"""UsuiTrack: a tracked low-rank frame per matrix, a moment of unit projected
gradients inside it, and a step of the moment's polar factor at `lr` of the
weight's norm. `docs/SPEC.md` is the design; the step numbers below are its.
"""

from __future__ import annotations

import math
import warnings
import weakref
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer

from . import frame
from .diagnostics import Diagnostics
from .polar import unit_polar
from .stochastic import round_into

# The frame's turn per basis update, in radians on every live plane. Constant,
# not scheduled (SPEC step 4, "The turn is constant").
GEODESIC_STEPSIZE = 0.02

# How a param group sizes its write (SPEC step 8): against the weight's norm, or
# by the matrix's shape alone.
STEP_RULES = ("trust", "fixed")

# Where a param group's frames start (SPEC step 2): a range finder on the first
# gradient, or drawn at random when the group is added.
FRAME_INITS = ("range_finder", "random")


@dataclass
class Prepared:
    """Phase one's output for one matrix: what `step()` needs once `G` is freed."""

    projected: Tensor
    """`Z = G Q`, oriented `[other, r]`, in the gradient's dtype."""
    tangent: Tensor | None
    """The Oja tangent `[d, r]`, fp32; `None` on the step that fits the frame."""
    raw_norm: Tensor
    """`||G||_F` before sanitizing, fp32: non-finite iff the gradient was."""


class UsuiTrack(Optimizer):
    """Optimizer for 2D weights that are shared linear maps.

    Biases, norms, embeddings and other lookup tables go to a separate
    optimizer (README, "Usage"). Per matrix it keeps a frame `[d, r]` and a
    moment `[other, r]`, both in the parameter's dtype.

    `release_matrix_grads` consumes each matrix gradient in a backward hook
    as soon as it is complete and frees it, rather than holding every gradient
    until `step()`. It takes exactly one backward per `step()`; gradient
    accumulation needs it off.

    A param group's `step_rule` is `"trust"` (default): `lr` is the relative
    step, against `||W||`. `"fixed"` sizes the step by the matrix's shape
    instead, so a zero-initialized matrix moves (SPEC step 8).

    A param group's `frame_init` is `"range_finder"` (default): each frame is
    fitted to its matrix's first gradient. `"random"` draws the frames when the
    group is added, on each param's device, so the model is placed first; it is
    for pretraining from a zero-initialized output, whose first gradients carry
    no subspace (SPEC step 2).
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict],
        lr: float = 4e-5,
        beta: float = 0.95,
        weight_decay: float = 0.0,
        rank_fraction: float = 0.05,
        side: str = "auto",
        release_matrix_grads: bool = False,
        diagnostics: bool = False,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not 0 <= beta < 1:
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
        if rank_fraction <= 0:
            raise ValueError(f"rank_fraction must be positive, got {rank_fraction}")
        self.release_matrix_grads = release_matrix_grads
        self.diagnostics: Diagnostics | None = Diagnostics() if diagnostics else None
        self._pending: dict[Tensor, Prepared] = {}
        self._group_of: dict[Tensor, dict] = {}
        self._frame_generators: dict[torch.device, torch.Generator] = {}
        owner = weakref.ref(self)

        def release(param: Tensor) -> None:
            optimizer = owner()
            if optimizer is not None:
                optimizer._prepare(param)

        self._release_hook = release
        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay, rank_fraction=rank_fraction, side=side, step_rule="trust", frame_init="range_finder")
        super().__init__(params, defaults)

    def add_param_group(self, param_group: dict) -> None:
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        problem = next((f"shape {tuple(p.shape)}" for p in group["params"] if p.ndim != 2), None)
        if group["side"] not in frame.SIDES:
            problem = f"side {group['side']!r}, not one of {frame.SIDES}"
        if group["step_rule"] not in STEP_RULES:
            problem = f"step_rule {group['step_rule']!r}, not one of {STEP_RULES}"
        if group["frame_init"] not in FRAME_INITS:
            problem = f"frame_init {group['frame_init']!r}, not one of {FRAME_INITS}"
        if problem is not None:
            self.param_groups.pop()
            raise ValueError(f"UsuiTrack takes 2D matrix parameters with a valid side, step_rule and frame_init; got {problem}")
        for param in group["params"]:
            self._group_of[param] = group
            if group["frame_init"] == "random":
                self.state[param]["frame"] = self._draw_frame(param, group)
            if self.release_matrix_grads and param.requires_grad:
                param.register_post_accumulate_grad_hook(self._release_hook)

    def _draw_frame(self, param: Tensor, group: dict) -> Tensor:
        # One private generator per device, so the draws follow the param order
        # and training's random stream never shifts.
        if param.device not in self._frame_generators:
            self._frame_generators[param.device] = torch.Generator(param.device).manual_seed(0)
        length = frame.orient(param, frame.resolve_side(param.shape, group["side"])).shape[1]
        return frame.draw(length, frame.rank(param.shape, group["rank_fraction"]), self._frame_generators[param.device], param.dtype)

    def load_state_dict(self, state_dict: dict) -> None:
        settings = [(group["step_rule"], group["frame_init"]) for group in self.param_groups]
        super().load_state_dict(state_dict)
        self._pending.clear()
        # A checkpoint from before `step_rule` or `frame_init` carries groups without them; they keep the ones configured.
        for group, (rule, init) in zip(self.param_groups, settings, strict=True):
            group.setdefault("step_rule", rule)
            group.setdefault("frame_init", init)
        # torch rebuilds the group dicts, so re-point the index at the live ones.
        self._group_of = {param: group for group in self.param_groups for param in group["params"]}
        # A param the checkpoint holds no state for gets its drawn frame back, so it does not fit one.
        for group in self.param_groups:
            for param in group["params"]:
                if group["frame_init"] == "random" and "frame" not in self.state[param]:
                    self.state[param]["frame"] = self._draw_frame(param, group)

    def zero_grad(self, set_to_none: bool = True) -> None:
        # A loop that bails out between backward and step (an OOM retry) may
        # call this with prepared work pending. Neither the moment nor the
        # frame has moved yet, so dropping it loses only those gradients.
        if self._pending:
            warnings.warn(
                f"UsuiTrack: zero_grad() dropped {len(self._pending)} prepared matrix gradients before step() applied them.",
                stacklevel=2,
            )
            self._pending.clear()
        super().zero_grad(set_to_none=set_to_none)

    def pop_diagnostics(self) -> dict[str, float]:
        """The reads since the last call, `usuitrack.diagnostics.READS` says how to
        read each; empty when diagnostics are off. Call on the logging cadence:
        this is the one device-to-host transfer."""

        if self.diagnostics is None:
            return {}
        return self.diagnostics.pop(param for group in self.param_groups for param in group["params"])

    @torch.no_grad()
    def _prepare(self, param: Tensor) -> None:
        """Phase one, SPEC steps 1-4: sanitize, project into the held frame, aim, free `G`."""

        if param in self._pending:
            raise RuntimeError(
                "UsuiTrack: a matrix received a second gradient before step(). "
                "release_matrix_grads takes one backward per step; accumulate gradients with it off."
            )
        grad = param.grad
        if grad.is_sparse:
            raise RuntimeError("UsuiTrack does not support sparse gradients")
        group = self._group_of[param]
        grad = frame.orient(grad, frame.resolve_side(param.shape, group["side"]))
        raw_norm = torch.linalg.vector_norm(grad, dtype=torch.float32)
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        state = self.state[param]
        if "frame" not in state:
            state["frame"] = frame.fit(grad, frame.rank(param.shape, group["rank_fraction"]))
            self._pending[param] = Prepared(grad @ state["frame"], None, raw_norm)
        else:
            projected, tangent = frame.project_and_aim(grad, state["frame"])
            self._pending[param] = Prepared(projected, tangent, raw_norm)
        param.grad = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    self._prepare(param)
            entries = [(param, self._pending.pop(param)) for param in group["params"] if param in self._pending]
            if entries:
                self._step_group(group, entries)
        if self.diagnostics is not None:
            self.diagnostics.step_end()
        return loss

    def _step_group(self, group: dict, entries: list[tuple[Tensor, Prepared]]) -> None:
        """Phase two: SPEC steps 5-8, then the frame moves and the moment is committed."""

        moments = [self._blend_moment(param, prepared, group["beta"]) for param, prepared in entries]
        for (param, _prepared), moment in zip(entries, moments, strict=True):
            self._write_weight(param, unit_polar(moment), group)

        # The weights were lifted through the held frame; only now does it move.
        # A matrix whose frame was fitted this step has no tangent and holds still.
        turning = [(param, prepared.tangent) for param, prepared in entries if prepared.tangent is not None]
        spectra = frame.tangent_spectra([tangent for _param, tangent in turning])
        turned = {
            param: frame.turn(self.state[param]["frame"], tangent, spectrum, GEODESIC_STEPSIZE)
            for (param, tangent), spectrum in zip(turning, spectra, strict=True)
        }
        if turned and self.diagnostics is not None:
            self.diagnostics.frame_turn(list(turned), [self.state[param]["frame"] for param in turned], list(turned.values()))

        # SPEC step 6: the frame is stored, the moment follows its in-span
        # rotation, then is rounded once.
        for (param, _prepared), moment in zip(entries, moments, strict=True):
            state = self.state[param]
            item = turned.get(param)
            if item is not None:
                state["frame"].copy_(item.frame)
                moment = moment @ item.rotation
            round_into(state["moment"], moment)

    def _blend_moment(self, param: Tensor, prepared: Prepared, beta: float) -> Tensor:
        """SPEC step 5: `M = beta M + (1 - beta) Z / ||Z||`, in fp32 until the commit."""

        state = self.state[param]
        if "moment" not in state:
            state["moment"] = torch.zeros_like(prepared.projected, dtype=param.dtype)
        projected = prepared.projected.float()
        projected_norm = projected.norm()
        unit = projected / projected_norm.clamp_min(1e-12)
        before = state["moment"].float()
        moment = beta * before + (1.0 - beta) * unit
        if self.diagnostics is not None:
            self.diagnostics.gradient(param, projected_norm, prepared.raw_norm)
            self.diagnostics.moment(param, unit, before, moment, beta)
        return moment

    def _write_weight(self, param: Tensor, direction: Tensor, group: dict) -> None:
        """SPEC step 8: `W -= lr size lift(O)`, decay folded in, rounded once."""

        side = frame.resolve_side(param.shape, group["side"])
        frame_q = self.state[param]["frame"]
        lift = frame.orient(direction @ frame_q.float().mT, side)
        if group["step_rule"] == "fixed":
            # Muon's aspect factor on the stored [out, in] shape, times the polar factor's norm.
            rows, cols = param.shape
            size = math.sqrt(max(1.0, rows / cols) * frame_q.shape[-1])
        else:
            size = torch.linalg.vector_norm(param, dtype=torch.float32)
        round_into(param, param.float() * (1.0 - group["lr"] * group["weight_decay"]) - group["lr"] * size * lift)
        if self.diagnostics is not None:
            # The lift is an isometry, so `||U|| = ||O|| size` without a pass over `W`.
            self.diagnostics.update(param, direction.norm() * size * group["lr"])
