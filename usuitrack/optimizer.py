"""UsuiTrack: a tracked low-rank frame per matrix, a moment of unit projected
gradients inside it, and a step of the moment's polar factor at `lr` of the
weight's norm. `docs/SPEC.md` is the design; the step numbers below are its.
"""

from __future__ import annotations

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
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict],
        lr: float = 1e-4,
        beta: float = 0.95,
        weight_decay: float = 0.0,
        rank_fraction: float = 0.1,
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
        owner = weakref.ref(self)

        def release(param: Tensor) -> None:
            optimizer = owner()
            if optimizer is not None:
                optimizer._prepare(param)

        self._release_hook = release
        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay, rank_fraction=rank_fraction, side=side)
        super().__init__(params, defaults)

    def add_param_group(self, param_group: dict) -> None:
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        problem = next((f"shape {tuple(p.shape)}" for p in group["params"] if p.ndim != 2), None)
        if group["side"] not in frame.SIDES:
            problem = f"side {group['side']!r}, not one of {frame.SIDES}"
        if problem is not None:
            self.param_groups.pop()
            raise ValueError(f"UsuiTrack takes 2D matrix parameters with a valid side; got {problem}")
        for param in group["params"]:
            self._group_of[param] = group
            if self.release_matrix_grads and param.requires_grad:
                param.register_post_accumulate_grad_hook(self._release_hook)

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self._pending.clear()
        # torch rebuilds the group dicts, so re-point the index at the live ones.
        self._group_of = {param: group for group in self.param_groups for param in group["params"]}

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
        for (param, _prepared), direction in zip(entries, unit_polar(moments), strict=True):
            self._write_weight(param, direction, group)

        # The weights were lifted through the held frame; only now does it move.
        turning = [index for index, (_param, prepared) in enumerate(entries) if prepared.tangent is not None]
        rotations: dict[int, Tensor] = {}
        if turning:
            params = [entries[index][0] for index in turning]
            before = [self.state[param]["frame"] for param in params]
            turned = frame.turn(before, [entries[index][1].tangent for index in turning], GEODESIC_STEPSIZE)
            if self.diagnostics is not None:
                self.diagnostics.frame_turn(params, before, turned)
            for index, param, item in zip(turning, params, turned, strict=True):
                self.state[param]["frame"].copy_(item.frame)
                rotations[index] = item.rotation

        # SPEC step 6: the moment follows the stored frame's in-span rotation,
        # then is rounded once.
        for index, ((param, _prepared), moment) in enumerate(zip(entries, moments, strict=True)):
            rotation = rotations.get(index)
            round_into(self.state[param]["moment"], moment if rotation is None else moment @ rotation)

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
        """SPEC step 8: `W -= lr ||W|| lift(O)`, decay folded in, rounded once."""

        side = frame.resolve_side(param.shape, group["side"])
        lift = frame.orient(direction @ self.state[param]["frame"].float().mT, side)
        weight_norm = torch.linalg.vector_norm(param, dtype=torch.float32)
        work = param.float() if param.dtype is torch.bfloat16 else param
        if group["weight_decay"]:
            work.mul_(1.0 - group["lr"] * group["weight_decay"])
        work.add_((lift * weight_norm).to(work.dtype), alpha=-group["lr"])
        if work is not param:
            round_into(param, work)
        if self.diagnostics is not None:
            # The lift is an isometry, so `||U|| = ||O|| ||W||` without a pass over `W`.
            self.diagnostics.update(param, direction.norm() * weight_norm * group["lr"])
