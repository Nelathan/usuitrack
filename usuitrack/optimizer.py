"""UsuiTrack: a tracked low-rank frame per matrix, a moment of unit projected
gradients inside it, and a step of the moment's polar factor at `lr` of the
weight's norm. `docs/SPEC.md` is the design; the step numbers below are its.
"""

from __future__ import annotations

import math
import warnings
import weakref
from dataclasses import dataclass, field
from typing import Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer

from . import frame
from .diagnostics import Diagnostics
from .polar import unit_polar
from .stochastic import round_into

# The frame moment's memory: `Y = beta Y + (1 - beta) a` (SPEC step 4).
TRACKER_BETA = 0.95

# How a param group sizes its write (SPEC step 8): against the weight's norm, or
# by the matrix's shape alone.
STEP_RULES = ("trust", "fixed")

# Where a param group's frames start (SPEC step 2): a range finder on the first
# gradient, or drawn at random when the group is added.
FRAME_INITS = ("range_finder", "random")

# A block's transient cap (SPEC "Blocks"): same-shaped matrices step together
# until their batched work would pass it. The stacked prototype's cap (FACTS on
# `lab`, "On a small deep model the step costs 38% of training"); as a peak on
# LFM3 or Anima, unmeasured.
BLOCK_BYTES = 256 * 2**20

# The state a block holds as one buffer per entry, each matrix's a view into it.
BLOCK_STATE = ("frame", "frame_moment", "frame_moment_span", "moment")


@dataclass
class Prepared:
    """Phase one's output for one matrix: what `step()` needs once `G` is freed."""

    projected: Tensor
    """`Z = G Q`, oriented `[other, r]`, in the gradient's dtype."""
    projected_norm: Tensor
    """`||Z||_F`, fp32, of the sanitized `Z`: taken once, for the blend and the reads."""
    aim: Tensor | None
    """The normalized action `[d, r]`, fp32; `None` on the step that fits the frame."""
    raw_norm: Tensor
    """`||G||_F` before sanitizing, fp32: non-finite iff the gradient was."""
    probed: bool = False
    """`raw_norm` is Molegrad's probe estimate, not the exact norm."""


@dataclass(eq=False)
class Block:
    """Same-shaped matrices of one group, in param order, stepped together (SPEC "Blocks").

    Each entry of `BLOCK_STATE` is one `[b, ...]` buffer, allocated with the
    first frame, and a matrix's state holds views into it, so the step reads
    and writes the block's state in place, with no stacking copy."""

    params: list[Tensor]
    side: str
    """The resolved side, `left` or `right`."""
    rank: int
    buffers: dict[str, Tensor] = field(default_factory=dict)
    init_norm: Tensor | None = None
    """`[b]`, fp32: each matrix's norm before its first write, under `norm_bounds`."""
    indices: dict[tuple[int, ...], Tensor] = field(default_factory=dict)
    """Device index tensors for steps that reach only some members, made once per subset."""

    def index(self, members: list[int]) -> Tensor | None:
        """`None` for the whole block, which then reads its buffers as they are."""

        if len(members) == len(self.params):
            return None
        key = tuple(members)
        if key not in self.indices:
            self.indices[key] = torch.tensor(members, device=self.params[0].device)
        return self.indices[key]

    def take(self, name: str, index: Tensor | None) -> Tensor:
        buffer = self.buffers[name]
        return buffer if index is None else buffer[index]

    def put(self, name: str, index: Tensor | None, value: Tensor) -> None:
        buffer = self.buffers[name]
        if index is None:
            buffer.copy_(value)
        else:
            buffer[index] = value.to(buffer.dtype)

    def put_rounded(self, name: str, index: Tensor | None, value: Tensor) -> None:
        """`round_into` the block's buffer, stochastically for bf16."""

        buffer = self.buffers[name]
        if index is None:
            round_into(buffer, value)
        else:
            rounded = buffer.new_empty(value.shape)
            round_into(rounded, value)
            buffer[index] = rounded


class UsuiTrack(Optimizer):
    """Optimizer for 2D weights that are shared linear maps.

    Biases, norms, embeddings and other lookup tables go to a separate
    optimizer (README, "Usage"). Per matrix it keeps a frame `[d, r]`, a
    moment `[other, r]` and the frame's moment, `[d, r]` in the parameter's
    dtype plus its in-span part `[r, r]` in fp32.

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

    A param group's `norm_bounds` is `(low, high)` in multiples of each
    matrix's Frobenius norm at its first step, either side `None` for open;
    the default `(None, None)` bounds nothing. After each write the norm is
    clamped into the bounds; `(1, 1)` pins it (SPEC step 8).
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict],
        lr: float = 4e-5,
        beta: float = 0.95,
        weight_decay: float = 0.0,
        rank_fraction: float = 0.05,
        side: str = "right",
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
        self._block_of: dict[Tensor, tuple[Block, int]] = {}
        self._checkpoint: dict | None = None
        self._settings: list | None = None
        self._frame_generators: dict[torch.device, torch.Generator] = {}
        owner = weakref.ref(self)

        def release(param: Tensor) -> None:
            optimizer = owner()
            if optimizer is not None:
                optimizer._prepare(param)

        self._release_hook = release
        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay, rank_fraction=rank_fraction, side=side, step_rule="trust", frame_init="range_finder", norm_bounds=(None, None))
        super().__init__(params, defaults)
        self.register_load_state_dict_post_hook(self._after_load, prepend=True)

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
        if not _valid_bounds(group["norm_bounds"]):
            problem = f"norm_bounds {group['norm_bounds']!r}, not (low, high) with each positive or None and low <= high"
        if problem is not None:
            self.param_groups.pop()
            raise ValueError(f"UsuiTrack takes 2D matrix parameters with a valid side, step_rule, frame_init and norm_bounds; got {problem}")
        group["norm_bounds"] = tuple(group["norm_bounds"])
        for param in group["params"]:
            self._group_of[param] = group
            if group["frame_init"] == "random":
                self._hold_frame(param, self._draw_frame(param, group))
            if self.release_matrix_grads and param.requires_grad:
                param.register_post_accumulate_grad_hook(self._release_hook)

    def _draw_frame(self, param: Tensor, group: dict) -> Tensor:
        # One private generator per device, so the draws follow the param order
        # and training's random stream never shifts.
        if param.device not in self._frame_generators:
            self._frame_generators[param.device] = torch.Generator(param.device).manual_seed(0)
        length = frame.orient(param, frame.resolve_side(param.shape, group["side"])).shape[1]
        return frame.draw(length, frame.rank(param.shape, group["rank_fraction"]), self._frame_generators[param.device], param.dtype)

    def _form_blocks(self, group: dict) -> None:
        """Split a group into blocks: one per side and shape (and device, dtype), in
        param order, a new one each time the open one's work would pass `BLOCK_BYTES`.
        Formed at the group's first frame, so side and rank are read when they first matter."""

        open_blocks: dict[tuple, Block] = {}
        for param in group["params"]:
            side = frame.resolve_side(param.shape, group["side"])
            rank = frame.rank(param.shape, group["rank_fraction"])
            key = (param.device, param.dtype, side, tuple(param.shape))
            block = open_blocks.get(key)
            if block is None or (len(block.params) + 1) * _work_bytes(param, rank) > BLOCK_BYTES:
                block = open_blocks[key] = Block([], side, rank)
            self._block_of[param] = (block, len(block.params))
            block.params.append(param)

    def _hold_frame(self, param: Tensor, value: Tensor) -> Tensor:
        """Store a matrix's first frame and bind its state to its block's buffers;
        the frame's moment and the moment start at zero. Returns the stored frame."""

        if param not in self._block_of:
            self._form_blocks(self._group_of[param])
        block, index = self._block_of[param]
        if not block.buffers:
            count = len(block.params)
            sample = block.params[0]
            other, length = frame.orient(sample, block.side).shape
            device, dtype = sample.device, sample.dtype
            block.buffers = {
                "frame": torch.zeros(count, length, block.rank, device=device, dtype=dtype),
                "frame_moment": torch.zeros(count, length, block.rank, device=device, dtype=dtype),
                # fp32: rounding `Y`'s large in-span part walks the frame (SPEC step 4).
                "frame_moment_span": torch.zeros(count, block.rank, block.rank, device=device, dtype=torch.float32),
                "moment": torch.zeros(count, other, block.rank, device=device, dtype=dtype),
            }
        state = self.state[param]
        for name in BLOCK_STATE:
            state[name] = block.buffers[name][index]
        state["frame"].copy_(value)
        return state["frame"]

    def _hold_init_norm(self, param: Tensor, value: Tensor | float) -> None:
        block, index = self._block_of[param]
        if block.init_norm is None:
            block.init_norm = torch.zeros(len(block.params), device=param.device, dtype=torch.float32)
        self.state[param]["init_norm"] = block.init_norm[index]
        self.state[param]["init_norm"].copy_(torch.as_tensor(value))

    def load_state_dict(self, state_dict: dict) -> None:
        # torch casts loaded state to the param's dtype, then runs the load post hooks,
        # Molegrad's among them; `_after_load` runs first and reads the fp32 entries from here.
        self._settings = [(group["step_rule"], group["frame_init"], group["norm_bounds"]) for group in self.param_groups]
        self._checkpoint = state_dict
        try:
            super().load_state_dict(state_dict)
        finally:
            self._checkpoint = self._settings = None

    def _after_load(self, _optimizer: Optimizer) -> None:
        self._pending.clear()
        # A checkpoint from before `step_rule`, `frame_init` or `norm_bounds` carries groups without them; they keep the ones configured.
        for group, (rule, init, bounds) in zip(self.param_groups, self._settings, strict=True):
            group.setdefault("step_rule", rule)
            group.setdefault("frame_init", init)
            group["norm_bounds"] = tuple(group.setdefault("norm_bounds", bounds))
        # torch rebuilds the group dicts, so re-point the index at the live ones.
        self._group_of = {param: group for group in self.param_groups for param in group["params"]}
        # The loaded state is packed into fresh blocks. The span and the init norm are fp32
        # (SPEC step 4, step 8), so they are taken again from the checkpoint, matched to
        # params by order as torch matches them.
        self._block_of = {}
        indices = [index for group in self._checkpoint["param_groups"] for index in group["params"]]
        live = [param for group in self.param_groups for param in group["params"]]
        for index, param in zip(indices, live, strict=True):
            loaded = dict(self.state[param])
            if "frame" not in loaded:
                continue
            saved = self._checkpoint["state"].get(index, {})
            self._hold_frame(param, loaded["frame"])
            for name in ("frame_moment", "moment"):
                if name in loaded:
                    self.state[param][name].copy_(loaded[name])
            if "frame_moment_span" in saved:
                self.state[param]["frame_moment_span"].copy_(saved["frame_moment_span"])
            if "init_norm" in saved:
                self._hold_init_norm(param, saved["init_norm"])
        # A param the checkpoint holds no state for gets its drawn frame back, so it does not fit one.
        for group in self.param_groups:
            for param in group["params"]:
                if group["frame_init"] == "random" and "frame" not in self.state[param]:
                    self._hold_frame(param, self._draw_frame(param, group))

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
            held = self._hold_frame(param, frame.fit(grad, frame.rank(param.shape, group["rank_fraction"])))
            projected, aim = grad @ held, None
        else:
            projected, aim = frame.project_and_aim(grad, state["frame"])
        self._pending[param] = Prepared(projected, torch.linalg.vector_norm(projected, dtype=torch.float32), aim, raw_norm)
        param.grad = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            # A block steps in one batched call per kind of phase-one work: the
            # matrices that fitted their frame this step hold it still.
            parts: dict[tuple[Block, bool, bool], list[tuple[int, Prepared]]] = {}
            for param in group["params"]:
                if param.grad is not None:
                    self._prepare(param)
                prepared = self._pending.pop(param, None)
                if prepared is not None:
                    block, index = self._block_of[param]
                    parts.setdefault((block, prepared.aim is None, prepared.probed), []).append((index, prepared))
            for (block, _fitted, _probed), entries in parts.items():
                self._step_block(group, block, entries)
        if self.diagnostics is not None:
            self.diagnostics.step_end()
        return loss

    def _step_block(self, group: dict, block: Block, entries: list[tuple[int, Prepared]]) -> None:
        """Phase two for a block's members, batched: SPEC steps 5-8, then the frame
        moves and the moment is committed."""

        members = [member for member, _prepared in entries]
        prepared = [item for _member, item in entries]
        index = block.index(members)
        params = [block.params[member] for member in members]
        moment = self._blend_moment(params, block.take("moment", index), prepared, group["beta"])
        frames = block.take("frame", index)
        self._write_weights(group, block, index, params, unit_polar(moment), frames)
        # The weights were lifted through the held frames; only now do they move.
        # Matrices whose frame was fitted this step have no aim and hold still.
        if prepared[0].aim is not None:
            moved = self._follow(block, index, torch.stack([item.aim for item in prepared]))
            if self.diagnostics is not None:
                self.diagnostics.frame_move(params, frames, moved)
            # SPEC step 6: the frame is stored, the moment follows its in-span
            # rotation, then is rounded once.
            block.put("frame", index, moved.frame)
            moment = moment @ moved.rotation
        block.put_rounded("moment", index, moment)

    def _follow(self, block: Block, index: Tensor | None, aims: Tensor) -> frame.Moved:
        """SPEC step 4: blend the aims into the frames' moment, commit it, and move each frame to `polar(Q + Y)`."""

        frames = block.take("frame", index)
        held = frames.float() @ block.take("frame_moment_span", index) + block.take("frame_moment", index).float()
        frame_moment = ((1.0 - TRACKER_BETA) * aims).add_(held, alpha=TRACKER_BETA)
        moved = frame.follow(frames, frame_moment)
        # Held against the new frame as stored: `Y = Q S + H`, `S` in fp32. `H` is small
        # where `Y`'s in-span part is large, so storing it rounds away only noise; each
        # update moves `H` by `1 - beta_f` of an aim, far above a bf16 half-ulp, so
        # round-to-nearest has no dead zone to lose a step in.
        new = moved.frame.float()
        span = new.mT @ frame_moment
        block.put("frame_moment_span", index, span)
        block.put("frame_moment", index, (frame_moment - new @ span).to(frames.dtype))
        return moved

    def _blend_moment(self, params: list[Tensor], held: Tensor, prepared: list[Prepared], beta: float) -> Tensor:
        """SPEC step 5: `M = beta M + (1 - beta) Z / ||Z||`, in fp32 until the commit."""

        projected_norm = torch.stack([item.projected_norm for item in prepared])
        unit = torch.stack([item.projected for item in prepared]).float() / projected_norm.clamp_min(1e-12)[:, None, None]
        before = held.float()
        moment = beta * before + (1.0 - beta) * unit
        if self.diagnostics is not None:
            raw_norm = torch.stack([item.raw_norm for item in prepared])
            self.diagnostics.gradient(params, projected_norm, raw_norm, prepared[0].probed)
            self.diagnostics.moment(params, unit, before, moment, beta)
        return moment

    def _write_weights(self, group: dict, block: Block, index: Tensor | None, params: list[Tensor], direction: Tensor, frames: Tensor) -> None:
        """SPEC step 8: `W -= lr size lift(O)` for each member, decay folded in, rounded once."""

        lift = frame.orient(direction @ frames.float().mT, block.side)
        # One member is written in place; more are stacked, then copied back in one call.
        weights = params[0].unsqueeze(0) if len(params) == 1 else torch.stack(params)
        if group["step_rule"] == "fixed":
            # Muon's aspect factor on the stored [out, in] shape, times the polar factor's norm.
            rows, cols = params[0].shape
            scale = group["lr"] * math.sqrt(max(1.0, rows / cols) * block.rank)
        else:
            scale = group["lr"] * torch.linalg.vector_norm(weights, dim=(-2, -1), keepdim=True, dtype=torch.float32)
        written = weights.float() * (1.0 - group["lr"] * group["weight_decay"]) - scale * lift
        low, high = group["norm_bounds"]
        if (low, high) != (None, None):
            written = self._bound_norm(block, index, params, weights, written, low, high)
        round_into(weights, written)
        if len(params) > 1:
            torch._foreach_copy_(params, list(weights.unbind(0)))
        if self.diagnostics is not None:
            # The lift is an isometry, so `||U|| = ||O|| lr size` without a pass over `W`.
            self.diagnostics.update(params, (torch.linalg.vector_norm(direction, dim=(-2, -1), keepdim=True) * scale).flatten())

    def _bound_norm(self, block: Block, index: Tensor | None, params: list[Tensor], weights: Tensor, written: Tensor, low: float | None, high: float | None) -> Tensor:
        """SPEC step 8, norm bounds: clamp `||W||` into `[low R, high R]`, `R` the norm before the first write."""

        fresh = [member for member, param in enumerate(params) if "init_norm" not in self.state[param]]
        if fresh:
            norms = torch.linalg.vector_norm(weights[fresh], dim=(-2, -1), dtype=torch.float32)
            # One host read, on a matrix's first step only.
            if bool((norms == 0.0).any()):
                raise ValueError("UsuiTrack: norm_bounds are relative to a matrix's first norm, and this matrix starts at zero")
            for member, norm in zip(fresh, norms.unbind(0)):
                self._hold_init_norm(params[member], norm)
        reference = (block.init_norm if index is None else block.init_norm[index])[:, None, None]
        norm = torch.linalg.vector_norm(written, dim=(-2, -1), keepdim=True)
        bounded = norm.clamp(None if low is None else low * reference, None if high is None else high * reference)
        return written * (bounded / norm.clamp_min(1e-12))


def _work_bytes(param: Tensor, rank: int) -> int:
    """A matrix's share of its block's transient: the stacked weight and the four fp32
    temporaries live at the write's peak (the lift, the decayed copy, the scaled lift, the
    result), and about four fp32 copies of its moment and frame."""

    rows, cols = param.shape
    return (param.element_size() + 16) * rows * cols + 16 * (rows + cols) * rank


def _valid_bounds(bounds) -> bool:
    if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
        return False
    low, high = bounds
    if any(side is not None and not (isinstance(side, (int, float)) and not isinstance(side, bool) and side > 0) for side in bounds):
        return False
    return low is None or high is None or low <= high
