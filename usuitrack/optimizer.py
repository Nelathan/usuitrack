from __future__ import annotations

import math
import warnings
import weakref
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer

from .diagnostics import DiagnosticsAccumulator
from .projector import ProjectionSide, SubspaceProjector
from .stochastic import copy_stochastic_, wants_stochastic_rounding
AURORA_PP_ITERATIONS = 1
AURORA_PP_BETA = 0.5
ORTHOGONALIZATION_SCALE_MODE = "muon"
NEWTON_SCHULZ_COEFFICIENTS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)
# Fixed step size for the Grassmann geodesic. Constant, not scheduled.
#
# There used to be a `max(0.01, 1/t)` anneal here, and its justification was a
# moving aim: while Adafactor's row/column variances warmed up, the conditioned
# Gram whose eigenspace the tracker targets was itself shifting, so the frame was
# chasing a target rather than fitting one. A hot start is the right answer to
# that. The schedule reached this value at basis update 100 and Adafactor's
# variance memory was 1/(1 - 0.99) = 100 steps; the agreement is not a
# coincidence. With the conditioning gone the aim is the leading eigenspace of
# `G^T G` from the first step and moves only as the model does, so the
# compensation has nothing left to compensate for.
#
# Removing it also makes the tracker observable. `sigma` is already
# self-annealing, so a decaying schedule on top of it made frame motion the
# product of two annealing terms, and no reading could separate "the tracker
# settled" from "the clock ran out". Constant step, and `rotation_rad_sum` is
# `sigma` up to a fixed factor.
BASIS_UPDATE_STEP = 0.01


@dataclass
class MatrixUpdate:
    param: Tensor
    projector: SubspaceProjector
    projected_exp_avg: Tensor
    original_shape: tuple[int, ...]
    oja_tangent: Tensor | None = None
    raw_grad_norm: Tensor | None = None


class UsuiTrack(Optimizer):
    """Eager UsuiTrack baseline optimizer.

    Matrix parameters keep optimizer state in projected space: an orthonormal
    basis plus a projected first moment. UsuiTrack accepts only 2D parameters;
    callers own any bias, norm, or other fallback optimizer separately. The
    default one-state basis tracker clips every full matrix gradient and updates
    its live basis on the configured cadence.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        beta: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        rank: int = 32,
        side: ProjectionSide | str = ProjectionSide.AUTO,
        grad_clip_norm: float = 1.0,
        basis_update_interval: int = 1,
        consume_grad: bool = True,
        release_matrix_grads: bool = False,
        compile_tensor_kernels: bool = False,
        stochastic_rounding: bool = True,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not 0 <= beta < 1:
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if grad_clip_norm <= 0:
            raise ValueError(f"grad_clip_norm must be positive, got {grad_clip_norm}")
        if basis_update_interval <= 0:
            raise ValueError(f"basis_update_interval must be positive, got {basis_update_interval}")
        if release_matrix_grads and not consume_grad:
            raise ValueError("release_matrix_grads requires consume_grad=True")

        defaults = dict(
            lr=lr,
            beta=beta,
            eps=eps,
            weight_decay=weight_decay,
            rank=rank,
            side=ProjectionSide(side).value,
            grad_clip_norm=grad_clip_norm,
            consume_grad=consume_grad,
            compile_tensor_kernels=compile_tensor_kernels,
            basis_update_interval=basis_update_interval,
            matrix_step=0,
            basis_update_step=0,
        )
        super().__init__(params, defaults)
        self._compiled_orthogonalize_update = torch.compile(UsuiTrack._orthogonalize_aurora_muon_tensor) if compile_tensor_kernels else None
        self._compiled_prepare_tracker_right = (
            torch.compile(UsuiTrack._prepare_tracker_right_tensors, dynamic=True)
            if compile_tensor_kernels
            else None
        )
        self._compiled_prepare_tracker_left = (
            torch.compile(UsuiTrack._prepare_tracker_left_tensors, dynamic=True)
            if compile_tensor_kernels
            else None
        )
        self._pending_matrix_updates: dict[Tensor, MatrixUpdate] = {}
        self.diagnostics_enabled = False
        # Convergence instrument, off even when diagnostics are on: it is the
        # only read that goes to zero iff the frame actually stops, and it is
        # the one diagnostic that costs persistent bytes. Sampled over a few
        # matrices rather than carried per parameter -- see pop_diagnostics().
        self.diagnostics_lag_enabled = False
        self.diagnostics_lag_matrices = 32
        self.diagnostics_lag_interval = 5
        self._lag_snapshots: dict[Tensor, Tensor] = {}
        self._lag_sampled: set[Tensor] | None = None
        self._diagnostics: DiagnosticsAccumulator | None = None
        self._step_update_norm_sq: Tensor | None = None
        self._matrix_grad_hook_handles = []
        self._matrix_param_groups: dict[Tensor, dict] = {}
        self.release_matrix_grads = release_matrix_grads
        self.stochastic_rounding = stochastic_rounding
        for group in self.param_groups:
            for param in group["params"]:
                if param.ndim == 2:
                    self._matrix_param_groups[param] = group
        if release_matrix_grads:
            optimizer_ref = weakref.ref(self)

            def release_grad(param: Tensor) -> None:
                optimizer = optimizer_ref()
                if optimizer is not None:
                    optimizer.prepare(param)

            for group in self.param_groups:
                if not group["consume_grad"] and any(param.ndim == 2 for param in group["params"]):
                    raise ValueError("release_matrix_grads requires consume_grad=True for every matrix parameter group")
                for param in group["params"]:
                    if param.ndim == 2 and param.requires_grad:
                        self._matrix_grad_hook_handles.append(param.register_post_accumulate_grad_hook(release_grad))

    def add_param_group(self, param_group: dict) -> None:
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        try:
            # effective_rank() caps rank at half a parameter's smaller side, so
            # a configured rank is a ceiling rather than a promise. This is
            # normal and not an error -- a model trained at rank 256 runs its
            # tall narrow modules at 32, the same way a LoRA config does -- but
            # saying nothing would be dishonest, so report it once at startup.
            # One summary per group, not one warning per parameter: on a real
            # model the same cap applies to hundreds of weights.
            clamped: dict[tuple[tuple[int, ...], int], int] = {}
            for param in group["params"]:
                if param.ndim != 2:
                    raise ValueError(
                        "UsuiTrack only supports 2D matrix parameters; "
                        f"got shape {tuple(param.shape)}"
                    )
                max_rank = SubspaceProjector(
                    rank=group["rank"], side=ProjectionSide(group["side"])
                ).effective_rank(param)
                if group["rank"] > max_rank:
                    key = (tuple(param.shape), max_rank)
                    clamped[key] = clamped.get(key, 0) + 1
            if clamped:
                detail = ", ".join(
                    f"{count}x{list(shape)}->rank {rank}"
                    for (shape, rank), count in sorted(clamped.items())
                )
                warnings.warn(
                    f"UsuiTrack: configured rank {group['rank']} exceeds half the smaller side "
                    f"of {sum(clamped.values())} of {len(group['params'])} matrix parameters; "
                    f"those run at a reduced rank ({detail}).",
                    stacklevel=2,
                )
        except Exception:
            self.param_groups.pop()
            raise

        matrix_param_groups = getattr(self, "_matrix_param_groups", None)
        if matrix_param_groups is not None:
            for param in group["params"]:
                matrix_param_groups[param] = group

    def zero_grad(self, set_to_none: bool = True) -> None:
        # A caller (an OOM-retry path, say) may need to bail out after prepare()
        # has already advanced moving-average state but before step() applied it.
        # That advance cannot be undone: those parameters carry one Oja/moment
        # step that never reached the weights. Not good, not worth killing a run
        # over -- so warn and drop the pending work rather than raise.
        if self._pending_matrix_updates:
            warnings.warn(
                f"UsuiTrack: zero_grad() discarded prepared updates for "
                f"{len(self._pending_matrix_updates)} matrix parameters. Their moment state "
                f"has advanced by one step that will not reach the weights.",
                stacklevel=2,
            )
        self._pending_matrix_updates.clear()
        super().zero_grad(set_to_none=set_to_none)

    def _diagnostics_sink(self) -> DiagnosticsAccumulator | None:
        """The live accumulator, or None when telemetry is off.

        Every accumulation site goes through here and does nothing when the
        result is None, so a run with diagnostics off pays one attribute read
        per site and no device work at all.
        """

        if not self.diagnostics_enabled:
            return None
        if self._diagnostics is None:
            self._diagnostics = DiagnosticsAccumulator()
        return self._diagnostics

    @torch.no_grad()
    def pop_diagnostics(self) -> dict[str, float]:
        """Telemetry accumulated since the last call, as a plain dict of floats.

        Set ``diagnostics_enabled = True`` and call this on whatever cadence the
        surrounding trainer already logs at -- every step is supported but not
        the intent, since this is where the accumulated device tensors are read
        back to the host. An empty dict means either telemetry is off or nothing
        has happened since the last read, and is always safe to skip logging.

        Means are over samples (matrix x step) since the last read;
        ``nonfinite_grads`` is a total. What is here answers a specific question, in this
        order of load-bearing-ness:

        ``rotation_rad_sum``
            How far the frame turned, as the sum of the per-plane angles of one
            geodesic -- the tangent's nuclear norm times the step size. Not the
            angle any single plane moves through. Falling is convergence; a
            Since the step size is a constant, this is ``sigma`` up to that
            constant, so it reads directly as the tracker's residual.
        ``tangent_concentration``
            Where that motion went, in ``[1/r, 1]``: the fraction of the
            tangent's energy in its leading direction. High means a confident
            aim drifting; low means the frame is spinning on a near-isotropic
            noise tail. Distinguishes a long smooth turn from churn, which
            ``rotation_rad_sum`` alone cannot.
        ``projected_grad_norm``, ``grad_to_moment_ratio``
            Scale of the gradient inside the frame, and how much of
            the moment is this batch rather than history. The ratio is against
            the moment *after* this step's update, which sits at ``1/(1 - beta)``
            on the first step and settles well below it once the moment has
            history; it can spike when a fresh gradient cancels the moment it
            just went into.
        ``update_to_param_ratio``
            Mean per-step weight motion against current weight norm. Flat
            through healthy training; mostly useful for finding a sane learning
            rate on an unfamiliar model.
        ``grad_moment_cosine``
            Agreement between this batch's projected gradient and the moment as
            it stood before this step -- does the batch confirm accumulated
            history, or contradict it. A persistence read, not a preference:
            unlike capture it cannot reward the frame for what it already holds.
            Read it beside ``tangent_concentration``: disagreement with a
            concentrated spectrum is structured drift, disagreement with a flat
            one is noise.
        ``basis_lag_rms_sin``
            Only present when ``diagnostics_lag_enabled``. RMS sine of the
            principal angles between each sampled frame and its own snapshot
            ``diagnostics_lag_interval`` basis updates ago. Zero iff every
            tracked plane has stopped. This is the only metric here that can
            distinguish a frame that has converged from one orbiting a fixed
            point at constant radius; every other read is floored by aim noise.
        ``nonfinite_grads``
            How many incoming matrix gradients arrived non-finite and were
            sanitized at the clip. This is the one guard left on the matrix path,
            and this counter is what would justify removing it too.
        """

        accumulator = self._diagnostics
        self._diagnostics = None
        if accumulator is None or not accumulator:
            return {}
        result = accumulator.reduce()
        update_norm = result.pop("update_norm", None)
        if update_norm is not None:
            param_norm = self._matrix_param_norm()
            result["update_to_param_ratio"] = update_norm / param_norm if param_norm > 0 else float("nan")
        return result

    @torch.no_grad()
    def _matrix_param_norm(self) -> float:
        """Global norm of the parameters this optimizer owns.

        A state quantity, not an accumulation, so it is measured here at read
        time rather than swept every step.
        """

        squares = [
            param.detach().float().pow(2).sum()
            for group in self.param_groups
            for param in group["params"]
        ]
        if not squares:
            return 0.0
        return float(torch.stack(squares).sum().sqrt())

    @torch.no_grad()
    def prepare(self, param: Tensor) -> None:
        """Consume and prepare one owned full matrix gradient exactly once."""

        group = self._matrix_group(param)
        if group is None:
            if not self._owns_param(param):
                raise ValueError("cannot prepare a parameter not owned by this optimizer")
            raise ValueError(f"prepare() only supports 2D matrix parameters, got shape {tuple(param.shape)}")
        if not group["consume_grad"]:
            raise RuntimeError("prepare() requires consume_grad=True for the parameter group")
        self._prepare_matrix_param(param, require_full_grad=True)

    @torch.no_grad()
    def _prepare_matrix_param(
        self,
        param: Tensor,
        require_full_grad: bool = False,
    ) -> MatrixUpdate:
        if param in self._pending_matrix_updates:
            raise RuntimeError(
                "matrix parameter is already prepared; prepare/release does not support gradient accumulation before step()"
            )
        grad = param.grad
        if require_full_grad and grad is None:
            raise RuntimeError("prepare() requires a live full matrix gradient")
        if grad is None:
            raise RuntimeError("matrix update requires a full grad")
        if grad is not None and grad.is_sparse:
            raise RuntimeError("UsuiTrack does not support sparse gradients")
        group = self._matrix_group(param)
        if group is None:
            raise ValueError("cannot prepare a matrix parameter not owned by this optimizer")
        update = self._prepare_matrix_update(param, grad, group)
        self._pending_matrix_updates[param] = update
        if group["consume_grad"]:
            param.grad = None
        return update

    def released_matrix_grad_norms(self) -> tuple[Tensor, ...]:
        """Raw full-gradient norms retained for telemetry after matrix grads are released."""

        return tuple(
            self._pending_matrix_updates[param].raw_grad_norm
            for group in self.param_groups
            for param in group["params"]
            if param in self._pending_matrix_updates
            and self._pending_matrix_updates[param].raw_grad_norm is not None
        )

    def _matrix_group(self, param: Tensor) -> dict | None:
        group = self._matrix_param_groups.get(param)
        if group is not None:
            return group
        if param.ndim != 2:
            return None
        for candidate_group in self.param_groups:
            if any(param is candidate for candidate in candidate_group["params"]):
                self._matrix_param_groups[param] = candidate_group
                return candidate_group
        return None

    def _owns_param(self, param: Tensor) -> bool:
        return any(param is candidate for group in self.param_groups for candidate in group["params"])

    def _validate_step_inputs(self) -> None:
        for group in self.param_groups:
            for param in group["params"]:
                pending = param in self._pending_matrix_updates
                grad = param.grad
                if pending and grad is not None:
                    raise RuntimeError("a prepared matrix parameter cannot also have a new live gradient")
                if self.release_matrix_grads and grad is not None:
                    raise RuntimeError("release_matrix_grads requires matrix gradients to be produced by backward hooks")
                if grad is not None and grad.is_sparse:
                    raise RuntimeError("UsuiTrack does not support sparse gradients")

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None and (self.release_matrix_grads or self._pending_matrix_updates):
            raise RuntimeError("optimizer closures cannot run while matrix updates are pending")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._validate_step_inputs()
        self._step_update_norm_sq = None

        for group in self.param_groups:
            matrix_updates = []

            for p in group["params"]:
                if p in self._pending_matrix_updates:
                    matrix_updates.append(self._pending_matrix_updates[p])
                    continue
                if p.grad is None:
                    continue
                matrix_updates.append(self._prepare_matrix_param(p))
            if matrix_updates:
                basis_update_due = self._basis_update_due(group)
                group["matrix_step"] += 1
                if basis_update_due:
                    group["basis_update_step"] += 1
            # Weights first, frame second. Everything in this step -- the
            # held-frame projection, the Oja tangent, and the lift back -- must
            # see the same frame `Q_t`. Moving the frame first meant the update
            # was projected down through `Q_t` and lifted back through
            # `Q_{t+1}`, so every step applied its own gradient rotated by
            # however far the tracker had just turned. History is unaffected:
            # the moment is read next step in the moved frame, where identity
            # transport is exact because the geodesic is a rigid rotation.
            self._apply_matrix_update_buckets(matrix_updates, group)
            self._apply_basis_updates(matrix_updates, group)

        # One sample per step, not per matrix: the quantity that means something
        # is the whole step's motion against the whole model's norm, so the
        # per-matrix squares are summed here and rooted once.
        diagnostics = self._diagnostics_sink()
        if diagnostics is not None and self._step_update_norm_sq is not None:
            diagnostics.add("update_norm", self._step_update_norm_sq.sqrt())
        self._step_update_norm_sq = None

        self._pending_matrix_updates.clear()
        return loss

    def _prepare_matrix_update(self, p: Tensor, grad: Tensor, group: dict) -> MatrixUpdate:
        state = self.state[p]
        projector = self._projector_from_state(p, group, state)

        if (
            projector.is_initialized
            and self._basis_update_due(group)
            and state.get("projected_exp_avg") is not None
        ):
            # Steady state, and after the first step this is every step at the
            # default cadence: one fused kernel per side does clipping, the
            # held-frame projection, the Oja tangent, and the moment update in a
            # single pass.
            return self._prepare_initialized_tracker_update(p, grad, projector, state, group)

        # Two cold cases are left, and neither wants a tangent: the first step,
        # which fits the frame rather than moving it, and a step where the basis
        # update is not due under basis_update_interval > 1.
        #
        # Clipping happens on the RAW gradient, upstream of every consumer. A
        # blip batch (grad norm spiking ~180x) otherwise reaches the frame and
        # the moment at full size, and a basis that has adapted to one is
        # corrupt for as long as it takes the tracker to turn back. Observed: a
        # single step-54 blip collapsed alignment for the whole back half of a
        # 100-step run. A guard installed downstream cannot protect upstream
        # state, which is why the projected-grad clip could not stop it and why
        # grad_clip_norm has no off switch.
        diagnostics = self._diagnostics_sink()
        self._record_incoming_grad_diagnostics(diagnostics, grad)
        grad, raw_grad_norm = self._sanitize_and_clip_grad_tensors(grad, float(group["grad_clip_norm"]))
        if not projector.is_initialized:
            self._initialize_projector(projector, grad, state)
        projected_grad = projector.project(grad)

        projected_exp_avg = state.get("projected_exp_avg")
        state["step"] = state.get("step", 0) + 1
        if projected_exp_avg is None:
            projected_exp_avg = torch.zeros_like(projected_grad)
        projected_exp_avg.mul_(group["beta"]).add_(projected_grad, alpha=1.0 - group["beta"])
        state["projected_exp_avg"] = projected_exp_avg
        self._record_projection_diagnostics(diagnostics, projected_grad, projected_exp_avg, group["beta"])

        return MatrixUpdate(
            param=p,
            projector=projector,
            projected_exp_avg=projected_exp_avg,
            original_shape=tuple(p.shape),
            oja_tangent=None,
            raw_grad_norm=raw_grad_norm,
        )

    def _prepare_initialized_tracker_update(
        self,
        p: Tensor,
        grad: Tensor,
        projector: SubspaceProjector,
        state: dict,
        group: dict,
    ) -> MatrixUpdate:
        diagnostics = self._diagnostics_sink()
        self._record_incoming_grad_diagnostics(diagnostics, grad)
        state["step"] = state.get("step", 0) + 1
        basis = projector.basis
        assert basis is not None
        is_right = projector._basis_side() is ProjectionSide.RIGHT
        prepare = (
            self._compiled_prepare_tracker_right if is_right else self._compiled_prepare_tracker_left
        ) or (
            self._prepare_tracker_right_tensors if is_right else self._prepare_tracker_left_tensors
        )
        projected_grad, oja_tangent, raw_grad_norm = prepare(
            grad,
            basis,
            state["projected_exp_avg"],
            float(group["grad_clip_norm"]),
            float(group["beta"]),
        )
        projected_exp_avg = state["projected_exp_avg"]
        # Deliberately outside the compiled kernel: measuring in there would put
        # the reductions in the graph whether or not anyone asked for them.
        self._record_projection_diagnostics(diagnostics, projected_grad, projected_exp_avg, group["beta"])

        return MatrixUpdate(
            param=p,
            projector=projector,
            projected_exp_avg=projected_exp_avg,
            original_shape=tuple(p.shape),
            oja_tangent=oja_tangent,
            raw_grad_norm=raw_grad_norm,
        )

    @staticmethod
    def _record_incoming_grad_diagnostics(diagnostics: DiagnosticsAccumulator | None, grad: Tensor) -> None:
        """Count matrix gradients that arrive non-finite.

        Stays a device tensor: an `isfinite().all()` read here would be a sync
        on the hot path, which is exactly what the sanitizer avoids being.
        """

        if diagnostics is None:
            return
        diagnostics.add_total("nonfinite_grads", (~torch.isfinite(grad)).any().float())

    @staticmethod
    def _record_projection_diagnostics(
        diagnostics: DiagnosticsAccumulator | None,
        projected_grad: Tensor,
        projected_exp_avg: Tensor | None,
        beta: float,
    ) -> None:
        if diagnostics is None:
            return
        projected_grad_norm = projected_grad.float().norm()
        diagnostics.add("projected_grad_norm", projected_grad_norm)
        if projected_exp_avg is None:
            return
        moment = projected_exp_avg.float()
        diagnostics.add("grad_to_moment_ratio", projected_grad_norm / moment.norm().clamp_min(1e-12))
        # Agreement is asked of the moment as it stood BEFORE this step's blend,
        # recovered exactly from the post-update moment. The post-update moment
        # already contains `1 - beta` of this very gradient, which would make the
        # measurement partly self-referential -- it would report agreement with
        # itself. Reconstructing costs one axpy and keeps the fused kernel free
        # of telemetry.
        if beta <= 0.0:
            # No memory, so there is no prior moment to agree or disagree with:
            # the moment *is* this gradient. Reporting 1.0 would be true and
            # useless; reporting nothing keeps the metric meaning one thing.
            return
        moment_before = (moment - (1.0 - beta) * projected_grad.float()) / beta
        diagnostics.add(
            "grad_moment_cosine",
            (projected_grad.float() * moment_before).sum()
            / (projected_grad_norm * moment_before.norm()).clamp_min(1e-12),
        )

    @staticmethod
    def _prepare_tracker_right_tensors(
        grad: Tensor,
        basis: Tensor,
        projected_exp_avg: Tensor,
        grad_clip_norm: float,
        beta: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """One fused pass: clip, project in the held frame, aim, accumulate.

        Both consumers -- the Oja tangent that steers the basis and the
        projected moment that becomes the update -- read the same raw clipped
        gradient, so the held-frame projection is computed once and shared.
        """

        grad, raw_grad_norm = UsuiTrack._sanitize_and_clip_grad_tensors(grad, grad_clip_norm)
        projected_grad = grad @ basis.mT
        work = grad.float()
        frame = basis.float().mT
        low = projected_grad.float()
        action = work.mT @ low
        rayleigh = frame.mT @ action
        rayleigh = 0.5 * (rayleigh + rayleigh.mT)
        tangent = action - frame @ rayleigh
        tangent = tangent / rayleigh.diagonal().mean().clamp_min(1e-12)
        projected_exp_avg.mul_(beta).add_(projected_grad, alpha=1.0 - beta)
        return projected_grad, tangent, raw_grad_norm

    @staticmethod
    def _prepare_tracker_left_tensors(
        grad: Tensor,
        basis: Tensor,
        projected_exp_avg: Tensor,
        grad_clip_norm: float,
        beta: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Left-side twin of `_prepare_tracker_right_tensors`."""

        grad, raw_grad_norm = UsuiTrack._sanitize_and_clip_grad_tensors(grad, grad_clip_norm)
        projected_grad = basis.mT @ grad
        work = grad.float()
        frame = basis.float()
        low = projected_grad.float()
        action = work @ low.mT
        rayleigh = frame.mT @ action
        rayleigh = 0.5 * (rayleigh + rayleigh.mT)
        tangent = action - frame @ rayleigh
        tangent = tangent / rayleigh.diagonal().mean().clamp_min(1e-12)
        projected_exp_avg.mul_(beta).add_(projected_grad, alpha=1.0 - beta)
        return projected_grad, tangent, raw_grad_norm

    @staticmethod
    def _sanitize_and_clip_grad_tensors(grad: Tensor, grad_clip_norm: float) -> tuple[Tensor, Tensor]:
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        raw_grad_norm = grad.float().norm().detach()
        clip_scale = (grad.new_tensor(grad_clip_norm) / raw_grad_norm.clamp_min(1e-12)).clamp(max=1.0)
        return grad.mul(clip_scale), raw_grad_norm

    def _apply_matrix_update_buckets(self, entries: list[MatrixUpdate], group: dict) -> None:
        if not entries:
            return

        buckets: dict[tuple, list[MatrixUpdate]] = {}
        for entry in entries:
            projected_exp_avg = entry.projected_exp_avg
            key = (tuple(projected_exp_avg.shape), entry.original_shape)
            buckets.setdefault(key, []).append(entry)

        for bucket_entries in buckets.values():
            if len(bucket_entries) == 1:
                update_hats = [
                    self._orthogonalize_update_runtime(
                        bucket_entries[0].projected_exp_avg,
                        group,
                        bucket_entries[0].original_shape,
                    )
                ]
            else:
                stacked = torch.stack([entry.projected_exp_avg for entry in bucket_entries])
                stacked_update_hats = self._orthogonalize_update_runtime(stacked, group, bucket_entries[0].original_shape)
                update_hats = list(stacked_update_hats.unbind(0))
            for entry, update_hat in zip(bucket_entries, update_hats, strict=True):
                self._apply_matrix_update(entry, update_hat, group)

    def _apply_basis_updates(self, entries: list[MatrixUpdate], group: dict) -> None:
        pending = [entry for entry in entries if entry.oja_tangent is not None]
        if not pending:
            return

        buckets: dict[tuple, list[MatrixUpdate]] = {}
        for entry in pending:
            tangent = entry.oja_tangent
            assert tangent is not None
            key = (tangent.device, tangent.dtype, tangent.shape[1])
            buckets.setdefault(key, []).append(entry)

        step_size = self._basis_update_step_size(group)
        diagnostics = self._diagnostics_sink()
        for bucket_entries in buckets.values():
            tangents = [entry.oja_tangent for entry in bucket_entries]
            assert all(tangent is not None for tangent in tangents)
            grams = torch.stack([tangent.mT @ tangent for tangent in tangents if tangent is not None])
            grams = 0.5 * (grams + grams.mT)
            # Bare, deliberately. A relative Tikhonov jitter used to be applied
            # here to rescue a solver failure, and across two models, two ranks
            # and ~3300 basis updates it never once fired. A failing eigh now
            # fails, which is what you want from a decomposition whose result
            # steers the frame.
            eigenvalues, eigenvectors = torch.linalg.eigh(grams)
            self._record_basis_motion_diagnostics(diagnostics, eigenvalues, step_size)
            geometry_buckets: dict[tuple, list[int]] = {}
            for index, entry in enumerate(bucket_entries):
                tangent = entry.oja_tangent
                assert tangent is not None
                geometry_buckets.setdefault((tuple(tangent.shape), entry.projector._basis_side()), []).append(index)
            for (_shape, side), indices in geometry_buckets.items():
                selected_entries = [bucket_entries[index] for index in indices]
                frames = torch.stack([entry.projector.canonical_basis() for entry in selected_entries])
                selected_tangents = torch.stack([entry.oja_tangent for entry in selected_entries if entry.oja_tangent is not None])
                selected_values = eigenvalues[indices]
                selected_vectors = eigenvectors[indices]
                new_frames = SubspaceProjector.oja_geodesic_from_eigh(
                    frames,
                    selected_tangents,
                    selected_values,
                    selected_vectors,
                    step_size,
                )
                for entry, new_frame in zip(selected_entries, new_frames, strict=True):
                    basis = new_frame.mT if side is ProjectionSide.RIGHT else new_frame
                    entry.projector.basis = basis.to(
                        device=entry.projector.basis.device,
                        dtype=entry.projector.basis.dtype,
                    ).contiguous()
                    entry.projector.resolved_side = side
                    state = self.state[entry.param]
                    state["basis"] = entry.projector.basis
                    state["projection_side_is_right"] = side is ProjectionSide.RIGHT
        self._record_basis_lag(pending, group)

    def _record_basis_lag(self, entries: list[MatrixUpdate], group: dict) -> None:
        """Principal-angle distance between each sampled frame and its own past.

        The one read that can show convergence. `rotation_rad_sum` cannot: it is
        floored by single-batch noise in the aim, so a frame orbiting a fixed
        point at constant radius reports the same rotation forever as one that
        is genuinely still travelling. Comparing the frame to *itself* some
        refreshes back cancels that -- an orbit returns, a drift does not.

        Reported as an RMS sine of the principal angles, which is exact and
        needs no decomposition: with `C = Q_old^T Q_now`, the residual
        `R = Q_now - Q_old C` has `||R||_F^2 = sum_i sin^2(theta_i)`. Forming
        `R` by subtracting the vectors rather than the squared cosines is what
        keeps it well conditioned at the small angles that matter here. It goes
        to zero if and only if every tracked plane has stopped moving.

        Sampled over `diagnostics_lag_matrices` parameters, not carried per
        parameter: this is the only diagnostic that costs persistent bytes
        (`[d,r]` per sampled matrix), and the VRAM-first rule bans that in the
        update, not in an opt-in instrument over a handful of tensors. The
        snapshots live outside `self.state` so they never enter a checkpoint.
        """

        diagnostics = self._diagnostics_sink()
        if diagnostics is None or not self.diagnostics_lag_enabled:
            return
        if group["basis_update_step"] % max(1, self.diagnostics_lag_interval):
            return
        if self._lag_sampled is None:
            # Deterministic sample, so two runs of the same config measure the
            # same tensors and their curves are comparable.
            matrices = [param for candidate in self.param_groups for param in candidate["params"]]
            self._lag_sampled = set(matrices[: max(0, self.diagnostics_lag_matrices)])
        for entry in entries:
            if entry.param not in self._lag_sampled:
                continue
            frame = entry.projector.canonical_basis().float()
            previous = self._lag_snapshots.get(entry.param)
            self._lag_snapshots[entry.param] = frame.clone()
            if previous is None or previous.shape != frame.shape:
                continue
            residual = frame - previous @ (previous.mT @ frame)
            rank = frame.shape[1]
            diagnostics.add("basis_lag_rms_sin", residual.norm() / math.sqrt(rank))

    @staticmethod
    def _record_basis_motion_diagnostics(
        diagnostics: DiagnosticsAccumulator | None,
        eigenvalues: Tensor,
        step_size: float,
    ) -> None:
        """How far the frame turns, and how concentrated that turn is.

        Both come free from the eigenvalues the geodesic already needs. The
        angle is the sum of the per-plane rotations -- the tangent's nuclear
        norm scaled by the step size -- so it is a measure of total motion
        across all `r` planes, not of how far any one plane moves.
        Concentration, the leading eigenvalue's share of the trace, is what
        separates a confident drift from a frame spinning on its noise tail:
        the same total angle can be either.
        """

        if diagnostics is None:
            return
        spectrum = eigenvalues.clamp_min(0.0)
        rotation = step_size * spectrum.sqrt().sum(dim=-1)
        concentration = spectrum.amax(dim=-1) / spectrum.sum(dim=-1).clamp_min(1e-12)
        samples = int(spectrum.shape[0])
        diagnostics.add("rotation_rad_sum", rotation.sum(), count=samples)
        diagnostics.add("tangent_concentration", concentration.sum(), count=samples)

    @staticmethod
    def _basis_update_step_size(group: dict) -> float:
        return BASIS_UPDATE_STEP

    @staticmethod
    def _basis_update_due(group: dict) -> bool:
        return (group["matrix_step"] + 1) % group["basis_update_interval"] == 0

    def _apply_matrix_update(self, entry: MatrixUpdate, update_hat: Tensor, group: dict) -> None:
        param = entry.param
        stochastic = self.stochastic_rounding and wants_stochastic_rounding(param)
        decay = 1.0 - group["lr"] * group["weight_decay"] if group["weight_decay"] else None

        if not stochastic:
            update = entry.projector.project_back(update_hat).to(dtype=param.dtype)
            self._record_update_norm(update, group["lr"])
            if decay is not None:
                param.mul_(decay)
            param.add_(update, alpha=-group["lr"])
            return

        # Lift in fp32 and never round the update on the way in: the whole
        # point is that `lr * update` is a fraction of a bf16 ulp, so any
        # intermediate cast discards it before the write can decide what to do
        # with it. Weight decay is folded into the same fp32 accumulator so the
        # parameter is rounded exactly once per step, not twice.
        work = param.float()
        if decay is not None:
            work.mul_(decay)
        update = entry.projector.project_back(update_hat.float())
        self._record_update_norm(update, group["lr"])
        work.add_(update, alpha=-group["lr"])
        copy_stochastic_(param, work)

    def _record_update_norm(self, update: Tensor, lr: float) -> None:
        """Accumulate this matrix's contribution to the step's total motion.

        Kept as a running square across the step and rooted once in `step()`, so
        the reported number is the norm of the whole step's update rather than a
        mean over matrices of unrelated sizes.
        """

        if self._diagnostics_sink() is None:
            return
        contribution = (update.float().norm() * lr).square().detach()
        self._step_update_norm_sq = (
            contribution if self._step_update_norm_sq is None else self._step_update_norm_sq + contribution
        )

    def _initialize_projector(
        self,
        projector: SubspaceProjector,
        grad: Tensor,
        state: dict,
    ) -> None:
        projector.fit(grad)
        state["basis"] = projector.basis
        resolved_side = projector.resolved_side if projector.resolved_side is not None else projector.side
        state["projection_side_is_right"] = resolved_side is ProjectionSide.RIGHT
        old_projected_exp_avg = state.get("projected_exp_avg")
        if old_projected_exp_avg is not None:
            projected_shape = tuple(projector.project(grad).shape)
            if tuple(old_projected_exp_avg.shape) != projected_shape:
                state.pop("projected_exp_avg", None)

    @staticmethod
    def _expected_projected_grad_shape(p: Tensor, projector: SubspaceProjector) -> tuple[int, int]:
        if projector.basis is None:
            raise RuntimeError("basis is not initialized")
        side = projector.resolved_side if projector.resolved_side is not None else projector.side
        if side is ProjectionSide.AUTO:
            side = projector.effective_side(p)
        if side is ProjectionSide.RIGHT:
            return (p.shape[0], projector.basis.shape[0])
        return (projector.basis.shape[1], p.shape[1])

    @staticmethod
    def _orthogonalize_update(update: Tensor, _group: dict, original_shape: tuple[int, ...] | None = None) -> Tensor:
        return UsuiTrack._orthogonalize_aurora(update, original_shape)

    def _orthogonalize_update_runtime(self, update: Tensor, group: dict, original_shape: tuple[int, ...] | None = None) -> Tensor:
        if self._compiled_orthogonalize_update is None:
            return self._orthogonalize_update(update, group, original_shape)
        if original_shape is None or len(original_shape) < 2:
            raise ValueError("compiled UsuiTrack orthogonalization requires the original parameter shape")
        rows = int(original_shape[0])
        cols = int(math.prod(original_shape[1:]))
        return self._compiled_orthogonalize_update(
            update,
            rows,
            cols,
        )

    @staticmethod
    def _orthogonalize_aurora_muon_tensor(
        update: Tensor,
        original_rows: int,
        original_cols: int,
    ) -> Tensor:
        aurora_update = UsuiTrack._aurora_leverage_uniform_polar(update)
        return aurora_update * math.sqrt(max(1.0, original_rows / original_cols))

    @staticmethod
    def _orthogonalize_aurora(update: Tensor, original_shape: tuple[int, ...] | None) -> Tensor:
        aurora_update = UsuiTrack._aurora_leverage_uniform_polar(update)
        return UsuiTrack._scale_orthogonalized_update(
            update,
            aurora_update,
            ORTHOGONALIZATION_SCALE_MODE,
            original_shape,
        )

    @staticmethod
    def _aurora_leverage_uniform_polar(
        update: Tensor,
        eps: float = 1e-7,
    ) -> Tensor:
        """Aurora-style leverage-uniform polar direction for rectangular projected moments.

        Aurora: https://github.com/tilde-research/aurora-release (Tilde Research).
        Reimplemented from the method; not a runtime dependency.

        UsuiTrack owns momentum, LR, weight decay, and full-matrix Muon scaling. This
        helper extracts only Aurora's rectangular direction map: diagonally
        precondition a non-square matrix before polar/NS so the large-side row
        leverage approaches the Stiefel target. For wide matrices, transpose to
        tall form, balance, then transpose back, matching Aurora's convention.
        """

        if update.ndim < 2:
            raise ValueError(f"Aurora orthogonalization expects at least 2D input, got shape {tuple(update.shape)}")
        if update.shape[-2] == update.shape[-1]:
            return UsuiTrack._newton_schulz_polar(update)

        transposed = update.shape[-2] < update.shape[-1]
        work = update.mT if transposed else update
        work32 = work.float()
        rows, cols = work32.shape[-2:]
        target_row_sq = cols / rows
        diagonal = work32.norm(dim=-1, keepdim=True).clamp_min(eps).reciprocal()
        balanced = None
        for iteration in range(AURORA_PP_ITERATIONS):
            balanced = UsuiTrack._newton_schulz_polar(diagonal * work32).float()
            if iteration < AURORA_PP_ITERATIONS - 1:
                row_sq = balanced.square().sum(dim=-1, keepdim=True).clamp_min(eps * eps)
                diagonal = diagonal * (target_row_sq / row_sq).pow(AURORA_PP_BETA)
        assert balanced is not None
        result = balanced.mT if transposed else balanced
        return result.to(device=update.device, dtype=update.dtype)

    @staticmethod
    def _newton_schulz_polar(update: Tensor) -> Tensor:
        return UsuiTrack._batched_newton_schulz(update)

    @staticmethod
    def _batched_newton_schulz(update: Tensor, eps: float = 1e-7) -> Tensor:
        if update.ndim < 2:
            raise ValueError(f"Newton-Schulz orthogonalization expects at least 2D input, got shape {tuple(update.shape)}")
        work = update.float()
        work = work / work.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
        transposed = work.shape[-2] > work.shape[-1]
        x = work.mT if transposed else work

        for a, b, c in NEWTON_SCHULZ_COEFFICIENTS:
            gram = x @ x.mT
            y = c * gram
            y.diagonal(dim1=-2, dim2=-1).add_(b)
            y = y @ gram
            y.diagonal(dim1=-2, dim2=-1).add_(a)
            x = y @ x

        result = x.mT if transposed else x
        return result.to(device=update.device, dtype=update.dtype)

    @staticmethod
    def _scale_orthogonalized_update(
        original_update: Tensor,
        orthogonalized_update: Tensor,
        scale_mode: str,
        original_shape: tuple[int, ...] | None,
    ) -> Tensor:
        if scale_mode == "none":
            return orthogonalized_update
        if scale_mode == "scale":
            return orthogonalized_update * math.sqrt(max(1.0, original_update.shape[-2] / original_update.shape[-1]))
        if scale_mode == "graft":
            if original_update.ndim > 2:
                original_norm = original_update.norm(dim=(-2, -1), keepdim=True)
                ortho_norm = orthogonalized_update.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
                return orthogonalized_update * (original_norm / ortho_norm)
            return orthogonalized_update * (original_update.norm() / orthogonalized_update.norm().clamp(min=1e-6))
        if scale_mode == "muon":
            if original_shape is None or len(original_shape) < 2:
                raise ValueError("muon scale mode requires the original parameter shape")
            rows = original_shape[0]
            cols = math.prod(original_shape[1:])
            return orthogonalized_update * math.sqrt(max(1.0, rows / cols))
        raise AssertionError(f"unexpected orthogonalization scale mode: {scale_mode}")

    @staticmethod
    def _projector_from_state(p: Tensor, group: dict, state: dict) -> SubspaceProjector:
        projector = SubspaceProjector(
            rank=group["rank"],
            side=ProjectionSide(group["side"]),
        )
        basis = state.get("basis")
        if basis is not None:
            projector.basis = basis
            is_right = state.get("projection_side_is_right")
            if is_right is None:
                projector.resolved_side = projector.effective_side(p)
            else:
                projector.resolved_side = ProjectionSide.RIGHT if is_right else ProjectionSide.LEFT
        return projector
