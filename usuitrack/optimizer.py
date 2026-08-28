from __future__ import annotations

import math
import warnings
import weakref
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer

from .projector import MAX_BASIS_ROTATION, ProjectionSide, SubspaceProjector
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
MIN_BASIS_UPDATE_STEP = 0.01


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
    default one-state basis tracker conditions every full matrix gradient and
    updates its live basis on the configured cadence.
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
        adafactor_beta2: float = 0.99,
        adafactor_eps: float = 1e-30,
        grad_clip_norm: float | None = 1.0,
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
        if not 0 <= adafactor_beta2 < 1:
            raise ValueError(f"adafactor_beta2 must be in [0, 1), got {adafactor_beta2}")
        if adafactor_eps <= 0:
            raise ValueError(f"adafactor_eps must be positive, got {adafactor_eps}")
        if grad_clip_norm is not None and grad_clip_norm <= 0:
            raise ValueError(f"grad_clip_norm must be positive when set, got {grad_clip_norm}")
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
            adafactor_beta2=adafactor_beta2,
            adafactor_eps=adafactor_eps,
            grad_clip_norm=grad_clip_norm,
            consume_grad=consume_grad,
            compile_tensor_kernels=compile_tensor_kernels,
            basis_update_interval=basis_update_interval,
            matrix_step=0,
            basis_update_step=0,
        )
        super().__init__(params, defaults)
        self._compiled_orthogonalize_update = torch.compile(UsuiTrack._orthogonalize_aurora_muon_tensor) if compile_tensor_kernels else None
        self._compiled_prepare_tracker_adafactor_right = (
            torch.compile(UsuiTrack._prepare_tracker_adafactor_right_tensors, dynamic=True)
            if compile_tensor_kernels
            else None
        )
        self._compiled_prepare_tracker_adafactor_left = (
            torch.compile(UsuiTrack._prepare_tracker_adafactor_left_tensors, dynamic=True)
            if compile_tensor_kernels
            else None
        )
        self._pending_matrix_updates: dict[Tensor, MatrixUpdate] = {}
        self._matrix_grad_hook_handles = []
        self._matrix_param_groups: dict[Tensor, dict] = {}
        self.release_matrix_grads = release_matrix_grads
        self.stochastic_rounding = stochastic_rounding
        # Oja rotation telemetry. Kept per-device as on-device tensors so the
        # accumulation costs no host sync; only pop_basis_rotation_angle()
        # reads them, and that is meant to be called at logging cadence.
        self._basis_rotation_sum: dict[torch.device, Tensor] = {}
        self._basis_rotation_count: dict[torch.device, int] = {}
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
            # effective_rank() clamps rank to a quarter of the tracked
            # dimension, so a configured rank is a ceiling rather than a
            # promise. Not fatal -- those parameters simply track a smaller
            # subspace -- but the caller asked for something it is not getting,
            # so report it. One summary per group, not one warning per
            # parameter: on a real model the same clamp fires for hundreds of
            # weights and the per-parameter form buries the signal.
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
                    f"UsuiTrack: configured rank {group['rank']} exceeds a quarter of the "
                    f"tracked dimension for {sum(clamped.values())} of {len(group['params'])} "
                    f"matrix parameters on side {group['side']}; they fall back to a smaller "
                    f"rank ({detail}).",
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
        # A caller (e.g. an OOM-retry path) may need to bail out after prepare()
        # already mutated moving-average state for some params but before step()
        # applied it. That mutation can't be undone -- the Oja/adafactor moments
        # for those params will reflect a step that never landed on the weights --
        # but that's a one-step drift, not worth blocking a clean recovery over.
        self._pending_matrix_updates.clear()
        super().zero_grad(set_to_none=set_to_none)

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
            self._apply_basis_updates(matrix_updates, group)
            self._apply_matrix_update_buckets(matrix_updates, group)

        self._pending_matrix_updates.clear()
        return loss

    def _prepare_matrix_update(self, p: Tensor, grad: Tensor, group: dict) -> MatrixUpdate:
        state = self.state[p]
        projector = self._projector_from_state(p, group, state)
        oja_tangent = None
        raw_grad_norm = None
        if self._can_use_initialized_tracker_adafactor_prepare(projector, state, group):
            return self._prepare_initialized_tracker_adafactor_update(
                p,
                grad,
                projector,
                state,
                group,
            )
        else:
            # Sync-free non-finite guard. A NaN/inf batch otherwise poisons every
            # downstream consumer at once: the clip scale (NaN norm -> NaN scale ->
            # whole grad NaN), adafactor's row/col vars, the tangent buffer, and
            # eventually the refresh SVD -- and an `isfinite().all()` branch would be
            # a device->host sync in the hot path. Zeroing non-finite elements drops
            # their contribution for one step instead.
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            # Clip the RAW gradient before it reaches adafactor. A blip batch (grad
            # norm spiking ~180x) otherwise poisons the adafactor row/col second
            # moment: grad_sq of the blip is ~30000x normal, and with beta2=0.99 that
            # spike decays over ~100 steps, over-dampening whole gradient directions
            # for the entire window -- the basis then tracks a *starved* residual and
            # each subsequent refresh adapts to the corruption (observed: a single
            # step-54 blip collapsed alignment for the whole back half of a 100-step
            # run). Clipping here protects adafactor's state, the basis refresh, and
            # the projection in one place -- upstream of everything, which is why the
            # projected-grad clip (downstream, moment-only) could not stop it.
            raw_grad_norm = grad.float().norm().detach()
            grad_clip_norm = group.get("grad_clip_norm")
            if grad_clip_norm is not None:
                clip_scale = (grad.new_tensor(float(grad_clip_norm)) / raw_grad_norm.clamp_min(1e-12)).clamp(max=1.0)
                grad = grad.mul(clip_scale)
            grad = self._adafactor_dampen_full_grad(grad, group, state)
            held_projected_grad = projector.project(grad) if projector.is_initialized else None
            if held_projected_grad is None:
                self._initialize_projector(projector, grad, state)
                projected_grad = projector.project(grad)
            else:
                # The held-frame projection supplies both the projected moment
                # update and Oja's covariance action. Basis motion is deferred
                # until every tangent can be batched in step().
                if self._basis_update_due(group):
                    oja_tangent = projector.oja_tangent(grad, projected=held_projected_grad)
                projected_grad = held_projected_grad

        projected_exp_avg = state.get("projected_exp_avg")
        state["step"] = state.get("step", 0) + 1
        if projected_exp_avg is None:
            projected_exp_avg = torch.zeros_like(projected_grad)
        projected_exp_avg.mul_(group["beta"]).add_(projected_grad, alpha=1.0 - group["beta"])
        state["projected_exp_avg"] = projected_exp_avg

        return MatrixUpdate(
            param=p,
            projector=projector,
            projected_exp_avg=projected_exp_avg,
            original_shape=tuple(p.shape),
            oja_tangent=oja_tangent,
            raw_grad_norm=raw_grad_norm,
        )

    @staticmethod
    def _can_use_initialized_tracker_adafactor_prepare(
        projector: SubspaceProjector,
        state: dict,
        _group: dict,
    ) -> bool:
        return (
            projector.is_initialized
            and UsuiTrack._basis_update_due(_group)
            and _group.get("grad_clip_norm") is not None
            and state.get("adafactor_row_var") is not None
            and state.get("adafactor_col_var") is not None
            and state.get("projected_exp_avg") is not None
        )

    def _prepare_initialized_tracker_adafactor_update(
        self,
        p: Tensor,
        grad: Tensor,
        projector: SubspaceProjector,
        state: dict,
        group: dict,
    ) -> MatrixUpdate:
        adafactor_step = state.get("adafactor_step", 0) + 1
        state["adafactor_step"] = adafactor_step
        state["step"] = state.get("step", 0) + 1
        basis = projector.basis
        assert basis is not None
        side = projector._basis_side()
        prepare = (
            self._compiled_prepare_tracker_adafactor_right
            if side is ProjectionSide.RIGHT
            else self._compiled_prepare_tracker_adafactor_left
        )
        if prepare is None:
            prepare = (
                self._prepare_tracker_adafactor_right_tensors
                if side is ProjectionSide.RIGHT
                else self._prepare_tracker_adafactor_left_tensors
            )
        _conditioned_grad, oja_tangent, raw_grad_norm = prepare(
            grad,
            basis,
            state["adafactor_row_var"],
            state["adafactor_col_var"],
            state["projected_exp_avg"],
            adafactor_step,
            float(group["grad_clip_norm"]),
            float(group["adafactor_beta2"]),
            float(group["adafactor_eps"]),
            float(group["beta"]),
        )
        projected_exp_avg = state["projected_exp_avg"]

        return MatrixUpdate(
            param=p,
            projector=projector,
            projected_exp_avg=projected_exp_avg,
            original_shape=tuple(p.shape),
            oja_tangent=oja_tangent,
            raw_grad_norm=raw_grad_norm,
        )

    @staticmethod
    def _prepare_tracker_adafactor_right_tensors(
        grad: Tensor,
        basis: Tensor,
        row_var: Tensor,
        col_var: Tensor,
        projected_exp_avg: Tensor,
        adafactor_step: int,
        grad_clip_norm: float,
        adafactor_beta2: float,
        adafactor_eps: float,
        beta: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        grad, raw_grad_norm = UsuiTrack._sanitize_and_clip_grad_tensors(grad, grad_clip_norm)
        conditioned_grad = UsuiTrack._adafactor_dampen_tensors(
            grad,
            row_var,
            col_var,
            adafactor_step,
            adafactor_beta2,
            adafactor_eps,
        )
        projected_grad = conditioned_grad @ basis.mT
        work = conditioned_grad.float()
        frame = basis.float().mT
        low = projected_grad.float()
        action = work.mT @ low
        rayleigh = frame.mT @ action
        rayleigh = 0.5 * (rayleigh + rayleigh.mT)
        tangent = action - frame @ rayleigh
        tangent = tangent / rayleigh.diagonal().mean().clamp_min(1e-12)
        projected_exp_avg.mul_(beta).add_(projected_grad, alpha=1.0 - beta)
        return conditioned_grad, tangent, raw_grad_norm

    @staticmethod
    def _prepare_tracker_adafactor_left_tensors(
        grad: Tensor,
        basis: Tensor,
        row_var: Tensor,
        col_var: Tensor,
        projected_exp_avg: Tensor,
        adafactor_step: int,
        grad_clip_norm: float,
        adafactor_beta2: float,
        adafactor_eps: float,
        beta: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        grad, raw_grad_norm = UsuiTrack._sanitize_and_clip_grad_tensors(grad, grad_clip_norm)
        conditioned_grad = UsuiTrack._adafactor_dampen_tensors(
            grad,
            row_var,
            col_var,
            adafactor_step,
            adafactor_beta2,
            adafactor_eps,
        )
        projected_grad = basis.mT @ conditioned_grad
        work = conditioned_grad.float()
        frame = basis.float()
        low = projected_grad.float()
        action = work @ low.mT
        rayleigh = frame.mT @ action
        rayleigh = 0.5 * (rayleigh + rayleigh.mT)
        tangent = action - frame @ rayleigh
        tangent = tangent / rayleigh.diagonal().mean().clamp_min(1e-12)
        projected_exp_avg.mul_(beta).add_(projected_grad, alpha=1.0 - beta)
        return conditioned_grad, tangent, raw_grad_norm

    @staticmethod
    def _sanitize_and_clip_grad_tensors(grad: Tensor, grad_clip_norm: float) -> tuple[Tensor, Tensor]:
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        raw_grad_norm = grad.float().norm().detach()
        clip_scale = (grad.new_tensor(grad_clip_norm) / raw_grad_norm.clamp_min(1e-12)).clamp(max=1.0)
        return grad.mul(clip_scale), raw_grad_norm

    @staticmethod
    def _adafactor_dampen_tensors(
        grad: Tensor,
        row_var: Tensor,
        col_var: Tensor,
        step: int,
        beta2: float,
        eps: float,
    ) -> Tensor:
        grad32 = grad.float()
        grad_sq = grad32.square() + eps
        row_var.mul_(beta2).add_(grad_sq.mean(dim=1), alpha=1.0 - beta2)
        col_var.mul_(beta2).add_(grad_sq.mean(dim=0), alpha=1.0 - beta2)
        bias_correction = 1.0 - beta2**step
        row_hat = row_var / bias_correction
        col_hat = col_var / bias_correction
        # Reconstruct the inverse factored RMS as two broadcast vectors rather
        # than a second matrix-sized tensor, then restore the raw gradient RMS.
        # The latter keeps Oja's quadratic covariance scale comparable to the
        # unconditioned gradient while retaining Adafactor's SNR reweighting.
        mean_row = row_hat.mean().clamp_min(eps)
        row_scale = (mean_row / row_hat.clamp_min(eps)).sqrt().unsqueeze(1)
        col_scale = col_hat.clamp_min(eps).rsqrt().unsqueeze(0)
        dampened = grad32 * row_scale * col_scale
        grad_rms = grad32.square().mean().sqrt().clamp_min(eps)
        return (dampened * grad_rms).to(dtype=grad.dtype)

    @staticmethod
    def _adafactor_dampen_full_grad(grad: Tensor, group: dict, state: dict) -> Tensor:
        """Adafactor-style row/col factored second moment on the full gradient,
        applied before basis tracking and before projection so the same dampened
        gradient feeds both consumers, then still passed through the same
        first-moment EMA as ema mode.
        """

        beta2 = group["adafactor_beta2"]
        eps = group["adafactor_eps"]
        step = state.get("adafactor_step", 0) + 1
        state["adafactor_step"] = step

        row_var = state.get("adafactor_row_var")
        col_var = state.get("adafactor_col_var")
        if row_var is None:
            row_var = torch.zeros(grad.shape[0], device=grad.device, dtype=torch.float32)
            col_var = torch.zeros(grad.shape[1], device=grad.device, dtype=torch.float32)
        state["adafactor_row_var"] = row_var
        state["adafactor_col_var"] = col_var
        return UsuiTrack._adafactor_dampen_tensors(grad, row_var, col_var, step, beta2, eps)

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
            # Sanitize once, here, where every tangent producer converges --
            # the projector method and both fused kernels. A non-finite tangent
            # otherwise reaches the geodesic through `tangent @ eigenvectors`
            # even when the Gram it built was cleaned up, because the two are
            # separate reads of the same tensor. Sync-free, like the gradient
            # guard in _prepare_matrix_update.
            tangent = torch.nan_to_num(tangent, nan=0.0, posinf=0.0, neginf=0.0)
            entry.oja_tangent = tangent
            key = (tangent.device, tangent.dtype, tangent.shape[1])
            buckets.setdefault(key, []).append(entry)

        step_size = self._basis_update_step_size(group)
        for bucket_entries in buckets.values():
            tangents = [entry.oja_tangent for entry in bucket_entries]
            assert all(tangent is not None for tangent in tangents)
            grams = torch.stack([tangent.mT @ tangent for tangent in tangents if tangent is not None])
            grams = 0.5 * (grams + grams.mT)
            # A near-zero-curvature direction makes the tangent's
            # `/ rayleigh.diagonal().mean().clamp_min(1e-12)` normalization
            # blow it up to a huge magnitude, which pushes its Gram matrix to
            # an extreme eigenvalue spread and makes the batched eigh solver
            # fail to converge on that one matrix in the batch. A small
            # relative Tikhonov jitter fixes the conditioning; the resulting
            # over-large rotation is bounded separately, by MAX_BASIS_ROTATION
            # in oja_geodesic_from_eigh.
            rank_dim = grams.shape[-1]
            trace = grams.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-12)
            jitter = (1e-6 * trace / rank_dim).view(-1, 1, 1)
            eye = torch.eye(rank_dim, device=grams.device, dtype=grams.dtype)
            grams = grams + jitter * eye
            try:
                eigenvalues, eigenvectors = torch.linalg.eigh(grams)
            except torch.linalg.LinAlgError:
                # Jitter isn't always enough -- some matrix in this batch is
                # still too ill-conditioned for eigh to converge. Retry
                # one-by-one so a single degenerate parameter doesn't block
                # the basis update for everything else in the bucket; a
                # parameter that still fails on its own just bails on this
                # step's basis update (keeps its current frame) instead of
                # forcing a geodesic move through garbage eigenvectors.
                values_list: list[Tensor] = []
                vectors_list: list[Tensor] = []
                keep_mask: list[bool] = []
                for i in range(grams.shape[0]):
                    try:
                        vals, vecs = torch.linalg.eigh(grams[i])
                    except torch.linalg.LinAlgError:
                        warnings.warn(
                            "UsuiTrack: skipping this step's basis update for "
                            "a matrix parameter -- its Gram matrix is too "
                            "ill-conditioned for eigh to converge even after "
                            "regularization.",
                            stacklevel=2,
                        )
                        keep_mask.append(False)
                        continue
                    values_list.append(vals)
                    vectors_list.append(vecs)
                    keep_mask.append(True)
                if not any(keep_mask):
                    continue
                bucket_entries = [entry for entry, keep in zip(bucket_entries, keep_mask) if keep]
                eigenvalues = torch.stack(values_list)
                eigenvectors = torch.stack(vectors_list)
            self._accumulate_basis_rotation(eigenvalues, step_size)
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

    def _accumulate_basis_rotation(self, eigenvalues: Tensor, step_size: float) -> None:
        """Record how far this step rotated the Oja frames, in radians.

        The principal angles between the old frame and the geodesic-moved one
        are exactly ``min(step_size * sqrt(eigenvalue_i), MAX_BASIS_ROTATION)``:
        the tangent is the frame-orthogonal residual ``action - frame @
        rayleigh``, so the geodesic turns strictly in the frame's complement and
        the rotation angles fall straight out of the eigendecomposition that
        _apply_basis_updates already computed. No extra matmul, and nothing
        leaves the device -- the per-matrix RMS angle is summed into an
        on-device accumulator that only pop_basis_rotation_angle() reads.

        Mirrors the clamp in oja_geodesic_from_eigh deliberately: this reports
        the rotation the frame actually took, not the one the tangent asked
        for. A reading pinned near MAX_BASIS_ROTATION is therefore itself the
        signal that the clamp is saturating.
        """
        angles = (float(step_size) * eigenvalues.clamp_min(0.0).sqrt()).clamp(max=MAX_BASIS_ROTATION)
        per_matrix = angles.pow(2).mean(dim=-1).sqrt()
        device = per_matrix.device
        total = per_matrix.sum()
        if device in self._basis_rotation_sum:
            self._basis_rotation_sum[device] += total
        else:
            self._basis_rotation_sum[device] = total
        # shape[0] is static metadata, not a value read -- no sync here.
        self._basis_rotation_count[device] = self._basis_rotation_count.get(device, 0) + per_matrix.shape[0]

    def pop_basis_rotation_angle(self) -> float | None:
        """Mean per-matrix RMS Oja rotation angle (radians) since the last call.

        Returns None when no basis update has happened since. This is the only
        place the accumulator is read back to the host, so call it at logging
        cadence, never per step.
        """
        count = sum(self._basis_rotation_count.values())
        if count == 0:
            return None
        total = sum(value.item() for value in self._basis_rotation_sum.values())
        self._basis_rotation_sum.clear()
        self._basis_rotation_count.clear()
        return total / count

    @staticmethod
    def _basis_update_step_size(group: dict) -> float:
        return max(MIN_BASIS_UPDATE_STEP, 1.0 / group["basis_update_step"])

    @staticmethod
    def _basis_update_due(group: dict) -> bool:
        return (group["matrix_step"] + 1) % group["basis_update_interval"] == 0

    def _apply_matrix_update(self, entry: MatrixUpdate, update_hat: Tensor, group: dict) -> None:
        param = entry.param
        stochastic = self.stochastic_rounding and wants_stochastic_rounding(param)
        decay = 1.0 - group["lr"] * group["weight_decay"] if group["weight_decay"] else None

        if not stochastic:
            update = entry.projector.project_back(update_hat).to(dtype=param.dtype)
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
        work.add_(entry.projector.project_back(update_hat.float()), alpha=-group["lr"])
        copy_stochastic_(param, work)

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
