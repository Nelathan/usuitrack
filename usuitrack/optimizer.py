from __future__ import annotations

import math
import warnings
import weakref
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer

from .diagnostics import DiagnosticsAccumulator, RankCalibrator
from .projector import ProjectionSide, SubspaceProjector
from .stochastic import copy_stochastic_, wants_stochastic_rounding
AURORA_PP_ITERATIONS = 1
AURORA_PP_BETA = 0.5
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
# settled" from "the clock ran out". Constant step, so what `transport_speed`
# reports is the tracker's own residual and not a schedule -- though the speed is
# now read from the frames rather than from `sigma`, which is what keeps that
# true under any experiment that transforms the tangent between the two.
GEODESIC_STEPSIZE = 0.01

# Control arm for the ortho-first integration: put the second polar map back, so
# the moment contributes direction only and the step is restored to full size
# however much the average cancelled. It isolates the spectral-fairness half of
# ortho-first (which survives here) from the agreement-as-magnitude half (which
# does not), and exists to be deleted with whichever arm loses.
REORTHOGONALIZE_MOMENT = False



@dataclass
class PreparedGrad:
    """One micro-batch's matrix gradient, projected in the frame the step holds.

    Produced once per `backward()` per matrix. Under gradient accumulation there
    are `k` of these per optimizer step and `accumulate()` folds them into the
    `MatrixUpdate` below one at a time, so this type never holds more than one
    micro-batch's tensors and the full matrix gradient is still freed as soon as
    it has been projected.
    """

    param: Tensor
    projector: SubspaceProjector
    projected_grad: Tensor
    original_shape: tuple[int, ...]
    oja_tangent: Tensor | None = None
    raw_grad_norm: Tensor | None = None


@dataclass
class MatrixUpdate:
    """One optimizer step's accumulated update for one matrix.

    Every field summed over micro-batches carries that micro-batch's importance
    weight, and `_resolve_accumulated` divides by `weight` before anything reads
    them, so the moment and the geodesic see a weighted *mean* and not a total.
    Without accumulation there is one micro-batch at weight 1.0 and this is the
    single step it always was.
    """

    param: Tensor
    projector: SubspaceProjector
    original_shape: tuple[int, ...]
    # Sum of `w_i * polar(projected grad_i)`, at the gradient's dtype until the
    # final fold promotes it to fp32 -- see `_fold_into`. The polar map runs per
    # micro-batch, ahead of this average -- see `_orthogonalized_directions`.
    direction: Tensor
    weight: float | Tensor
    projected_grad_norm: Tensor
    micro_batches: int = 0
    oja_tangent: Tensor | None = None
    raw_grad_norm: Tensor | None = None
    # `sum w_i^2 ||d_i||^2` and `sum w_i^2`. Together with `direction` and
    # `weight` these recover the mean pairwise cosine between the micro-batch
    # directions without keeping the directions -- see
    # `_record_micro_batch_agreement`.
    direction_square_sum: Tensor | None = None
    weight_square_sum: float | Tensor = 0.0
    # Filled by `_integrate_step`, which is where the averaged direction enters
    # the moment; `_commit_moments` rounds it into state.
    projected_exp_avg: Tensor | None = None
    transport_speed: Tensor | None = None
    # The in-span rotation between the frame this step's moment was measured in
    # and the frame it will be read in next step. `None` when the basis did not
    # move, which is the only case where identity transport is exact on its own.
    frame_rotation: Tensor | None = None


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
        beta: float = 0.9,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        rank: int = 32,
        side: ProjectionSide | str = ProjectionSide.AUTO,
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
            consume_grad=consume_grad,
            compile_tensor_kernels=compile_tensor_kernels,
            basis_update_interval=basis_update_interval,
            matrix_step=0,
            basis_update_step=0,
        )
        super().__init__(params, defaults)
        # `torch.compile` wraps the same function the eager path calls, never a
        # second copy of the maths -- see AGENTS.md, "No parallel compiled and
        # uncompiled implementations of the same maths".
        self._compiled_orthogonalize_update = (
            torch.compile(UsuiTrack._orthogonalize_update) if compile_tensor_kernels else None
        )
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
        # Two stages, and the split is what makes gradient accumulation cheap.
        # `_pending_matrix_updates` holds at most one micro-batch, cleared by
        # every `accumulate()`; `_accumulated` holds the weighted sums that
        # `step()` applies. Neither enters a checkpoint: they are alive only
        # between a backward and the step that consumes it.
        self._pending_matrix_updates: dict[Tensor, PreparedGrad] = {}
        self._accumulated: dict[Tensor, MatrixUpdate] = {}
        # "off", "core" or "full". The line between the two live tiers is state
        # and flops, not usefulness: `core` is everything derivable from tensors
        # the step already formed, so it can be left on for a whole training run
        # without a decision. `full` adds the reads that need a frame snapshot
        # -- `transport_lag`, `transport_curve`, `transport_spin` -- which is
        # `[d,r]` over `diagnostics_lag_matrices` sampled matrices, a fixed cost
        # that does not grow with the model. Off is one attribute read.
        self._diagnostics_tier = "off"
        self.diagnostics_lag_matrices = 32
        # The window the snapshot spans. This is also the projected moment's
        # memory, so `1 / (1 - beta)` is the interval at which `transport_lag`
        # reads the moment's own smear rather than an arbitrary window.
        self.diagnostics_lag_interval = 10
        self._lag_snapshots: dict[Tensor, Tensor] = {}
        self._lag_path: dict[Tensor, Tensor] = {}
        self._lag_sampled: set[Tensor] | None = None
        self._diagnostics: DiagnosticsAccumulator | None = None
        # Off by default and independent of the diagnostics tier: set it to a
        # `RankCalibrator` before a run at an oversized rank to collect the
        # live-plane counts a rank table is then sized from by hand. Groups need
        # a `calibration_label` key for their counts to be bucketed.
        self.rank_calibrator: RankCalibrator | None = None
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
        # or accumulate() has consumed gradients but before step() applied them.
        # Neither writes the moment or the frame -- both happen in step() -- so
        # what is lost is the gradients themselves, plus a basis fitted on a
        # first step. Not worth killing a run over, so warn and drop the work.
        outstanding = len(self._pending_matrix_updates) + len(self._accumulated)
        if outstanding:
            warnings.warn(
                f"UsuiTrack: zero_grad() discarded prepared updates for "
                f"{outstanding} matrix parameters. Those gradients never reached "
                f"the weights.",
                stacklevel=2,
            )
        self._pending_matrix_updates.clear()
        self._accumulated.clear()
        super().zero_grad(set_to_none=set_to_none)

    DIAGNOSTIC_TIERS = ("off", "core", "full")

    @property
    def diagnostics(self) -> str:
        return self._diagnostics_tier

    @diagnostics.setter
    def diagnostics(self, tier: str) -> None:
        """Validated, because the failure this prevents is silent.

        The tiers are read as string comparisons on the hot path, so a typo does
        not raise -- ``"ful"`` is not ``"off"``, which enables core, and it is not
        ``"full"``, which leaves out exactly the snapshot reads that were asked
        for. The run then looks fine and quietly omits its most expensive
        telemetry. This is a knob set by hand mid-experiment, so it gets a guard.

        Leaving ``"full"`` also drops the snapshot state. Those buffers are only
        meaningful as a comparison against a live window; kept across a tier
        change they would make the first reading after re-enabling a comparison
        against an arbitrarily old frame.
        """

        if tier not in self.DIAGNOSTIC_TIERS:
            raise ValueError(f"diagnostics must be one of {self.DIAGNOSTIC_TIERS}, got {tier!r}")
        if tier != "full":
            self._lag_snapshots.clear()
            self._lag_path.clear()
        self._diagnostics_tier = tier

    def _diagnostics_sink(self) -> DiagnosticsAccumulator | None:
        """The live accumulator, or None when telemetry is off.

        Every accumulation site goes through here and does nothing when the
        result is None, so a run with diagnostics off pays one attribute read
        per site and no device work at all.
        """

        if self._diagnostics_tier == "off":
            return None
        if self._diagnostics is None:
            self._diagnostics = DiagnosticsAccumulator()
        return self._diagnostics

    @torch.no_grad()
    def pop_diagnostics(self) -> dict[str, float]:
        """Telemetry accumulated since the last call, as a plain dict of floats.

        Set ``diagnostics`` to ``"core"`` or ``"full"`` and call this on the cadence the
        surrounding trainer already logs at -- every step is supported but not
        the intent, since this is where the accumulated device tensors are read
        back to the host. An empty dict means either telemetry is off or nothing
        has happened since the last read, and is always safe to skip logging.

        Two live tiers, split by cost rather than by usefulness.
        ``diagnostics = "core"`` is every read derivable from tensors the step
        already formed -- no state, no decomposition, no extra pass -- so it can
        be left on for a whole run without a decision. ``"full"`` adds the three
        reads that need a frame snapshot, which is ``[d,r]`` over
        ``diagnostics_lag_matrices`` sampled matrices: 16 MB at rank 128 on a
        1024-wide model, and a fixed cost that does not grow with the model
        because the sample count is fixed. It buys the only reads that can tell
        a converged frame from one orbiting a fixed point, so "full" is the tier
        to run when the question is about the tracker rather than the loss.

        Means are over samples (matrix x step) since the last read;
        ``nonfinite_grads`` is a total. What is here answers a specific question, in this
        order of load-bearing-ness:

        ``transport_speed``, ``transport_curve``, ``transport_spin``
            The frame's motion, read as three quantities that only mean
            something together. All three are per-plane RMS sines, so they are
            comparable to each other and across ranks.

            ``transport_speed`` is how far the subspace moved in one geodesic.
            ``transport_curve`` is `1 - lag / path` over the last
            ``diagnostics_lag_interval`` basis updates: the fraction of that
            travel which cancelled, zero for a straight drift and approaching
            one for a frame churning in place. ``transport_spin`` is rotation of
            the frame's columns *within* the span they already had -- motion
            that moves the subspace not at all and scrambles the projected
            moment one-for-one, since transport is the identity in these
            coordinates.

            How to read them. High speed is fine if curve is low: the frame is
            travelling, and it will slow as the aim converges. High curve is
            fine if speed is low: small motion that mostly cancels is a frame
            sitting on its fixed point. High speed *and* high curve is churn --
            the tracker is working hard and going nowhere, and it is integrating
            batch noise into the frame while it does. Low speed and low curve is
            ambiguous between a settled frame and a starved one, and spin is
            what separates them: a settled frame is still in every sense, while
            a frame with spin is quietly rotating its own coordinates under a
            moment that assumes they are fixed.

            ``transport_curve`` and ``transport_spin`` need the frame snapshot
            and so appear only at ``diagnostics = "full"``. ``transport_speed``
            does not: the frames before and after the geodesic are both already
            in hand, so it is one matmul and a norm.

            This replaced ``rotation_rad_sum``, which was `sum_i eta sigma_i` --
            the tangent's nuclear norm. That is a real quantity but not a
            distance: displacement follows ``||sigma||_2`` and it followed
            ``sum_i sigma_i``, so the two differ by `sqrt(participation)` and do
            not move together. It also grew with rank and shared its unit with
            nothing else here, which made it unreadable against the lag it was
            meant to be compared with.
        ``tangent_concentration``, ``tangent_effective_planes``
            Where that motion went, both read from the same eigenvalues.
            Concentration is the leading plane's share of the tangent's energy in
            ``[1/r, 1]`` -- the head of the spectrum. The plane count is the
            effective number of planes carrying it, ``(sum lambda)^2 / sum
            lambda^2`` in ``[1, r]`` -- the bulk. They separate because the
            spectrum is a power law with no edge: a high head does not imply a
            short tail. High concentration with few planes is a confident aim
            drifting; low concentration with many is a frame turning on a
            near-isotropic noise tail, which is a mechanism for integrating batch
            noise into the basis. They are also near-perfectly anti-correlated on
            a real run (``-0.95`` over two Anima runs), so a story in which one
            causes the other is a story about one number. Both are magnitude
            statistics, and the geodesic turns every live plane by ``eta``
            whatever the eigenvalue said -- so neither describes the motion, only
            the aim it was formed from.
        ``projected_grad_norm``, ``raw_grad_norm``, ``grad_capture``,
        ``moment_persistence``
            Scale of the gradient inside the frame, outside it, their ratio, and
            how much coherent signal the average holds once the
            incomplete-cancellation floor of an independent stream is subtracted.
            Only the ratio can say whether the basis catches the gradient: the
            projected norm alone falls with a leaner rank even when every kept
            plane catches as much as before. 0 is a white stream, 1 a direction
            held for the whole memory, negative anti-correlated. The raw norm
            ratio the step size follows is recoverable from it as
            ``p (1 - w^2) + w^2`` with ``w^2 = (1 - beta)/(1 + beta)``.
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
        ``micro_batch_agreement``
            Only present when a step accumulated two or more micro-batches. The
            mean pairwise cosine between their orthogonalized directions, which
            is the aim's signal-to-noise ratio ``|s|^2 / (|s|^2 + |n|^2)``: a
            pure-noise aim reads zero, so this is an absolute level and not a
            quantity needing a reference run. It is the only read here that
            compares independently-sampled batches rather than one batch against
            history, so it is the one that can say whether a frame has arrived or
            is orbiting on noise. It also prices accumulation's effect on the
            step: the mean of `k` disagreeing directions is shorter than each of
            them, so a low reading means a smaller step at the same ``lr``, which
            ``update_to_param_ratio`` will show.
        ``transport_lag``
            Only present at ``diagnostics = "full"``. The net distance each
            sampled frame covered over the last ``diagnostics_lag_interval``
            basis updates -- the numerator ``transport_curve`` divides. Zero iff
            every tracked plane has stopped. Read directly, it is also the
            projected moment's smear: under identity transport a contribution
            from `k` updates ago has been misaligned by exactly these principal
            angles, so setting ``diagnostics_lag_interval`` to the moment's
            memory ``1 / (1 - beta)`` makes this read the fraction of the
            moment's own history that no longer names the direction it was
            accumulated in.
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
    ) -> PreparedGrad:
        if param in self._pending_matrix_updates:
            raise RuntimeError(
                "matrix parameter is already prepared; call accumulate() between backward passes "
                "to fold each micro-batch into the pending step"
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
        """Raw full-gradient norms retained for telemetry after matrix grads are released.

        Reads whichever stage holds the current step's work: the pending
        micro-batch before `accumulate()`, and the accumulated weighted mean
        after it, so a caller that reads this between the last backward and
        `step()` sees the same quantity with or without accumulation.
        """

        norms = []
        for group in self.param_groups:
            for param in group["params"]:
                pending = self._pending_matrix_updates.get(param)
                if pending is not None and pending.raw_grad_norm is not None:
                    norms.append(pending.raw_grad_norm)
                    continue
                entry = self._accumulated.get(param)
                if entry is not None and entry.raw_grad_norm is not None:
                    norms.append(entry.raw_grad_norm / entry.weight)
        return tuple(norms)

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
    def accumulate(self, weight: float | Tensor = 1.0, final: bool = False) -> None:
        """Fold one micro-batch into the step `step()` will apply.

        Call it after every ``backward()``, then ``step()`` once. A loop that
        never calls it gets the single-batch step, folded by ``step()`` itself;
        `k` calls are gradient accumulation. The accumulation lives in projected
        space -- `[d, r]` per matrix against `[m, n]` for a gradient -- so a
        ``release_matrix_grads`` run still consumes and frees one full matrix
        gradient at a time.

        ``weight`` is the micro-batch's **share of the group's loss, not its
        gradient magnitude**: its loss token count, so the average weighs each
        micro-batch as one big batch would have. The weights normalize by their
        own sum, so their scale is free and 1.0 everywhere is the unweighted
        mean.

        ``final`` marks the last micro-batch of the step. The running sums are
        stored at the gradient's dtype only because they stay resident through
        the *next* backward; nothing follows the last one, so its fold is kept
        at full precision. ``step()`` folds whatever is still pending as final,
        which is what makes the un-accumulated step exactly the single-batch
        path, with no rounding in it.

        Three things hold across the group of micro-batches, and each is
        load-bearing.

        **The basis does not move.** Every micro-batch projects in one frame, so
        the sum of their Oja tangents is a sum of vectors in a single tangent
        space rather than an average over basepoints, which is what makes the
        arithmetic mean the geometrically correct one and needs no spherical
        machinery.

        **The polar map runs per micro-batch, ahead of the average.** Averaging
        raw projected gradients and orthogonalizing once would hand the step to
        whichever micro-batch was loudest -- the same failure the moment's own
        ordering exists to prevent.

        **The moment is not touched until `step()`.** It costs one accumulator
        per matrix and it buys honest telemetry: ``grad_moment_cosine`` and
        ``moment_persistence`` read the step that was applied, never a partial
        one.
        """

        self._validate_step_inputs()
        for group in self.param_groups:
            prepared: list[PreparedGrad] = []
            for p in group["params"]:
                if p not in self._pending_matrix_updates:
                    if p.grad is None:
                        continue
                    self._prepare_matrix_param(p)
                prepared.append(self._pending_matrix_updates.pop(p))
            if not prepared:
                continue
            directions = self._orthogonalized_directions(prepared)
            for entry, direction in zip(prepared, directions, strict=True):
                storage = torch.float32 if final else entry.projected_grad.dtype
                self._fold_micro_batch(entry, direction, weight, storage)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None and (
            self.release_matrix_grads or self._pending_matrix_updates or self._accumulated
        ):
            raise RuntimeError("optimizer closures cannot run while matrix updates are pending")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Whatever the caller did not fold in itself. A no-op when every
        # micro-batch already went through `accumulate()`, and the whole of the
        # un-accumulated path when none did.
        self.accumulate(final=True)
        self._step_update_norm_sq = None

        for group in self.param_groups:
            matrix_updates = [
                self._accumulated[p] for p in group["params"] if p in self._accumulated
            ]
            if matrix_updates:
                basis_update_due = self._basis_update_due(group)
                group["matrix_step"] += 1
                if basis_update_due:
                    group["basis_update_step"] += 1
            self._resolve_accumulated(matrix_updates)
            # Weights first, frame second. Everything in this step -- the
            # held-frame projection, the Oja tangent, and the lift back -- must
            # see the same frame `Q_t`. Moving the frame first meant the update
            # was projected down through `Q_t` and lifted back through
            # `Q_{t+1}`, so every step applied its own gradient rotated by
            # however far the tracker had just turned. History is unaffected:
            # the moment is read next step in the moved frame, where identity
            # transport is exact because the geodesic is a rigid rotation.
            self._integrate_step(matrix_updates, group)
            self._apply_basis_updates(matrix_updates, group)
            self._commit_moments(matrix_updates)

        # One sample per step, not per matrix: the quantity that means something
        # is the whole step's motion against the whole model's norm, so the
        # per-matrix squares are summed here and rooted once.
        diagnostics = self._diagnostics_sink()
        if diagnostics is not None and self._step_update_norm_sq is not None:
            diagnostics.add("update_norm", self._step_update_norm_sq.sqrt())
        self._step_update_norm_sq = None

        self._pending_matrix_updates.clear()
        self._accumulated.clear()
        return loss

    @torch.no_grad()
    def _fold_micro_batch(
        self, prepared: PreparedGrad, direction: Tensor, weight: float | Tensor, storage: torch.dtype
    ) -> None:
        """Add one orthogonalized micro-batch into this step's weighted sums."""

        direction = direction.float() * weight
        # `w_i^2 ||d_i||^2`, the weighted-sum form the agreement read needs.
        square = direction.square().sum()
        projected_norm = prepared.projected_grad.float().norm() * weight
        tangent = None if prepared.oja_tangent is None else prepared.oja_tangent.float() * weight
        raw_norm = None if prepared.raw_grad_norm is None else prepared.raw_grad_norm.float() * weight

        entry = self._accumulated.get(prepared.param)
        if entry is None:
            self._accumulated[prepared.param] = MatrixUpdate(
                param=prepared.param,
                projector=prepared.projector,
                original_shape=prepared.original_shape,
                direction=self._fold_into(None, direction, storage),
                weight=weight,
                projected_grad_norm=projected_norm,
                micro_batches=1,
                oja_tangent=(
                    None if tangent is None else self._fold_into(None, tangent, storage)
                ),
                raw_grad_norm=raw_norm,
                direction_square_sum=square,
                weight_square_sum=weight * weight,
            )
            return

        # The projector is rebuilt per prepare over the same basis tensor; the
        # latest one is the object `_apply_basis_updates` will write through.
        entry.projector = prepared.projector
        entry.direction = self._fold_into(entry.direction, direction, storage)
        entry.weight = entry.weight + weight
        entry.projected_grad_norm = entry.projected_grad_norm + projected_norm
        entry.micro_batches += 1
        if tangent is not None:
            entry.oja_tangent = self._fold_into(entry.oja_tangent, tangent, storage)
        if raw_norm is not None:
            entry.raw_grad_norm = raw_norm if entry.raw_grad_norm is None else entry.raw_grad_norm + raw_norm
        entry.direction_square_sum = entry.direction_square_sum + square
        entry.weight_square_sum = entry.weight_square_sum + weight * weight

    @staticmethod
    @torch.no_grad()
    def _fold_into(total: Tensor | None, increment: Tensor, dtype: torch.dtype) -> Tensor:
        """Add one fp32 increment into a running sum stored in `dtype`.

        The folded direction and the summed Oja tangent are the only `[d, r]`
        buffers that stay resident *across* a micro-batch group, so they are
        live alongside the next backward's activations. Stored at the gradient's
        dtype -- the rule the moment follows -- they halve that cost on a bf16
        model, which is what let a rank-256 `k=4` calibration fit a 12 GB card.
        A final fold asks for fp32 and promotes the sum, so the last increment
        is added exactly.

        A bf16 write is stochastically rounded: increment `i` of `k` is about
        `1/k` of the total, and round-to-nearest would drop it. Each sum keeps
        its expectation. What it does not keep is the square: the agreement read
        takes `||S||^2`, and unbiased rounding error `e` adds `E||e||^2` to it.
        With per-element error below `ulp / 2` and `ulp` at most `|x| / 128`
        that is under `2e-5 ||S||^2` per rounded fold -- about 0.5% of the
        `(k-1) a Q` signal at `k = 4`, `a = 0.003`.
        """

        if total is None:
            total = torch.zeros(increment.shape, dtype=dtype, device=increment.device)
        elif total.dtype != dtype:
            total = total.to(dtype)
        if wants_stochastic_rounding(total):
            copy_stochastic_(total, total.float().add_(increment))
        else:
            total.add_(increment.to(total.dtype))
        return total

    @torch.no_grad()
    def _resolve_accumulated(self, entries: list[MatrixUpdate]) -> None:
        """Turn the weighted sums into weighted means, once, before anything reads them.

        Also the last point at which the micro-batches can still be told apart,
        so the agreement read happens here.
        """

        diagnostics = self._diagnostics_sink()
        for entry in entries:
            self._record_micro_batch_agreement(diagnostics, entry)
            # Back to fp32 as the means leave the accumulator: bf16 is the
            # storage that had to be small, not the arithmetic downstream.
            entry.direction = entry.direction.float() / entry.weight
            entry.projected_grad_norm = entry.projected_grad_norm / entry.weight
            if entry.oja_tangent is not None:
                entry.oja_tangent = entry.oja_tangent.float() / entry.weight
            if entry.raw_grad_norm is not None:
                entry.raw_grad_norm = entry.raw_grad_norm / entry.weight

    @staticmethod
    def _record_micro_batch_agreement(
        diagnostics: DiagnosticsAccumulator | None, entry: MatrixUpdate
    ) -> None:
        """Mean pairwise cosine between the micro-batch directions this step averaged.

        Independent micro-batches share only the signal, so pure noise reads
        exactly zero -- an absolute level, not one needing a reference run. For
        raw gradients the expectation would be ``|s|^2 / (|s|^2 + |n|^2)``; the
        polar map is nonlinear, so here it is a monotone proxy for that ratio
        with the same zero, not the formula. Published and wired to nothing.

        Recovered from the sums, so the directions need not be kept. With
        `S = sum w_i d_i` and `Q = sum w_i^2 ||d_i||^2`, the numerator
        `||S||^2 - Q` is exactly the weighted sum of pairwise inner products.
        Dividing by `Q (W^2 - w2) / w2` turns it into a cosine by assuming the
        `||d_i||` are equal; the Newton-Schulz schedule leaves singular values
        near but not at one, so that normalization is approximate while the
        sign and the zero are exact. One norm of a tensor already in hand, no
        sync.
        """

        if diagnostics is None or entry.micro_batches < 2 or entry.direction_square_sum is None:
            return
        square_sum = entry.direction_square_sum
        pairs = entry.weight * entry.weight - entry.weight_square_sum
        denominator = square_sum * pairs / entry.weight_square_sum
        agreement = (entry.direction.float().square().sum() - square_sum) / denominator
        diagnostics.add("micro_batch_agreement", agreement)

    def _prepare_matrix_update(self, p: Tensor, grad: Tensor, group: dict) -> PreparedGrad:
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
        # There is no gradient clipping, because with the polar map ahead of the
        # average there is nothing left for it to protect. Every consumer of the
        # gradient is now insensitive to its magnitude: the aim divides a
        # quadratic by a quadratic, and the update orthogonalizes each step
        # before it enters the moment, so a blip contributes one unit-spectrum
        # direction weighted like any other step. A clip is exercised on every
        # step of a real run and changes the trajectory by float noise
        # (`test_the_update_is_invariant_to_gradient_scale`). Under the old order
        # it was load-bearing -- the moment absorbed a 180x gradient at full
        # size, and one step-54 blip collapsed alignment for the back half of a
        # 100-step run. Sanitizing non-finite gradients is a different job and
        # stays.
        diagnostics = self._diagnostics_sink()
        self._record_incoming_grad_diagnostics(diagnostics, grad)
        grad, raw_grad_norm = self._sanitize_grad_tensors(grad)
        if not projector.is_initialized:
            self._initialize_projector(projector, grad, state)
        projected_grad = projector.project(grad)

        return PreparedGrad(
            param=p,
            projector=projector,
            projected_grad=projected_grad,
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
    ) -> PreparedGrad:
        diagnostics = self._diagnostics_sink()
        self._record_incoming_grad_diagnostics(diagnostics, grad)
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
        )

        return PreparedGrad(
            param=p,
            projector=projector,
            projected_grad=projected_grad,
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
        entry: MatrixUpdate,
        moment: Tensor,
        beta: float,
    ) -> None:
        if diagnostics is None:
            return
        direction = entry.direction
        raw_grad_norm = entry.raw_grad_norm
        # Already the micro-batch mean: `_resolve_accumulated` divided both the
        # norm and the direction by the weight they were summed with, so every
        # read below describes the step that is about to be applied.
        projected_norm = entry.projected_grad_norm
        diagnostics.add("projected_grad_norm", projected_norm)
        if raw_grad_norm is not None:
            # The norm alone cannot answer "is the frame catching the gradient":
            # it falls when the basis is leaner even though every kept plane
            # catches as much as before. The ratio is the read that can, and it
            # needs the raw norm published beside it because the two move for
            # different reasons -- capture is the basis's business, the raw norm
            # is the model's.
            diagnostics.add("raw_grad_norm", raw_grad_norm)
            diagnostics.add("grad_capture", projected_norm / raw_grad_norm.clamp_min(1e-12))
        direction_norm = direction.norm()
        if beta <= 0.0:
            # `w` below is 1 and the read is 0/0: with no memory the moment is
            # this direction and there is no averaging to have a floor.
            return
        # Persistence with the noise floor taken out. An EMA of *independent*
        # constant-norm directions already has norm `w = sqrt((1-beta)/(1+beta))`
        # -- 0.229 at beta 0.9 -- purely from incomplete cancellation, so the raw
        # ratio never approaches zero and cannot be compared across beta. Writing
        # `d = mu + n`, `||M||^2 = ||mu||^2 + (1 - ||mu||^2) w^2`, so this inverts
        # to the mean's share of the direction's energy: 0 is a white stream, 1 a
        # direction held throughout, and negative is anti-correlated, which is
        # what the raw gradient stream reads before the polar map.
        #
        # Under accumulation the floor is approximate rather than exact: it
        # assumes a constant-norm stream, and the mean of `k` micro-batch
        # directions has a norm that falls with how much they disagree. Read
        # `micro_batch_agreement` beside it, which measures exactly that.
        #
        # It bounds the mean's *energy*, not its worth. A mean far too small to
        # move this read still decides where a thousand accumulated steps land,
        # because it is the only component the weight sum does not cancel.
        floor_sq = (1.0 - beta) / (1.0 + beta)
        ratio_sq = (moment.norm() / direction_norm.clamp_min(1e-12)).square()
        diagnostics.add("moment_persistence", (ratio_sq - floor_sq) / (1.0 - floor_sq))
        # Agreement is asked of the moment as it stood BEFORE this step's blend,
        # recovered exactly from the post-update moment. The post-update moment
        # already contains `1 - beta` of this very direction, which would make
        # the measurement partly self-referential -- it would report agreement
        # with itself. Reconstructing costs one axpy.
        #
        # Read as an overshoot meter: positive means the step under-travels and
        # the next gradient still points the way the last one did, negative means
        # it overshoots the valley and the gradient has flipped behind it. Zero is
        # critically damped, which makes this a learning-rate read and not merely
        # a description of the moment.
        moment_before = (moment - (1.0 - beta) * direction) / beta
        diagnostics.add(
            "grad_moment_cosine",
            (direction * moment_before).sum() / (direction_norm * moment_before.norm()).clamp_min(1e-12),
        )

    @staticmethod
    def _prepare_tracker_right_tensors(
        grad: Tensor,
        basis: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """One fused pass: clip, project in the held frame, aim.

        Both consumers -- the Oja tangent that steers the basis and the
        projected gradient that becomes the update -- read the same sanitized
        gradient, so the held-frame projection is computed once and shared. The
        moment is not built here: it integrates orthogonalized directions, and
        the polar map that produces them is batched across matrices.
        """

        grad, raw_grad_norm = UsuiTrack._sanitize_grad_tensors(grad)
        projected_grad = grad @ basis.mT
        work = grad.float()
        frame = basis.float().mT
        low = projected_grad.float()
        action = work.mT @ low
        rayleigh = frame.mT @ action
        rayleigh = 0.5 * (rayleigh + rayleigh.mT)
        tangent = action - frame @ rayleigh
        tangent = tangent / rayleigh.diagonal().mean().clamp_min(1e-12)
        return projected_grad, tangent, raw_grad_norm

    @staticmethod
    def _prepare_tracker_left_tensors(
        grad: Tensor,
        basis: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Left-side twin of `_prepare_tracker_right_tensors`."""

        grad, raw_grad_norm = UsuiTrack._sanitize_grad_tensors(grad)
        projected_grad = basis.mT @ grad
        work = grad.float()
        frame = basis.float()
        low = projected_grad.float()
        action = work @ low.mT
        rayleigh = frame.mT @ action
        rayleigh = 0.5 * (rayleigh + rayleigh.mT)
        tangent = action - frame @ rayleigh
        tangent = tangent / rayleigh.diagonal().mean().clamp_min(1e-12)
        return projected_grad, tangent, raw_grad_norm

    @staticmethod
    def _sanitize_grad_tensors(grad: Tensor) -> tuple[Tensor, Tensor]:
        """Replace non-finite entries and report the raw norm for telemetry.

        The norm is measured, never applied: nothing downstream scales by it.
        """

        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        return grad, grad.float().norm().detach()

    def _orthogonalized_directions(self, prepared: list[PreparedGrad]) -> list[Tensor]:
        """The polar map over one micro-batch's projected gradients.

        **The polar map runs before the average, not after** -- before the
        moment's average, and under accumulation before the micro-batches'
        average too. Every contribution enters with a flat spectrum, so one
        batch's dominant plane cannot own the memory by being large, the same
        reordering that fixed the aim where a directional burst used to own the
        turn. What the averages keep is magnitude as agreement: planes that
        persist stay long, planes that oscillate cancel toward zero, and the
        step shrinks where the direction is contested instead of being restored
        to full size by a second polar map. Under the old order that restoration
        was unconditional, which is why a moment that had cancelled to a
        fifteenth of the gradient still took a full-sized step.
        `REORTHOGONALIZE_MOMENT` puts it back as the control.
        """

        return self._bucketed_polar(
            [entry.projected_grad for entry in prepared],
            [entry.original_shape for entry in prepared],
        )

    def _bucketed_polar(self, tensors: list[Tensor], shapes: list[tuple[int, ...]]) -> list[Tensor]:
        """One Newton-Schulz call per group of entries sharing shape and scale.

        Keyed on the projected shape (torch.stack's actual requirement) and the
        scale, not the original parameter shape that scale is computed from.
        Several original shapes can share a scale -- `_muon_aspect_scale` clamps
        to 1.0 for every matrix tracked on its wider side, so a fleet of
        differently-shaped `w2`-like matrices collapses into one bucket. The
        scale each entry receives is exact, the same float it would have gotten
        alone; the orthogonalized direction is not bitwise identical to running
        the matrix solo, at the same ulp-level float tolerance every other
        batched-vs-solo comparison in this optimizer already carries. The scale
        is a per-matrix constant, so applying it here and averaging afterwards is
        the same magnitude convention as applying it last.
        """

        buckets: dict[tuple, list[int]] = {}
        for index, (tensor, shape) in enumerate(zip(tensors, shapes, strict=True)):
            key = (tuple(tensor.shape), UsuiTrack._muon_aspect_scale(shape))
            buckets.setdefault(key, []).append(index)

        mapped: list[Tensor | None] = [None] * len(tensors)
        for (_projected_shape, scale), indices in buckets.items():
            directions = self._orthogonalize_bucket([tensors[index] for index in indices], scale)
            for index, direction in zip(indices, directions, strict=True):
                mapped[index] = direction
        assert all(direction is not None for direction in mapped)
        return mapped  # type: ignore[return-value]

    def _integrate_step(self, entries: list[MatrixUpdate], group: dict) -> None:
        """Blend this step's averaged direction into the moment and move the weights."""

        if not entries:
            return

        beta = float(group["beta"])
        diagnostics = self._diagnostics_sink()
        for entry in entries:
            state = self.state[entry.param]
            state["step"] = state.get("step", 0) + 1
            stored = state.get("projected_exp_avg")
            if stored is None:
                # Allocated at the first step's end, not at its first fold: its
                # existence is what tells `_prepare_matrix_update` the frame is
                # past the step that fitted it, and micro-batches 2..k of that
                # step must still see the frame as just fitted, or they aim and
                # turn it where a single batch would not. The parameter's dtype
                # is the gradient's, so the moment keeps its storage dtype.
                stored = state["projected_exp_avg"] = torch.zeros(
                    entry.direction.shape, dtype=entry.param.dtype, device=entry.direction.device
                )
            # fp32 from here through the frame rotation to a single rounded
            # commit in `_commit_moments`. `copy=True` because `.float()` on
            # an already-fp32 moment aliases the stored tensor, and this must
            # not write through to state before the rotation has been applied.
            moment = stored.to(dtype=torch.float32, copy=True)
            moment.mul_(beta).add_(entry.direction, alpha=1.0 - beta)
            entry.projected_exp_avg = moment
            self._record_projection_diagnostics(diagnostics, entry, moment, beta)

        moments = [entry.projected_exp_avg for entry in entries]
        update_hats = (
            self._bucketed_polar(moments, [entry.original_shape for entry in entries])
            if REORTHOGONALIZE_MOMENT
            else moments
        )
        for entry, update_hat in zip(entries, update_hats, strict=True):
            self._apply_matrix_update(entry, update_hat, group)

    def _orthogonalize_bucket(self, tensors: list[Tensor], scale: float) -> list[Tensor]:
        if len(tensors) == 1:
            return [self._orthogonalize_update_runtime(tensors[0], scale)]
        stacked = self._orthogonalize_update_runtime(torch.stack(tensors), scale)
        return list(stacked.unbind(0))

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
            self._record_tangent_spectrum_diagnostics(diagnostics, eigenvalues)
            geometry_buckets: dict[tuple, list[int]] = {}
            for index, entry in enumerate(bucket_entries):
                tangent = entry.oja_tangent
                assert tangent is not None
                geometry_buckets.setdefault((tuple(tangent.shape), entry.projector._basis_side()), []).append(index)
            for (_shape, side), indices in geometry_buckets.items():
                selected_entries = [bucket_entries[index] for index in indices]
                frames = torch.stack([entry.projector.canonical_basis() for entry in selected_entries])
                selected_tangents = torch.stack([entry.oja_tangent for entry in selected_entries if entry.oja_tangent is not None])
                selected_vectors = eigenvectors[indices]
                selected_tangents, selected_values = self._polar_tangent(
                    selected_entries,
                    selected_tangents,
                    eigenvalues[indices],
                    selected_vectors,
                )
                new_frames = SubspaceProjector.oja_geodesic_from_eigh(
                    frames,
                    selected_tangents,
                    selected_values,
                    selected_vectors,
                    step_size,
                )
                self._record_followed_step(selected_entries, frames, new_frames)
                for entry, new_frame in zip(selected_entries, new_frames, strict=True):
                    basis = new_frame.mT if side is ProjectionSide.RIGHT else new_frame
                    # Round-to-nearest, deliberately, and unlike the moment's
                    # accumulate this one is measured. Stochastic
                    # rounding exists to stop a sub-ulp *update* vanishing, and
                    # the frame has no sub-ulp update to lose -- the geodesic
                    # moves it by a real angle every time. What bf16 storage
                    # costs the frame is `transport_spin`, in-span rotation that
                    # scrambles the projected moment's coordinates for no
                    # subspace progress, and that is variance rather than bias:
                    # measured, stochastic rounding here makes it `sqrt(2)` times
                    # worse at every window, which is exactly the extra variance
                    # of a full-ulp uniform draw over a half-ulp bound.
                    entry.projector.basis = basis.to(
                        device=entry.projector.basis.device,
                        dtype=entry.projector.basis.dtype,
                    ).contiguous()
                    entry.projector.resolved_side = side
                    state = self.state[entry.param]
                    state["basis"] = entry.projector.basis
                    state["projection_side_is_right"] = side is ProjectionSide.RIGHT
                self._record_frame_rotation(selected_entries, frames)
        self._record_basis_lag(pending, group)

    @torch.no_grad()
    def _record_frame_rotation(self, entries: list[MatrixUpdate], old_frames: Tensor) -> None:
        """The in-span rotation identity transport does not account for.

        Transport is the identity in moving-frame coordinates, which is exact
        for the geodesic itself: a single Grassmann step along a horizontal
        tangent gives `Q^T Q+ = V cos(theta) V^T`, symmetric, so the moment's
        coordinates carry over untouched. What is not exact is the frame that
        actually gets stored. Rounding the geodesic's result into bf16 rotates
        the frame's columns *inside* the span they already had -- motion that
        advances the subspace not at all and scrambles the moment one-for-one.
        `transport_spin` has always measured it; nothing corrected it.

        So read the overlap from the frames as they are **stored**, not as they
        were computed -- the rounding is the whole quantity being chased -- and
        take its orthogonal polar factor. The symmetric part is the geodesic and
        is already handled by identity transport; the rotation is the residue.
        An ideal step gives the identity here, so this is a no-op exactly when
        transport was already right.
        """

        new_frames = torch.stack([entry.projector.canonical_basis() for entry in entries])
        overlap = old_frames.mT @ new_frames
        rotation = SubspaceProjector.polar_factor(overlap)
        for entry, matrix in zip(entries, rotation.unbind(0), strict=True):
            entry.frame_rotation = matrix

    @torch.no_grad()
    def _commit_moments(self, entries: list[MatrixUpdate]) -> None:
        """Rotate into the new frame, then round once.

        Order is the whole correctness argument. This step's projected gradient
        was measured in `Q_t` and has already been blended into a moment that is
        also in `Q_t`, and the weight update lifted through `Q_t` as well. Only
        now, with nothing left to read in the old coordinates, does the moment
        move to `Q_{t+1}`. Rotating before the blend would add a `Q_t` gradient
        onto a `Q_{t+1}` moment, which is the coordinate mismatch the old
        project-up-project-down transport used to make on every step.

        The moment has been fp32 since it was accumulated, so this is the
        step's only rounding of it, and stochastic rounding makes that write
        unbiased -- which matters more here than for the parameter, because a
        rotation is a rearrangement whose systematic rounding would accumulate
        against the very coordinates it exists to preserve.
        """

        for entry in entries:
            moment = entry.projected_exp_avg
            rotation = entry.frame_rotation
            if rotation is not None:
                is_right = entry.projector._basis_side() is ProjectionSide.RIGHT
                moment = moment @ rotation if is_right else rotation.mT @ moment
            stored = self.state[entry.param]["projected_exp_avg"]
            if self.stochastic_rounding and stored.dtype in (torch.bfloat16, torch.float16):
                copy_stochastic_(stored, moment)
            else:
                stored.copy_(moment)

    @torch.no_grad()
    def _polar_tangent(
        self,
        entries: list[MatrixUpdate],
        tangents: Tensor,
        eigenvalues: Tensor,
        eigenvectors: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Turn every live plane by the same angle, `eta`, with no gain on top.

        **Magnitude decides nothing.** The tangent is replaced by its polar
        factor, which is the same distrust of magnitude the weight update already
        applies: Newton-Schulz throws away the projected moment's singular values
        because direction is what survives a noisy batch and magnitude is not,
        and the tangent's singular values deserve no more credit. They would only
        decide how the frame's motion is divided between planes, and the leading
        plane -- which owns most of the displacement -- is measurably the least
        persistent thing in the aim. `polar(Delta) = Delta V diag(1/sigma) V^T`
        falls straight out of the eigendecomposition the geodesic already needs,
        so this costs two `[r, r]` matmuls and no Newton-Schulz.

        **Dead planes are held still**, not merely handed a zero tangent column.
        A plane is live only if it clears the Gram's own numerical noise floor:
        a symmetric `[r, r]` decomposition carries backward error of order
        `r * eps * lambda_max`, which is `sqrt(r * eps) * sigma_max` in these
        units -- `2.8e-3` at rank 64 in fp32. The threshold here used to be
        `1e-6 * sigma_max`, six orders below what fp32 can resolve, so no plane
        was ever dead. Measured, that mattered and not equally: LFM at rank 128
        resolves 79% of its planes, Anima at rank 64 resolves 45%. Under the old
        threshold the rest were driven anyway -- `1 / sigma` promotes a rounding
        artifact to a unit-norm direction and the polar step turns it by the same
        angle as the plane carrying the signal -- and on Anima those noise
        directions fed back into the next tangent Gram until `eigh` failed within
        twenty steps. The zero must reach the *angle*, not just the tangent: the
        geodesic reads `cos(eta * sigma)` on the frame's own component along each
        eigenvector, so a dead plane handed a live angle contracts that component
        against a tangent term that is zero, which is not a rotation.

        **There is no gain, and the reason is that nothing measured one.** The
        turn used to be scaled by how well the aim repeated -- the mean squared
        cosine of the principal angles between consecutive top-`k` aims, divided
        by an attainable ceiling. The shape of that idea is still right: a
        converged frame orbits on batch noise and should stop turning. What has
        never existed is a sensor for it. The two-lag form that would derive the
        anchor instead of remembering it is falsified in `ARCHIVE.md` -- the
        aim's persistence *shape* is stationary while its level is not, so the
        ratio holds at ~0.81 and anneals nothing. The ceiling that replaced it is
        `effective_rank / k`, a magnitude functional gating a step this method
        exists to make magnitude-free, and on Anima it annealed the turn by 1.21x
        over 2100 steps -- close to inert. A controller that cannot tell arrival
        from batch noise is not a controller. Constant turn until something
        measures the thing the gain was for.

        Its price is named: nothing here slows down near the answer. `eta` is the
        only remaining term, and scheduling it on the prior that aiming gets
        harder as the loss surface smooths is a knob we have, not a sensor.
        """

        sigma = eigenvalues.clamp_min(0.0).sqrt()
        noise_floor = math.sqrt(sigma.shape[-1] * torch.finfo(sigma.dtype).eps)
        live = sigma > noise_floor * sigma.amax(dim=-1, keepdim=True).clamp_min(1e-30)
        inverse = torch.where(live, 1.0 / sigma.clamp_min(1e-30), torch.zeros_like(sigma))

        # `Delta V diag(1/sigma) V^T`, the polar factor: unit-length plane
        # directions carried back into the eigenbasis the geodesic reads.
        directions = (tangents @ eigenvectors) * inverse.unsqueeze(-2).to(tangents.dtype)
        polar = directions @ eigenvectors.mT

        if self.rank_calibrator is not None:
            group = self._matrix_param_groups.get(entries[0].param)
            label = group.get("calibration_label") if group is not None else None
            if label is not None:
                self.rank_calibrator.observe(label, sigma.shape[-1], [entry.param for entry in entries], live.sum(dim=-1))

        diagnostics = self._diagnostics_sink()
        if diagnostics is not None:
            diagnostics.add("tangent_live_fraction", live.sum() / sigma.shape[-1], count=live.shape[0])

        # The geodesic reads its per-plane angles as `sqrt(values)` and the polar
        # factor's own singular values are all one, so unit values are what make
        # every live plane turn by exactly `eta`.
        values = live.to(eigenvalues.dtype)
        return polar, values

    def _lag_sample(self) -> set[Tensor]:
        """The matrices the snapshot instruments follow, chosen once.

        Deterministic, so two runs of the same config measure the same tensors
        and their curves are comparable.
        """

        if self._lag_sampled is None:
            matrices = [param for group in self.param_groups for param in group["params"]]
            self._lag_sampled = set(matrices[: max(0, self.diagnostics_lag_matrices)])
        return self._lag_sampled

    @torch.no_grad()
    def _record_followed_step(self, entries: list[MatrixUpdate], before: Tensor, after: Tensor) -> None:
        """Distance each sampled frame actually moved, measured from the frames.

        Measured from the frames, which is the only reason this method exists.
        `transport_speed` used to be read from the tangent's eigenvalues *before*
        the geodesic ran -- what the aim proposed rather than what the frame did.
        Those agree only while nothing stands between them. An orthogonalized
        tangent turns every live plane by `eta` regardless of what `sigma` said,
        and measured on a real run the proposal read 2.2x the motion followed, so
        every ortho arm's speed was reporting a turn that never happened.

        `transport_curve` is a ratio, so its numerator and denominator have to
        be the same kind of measurement. This is the denominator: the same
        chordal distance `transport_lag` reports, over one update instead of a
        window, taken from the frame that was actually written.
        """

        sample = self._lag_sample()
        indices = [index for index, entry in enumerate(entries) if entry.param in sample]
        if not indices:
            return
        start = before[indices].float()
        end = after[indices].float()
        residual = end - start @ (start.mT @ end)
        followed = residual.flatten(1).norm(dim=-1) / math.sqrt(end.shape[-1])
        for entry, distance in zip((entries[index] for index in indices), followed.unbind(0), strict=True):
            entry.transport_speed = distance
        # Published here rather than from the eigenvalues, because the frame is
        # the thing the name promises. The two agree in the release and come
        # apart under any experiment that transforms the geodesic -- measured, an
        # orthogonalized tangent turns every plane by `eta` while the raw
        # spectrum proposed 2.2x that, so the eigenvalue read was reporting an
        # aim the frame never followed.
        diagnostics = self._diagnostics
        if diagnostics is not None:
            diagnostics.add("transport_speed", followed.sum(), count=len(indices))

    def _record_basis_lag(self, entries: list[MatrixUpdate], group: dict) -> None:
        """Where each sampled frame ended up against where it has been.

        `transport_speed` says how fast the frame moves and cannot say whether
        that motion goes anywhere: it is floored by single-batch noise in the
        aim, so a frame orbiting a fixed point at constant radius reports the
        same speed forever as one genuinely travelling. Comparing the frame to
        *itself* some refreshes back cancels that -- an orbit returns, a drift
        does not. Three reads come out of the one comparison, and they are only
        meaningful together.

        `transport_lag` is the net distance covered over the window, in the same
        per-plane RMS-sine unit as the speed. Exact and needing no
        decomposition: with `C = Q_old^T Q_now`, the residual `R = Q_now -
        Q_old C` has `||R||_F^2 = sum_i sin^2(theta_i)`. Subtracting the vectors
        rather than the squared cosines is what keeps it well conditioned at the
        small angles that matter here.

        `transport_curve` is `1 - lag / path`, the fraction of the window's
        travel that cancelled. Zero is a straight drift; approaching one is a
        frame churning in place. This is what distinguishes a tracker that is
        settling from one that is merely slow -- speed alone cannot, since
        halving the step size halves both a productive drift and a useless
        orbit.

        `transport_spin` is rotation of the frame's columns *inside* the span
        they already had, which moves the subspace not at all and scrambles the
        moment one-for-one, since transport is the identity in these
        coordinates. It is the skew part of `C`: a single Grassmann geodesic
        along a horizontal tangent gives `C = V cos(theta) V^T`, exactly
        symmetric, so a nonzero skew is either genuine holonomy -- the product
        of several such symmetric steps need not be symmetric -- or retraction
        and rounding error. Both break identity transport, so this is the
        measurement that says whether the moment's coordinates still mean what
        they meant, rather than an argument that they should.

        Sampled over `diagnostics_lag_matrices` parameters, not carried per
        parameter: this is the only diagnostic that costs persistent bytes
        (`[d,r]` per sampled matrix), and the VRAM-first rule bans that in the
        update, not in an opt-in instrument over a handful of tensors. The
        snapshots live outside `self.state` so they never enter a checkpoint.
        """

        diagnostics = self._diagnostics_sink()
        if diagnostics is None or self._diagnostics_tier != "full":
            return
        # The path has to accumulate on every basis update while the snapshot is
        # only taken on the window boundary, so this runs unconditionally and the
        # window check happens per entry below.
        due = group["basis_update_step"] % max(1, self.diagnostics_lag_interval) == 0
        sample = self._lag_sample()
        for entry in entries:
            if entry.param not in sample:
                continue
            if entry.transport_speed is not None:
                walked = self._lag_path.get(entry.param)
                self._lag_path[entry.param] = entry.transport_speed if walked is None else walked + entry.transport_speed
            if not due:
                continue
            frame = entry.projector.canonical_basis().float()
            previous = self._lag_snapshots.get(entry.param)
            self._lag_snapshots[entry.param] = frame.clone()
            path = self._lag_path.pop(entry.param, None)
            if previous is None or previous.shape != frame.shape:
                continue
            root = math.sqrt(frame.shape[1])
            overlap = previous.mT @ frame
            lag = (frame - previous @ overlap).norm() / root
            diagnostics.add("transport_lag", lag)
            diagnostics.add("transport_spin", (overlap - overlap.mT).norm() / (2.0 * root))
            if path is not None:
                diagnostics.add("transport_curve", 1.0 - lag / path.clamp_min(1e-12))

    @staticmethod
    def _record_tangent_spectrum_diagnostics(
        diagnostics: DiagnosticsAccumulator | None,
        eigenvalues: Tensor,
    ) -> None:
        """The shape of the aim's spectrum: how concentrated, how many planes.

        `tangent_concentration` is the leading eigenvalue's share of the trace and
        `tangent_effective_planes` the spectrum's effective rank. Together they
        separate a confident drift from a frame spinning on its noise tail -- the
        same displacement can be either.

        The plane count is published raw, not as a fraction of `r`. Divided by
        the rank it is not comparable across rank settings: a leaner table raises
        the fraction while the spectrum is unchanged, and one run read that as a
        third more planes carrying the aim when the count had moved by six
        percent.

        This used to publish `transport_speed` from these eigenvalues too, as the
        displacement the geodesic *would* produce. That reading is gone: speed is
        measured from the frames in `_record_followed_step`, because the two agree
        only while nothing transforms the geodesic between the two points.
        """

        if diagnostics is None:
            return
        spectrum = eigenvalues.clamp_min(0.0)
        energy = spectrum.sum(dim=-1).clamp_min(1e-12)
        concentration = spectrum.amax(dim=-1) / energy
        planes = energy.square() / spectrum.square().sum(dim=-1).clamp_min(1e-12)
        samples = int(spectrum.shape[0])
        diagnostics.add("tangent_concentration", concentration.sum(), count=samples)
        diagnostics.add("tangent_effective_planes", planes.sum(), count=samples)

    @staticmethod
    def _basis_update_step_size(group: dict) -> float:
        return GEODESIC_STEPSIZE

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
    def _muon_aspect_scale(original_shape: tuple[int, ...]) -> float:
        """Muon's aspect factor, read from the parameter the update will land on.

        In Muon, which orthogonalizes the full `[m, n]` update, this makes
        `||U||_F == sqrt(m)` whichever way the matrix is stored. Here it
        multiplies the projected moment's polar factor instead, `[d, r]` rather
        than `[m, n]`, so the guarantee becomes `||U||_F == sqrt(m) *
        sqrt(r/min(m,n))`. Measured against a unit scale and a rank-compensating
        one at matched effective step, this rule still wins both eval heads --
        see ARCHIVE.md, "the Muon aspect factor is not double duty".
        """

        rows = original_shape[0]
        cols = math.prod(original_shape[1:])
        return math.sqrt(max(1.0, rows / cols))

    @staticmethod
    def _orthogonalize_update(update: Tensor, scale: float) -> Tensor:
        """Direction from the balanced polar map, magnitude from the aspect scale.

        The only implementation: `torch.compile` wraps this function, so the
        compiled and eager paths cannot disagree about what an update is. The
        scale arrives as a number because it is a pure function of the parameter
        shape, which keeps integer arithmetic out of the graph and leaves this
        with one tensor operation and one multiply.
        """

        return UsuiTrack._balanced_polar_direction(update) * scale

    def _orthogonalize_update_runtime(self, update: Tensor, scale: float) -> Tensor:
        kernel = self._compiled_orthogonalize_update or UsuiTrack._orthogonalize_update
        return kernel(update, scale)

    @staticmethod
    def _balanced_polar_direction(
        update: Tensor,
        eps: float = 1e-7,
    ) -> Tensor:
        """Balance the rows, then orthogonalize. Two operations, two lineages.

        The balancing is Aurora's
        (https://github.com/tilde-research/aurora-release, Tilde Research):
        diagonally precondition a rectangular matrix so its large-side row
        leverage approaches the Stiefel target, orienting wide input to tall form
        first and transposing back after, per Aurora's convention. It changes
        how force is distributed across rows and orthogonalizes nothing.

        The orthogonalization is the Newton-Schulz polar map of the Muon lineage
        (`_newton_schulz_polar`). It is what discards the moment's singular
        values, and it is the step the surrounding code means when it says the
        update's magnitude comes from `lr` rather than from the gradient.

        Both are reimplemented from the methods; neither is a runtime dependency.
        Square input needs no balancing and goes straight to the polar map.
        UsuiTrack owns momentum, LR, weight decay, and the aspect scale.
        """

        if update.ndim < 2:
            raise ValueError(f"the polar direction map expects at least 2D input, got shape {tuple(update.shape)}")
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
    def _newton_schulz_polar(update: Tensor, eps: float = 1e-7) -> Tensor:
        if update.ndim < 2:
            raise ValueError(f"Newton-Schulz orthogonalization expects at least 2D input, got shape {tuple(update.shape)}")
        # fp32, where both references this is ported from run bf16. Measured:
        # the iteration is 1.5-2x faster in bf16 through this call and worth
        # nothing end to end, because the polar map is not a meaningful share of
        # a training step. See PLAN.md P12.
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
