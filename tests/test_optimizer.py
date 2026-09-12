import copy
import math

import pytest
import torch
from torch import Tensor

from usuitrack import SubspaceProjector, UsuiTrack


# A representative transformer-ish weight. Small toy shapes like (8, 4) put
# effective_rank() straight into its floor -- a quarter of 4 rounds to 1 -- so
# every test silently exercised a rank-1 basis instead of the regime the
# optimizer is designed for. RANK sits below the quarter cap of COLS so the
# configured rank is the rank actually used.
ROWS, COLS, RANK = 256, 128, 16


def test_rejects_non_matrix_and_excessive_rank():
    with pytest.raises(ValueError, match="only supports 2D"):
        UsuiTrack([torch.nn.Parameter(torch.randn(4))])
    # Over-cap rank is not fatal: effective_rank() caps it at half the smaller
    # side and says so once at startup.
    with pytest.warns(UserWarning, match="exceeds half the smaller side"):
        UsuiTrack([torch.nn.Parameter(torch.randn(ROWS, COLS))], rank=COLS)


def test_basis_moves_every_step_by_default():
    weight = torch.nn.Parameter(torch.randn(ROWS, COLS))
    optimizer = UsuiTrack([weight], lr=0.01, rank=RANK, side="right")

    weight.grad = torch.randn_like(weight)
    optimizer.step()
    initial_basis = optimizer.state[weight]["basis"].clone()

    weight.grad = torch.randn_like(weight)
    optimizer.step()
    assert not torch.equal(initial_basis, optimizer.state[weight]["basis"])
    assert optimizer.param_groups[0]["basis_update_step"] == 2


def test_basis_update_interval_gates_geodesic_not_moment():
    weight = torch.nn.Parameter(torch.randn(ROWS, COLS))
    optimizer = UsuiTrack([weight], lr=0.01, rank=RANK, side="right", basis_update_interval=2)

    weight.grad = torch.randn_like(weight)
    optimizer.step()
    initial_basis = optimizer.state[weight]["basis"].clone()
    assert optimizer.param_groups[0]["basis_update_step"] == 0

    weight.grad = torch.randn_like(weight)
    optimizer.step()
    assert not torch.equal(initial_basis, optimizer.state[weight]["basis"])
    assert optimizer.param_groups[0]["basis_update_step"] == 1
    assert "projected_exp_avg" in optimizer.state[weight]


def test_prepare_release_matches_ordinary_step_exactly():
    """Validates the release_matrix_grads=True fast path (README's early-release
    mode) produces bit-identical parameters to the ordinary step() path."""

    torch.manual_seed(0)
    ordinary_weight = torch.nn.Parameter(torch.randn(ROWS, COLS))
    released_weight = torch.nn.Parameter(ordinary_weight.detach().clone())
    ordinary = UsuiTrack([ordinary_weight], lr=0.01, rank=RANK, side="right")
    released = UsuiTrack([released_weight], lr=0.01, rank=RANK, side="right", release_matrix_grads=True)

    for _ in range(3):
        gradient = torch.randn_like(ordinary_weight)
        ordinary_weight.grad = gradient.clone()
        released_weight.grad = gradient.clone()
        released.prepare(released_weight)
        assert released_weight.grad is None
        ordinary.step()
        released.step()
        torch.testing.assert_close(ordinary_weight, released_weight, rtol=0, atol=0)


def test_plain_gradient_accumulation_without_release_matches_manual_grad_sum():
    """The summed-gradient path, which is not what accumulate() does.

    Two backward() calls before one step() with no accumulate() between them
    leave a summed gradient in param.grad, and step() reads it as one gradient
    -- identical to a single step() on the pre-summed tensor. That is a larger
    gradient for one step, not a larger batch: the polar map discards the
    magnitude, so the two micro-batches never get to disagree. accumulate() is
    the path that averages them; this one is what happens without it."""

    torch.manual_seed(2)
    accumulated = torch.nn.Linear(COLS, ROWS, bias=False)
    presummed = torch.nn.Linear(COLS, ROWS, bias=False)
    presummed.weight.data.copy_(accumulated.weight.data)

    x1, x2 = torch.randn(8, COLS), torch.randn(8, COLS)
    accumulated(x1).sum().backward()
    accumulated(x2).sum().backward()

    grad1 = torch.autograd.grad(presummed(x1).sum(), presummed.weight)[0]
    grad2 = torch.autograd.grad(presummed(x2).sum(), presummed.weight)[0]
    presummed.weight.grad = grad1 + grad2

    torch.testing.assert_close(accumulated.weight.grad, presummed.weight.grad, rtol=0, atol=0)

    UsuiTrack([accumulated.weight], lr=0.01, rank=RANK).step()
    UsuiTrack([presummed.weight], lr=0.01, rank=RANK).step()
    torch.testing.assert_close(accumulated.weight, presummed.weight, rtol=0, atol=0)


def test_prepare_is_exactly_once():
    weight = torch.nn.Parameter(torch.randn(ROWS, COLS))
    optimizer = UsuiTrack([weight], rank=RANK)
    weight.grad = torch.randn_like(weight)
    optimizer.prepare(weight)
    with pytest.raises(RuntimeError, match="already prepared"):
        optimizer.prepare(weight)
    optimizer.step()


def test_zero_grad_drops_pending_prepared_work():
    """zero_grad() is an escape hatch for callers that must bail out (e.g. an
    OOM-retry loop) after prepare() already mutated moving-average state for
    some params. It can't undo that mutation, but it must not block recovery:
    it just drops the pending update so the next prepare()/step() starts
    clean."""

    weight = torch.nn.Parameter(torch.randn(ROWS, COLS))
    optimizer = UsuiTrack([weight], rank=RANK)
    weight.grad = torch.randn_like(weight)
    optimizer.prepare(weight)
    with pytest.warns(UserWarning, match="discarded prepared updates"):
        optimizer.zero_grad()
    assert not optimizer._pending_matrix_updates
    assert weight.grad is None


def test_state_dict_resume_matches_uninterrupted_run():
    torch.manual_seed(1)
    first = torch.nn.Parameter(torch.randn(ROWS, COLS))
    first_optimizer = UsuiTrack([first], lr=0.01, rank=RANK)
    for _ in range(2):
        first.grad = torch.randn_like(first)
        first_optimizer.step()

    second = torch.nn.Parameter(first.detach().clone())
    second_optimizer = UsuiTrack([second], lr=0.01, rank=RANK)
    second_optimizer.load_state_dict(copy.deepcopy(first_optimizer.state_dict()))

    gradient = torch.randn_like(first)
    first.grad = gradient.clone()
    second.grad = gradient.clone()
    first_optimizer.step()
    second_optimizer.step()
    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("side", ["right", "left"])
def test_live_kernel_tangent_is_horizontal(side):
    """Oja's tangent must have no component along the current frame -- that is
    what makes the geodesic turn strictly in the frame's complement and keeps
    the retraction exact. Checked against the fused kernel because after the
    first step that is the only code that builds a tangent."""
    from usuitrack import SubspaceProjector

    torch.manual_seed(0)
    gradient = torch.randn(ROWS, COLS)
    projector = SubspaceProjector(rank=RANK, side=side)
    projector.fit(gradient)
    basis = projector.basis
    assert basis is not None

    kernel = (
        UsuiTrack._prepare_tracker_right_tensors
        if side == "right"
        else UsuiTrack._prepare_tracker_left_tensors
    )
    projected_shape = (ROWS, RANK) if side == "right" else (RANK, COLS)
    # A gradient the frame was *not* fitted on: a frame fitted on G is the
    # leading eigenspace of `G^T G`, so the Oja action lands entirely inside it
    # and the horizontal residual is exactly zero, which makes this assertion
    # vacuous.
    _projected, tangent, _norm = kernel(
        torch.randn(ROWS, COLS),
        basis,
    )

    frame = projector.canonical_basis()
    overlap = frame.mT @ tangent
    assert float(overlap.abs().max()) < 1e-3 * float(tangent.abs().max()), overlap.abs().max()


def _two_matrix_optimizer(**kwargs):
    params = [
        torch.nn.Parameter(torch.randn(ROWS, COLS)),
        torch.nn.Parameter(torch.randn(ROWS, COLS)),
    ]
    return params, UsuiTrack(params, lr=1e-3, rank=RANK, **kwargs)


def _run(params, optimizer, steps):
    for _ in range(steps):
        for param in params:
            param.grad = torch.randn_like(param)
        optimizer.step()


def test_a_fitted_frame_is_a_fixed_point_of_its_own_gradient():
    """The frame has nothing left to learn from the gradient that fitted it.

    The Oja action is `(G^T G) B^T`, and a frame fitted on `G` *is* the leading
    eigenspace of `G^T G`, so the action lands entirely inside the frame and the
    horizontal residual is zero. The tracker therefore stops turning when the
    gradient's principal subspace stops moving, rather than churning on a
    stationary target.
    """
    torch.manual_seed(0)
    gradient = torch.randn(ROWS, COLS)
    projector = SubspaceProjector(rank=RANK, side="right")
    projector.fit(gradient)
    basis = projector.basis
    assert basis is not None

    _projected, tangent, _norm = UsuiTrack._prepare_tracker_right_tensors(gradient, basis)
    assert float(tangent.abs().max()) < 1e-5, float(tangent.abs().max())


def test_kernel_matches_a_hand_rolled_step():
    """The fused kernel is the written-out projection, not a path that merely runs."""
    torch.manual_seed(0)
    gradient = torch.randn(ROWS, COLS)
    projector = SubspaceProjector(rank=RANK, side="right")
    projector.fit(torch.randn(ROWS, COLS))
    basis = projector.basis
    assert basis is not None

    projected, _tangent, _norm = UsuiTrack._prepare_tracker_right_tensors(gradient, basis)

    assert torch.allclose(projected, gradient @ basis.mT, atol=1e-6)


def test_the_moment_integrates_orthogonalized_directions():
    """The polar map runs before the average, and the average is the update.

    Written out at the integration site, because that is where the order lives:
    one step from a zero moment must leave `(1 - beta)` of the *orthogonalized*
    projected gradient in state, not `(1 - beta)` of the gradient itself, and the
    weight must move by that moment rather than by a re-orthogonalized copy of
    it.
    """
    torch.manual_seed(0)
    weight = torch.nn.Parameter(torch.randn(ROWS, COLS))
    before = weight.detach().clone()
    optimizer = UsuiTrack([weight], lr=0.1, rank=RANK, side="right", beta=0.9, weight_decay=0.0)

    gradient = torch.randn(ROWS, COLS)
    weight.grad = gradient.clone()
    optimizer.step()

    state = optimizer.state[weight]
    basis = state["basis"]
    direction = UsuiTrack._orthogonalize_update(gradient @ basis.mT, UsuiTrack._muon_aspect_scale((ROWS, COLS)))

    torch.testing.assert_close(state["projected_exp_avg"].float(), 0.1 * direction, atol=1e-3, rtol=0)
    # The frame does not move on the step that fits it, so the update lifts
    # through the same basis the moment lives in.
    expected_step = (0.1 * direction) @ basis
    torch.testing.assert_close((before - weight.detach()) / 0.1, expected_step, atol=1e-3, rtol=0)


def test_a_contested_direction_takes_a_smaller_step_than_an_agreed_one():
    """Disagreement costs step size, which is the whole point of ortho-first.

    Both arms hand the moment a unit-spectrum direction of identical size every
    step; only the sign pattern differs. `G` and `-G` share `G^T G`, so a frame
    fitted on either is a fixed point of both streams and the frame contributes
    nothing to the difference. Under the old order -- average, then
    orthogonalize -- the two arms would take steps of exactly the same size,
    because the polar map restored whatever magnitude the cancellation removed.
    """

    def final_step_norm(signs: list[int]) -> float:
        torch.manual_seed(0)
        weight = torch.nn.Parameter(torch.randn(ROWS, COLS))
        optimizer = UsuiTrack([weight], lr=0.01, rank=RANK, side="right", beta=0.9)
        gradient = torch.randn(ROWS, COLS)
        for sign in signs:
            before = weight.detach().clone()
            weight.grad = gradient * sign
            optimizer.step()
            motion = float((weight.detach() - before).norm())
        return motion

    agreed = final_step_norm([1] * 8)
    contested = final_step_norm([1, -1] * 4)
    assert contested < 0.5 * agreed, (contested, agreed)


def test_moment_persistence_reads_zero_on_an_independent_stream():
    """The metric must subtract its own floor, or it can never reach zero.

    An EMA of independent constant-norm directions has norm
    `sqrt((1-beta)/(1+beta))` from incomplete cancellation alone -- 0.229 at
    beta 0.9 -- so the raw ratio reports "agreement" for a stream that has none.
    Fed white noise the corrected read sits at zero; fed one repeated direction
    it sits at one.
    """
    def persistence(repeat: bool) -> float:
        torch.manual_seed(0)
        weight = torch.nn.Parameter(torch.randn(ROWS, COLS))
        optimizer = UsuiTrack([weight], lr=1e-4, rank=RANK, side="right", beta=0.9)
        optimizer.diagnostics = "full"
        fixed = torch.randn(ROWS, COLS)
        for _ in range(200):
            weight.grad = fixed.clone() if repeat else torch.randn(ROWS, COLS)
            optimizer.step()
            if _ == 149:
                optimizer.pop_diagnostics()  # drop the acquisition transient
        return optimizer.pop_diagnostics()["moment_persistence"]

    assert abs(persistence(repeat=False)) < 0.05, persistence(repeat=False)
    assert persistence(repeat=True) > 0.9, persistence(repeat=True)


def test_the_update_is_invariant_to_gradient_scale():
    """Why there is no gradient clip: nothing downstream can see the magnitude.

    The polar map runs on each step's projected gradient before it enters the
    moment, so a per-step rescale -- which is exactly what a clip is -- cannot
    reach the average. The aim is scale-free for its own reason: `action` and
    `rayleigh` are both quadratic in `G`, so their ratio cancels the factor.
    A 180x blip is the case the clip existed for, and it is included here.
    """

    def trajectory(scales: list[float]) -> tuple[Tensor, Tensor]:
        torch.manual_seed(0)
        weight = torch.nn.Parameter(torch.randn(ROWS, COLS))
        optimizer = UsuiTrack([weight], lr=0.01, rank=RANK, side="right", beta=0.9)
        generator = torch.Generator().manual_seed(7)
        for scale in scales:
            weight.grad = torch.randn(ROWS, COLS, generator=generator) * scale
            optimizer.step()
        return weight.detach().clone(), optimizer.state[weight]["projected_exp_avg"].float()

    blip = [180.0 if step == 5 else 1.0 for step in range(12)]
    scaled, scaled_moment = trajectory(blip)
    unit, unit_moment = trajectory([1.0] * 12)

    torch.testing.assert_close(scaled, unit, atol=1e-5, rtol=0)
    torch.testing.assert_close(scaled_moment, unit_moment, atol=1e-4, rtol=0)


def test_polar_factor_of_a_frame_overlap_is_orthogonal():
    """`_record_frame_rotation` needs the rotation part of the overlap alone."""
    torch.manual_seed(0)
    first = torch.linalg.qr(torch.randn(ROWS, RANK))[0]
    second = torch.linalg.qr(first + 0.01 * torch.randn(ROWS, RANK))[0]
    overlap = first.mT @ second

    rotation = SubspaceProjector.polar_factor(overlap)

    identity = torch.eye(RANK)
    # One Polar Express iteration, so orthogonality holds to ~1e-5 rather than
    # to machine precision. That is two orders below the in-span rotation this
    # exists to remove, which is what the tolerance has to clear.
    torch.testing.assert_close(rotation.mT @ rotation, identity, atol=1e-4, rtol=0)
    # The residue after the rotation is taken out is the symmetric part the
    # geodesic already accounts for.
    remainder = overlap @ rotation.mT
    torch.testing.assert_close(remainder, remainder.mT, atol=1e-4, rtol=0)


def test_frame_rotation_is_the_identity_when_transport_is_exact():
    """In fp32 the geodesic's own overlap is symmetric, so nothing is corrected.

    This is the property that makes the correction safe: it fires on the
    deviation between the frame computed and the frame stored, and on a path
    with no rounding there is no deviation.
    """

    torch.manual_seed(0)
    params, optimizer = _two_matrix_optimizer()
    # Captured from the commit, because step() clears its own bookkeeping before
    # returning -- read afterwards this asserted nothing at all.
    rotations = []
    commit = optimizer._commit_moments

    def capture(entries):
        rotations.extend(entry.frame_rotation for entry in entries)
        return commit(entries)

    optimizer._commit_moments = capture
    _run(params, optimizer, 4)

    observed = [rotation for rotation in rotations if rotation is not None]
    assert len(observed) == 6, "one rotation per matrix per basis update after the fitting step"
    for rotation in observed:
        torch.testing.assert_close(rotation, torch.eye(rotation.shape[-1]), atol=1e-5, rtol=0)


def test_diagnostics_are_inert_until_enabled():
    torch.manual_seed(0)
    params, optimizer = _two_matrix_optimizer()
    _run(params, optimizer, 3)

    assert optimizer.pop_diagnostics() == {}
    # Nothing was allocated to hold telemetry nobody asked for.
    assert optimizer._diagnostics is None


def test_core_diagnostics_read_sane_values():
    """No fixed key set here on purpose: a metric added or removed at `core`
    should need no test update to be noticed, only a look at what it reads."""

    torch.manual_seed(0)
    params, optimizer = _two_matrix_optimizer()
    optimizer.diagnostics = "core"
    # Three steps: the first fits the frame, so basis motion only exists after it.
    _run(params, optimizer, 3)

    diagnostics = optimizer.pop_diagnostics()
    assert -1.0 <= diagnostics["grad_moment_cosine"] <= 1.0
    assert all(isinstance(value, float) for value in diagnostics.values())
    assert all(value == value for value in diagnostics.values()), diagnostics

    assert diagnostics["transport_speed"] > 0
    assert 1.0 / RANK <= diagnostics["tangent_concentration"] <= 1.0
    assert 1.0 <= diagnostics["tangent_effective_planes"] <= RANK
    # Capture is a fraction of the gradient the frame holds, so it cannot exceed one.
    assert 0.0 < diagnostics["grad_capture"] <= 1.0 + 1e-6
    assert diagnostics["raw_grad_norm"] > 0
    # A healthy aim resolves every plane above the Gram's noise floor.
    assert diagnostics["tangent_live_fraction"] == 1.0
    assert diagnostics["update_to_param_ratio"] > 0
    assert diagnostics["nonfinite_grads"] == 0.0


def test_pop_diagnostics_clears_the_window():
    torch.manual_seed(0)
    params, optimizer = _two_matrix_optimizer()
    optimizer.diagnostics = "core"
    _run(params, optimizer, 2)

    assert optimizer.pop_diagnostics()
    # A second read with no steps in between has nothing to report, rather than
    # repeating the previous window's numbers.
    assert optimizer.pop_diagnostics() == {}
    _run(params, optimizer, 1)
    assert optimizer.pop_diagnostics()


def test_nonfinite_gradients_are_counted_not_averaged():
    torch.manual_seed(0)
    params, optimizer = _two_matrix_optimizer()
    optimizer.diagnostics = "core"
    _run(params, optimizer, 2)
    optimizer.pop_diagnostics()

    for param in params:
        param.grad = torch.randn_like(param)
    params[0].grad[0, 0] = float("nan")
    optimizer.step()

    diagnostics = optimizer.pop_diagnostics()
    assert diagnostics["nonfinite_grads"] == 1.0
    assert torch.isfinite(params[0]).all()


def test_diagnostics_do_not_change_the_trajectory():
    torch.manual_seed(0)
    quiet_params, quiet = _two_matrix_optimizer()
    torch.manual_seed(0)
    loud_params, loud = _two_matrix_optimizer()
    loud.diagnostics = "core"

    torch.manual_seed(1)
    _run(quiet_params, quiet, 4)
    torch.manual_seed(1)
    _run(loud_params, loud, 4)

    for quiet_param, loud_param in zip(quiet_params, loud_params, strict=True):
        assert torch.equal(quiet_param, loud_param)


def test_no_second_moment_state_is_allocated():
    """UsuiTrack keeps a basis and a projected first moment. Nothing else."""
    torch.manual_seed(0)
    params, optimizer = _two_matrix_optimizer()
    _run(params, optimizer, 4)

    for param in params:
        assert set(optimizer.state[param]) == {
            "basis",
            "projection_side_is_right",
            "projected_exp_avg",
            "step",
        }


def test_agreement_reads_the_moment_before_this_step_not_after():
    """The post-update moment contains `1-beta` of this very gradient, so a
    cosine against it would partly measure the gradient against itself. A first
    step, where the prior moment is exactly zero, is the sharpest case: the
    post-update moment is a pure multiple of this gradient, so a naive read
    would report perfect agreement where there is no history to agree with."""
    torch.manual_seed(0)
    params, optimizer = _two_matrix_optimizer()
    optimizer.diagnostics = "core"
    _run(params, optimizer, 1)

    cosine = optimizer.pop_diagnostics()["grad_moment_cosine"]
    assert abs(cosine) < 1e-6, cosine


def test_basis_lag_is_off_until_asked_for_and_measures_frame_motion():
    torch.manual_seed(0)
    params, optimizer = _two_matrix_optimizer()
    optimizer.diagnostics = "core"
    _run(params, optimizer, 8)
    assert "transport_lag" not in optimizer.pop_diagnostics()
    assert optimizer._lag_snapshots == {}

    optimizer.diagnostics = "full"
    optimizer.diagnostics_lag_interval = 2
    _run(params, optimizer, 8)
    drained = optimizer.pop_diagnostics()
    lag = drained["transport_lag"]
    # A frame under random gradients keeps moving, so the sine is real but is a
    # sine: bounded by 1 whatever the frame does.
    assert 0.0 < lag <= 1.0, lag
    # Curve is a fraction of a path that cancelled, so it cannot exceed one and
    # cannot be negative unless the lag exceeded the path that produced it.
    assert 0.0 <= drained["transport_curve"] < 1.0, drained["transport_curve"]
    assert drained["transport_spin"] >= 0.0


def test_basis_lag_metric_is_the_sine_of_the_principal_angle():
    """Check the quantity itself, not the plumbing: zero for a frame compared
    against itself, and exactly sin(theta) for a frame rotated by theta in one
    plane. This is the property the instrument exists for -- it must vanish iff
    motion vanishes, and it must not saturate at the small angles we expect."""
    torch.manual_seed(0)
    frame, _ = torch.linalg.qr(torch.randn(64, 4))

    def rms_sin(current, previous):
        residual = current - previous @ (previous.mT @ current)
        return float(residual.norm() / (current.shape[1] ** 0.5))

    assert rms_sin(frame, frame) < 1e-6

    for theta in (1e-4, 1e-2, 0.3):
        rotated = frame.clone()
        # Rotate one plane by theta, out of the span, keeping the frame orthonormal.
        outside = torch.linalg.qr(torch.randn(64, 1))[0]
        outside = outside - frame @ (frame.mT @ outside)
        outside = outside / outside.norm()
        rotated[:, 0] = frame[:, 0] * math.cos(theta) + outside[:, 0] * math.sin(theta)
        # One plane of four moved by theta, so the RMS sine is sin(theta)/2.
        assert abs(rms_sin(rotated, frame) - math.sin(theta) / 2) < 1e-5, theta


def test_spin_separates_in_span_rotation_from_subspace_motion():
    """The property spin exists for: it must see the motion lag is blind to.

    Rotating a frame's columns among themselves moves the subspace not at all,
    so lag reads zero -- and it renames every coordinate the projected moment is
    stored in, which under identity transport is total corruption. Spin is the
    skew part of `Q_old^T Q_new`, which is exactly zero for the symmetric
    overlap a single Grassmann geodesic produces and nonzero here.
    """
    torch.manual_seed(0)
    frame, _ = torch.linalg.qr(torch.randn(64, 8))

    def reads(current, previous):
        root = current.shape[1] ** 0.5
        overlap = previous.mT @ current
        lag = float((current - previous @ overlap).norm() / root)
        spin = float((overlap - overlap.mT).norm() / (2.0 * root))
        return lag, spin

    # A pure in-span rotation: same span, different columns.
    theta = 0.2
    rotation = torch.eye(8)
    rotation[0, 0] = rotation[1, 1] = math.cos(theta)
    rotation[0, 1], rotation[1, 0] = -math.sin(theta), math.sin(theta)
    lag, spin = reads(frame @ rotation, frame)
    assert lag < 1e-6, lag
    assert spin > 0.05, spin

    # One exact Grassmann geodesic: the overlap is `V cos(theta) V^T`, symmetric,
    # so all of the motion lands in lag and none of it in spin.
    tangent = torch.randn(64, 8)
    tangent = tangent - frame @ (frame.mT @ tangent)
    values, vectors = torch.linalg.eigh(0.5 * (tangent.mT @ tangent + (tangent.mT @ tangent).mT))
    moved = SubspaceProjector.oja_geodesic_from_eigh(frame, tangent, values, vectors, 0.05)
    lag, spin = reads(moved, frame)
    assert lag > 1e-3, lag
    assert spin < 1e-5, spin


def test_the_frame_turns_from_the_very_first_basis_update():
    """There is no cold start any more, and that is a deliberate change.

    The gain used to return zero with no history -- no evidence the aim repeats,
    no turn -- which is what kept the first update off the stability cliff back
    when a full-magnitude turn at an `eta` chosen for a scale near 0.05 failed
    `eigh` outright. Masking dead planes removed that cliff: the failure was
    rounding artifacts promoted to unit-norm directions by `1 / sigma`, not step
    size. With a constant turn there is no history to wait for, so the first
    update turns like every other one.
    """

    def moved(before, after):
        residual = after.mT - before.mT @ (before @ after.mT)
        return float(residual.norm() / math.sqrt(RANK))

    torch.manual_seed(0)
    weight = torch.nn.Parameter(torch.randn(ROWS, COLS))
    optimizer = UsuiTrack([weight], lr=0.01, rank=RANK, side="right")

    _run([weight], optimizer, 1)
    first = optimizer.state[weight]["basis"].clone().float()
    _run([weight], optimizer, 1)
    assert moved(first, optimizer.state[weight]["basis"].float()) > 1e-3


def test_every_live_plane_turns_by_exactly_eta():
    """The turn is `eta`, not `eta` times a meter, and that is now an equality.

    Every live plane of the polar tangent has singular value one, so the geodesic
    takes the angle `eta` on each of them and the chordal residual per plane is
    `sin(eta)`. Under the agreement gain this could only be asserted as an upper
    bound; with a constant turn the bound is the value, which is the sharpest
    statement available that no magnitude survives into the frame's motion.
    """

    torch.manual_seed(0)
    weight = torch.nn.Parameter(torch.randn(ROWS, COLS))
    optimizer = UsuiTrack([weight], lr=0.01, rank=RANK, side="right")
    optimizer.diagnostics = "core"
    _run([weight], optimizer, 12)

    before = optimizer.state[weight]["basis"].clone().float()
    _run([weight], optimizer, 1)
    after = optimizer.state[weight]["basis"].float()

    drained = optimizer.pop_diagnostics()
    # The equality below only means "every plane turned by eta" if every plane
    # was live to begin with.
    assert drained["tangent_live_fraction"] == 1.0
    # Chordal distance per plane, the same unit `transport_speed` reports.
    residual = after.mT - before.mT @ (before @ after.mT)
    moved = float(residual.norm() / math.sqrt(RANK))
    assert moved == pytest.approx(math.sin(0.01), rel=2e-3), moved


def test_diagnostics_tier_rejects_a_typo_instead_of_silently_downgrading():
    """The failure this guards is silent, which is why it is worth a guard.

    Tiers are string comparisons on the hot path: `"ful"` is not `"off"`, so core
    switches on, and it is not `"full"`, so the snapshot reads stay off. The run
    then looks healthy while omitting exactly the telemetry that was asked for.
    """

    weight = torch.nn.Parameter(torch.randn(ROWS, COLS))
    optimizer = UsuiTrack([weight], lr=0.01, rank=RANK, side="right")

    with pytest.raises(ValueError, match="diagnostics must be one of"):
        optimizer.diagnostics = "ful"
    assert optimizer.diagnostics == "off"

    for tier in ("core", "full", "off"):
        optimizer.diagnostics = tier
        assert optimizer.diagnostics == tier


def test_leaving_the_full_tier_drops_the_snapshot_state():
    """Snapshots only mean something against a live window.

    Kept across a tier change, the first reading after re-enabling would compare
    the frame against one from an arbitrarily distant past and report it as one
    window's travel.
    """

    torch.manual_seed(0)
    weight = torch.nn.Parameter(torch.randn(ROWS, COLS))
    optimizer = UsuiTrack([weight], lr=0.01, rank=RANK, side="right")
    optimizer.diagnostics = "full"
    optimizer.diagnostics_lag_interval = 2
    _run([weight], optimizer, 6)
    assert optimizer._lag_snapshots

    optimizer.diagnostics = "core"
    assert optimizer._lag_snapshots == {}
    assert optimizer._lag_path == {}


def test_compiled_and_eager_kernels_produce_the_same_update():
    """The guard for the defect this file did not have.

    `compile_tensor_kernels` used to select a *second* implementation of the
    orthogonalization -- one that hardcoded the aspect scale while the eager one
    read a four-way mode constant -- so the optimizer's update depended on a
    compile flag and nothing here was watching. There is one implementation now
    and `torch.compile` wraps it, which is a property worth holding rather than
    a state of affairs worth assuming: this fails the moment a compiled path
    grows its own copy of the maths again.

    Tolerance rather than bitwise identity, because Inductor is free to fuse and
    reorder float operations. What is being asserted is that the two paths
    compute the same function, not that they emit the same instructions.
    """

    torch.manual_seed(0)
    base = torch.randn(ROWS, COLS)
    gradients = [torch.randn(ROWS, COLS) for _ in range(4)]

    trained = []
    for compiled in (False, True):
        weight = torch.nn.Parameter(base.clone())
        optimizer = UsuiTrack(
            [weight], lr=0.01, rank=RANK, side="right", compile_tensor_kernels=compiled
        )
        for gradient in gradients:
            weight.grad = gradient.clone()
            optimizer.step()
        trained.append(weight.detach().clone())

    torch.testing.assert_close(trained[0], trained[1], rtol=1e-5, atol=1e-6)


def test_the_aspect_scale_is_muon_at_full_rank_and_attenuated_below_it():
    """What the scale still guarantees, and what it stopped guaranteeing.

    Muon's factor exists to make the orthogonalized update's Frobenius norm come
    out at `sqrt(rows)` however the matrix is stored -- transposing a weight must
    not change how hard it is pushed. That holds only while the orthogonalized
    object has the parameter's own two dimensions. Ours is the projected moment,
    `[d, r]`, whose polar factor has norm `sqrt(r)`, so the product carries a
    residual `sqrt(r / min(m, n))` that varies across the fleet with `min(m, n)`
    and not with anything anyone chose.

    Pinned here because the term reads as settled and is not: a future arm that
    changes it should have to change this test and say why.
    """

    tall, wide = (4608, 1024), (1024, 4608)
    assert UsuiTrack._muon_aspect_scale(tall) == pytest.approx(math.sqrt(4.5))
    # Clamped at one below the diagonal, which is what makes the two orientations
    # differ by the aspect ratio rather than agree.
    assert UsuiTrack._muon_aspect_scale(wide) == 1.0

    # At full rank the invariant holds: `sqrt(min(m,n)) * scale == sqrt(rows)`.
    for shape in (tall, wide):
        full_rank_norm = math.sqrt(min(shape)) * UsuiTrack._muon_aspect_scale(shape)
        assert full_rank_norm == pytest.approx(math.sqrt(shape[0]))

    # Below it the norm is short by `sqrt(r / min(m, n))`, and that shortfall is
    # a function of the matrix's shape -- the spread nobody picked.
    rank = 128
    attenuation = [math.sqrt(rank / min(shape)) for shape in ((4096, 1024), (4096, 512))]
    assert attenuation[0] == pytest.approx(0.3536, abs=1e-4)
    assert attenuation[1] == pytest.approx(0.5, abs=1e-4)


def test_matrices_sharing_a_scale_but_not_a_shape_still_bucket_correctly():
    """The bucketing win: keying on `(projected shape, scale)` instead of
    `(projected shape, original shape)` lets differently-shaped matrices that
    happen to share a scale batch into one Newton-Schulz call. `projected_exp_avg`
    for a right-tracked matrix is `(rows, r)` and never encodes `cols`, so two
    matrices with equal `rows` but different `cols` were split by the old key
    even though their moments already stack; here they also share a scale
    (`rows < cols` for both, so `_muon_aspect_scale` clamps to `1.0`) and so now
    land in one bucket. Must train identically whether they share an optimizer
    or not -- the grouping is bookkeeping, not maths.
    """

    torch.manual_seed(5)
    shapes = [(64, 512), (64, 768)]
    base = {shape: torch.randn(*shape) for shape in shapes}
    gradients = {shape: [torch.randn(*shape) for _ in range(6)] for shape in shapes}
    for shape in shapes:
        assert UsuiTrack._muon_aspect_scale(shape) == 1.0

    together = {shape: torch.nn.Parameter(base[shape].clone()) for shape in shapes}
    optimizer = UsuiTrack(list(together.values()), lr=0.01, rank=RANK, side="right")
    for step in range(6):
        for shape, weight in together.items():
            weight.grad = gradients[shape][step].clone()
        optimizer.step()

    separate = {shape: torch.nn.Parameter(base[shape].clone()) for shape in shapes}
    solo_optimizers = {shape: UsuiTrack([separate[shape]], lr=0.01, rank=RANK, side="right") for shape in shapes}
    for step in range(6):
        for shape in shapes:
            separate[shape].grad = gradients[shape][step].clone()
            solo_optimizers[shape].step()

    # Not bitwise: batched and per-matrix Newton-Schulz call different BLAS
    # kernels, so they round differently at the ulp level -- the same tolerance
    # every other batched-vs-solo comparison in this file already carries.
    for shape in shapes:
        torch.testing.assert_close(together[shape], separate[shape], rtol=1e-5, atol=1e-6)


def test_accumulated_micro_batches_average_to_the_single_step_they_repeat():
    """`k` copies of one gradient are that gradient.

    The invariant that says the accumulation is an average and not a sum: the
    weighted mean of `k` identical orthogonalized directions is the direction,
    and the mean of `k` identical Oja tangents is the tangent, so the step and
    the geodesic land where one micro-batch would have put them. A sum would
    scale the moment's increment by `k` and the frame's aim with it.
    """

    torch.manual_seed(0)
    single = torch.nn.Parameter(torch.randn(ROWS, COLS))
    accumulated = torch.nn.Parameter(single.detach().clone())
    one = UsuiTrack([single], lr=0.01, rank=RANK, side="right")
    many = UsuiTrack([accumulated], lr=0.01, rank=RANK, side="right")

    for _ in range(3):
        gradient = torch.randn_like(single)
        single.grad = gradient.clone()
        one.step()
        for _ in range(4):
            accumulated.grad = gradient.clone()
            many.accumulate()
        many.step()
        torch.testing.assert_close(accumulated, single)


def test_importance_weights_are_a_weighted_mean_over_micro_batches():
    """One micro-batch at weight 3 is three micro-batches at weight 1.

    Stated as an equivalence rather than against a recomputed mean, so the test
    does not reimplement the polar map to check it. The sums normalize, so the
    weights carry only their ratio.
    """

    torch.manual_seed(3)
    weighted = torch.nn.Parameter(torch.randn(ROWS, COLS))
    repeated = torch.nn.Parameter(weighted.detach().clone())
    by_weight = UsuiTrack([weighted], lr=0.01, rank=RANK, side="right")
    by_repeat = UsuiTrack([repeated], lr=0.01, rank=RANK, side="right")

    for _ in range(3):
        first = torch.randn_like(weighted)
        second = torch.randn_like(weighted)

        weighted.grad = first.clone()
        by_weight.accumulate(weight=1.0)
        weighted.grad = second.clone()
        by_weight.accumulate(weight=3.0)
        by_weight.step()

        repeated.grad = first.clone()
        by_repeat.accumulate()
        for _ in range(3):
            repeated.grad = second.clone()
            by_repeat.accumulate()
        by_repeat.step()

        torch.testing.assert_close(weighted, repeated)


def test_accumulation_moves_neither_the_moment_nor_the_basis():
    """Both move exactly once per step, which is what keeps the telemetry honest.

    The frame being held is also a correctness requirement and not only a
    reporting one: every micro-batch's tangent has to live in one tangent space
    for their sum to be a vector rather than an average over basepoints.
    """

    torch.manual_seed(4)
    weight = torch.nn.Parameter(torch.randn(ROWS, COLS))
    optimizer = UsuiTrack([weight], lr=0.01, rank=RANK, side="right")
    weight.grad = torch.randn_like(weight)
    optimizer.step()

    basis = optimizer.state[weight]["basis"].clone()
    moment = optimizer.state[weight]["projected_exp_avg"].clone()
    for _ in range(4):
        weight.grad = torch.randn_like(weight)
        optimizer.accumulate()
        torch.testing.assert_close(optimizer.state[weight]["basis"], basis, rtol=0, atol=0)
        torch.testing.assert_close(optimizer.state[weight]["projected_exp_avg"], moment, rtol=0, atol=0)

    optimizer.step()
    assert not torch.equal(optimizer.state[weight]["basis"], basis)
    assert not torch.equal(optimizer.state[weight]["projected_exp_avg"], moment)


def test_micro_batch_agreement_spans_repeats_to_opposites():
    """The meter's two analytic fixed points, and its absence at one micro-batch.

    Repeats agree perfectly and opposites disagree perfectly, so 1 and -1 are
    exact rather than approximate -- the polar map is an odd function of its
    input, which is what makes the opposite case land on -1 and not merely
    negative.
    """

    torch.manual_seed(5)
    weight = torch.nn.Parameter(torch.randn(ROWS, COLS))
    optimizer = UsuiTrack([weight], lr=0.01, rank=RANK, side="right")
    optimizer.diagnostics = "core"
    gradient = torch.randn_like(weight)

    weight.grad = gradient.clone()
    optimizer.step()
    assert "micro_batch_agreement" not in optimizer.pop_diagnostics()

    for _ in range(3):
        weight.grad = gradient.clone()
        optimizer.accumulate()
    optimizer.step()
    assert optimizer.pop_diagnostics()["micro_batch_agreement"] == pytest.approx(1.0, abs=1e-3)

    weight.grad = gradient.clone()
    optimizer.accumulate()
    weight.grad = -gradient
    optimizer.accumulate()
    optimizer.step()
    assert optimizer.pop_diagnostics()["micro_batch_agreement"] == pytest.approx(-1.0, abs=1e-3)


def test_rank_calibrator_geometric_mean_sits_between_mean_and_median():
    """The estimator matching a power-law quantity, on a deliberately skewed label.

    Live counts follow `live ~ r^alpha`, so their spread is log-normal-ish and
    the arithmetic mean is pulled by the high tail. The geometric mean is the
    central estimator of that shape and lands below the arithmetic one.
    """

    from usuitrack.diagnostics import RankCalibrator

    calibrator = RankCalibrator()
    params = [object() for _ in range(4)]
    counts = torch.tensor([2.0, 4.0, 8.0, 64.0])
    for _ in range(3):
        calibrator.observe("skewed", 128, params, counts)
        calibrator.roll()

    row = calibrator.report()["skewed"]
    assert row["geomean"] == pytest.approx(8.0, rel=1e-5)
    assert row["geomean"] < row["mean"]
    # torch.median takes the lower of the two middle values, not their average.
    assert row["median"] == pytest.approx(4.0)


def test_rank_calibrator_geometric_mean_floors_a_dead_frame_at_one_plane():
    """`log(0)` is not a measurement, and no rank names fewer than one plane.

    The pair is the honest report: the geometric mean says one, the arithmetic
    mean says zero, and only together do they say the frames were dead.
    """

    from usuitrack.diagnostics import RankCalibrator

    calibrator = RankCalibrator()
    params = [object(), object()]
    for _ in range(3):
        calibrator.observe("dead", 32, params, torch.zeros(2))
        calibrator.roll()

    row = calibrator.report()["dead"]
    assert row["geomean"] == pytest.approx(1.0)
    assert row["mean"] == pytest.approx(0.0)
