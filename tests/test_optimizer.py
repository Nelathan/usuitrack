import copy
import math

import pytest
import torch

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
    """UsuiTrack has no accumulation-aware bookkeeping: it just reads
    param.grad at step() time. Two backward() calls before one step() (no
    release_matrix_grads, no zero_grad in between) must therefore behave
    identically to a single step() on the pre-summed gradient."""

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
        torch.zeros(projected_shape),
        1.0,  # grad_clip_norm
        0.95,  # beta
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

    _projected, tangent, _norm = UsuiTrack._prepare_tracker_right_tensors(
        gradient, basis, torch.zeros(ROWS, RANK), 1.0, 0.95
    )
    assert float(tangent.abs().max()) < 1e-5, float(tangent.abs().max())


def test_kernel_matches_a_hand_rolled_step():
    """The fused kernel is the written-out update, not a path that merely runs."""
    torch.manual_seed(0)
    gradient = torch.randn(ROWS, COLS)
    projector = SubspaceProjector(rank=RANK, side="right")
    projector.fit(torch.randn(ROWS, COLS))
    basis = projector.basis
    assert basis is not None

    moment = torch.randn(ROWS, RANK).mul_(1e-3)
    expected = moment.clone()
    projected, _tangent, _norm = UsuiTrack._prepare_tracker_right_tensors(
        gradient, basis, moment, 1.0, 0.95
    )

    clipped = gradient * (1.0 / gradient.float().norm()).clamp(max=1.0)
    assert torch.allclose(projected, clipped @ basis.mT, atol=1e-6)
    expected.mul_(0.95).add_(projected, alpha=0.05)
    assert torch.allclose(moment, expected, atol=1e-6)


def test_diagnostics_are_inert_until_enabled():
    torch.manual_seed(0)
    params, optimizer = _two_matrix_optimizer()
    _run(params, optimizer, 3)

    assert optimizer.pop_diagnostics() == {}
    # Nothing was allocated to hold telemetry nobody asked for.
    assert optimizer._diagnostics is None


def test_diagnostics_report_every_documented_metric():
    torch.manual_seed(0)
    params, optimizer = _two_matrix_optimizer()
    optimizer.diagnostics = "core"
    # Three steps: the first fits the frame, so basis motion only exists after it.
    _run(params, optimizer, 3)

    diagnostics = optimizer.pop_diagnostics()
    assert set(diagnostics) == {
        "transport_speed",
        "turn_fraction",
        "tangent_live_fraction",
        "agreement_ceiling",
        "tangent_concentration",
        "tangent_participation",
        "projected_grad_norm",
        "grad_to_moment_ratio",
        "grad_moment_cosine",
        "update_to_param_ratio",
        "nonfinite_grads",
    }
    assert -1.0 <= diagnostics["grad_moment_cosine"] <= 1.0
    assert all(isinstance(value, float) for value in diagnostics.values())
    assert all(value == value for value in diagnostics.values()), diagnostics

    assert diagnostics["transport_speed"] > 0
    assert 1.0 / RANK <= diagnostics["tangent_concentration"] <= 1.0
    assert 1.0 / RANK <= diagnostics["tangent_participation"] <= 1.0
    # Structural, not measured: the controller can only ever scale the turn down.
    assert 0.0 <= diagnostics["turn_fraction"] <= 1.0
    # A healthy aim resolves every plane above the Gram's noise floor.
    assert diagnostics["tangent_live_fraction"] == 1.0
    assert diagnostics["agreement_ceiling"] > 0.0
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


def test_agreement_controller_holds_the_frame_until_the_aim_has_repeated():
    """No history means no evidence the aim repeats, so the first turn is zero.

    This is the property that keeps the first basis update off the stability
    cliff. Before it was there the cold start took a full-magnitude turn at an
    `eta` chosen for a scale near 0.05, and `eigh` on the tangent Gram failed
    outright. It costs exactly one basis update.

    "Held" is not bitwise identity: a scale of zero still runs the geodesic and
    its Polar-Express retraction, so the frame comes back changed at the
    retraction's own error. That is the level the check is written against.
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
    # Step two has a stored aim but no gain yet, so the frame is still held.
    held = moved(first, optimizer.state[weight]["basis"].float())
    assert held < 1e-5, held

    _run([weight], optimizer, 3)
    assert moved(first, optimizer.state[weight]["basis"].float()) > 100 * held


def test_turn_fraction_never_exceeds_the_bare_step():
    """`scale <= 1` is a bound on the geodesic, not an observation about it.

    Every live plane of the polar tangent has singular value one, so the angle
    the geodesic takes is `eta * scale` exactly. The clamp is therefore the whole
    guarantee that annealing can only ever slow the frame relative to turning
    every plane by `eta`, whatever the meter reads.
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
    assert 0.0 <= drained["turn_fraction"] <= 1.0
    # Chordal distance per plane, the same unit `transport_speed` reports, and
    # bounded above by the angle a scale of one would produce.
    residual = after.mT - before.mT @ (before @ after.mT)
    moved = float(residual.norm() / math.sqrt(RANK))
    assert moved <= math.sin(0.01) + 1e-6, moved


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
