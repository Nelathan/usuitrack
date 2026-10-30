import copy
import math

import pytest
import torch
from torch import Tensor

from usuitrack import UsuiTrack
from usuitrack import frame
from usuitrack.diagnostics import READS
from usuitrack.optimizer import GEODESIC_STEPSIZE, write_weight
from usuitrack.polar import unit_polar


# A representative transformer-ish weight. Small toy shapes like (8, 4) put the
# rank straight into its floor, so every test would silently exercise a rank-1
# frame. RANK_FRACTION is chosen so this shape runs at exactly RANK, below the cap.
ROWS, COLS, RANK = 256, 128, 16
RANK_FRACTION = RANK / math.sqrt(ROWS * COLS)


def _two_matrix_optimizer(**kwargs):
    params = [
        torch.nn.Parameter(torch.randn(ROWS, COLS)),
        torch.nn.Parameter(torch.randn(ROWS, COLS)),
    ]
    return params, UsuiTrack(params, lr=1e-3, rank_fraction=RANK_FRACTION, **kwargs)


def _run(params, optimizer, steps):
    for _ in range(steps):
        for param in params:
            param.grad = torch.randn_like(param)
        optimizer.step()


def _backward(param: Tensor, gradient: Tensor) -> None:
    """Deliver `gradient` through autograd, so post-accumulate hooks fire."""
    (param * gradient).sum().backward()


def test_rejects_non_matrix():
    with pytest.raises(ValueError, match="2D matrix"):
        UsuiTrack([torch.nn.Parameter(torch.randn(4))])


def test_rejects_an_unknown_side():
    with pytest.raises(ValueError, match="side"):
        UsuiTrack([torch.nn.Parameter(torch.randn(ROWS, COLS))], side="residual")


def test_frame_moves_every_step_after_the_one_that_fits_it():
    weight = torch.nn.Parameter(torch.randn(ROWS, COLS))
    optimizer = UsuiTrack([weight], lr=0.01, rank_fraction=RANK_FRACTION)

    _run([weight], optimizer, 1)
    fitted = optimizer.state[weight]["frame"].clone()
    _run([weight], optimizer, 1)
    moved = optimizer.state[weight]["frame"]
    residual = moved - fitted @ (fitted.mT @ moved)
    assert float(residual.norm() / math.sqrt(RANK)) > 1e-3


def test_only_a_frame_and_a_moment_are_kept():
    """UsuiTrack keeps a frame and a projected first moment. Nothing else."""
    torch.manual_seed(0)
    params, optimizer = _two_matrix_optimizer()
    _run(params, optimizer, 4)

    for param in params:
        state = optimizer.state[param]
        assert set(state) == {"frame", "moment"}
        assert tuple(state["frame"].shape) == (COLS, RANK)
        assert tuple(state["moment"].shape) == (ROWS, RANK)


def test_release_matches_ordinary_step_exactly():
    torch.manual_seed(0)
    ordinary_weight = torch.nn.Parameter(torch.randn(ROWS, COLS))
    released_weight = torch.nn.Parameter(ordinary_weight.detach().clone())
    ordinary = UsuiTrack([ordinary_weight], lr=0.01, rank_fraction=RANK_FRACTION)
    released = UsuiTrack([released_weight], lr=0.01, rank_fraction=RANK_FRACTION, release_matrix_grads=True)

    for _ in range(3):
        gradient = torch.randn(ROWS, COLS)
        _backward(ordinary_weight, gradient)
        _backward(released_weight, gradient)
        assert released_weight.grad is None
        ordinary.step()
        released.step()
        torch.testing.assert_close(ordinary_weight, released_weight, rtol=0, atol=0)


def test_release_hooks_cover_a_group_added_later():
    weight = torch.nn.Parameter(torch.randn(ROWS, COLS))
    late = torch.nn.Parameter(torch.randn(ROWS, COLS))
    optimizer = UsuiTrack([weight], rank_fraction=RANK_FRACTION, release_matrix_grads=True)
    optimizer.add_param_group({"params": [late]})

    _backward(late, torch.randn(ROWS, COLS))
    assert late.grad is None


def test_plain_gradient_accumulation_matches_a_presummed_gradient():
    """Two backward() calls before one step() leave a summed gradient, and
    step() reads it as one gradient: the estimator a batch of both gives."""

    torch.manual_seed(2)
    accumulated = torch.nn.Parameter(torch.randn(ROWS, COLS))
    presummed = torch.nn.Parameter(accumulated.detach().clone())
    first, second = torch.randn(ROWS, COLS), torch.randn(ROWS, COLS)

    _backward(accumulated, first)
    _backward(accumulated, second)
    presummed.grad = first + second

    UsuiTrack([accumulated], lr=0.01, rank_fraction=RANK_FRACTION).step()
    UsuiTrack([presummed], lr=0.01, rank_fraction=RANK_FRACTION).step()
    torch.testing.assert_close(accumulated, presummed, rtol=0, atol=0)


def test_accumulation_under_release_is_refused_with_the_reason():
    weight = torch.nn.Parameter(torch.randn(ROWS, COLS))
    optimizer = UsuiTrack([weight], rank_fraction=RANK_FRACTION, release_matrix_grads=True)
    _backward(weight, torch.randn(ROWS, COLS))
    assert optimizer._pending
    with pytest.raises(RuntimeError, match="accumulate gradients with it off"):
        _backward(weight, torch.randn(ROWS, COLS))


def test_zero_grad_drops_pending_prepared_work():
    """An escape hatch for a loop that bails out between backward and step."""

    weight = torch.nn.Parameter(torch.randn(ROWS, COLS))
    optimizer = UsuiTrack([weight], rank_fraction=RANK_FRACTION, release_matrix_grads=True)
    _backward(weight, torch.randn(ROWS, COLS))
    with pytest.warns(UserWarning, match="dropped 1 prepared"):
        optimizer.zero_grad()
    before = weight.detach().clone()
    optimizer.step()
    assert torch.equal(weight, before)


def test_state_dict_resume_matches_uninterrupted_run():
    torch.manual_seed(1)
    first = torch.nn.Parameter(torch.randn(ROWS, COLS))
    first_optimizer = UsuiTrack([first], lr=0.01, rank_fraction=RANK_FRACTION)
    _run([first], first_optimizer, 2)

    second = torch.nn.Parameter(first.detach().clone())
    second_optimizer = UsuiTrack([second], lr=0.01, rank_fraction=RANK_FRACTION)
    second_optimizer.load_state_dict(copy.deepcopy(first_optimizer.state_dict()))

    gradient = torch.randn_like(first)
    first.grad = gradient.clone()
    second.grad = gradient.clone()
    first_optimizer.step()
    second_optimizer.step()
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_a_loaded_group_is_the_one_the_step_reads():
    """torch rebuilds the group dicts on load; a hyperparameter changed through
    `param_groups` afterwards must reach the matrices that release hooks prepare."""

    torch.manual_seed(1)
    weight = torch.nn.Parameter(torch.randn(ROWS, COLS))
    optimizer = UsuiTrack([weight], rank_fraction=RANK_FRACTION, release_matrix_grads=True)
    optimizer.load_state_dict(optimizer.state_dict())
    optimizer.param_groups[0]["side"] = "left"

    _backward(weight, torch.randn(ROWS, COLS))
    optimizer.step()
    assert tuple(optimizer.state[weight]["frame"].shape) == (ROWS, RANK)


def test_the_moment_averages_unit_gradients_and_the_step_is_its_polar_factor():
    """The average runs on gradients, and the polar map runs once, on the average.

    One step from a zero moment leaves `(1 - beta)` of the projected gradient at
    unit norm in state -- not of its polar factor -- and the weight moves by the
    polar factor of that moment at unit norm, lifted through the frame that
    measured it, at `lr` of the weight's norm.
    """
    torch.manual_seed(0)
    weight = torch.nn.Parameter(torch.randn(ROWS, COLS))
    before = weight.detach().clone()
    optimizer = UsuiTrack([weight], lr=0.1, rank_fraction=RANK_FRACTION, beta=0.9)

    gradient = torch.randn(ROWS, COLS)
    weight.grad = gradient.clone()
    optimizer.step()

    state = optimizer.state[weight]
    q = state["frame"]
    projected = gradient @ q
    torch.testing.assert_close(state["moment"], 0.1 * projected / projected.norm(), atol=1e-6, rtol=0)
    direction = unit_polar(projected)
    torch.testing.assert_close((before - weight.detach()) / (0.1 * before.norm()), direction @ q.mT, atol=1e-4, rtol=0)


def test_a_contested_direction_steps_as_far_as_an_agreed_one():
    """The moment decides where the step points, never how far it goes.

    Both arms feed gradients of identical size; only the sign pattern differs,
    so one moment grows to full length and the other cancels. `G` and `-G`
    share `G^T G`, so the frame contributes nothing to the difference.
    """

    lr = 0.01

    def final_relative_step(signs: list[int]) -> float:
        torch.manual_seed(0)
        weight = torch.nn.Parameter(torch.randn(ROWS, COLS))
        optimizer = UsuiTrack([weight], lr=lr, rank_fraction=RANK_FRACTION, beta=0.9)
        gradient = torch.randn(ROWS, COLS)
        for sign in signs:
            before = weight.detach().clone()
            weight.grad = gradient * sign
            optimizer.step()
            motion = float((weight.detach() - before).norm() / before.norm())
        return motion

    assert final_relative_step([1] * 8) == pytest.approx(lr, rel=0.1)
    assert final_relative_step([1, -1] * 4) == pytest.approx(lr, rel=0.1)


def test_the_update_is_invariant_to_gradient_scale():
    """Why there is no gradient clip: nothing downstream sees the magnitude.

    Each projected gradient enters the moment at unit norm, and the Oja tangent
    divides two quadratics in `G`. A 180x blip is included.
    """

    def trajectory(scales: list[float]) -> tuple[Tensor, Tensor]:
        torch.manual_seed(0)
        weight = torch.nn.Parameter(torch.randn(ROWS, COLS))
        optimizer = UsuiTrack([weight], lr=0.01, rank_fraction=RANK_FRACTION, beta=0.9)
        generator = torch.Generator().manual_seed(7)
        for scale in scales:
            weight.grad = torch.randn(ROWS, COLS, generator=generator) * scale
            optimizer.step()
        return weight.detach().clone(), optimizer.state[weight]["moment"].clone()

    scaled, scaled_moment = trajectory([180.0 if step == 5 else 1.0 for step in range(12)])
    unit, unit_moment = trajectory([1.0] * 12)
    torch.testing.assert_close(scaled, unit, atol=1e-5, rtol=0)
    torch.testing.assert_close(scaled_moment, unit_moment, atol=1e-4, rtol=0)


def test_lr_is_the_relative_step_in_every_module():
    """SPEC step 8: `||dW|| / ||W||` is `lr` whatever the shape, side or scale.
    The polar map's singular values are not exactly one, hence a tolerance on
    shapes; the weight's scale enters exactly."""

    lr = 1e-2

    def relative_step(shape: tuple[int, int], weight_scale: float) -> float:
        torch.manual_seed(0)
        weight = torch.nn.Parameter(weight_scale * torch.randn(*shape))
        before = weight.detach().clone()
        optimizer = UsuiTrack([weight], lr=lr, rank_fraction=RANK_FRACTION)
        weight.grad = torch.randn(*shape)
        optimizer.step()
        return ((before - weight.detach()).norm() / before.norm()).item()

    reference = relative_step((ROWS, COLS), 1.0)
    assert relative_step((ROWS, COLS), 30.0) == pytest.approx(reference, rel=1e-5)
    for shape in ((ROWS, COLS), (COLS, ROWS), (512, 32), (32, 512)):
        assert relative_step(shape, 1.0) == pytest.approx(lr, rel=0.1)


def _write_inputs(side: str, dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor]:
    """The layouts the optimizer hands over: the polar factor column-major, the
    frame row-major. Each other layout is another compiled graph."""
    other, frame_length = (ROWS, COLS) if side == "right" else (COLS, ROWS)
    direction = torch.linalg.qr(torch.randn(other, RANK))[0]
    basis = torch.linalg.qr(torch.randn(frame_length, RANK))[0].to(dtype).contiguous()
    return (0.01 * torch.randn(ROWS, COLS)).to(dtype), direction, basis


@pytest.mark.parametrize("side", ["left", "right"])
@torch.no_grad()
def test_the_compiled_write_is_the_eager_write(side):
    """The compiled weight write against its own eager function, fp32."""

    torch.manual_seed(0)
    weight, direction, basis = _write_inputs(side, torch.float32)
    compiled, eager = weight.clone(), weight.clone()
    compiled_norm = write_weight(compiled, direction, basis, side, 1e-2, 0.1)
    eager_norm = write_weight.__wrapped__(eager, direction, basis, side, 1e-2, 0.1)
    torch.testing.assert_close(compiled, eager)
    torch.testing.assert_close(compiled_norm, eager_norm)


@pytest.mark.parametrize("side", ["left", "right"])
@torch.no_grad()
def test_the_compiled_bf16_write_lands_the_fp32_step_in_expectation(side):
    """Inductor draws its own rounding bits, so bf16 matches eager only in
    expectation. A step of a fraction of an ulp, averaged over draws, lands on
    the fp32 step to within the rounding noise (~0.05 at 400 draws); rounding to
    nearest misses it by ~0.6."""

    torch.manual_seed(0)
    start, direction, basis = _write_inputs(side, torch.bfloat16)
    exact = start.float()
    write_weight.__wrapped__(exact, direction, basis, side, 5e-4, 0.1)
    landed = torch.zeros_like(exact)
    for _ in range(400):
        weight = start.clone()
        write_weight(weight, direction, basis, side, 5e-4, 0.1)
        landed += weight.float() / 400
    assert ((landed - exact).norm() / (exact - start.float()).norm()).item() < 0.1


@pytest.mark.parametrize("side", ["left", "right"])
def test_a_forced_side_is_the_transposed_problem(side):
    """SPEC identity: transposing the problem and swapping sides transposes the update."""

    torch.manual_seed(0)
    base = torch.randn(ROWS, COLS)
    gradients = [torch.randn(ROWS, COLS) for _ in range(4)]
    swapped = {"left": "right", "right": "left"}[side]

    weight = torch.nn.Parameter(base.clone())
    transposed = torch.nn.Parameter(base.mT.clone())
    optimizer = UsuiTrack([weight], lr=0.01, rank_fraction=RANK_FRACTION, side=side)
    mirror = UsuiTrack([transposed], lr=0.01, rank_fraction=RANK_FRACTION, side=swapped)
    for gradient in gradients:
        weight.grad = gradient.clone()
        transposed.grad = gradient.mT.clone()
        optimizer.step()
        mirror.step()
    torch.testing.assert_close(weight, transposed.mT, rtol=1e-5, atol=1e-6)


def test_matrices_stepped_together_match_each_stepped_alone():
    """Same-rank tangent Grams share one `eigh` across shapes; a third rank does
    not. Grouping is bookkeeping, not maths. Not bitwise: batched and solo
    calls round differently at the ulp level."""

    torch.manual_seed(5)
    shapes = [(64, 512), (64, 520), (96, 96)]
    ranks = [frame.rank(torch.Size(shape), RANK_FRACTION) for shape in shapes]
    assert ranks[0] == ranks[1] != ranks[2]
    base = {shape: torch.randn(*shape) for shape in shapes}
    gradients = {shape: [torch.randn(*shape) for _ in range(6)] for shape in shapes}

    together = {shape: torch.nn.Parameter(base[shape].clone()) for shape in shapes}
    optimizer = UsuiTrack(list(together.values()), lr=0.01, rank_fraction=RANK_FRACTION, side="right")
    for step in range(6):
        for shape, weight in together.items():
            weight.grad = gradients[shape][step].clone()
        optimizer.step()

    separate = {shape: torch.nn.Parameter(base[shape].clone()) for shape in shapes}
    solo = {shape: UsuiTrack([separate[shape]], lr=0.01, rank_fraction=RANK_FRACTION, side="right") for shape in shapes}
    for step in range(6):
        for shape in shapes:
            separate[shape].grad = gradients[shape][step].clone()
            solo[shape].step()

    for shape in shapes:
        torch.testing.assert_close(together[shape], separate[shape], rtol=1e-5, atol=1e-6)


def test_the_frame_turns_from_the_very_first_basis_update():
    torch.manual_seed(0)
    weight = torch.nn.Parameter(torch.randn(ROWS, COLS))
    optimizer = UsuiTrack([weight], lr=0.01, rank_fraction=RANK_FRACTION)
    _run([weight], optimizer, 1)
    first = optimizer.state[weight]["frame"].clone()
    _run([weight], optimizer, 1)
    after = optimizer.state[weight]["frame"]
    assert float((after - first @ (first.mT @ after)).norm() / math.sqrt(RANK)) > 1e-3


def test_every_live_plane_turns_by_exactly_eta():
    """The turn is `eta`, not `eta` times a meter: each live plane of the polar
    tangent has singular value one, so the chordal residual per plane is `sin(eta)`."""

    torch.manual_seed(0)
    weight = torch.nn.Parameter(torch.randn(ROWS, COLS))
    optimizer = UsuiTrack([weight], lr=0.01, rank_fraction=RANK_FRACTION, diagnostics=True)
    _run([weight], optimizer, 12)

    before = optimizer.state[weight]["frame"].clone()
    _run([weight], optimizer, 1)
    after = optimizer.state[weight]["frame"]

    # The equality only means "every plane turned by eta" if every plane was live.
    assert optimizer.pop_diagnostics()["tangent_live_fraction"] == 1.0
    moved = float((after - before @ (before.mT @ after)).norm() / math.sqrt(RANK))
    assert moved == pytest.approx(math.sin(GEODESIC_STEPSIZE), rel=2e-3), moved


def test_diagnostics_are_inert_until_enabled():
    torch.manual_seed(0)
    params, optimizer = _two_matrix_optimizer()
    _run(params, optimizer, 3)
    assert optimizer.diagnostics is None
    assert optimizer.pop_diagnostics() == {}


def test_every_read_is_published_sane_and_describes_itself():
    torch.manual_seed(0)
    params, optimizer = _two_matrix_optimizer(diagnostics=True)
    # Three steps: the first fits the frame, so frame motion only exists after it.
    _run(params, optimizer, 3)

    diagnostics = optimizer.pop_diagnostics()
    assert set(diagnostics) == set(READS)
    assert all(isinstance(value, float) and value == value for value in diagnostics.values()), diagnostics
    assert -1.0 <= diagnostics["grad_moment_cosine"] <= 1.0
    # Capture is the share of the gradient the frame holds, so it cannot exceed one.
    assert 0.0 < diagnostics["grad_capture"] <= 1.0 + 1e-6
    assert diagnostics["tangent_live_fraction"] == 1.0
    assert diagnostics["nonfinite_grads"] == 0.0


def test_relative_step_reads_the_whole_step_against_the_whole_weight():
    """In fp32 the parameters move by exactly `lr U`, so the read is the joint
    Frobenius ratio of every matrix's change, not a mean over matrices."""

    torch.manual_seed(0)
    params, optimizer = _two_matrix_optimizer(diagnostics=True)
    optimizer.param_groups[0]["lr"] = 1e-1
    lone = torch.nn.Parameter(torch.randn(ROWS, COLS) * 3)
    optimizer.add_param_group({"params": [lone], "lr": 1e-1})
    all_params = [*params, lone]
    _run(all_params, optimizer, 2)
    optimizer.pop_diagnostics()

    before = [param.detach().clone() for param in all_params]
    _run(all_params, optimizer, 1)
    change = torch.stack([(param.detach() - old).norm() for param, old in zip(all_params, before, strict=True)])
    norm = torch.stack([param.detach().norm() for param in all_params])
    expected = change.square().sum().sqrt() / norm.square().sum().sqrt()
    assert optimizer.pop_diagnostics()["relative_step"] == pytest.approx(float(expected), rel=1e-4)


def test_pop_diagnostics_clears_the_window():
    torch.manual_seed(0)
    params, optimizer = _two_matrix_optimizer(diagnostics=True)
    _run(params, optimizer, 2)

    assert optimizer.pop_diagnostics()
    assert optimizer.pop_diagnostics() == {}
    _run(params, optimizer, 1)
    assert optimizer.pop_diagnostics()


def test_nonfinite_gradients_are_counted_and_sanitized():
    torch.manual_seed(0)
    params, optimizer = _two_matrix_optimizer(diagnostics=True)
    _run(params, optimizer, 2)
    optimizer.pop_diagnostics()

    for param in params:
        param.grad = torch.randn_like(param)
    params[0].grad[0, 0] = float("nan")
    optimizer.step()

    drained = optimizer.pop_diagnostics()
    assert drained["nonfinite_grads"] == 1.0
    # The bad gradient is flagged, not allowed to take the window's means to NaN.
    assert math.isfinite(drained["grad_capture"])
    assert torch.isfinite(params[0]).all()


def test_diagnostics_do_not_change_the_trajectory():
    torch.manual_seed(0)
    quiet_params, quiet = _two_matrix_optimizer()
    torch.manual_seed(0)
    loud_params, loud = _two_matrix_optimizer(diagnostics=True)

    torch.manual_seed(1)
    _run(quiet_params, quiet, 4)
    torch.manual_seed(1)
    _run(loud_params, loud, 4)
    for quiet_param, loud_param in zip(quiet_params, loud_params, strict=True):
        assert torch.equal(quiet_param, loud_param)


def test_agreement_reads_the_moment_before_this_step_not_after():
    """On a first step the prior moment is zero, so a cosine against the
    post-blend moment would report perfect agreement with no history."""
    torch.manual_seed(0)
    params, optimizer = _two_matrix_optimizer(diagnostics=True)
    _run(params, optimizer, 1)
    assert abs(optimizer.pop_diagnostics()["grad_moment_cosine"]) < 1e-6
