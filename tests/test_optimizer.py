import copy
import math

import pytest
import torch
from torch import Tensor

from usuitrack import UsuiTrack
from usuitrack import frame
from usuitrack.diagnostics import READS
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


def test_only_the_frame_its_moment_and_the_moment_are_kept():
    """UsuiTrack keeps a frame, the frame's moment and a projected first moment. Nothing else."""
    torch.manual_seed(0)
    params, optimizer = _two_matrix_optimizer()
    _run(params, optimizer, 4)

    for param in params:
        state = optimizer.state[param]
        assert set(state) == {"frame", "frame_moment", "frame_moment_span", "moment"}
        assert tuple(state["frame"].shape) == tuple(state["frame_moment"].shape) == (COLS, RANK)
        assert tuple(state["frame_moment_span"].shape) == (RANK, RANK)
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

    # Two steps: the first's write reads only the held frame, so only the second sees the frame's moment.
    for _ in range(2):
        gradient = torch.randn_like(first)
        first.grad = gradient.clone()
        second.grad = gradient.clone()
        first_optimizer.step()
        second_optimizer.step()
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def _frame_moment(state: dict) -> Tensor:
    """`Y`, as the optimizer holds it against its stored frame: `Q S + H`."""
    return state["frame"].float() @ state["frame_moment_span"] + state["frame_moment"].float()


def test_a_checkpoint_without_the_frame_moment_resumes_it_from_zero():
    """README and SPEC promise it: the frame's moment starts at zero, as on the step after a fit."""
    torch.manual_seed(1)
    weight = torch.nn.Parameter(torch.randn(ROWS, COLS))
    optimizer = UsuiTrack([weight], lr=0.01, rank_fraction=RANK_FRACTION)
    _run([weight], optimizer, 3)
    saved = copy.deepcopy(optimizer.state_dict())
    for state in saved["state"].values():
        del state["frame_moment"], state["frame_moment_span"]
    resumed = torch.nn.Parameter(weight.detach().clone())
    fresh = UsuiTrack([resumed], lr=0.01, rank_fraction=RANK_FRACTION)
    fresh.load_state_dict(saved)
    held = fresh.state[resumed]["frame"].clone()
    resumed.grad = torch.randn_like(resumed)
    _projected, aim = frame.project_and_aim(resumed.grad.clone(), held)
    fresh.step()
    torch.testing.assert_close(_frame_moment(fresh.state[resumed]), 0.05 * aim)


def test_a_bf16_matrix_holds_its_frame_moments_span_in_fp32_through_a_load():
    """Rounding `Y`'s large in-span part walks the frame (SPEC step 4), so it stays fp32,
    also after a load, where torch casts state to the param's dtype."""
    torch.manual_seed(1)
    weight = torch.nn.Parameter(torch.randn(ROWS, COLS, dtype=torch.bfloat16))
    optimizer = UsuiTrack([weight], lr=0.01, rank_fraction=RANK_FRACTION)
    _run([weight], optimizer, 2)
    assert optimizer.state[weight]["frame_moment_span"].dtype == torch.float32
    resumed = torch.nn.Parameter(weight.detach().clone())
    loaded = UsuiTrack([resumed], lr=0.01, rank_fraction=RANK_FRACTION)
    loaded.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    assert torch.equal(loaded.state[resumed]["frame_moment_span"], optimizer.state[weight]["frame_moment_span"])
    assert loaded.state[resumed]["frame_moment"].dtype == loaded.state[resumed]["frame"].dtype == torch.bfloat16


def test_the_frame_moment_averages_aims_at_the_held_frame():
    """`Y = 0.95 Y + 0.05 a`, `a` aimed at the frame held when the gradient arrived,
    and the frame moves to `polar(Q + Y)`."""
    torch.manual_seed(1)
    weight = torch.nn.Parameter(torch.randn(ROWS, COLS))
    optimizer = UsuiTrack([weight], lr=0.01, rank_fraction=RANK_FRACTION)
    _run([weight], optimizer, 1)
    expected = None
    for _ in range(3):
        held = optimizer.state[weight]["frame"].clone()
        weight.grad = torch.randn_like(weight)
        _projected, aim = frame.project_and_aim(weight.grad.clone(), held)
        expected = 0.05 * aim if expected is None else 0.95 * expected + 0.05 * aim
        optimizer.step()
        torch.testing.assert_close(_frame_moment(optimizer.state[weight]), expected)
        torch.testing.assert_close(optimizer.state[weight]["frame"], frame.follow(held, expected).frame)


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

    Each projected gradient enters the moment at unit norm, and the frame's aim
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
    """One optimizer over several shapes and ranks steps each matrix as its own
    optimizer would: nothing in a step couples matrices."""

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


@pytest.mark.parametrize("block_bytes", [None, 1])
def test_a_block_steps_its_matrices_as_each_would_step_alone(monkeypatch, block_bytes):
    """Same-shaped matrices step batched, as one block or, under a cap of one byte, one each;
    a step that reaches only some members leaves the others' weights and state as they were."""

    import usuitrack.optimizer

    if block_bytes is not None:
        monkeypatch.setattr(usuitrack.optimizer, "BLOCK_BYTES", block_bytes)
    torch.manual_seed(6)
    shapes = [(64, 512), (96, 96), (64, 512), (64, 512)]
    base = [torch.randn(*shape) for shape in shapes]
    # The middle (64, 512) gets no gradient on step 3, so its block steps a subset there.
    gradients = [[None if (step, k) == (3, 2) else torch.randn(*shape) for k, shape in enumerate(shapes)] for step in range(6)]

    together = [torch.nn.Parameter(weight.clone()) for weight in base]
    optimizer = UsuiTrack(together, lr=0.01, rank_fraction=RANK_FRACTION, side="right")
    separate = [torch.nn.Parameter(weight.clone()) for weight in base]
    solo = [UsuiTrack([weight], lr=0.01, rank_fraction=RANK_FRACTION, side="right") for weight in separate]
    for step in range(6):
        for k in range(len(shapes)):
            gradient = gradients[step][k]
            together[k].grad = None if gradient is None else gradient.clone()
            if gradient is not None:
                separate[k].grad = gradient.clone()
                solo[k].step()
        optimizer.step()

    for k in range(len(shapes)):
        torch.testing.assert_close(together[k], separate[k], rtol=1e-5, atol=1e-6)
        for name in ("frame", "moment", "frame_moment", "frame_moment_span"):
            torch.testing.assert_close(optimizer.state[together[k]][name], solo[k].state[separate[k]][name], rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the Triton write runs on CUDA")
@pytest.mark.parametrize("side,bounds,fixed", [("right", (None, None), False), ("left", (None, 0.99), False), ("right", (1.0, 1.0), True)])
def test_the_cuda_write_lands_where_the_eager_write_lands(side, bounds, fixed):
    """Against the eager maths in fp32: fp32 weights take the same step up to the lift's
    TF32 error, and bf16 weights take it in expectation, within an ulp of it."""
    from usuitrack.optimizer import Block, _write_eager
    from usuitrack.write import write_block

    torch.manual_seed(0)
    count, rows, cols, rank = 3, 200, 136, 40
    length, other = (cols, rows) if side == "right" else (rows, cols)
    start = torch.randn(count, rows, cols, device="cuda") * 0.02
    frames = torch.linalg.qr(torch.randn(count, length, rank, device="cuda"))[0].bfloat16()
    direction = unit_polar(torch.randn(count, other, rank, device="cuda"))
    reference = torch.linalg.vector_norm(start, dim=(-2, -1))
    lr, decay = 2e-2, 1.0 - 2e-2 * 0.1
    fixed_scale = lr * math.sqrt(rows / cols * rank) if fixed else None
    for dtype in (torch.float32, torch.bfloat16):
        params = [weight.to(dtype) for weight in start]
        before = torch.stack(params).float()
        expected = list(before.clone().unbind(0))
        block = Block(expected, side, rank)
        _write_eager(block, expected, direction, frames, decay, lr, fixed_scale, bounds, reference)
        expected = torch.stack(expected)
        step = expected - before
        write_block(params, block.pointer_table(params), side, direction, frames, decay, lr, fixed_scale, bounds, reference)
        written = torch.stack(params).float()
        if dtype is torch.float32:
            assert torch.linalg.vector_norm(written - expected) < 1e-3 * torch.linalg.vector_norm(step)
        else:
            assert abs(((written - before) * step).sum() / step.square().sum() - 1.0) < 2e-2
            ulp = expected.abs().bfloat16().float() * 2.0**-7
            assert ((written - expected).abs() <= ulp * 1.01 + 1e-3 * step.abs().max()).all()
        if bounds == (1.0, 1.0):
            torch.testing.assert_close(torch.linalg.vector_norm(written, dim=(-2, -1)), reference, rtol=1e-2 if dtype is torch.bfloat16 else 1e-5, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the Triton write runs on CUDA")
def test_the_cuda_write_lands_a_sub_ulp_step_in_expectation():
    """A bf16 step of about a hundredth of an ulp moves each entry by a whole ulp with
    that probability, so the step survives on average rather than rounding away."""
    from usuitrack.optimizer import Block, _write_eager
    from usuitrack.write import write_block

    torch.manual_seed(0)
    count, rows, cols, rank = 4, 512, 384, 32
    start = (torch.randn(count, rows, cols, device="cuda") * 0.02).bfloat16()
    frames = torch.linalg.qr(torch.randn(count, cols, rank, device="cuda"))[0].bfloat16()
    direction = unit_polar(torch.randn(count, rows, rank, device="cuda"))
    before = start.float()
    expected = list(before.clone().unbind(0))
    _write_eager(Block(expected, "right", rank), expected, direction, frames, 1.0, 1e-5, None, (None, None), None)
    step = torch.stack(expected) - before
    params = [weight.clone() for weight in start]
    write_block(params, Block(params, "right", rank).pointer_table(params), "right", direction, frames, 1.0, 1e-5, None, (None, None), None)
    moved = torch.stack(params).float() - before
    # bf16's spacing at each entry, from its fp32 bits (bf16 is their top half).
    magnitude = before.abs()
    ulp = (magnitude.view(torch.int32) + (1 << 16)).view(torch.float32) - magnitude
    # Entries near zero take steps of many ulps, and at a power of two the spacing below is half.
    sub_ulp = (step.abs() < 0.5 * ulp) & (magnitude.view(torch.int32) & 0x7F0000 != 0)
    assert sub_ulp.float().mean() > 0.9
    expected_moves = (step.abs() / ulp)[sub_ulp].sum()
    assert abs((moved != 0)[sub_ulp].sum() / expected_moves - 1.0) < 0.05
    assert abs((moved * step).sum() / step.square().sum() - 1.0) < 0.1


def test_a_block_resumes_from_a_checkpoint_as_if_never_stopped():
    torch.manual_seed(2)
    first = [torch.nn.Parameter(torch.randn(ROWS, COLS)) for _ in range(3)]
    first_optimizer = UsuiTrack(first, lr=0.01, rank_fraction=RANK_FRACTION)
    _run(first, first_optimizer, 2)

    second = [torch.nn.Parameter(weight.detach().clone()) for weight in first]
    second_optimizer = UsuiTrack(second, lr=0.01, rank_fraction=RANK_FRACTION)
    second_optimizer.load_state_dict(copy.deepcopy(first_optimizer.state_dict()))
    for _ in range(2):
        for one, other in zip(first, second, strict=True):
            other.grad = torch.randn_like(one)
            one.grad = other.grad.clone()
        first_optimizer.step()
        second_optimizer.step()
    for one, other in zip(first, second, strict=True):
        torch.testing.assert_close(one, other, rtol=0, atol=0)


def test_the_frame_moves_from_the_very_first_update_after_its_fit():
    torch.manual_seed(0)
    weight = torch.nn.Parameter(torch.randn(ROWS, COLS))
    optimizer = UsuiTrack([weight], lr=0.01, rank_fraction=RANK_FRACTION)
    _run([weight], optimizer, 1)
    first = optimizer.state[weight]["frame"].clone()
    _run([weight], optimizer, 1)
    after = optimizer.state[weight]["frame"]
    assert float((after - first @ (first.mT @ after)).norm() / math.sqrt(RANK)) > 1e-3


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
    # The `G` path holds the exact norm; only Molegrad publishes the probe ratio.
    assert set(diagnostics) == set(READS) - {"grad_capture_probe"}
    assert all(isinstance(value, float) and value == value for value in diagnostics.values()), diagnostics
    assert -1.0 <= diagnostics["grad_moment_cosine"] <= 1.0
    # Capture is the share of the gradient the frame holds, so it cannot exceed one.
    assert 0.0 < diagnostics["grad_capture"] <= 1.0 + 1e-6
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


def test_a_zero_matrix_never_moves():
    """SPEC step 8: the step is sized against `||W||`, which is zero."""

    weight = torch.nn.Parameter(torch.zeros(ROWS, COLS))
    optimizer = UsuiTrack([weight], lr=0.1, rank_fraction=RANK_FRACTION)
    for _ in range(3):
        weight.grad = torch.randn(ROWS, COLS)
        optimizer.step()
    assert torch.equal(weight, torch.zeros(ROWS, COLS))


@pytest.mark.parametrize("release", [False, True])
def test_a_zero_or_non_finite_first_gradient_fits_an_orthonormal_frame_and_moves_nothing(release):
    """SPEC step 2: the fit takes the first gradient as it is, sanitized; a zero
    one fits an arbitrary orthonormal frame, and its zero moment writes nothing,
    decay aside."""

    torch.manual_seed(0)
    base = torch.randn(ROWS, COLS)
    for empty in (torch.zeros(ROWS, COLS), torch.full((ROWS, COLS), float("nan"))):
        weight = torch.nn.Parameter(base.clone())
        optimizer = UsuiTrack([weight], lr=0.01, rank_fraction=RANK_FRACTION, release_matrix_grads=release)
        if release:
            _backward(weight, empty)
        else:
            weight.grad = empty.clone()
        optimizer.step()
        basis = optimizer.state[weight]["frame"]
        torch.testing.assert_close(basis.mT @ basis, torch.eye(basis.shape[-1]), atol=1e-5, rtol=0)
        assert torch.equal(weight, base)


def test_random_frames_are_drawn_at_build_orthonormal_and_reproducible():
    """`frame_init="random"`: every matrix holds a frame before its first
    gradient, drawn from a private generator, so the same build draws the same
    frames and training's random stream never shifts (SPEC step 2)."""

    def build():
        weights = [torch.nn.Parameter(torch.zeros(ROWS, COLS)), torch.nn.Parameter(torch.zeros(COLS, ROWS))]
        return weights, UsuiTrack([{"params": weights, "frame_init": "random"}], rank_fraction=RANK_FRACTION)

    torch.manual_seed(0)
    untouched = torch.rand(4)
    torch.manual_seed(0)
    weights, optimizer = build()
    assert torch.equal(torch.rand(4), untouched)
    frames = [optimizer.state[weight]["frame"] for weight in weights]
    for weight, basis in zip(weights, frames, strict=True):
        side = optimizer.param_groups[0]["side"]
        assert tuple(basis.shape) == (frame.orient(weight, side).shape[1], frame.rank(weight.shape, RANK_FRACTION))
        torch.testing.assert_close(basis.mT @ basis, torch.eye(basis.shape[-1]), atol=1e-5, rtol=0)
    again_weights, again = build()
    assert all(torch.equal(basis, again.state[weight]["frame"]) for weight, basis in zip(again_weights, frames, strict=True))
    with pytest.raises(ValueError, match="frame_init"):
        UsuiTrack([{"params": [torch.nn.Parameter(torch.zeros(ROWS, COLS))], "frame_init": "eigh"}])


def test_a_load_without_state_keeps_a_random_frame():
    """A param the checkpoint holds no state for still starts from a drawn
    frame after the load, not from a fit to its first gradient."""

    weight = torch.nn.Parameter(torch.zeros(ROWS, COLS))
    optimizer = UsuiTrack([{"params": [weight], "frame_init": "random"}], rank_fraction=RANK_FRACTION)
    optimizer.load_state_dict({"state": {}, "param_groups": optimizer.state_dict()["param_groups"]})
    basis = optimizer.state[weight]["frame"]
    torch.testing.assert_close(basis.mT @ basis, torch.eye(basis.shape[-1]), atol=1e-5, rtol=0)


def _fixed_size(shape):
    """SPEC step 8, fixed rule: the norm of a step whose moment fills its rank."""

    rows, cols = shape
    return math.sqrt(max(1.0, rows / cols) * frame.rank(torch.Size(shape), RANK_FRACTION))


@pytest.mark.parametrize("shape", [(ROWS, COLS), (COLS, ROWS), (512, 32), (32, 512)])
@pytest.mark.parametrize("scale", [1.0, 30.0])
def test_a_fixed_step_is_sized_by_the_shape_and_not_by_the_weight(shape, scale):
    torch.manual_seed(0)
    lr = 1e-2
    weight = torch.nn.Parameter(scale * torch.randn(*shape))
    before = weight.detach().clone()
    optimizer = UsuiTrack([{"params": [weight], "step_rule": "fixed"}], lr=lr, rank_fraction=RANK_FRACTION)
    weight.grad = torch.randn(*shape)
    optimizer.step()
    assert (before - weight.detach()).norm().item() == pytest.approx(lr * _fixed_size(shape), rel=0.1)


def test_a_zero_matrix_moves_under_the_fixed_rule_from_its_first_nonzero_gradient():
    torch.manual_seed(0)
    lr = 1e-2
    weight = torch.nn.Parameter(torch.zeros(ROWS, COLS))
    optimizer = UsuiTrack([{"params": [weight], "step_rule": "fixed", "frame_init": "random"}], lr=lr, rank_fraction=RANK_FRACTION)
    weight.grad = torch.zeros(ROWS, COLS)
    optimizer.step()
    assert torch.equal(weight, torch.zeros(ROWS, COLS))
    weight.grad = torch.randn(ROWS, COLS)
    optimizer.step()
    assert weight.detach().norm().item() == pytest.approx(lr * _fixed_size((ROWS, COLS)), rel=0.1)


def test_the_trust_rule_is_the_default_and_the_fixed_rule_changes_only_the_size():
    torch.manual_seed(0)
    base = torch.randn(ROWS, COLS)
    gradients = [torch.randn(ROWS, COLS) for _ in range(3)]
    ran = {}
    for rule in (None, "trust", "fixed"):
        weight = torch.nn.Parameter(base.clone())
        group = {"params": [weight], **({} if rule is None else {"step_rule": rule})}
        optimizer = UsuiTrack([group], lr=0.01, rank_fraction=RANK_FRACTION)
        for gradient in gradients:
            weight.grad = gradient.clone()
            optimizer.step()
        ran[rule] = (weight.detach(), optimizer.state[weight]["frame"].clone())
    assert torch.equal(ran[None][0], ran["trust"][0])
    assert not torch.equal(ran[None][0], ran["fixed"][0])
    # The direction is the same, so the frames (which read only gradients) agree.
    # They need not be bitwise equal: the weights differ, but nothing reads them.
    assert torch.equal(ran[None][1], ran["fixed"][1])


def test_groups_with_different_rules_step_by_their_own_in_one_optimizer():
    torch.manual_seed(0)
    lr = 1e-2
    trusted = torch.nn.Parameter(30 * torch.randn(ROWS, COLS))
    fixed = torch.nn.Parameter(30 * torch.randn(ROWS, COLS))
    optimizer = UsuiTrack([{"params": [trusted]}, {"params": [fixed], "step_rule": "fixed"}], lr=lr, rank_fraction=RANK_FRACTION)
    before = [trusted.detach().clone(), fixed.detach().clone()]
    for param in (trusted, fixed):
        param.grad = torch.randn(ROWS, COLS)
    optimizer.step()
    trusted_step = (before[0] - trusted.detach()).norm().item()
    fixed_step = (before[1] - fixed.detach()).norm().item()
    assert trusted_step == pytest.approx(lr * before[0].norm().item(), rel=0.1)
    assert fixed_step == pytest.approx(lr * _fixed_size((ROWS, COLS)), rel=0.1)


def test_an_unknown_step_rule_is_refused():
    with pytest.raises(ValueError, match="step_rule"):
        UsuiTrack([{"params": [torch.nn.Parameter(torch.randn(ROWS, COLS))], "step_rule": "relative"}])


def test_the_step_rule_survives_a_checkpoint_round_trip():
    """A saved rule is the loaded one; a checkpoint from before the key keeps the configured one."""

    saved_from = UsuiTrack([{"params": [torch.nn.Parameter(torch.randn(ROWS, COLS))], "step_rule": "fixed"}], rank_fraction=RANK_FRACTION)
    restored = UsuiTrack([{"params": [torch.nn.Parameter(torch.randn(ROWS, COLS))]}], rank_fraction=RANK_FRACTION)
    restored.load_state_dict(saved_from.state_dict())
    assert restored.param_groups[0]["step_rule"] == "fixed"

    older = saved_from.state_dict()
    for group in older["param_groups"]:
        del group["step_rule"]
    configured = UsuiTrack([{"params": [torch.nn.Parameter(torch.randn(ROWS, COLS))], "step_rule": "fixed"}], rank_fraction=RANK_FRACTION)
    configured.load_state_dict(older)
    assert configured.param_groups[0]["step_rule"] == "fixed"


def test_relative_step_stays_the_change_over_the_weight_under_the_fixed_rule():
    torch.manual_seed(0)
    weight = torch.nn.Parameter(torch.randn(ROWS, COLS))
    optimizer = UsuiTrack([{"params": [weight], "step_rule": "fixed"}], lr=0.01, rank_fraction=RANK_FRACTION, diagnostics=True)
    before = weight.detach().clone()
    weight.grad = torch.randn(ROWS, COLS)
    optimizer.step()
    expected = (weight.detach() - before).norm() / weight.detach().norm()
    assert optimizer.pop_diagnostics()["relative_step"] == pytest.approx(float(expected), rel=1e-4)


def test_bounds_of_one_pin_the_norm_and_the_step_stays_relative():
    torch.manual_seed(0)
    lr = 1e-2
    weight = torch.nn.Parameter(3.0 * torch.randn(ROWS, COLS))
    start = weight.detach().norm().item()
    optimizer = UsuiTrack([{"params": [weight], "norm_bounds": (1.0, 1.0)}], lr=lr, rank_fraction=RANK_FRACTION)
    for _ in range(20):
        before = weight.detach().clone()
        weight.grad = torch.randn(ROWS, COLS)
        optimizer.step()
        assert weight.detach().norm().item() == pytest.approx(start, rel=1e-5)
        assert (weight.detach() - before).norm().item() / start == pytest.approx(lr, rel=0.15)


def test_a_ceiling_caps_growth_and_a_floor_holds_against_decay():
    torch.manual_seed(0)
    gradients = [torch.randn(ROWS, COLS) for _ in range(30)]
    for bounds, decay in (((None, 1.1), 0.0), ((0.9, None), 5.0)):
        weight = torch.nn.Parameter(torch.randn(ROWS, COLS))
        start = weight.detach().norm().item()
        optimizer = UsuiTrack([{"params": [weight], "norm_bounds": bounds}], lr=0.05, weight_decay=decay, rank_fraction=RANK_FRACTION)
        for gradient in gradients:
            weight.grad = gradient.clone()
            optimizer.step()
        low, high = bounds
        ratio = weight.detach().norm().item() / start
        if high is not None:
            assert ratio == pytest.approx(high, rel=1e-5)  # unbounded, these steps grow it past 1.1
        if low is not None:
            assert ratio == pytest.approx(low, rel=1e-5)  # unbounded, this decay shrinks it below 0.9


def test_bounds_that_do_not_bind_leave_the_write_unchanged():
    torch.manual_seed(0)
    base = torch.randn(ROWS, COLS)
    gradients = [torch.randn(ROWS, COLS) for _ in range(3)]
    ran = []
    for group in ({}, {"norm_bounds": (None, None)}, {"norm_bounds": (0.01, 100.0)}):
        weight = torch.nn.Parameter(base.clone())
        optimizer = UsuiTrack([{"params": [weight], **group}], lr=0.01, rank_fraction=RANK_FRACTION)
        for gradient in gradients:
            weight.grad = gradient.clone()
            optimizer.step()
        ran.append(weight.detach())
    assert all(torch.equal(ran[0], other) for other in ran[1:])


def test_the_first_norm_of_a_bf16_matrix_survives_a_checkpoint_unrounded():
    torch.manual_seed(0)
    weight = torch.nn.Parameter((2.0 * torch.randn(ROWS, COLS)).bfloat16())
    optimizer = UsuiTrack([{"params": [weight], "norm_bounds": (1.0, 1.0)}], rank_fraction=RANK_FRACTION)
    weight.grad = torch.randn(ROWS, COLS).bfloat16()
    optimizer.step()
    restored_weight = torch.nn.Parameter(weight.detach().clone() * 5.0)  # a drifted copy; the pin must pull it back to the saved norm
    restored = UsuiTrack([{"params": [restored_weight], "norm_bounds": (1.0, 1.0)}], rank_fraction=RANK_FRACTION)
    restored.load_state_dict(optimizer.state_dict())
    assert restored.state[restored_weight]["init_norm"] == optimizer.state[weight]["init_norm"]
    restored_weight.grad = torch.randn(ROWS, COLS).bfloat16()
    restored.step()
    assert restored_weight.detach().float().norm().item() == pytest.approx(optimizer.state[weight]["init_norm"].item(), rel=1e-2)


def test_bounds_refuse_a_zero_matrix_and_malformed_settings():
    weight = torch.nn.Parameter(torch.zeros(ROWS, COLS))
    optimizer = UsuiTrack([{"params": [weight], "step_rule": "fixed", "norm_bounds": (None, 1.0)}], rank_fraction=RANK_FRACTION)
    weight.grad = torch.randn(ROWS, COLS)
    for _ in range(2):  # a caught refusal leaves nothing behind that would zero it on the next step
        with pytest.raises(ValueError, match="starts at zero"):
            optimizer.step()
        weight.grad = torch.randn(ROWS, COLS)
    for bounds in ((2.0, 1.0), (0.0, 1.0), (1.0,), "pin", (None, -1.0), (True, None)):
        with pytest.raises(ValueError, match="norm_bounds"):
            UsuiTrack([{"params": [torch.nn.Parameter(torch.randn(ROWS, COLS))], "norm_bounds": bounds}])
