import math

import pytest
import torch

from usuitrack import UsuiTrack, copy_stochastic_
from usuitrack.stochastic import StochasticAdamW

ROWS, COLS, RANK = 256, 160, 32
RANK_FRACTION = RANK / math.sqrt(ROWS * COLS)


def test_copy_stochastic_is_unbiased_for_sub_ulp_values():
    """A value a fraction f of the way between two bf16 neighbours must land on
    the upper one with probability f, so the mean survives the rounding."""
    torch.manual_seed(0)
    low = torch.tensor(0.01, dtype=torch.bfloat16)
    ulp = torch.nextafter(low, torch.tensor(1.0, dtype=torch.bfloat16)) - low
    fraction = 0.25
    source = low.float() + fraction * ulp.float()

    target = torch.empty(200_000, dtype=torch.bfloat16)
    copy_stochastic_(target, source.expand_as(target).contiguous())

    hits = (target > low).float().mean().item()
    assert abs(hits - fraction) < 0.01
    assert torch.isclose(target.float().mean(), source, rtol=0, atol=float(ulp) * 0.02)


def test_round_into_fp16_keeps_fp16_precision():
    """Stochastic rounding is bf16's bit layout; an fp16 target is rounded to
    its own grid, not to bf16's eight-times coarser one."""
    from usuitrack.stochastic import round_into

    source = torch.full((1000,), 1.0 + 3e-4)
    target = torch.empty(1000, dtype=torch.float16)
    round_into(target, source)
    assert torch.equal(target, source.half())


def test_sub_ulp_updates_accumulate_instead_of_vanishing(monkeypatch):
    """The regime that motivates this module: bf16 weights whose per-step
    update is a fraction of an ulp. UsuiTrack orthogonalizes the direction, so
    the step size is `lr` and not gradient magnitude -- which is exactly when
    round-to-nearest writes back the value it started with, every step.
    Stochastic rounding must instead track the fp32 trajectory. The nearest
    arm replaces the optimizer's rounded write with a plain copy.
    """
    import usuitrack.optimizer
    torch.manual_seed(0)
    start = torch.full((ROWS, COLS), 0.01)
    gradient = torch.full((ROWS, COLS), 1.0)

    def run(dtype, stochastic):
        weight = torch.nn.Parameter(start.clone().to(dtype))
        origin = weight.detach().clone().float()
        if not stochastic:
            monkeypatch.setattr(usuitrack.optimizer, "round_into", lambda target, source: target.copy_(source))
        optimizer = UsuiTrack([weight], lr=1e-5, rank_fraction=RANK_FRACTION, side="right")
        for _ in range(400):
            weight.grad = gradient.clone().to(dtype)
            optimizer.step()
        assert torch.isfinite(weight).all()
        detached = weight.detach().float()
        monkeypatch.undo()
        return detached - origin, (detached != origin).float().mean().item()

    reference_drift, _ = run(torch.float32, False)
    nearest_drift, nearest_moved = run(torch.bfloat16, False)
    stochastic_drift, stochastic_moved = run(torch.bfloat16, True)

    # Nearest rounding freezes the tensor; stochastic rounding does not.
    assert nearest_moved < 0.05
    assert stochastic_moved > 10 * max(nearest_moved, 1e-3)

    # And the drift it does accumulate is the fp32 drift, not noise.
    assert reference_drift.mean() != 0.0
    ratio = stochastic_drift.mean() / reference_drift.mean()
    assert 0.75 < ratio < 1.25, ratio
    assert abs(nearest_drift.mean()) < 0.25 * abs(reference_drift.mean())


def test_stochastic_adamw_tracks_fp32_where_plain_bf16_adamw_stalls():
    """The fallback set (biases, norm gains, excluded matrices) is where bf16
    round-to-nearest bites hardest. At a step size well under an ulp, plain
    bf16 AdamW freezes while the stochastic one follows the fp32 trajectory."""
    torch.manual_seed(0)
    start = torch.full((ROWS, COLS), 0.02)
    gradient = torch.full((ROWS, COLS), 1.0)
    steps, lr = 300, 1e-7
    ulp = float(
        torch.tensor(0.02, dtype=torch.bfloat16).nextafter(torch.tensor(1.0, dtype=torch.bfloat16))
        - torch.tensor(0.02, dtype=torch.bfloat16)
    )
    # Adam normalises the update to about lr per element, so this is the regime
    # the whole module exists for.
    assert lr * steps < ulp

    def drift(parameter, optimizer):
        # Measure from each parameter's own starting value: casting 0.02 into
        # bf16 already moves it, and that offset is not something the optimizer
        # did.
        origin = parameter.detach().clone().float()
        for _ in range(steps):
            parameter.grad = gradient.clone().to(parameter.dtype)
            optimizer.step()
        return (parameter.detach().float() - origin).mean().item()

    fp32 = torch.nn.Parameter(start.clone())
    fp32_drift = drift(fp32, torch.optim.AdamW([fp32], lr=lr, weight_decay=0.0))

    nearest = torch.nn.Parameter(start.clone().bfloat16())
    nearest_drift = drift(nearest, torch.optim.AdamW([nearest], lr=lr, weight_decay=0.0))

    stochastic = torch.nn.Parameter(start.clone().bfloat16())
    stochastic_drift = drift(stochastic, StochasticAdamW([stochastic], lr=lr, weight_decay=0.0))

    assert torch.isfinite(stochastic).all()
    assert abs(nearest_drift) < 0.02 * abs(fp32_drift)
    assert abs(stochastic_drift - fp32_drift) < 0.15 * abs(fp32_drift)


def test_stochastic_adamw_leaves_fp32_parameters_alone():
    torch.manual_seed(0)
    weight = torch.nn.Parameter(torch.randn(16, 8))
    optimizer = StochasticAdamW([weight], lr=1e-3)
    weight.grad = torch.randn_like(weight)
    optimizer.step()
    assert optimizer.state[weight]["exp_avg"].dtype is torch.float32
    assert torch.isfinite(weight).all()


def _twin_optimizers(dtype: torch.dtype):
    """The same AdamW on CPU (the eager loop) and CUDA (the kernel): two groups apart in every
    scalar, sizes across a launch block, one tensor that skips a step."""
    torch.manual_seed(0)
    shapes = [(3,), (1000,), (33, 65), (4096,)]
    cpu = [torch.nn.Parameter(torch.randn(*shape).to(dtype)) for shape in shapes]
    cuda = [torch.nn.Parameter(p.detach().cuda()) for p in cpu]

    def build(params):
        return StochasticAdamW([{"params": params[:2]}, {"params": params[2:], "lr": 3e-3, "betas": (0.8, 0.95), "eps": 1e-6, "weight_decay": 0.1}],
                               lr=1e-2, betas=(0.9, 0.99))

    return cpu, cuda, build(cpu), build(cuda)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the Triton step runs on CUDA")
@pytest.mark.parametrize("dtype,tolerance", [(torch.float32, 1e-6), (torch.float16, 1e-3)])
def test_the_cuda_step_lands_where_the_eager_step_lands(dtype, tolerance):
    """fp16 weights round to nearest from the same fp32 result, so they agree to an fp16 ulp."""
    cpu, cuda, eager, kernel = _twin_optimizers(dtype)
    for step in range(5):
        for k, (a, b) in enumerate(zip(cpu, cuda)):
            gradient = torch.randn_like(a) if not (step == 2 and k == 1) else None
            a.grad, b.grad = gradient, None if gradient is None else gradient.cuda()
        eager.step()
        kernel.step()
    for a, b in zip(cpu, cuda):
        torch.testing.assert_close(b.cpu(), a, rtol=tolerance, atol=tolerance)
        for name in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(kernel.state[b][name].cpu(), eager.state[a][name], rtol=1e-5, atol=1e-7)
        assert kernel.state[b]["step"] == eager.state[a]["step"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the Triton step runs on CUDA")
def test_the_cuda_step_rounds_bf16_within_an_ulp_and_lands_a_sub_ulp_step_in_expectation():
    """Against the fp32 maths: every element within one bf16 ulp of it, and a step of a
    hundredth of an ulp landing on average, in the weight and in both moments."""
    weight = torch.nn.Parameter(torch.ones(1 << 20, device="cuda", dtype=torch.bfloat16))
    optimizer = StochasticAdamW([weight], lr=1e-4, betas=(0.9, 0.99))
    weight.grad = torch.full_like(weight, 0.25)
    optimizer.step()
    # One step from zero moments: m = 0.025, v = 6.25e-4, so the bias-corrected update is lr, 1/39 of an ulp at 1.
    expected = {"weight": 1.0 - 1e-4, "exp_avg": 0.025, "exp_avg_sq": 6.25e-4}
    values = {"weight": weight.detach(), "exp_avg": optimizer.state[weight]["exp_avg"], "exp_avg_sq": optimizer.state[weight]["exp_avg_sq"]}
    for name, value in values.items():
        target = expected[name]
        ulp = 2.0 ** (math.floor(math.log2(target)) - 7)  # bf16 keeps 8 significant bits
        assert float((value.float() - target).abs().max()) <= ulp
        assert abs(float(value.float().mean()) - target) < 0.02 * ulp


def test_a_copied_optimizer_steps():
    """deepcopy and pickle carry only torch's optimizer state; the CUDA step's tables are rebuilt."""
    import copy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight = torch.nn.Parameter(torch.randn(100, device=device, dtype=torch.bfloat16))
    optimizer = StochasticAdamW([weight], lr=1e-3)
    weight.grad = torch.randn_like(weight)
    optimizer.step()
    copied = copy.deepcopy(optimizer)
    twin = copied.param_groups[0]["params"][0]
    twin.grad = torch.randn_like(twin)
    copied.step()
    assert copied.state[twin]["step"] == 2
