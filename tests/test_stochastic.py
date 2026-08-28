import torch

from usuitrack import UsuiTrack, copy_stochastic_
from usuitrack.stochastic import StochasticAdamW

ROWS, COLS, RANK = 256, 128, 16


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


def test_sub_ulp_updates_accumulate_instead_of_vanishing():
    """The regime that motivates this module: bf16 weights whose per-step
    update is a fraction of an ulp. UsuiTrack orthogonalizes the direction, so
    the step size is `lr` and not gradient magnitude -- which is exactly when
    round-to-nearest writes back the value it started with, every step.
    Stochastic rounding must instead track the fp32 trajectory.
    """
    torch.manual_seed(0)
    start = torch.full((ROWS, COLS), 0.01)
    gradient = torch.full((ROWS, COLS), 1.0)

    def run(dtype, stochastic):
        weight = torch.nn.Parameter(start.clone().to(dtype))
        origin = weight.detach().clone().float()
        optimizer = UsuiTrack(
            [weight], lr=1e-5, rank=RANK, side="right", stochastic_rounding=stochastic
        )
        for _ in range(400):
            weight.grad = gradient.clone().to(dtype)
            optimizer.step()
        assert torch.isfinite(weight).all()
        detached = weight.detach().float()
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
