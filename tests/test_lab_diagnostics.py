"""The lab reads in `experiments/lab_diagnostics.py`, on the release optimizer."""

import math

import pytest
import torch

from experiments.lab_diagnostics import ALL_READS, LabDiagnostics
from usuitrack import UsuiTrack

ROWS, COLS, RANK = 256, 128, 16
RANK_FRACTION = RANK / math.sqrt(ROWS * COLS)


def _optimizer(diagnostics, params=None):
    params = params or [torch.nn.Parameter(torch.randn(ROWS, COLS)) for _ in range(2)]
    optimizer = UsuiTrack(params, lr=1e-3, rank_fraction=RANK_FRACTION, beta=0.9)
    optimizer.diagnostics = diagnostics
    return params, optimizer


def _run(params, optimizer, steps, gradient=None):
    for _ in range(steps):
        for param in params:
            param.grad = gradient.clone() if gradient is not None else torch.randn_like(param)
        optimizer.step()


def test_every_lab_read_is_published_and_describes_itself():
    torch.manual_seed(0)
    params, optimizer = _optimizer(LabDiagnostics(snapshot_interval=2))
    _run(params, optimizer, 8)
    drained = optimizer.pop_diagnostics()
    assert set(drained) == set(ALL_READS)
    assert 0.0 < drained["transport_lag"] <= 1.0
    assert 0.0 <= drained["transport_curve"] < 1.0
    assert drained["transport_spin"] >= 0.0
    assert 1.0 / RANK <= drained["tangent_concentration"] <= 1.0
    # fp32 frames store the geodesic exactly, so there is no in-span turn to correct.
    assert drained["transport_correction"] < 1e-4


def test_moment_persistence_reads_zero_on_an_independent_stream_and_one_on_a_held_one():
    """The read subtracts the incomplete-cancellation floor, or it never reaches zero."""

    def persistence(repeat: bool) -> float:
        torch.manual_seed(0)
        params, optimizer = _optimizer(LabDiagnostics(), [torch.nn.Parameter(torch.randn(ROWS, COLS))])
        held = torch.randn(ROWS, COLS) if repeat else None
        _run(params, optimizer, 150, held)
        optimizer.pop_diagnostics()
        _run(params, optimizer, 50, held)
        return optimizer.pop_diagnostics()["moment_persistence"]

    assert abs(persistence(repeat=False)) < 0.05
    assert persistence(repeat=True) > 0.9


def test_role_reads_partition_the_fleet_only_when_asked_for():
    torch.manual_seed(0)
    params = [torch.nn.Parameter(torch.randn(ROWS, COLS)) for _ in range(3)]
    roles = {params[0]: "pair", params[1]: "pair", params[2]: "lone"}
    _params, optimizer = _optimizer(LabDiagnostics(roles=roles), params)
    _run(params, optimizer, 2)
    assert not [key for key in optimizer.pop_diagnostics() if "/" in key]

    optimizer.diagnostics = LabDiagnostics(roles=roles, role_reads=("grad_capture",))
    _run(params, optimizer, 1)
    drained = optimizer.pop_diagnostics()
    assert sorted(key for key in drained if "/" in key) == ["grad_capture/lone", "grad_capture/pair"]
    pair, lone = drained["grad_capture/pair"], drained["grad_capture/lone"]
    assert drained["grad_capture"] == pytest.approx((2 * pair + lone) / 3, rel=1e-6)


def test_frame_motion_is_sampled_across_param_groups():
    """The harness splits matrices into a group per side; the sample must not
    stop at the first group the step happens to turn."""

    torch.manual_seed(0)
    left = [torch.nn.Parameter(torch.randn(COLS, ROWS)) for _ in range(2)]
    right = [torch.nn.Parameter(torch.randn(ROWS, COLS)) for _ in range(2)]
    diagnostics = LabDiagnostics(sample=4)
    optimizer = UsuiTrack(
        [{"params": left, "side": "left"}, {"params": right, "side": "right"}],
        lr=1e-3,
        rank_fraction=RANK_FRACTION,
    )
    optimizer.diagnostics = diagnostics
    _run(left + right, optimizer, 3)
    assert diagnostics._sampled == set(left + right)


def test_role_reads_reject_an_unknown_read():
    with pytest.raises(ValueError, match="role_reads"):
        LabDiagnostics(role_reads=("grad_captur",))


def test_lag_is_the_sine_of_the_principal_angle_and_spin_sees_only_in_span_rotation():
    """Lag vanishes iff the subspace stopped; spin is blind to subspace motion
    and sees a pure in-span rotation, which renames every moment coordinate."""
    torch.manual_seed(0)
    frame, _ = torch.linalg.qr(torch.randn(64, 8))

    def reads(current, previous):
        root = current.shape[1] ** 0.5
        overlap = previous.mT @ current
        return float((current - previous @ overlap).norm() / root), float((overlap - overlap.mT).norm() / (2.0 * root))

    outside = torch.randn(64, 1)
    outside = outside - frame @ (frame.mT @ outside)
    outside = outside / outside.norm()
    for theta in (1e-4, 1e-2, 0.3):
        rotated = frame.clone()
        rotated[:, 0] = frame[:, 0] * math.cos(theta) + outside[:, 0] * math.sin(theta)
        # One plane of eight moved by theta, so the RMS sine is sin(theta)/sqrt(8).
        assert abs(reads(rotated, frame)[0] - math.sin(theta) / math.sqrt(8)) < 1e-5

    spin = torch.eye(8)
    spin[0, 0] = spin[1, 1] = math.cos(0.2)
    spin[0, 1], spin[1, 0] = -math.sin(0.2), math.sin(0.2)
    lag, in_span = reads(frame @ spin, frame)
    assert lag < 1e-6 and in_span > 0.05
