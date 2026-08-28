"""The Oja tangent is written twice: once readably on SubspaceProjector, once
inlined into the fused per-side kernels in optimizer.py so torch.compile sees
flat tensor ops. Nothing structural keeps the two in agreement, and they have
already drifted -- two numerical guards were once added to the projector method
alone, which is the copy a default-configured run never executes.

These tests pin both halves of that: that the fused kernels are what actually
runs, and that they compute the same thing the readable copy does.
"""

import math

import pytest
import torch

from usuitrack import ProjectionSide, SubspaceProjector, UsuiTrack
from usuitrack.projector import MAX_BASIS_ROTATION

ROWS, COLS, RANK = 256, 128, 16


def _fused_kernel(side: ProjectionSide):
    return (
        UsuiTrack._prepare_tracker_adafactor_right_tensors
        if side is ProjectionSide.RIGHT
        else UsuiTrack._prepare_tracker_adafactor_left_tensors
    )


def _starved_gradient(projector: SubspaceProjector, side: ProjectionSide) -> torch.Tensor:
    """A gradient carrying almost no energy in the fitted frame's directions.

    This is the regime that makes the tangent explode: `rayleigh` measures the
    gradient's energy inside the frame, and the tangent divides by its mean
    diagonal. Drive that toward zero and the tangent grows without bound --
    which is exactly where a magnitude guard present in one copy of the Oja
    math and absent in the other shows up as a difference.
    """
    basis = projector.basis
    assert basis is not None
    raw = torch.randn(ROWS, COLS)
    if side is ProjectionSide.RIGHT:
        return raw - (raw @ basis.mT) @ basis
    return raw - basis @ (basis.mT @ raw)


@pytest.mark.parametrize("side", [ProjectionSide.RIGHT, ProjectionSide.LEFT])
@pytest.mark.parametrize("regime", ["ordinary", "starved"])
def test_fused_kernel_tangent_matches_the_projector_method(side, regime):
    torch.manual_seed(0)
    projector = SubspaceProjector(rank=RANK, side=side)
    projector.fit(torch.randn(ROWS, COLS))
    basis = projector.basis
    assert basis is not None
    gradient = torch.randn(ROWS, COLS) if regime == "ordinary" else _starved_gradient(projector, side)

    projected_shape = (ROWS, RANK) if side is ProjectionSide.RIGHT else (RANK, COLS)
    conditioned_grad, fused_tangent, _ = _fused_kernel(side)(
        gradient,
        basis,
        torch.rand(ROWS).mul_(1e-3),
        torch.rand(COLS).mul_(1e-3),
        torch.zeros(projected_shape),
        3,  # adafactor_step
        1.0,  # grad_clip_norm
        0.99,  # adafactor_beta2
        1e-30,  # adafactor_eps
        0.95,  # beta
    )

    # The kernel folds clipping, adafactor, the tangent, and the moment update
    # into one pass; feeding its own conditioned gradient back through the
    # readable copy isolates the tangent math, which is the part duplicated.
    reference_tangent = projector.oja_tangent(
        conditioned_grad, projected=projector.project(conditioned_grad)
    )
    action_norm = conditioned_grad.float().norm() * projector.project(conditioned_grad).float().norm()
    blowup = (fused_tangent.norm() / action_norm.clamp_min(1e-12)).item()
    if regime == "starved":
        # Confirms the case is actually pathological, so this stays a
        # regression guard rather than a second copy of the ordinary case.
        assert blowup > 1e3, blowup
    torch.testing.assert_close(
        fused_tangent, reference_tangent, rtol=1e-5, atol=1e-6 * max(1.0, blowup)
    )


def test_the_fused_kernel_is_the_path_a_default_run_takes():
    """A guard against fixing the copy that does not run. Under the default
    grad_clip_norm the projector method is never called after the basis is
    initialized -- any change to the Oja math has to land in the kernels."""

    calls = {"projector": 0}
    original = SubspaceProjector.oja_tangent

    def counting(self, matrix, projected=None):
        calls["projector"] += 1
        return original(self, matrix, projected=projected)

    SubspaceProjector.oja_tangent = counting
    try:
        weight = torch.nn.Parameter(torch.randn(ROWS, COLS))
        optimizer = UsuiTrack([weight], lr=1e-3, rank=RANK, side="right")
        for _ in range(4):
            weight.grad = torch.randn_like(weight)
            optimizer.step()
    finally:
        SubspaceProjector.oja_tangent = original

    assert calls["projector"] == 0


def test_geodesic_clamps_rotation_and_stays_on_the_stiefel_manifold():
    """A pathological tangent must not turn the frame further than
    MAX_BASIS_ROTATION, and clamping the angle must not cost orthonormality."""

    torch.manual_seed(0)
    frame = torch.linalg.qr(torch.randn(COLS, RANK))[0]
    direction = torch.randn(COLS, RANK)
    # Horizontal: strip the component lying along the frame, the way
    # oja_tangent's `action - frame @ rayleigh` residual does.
    direction = direction - frame @ (frame.mT @ direction)
    tangent = direction * 1e4  # a tangent that would otherwise spin the frame wildly

    gram = tangent.mT @ tangent
    values, vectors = torch.linalg.eigh(0.5 * (gram + gram.mT))
    moved = SubspaceProjector.oja_geodesic_from_eigh(frame, tangent, values, vectors, 1.0)

    torch.testing.assert_close(moved.mT @ moved, torch.eye(RANK), atol=3e-3, rtol=0)

    # Principal angles between the two subspaces are acos of the singular
    # values of frame.mT @ moved; none may exceed the clamp.
    cosines = torch.linalg.svdvals(frame.mT @ moved).clamp(-1.0, 1.0)
    largest_angle = torch.arccos(cosines.min()).item()
    assert largest_angle <= MAX_BASIS_ROTATION + 1e-2, largest_angle
    # And it really is pressed against the ceiling, not accidentally small.
    assert largest_angle > 0.5 * MAX_BASIS_ROTATION, largest_angle
    assert MAX_BASIS_ROTATION == pytest.approx(math.pi / 4)
