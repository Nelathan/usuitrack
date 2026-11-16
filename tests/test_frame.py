import math

import pytest
import torch

from usuitrack import frame

# RANK_FRACTION is chosen so (ROWS, COLS) runs at exactly RANK, below the cap.
ROWS, COLS, RANK = 256, 128, 32
RANK_FRACTION = RANK / math.sqrt(ROWS * COLS)


def _horizontal_tangent(q: torch.Tensor) -> torch.Tensor:
    tangent = torch.randn_like(q)
    return tangent - q @ (q.mT @ tangent)


def test_fit_is_orthonormal_and_deterministic_on_zero():
    q = frame.fit(torch.randn(ROWS, COLS), RANK)
    assert tuple(q.shape) == (COLS, RANK)
    torch.testing.assert_close(q.mT @ q, torch.eye(RANK), atol=1e-5, rtol=0)
    zero = torch.zeros(ROWS, COLS)
    torch.testing.assert_close(frame.fit(zero, RANK), frame.fit(zero, RANK), rtol=0, atol=0)


def test_fitting_leaves_the_training_random_stream_alone():
    """The fit draws from its own generator, so a run's batches and dropout do
    not depend on when its matrices fit (SPEC step 2)."""
    gradient = torch.randn(ROWS, COLS)
    torch.manual_seed(0)
    untouched = torch.rand(4)
    torch.manual_seed(0)
    frame.fit(gradient, RANK)
    assert torch.equal(torch.rand(4), untouched)


def test_the_aim_is_blind_to_the_gradients_scale():
    """The action is quadratic in `G` and so is its Rayleigh quotient; the aim
    divides one by the other, so a gradient of tiny entries (a matrix behind a
    gate still near zero) aims like any other."""
    torch.manual_seed(0)
    q = frame.fit(torch.randn(ROWS, COLS), RANK)
    gradient = torch.randn(ROWS, COLS)
    _projected, aim = frame.project_and_aim(gradient, q)
    _projected, tiny = frame.project_and_aim(1e-8 * gradient, q)
    torch.testing.assert_close(tiny, aim, rtol=1e-4, atol=1e-5)
    assert float((q * aim).sum()) / RANK == pytest.approx(1.0, rel=1e-5)


def test_a_fitted_frame_is_a_fixed_point_of_its_own_gradient():
    """A frame fitted on `G` is the leading eigenspace of `G^T G` where the
    gradient determines one, so the aim lands inside it and the frame holds:
    the tracker stops when the gradient's principal subspace stops moving."""
    torch.manual_seed(0)
    gradient = torch.randn(ROWS, RANK) @ torch.randn(RANK, COLS)
    q = frame.fit(gradient, RANK)
    _projected, aim = frame.project_and_aim(gradient, q)
    moved = frame.follow(q, 0.05 * aim)
    assert float((moved.frame - q @ (q.mT @ moved.frame)).norm()) < 1e-4
    torch.testing.assert_close(moved.frame, q, atol=1e-3, rtol=0)


def test_a_gradient_of_lower_rank_than_the_frame_lies_wholly_inside_it():
    """A zero-initialized model hands its upstream matrices a first gradient of
    rank far below the frame's, nearly rank one (SPEC step 2); the fit holds
    all of it, and the planes it does not span are still orthonormal."""
    torch.manual_seed(0)
    gradient = 1e3 * torch.randn(ROWS, 1) @ torch.randn(1, COLS) + torch.randn(ROWS, 2) @ torch.randn(2, COLS)
    q = frame.fit(gradient, RANK)
    torch.testing.assert_close(q.mT @ q, torch.eye(RANK), atol=1e-5, rtol=0)
    assert float((gradient - gradient @ q @ q.mT).norm() / gradient.norm()) < 1e-4  # the fp32 Gram squares the tail's 1e-3


def test_the_followed_frame_is_orthonormal_however_far_the_moment_has_grown():
    """Newton-Schulz alone lands a few percent off orthonormal; the retraction finishes it."""
    torch.manual_seed(0)
    q = torch.linalg.qr(torch.randn(COLS, RANK))[0]
    for scale in (0.05, 1.0, 3.0):
        moved = frame.follow(q, scale * torch.randn(COLS, RANK))
        torch.testing.assert_close(moved.frame.mT @ moved.frame, torch.eye(RANK), atol=1e-3, rtol=0)


def test_planes_the_aim_does_not_leave_hold_still():
    """A moment with two directions outside the span moves two planes; the rest of the frame stays put."""
    torch.manual_seed(0)
    q = torch.linalg.qr(torch.randn(COLS, RANK))[0]
    low_rank = 0.1 * _horizontal_tangent(q)[:, :2] @ torch.randn(2, RANK)
    moved = frame.follow(q, low_rank).frame
    away = moved - q @ (q.mT @ moved)
    assert float(away.norm()) > 1e-2
    assert torch.linalg.matrix_rank(away, atol=1e-4) == 2


def test_a_horizontal_move_carries_no_rotation_until_bf16_storage_adds_one():
    """A horizontal moment moves the frame with a symmetric overlap, so there is
    no in-span rotation to carry the moment by in fp32; bf16 storage adds one.
    Taken at a per-step angle of ~0.02, the size the twin moves by: the
    rotation comes from one retraction of the overlap, exact only at small angles."""
    torch.manual_seed(0)
    q = torch.linalg.qr(torch.randn(COLS, RANK))[0]
    horizontal = 0.002 * _horizontal_tangent(q)
    torch.testing.assert_close(frame.follow(q, horizontal).rotation, torch.eye(RANK), atol=1e-4, rtol=0)

    rotation = frame.follow(q.bfloat16(), horizontal).rotation
    torch.testing.assert_close(rotation.mT @ rotation, torch.eye(RANK), atol=1e-3, rtol=0)
    assert float((rotation - rotation.mT).abs().max()) > 1e-4


def test_the_frame_settles_on_a_held_gradients_leading_eigenspace():
    """Followed under one gradient, from a random start, the frame converges to
    the top `r` eigenspace of its Gram."""
    torch.manual_seed(0)
    spectrum = torch.cat([torch.full((RANK,), 3.0), torch.linspace(1.0, 0.1, COLS - RANK)])
    basis = torch.linalg.qr(torch.randn(COLS, COLS))[0]
    gradient = torch.randn(ROWS, COLS) @ basis @ torch.diag(spectrum) @ basis.mT
    top = torch.linalg.eigh(gradient.mT @ gradient).eigenvectors[:, -RANK:]
    q = torch.linalg.qr(torch.randn(COLS, RANK))[0]
    frame_moment = torch.zeros_like(q)
    for _ in range(300):
        _projected, aim = frame.project_and_aim(gradient, q)
        frame_moment = 0.95 * frame_moment + 0.05 * aim
        q = frame.follow(q, frame_moment).frame
    assert float((top.mT @ q).norm().square()) / RANK > 0.99


def test_rank_is_a_fraction_of_the_size_to_the_nearest_32_capped_at_256_and_half_the_smaller_side():
    """Anima's patch embedding is the case the half cap exists for: (2048, 68) can carry a
    rank-68 gradient at most, so at 0.2, which asks for 64 planes, it runs at 34."""
    assert frame.rank(torch.Size((1024, 1024)), 0.1) == 96
    assert frame.rank(torch.Size((1024, 1024)), 0.05) == 64
    assert frame.rank(torch.Size((4608, 1024)), 0.1) == 224
    assert frame.rank(torch.Size((1024, 4608)), 0.05) == 96
    assert frame.rank(torch.Size((128, 128)), 0.05) == 32
    assert frame.rank(torch.Size((8192, 8192)), 0.05) == 256
    assert frame.rank(torch.Size((2048, 68)), 0.2) == 34
    assert frame.rank(torch.Size((64, 1)), 0.1) == 1
