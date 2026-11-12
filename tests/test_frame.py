import math

import pytest
import torch

from usuitrack import frame

# RANK_FRACTION is chosen so (ROWS, COLS) runs at exactly RANK, below the cap.
ROWS, COLS, RANK = 256, 128, 16
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


def test_one_stacked_eigh_gives_each_grams_own_spectrum():
    """Grams of different ranks share one padded `eigh`; each comes back as its
    own decomposition, including one from a tangent of tiny entries."""

    torch.manual_seed(0)
    tangents = [torch.randn(COLS, 16), torch.randn(ROWS, 5), 1e-11 * torch.randn(COLS, 9), torch.randn(COLS, 16)]
    for tangent, (values, vectors) in zip(tangents, frame.tangent_spectra(tangents), strict=True):
        gram = tangent.mT @ tangent
        rank = gram.shape[-1]
        assert tuple(values.shape) == (rank,) and tuple(vectors.shape) == (rank, rank)
        torch.testing.assert_close(vectors.mT @ vectors, torch.eye(rank), atol=1e-5, rtol=0)
        torch.testing.assert_close(vectors @ torch.diag(values) @ vectors.mT, gram, atol=1e-4 * gram.abs().max().item(), rtol=1e-4)
        torch.testing.assert_close(values, torch.linalg.eigvalsh(gram), atol=1e-4 * gram.abs().max().item(), rtol=1e-4)


def test_fitting_leaves_the_training_random_stream_alone():
    """The fit draws from its own generator, so a run's batches and dropout do
    not depend on when its matrices fit (SPEC step 2)."""
    gradient = torch.randn(ROWS, COLS)
    torch.manual_seed(0)
    untouched = torch.rand(4)
    torch.manual_seed(0)
    frame.fit(gradient, RANK)
    assert torch.equal(torch.rand(4), untouched)


def test_tangent_is_horizontal():
    """Oja's tangent has no component along the frame, so the geodesic turns
    strictly into the complement. Taken on a gradient the frame was not fitted
    on: a fitted frame is a fixed point, which would make this vacuous."""
    torch.manual_seed(0)
    q = frame.fit(torch.randn(ROWS, COLS), RANK)
    _projected, tangent = frame.project_and_aim(torch.randn(ROWS, COLS), q)
    assert float((q.mT @ tangent).abs().max()) < 1e-3 * float(tangent.abs().max())


def test_a_fitted_frame_is_a_fixed_point_of_its_own_gradient():
    """A frame fitted on `G` is the leading eigenspace of `G^T G` where the
    gradient determines one, so the Oja action lands inside it: the tracker
    stops when the gradient's principal subspace stops moving."""
    torch.manual_seed(0)
    gradient = torch.randn(ROWS, RANK) @ torch.randn(RANK, COLS)
    _projected, tangent = frame.project_and_aim(gradient, frame.fit(gradient, RANK))
    assert float(tangent.abs().max()) < 1e-5


def test_a_gradient_of_lower_rank_than_the_frame_lies_wholly_inside_it():
    """A zero-initialized model hands its upstream matrices a first gradient of
    rank far below the frame's, nearly rank one (SPEC step 2); the fit holds
    all of it, and the planes it does not span are still orthonormal."""
    torch.manual_seed(0)
    gradient = 1e3 * torch.randn(ROWS, 1) @ torch.randn(1, COLS) + torch.randn(ROWS, 2) @ torch.randn(2, COLS)
    q = frame.fit(gradient, RANK)
    torch.testing.assert_close(q.mT @ q, torch.eye(RANK), atol=1e-5, rtol=0)
    assert float((gradient - gradient @ q @ q.mT).norm() / gradient.norm()) < 1e-4  # the fp32 Gram squares the tail's 1e-3


def _turn(q, tangent, eta):
    (spectrum,) = frame.tangent_spectra([tangent])
    return frame.turn(q, tangent, spectrum, eta)


def test_tangent_spectra_match_each_gram_alone_across_ranks():
    """Same-rank Grams share one `eigh`; grouping is bookkeeping, not maths.
    Not bitwise: batched and solo calls round differently at the ulp level."""
    torch.manual_seed(0)
    tangents = [torch.randn(COLS, rank) for rank in (RANK, RANK + 2, RANK, RANK + 2)]
    for tangent, (eigenvalues, vectors) in zip(tangents, frame.tangent_spectra(tangents), strict=True):
        gram = tangent.mT @ tangent
        torch.testing.assert_close(eigenvalues, torch.linalg.eigvalsh(gram), rtol=1e-5, atol=1e-4)
        torch.testing.assert_close(vectors @ torch.diag(eigenvalues) @ vectors.mT, gram, rtol=1e-5, atol=1e-3)


def test_turn_stays_on_the_stiefel_manifold_and_turns_each_plane_by_eta():
    torch.manual_seed(0)
    q = torch.linalg.qr(torch.randn(COLS, RANK))[0]
    tangent = _horizontal_tangent(q)
    for eta in (0.01, 0.2):
        turned = _turn(q, tangent, eta)
        torch.testing.assert_close(turned.frame.mT @ turned.frame, torch.eye(RANK), atol=1e-4, rtol=0)
        moved = (turned.frame - q @ (q.mT @ turned.frame)).norm() / math.sqrt(RANK)
        assert float(moved) == math.sin(eta) or abs(float(moved) / math.sin(eta) - 1) < 2e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuSOLVER's fp32 eigh is the one that fails")
def test_a_tiny_tangent_turns_the_frame_and_keeps_it_orthonormal():
    """A matrix behind a gate still near zero hands the frame a tangent of
    ~1e-11 entries; its Gram's spectrum must stay orthonormal, and so must the turn."""
    torch.manual_seed(0)
    q = torch.linalg.qr(torch.randn(512, 72, device="cuda"))[0]
    tangent = 1e-11 * _horizontal_tangent(q) @ torch.diag(torch.logspace(0, -3, 72, device="cuda"))
    (spectrum,) = frame.tangent_spectra([tangent])
    torch.testing.assert_close(spectrum[1].mT @ spectrum[1], torch.eye(72, device="cuda"), atol=1e-5, rtol=0)
    turned = frame.turn(q, tangent, spectrum, 0.05)
    torch.testing.assert_close(turned.frame.mT @ turned.frame, torch.eye(72, device="cuda"), atol=1e-4, rtol=0)
    assert bool(turned.live.any()) and float((turned.frame - q).abs().max()) > 1e-3


def test_dead_planes_hold_still():
    """A tangent of rank 2 turns two planes; the rest of the frame stays put."""
    torch.manual_seed(0)
    q = torch.linalg.qr(torch.randn(COLS, RANK))[0]
    low_rank = _horizontal_tangent(q)[:, :2] @ torch.randn(2, RANK)
    turned = _turn(q, low_rank, 0.05)
    assert int(turned.live.sum()) == 2
    moved = turned.frame - q @ (q.mT @ turned.frame)
    assert torch.linalg.matrix_rank(moved, atol=1e-4) == 2


def test_stored_rotation_is_identity_in_fp32_and_fires_in_bf16():
    """The moment's correction fires on the in-span turn storage adds, and only then."""
    torch.manual_seed(0)
    q = torch.linalg.qr(torch.randn(COLS, RANK))[0]
    tangent = _horizontal_tangent(q)
    exact = _turn(q, tangent, 0.02)
    torch.testing.assert_close(exact.rotation, torch.eye(RANK), atol=1e-5, rtol=0)

    rounded = _turn(q.bfloat16(), tangent, 0.02)
    rotation = rounded.rotation
    torch.testing.assert_close(rotation.mT @ rotation, torch.eye(RANK), atol=1e-4, rtol=0)
    assert float((rotation - rotation.mT).abs().max()) > 1e-4


def test_rank_is_a_fraction_of_the_size_capped_at_half_the_smaller_side():
    """Anima's patch embedding is the case the cap exists for: (2048, 68) asks
    for 37 planes at 0.1 and can carry a rank-68 gradient at most, so it runs at 34."""
    assert frame.rank(torch.Size((1024, 1024)), 0.1) == 102
    assert frame.rank(torch.Size((4608, 1024)), 0.1) == 217
    assert frame.rank(torch.Size((1024, 4608)), 0.1) == 217
    assert frame.rank(torch.Size((2048, 68)), 0.1) == 34
    assert frame.rank(torch.Size((64, 1)), 0.1) == 1
