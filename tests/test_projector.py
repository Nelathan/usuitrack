import math

import torch

from usuitrack import ProjectionSide, SubspaceProjector


# A representative transformer-ish weight. Small toy shapes like (8, 4) put
# effective_rank() straight into its floor, so every test silently exercised a
# rank-1 basis instead of the regime the optimizer is designed for.
# RANK_FRACTION is chosen so this shape runs at exactly RANK, below the cap.
ROWS, COLS, RANK = 256, 128, 16
RANK_FRACTION = RANK / math.sqrt(ROWS * COLS)


def test_eigh_init_is_orthonormal_both_sides():
    for side, shape in ((ProjectionSide.RIGHT, (ROWS, COLS)), (ProjectionSide.LEFT, (COLS, ROWS))):
        projector = SubspaceProjector(rank_fraction=RANK_FRACTION, side=side)
        basis = projector.fit(torch.randn(*shape))
        expected_shape = (RANK, shape[1]) if side is ProjectionSide.RIGHT else (shape[0], RANK)
        assert tuple(basis.shape) == expected_shape
        assert float(projector.orthonormality_error()) < 1e-5


def test_zero_input_gives_deterministic_frame():
    zero = torch.zeros(ROWS, COLS)
    first = SubspaceProjector(rank_fraction=RANK_FRACTION, side="right").fit(zero)
    second = SubspaceProjector(rank_fraction=RANK_FRACTION, side="right").fit(zero)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_project_and_lift_round_trip_shape():
    for side, shape in (("right", (ROWS, COLS)), ("left", (COLS, ROWS))):
        matrix = torch.randn(*shape)
        projector = SubspaceProjector(rank_fraction=RANK_FRACTION, side=side)
        lifted = projector.project_back(projector.project(matrix))
        assert tuple(lifted.shape) == shape


def test_geodesic_stays_on_the_stiefel_manifold():
    """The geodesic must return an orthonormal frame for any horizontal tangent,
    at any step size. The tangent is built here rather than taken from the
    optimizer so this stays a test of the retraction alone; the tangent's own
    horizontality is checked against the live kernel in test_optimizer.py."""
    torch.manual_seed(0)
    frame = torch.linalg.qr(torch.randn(COLS, RANK))[0]
    tangent = torch.randn(COLS, RANK)
    tangent = tangent - frame @ (frame.mT @ tangent)  # horizontal by construction

    gram = tangent.mT @ tangent
    values, vectors = torch.linalg.eigh(0.5 * (gram + gram.mT))
    for step_size in (0.01, 0.5, 5.0):
        moved = SubspaceProjector.oja_geodesic_from_eigh(frame, tangent, values, vectors, step_size)
        torch.testing.assert_close(moved.mT @ moved, torch.eye(RANK), atol=3e-3, rtol=0)


def test_effective_rank_is_a_fraction_of_the_size_capped_at_half_the_smaller_side():
    """One fraction sizes every matrix by `sqrt(m * n)`; matrices of one shape
    share a rank whichever way they are stored. Anima's patch embedding is the
    case the cap exists for: a (2048, 68) weight asks for 37 planes at 0.1 and
    can carry a rank-68 gradient at most, so it runs at 34."""
    for side in ("auto", "left", "right"):
        projector = SubspaceProjector(rank_fraction=0.1, side=side)
        assert projector.effective_rank(torch.zeros(1024, 1024)) == 102
        assert projector.effective_rank(torch.zeros(4608, 1024)) == 217
        assert projector.effective_rank(torch.zeros(1024, 4608)) == 217
        assert projector.effective_rank(torch.zeros(2048, 68)) == 34
    # A degenerate side still yields a usable rank.
    assert SubspaceProjector(rank_fraction=0.1).effective_rank(torch.zeros(64, 1)) == 1
