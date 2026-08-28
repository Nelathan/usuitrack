import torch

from usuitrack import ProjectionSide, SubspaceProjector


# A representative transformer-ish weight. Small toy shapes like (8, 4) put
# effective_rank() straight into its floor -- a quarter of 4 rounds to 1 -- so
# every test silently exercised a rank-1 basis instead of the regime the
# optimizer is designed for. RANK sits below the quarter cap of COLS so the
# configured rank is the rank actually used.
ROWS, COLS, RANK = 256, 128, 16


def test_eigh_init_is_orthonormal_both_sides():
    for side, shape in ((ProjectionSide.RIGHT, (ROWS, COLS)), (ProjectionSide.LEFT, (COLS, ROWS))):
        projector = SubspaceProjector(rank=RANK, side=side)
        basis = projector.fit(torch.randn(*shape))
        expected_shape = (RANK, shape[1]) if side is ProjectionSide.RIGHT else (shape[0], RANK)
        assert tuple(basis.shape) == expected_shape
        assert float(projector.orthonormality_error()) < 1e-5


def test_zero_input_gives_deterministic_frame():
    zero = torch.zeros(ROWS, COLS)
    first = SubspaceProjector(rank=RANK, side="right").fit(zero)
    second = SubspaceProjector(rank=RANK, side="right").fit(zero)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_project_and_lift_round_trip_shape():
    for side, shape in (("right", (ROWS, COLS)), ("left", (COLS, ROWS))):
        matrix = torch.randn(*shape)
        projector = SubspaceProjector(rank=RANK, side=side)
        lifted = projector.project_back(projector.project(matrix))
        assert tuple(lifted.shape) == shape


def test_oja_tangent_is_horizontal_and_geodesic_stays_on_stiefel():
    matrix = torch.randn(ROWS, COLS)
    projector = SubspaceProjector(rank=RANK, side="right")
    projector.fit(matrix)
    projected = projector.project(matrix)
    tangent = projector.oja_tangent(matrix, projected=projected)
    frame = projector.canonical_basis()

    # Horizontal: the tangent has no component along the current frame.
    torch.testing.assert_close(frame.mT @ tangent, torch.zeros(RANK, RANK), atol=1e-5, rtol=0)

    gram = tangent.mT @ tangent
    values, vectors = torch.linalg.eigh(0.5 * (gram + gram.mT))
    moved = SubspaceProjector.oja_geodesic_from_eigh(frame, tangent, values, vectors, 0.5)
    torch.testing.assert_close(moved.mT @ moved, torch.eye(RANK), atol=3e-3, rtol=0)
