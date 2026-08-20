import torch

from usuitrack import ProjectionSide, SubspaceProjector


def test_eigh_init_is_orthonormal_both_sides():
    for side, shape in ((ProjectionSide.RIGHT, (8, 4)), (ProjectionSide.LEFT, (4, 8))):
        projector = SubspaceProjector(rank=3, side=side)
        basis = projector.fit(torch.randn(*shape))
        expected_shape = (3, shape[1]) if side is ProjectionSide.RIGHT else (shape[0], 3)
        assert tuple(basis.shape) == expected_shape
        assert float(projector.orthonormality_error()) < 1e-5


def test_zero_input_gives_deterministic_frame():
    zero = torch.zeros(8, 4)
    first = SubspaceProjector(rank=3, side="right").fit(zero)
    second = SubspaceProjector(rank=3, side="right").fit(zero)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_project_and_lift_round_trip_shape():
    for side, shape in (("right", (8, 4)), ("left", (4, 8))):
        matrix = torch.randn(*shape)
        projector = SubspaceProjector(rank=3, side=side)
        lifted = projector.project_back(projector.project(matrix))
        assert tuple(lifted.shape) == shape


def test_oja_tangent_is_horizontal_and_geodesic_stays_on_stiefel():
    matrix = torch.randn(8, 4)
    projector = SubspaceProjector(rank=3, side="right")
    projector.fit(matrix)
    projected = projector.project(matrix)
    tangent = projector.oja_tangent(matrix, projected=projected)
    frame = projector.canonical_basis()

    # Horizontal: the tangent has no component along the current frame.
    torch.testing.assert_close(frame.mT @ tangent, torch.zeros(3, 3), atol=1e-5, rtol=0)

    gram = tangent.mT @ tangent
    values, vectors = torch.linalg.eigh(0.5 * (gram + gram.mT))
    moved = SubspaceProjector.oja_geodesic_from_eigh(frame, tangent, values, vectors, 0.5)
    torch.testing.assert_close(moved.mT @ moved, torch.eye(3), atol=3e-3, rtol=0)
