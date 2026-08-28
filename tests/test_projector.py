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


def test_effective_rank_caps_at_half_the_smaller_side():
    """A configured rank is a ceiling, not a promise. Anima's tall narrow
    modules are the case that matters: a (2048, 64) weight can only ever carry
    a rank-64 gradient, and it is a bottleneck, so it runs at 32 no matter what
    the rest of the model is configured for."""
    tall_narrow = torch.zeros(2048, 64)
    for side in ("auto", "left", "right"):
        assert SubspaceProjector(rank=256, side=side).effective_rank(tall_narrow) == 32
    # Below the cap the configured rank is used unchanged.
    assert SubspaceProjector(rank=16, side="auto").effective_rank(tall_narrow) == 16
    # A degenerate side still yields a usable rank.
    assert SubspaceProjector(rank=8, side="auto").effective_rank(torch.zeros(64, 1)) == 1
