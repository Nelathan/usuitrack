import pytest
import torch

from usuitrack import frame, polar


def _iterated(matrix: torch.Tensor, schedule) -> torch.Tensor:
    """SPEC step 7's iteration on X itself, in float64: X <- (a I + b H + c H^2) X, H = X X^T."""
    x = matrix.double()
    transposed = x.shape[-2] > x.shape[-1]
    x = x.mT if transposed else x
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-7)
    for a, b, c in schedule:
        h = x @ x.mT
        eye = torch.eye(h.shape[-1], dtype=h.dtype)
        x = (a * eye + b * h + c * h @ h) @ x
    return (x.mT if transposed else x).float()


@pytest.mark.parametrize("shape", [(96, 20), (20, 96), (3, 300, 40), (64, 64)])
@pytest.mark.parametrize("schedule", [polar.NEWTON_SCHULZ_COEFFICIENTS, frame.FOLLOW_SCHEDULE])
def test_the_gram_route_is_the_iteration_on_the_matrix(shape, schedule):
    """Every iterate is X C with C a polynomial in X^T X, so running the schedule on the
    Gram lands where the iteration on X does."""
    matrix = torch.randn(*shape) @ torch.diag(torch.logspace(0, -2, shape[-1]))
    torch.testing.assert_close(polar.newton_schulz(matrix, schedule), _iterated(matrix, schedule), rtol=1e-4, atol=1e-5)


def test_a_zero_matrix_maps_to_zero():
    assert polar.newton_schulz(torch.zeros(3, 50, 8)).abs().max() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the Triton kernel runs on CUDA")
@pytest.mark.parametrize("rank", [1, 20, 32, 77, polar.KERNEL_RANK])
@pytest.mark.parametrize("schedule", [polar.NEWTON_SCHULZ_COEFFICIENTS, frame.FOLLOW_SCHEDULE])
def test_the_cuda_kernel_lands_where_the_eager_loop_lands(rank, schedule):
    """Against the eager loop on CPU in fp32, for a frame-like and a moment-like Gram; the
    kernel's tf32x3 dots stay within fp32 rounding of it."""
    torch.manual_seed(0)
    q = torch.linalg.qr(torch.randn(4, 3 * rank + 8, rank))[0]
    for x in (q + 3.0 * torch.randn_like(q) / q.shape[-2] ** 0.5, torch.randn_like(q) @ torch.diag(torch.logspace(0, -2, rank))):
        gram = x.mT @ x
        expected = polar.gram_polynomial(gram, schedule)
        torch.testing.assert_close(polar.gram_polynomial(gram.cuda(), schedule).cpu(), expected, rtol=1e-4, atol=1e-5)
