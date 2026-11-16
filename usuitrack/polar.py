"""The update's direction: Aurora's row balance, then the Newton-Schulz polar map.

SPEC step 7. Both are reimplemented from their methods; neither is a runtime
dependency.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

# Per-step-tuned quintic schedule of the Muon lineage (SPEC step 7 credits it).
NEWTON_SCHULZ_COEFFICIENTS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)
# The r x r iterations run in one Triton program per matrix up to this rank (`gram.py`): one program cannot
# spread a wider matrix across the GPU, and batched GEMMs carry it faster there.
KERNEL_RANK = 128


def newton_schulz(matrix: Tensor, schedule: tuple[tuple[float, float, float], ...] = NEWTON_SCHULZ_COEFFICIENTS) -> Tensor:
    """Approximate polar factor of `matrix`, or of each matrix in a `[b, m, n]` stack, in fp32.

    Every iterate is `X C` for an `r x r` polynomial `C` in the short-side Gram `X^T X`, so
    the schedule runs on the Gram and one product closes it (SPEC step 7).
    """

    work = matrix.float()
    transposed = work.shape[-2] < work.shape[-1]
    tall = work.mT if transposed else work
    polar = tall @ gram_polynomial(tall.mT @ tall, schedule)
    return polar.mT if transposed else polar


def gram_polynomial(gram: Tensor, schedule: tuple[tuple[float, float, float], ...]) -> Tensor:
    """`C` with `X C` the schedule's last iterate, from `G = X^T X` `[..., r, r]`: on CUDA up
    to `KERNEL_RANK` the Triton kernel (`gram.py`), elsewhere this loop."""

    if gram.is_cuda and gram.shape[-1] <= KERNEL_RANK:
        from .gram import polynomial

        return polynomial(gram, schedule)
    # Newton-Schulz's Frobenius normalization of X: ||X||^2 = trace G.
    scale = gram.diagonal(dim1=-2, dim2=-1).sum(-1).sqrt().clamp_min(1e-7).reciprocal()[..., None, None]
    eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
    c = scale * eye
    m = gram * scale.square()
    # With M = C^T G C the iterate's Gram, an iteration is C <- P C, M <- P^2 M:
    # every factor is a polynomial in G, so they commute.
    for a, b, coefficient in schedule:
        p = a * eye + b * m + coefficient * (m @ m)
        c = p @ c
        m = p @ (p @ m)
    return c


def balanced_polar(matrix: Tensor) -> Tensor:
    """Aurora's leverage balance, one iteration, then the polar map; fp32,
    per matrix of a stack.

    The tall orientation's rows are scaled to unit norm before the polar map;
    a square matrix has nothing to balance.
    """

    if matrix.shape[-2] == matrix.shape[-1]:
        return newton_schulz(matrix)
    transposed = matrix.shape[-2] < matrix.shape[-1]
    work = (matrix.mT if transposed else matrix).float()
    rows = work.norm(dim=-1, keepdim=True).clamp_min(1e-7).reciprocal()
    polar = newton_schulz(rows * work)
    return polar.mT if transposed else polar


def unit_polar(moment: Tensor) -> Tensor:
    """The moment's balanced polar factor at unit Frobenius norm.

    A polar factor of rank `r` has norm `sqrt(r)`, `r` the short side, so
    `1/sqrt(r)` makes every step a unit; its size is set against `||W||`
    where it lands.
    """

    return balanced_polar(moment) * (1.0 / math.sqrt(min(moment.shape[-2:])))
