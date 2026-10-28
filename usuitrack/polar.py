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


def newton_schulz(matrix: Tensor) -> Tensor:
    """Approximate polar factor of `matrix` (batched over leading dims), in fp32."""

    work = matrix.float()
    work = work / work.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-7)
    transposed = work.shape[-2] > work.shape[-1]
    x = work.mT if transposed else work
    for a, b, c in NEWTON_SCHULZ_COEFFICIENTS:
        gram = x @ x.mT
        y = c * gram
        y.diagonal(dim1=-2, dim2=-1).add_(b)
        y = y @ gram
        y.diagonal(dim1=-2, dim2=-1).add_(a)
        x = y @ x
    return x.mT if transposed else x


def balanced_polar(matrix: Tensor) -> Tensor:
    """Aurora's leverage balance, one iteration, then the polar map; fp32.

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


def unit_polar(moments: list[Tensor]) -> list[Tensor]:
    """Each moment's balanced polar factor at unit Frobenius norm.

    A polar factor of rank `r` has norm `sqrt(r)`, `r` the short side, so
    `1/sqrt(r)` makes every step a unit; its size is set against `||W||`
    where it lands. Same-shaped moments share one batched call.
    """

    buckets: dict[tuple[int, ...], list[int]] = {}
    for index, moment in enumerate(moments):
        buckets.setdefault(tuple(moment.shape), []).append(index)
    directions: list[Tensor] = [torch.empty(0)] * len(moments)
    for shape, indices in buckets.items():
        scale = 1.0 / math.sqrt(min(shape))
        batch = balanced_polar(torch.stack([moments[index] for index in indices])) * scale
        for index, direction in zip(indices, batch.unbind(0), strict=True):
            directions[index] = direction
    return directions
