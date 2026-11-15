"""The tracked frame: an orthonormal `Q: [d, r]` on one side of a matrix.

Every function takes the matrix *oriented* so that the frame's side is its
columns -- `G` for `right`, `G^T` for `left` -- which makes the left case the
right case on a transpose (SPEC "Coordinates"). Then the projection is `G Q`,
the lift `Z Q^T`, and the moment lives in `[other side, r]`.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor

from .polar import newton_schulz

SIDES = ("auto", "left", "right")

# Polar Express's converged triple: one step pulls a near-orthonormal frame back
# onto the Stiefel manifold (SPEC step 4).
RETRACTION_COEFFICIENTS = (1.875, -1.25, 0.375)


def resolve_side(shape: torch.Size, side: str) -> str:
    """`auto` puts the frame on the shorter side, `right` (length `n`) when
    `m >= n`: a shape guess, not an architectural one."""

    if side == "auto":
        return "right" if shape[0] >= shape[1] else "left"
    return side


def orient(matrix: Tensor, side: str) -> Tensor:
    return matrix if side == "right" else matrix.mT


def rank(shape: torch.Size, rank_fraction: float) -> int:
    """`rank_fraction * sqrt(m n)`, at most half the shorter side (SPEC "Coordinates")."""

    rows, cols = shape
    wanted = max(1, round(rank_fraction * math.sqrt(rows * cols)))
    return min(wanted, max(1, min(rows, cols) // 2))


def fit(grad: Tensor, rank: int) -> Tensor:
    """A `rank`-dim orthonormal frame approximating the leading eigenspace of the side Gram `G^T G`, in
    `grad`'s dtype: `range_find` on the fixed `sketch` (SPEC step 2)."""

    return range_find(grad, sketch(grad.shape[1], rank, grad.device))


def sketch(length: int, rank: int, device: torch.device) -> Tensor:
    """The range finder's Gaussian test matrix `[length, rank]`, fp32, from a fixed seed: the fit is then a
    function of the gradient alone, and training's random stream never shifts."""

    generator = torch.Generator(device).manual_seed(0)
    return torch.randn(length, rank, device=device, dtype=torch.float32, generator=generator)


def range_find(grad: Tensor, test: Tensor) -> Tensor:
    """The frame `fit` returns, from a drawn `sketch`: a random range finder with one power step. No
    eigendecomposition, so nothing to fail on a nearly rank-one Gram, and nothing a compiled backward cannot
    trace; a zero `grad` gives an arbitrary orthonormal frame (QR of zero)."""

    work = grad.float()
    work = work / work.norm().clamp_min(1e-12)
    gram = work.mT @ work
    q = torch.linalg.qr(gram @ test).Q
    q = torch.linalg.qr(gram @ q).Q
    return q.to(grad.dtype).contiguous()


def draw(length: int, rank: int, generator: torch.Generator, dtype: torch.dtype) -> Tensor:
    """A random `rank`-dim orthonormal frame on a side of `length`, in `dtype`: the QR of a Gaussian draw from
    `generator`, for a matrix whose first gradients carry no subspace to fit (SPEC step 2, `frame_init`)."""

    gaussian = torch.randn(length, rank, device=generator.device, dtype=torch.float32, generator=generator)
    return torch.linalg.qr(gaussian).Q.to(dtype).contiguous()


def project_and_aim(grad: Tensor, frame: Tensor) -> tuple[Tensor, Tensor]:
    """The held-frame projection `Z = G Q`, and the aim built from it.

    SPEC steps 3-4. `Z` keeps the gradient's dtype; the aim is fp32.
    """

    projected = grad @ frame
    return projected, aim(grad.float().mT @ projected.float(), frame)


def aim(action: Tensor, frame: Tensor) -> Tensor:
    """The frame's action `A = G^T G Q` over its mean Rayleigh quotient, fp32:
    `Q^T a` has mean diagonal 1, so the aim is blind to the gradient's scale
    (SPEC step 4). The floor guards only an exactly zero gradient: the twin
    reads the aim's size, so a larger one would stall a matrix of tiny entries."""

    rayleigh = (frame.float() * action).sum() / frame.shape[-1]
    return action / rayleigh.clamp_min(1e-30)


def retract(frame: Tensor) -> Tensor:
    """One Polar Express step toward the nearest orthonormal frame; exact enough
    only for a frame already within small principal angles of one."""

    gram = frame.mT @ frame
    a, b, c = RETRACTION_COEFFICIENTS
    correction = c * gram
    correction.diagonal().add_(b)
    correction = correction @ gram
    correction.diagonal().add_(a)
    return frame @ correction


class Moved(NamedTuple):
    frame: Tensor
    """The moved frame, rounded to the stored frame's dtype."""
    rotation: Tensor
    """`[r, r]`, fp32: the in-span rotation between the old and new frames as
    stored, which the moment is carried by (SPEC step 6)."""


def follow(stored: Tensor, frame_moment: Tensor) -> Moved:
    """The frame as the polar factor of `Q + Y`, `Y` the moment of aims (SPEC step 4).

    `Q` anchors the target: `T = Q + Y` has its singular values at or above one
    while the symmetric part of `Q^T Y` is positive semidefinite, so no plane
    collapses, and an aim inside the span holds the frame exactly. Newton-Schulz lands within a few percent of
    orthonormal and the retraction finishes it; there is no `eigh` and nothing
    read back to the host.
    """

    q = stored.float()
    # Round-to-nearest, unlike the moment: the frame moves by a real angle
    # every update, so there is no sub-ulp step to lose, and stochastic
    # rounding measured sqrt(2) more in-span rotation.
    new = retract(newton_schulz(q + frame_moment)).to(stored.dtype)
    rotation = retract(q.mT @ new.float())
    return Moved(new, rotation)
