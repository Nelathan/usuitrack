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
    """Top-`rank` eigenvectors of the side Gram `G^T G` (SPEC step 2), in `grad`'s dtype."""

    work = grad.float()
    work = work / work.norm().clamp_min(1e-12)
    gram = work.mT @ work
    _values, vectors = torch.linalg.eigh(0.5 * (gram + gram.mT))
    return vectors[:, -rank:].to(grad.dtype).contiguous()


def project_and_aim(grad: Tensor, frame: Tensor) -> tuple[Tensor, Tensor]:
    """The held-frame projection `Z = G Q`, and the Oja tangent built from it.

    SPEC steps 3-4. `Z` keeps the gradient's dtype; the tangent is fp32 and
    horizontal, `Q^T Delta = 0`.
    """

    projected = grad @ frame
    q = frame.float()
    action = grad.float().mT @ projected.float()
    rayleigh = q.mT @ action
    rayleigh = 0.5 * (rayleigh + rayleigh.mT)
    tangent = (action - q @ rayleigh) / rayleigh.diagonal().mean().clamp_min(1e-12)
    return projected, tangent


def retract(frame: Tensor) -> Tensor:
    """One Polar Express step toward the nearest orthonormal frame; exact enough
    only for a frame already within small principal angles of one."""

    gram = frame.mT @ frame
    a, b, c = RETRACTION_COEFFICIENTS
    correction = c * gram
    correction.diagonal(dim1=-2, dim2=-1).add_(b)
    correction = correction @ gram
    correction.diagonal(dim1=-2, dim2=-1).add_(a)
    return frame @ correction


class Turned(NamedTuple):
    frame: Tensor
    """The moved frame, rounded to the stored frame's dtype."""
    rotation: Tensor
    """`[r, r]`, fp32: the in-span rotation between the old and new frames as stored.
    The identity unless rounding turned the columns inside their span (SPEC step 6)."""
    eigenvalues: Tensor
    """The tangent Gram's spectrum, ascending."""
    live: Tensor
    """`[r]` bool: the planes that turned."""


def turn(frames: list[Tensor], tangents: list[Tensor], eta: float) -> list[Turned]:
    """Move each frame along its polar tangent by `eta` on every live plane.

    SPEC step 4. With `Delta^T Delta = V diag(sigma^2) V^T`, the polar tangent
    is `Delta V diag(1/sigma) V^T`, so the geodesic is
    `[(Q V) cos(eta) + (Delta V) diag(sin(eta) / sigma)] V^T` on live planes.
    A plane is dead below the Gram's own backward error, `sqrt(r eps) sigma_max`,
    and is held still: angle zero, not merely a zero tangent column. Batched
    over frames of one shape.
    """

    turned: list[Turned | None] = [None] * len(frames)
    buckets: dict[tuple[int, ...], list[int]] = {}
    for index, frame in enumerate(frames):
        buckets.setdefault(tuple(frame.shape), []).append(index)
    for indices in buckets.values():
        stored = torch.stack([frames[index] for index in indices])
        q = stored.float()
        tangent = torch.stack([tangents[index] for index in indices])
        gram = tangent.mT @ tangent
        eigenvalues, vectors = torch.linalg.eigh(0.5 * (gram + gram.mT))
        sigma = eigenvalues.clamp_min(0.0).sqrt()
        floor = math.sqrt(sigma.shape[-1] * torch.finfo(sigma.dtype).eps)
        live = sigma > floor * sigma.amax(dim=-1, keepdim=True).clamp_min(1e-30)
        inverse = torch.where(live, 1.0 / sigma.clamp_min(1e-30), torch.zeros_like(sigma))
        cosine = torch.cos(eta * live.float())
        moved = ((q @ vectors) * cosine.unsqueeze(-2) + (tangent @ vectors) * (inverse * math.sin(eta)).unsqueeze(-2)) @ vectors.mT
        # Round-to-nearest, unlike the moment: the frame moves by a real angle
        # every update, so there is no sub-ulp step to lose, and stochastic
        # rounding measured sqrt(2) more in-span rotation.
        new = retract(moved).to(stored.dtype)
        rotation = retract(q.mT @ new.float())
        for position, index in enumerate(indices):
            turned[index] = Turned(new[position], rotation[position], eigenvalues[position], live[position])
    return turned  # type: ignore[return-value]
