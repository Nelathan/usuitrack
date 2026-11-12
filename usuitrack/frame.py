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
    """A `rank`-dim orthonormal frame on the leading eigenspace of the side Gram `G^T G`, in `grad`'s dtype:
    `range_find` on the fixed `sketch` (SPEC step 2)."""

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
    """The held-frame projection `Z = G Q`, and the Oja tangent built from it.

    SPEC steps 3-4. `Z` keeps the gradient's dtype; the tangent is fp32 and
    horizontal, `Q^T Delta = 0`.
    """

    projected = grad @ frame
    return projected, aim(grad.float().mT @ projected.float(), frame)


def aim(action: Tensor, frame: Tensor) -> Tensor:
    """The Oja tangent from the frame's action `A = G^T G Q`, fp32: its
    horizontal part, scaled by the mean Rayleigh quotient (SPEC step 4)."""

    q = frame.float()
    rayleigh = q.mT @ action
    rayleigh = 0.5 * (rayleigh + rayleigh.mT)
    return (action - q @ rayleigh) / rayleigh.diagonal().mean().clamp_min(1e-12)


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


def tangent_spectra(tangents: list[Tensor]) -> list[tuple[Tensor, Tensor]]:
    """`eigh` of each tangent Gram `Delta^T Delta`: eigenvalues ascending, vectors.

    The Grams are `[r, r]`, where `eigh` is launch-bound and reads its status
    back to the host, so all of them share one call: each is padded to the
    largest rank with `-1` on the padded diagonal, a block apart whose
    eigenpairs sort first and are dropped. They are the only per-matrix tensors
    stacked, and they are small. Each Gram is divided by its mean diagonal for
    the `eigh` and its eigenvalues multiplied back: cuSOLVER's fp32 `eigh`
    returns non-orthonormal vectors for a Gram whose entries are tiny (a matrix
    behind a still-closed gate, ~1e-22), and the turn then throws the frame off
    the manifold. A scaled Gram is PSD, so its eigenvalues stay above the
    padding's.
    """

    if not tangents:
        return []
    grams = [tangent.mT @ tangent for tangent in tangents]
    width = max(gram.shape[-1] for gram in grams)
    padding = -torch.eye(width, device=grams[0].device)
    scales = [gram.diagonal().mean().clamp_min(1e-30) for gram in grams]
    stacked = torch.stack([torch.block_diag(gram / scale, padding[gram.shape[-1] :, gram.shape[-1] :]) for gram, scale in zip(grams, scales, strict=True)])
    eigenvalues, vectors = torch.linalg.eigh(0.5 * (stacked + stacked.mT))
    spectra = []
    for gram, scale, values, basis in zip(grams, scales, eigenvalues, vectors, strict=True):
        rank = gram.shape[-1]
        spectra.append((values[width - rank :] * scale, basis[:rank, width - rank :]))
    return spectra


def turn(stored: Tensor, tangent: Tensor, spectrum: tuple[Tensor, Tensor], eta: float) -> Turned:
    """Move the frame along its polar tangent by `eta` on every live plane.

    SPEC step 4. With `Delta^T Delta = V diag(sigma^2) V^T`, the polar tangent
    is `Delta V diag(1/sigma) V^T`, so the geodesic is
    `[(Q V) cos(eta) + (Delta V) diag(sin(eta) / sigma)] V^T` on live planes.
    A plane is dead below the Gram's own backward error, `sqrt(r eps) sigma_max`,
    and is held still: angle zero, not merely a zero tangent column.
    `spectrum` is the tangent Gram's, from `tangent_spectra`.
    """

    q = stored.float()
    eigenvalues, vectors = spectrum
    sigma = eigenvalues.clamp_min(0.0).sqrt()
    floor = math.sqrt(sigma.shape[-1] * torch.finfo(sigma.dtype).eps)
    live = sigma > floor * sigma.amax().clamp_min(1e-30)
    inverse = torch.where(live, 1.0 / sigma.clamp_min(1e-30), torch.zeros_like(sigma))
    cosine = torch.cos(eta * live.float())
    moved = ((q @ vectors) * cosine + (tangent @ vectors) * (inverse * math.sin(eta))) @ vectors.mT
    # Round-to-nearest, unlike the moment: the frame moves by a real angle
    # every update, so there is no sub-ulp step to lose, and stochastic
    # rounding measured sqrt(2) more in-span rotation.
    new = retract(moved).to(stored.dtype)
    rotation = retract(q.mT @ new.float())
    return Turned(new, rotation, eigenvalues, live)
