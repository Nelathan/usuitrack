from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

import torch
from torch import Tensor

# Final Polar Express coefficient triple, converged to Newton-Schulz's
# steady-state fixed point. See optimizer.py's NEWTON_SCHULZ_COEFFICIENTS
# for the full warm-up schedule used elsewhere in the update.
_POLAR_EXPRESS_FINAL_COEFFICIENTS = (1.875, -1.25, 0.375)

# Hard ceiling on how far one geodesic step may turn the frame, in radians.
#
# This is the invariant every numerical guard around the Oja update was really
# reaching for. Before this clamp the geodesic's principal angles are exactly
# `step_size * sqrt(eigenvalue_i)` (see oja_geodesic_from_eigh), so bounding
# them is a single clamp in the coordinates that mean something, in the one
# place every tangent producer converges -- rather than a collection of proxies
# (a rank ratio, a tangent magnitude, a Gram condition number) scattered across
# the paths that build a tangent, each of which only correlates with rotation.
#
# A quarter turn is a pathology bound, not a tuning knob: a basis that swings
# more than 45 degrees in one step is not tracking a subspace, it is thrashing,
# and the eigenvector directions that produced such a tangent are not
# trustworthy anyway. Healthy runs sit orders of magnitude below it -- on a
# synthetic low-rank-plus-noise bench, 1.4e-2 rad at peak during acquisition
# settling to ~2e-4. Use UsuiTrack.pop_basis_rotation_angle() to see where a
# given run actually lives.
MAX_BASIS_ROTATION = math.pi / 4


class ProjectionSide(StrEnum):
    AUTO = "auto"
    LEFT = "left"
    RIGHT = "right"


@dataclass
class SubspaceProjector:
    """One-sided rank-r projector with stable side-Gram EIGH initialization."""

    rank: int = 32
    side: ProjectionSide | str = ProjectionSide.AUTO
    basis: Tensor | None = None
    resolved_side: ProjectionSide | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError(f"rank must be positive, got {self.rank}")
        self.side = ProjectionSide(self.side)

    @property
    def is_initialized(self) -> bool:
        return self.basis is not None

    def effective_side(self, matrix: Tensor) -> ProjectionSide:
        self._check_matrix(matrix)
        if self.side is ProjectionSide.AUTO:
            return ProjectionSide.RIGHT if matrix.shape[0] >= matrix.shape[1] else ProjectionSide.LEFT
        return ProjectionSide(self.side)

    def tracked_dim(self, matrix: Tensor) -> int:
        """Width of the space the basis lives in, which is the side it tracks."""
        self._check_matrix(matrix)
        return matrix.shape[1] if self.effective_side(matrix) is ProjectionSide.RIGHT else matrix.shape[0]

    def effective_rank(self, matrix: Tensor) -> int:
        # Cap at a quarter of the tracked dimension, not the full dimension.
        # Oja needs contrast in the signal: if nearly all the gradient mass
        # already lies inside the basis, the residual that drives rotation is
        # numerical noise, and the geodesic step size is fixed rather than
        # proportional to how well-aligned the basis already is -- so a
        # near-complete basis does not settle down, it rotates on noise at
        # full speed. Keeping three-quarters of the ambient space outside the
        # basis guarantees there is always real signal in the complement.
        # Observed without it: NaN loss within a couple of steps at exactly
        # full rank, and loss spikes at a half-dimension cap.
        #
        # The quarter is measured against the tracked side rather than
        # min(shape) because that is the space the complement has to exist in.
        # The two coincide whenever the side is `auto` (which picks the smaller
        # side) and for square-ish weights, so this leaves LLM-shaped layers
        # untouched -- a 1024x4096 MLP still resolves to rank 256. They diverge
        # only where a side hint deliberately tracks the larger side, which is
        # exactly where min(shape) was the wrong yardstick: a (2048, 68) input
        # projection tracking its 2048-wide output space was being clamped to
        # rank 17 by its 68-wide data side, a space the basis does not live in.
        # Round to nearest rather than floor: the quarter is a contrast
        # requirement, not a hard boundary, so a dimension of 130 wants 33 and
        # not 32. Floor of at least 1 keeps a degenerate dimension usable.
        return min(self.rank, max(1, round(self.tracked_dim(matrix) / 4)))

    @torch.no_grad()
    def fit(self, matrix: Tensor) -> Tensor:
        self._check_matrix(matrix)
        side = self.effective_side(matrix)
        rank = self.effective_rank(matrix)
        work = self._spectral_input(matrix)
        work = work / work.norm().clamp_min(1e-12)
        _values, vectors = self._side_gram_eigh(work, side)
        basis = vectors[:, -rank:].mT if side is ProjectionSide.RIGHT else vectors[:, -rank:]
        self.basis = basis.to(device=matrix.device, dtype=matrix.dtype).contiguous()
        self.resolved_side = side
        return self.basis

    @staticmethod
    def _side_gram_eigh(work: Tensor, side: ProjectionSide) -> tuple[Tensor, Tensor]:
        gram = work.mT @ work if side is ProjectionSide.RIGHT else work @ work.mT
        gram = 0.5 * (gram + gram.mT)
        try:
            return torch.linalg.eigh(gram)
        except RuntimeError:
            eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
            jitter = 1e-6 * (gram.diagonal().sum() / max(1, gram.shape[0])).clamp_min(1e-12)
            return torch.linalg.eigh(gram + jitter * eye)

    @torch.no_grad()
    def canonical_basis(self) -> Tensor:
        if self.basis is None:
            raise RuntimeError("cannot read a canonical basis before fitting")
        work = self.basis.float() if self.basis.dtype in (torch.float16, torch.bfloat16) else self.basis
        return work.mT if self._basis_side() is ProjectionSide.RIGHT else work

    @torch.no_grad()
    def project(self, matrix: Tensor) -> Tensor:
        basis = self._basis_for(matrix)
        return matrix @ basis.mT if self.effective_side(matrix) is ProjectionSide.RIGHT else basis.mT @ matrix

    @torch.no_grad()
    def project_back(self, projected: Tensor) -> Tensor:
        if self.basis is None:
            raise RuntimeError("cannot project back before fitting a basis")
        # matmul does not promote mixed dtypes the way elementwise ops do, so a
        # caller lifting an fp32 update through a bf16 basis has to be met here
        # rather than forced to round its update down to the basis dtype first.
        basis = self.basis if self.basis.dtype == projected.dtype else self.basis.to(projected.dtype)
        if self._basis_side() is ProjectionSide.RIGHT:
            if projected.ndim != 2 or projected.shape[1] != basis.shape[0]:
                raise ValueError(f"right-basis projected tensor does not match basis {tuple(basis.shape)}")
            return projected @ basis
        if projected.ndim != 2 or projected.shape[0] != basis.shape[1]:
            raise ValueError(f"left-basis projected tensor does not match basis {tuple(basis.shape)}")
        return basis @ projected

    @torch.no_grad()
    def project_and_back(self, matrix: Tensor) -> Tensor:
        return self.project_back(self.project(matrix))

    @torch.no_grad()
    def oja_tangent(self, matrix: Tensor, projected: Tensor | None = None) -> Tensor:
        if self.basis is None:
            raise RuntimeError("cannot compute an Oja tangent before fitting a basis")
        self._check_basis_matches(matrix)
        side = self._basis_side()
        work = matrix.float() if matrix.dtype in (torch.float16, torch.bfloat16) else matrix
        frame = self.canonical_basis()
        low = projected.float() if projected is not None else (work @ frame if side is ProjectionSide.RIGHT else frame.mT @ work)
        action = work.mT @ low if side is ProjectionSide.RIGHT else work @ low.mT
        rayleigh = frame.mT @ action
        rayleigh = 0.5 * (rayleigh + rayleigh.mT)
        # The tangent is deliberately left unbounded here. Its magnitude is not
        # the quantity that has to stay sane -- the rotation it induces is, and
        # that is clamped in oja_geodesic_from_eigh, downstream of both this
        # method and the fused kernels in optimizer.py that reimplement it.
        return (action - frame @ rayleigh) / rayleigh.diagonal().mean().clamp_min(1e-12)

    @staticmethod
    def oja_geodesic_from_eigh(frame: Tensor, tangent: Tensor, eigenvalues: Tensor, eigenvectors: Tensor, step_size: float) -> Tensor:
        sigma = eigenvalues.clamp_min(0.0).sqrt()
        # Clamping the angle alone is exact, not an approximation: the second
        # term's `tangent @ eigenvectors / sigma` is already the unit-norm
        # geodesic direction, so scaling it by sin of the clamped angle lands
        # on the honest geodesic point at that angle. Orthonormality is
        # preserved by construction, whether or not the clamp bites.
        rotation = (step_size * sigma).clamp(max=MAX_BASIS_ROTATION)
        sin_over_sigma = torch.where(sigma > 1e-7, torch.sin(rotation) / sigma.clamp_min(1e-12), torch.full_like(sigma, step_size))
        moved = ((frame @ eigenvectors) * torch.cos(rotation).unsqueeze(-2) + (tangent @ eigenvectors) * sin_over_sigma.unsqueeze(-2)) @ eigenvectors.mT
        return SubspaceProjector._polar_express_stiefel_correction(moved)

    @staticmethod
    def _polar_express_stiefel_correction(frame: Tensor) -> Tensor:
        gram = frame.mT @ frame
        a, b, c = _POLAR_EXPRESS_FINAL_COEFFICIENTS
        correction = c * gram
        correction.diagonal(dim1=-2, dim2=-1).add_(b)
        correction = correction @ gram
        correction.diagonal(dim1=-2, dim2=-1).add_(a)
        return frame @ correction

    @torch.no_grad()
    def orthonormality_error(self) -> Tensor:
        if self.basis is None:
            raise RuntimeError("cannot measure orthonormality before fitting a basis")
        basis = self.basis.float()
        gram = basis @ basis.mT if self._basis_side() is ProjectionSide.RIGHT else basis.mT @ basis
        return (gram - torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)).abs().max()

    def _basis_for(self, matrix: Tensor) -> Tensor:
        if self.basis is None:
            return self.fit(matrix)
        self._check_basis_matches(matrix)
        return self.basis

    def _basis_side(self) -> ProjectionSide:
        if self.basis is None:
            raise RuntimeError("basis has not been fitted")
        if self.side is not ProjectionSide.AUTO:
            return ProjectionSide(self.side)
        if self.resolved_side is None:
            raise RuntimeError("basis side has not been resolved")
        return self.resolved_side

    def _check_basis_matches(self, matrix: Tensor) -> None:
        self._check_matrix(matrix)
        assert self.basis is not None
        side = self.effective_side(matrix)
        rank = self.effective_rank(matrix)
        expected = (rank, matrix.shape[1]) if side is ProjectionSide.RIGHT else (matrix.shape[0], rank)
        if tuple(self.basis.shape) != expected:
            raise ValueError(f"basis shape {tuple(self.basis.shape)} does not match expected {expected}")
        if self.basis.device != matrix.device:
            raise ValueError(f"basis device {self.basis.device} does not match matrix device {matrix.device}")

    @staticmethod
    def _check_matrix(matrix: Tensor) -> None:
        if matrix.ndim != 2 or min(matrix.shape) == 0:
            raise ValueError(f"SubspaceProjector requires a non-empty 2D tensor, got shape {tuple(matrix.shape)}")

    @staticmethod
    def _spectral_input(matrix: Tensor) -> Tensor:
        matrix = matrix.float() if matrix.dtype in (torch.float16, torch.bfloat16) else matrix
        if not torch.isfinite(matrix).all():
            raise RuntimeError("cannot fit a projection basis from non-finite matrix values")
        return matrix
