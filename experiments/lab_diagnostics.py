"""Lab reads on top of the release's: frame motion, spectrum shape, persistence,
and per-role breakouts. For debugging the tracker, not for reading a run cold.

Install with `optimizer.diagnostics = LabDiagnostics(...)`. Frame-motion reads
are taken on a fixed sample of matrices, so their cost does not grow with the
model; snapshot reads (`snapshot_interval > 0`) keep one `[d, r]` frame per
sampled matrix outside optimizer state, so they never enter a checkpoint.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from usuitrack.diagnostics import READS, Diagnostics

LAB_READS = {
    "moment_persistence": (
        "Coherent share of the moment once the incomplete-cancellation floor of an independent "
        "stream, w^2 = (1-beta)/(1+beta), is removed: 0 white, 1 a held direction. Lags, and a "
        "large lr drains it."
    ),
    "tangent_concentration": "Leading eigenvalue's share of the tangent Gram, in [1/r, 1]: a confident aim against a noise tail.",
    "transport_speed": "Chordal distance per plane a sampled frame moved in one update; fixed by eta under the constant turn.",
    "transport_correction": "||skew(R_s)|| / sqrt(r): the in-span rotation the moment is turned by this step; zero for exact storage.",
    "transport_lag": "Net distance per plane a sampled frame covered over the snapshot window.",
    "transport_curve": "1 - lag / path over the window: 0 straight drift, toward 1 churn in place.",
}

# Reads sampled once per matrix, which is what a role can split.
ROLE_READS = ("grad_capture", "grad_moment_cosine", "moment_persistence", "projected_grad_norm", "nonfinite_grads")

ALL_READS = {**READS, **LAB_READS}


class LabDiagnostics(Diagnostics):
    def __init__(
        self,
        roles: dict[Tensor, str] | None = None,
        role_reads: tuple[str, ...] = (),
        sample: int = 32,
        snapshot_interval: int = 0,
    ) -> None:
        super().__init__()
        unknown = set(role_reads) - set(ROLE_READS)
        if unknown:
            raise ValueError(f"role_reads must be among {ROLE_READS}, got {sorted(unknown)}")
        self.roles = roles or {}
        self.role_reads = frozenset(role_reads)
        self.sample_size = sample
        self.snapshot_interval = snapshot_interval
        self._sampled: set[Tensor] = set()
        self._updates: dict[Tensor, int] = {}
        self._snapshots: dict[Tensor, Tensor] = {}
        self._paths: dict[Tensor, Tensor] = {}
        self._param: Tensor | None = None

    def add(self, name: str, value: Tensor, count: int = 1) -> None:
        super().add(name, value, count)
        role = self.roles.get(self._param) if self._param is not None else None
        if role is not None and name in self.role_reads:
            super().add(f"{name}/{role}", value, count)
            if name in self._totals:
                self._totals.add(f"{name}/{role}")

    def add_total(self, name: str, value: Tensor) -> None:
        self._totals.add(name)
        self.add(name, value)

    def gradient(self, param: Tensor, projected_norm: Tensor, raw_norm: Tensor) -> None:
        self._param = param
        super().gradient(param, projected_norm, raw_norm)
        self._param = None

    def moment(self, param: Tensor, unit_grad: Tensor, before: Tensor, after: Tensor, beta: float) -> None:
        self._param = param
        super().moment(param, unit_grad, before, after, beta)
        if beta > 0.0:
            floor = (1.0 - beta) / (1.0 + beta)
            self.add("moment_persistence", (after.norm().square() - floor) / (1.0 - floor))
        self._param = None

    def frame_turn(self, params: list[Tensor], before: list[Tensor], turned: list) -> None:
        super().frame_turn(params, before, turned)
        for item in turned:
            spectrum = item.eigenvalues.clamp_min(0.0)
            self.add("tangent_concentration", spectrum.max() / spectrum.sum().clamp_min(1e-12))
            rotation = item.rotation
            root = math.sqrt(rotation.shape[-1])
            self.add("transport_correction", (rotation - rotation.mT).norm() / (2.0 * root))

        # The first `sample` matrices to turn, across every group.
        for param in params:
            if len(self._sampled) < self.sample_size:
                self._sampled.add(param)
        for param, old, item in zip(params, before, turned, strict=True):
            if param not in self._sampled:
                continue
            start, end = old.float(), item.frame.float()
            root = math.sqrt(end.shape[-1])
            speed = (end - start @ (start.mT @ end)).norm() / root
            self.add("transport_speed", speed)
            if self.snapshot_interval <= 0:
                continue
            path = self._paths.get(param)
            self._paths[param] = speed if path is None else path + speed
            self._updates[param] = self._updates.get(param, 0) + 1
            if self._updates[param] % self.snapshot_interval:
                continue
            previous = self._snapshots.get(param)
            self._snapshots[param] = end.clone()
            path = self._paths.pop(param)
            if previous is None:
                continue
            overlap = previous.mT @ end
            lag = (end - previous @ overlap).norm() / root
            self.add("transport_lag", lag)
            self.add("transport_curve", 1.0 - lag / path.clamp_min(1e-12))
