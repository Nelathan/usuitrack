"""Lab reads on top of the release's: frame motion, persistence,
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
from usuitrack.frame import Moved

LAB_READS = {
    "moment_persistence": (
        "Coherent share of the moment once the incomplete-cancellation floor of an independent "
        "stream, w^2 = (1-beta)/(1+beta), is removed: 0 white, 1 a held direction. Lags, and a "
        "large lr drains it."
    ),
    "transport_speed": "Chordal distance per plane a sampled frame moved in one update.",
    "transport_correction": "||skew(R_s)|| / sqrt(r): the in-span rotation the moment is turned by this step.",
    "transport_lag": "Net distance per plane a sampled frame covered over the snapshot window.",
    "transport_curve": "1 - lag / path over the window: 0 straight drift, toward 1 churn in place.",
}

# Reads sampled once per matrix, which is what a role can split.
ROLE_READS = ("grad_capture", "grad_capture_probe", "grad_moment_cosine", "moment_persistence", "projected_grad_norm", "nonfinite_grads")

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
        self._indices: dict[tuple, Tensor] = {}

    def add_each(self, name: str, values: Tensor, params: list[Tensor], total: bool = False) -> None:
        super().add_each(name, values, params, total)
        if name not in self.role_reads:
            return
        by_role: dict[str, list[int]] = {}
        for member, param in enumerate(params):
            role = self.roles.get(param)
            if role is not None:
                by_role.setdefault(role, []).append(member)
        for role, members in by_role.items():
            super().add_each(f"{name}/{role}", values[self._index(members, values.device)], [params[member] for member in members], total)

    def _index(self, members: list[int], device: torch.device) -> Tensor:
        # Made once per member set, so a role split costs no host-to-device copy per step.
        key = (tuple(members), device)
        if key not in self._indices:
            self._indices[key] = torch.tensor(members, device=device)
        return self._indices[key]

    def moment(self, params: list[Tensor], unit_grad: Tensor, before: Tensor, after: Tensor, beta: float) -> None:
        super().moment(params, unit_grad, before, after, beta)
        if beta > 0.0:
            floor = (1.0 - beta) / (1.0 + beta)
            norm = torch.linalg.vector_norm(after, dim=(-2, -1))
            self.add_each("moment_persistence", (norm.square() - floor) / (1.0 - floor), params)

    def frame_move(self, params: list[Tensor], before: Tensor, moved: Moved) -> None:
        super().frame_move(params, before, moved)
        rotation = moved.rotation
        root = math.sqrt(rotation.shape[-1])
        self.add_each("transport_correction", torch.linalg.vector_norm(rotation - rotation.mT, dim=(-2, -1)) / (2.0 * root), params)

        # The first `sample` matrices to move, across every group.
        for param in params:
            if len(self._sampled) >= self.sample_size:
                break
            self._sampled.add(param)
        members = [member for member, param in enumerate(params) if param in self._sampled]
        if not members:
            return
        index = self._index(members, before.device)
        sampled = [params[member] for member in members]
        start, end = before[index].float(), moved.frame[index].float()
        speeds = torch.linalg.vector_norm(end - start @ (start.mT @ end), dim=(-2, -1)) / root
        self.add_each("transport_speed", speeds, sampled)
        if self.snapshot_interval <= 0:
            return
        for param, speed, current in zip(sampled, speeds.unbind(0), end.unbind(0), strict=True):
            path = self._paths.get(param)
            self._paths[param] = speed if path is None else path + speed
            self._updates[param] = self._updates.get(param, 0) + 1
            if self._updates[param] % self.snapshot_interval:
                continue
            previous = self._snapshots.get(param)
            self._snapshots[param] = current.clone()
            path = self._paths.pop(param)
            if previous is None:
                continue
            overlap = previous.mT @ current
            lag = (current - previous @ overlap).norm() / root
            self.add("transport_lag", lag)
            self.add("transport_curve", 1.0 - lag / path.clamp_min(1e-12))
