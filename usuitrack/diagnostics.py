"""Opt-in reads of what the optimizer is doing, accumulated on-device.

Nothing here touches the host while the optimizer runs: every read arrives as
a 0-dim device tensor and is summed there, and `pop` makes the one transfer,
on the caller's logging cadence. A drained value is the mean over every
matrix and step since the last `pop`; `nonfinite_grads` is a total.

`READS` says how to read each key. A subclass adds reads by overriding the
event methods, which the optimizer calls only while diagnostics are on, once per
block step: `params` are the block's members that stepped, and each tensor
carries them on its leading dimension.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import torch
from torch import Tensor

from .frame import Moved

READS = {
    "relative_step": (
        "||dW|| / ||W|| over all matrices, before rounding. Equals lr when every matrix "
        "stepped and every moment fills its rank; a matrix whose gradient carries k of r "
        "planes steps at sqrt(k/r) of lr and pulls the read down. A fixed-rule group steps "
        "lr sqrt(max(1, m/n) r) in norm whatever its weight, so it reads that over its ||W||."
    ),
    "grad_capture": (
        "||G Q|| / ||G||: the share of the gradient the tracked frame holds. Falls when "
        "the rank is too small or the frame lags a moving gradient. Compare runs at equal rank_fraction. "
        "A matrix in its full space reads 1. "
        "A non-finite gradient counts as zero."
    ),
    "grad_capture_probe": (
        "grad_capture under Molegrad, over the probe sketch's estimate of ||G||: a ratio of an exact "
        "||G Q|| to a noisy norm, so it can read above 1 (4-11% spread per matrix and step). A guide "
        "for sizing rank_fraction, not a measurement of the frame."
    ),
    "grad_moment_cosine": (
        "Cosine of this step's projected gradient against the moment before it. Positive: "
        "steps under-travel and the gradient keeps its direction. Near zero: critically damped. "
        "Negative: overshoot. Noisy; read its trend against lr."
    ),
    "projected_grad_norm": (
        "||G Q||, per matrix. The step ignores gradient magnitude, so bursts do not reach "
        "it, but they mark outlier batches that a fallback optimizer does see."
    ),
    "nonfinite_grads": (
        "Matrix gradients that arrived with NaN or Inf since the last read. Those entries "
        "are zeroed and the step proceeds; any count is worth finding upstream."
    ),
}


class Diagnostics:
    """The release reads, `READS`."""

    def __init__(self) -> None:
        self._sums: dict[str, Tensor] = {}
        self._counts: defaultdict[str, int] = defaultdict(int)
        self._totals: set[str] = set()
        self._step_update_sq: Tensor | None = None

    def add(self, name: str, value: Tensor, count: int = 1) -> None:
        """One sample, or `count` pre-summed samples, of a mean-reported read."""

        sample = value.detach().float().reshape(())
        current = self._sums.get(name)
        self._sums[name] = sample if current is None else current + sample.to(current.device)
        self._counts[name] += count

    def add_each(self, name: str, values: Tensor, params: list[Tensor], total: bool = False) -> None:
        """One sample per matrix of `params`, `values` `[b]`; a `total` is a count
        reported as a total, which a mean over a quiet window would hide."""

        self.add(name, values.sum(), count=len(params))
        if total:
            self._totals.add(name)

    def gradient(self, params: list[Tensor], projected_norm: Tensor, raw_norm: Tensor, probed: bool) -> None:
        self.add_each("projected_grad_norm", projected_norm, params)
        # A non-finite gradient reads as zero capture rather than taking the
        # whole window's mean to NaN; `nonfinite_grads` flags that window.
        finite = torch.isfinite(raw_norm)
        capture = projected_norm / raw_norm.clamp_min(1e-12)
        name = "grad_capture_probe" if probed else "grad_capture"
        self.add_each(name, torch.where(finite, capture, 0.0), params)
        self.add_each("nonfinite_grads", (~finite).float(), params, total=True)

    def moment(self, params: list[Tensor], unit_grad: Tensor, before: Tensor, after: Tensor, beta: float) -> None:
        """`before` is the moment as it stood ahead of this step's blend, so the
        cosine does not measure the gradient against itself."""

        # `unit_grad` has norm one up to the clamp, or is zero for a zero gradient.
        dims = (-2, -1)
        cosine = (unit_grad * before).sum(dim=dims) / torch.linalg.vector_norm(before, dim=dims).clamp_min(1e-12)
        self.add_each("grad_moment_cosine", cosine, params)

    def frame_move(self, params: list[Tensor], before: Tensor, moved: Moved) -> None:
        """Per frame update: the stored frames before, and `Moved` after; a hook the release reads nothing from."""

    def update(self, params: list[Tensor], update_norm: Tensor) -> None:
        square = update_norm.square().sum()
        if self._step_update_sq is not None:
            square = self._step_update_sq + square.to(self._step_update_sq.device)
        self._step_update_sq = square

    def step_end(self) -> None:
        # One sample per step: the whole step's motion, rooted once.
        if self._step_update_sq is not None:
            self.add("update_norm", self._step_update_sq.sqrt())
            self._step_update_sq = None

    @torch.no_grad()
    def pop(self, params: Iterable[Tensor]) -> dict[str, float]:
        """Every read since the last call, as floats; empty if nothing ran."""

        if not self._sums:
            return {}
        if "update_norm" in self._sums:
            # Joins the sums so it crosses with them, in the same transfer.
            for param in params:
                self.add("param_norm_sq", param.detach().float().square().sum(), count=0)
        names = sorted(self._sums)
        by_device: defaultdict[torch.device, list[str]] = defaultdict(list)
        for name in names:
            by_device[self._sums[name].device].append(name)
        result: dict[str, float] = {}
        for device_names in by_device.values():
            values = torch.stack([self._sums[name] for name in device_names]).cpu().tolist()
            result.update(zip(device_names, values, strict=True))
        param_norm = result.pop("param_norm_sq", 0.0) ** 0.5
        for name in result:
            if name not in self._totals:
                result[name] = result[name] / self._counts[name]
        update_norm = result.pop("update_norm", None)
        if update_norm is not None:
            result["relative_step"] = update_norm / param_norm if param_norm > 0 else float("nan")
        self._sums.clear()
        self._counts.clear()
        return result
