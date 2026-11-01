"""Opt-in reads of what the optimizer is doing, accumulated on-device.

Nothing here touches the host while the optimizer runs: every read arrives as
a 0-dim device tensor and is summed there, and `pop` makes the one transfer,
on the caller's logging cadence. A drained value is the mean over every
matrix and step since the last `pop`; `nonfinite_grads` is a total.

`READS` says how to read each key. A subclass adds reads by overriding the
event methods, which the optimizer calls only while diagnostics are on.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import torch
from torch import Tensor

READS = {
    "relative_step": (
        "||dW|| / ||W|| over all matrices, before rounding. Equals lr when every matrix "
        "stepped and every moment fills its rank; a matrix whose gradient carries k of r "
        "planes steps at sqrt(k/r) of lr and pulls the read down."
    ),
    "grad_capture": (
        "||G Q|| / ||G||: the share of the gradient the tracked frame holds. Falls when "
        "the rank is too small or the frame lags a moving gradient. Compare runs at equal rank_fraction. "
        "A non-finite gradient counts as zero."
    ),
    "grad_moment_cosine": (
        "Cosine of this step's projected gradient against the moment before it. Positive: "
        "steps under-travel and the gradient keeps its direction. Near zero: critically damped. "
        "Negative: overshoot. Noisy; read its trend against lr."
    ),
    "tangent_live_fraction": (
        "Share of tracked planes the frame update can resolve above float noise. Below 1, "
        "rank_fraction asks for more planes than the gradient carries at this batch size: "
        "lower the rank or raise the batch."
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

    def add_total(self, name: str, value: Tensor) -> None:
        """A count reported as a total, which a mean over a quiet window would hide."""

        self.add(name, value)
        self._totals.add(name)

    def gradient(self, param: Tensor, projected_norm: Tensor, raw_norm: Tensor) -> None:
        self.add("projected_grad_norm", projected_norm)
        # A non-finite gradient reads as zero capture rather than taking the
        # whole window's mean to NaN; `nonfinite_grads` flags that window.
        capture = projected_norm / raw_norm.clamp_min(1e-12)
        self.add("grad_capture", torch.where(torch.isfinite(raw_norm), capture, 0.0))
        self.add_total("nonfinite_grads", (~torch.isfinite(raw_norm)).float())

    def moment(self, param: Tensor, unit_grad: Tensor, before: Tensor, after: Tensor, beta: float) -> None:
        """`before` is the moment as it stood ahead of this step's blend, so the
        cosine does not measure the gradient against itself."""

        cosine = (unit_grad * before).sum() / (unit_grad.norm() * before.norm()).clamp_min(1e-12)
        self.add("grad_moment_cosine", cosine)

    def frame_turn(self, params: list[Tensor], before: list[Tensor], turned: list) -> None:
        """Per basis update: the stored frames before, and `frame.Turned` after."""

        for item in turned:
            self.add("tangent_live_fraction", item.live.float().mean())

    def update(self, param: Tensor, update_norm: Tensor) -> None:
        square = update_norm.square()
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
