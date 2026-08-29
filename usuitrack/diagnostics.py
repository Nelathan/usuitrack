from __future__ import annotations

from collections import defaultdict
from typing import Mapping

import torch
from torch import Tensor


def tensor_state_bytes(state: Mapping) -> int:
    total = 0
    for value in state.values():
        if torch.is_tensor(value):
            total += value.numel() * value.element_size()
        elif isinstance(value, Mapping):
            total += tensor_state_bytes(value)
    return total


def optimizer_state_bytes_by_category(optimizer: torch.optim.Optimizer) -> dict[str, int]:
    """Return optimizer state bytes split by projected-matrix versus fallback state.

    UsuiTrack owns only projected matrix state. The ``fallback`` category is for
    any separately-owned optimizer (e.g. AdamW) the caller uses for non-matrix
    parameters.
    """

    totals: defaultdict[str, int] = defaultdict(int)
    for state in optimizer.state.values():
        state_bytes = tensor_state_bytes(state)
        totals["total"] += state_bytes
        if "projected_exp_avg" in state or "basis" in state:
            totals["matrix"] += state_bytes
        elif state:
            totals["fallback"] += state_bytes

    return {"matrix": totals["matrix"], "fallback": totals["fallback"], "total": totals["total"]}


def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    return optimizer_state_bytes_by_category(optimizer)["total"]


class DiagnosticsAccumulator:
    """Collects telemetry on-device during the step, reduces it once at read.

    The one rule this type exists to enforce: nothing here touches the host
    while the optimizer is running. Every measurement arrives as a 0-dim device
    tensor and is summed on-device; the single device-to-host transfer happens
    in ``reduce()``, which the caller runs on its own logging cadence. A stray
    ``float()`` or ``.item()`` at an accumulation site would put a full device
    sync in the middle of every training step, which is exactly the failure this
    optimizer spent a field debugging session removing from a training loop.

    Means are over samples, not over steps: a quantity measured once per matrix
    per step contributes one sample per matrix per step, so a run's mean weights
    every tracked matrix equally regardless of how many steps it appeared in.
    Counters (``bump``) are plain host-side integers, already known without a
    transfer, and are reported as totals since the last read.
    """

    def __init__(self) -> None:
        self._sums: dict[str, Tensor] = {}
        self._counts: defaultdict[str, int] = defaultdict(int)
        self._counters: defaultdict[str, int] = defaultdict(int)
        self._report_as_total: set[str] = set()

    def __bool__(self) -> bool:
        return bool(self._sums or self._counters)

    def add(self, name: str, value: Tensor, count: int = 1) -> None:
        """Add one (or ``count`` pre-summed) samples of a scalar measurement."""

        sample = value.detach().float().reshape(())
        current = self._sums.get(name)
        if current is None:
            self._sums[name] = sample
        else:
            self._sums[name] = current + sample.to(current.device)
        self._counts[name] += count

    def add_total(self, name: str, value: Tensor) -> None:
        """Accumulate an event count that only exists as a device tensor.

        Same on-device path as ``add``, reported as a running total rather than
        a mean -- for things like "how many gradients were non-finite", where
        the host cannot know the answer without the sync this type exists to
        avoid, and where an average would hide a single rare firing.
        """

        self.add(name, value)
        self._report_as_total.add(name)

    def bump(self, name: str, amount: int = 1) -> None:
        """Add to a host-side event counter (retries, failures, and the like)."""

        self._counters[name] += amount

    def reduce(self) -> dict[str, float]:
        """Mean of every accumulated measurement plus every counter total.

        One host transfer per device, regardless of how many metrics were
        collected -- the sums are stacked and moved together.
        """

        result: dict[str, float] = {}
        by_device: defaultdict[torch.device, list[str]] = defaultdict(list)
        for name, value in self._sums.items():
            by_device[value.device].append(name)
        for names in by_device.values():
            names.sort()
            values = torch.stack([self._sums[name] for name in names]).cpu().tolist()
            for name, value in zip(names, values, strict=True):
                if name in self._report_as_total:
                    result[name] = value
                    continue
                count = self._counts[name]
                result[name] = value / count if count else float("nan")
        for name, count in self._counters.items():
            result[name] = float(count)
        return result
