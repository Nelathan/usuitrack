"""Deciding which parameters UsuiTrack should own.

`ndim == 2` is a structural test, and it is not the precondition the optimizer
actually has. UsuiTrack tracks a low-rank subspace of a *shared linear map*: it
assumes the rows (or columns) of a weight are coordinates in one common space,
so that a basis fitted from a few batches means something for all of them. Two
kinds of 2D tensor satisfy the shape and violate the assumption, and both were
found the expensive way -- as a run going non-finite thousands of steps in:

* **Lookup tables** (`nn.Embedding`, and any weight whose rows are independent
  per-token vectors). There is no shared map; each row is its own vector, and
  the gradient is row-sparse, so most rows sit untouched for thousands of steps
  with their Adafactor row variance pinned at the eps floor while the basis
  only ever sees the handful of rows a batch lights up. Muon-lineage optimizers
  exclude embedding tables for the same reason.

* **Multiplicative gates** (AdaLN/FiLM modulation linears, and anything whose
  output scales or shifts another layer's output rather than feeding forward).
  These are well-conditioned, ordinary-shaped matrices; the problem is not
  numerical. A small tracking error in a gate is multiplied through everything
  downstream of it instead of staying additive and local.

Neither is detectable from shape, dtype, or gradient statistics cheaply enough
to guess at runtime, and both are named consistently within an architecture. So
the split is: this module owns the structural rule and the routing mechanics,
and the caller supplies a `RoutingPolicy` naming the weights its architecture
puts in those two categories.

The same policy carries the `side` hint, for the same reason -- it is
architecture semantics, not shape. The rule that works for transformers is
**track the residual stream**: weights that *read* the stream (attention q/k/v,
the feed-forward up-projection, the patch/token embedder's output) track their
input side, and weights that *write* it (attention out-projection, the
feed-forward down-projection, the final projection) track their output side.
That keeps every tracker's basis living in the one space the whole network
shares, where the gradient really is low-rank, instead of in a per-layer data
space that happens to be narrow. Shape-only `auto` gets this right for
square-ish weights and wrong at the edges of the network, which is where the
worst-conditioned bases show up in practice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from torch import Tensor

from .projector import ProjectionSide

__all__ = ["RoutingPolicy", "Routing", "route_parameter", "route_parameters"]


@dataclass(frozen=True)
class RoutingPolicy:
    """Architecture-specific name hints, matched as substrings of the
    parameter's qualified name.

    `exclude` wins over both side hints; a name matching neither side hint
    falls through to `ProjectionSide.AUTO`, which resolves by shape. Empty
    tuples everywhere give pure shape-based routing, which is the right
    starting point for an architecture nobody has characterized yet.
    """

    exclude: tuple[str, ...] = ()
    track_right: tuple[str, ...] = ()
    track_left: tuple[str, ...] = ()


@dataclass
class Routing:
    """The result of a routing pass, keyed the way an optimizer wants it."""

    matrix: dict[ProjectionSide, list[tuple[str, Tensor]]] = field(default_factory=dict)
    fallback: list[tuple[str, Tensor]] = field(default_factory=list)

    def describe(self) -> str:
        """A one-line-per-group summary, for logging what a policy actually did.

        Worth printing once at startup: a policy is a list of substrings, and
        the failure mode is a hint that silently matches nothing after an
        upstream rename.
        """
        parts = [
            f"{side.value}={len(entries)}"
            for side, entries in sorted(self.matrix.items(), key=lambda item: item[0].value)
        ]
        parts.append(f"fallback={len(self.fallback)}")
        return "UsuiTrack routing: " + ", ".join(parts)


def route_parameter(name: str, param: Tensor, policy: RoutingPolicy) -> ProjectionSide | None:
    """Which side UsuiTrack should track for this parameter, or None for the
    caller's fallback optimizer."""

    if param.ndim != 2:
        return None
    if any(hint in name for hint in policy.exclude):
        return None
    if any(hint in name for hint in policy.track_right):
        return ProjectionSide.RIGHT
    if any(hint in name for hint in policy.track_left):
        return ProjectionSide.LEFT
    return ProjectionSide.AUTO


def route_parameters(
    named_parameters: Iterable[tuple[str, Tensor]],
    policy: RoutingPolicy,
) -> Routing:
    """Split named parameters into UsuiTrack's matrix groups and a fallback list.

    Does not filter on `requires_grad`; callers that freeze parts of a model
    should filter before calling, so that this stays a statement about what a
    weight *is* rather than about what a particular run trains.
    """

    routing = Routing()
    for name, param in named_parameters:
        side = route_parameter(name, param, policy)
        if side is None:
            routing.fallback.append((name, param))
        else:
            routing.matrix.setdefault(side, []).append((name, param))
    return routing
