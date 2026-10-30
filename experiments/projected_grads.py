"""Weight gradients without `G` (PLAN S2, `docs/PROJECTED_ACTIVATION_2.md`).

Each tracked Linear's weight is frozen for autograd, so no `dW = delta^T X` is
ever formed. Its backward has `X` (recomputed under checkpointing) and `delta`
in hand and returns, as the "gradients" of persistent leaves, what the
optimizer consumes:

- `Z`, exact: `delta^T (X Q)` on the input side, `X^T (delta Q)` (oriented) on
  the output side;
- the frame's action, from the streams (ARCHIVE, "frames track from the
  streams"): `X^T (w^2 * X Q)` with `w_t = ||delta_t||` on the input side,
  `delta^T (delta Q)` on the output side;
- `||G P||^2` for 16 Gaussian probes `P`, whose mean over probes is an unbiased
  estimate of `||G||^2`, for the reads that divide by it.

Autograd accumulates the leaves like any gradient: no hooks and no graph
breaks, and accumulation over micro-batches is exact, since all three are sums
over tokens. The first step runs the ordinary path, so the optimizer fits each
frame from `G`; the matrices switch over after it. Lab-only: it patches the
optimizer instance, never the release.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from usuitrack import frame
from usuitrack import optimizer as usuitrack_optimizer

PROBES = 16


class _Projected(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, basis: Tensor, probes: Tensor, right: bool, z: Tensor, action: Tensor, probed: Tensor) -> Tensor:
        ctx.save_for_backward(x, weight, basis, probes)
        ctx.right = right
        return F.linear(x, weight)

    @staticmethod
    def backward(ctx, grad: Tensor):
        x, weight, basis, probes = ctx.saved_tensors
        grad_x = grad @ weight
        stream = x.reshape(-1, x.shape[-1])
        delta = grad.reshape(-1, grad.shape[-1])
        if ctx.right:
            projected_x = stream @ basis
            z = delta.mT @ projected_x
            # ||delta_t||^2 without an fp32 copy of delta; zero past the last
            # supervised token, so it also masks the rows G never sees.
            weights = torch.linalg.vector_norm(delta, dim=-1, keepdim=True, dtype=torch.float32).square()
            action = stream.mT @ (projected_x * weights).to(stream.dtype)
        else:
            projected_delta = delta @ basis
            z = stream.mT @ projected_delta
            action = delta.mT @ projected_delta
        probed = (delta.mT @ (stream @ probes)).float().square().sum(dim=0)
        return grad_x, None, None, None, None, z, action.float(), probed


def _anchor(shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> Tensor:
    """A leaf whose value is never read, only the grad autograd gives it: one
    zero expanded to `shape`, so it holds no storage of its own."""
    return torch.zeros((), dtype=dtype, device=device).expand(shape).requires_grad_()


class ProjectedLinear(torch.nn.Linear):
    """A Linear whose weight gradient arrives projected; `ProjectedGrads` swaps
    a tracked Linear's class to this once its frame is fitted."""

    basis: Tensor
    probes: Tensor
    right: bool
    z: Tensor
    action: Tensor
    probed: Tensor

    def forward(self, x: Tensor) -> Tensor:
        if not torch.is_grad_enabled():
            return F.linear(x, self.weight)
        return _Projected.apply(x, self.weight, self.basis, self.probes, self.right, self.z, self.action, self.probed)


class ProjectedGrads:
    def __init__(self, model: torch.nn.Module, optimizer: usuitrack_optimizer.UsuiTrack) -> None:
        self.optimizer = optimizer
        groups = {param: group for group in optimizer.param_groups for param in group["params"]}
        self._waiting: list[tuple[torch.nn.Linear, dict]] = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and module.weight in groups:
                if module.bias is not None:
                    raise ValueError(f"{name}: ProjectedGrads takes bias-free Linears")
                self._waiting.append((module, groups[module.weight]))
        self._projected: list[ProjectedLinear] = []
        self._probes: dict[tuple[int, torch.device], Tensor] = {}
        self._generator: torch.Generator | None = None
        original = optimizer.step

        def step(closure=None):
            loss = original(closure)
            self._switch()
            self._draw()
            return loss

        optimizer.step = step  # type: ignore[method-assign]
        # The class swap gives every compiled decoder-layer type a second
        # variant in training and in eval: past dynamo's default 8 a frame
        # falls back to eager for the rest of the run.
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 32)

    @torch.no_grad()
    def prepare(self) -> None:
        """Hand each projected matrix's accumulated leaves to the optimizer as
        its phase-one work; call after the step's backwards, before `step()`."""

        for module in self._projected:
            if module.z.grad is None:
                continue
            param = module.weight
            if param in self.optimizer._pending:
                raise RuntimeError("ProjectedGrads: a matrix was prepared twice before step()")
            z, action, probed = module.z.grad, module.action.grad, module.probed.grad
            module.z.grad = module.action.grad = module.probed.grad = None
            z_norm = torch.linalg.vector_norm(z, dtype=torch.float32)
            estimate = probed.mean().sqrt()
            raw_norm = torch.where(torch.isfinite(z_norm) & torch.isfinite(estimate), estimate, torch.nan)
            z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
            self.optimizer._pending[param] = usuitrack_optimizer.Prepared(z, frame.aim(action, module.basis), raw_norm)

    @torch.no_grad()
    def _switch(self) -> None:
        # A matrix whose frame the optimizer has fitted leaves the ordinary path.
        still = []
        for module, group in self._waiting:
            state = self.optimizer.state.get(module.weight, {})
            if "frame" not in state:
                still.append((module, group))
                continue
            weight = module.weight
            right = frame.resolve_side(weight.shape, group["side"]) == "right"
            basis = state["frame"]
            other = weight.shape[0] if right else weight.shape[1]
            module.__class__ = ProjectedLinear
            module.basis = basis
            module.right = right
            module.probes = self._probe(weight.shape[1], weight.device, weight.dtype)
            module.z = _anchor((other, basis.shape[-1]), weight.dtype, weight.device)
            module.action = _anchor(basis.shape, torch.float32, weight.device)
            module.probed = _anchor((PROBES,), torch.float32, weight.device)
            weight.requires_grad_(False)
            self._projected.append(module)
        self._waiting = still

    def _probe(self, size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        key = (size, device)
        if key not in self._probes:
            self._probes[key] = torch.empty(size, PROBES, device=device, dtype=dtype)
            self._draw()
        return self._probes[key]

    @torch.no_grad()
    def _draw(self) -> None:
        # Fresh probes each step, in place so compiled graphs keep their inputs,
        # from a private generator so training's random stream never shifts.
        for probes in self._probes.values():
            if self._generator is None:
                self._generator = torch.Generator(probes.device).manual_seed(0)
            probes.copy_(torch.randn(probes.shape, device=probes.device, generator=self._generator))
