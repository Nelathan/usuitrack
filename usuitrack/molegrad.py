"""Weight gradients without `G`: each tracked Linear's backward returns `Z`,
the diagonal aim from its input and output gradient, and a probe sketch of
`G` (`docs/SPEC.md`, "Molegrad").
"""

from __future__ import annotations

import collections
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor

from . import frame
from .optimizer import Prepared, UsuiTrack

PROBES = 16


class _Projected(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor | None, basis: Tensor, probes: Tensor, right: bool, fit: bool, z: Tensor, action: Tensor, probed: Tensor) -> Tensor:
        ctx.save_for_backward(x, weight, basis, probes)
        ctx.right = right
        ctx.fit = fit
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad: Tensor):
        x, weight, basis, probes = ctx.saved_tensors
        grad_x = grad @ weight
        stream = x.reshape(-1, x.shape[-1])
        delta = grad.reshape(-1, grad.shape[-1])
        grad_bias = delta.sum(dim=0) if ctx.needs_input_grad[2] else None
        fitted = None
        if ctx.fit:
            gradient = delta.mT @ stream if ctx.right else stream.mT @ delta
            gradient = torch.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
            basis = fitted = frame.fit(gradient, basis.shape[-1])
        rank = basis.shape[-1]
        # The probes ride with the frame in one GEMM over each stream: [Q | P].
        if ctx.right:
            projected_x = stream @ torch.cat([basis, probes], dim=1)
            z_probed = delta.mT @ projected_x
            z, probed = z_probed[:, :rank], z_probed[:, rank:]
            # ||delta_t||^2 without an fp32 copy of delta; zero past the last
            # supervised token, so it also masks the rows G never sees.
            weights = torch.linalg.vector_norm(delta, dim=-1, keepdim=True, dtype=torch.float32).square()
            action = stream.mT @ (projected_x[:, :rank] * weights).to(stream.dtype)
        else:
            projected_delta = delta @ basis
            z = stream.mT @ projected_delta
            action_probed = delta.mT @ torch.cat([projected_delta, stream @ probes], dim=1)
            action, probed = action_probed[:, :rank], action_probed[:, rank:]
        return grad_x, None, grad_bias, fitted, None, None, None, z, action.float(), probed.float()


def _anchor(shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> Tensor:
    """A leaf for a grad only: its value is never read, so it holds no storage."""
    return torch.zeros((), dtype=dtype, device=device).expand(shape).requires_grad_()


class MoleLinear(torch.nn.Linear):
    """A Linear whose weight gradient arrives projected; `Molegrad` swaps
    every tracked Linear's class to this."""

    basis: Tensor
    probes: Tensor
    right: bool
    z: Tensor
    action: Tensor
    probed: Tensor

    def forward(self, x: Tensor) -> Tensor:
        if not torch.is_grad_enabled():
            return F.linear(x, self.weight, self.bias)
        # A basis that still requires grad is waiting for its fit.
        return _Projected.apply(x, self.weight, self.bias, self.basis, self.probes, self.right, self.basis.requires_grad, self.z, self.action, self.probed)


class Molegrad:
    """Trains `optimizer`'s matrices from projections, never forming `G`; build
    it after the optimizer, before the first backward."""

    def __init__(self, model: torch.nn.Module, optimizer: UsuiTrack) -> None:
        self.optimizer = optimizer
        groups = {param: group for group in optimizer.param_groups for param in group["params"]}
        self._projected: list[MoleLinear] = []
        self._fitted: set[Tensor] = set()
        self._probes: dict[tuple[int, torch.device], Tensor] = {}
        self._generators: dict[torch.device, torch.Generator] = {}
        if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            raise ValueError("Molegrad: the projected gradients are not all-reduced across processes")
        owners: dict[Tensor, list[tuple[str, torch.nn.Module]]] = collections.defaultdict(list)
        for name, module in model.named_modules():
            for param in module.parameters(recurse=False):
                owners[param].append((name, module))
        for param in groups:
            if len(owners[param]) > 1:
                (name, _module), (other, _other_module) = owners[param][:2]
                raise ValueError(f"{name}: its weight is shared with {other}, which would get no gradient")
        for param in list(groups):
            for name, module in owners[param]:
                if isinstance(module, torch.nn.Linear) and module.weight is param:
                    if type(module) is not torch.nn.Linear:
                        raise ValueError(f"{name}: {type(module).__name__} has its own forward, which projecting would replace")
                    self._project(module, groups.pop(param))
        if groups:
            raise ValueError(f"Molegrad: {len(groups)} tracked matrices are not Linears and would still form G")
        optimizer.register_load_state_dict_post_hook(self._after_load)
        optimizer.register_step_pre_hook(self._before_step)
        # torch has no zero_grad hook, so the optimizer's own is wrapped.
        self._zero_grad = optimizer.zero_grad
        optimizer.zero_grad = self._discard_then_zero_grad

    def _before_step(self, optimizer: UsuiTrack, args: tuple, kwargs: dict) -> None:
        # `args` starts with the optimizer itself.
        closure = args[1] if len(args) > 1 else kwargs.get("closure")
        if closure is not None:
            raise ValueError("Molegrad: a closure's backward would run after step() collected the gradients; call backward() before step()")
        self.prepare()

    def _after_load(self, optimizer: UsuiTrack) -> None:
        self._discard()
        self._bind()

    def _discard_then_zero_grad(self, set_to_none: bool = True) -> None:
        dropped = self._discard()
        if dropped:
            warnings.warn(f"Molegrad: zero_grad() dropped the projected gradients of {dropped} matrices before step() applied them.", stacklevel=2)
        self._zero_grad(set_to_none)

    def _discard(self) -> int:
        """Drop leaves no step consumed; a frame they fitted stays, as on the `G` path."""

        dropped = 0
        for module in self._projected:
            dropped += module.z.grad is not None
            module.z.grad = module.action.grad = module.probed.grad = None
        self._fitted.clear()
        return dropped

    @torch.no_grad()
    def prepare(self) -> None:
        """Hand each projected matrix's accumulated leaves to the optimizer as
        its phase-one work. `step()` calls it; calling it earlier exposes the
        pending raw norms, and the step's own call is then a no-op."""

        prepared = False
        for module in self._projected:
            if module.z.grad is None:
                continue
            param = module.weight
            if param in self.optimizer._pending:
                raise RuntimeError("Molegrad: a matrix was prepared twice before step()")
            z, action, probed = module.z.grad, module.action.grad, module.probed.grad
            module.z.grad = module.action.grad = module.probed.grad = None
            z_norm = torch.linalg.vector_norm(z, dtype=torch.float32)
            estimate = probed.square().sum(dim=0).mean().sqrt()
            raw_norm = torch.where(torch.isfinite(z_norm) & torch.isfinite(estimate), estimate, torch.nan)
            z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
            tangent = None if param in self._fitted else frame.aim(action, module.basis)
            self.optimizer._pending[param] = Prepared(z, tangent, raw_norm)
            prepared = True
        if prepared:
            self._fitted.clear()
            self._draw()

    def _project(self, module: torch.nn.Linear, group: dict) -> None:
        weight = module.weight
        right = frame.resolve_side(weight.shape, group["side"]) == "right"
        rank = frame.rank(weight.shape, group["rank_fraction"])
        length, other = (weight.shape[1], weight.shape[0]) if right else (weight.shape[0], weight.shape[1])
        module.__class__ = MoleLinear
        module.right = right
        module.basis = torch.zeros(length, rank, dtype=weight.dtype, device=weight.device)
        module.probes = self._probe(weight.shape[1], weight.device, weight.dtype)
        module.z = _anchor((other, rank), weight.dtype, weight.device)
        module.action = _anchor((length, rank), torch.float32, weight.device)
        module.probed = _anchor((weight.shape[0], PROBES), torch.float32, weight.device)
        weight.requires_grad_(False)
        self._projected.append(module)
        self._bind_module(module)

    def _bind(self) -> None:
        for module in self._projected:
            self._bind_module(module)

    def _bind_module(self, module: MoleLinear) -> None:
        """The frame lives only in optimizer state, which a load replaces: the
        module reads it from there, or holds a leaf that waits for its fit."""

        weight = module.weight
        held = self.optimizer.state[weight].get("frame")
        if held is not None:
            module.basis = held
            return
        basis = torch.zeros_like(module.basis, requires_grad=True)
        basis.register_post_accumulate_grad_hook(lambda basis: self._store_fit(weight, basis))
        module.basis = basis

    @torch.no_grad()
    def _store_fit(self, weight: Tensor, basis: Tensor) -> None:
        # Each call of the Linear fits its own frame, and autograd sums them.
        gram = basis.grad.float().mT @ basis.grad.float()
        if not torch.allclose(gram, torch.eye(gram.shape[0], device=gram.device), atol=0.1):
            raise RuntimeError("Molegrad: a Linear had more than one call in the backward that fits its frame; call it once until its first step")
        basis.copy_(basis.grad)
        basis.grad = None
        basis.requires_grad_(False)
        self.optimizer.state[weight]["frame"] = basis
        self._fitted.add(weight)

    def _probe(self, size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        key = (size, device, dtype)
        if key not in self._probes:
            self._probes[key] = torch.empty(size, PROBES, device=device, dtype=dtype)
            self._draw()
        return self._probes[key]

    @torch.no_grad()
    def _draw(self) -> None:
        # Fresh probes each step, in place so compiled graphs keep their inputs,
        # from a private generator so training's random stream never shifts.
        for probes in self._probes.values():
            if probes.device not in self._generators:
                self._generators[probes.device] = torch.Generator(probes.device).manual_seed(0)
            generator = self._generators[probes.device]
            probes.copy_(torch.randn(probes.shape, device=probes.device, generator=generator))
