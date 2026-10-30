"""Frames tracked from the streams around each matrix, read against `G` (PLAN S4).

`G = delta^T X`: the input side's span lies in `X`'s, the output side's in
`delta`'s. `StreamFrames` turns shadow frames by the optimizer's Oja aim and
constant turn on streams instead of `G`, never touching training, and reads
each against the gradient's own frame. Shadows start as copies of the
optimizer's frame after its fit, so a read says whether a stream holds the
gradient's span, not how fast it finds it. Variants:

- output side, `delta`: the action `delta^T delta Q`.
- input side, `x`: `X^T X Q`, S3's activation frame.
- input side, `weighted`: `X` with each token's row scaled by `||delta_t||`,
  the diagonal of `delta delta^T` in `G^T G = X^T delta delta^T X`.
- input side, `agreement`: `delta delta^T` replaced by its projection on a
  second shadow `Q_o` tracking `delta` on the matrix's output side; the action
  is `X^T (delta Q_o) C` with the `[r, r]` core `C = (delta Q_o)^T (X Q)`, the
  gradient seen through both frames. Exact when `delta` lies in `Q_o`.
- input side, `sum`: the trace-normalized Grams of `X` and of the residual
  gradient `g` at the norm that produced `X`, both in `d_model` coordinates,
  added.

Streams are taken in backward, where checkpointing recomputes `X` just before
the matrix's own `delta` arrives; nothing is kept across layers. Every action is
a chain of skinny GEMMs in the streams' own dtype, accumulated in fp32: no fp32
copy of a `[tokens, d]` stream and no `[d, d]` Gram, which would not fit beside
an LFM3 backward. Lab-only: it wraps the optimizer instance's `_prepare`, never
the release.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import torch
from torch import Tensor

from usuitrack import frame
from usuitrack import optimizer as usuitrack_optimizer

INPUT_VARIANTS = ("x", "weighted", "agreement", "sum")
OUTPUT_VARIANTS = ("delta",)

STREAM_READS = {
    "grad_capture_input": "||G Q|| / ||G|| over input-side matrices: the gradient frame's own capture there.",
    "grad_capture_output": "||Q^T G|| / ||G|| over output-side matrices.",
    **{f"stream_{v}_overlap": f"||Q^T Q_{v}||^2 / r: the `{v}` frame against the gradient's; 1 is the same span." for v in INPUT_VARIANTS + OUTPUT_VARIANTS},
    **{f"stream_{v}_grad_capture": f"||G Q_{v}|| / ||G|| (oriented): the gradient share the `{v}` frame holds." for v in INPUT_VARIANTS + OUTPUT_VARIANTS},
}


class _Tap(torch.autograd.Function):
    """Identity whose backward hands the incoming gradient to `sink`."""

    @staticmethod
    def forward(ctx, value: Tensor, sink: Callable[[Tensor], None]) -> Tensor:
        ctx.sink = sink
        return value.view_as(value)

    @staticmethod
    def backward(ctx, grad: Tensor) -> tuple[Tensor, None]:
        ctx.sink(grad)
        return grad, None


def _action(stream: Tensor, shadow: Tensor) -> Tensor:
    """`S^T S Q` as two skinny GEMMs, fp32 out."""

    return (stream.mT @ (stream @ shadow.to(stream.dtype))).float()


def _subspace(stream: Tensor, rank: int) -> Tensor:
    """The top-`rank` span of `S^T S` by subspace iteration from a random start:
    `frame.fit` without the `[d, d]` Gram. Its own generator, so the shadow never
    shifts the training run's random stream."""

    generator = torch.Generator(stream.device).manual_seed(0)
    q = torch.randn(stream.shape[-1], rank, device=stream.device, generator=generator)
    for _ in range(6):
        q = torch.linalg.qr(_action(stream, q)).Q
    return q


def _in_backward() -> bool:
    # Checkpointing recomputes the forward inside the backward graph task.
    return torch._C._current_graph_task_id() != -1


def lfm_sites(model: torch.nn.Module) -> dict[str, str]:
    """Linear name -> its block's input norm, for LFM2/LFM3 decoder layers: the
    operator (conv or attention) block's is `operator_norm`, the MLP's
    `ffn_norm`. Only input-side matrices consult it, and those read that norm's
    output; output-side entries are never used."""

    sites = {}
    for name, module in model.named_modules():
        match = re.fullmatch(r"(.*\.layers\.\d+)\.(conv|self_attn|feed_forward)\.\w+", name)
        if match and isinstance(module, torch.nn.Linear):
            norm = "ffn_norm" if match.group(2) == "feed_forward" else "operator_norm"
            sites[name] = f"{match.group(1)}.{norm}"
    return sites


class StreamFrames:
    def __init__(self, model: torch.nn.Module, optimizer: usuitrack_optimizer.UsuiTrack, sites: dict[str, str]) -> None:
        self.optimizer = optimizer
        self.mask: Tensor | None = None
        self._side = {param: group["side"] for group in optimizer.param_groups for param in group["params"]}
        self._shadows: dict[tuple[Tensor, str], Tensor] = {}
        self._tangents: dict[Tensor, dict[str, Tensor]] = {}
        self._x: dict[Tensor, Tensor] = {}
        self._site_x: dict[str, Tensor] = {}
        self._site_params: dict[str, list[Tensor]] = {}
        modules = dict(model.named_modules())
        for name, module in modules.items():
            if not isinstance(module, torch.nn.Linear) or module.weight not in self._side:
                continue
            side = self._side[module.weight]
            if side not in ("left", "right"):
                raise ValueError(f"{name}: StreamFrames needs an explicit side, got {side!r}")
            if side == "right":
                site = sites.get(name)
                if site is None:
                    raise ValueError(f"{name}: an input-side matrix with no norm site")
                self._site_params.setdefault(site, []).append(module.weight)
                module.register_forward_pre_hook(lambda m, args, site=site: self._take_x(m.weight, site, args[0]))
            module.register_forward_hook(lambda m, args, out: self._tap(out, lambda grad, w=m.weight: self._on_delta(w, grad)))
        for site in self._site_params:
            modules[site].register_forward_pre_hook(lambda m, args, site=site: (self._tap(args[0], lambda grad: self._on_residual(site, grad)), *args[1:]))
        original = optimizer._prepare

        def prepare(param: Tensor) -> None:
            tangents = self._tangents.pop(param, None)
            if tangents is not None:
                self._read(param)
            original(param)
            if tangents is not None:
                self._turn(param, tangents)
            elif param in self._side and (param, "x" if self._side[param] == "right" else "delta") not in self._shadows:
                self._start(param)

        optimizer._prepare = prepare  # type: ignore[method-assign]

    def set_batch(self, labels: Tensor) -> None:
        """As `ActivationFrames.set_batch`: rows before a sequence's last
        supervised label; anything a backward left unconsumed is dropped."""

        self._tangents.clear()
        self._x.clear()
        self._site_x.clear()
        positions = torch.arange(labels.shape[-1], device=labels.device)
        last = torch.where(labels != -100, positions, -1).amax(dim=-1, keepdim=True)
        self.mask = (positions < last).reshape(-1, 1)

    @torch.compiler.disable
    def _tap(self, value: Tensor, sink: Callable[[Tensor], None]) -> Tensor:
        # Tapped on the recompute too, so it rebuilds the graph it replays; only
        # the first forward's tap is ever backpropagated.
        if not torch.is_grad_enabled():
            return value
        return _Tap.apply(value, sink)

    @torch.compiler.disable
    @torch.no_grad()
    def _take_x(self, param: Tensor, site: str, inputs: Tensor) -> None:
        if not _in_backward():
            return
        # One masked copy per site: q/k/v and gate/up read the same `X`.
        x = self._site_x.get(site)
        if x is None:
            x = inputs.detach().reshape(-1, inputs.shape[-1])
            if self.mask is not None:
                x = x * self.mask.to(x.dtype)
            self._site_x[site] = x
        self._x[param] = x

    def _start(self, param: Tensor) -> None:
        stored = self.optimizer.state.get(param, {}).get("frame")
        if stored is None:
            return
        variants = INPUT_VARIANTS if self._side[param] == "right" else OUTPUT_VARIANTS
        for variant in variants:
            self._shadows[(param, variant)] = stored.clone()

    @torch.no_grad()
    def _on_delta(self, param: Tensor, grad: Tensor) -> None:
        if (param, "x" if self._side[param] == "right" else "delta") not in self._shadows:
            return
        delta = grad.reshape(-1, grad.shape[-1])
        tangents = self._tangents.setdefault(param, {})
        if self._side[param] == "left":
            shadow = self._shadows[(param, "delta")]
            tangents["delta"] = frame.aim(_action(delta, shadow), shadow)
            return
        x = self._x.pop(param, None)
        if x is None:
            return
        shadow = self._shadows[(param, "x")]
        tangents["x"] = frame.aim(_action(x, shadow), shadow)
        # X^T diag(||delta_t||^2) X Q, weighting the projection instead of X.
        shadow = self._shadows[(param, "weighted")]
        weights = delta.norm(dim=1, keepdim=True).float().square()
        tangents["weighted"] = frame.aim((x.mT @ ((x @ shadow) * weights).to(x.dtype)).float(), shadow)
        far = self._shadows.get((param, "far"))
        if far is None:
            far = self._shadows[(param, "far")] = _subspace(delta, shadow.shape[-1]).to(param.dtype)
        projected_far = delta @ far
        tangents["far"] = frame.aim((delta.mT @ projected_far).float(), far)
        agreement = self._shadows[(param, "agreement")]
        core = projected_far.float().mT @ (x @ agreement).float()
        tangents["agreement"] = frame.aim((x.mT @ (projected_far.float() @ core).to(x.dtype)).float(), agreement)

    @torch.no_grad()
    def _on_residual(self, site: str, grad: Tensor) -> None:
        x = self._site_x.pop(site, None)
        if x is None:
            return
        g = grad.reshape(-1, grad.shape[-1])
        x_energy = x.norm().float().square().clamp_min(1e-12)
        g_energy = g.norm().float().square().clamp_min(1e-12)
        params = [param for param in self._site_params[site] if (param, "sum") in self._shadows]
        frames = [self._shadows[(param, "sum")] for param in params]
        tangents = []
        for shadow in frames:
            action = _action(x, shadow) / x_energy + _action(g, shadow) / g_energy
            tangents.append(frame.aim(action, shadow))
        # The site's matrices were read at their own `_prepare`, earlier in this
        # backward, so the sum frames turn here, after their reads.
        for shadow, tangent, spectrum in zip(frames, tangents, frame.tangent_spectra(tangents), strict=True):
            shadow.copy_(frame.turn(shadow, tangent, spectrum, usuitrack_optimizer.GEODESIC_STEPSIZE).frame)

    @torch.no_grad()
    def _read(self, param: Tensor) -> None:
        diagnostics = self.optimizer.diagnostics
        stored = self.optimizer.state.get(param, {}).get("frame")
        if diagnostics is None or stored is None:
            return
        side = self._side[param]
        grad = frame.orient(torch.nan_to_num(param.grad.float()), side)
        grad_norm = grad.norm().clamp_min(1e-12)
        q = stored.float()
        diagnostics.add("grad_capture_input" if side == "right" else "grad_capture_output", (grad @ q).norm() / grad_norm)
        for variant in INPUT_VARIANTS if side == "right" else OUTPUT_VARIANTS:
            shadow = self._shadows[(param, variant)].float()
            diagnostics.add(f"stream_{variant}_grad_capture", (grad @ shadow).norm() / grad_norm)
            diagnostics.add(f"stream_{variant}_overlap", (q.mT @ shadow).square().sum() / q.shape[-1])

    @torch.no_grad()
    def _turn(self, param: Tensor, tangents: dict[str, Tensor]) -> None:
        names = list(tangents)
        frames = [self._shadows[(param, name)] for name in names]
        spectra = frame.tangent_spectra([tangents[name] for name in names])
        for shadow, name, spectrum in zip(frames, names, spectra, strict=True):
            shadow.copy_(frame.turn(shadow, tangents[name], spectrum, usuitrack_optimizer.GEODESIC_STEPSIZE).frame)
