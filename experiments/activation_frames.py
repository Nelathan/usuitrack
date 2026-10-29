"""Input-side frames tracked from activations (PLAN S3).

On the input side `G = delta^T X`, so the gradient's input span lies in the
activations'. A forward pre-hook on each input-side matrix takes the action
`X^T X Q`, and `ActivationFrames` either

- `shadow`: turns a second frame `Q_x` by the same Oja aim and constant turn
  as the optimizer's, never touching training, and reads both frames against
  both `G` and `X`; or
- `aim`: hands the optimizer the tangent from `X` in place of the one from
  `G`, so the frame follows the activations while the step still projects `G`.

Lab-only: it patches the optimizer instance, never the release. Rows of `X`
past a sequence's last supervised token are zeroed, since `delta` is exactly
zero there and `G` never sees them.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from usuitrack import frame
from usuitrack import optimizer as usuitrack_optimizer

MODES = ("off", "shadow", "aim")

ACTIVATION_READS = {
    "xframe_overlap": "||Q^T Q_x||^2 / r on input-side matrices: 1 is the same span, r/d chance. Shadow only.",
    "grad_capture_input": "||G Q|| / ||G|| over input-side matrices only: grad_capture where an activation frame applies.",
    "grad_capture_xframe": "||G Q_x|| / ||G||: the gradient share the activation frame would hold. Shadow only.",
    "act_capture_input": "||X Q|| / ||X||: the share of the (masked) activations the optimizer's frame holds.",
    "act_capture_xframe": "||X Q_x|| / ||X||: the share of the activations their own frame holds. Shadow only.",
}


class ActivationFrames:
    def __init__(self, model: torch.nn.Module, optimizer: usuitrack_optimizer.UsuiTrack, mode: Literal["shadow", "aim"]) -> None:
        if mode not in ("shadow", "aim"):
            raise ValueError(f"mode must be shadow or aim, got {mode!r}")
        self.mode = mode
        self.optimizer = optimizer
        self.mask: Tensor | None = None
        self._shadows: dict[Tensor, Tensor] = {}
        self._captures: dict[Tensor, dict[str, Tensor | None]] = {}
        right = {param for group in optimizer.param_groups if group["side"] == "right" for param in group["params"]}
        for module in model.modules():
            if isinstance(module, torch.nn.Linear) and module.weight in right:
                module.register_forward_pre_hook(self._hook)
        original = optimizer._prepare

        def prepare(param: Tensor) -> None:
            capture = self._captures.pop(param, None)
            if capture is not None:
                self._read_gradient(param, capture)
            original(param)
            if capture is not None:
                self._apply(param, capture)

        optimizer._prepare = prepare  # type: ignore[method-assign]

    def set_batch(self, labels: Tensor) -> None:
        """Mark the rows `G` can see: every position before a sequence's last
        supervised label (a causal model's earlier tokens feed it). A capture
        no `_prepare` consumed (a frozen matrix, a failed backward) is dropped,
        so the next G is never paired with an old X."""

        self._captures.clear()
        positions = torch.arange(labels.shape[-1], device=labels.device)
        last = torch.where(labels != -100, positions, -1).amax(dim=-1, keepdim=True)
        self.mask = (positions < last).reshape(-1, 1)

    def _hook(self, module: torch.nn.Module, args: tuple) -> None:
        # Eval runs under no_grad, and G never sees it.
        if torch.is_grad_enabled():
            self._capture(module.weight, args[0])

    @torch.compiler.disable
    @torch.no_grad()
    def _capture(self, param: Tensor, inputs: Tensor) -> None:
        # The checkpoint recompute repeats the forward's `X`.
        if param in self._captures:
            return
        state = self.optimizer.state.get(param, {})
        stored = state.get("frame")
        x = inputs.detach().reshape(-1, inputs.shape[-1])
        if self.mask is not None:
            x = x * self.mask.to(x.dtype)
        x_norm = torch.linalg.vector_norm(x, dtype=torch.float32).clamp_min(1e-12)
        capture: dict[str, Tensor | None] = {"tangent": None}
        if stored is not None:
            projected, tangent = frame.project_and_aim(x, stored)
            capture["act_capture"] = projected.float().norm() / x_norm
            if self.mode == "aim":
                capture["tangent"] = tangent
        if self.mode == "shadow":
            shadow = self._shadows.get(param)
            if shadow is None:
                group = self.optimizer._group_of[param]
                self._shadows[param] = frame.fit(x, frame.rank(param.shape, group["rank_fraction"])).to(param.dtype)
            else:
                projected, tangent = frame.project_and_aim(x, shadow)
                capture["act_capture_xframe"] = projected.float().norm() / x_norm
                capture["tangent"] = tangent
        self._captures[param] = capture

    @torch.no_grad()
    def _read_gradient(self, param: Tensor, capture: dict[str, Tensor | None]) -> None:
        diagnostics = self.optimizer.diagnostics
        stored = self.optimizer.state.get(param, {}).get("frame")
        if diagnostics is None or stored is None:
            return
        grad = torch.nan_to_num(param.grad.float())
        grad_norm = grad.norm().clamp_min(1e-12)
        q = stored.float()
        diagnostics.add("grad_capture_input", (grad @ q).norm() / grad_norm)
        diagnostics.add("act_capture_input", capture["act_capture"])
        shadow = self._shadows.get(param)
        if self.mode == "shadow" and shadow is not None and "act_capture_xframe" in capture:
            q_x = shadow.float()
            diagnostics.add("grad_capture_xframe", (grad @ q_x).norm() / grad_norm)
            diagnostics.add("act_capture_xframe", capture["act_capture_xframe"])
            diagnostics.add("xframe_overlap", (q.mT @ q_x).square().sum() / q.shape[-1])

    @torch.no_grad()
    def _apply(self, param: Tensor, capture: dict[str, Tensor | None]) -> None:
        tangent = capture["tangent"]
        if tangent is None:
            return
        if self.mode == "aim":
            prepared = self.optimizer._pending[param]
            if prepared.tangent is not None:
                prepared.tangent = tangent
            return
        shadow = self._shadows[param]
        (spectrum,) = frame.tangent_spectra([tangent])
        shadow.copy_(frame.turn(shadow, tangent, spectrum, usuitrack_optimizer.GEODESIC_STEPSIZE).frame)
