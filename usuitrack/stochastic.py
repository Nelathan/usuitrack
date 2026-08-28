"""Stochastic rounding for bf16 weights, and a bf16 AdamW that uses it.

Why UsuiTrack needs this. A subspace tracker deliberately produces small,
well-conditioned steps: the update direction is orthogonalized, so its scale
comes from `lr` alone rather than from gradient magnitude. Against bf16 weights
that is exactly the regime round-to-nearest destroys. Measured on a 2B DiT
finetune at lr 2e-5: weights of order 1e-2 have a bf16 ulp of ~3.05e-5 while a
step's per-element update is ~4e-7 -- about 1/76 of an ulp. Round-to-nearest
discards every one of them, and over 800 steps 96.7% of the model's elements
were bit-identical. Not slow convergence; no convergence, with the signal
thrown away on arrival.

Stochastic rounding lands a 1/76-ulp update with probability 1/76 and keeps the
expectation equal to the fp32 result, so small steps accumulate honestly
instead of vanishing. That is what makes a low, smooth learning rate usable --
the alternative is raising lr until updates clear an ulp, which is a different
and worse optimizer.

Credits:
  * `copy_stochastic_` is Nerogar's fast PyTorch implementation, from
    https://github.com/pytorch/pytorch/issues/120376#issuecomment-1974828905
  * ported here via lodestone-rock's torchastic (Apache-2.0),
    https://github.com/lodestone-rock/torchastic, whose bf16-state AdamW
    `StochasticAdamW` below also follows.
  * method: Zamirai et al., "Revisiting BFloat16 Training",
    https://arxiv.org/abs/2010.06192

Reimplemented from those sources; not a runtime dependency.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.optim import Optimizer


@torch.no_grad()
def copy_stochastic_(target: Tensor, source: Tensor) -> None:
    """Round `source` (fp32) into `target` (bf16) stochastically.

    bf16 keeps the top 16 bits of an fp32 word, so round-to-nearest on the
    discarded low 16 bits is what loses sub-ulp updates. Adding a uniform
    random 16-bit value before truncating makes the carry into bit 16 fire with
    probability equal to the fraction being discarded, which is stochastic
    rounding with no branching and no extra precision.
    """
    result = torch.randint_like(source, dtype=torch.int32, low=0, high=(1 << 16))
    result.add_(source.view(dtype=torch.int32))
    result.bitwise_and_(-65536)  # -65536 == 0xFFFF0000 as a signed int32
    target.copy_(result.view(dtype=torch.float32))


def wants_stochastic_rounding(tensor: Tensor) -> bool:
    """bf16 is the only dtype where this buys anything: fp16 lays its bits out
    differently, and fp32/fp64 keep the increment without help."""
    return tensor.dtype is torch.bfloat16


class StochasticAdamW(Optimizer):
    """AdamW that keeps parameters and moments in bf16 with stochastic rounding.

    The companion to UsuiTrack's matrix path: UsuiTrack owns 2D weights, and
    every caller needs somewhere to put biases, norm gains, and any matrix the
    tracker excludes. Those are the weights that suffer *most* from bf16
    round-to-nearest -- measured on the same run, 1D gains and biases moved 74
    of 16,384 elements in 800 steps -- so a fallback without stochastic
    rounding quietly cancels out the matrix path's gains.

    This buys no memory: `torch.optim.AdamW` already keeps its moments in the
    parameter dtype, so a bf16 model already had bf16 moments. What it buys is
    that the moment recurrences and the parameter update are accumulated in
    fp32 and written back through `copy_stochastic_`, instead of torch running
    `mul_`/`addcmul_` directly on bf16 storage where every one of those
    in-place ops rounds to nearest.

    Deviates from torchastic in two places, both to match `torch.optim.AdamW`
    so this stays a drop-in: decoupled weight decay uses `lr` rather than the
    bias-corrected step size, and non-bf16 parameters are handled in their own
    dtype instead of asserted away.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"lr must be non-negative, got {lr}")
        if eps < 0.0:
            raise ValueError(f"eps must be non-negative, got {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"betas must each be in [0, 1), got {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("StochasticAdamW does not support sparse gradients")

                state = self.state[p]
                stochastic = wants_stochastic_rounding(p)
                state_dtype = p.dtype if stochastic else torch.float32
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=state_dtype)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=state_dtype)
                state["step"] += 1
                step = state["step"]

                grad = p.grad.float()
                exp_avg = state["exp_avg"].float()
                exp_avg_sq = state["exp_avg_sq"].float()

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2_sqrt = (1.0 - beta2**step) ** 0.5
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

                work = p.float() if stochastic else p
                if weight_decay:
                    work.mul_(1.0 - lr * weight_decay)
                work.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)

                if stochastic:
                    # One rounding per tensor per step, at the end, after the
                    # whole update has been accumulated in fp32.
                    copy_stochastic_(p, work)
                    copy_stochastic_(state["exp_avg"], exp_avg)
                    copy_stochastic_(state["exp_avg_sq"], exp_avg_sq)
                else:
                    state["exp_avg"].copy_(exp_avg)
                    state["exp_avg_sq"].copy_(exp_avg_sq)

        return loss
