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


@torch.no_grad()
def round_into(target: Tensor, source: Tensor) -> None:
    """Write fp32 `source` into `target`: stochastically for bf16, otherwise to
    nearest. `copy_stochastic_` truncates to bf16's bit layout, so on any other
    dtype it would round to the bf16 grid."""
    if wants_stochastic_rounding(target):
        copy_stochastic_(target, source)
    else:
        target.copy_(source)


# The dtypes the CUDA step's kernel takes (`adamw.py`); others, and non-contiguous tensors, take the eager loop.
KERNEL_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


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
        self._tables: dict[tuple, tuple] = {}

    def __setstate__(self, state):
        super().__setstate__(state)
        # The CUDA step's address tables (`adamw.table`), per launch: the addresses they were built from, the table.
        self._tables = {}

    @torch.no_grad()
    def step(self, closure=None):
        """On CUDA one Triton launch per device and dtype (`adamw.py`); elsewhere a loop over tensors, the same
        maths, which a test holds the kernel to."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        launches: dict[tuple, list] = {}
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("StochasticAdamW does not support sparse gradients")
                state = self.state[p]
                if not state:
                    state_dtype = p.dtype if wants_stochastic_rounding(p) else torch.float32
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=state_dtype)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=state_dtype)
                state["step"] += 1
                scalars = (beta1, beta2, group["eps"], 1.0 - lr * group["weight_decay"],
                           lr / (1.0 - beta1 ** state["step"]), 1.0 / (1.0 - beta2 ** state["step"]) ** 0.5)
                if p.is_cuda and p.dtype in KERNEL_DTYPES and p.is_contiguous() and p.grad.is_contiguous():
                    launches.setdefault((p.device, p.dtype, state["exp_avg"].dtype), []).append((p, state, scalars))
                else:
                    _step_eager(p, state["exp_avg"], state["exp_avg_sq"], scalars)

        for launch, members in launches.items():
            from . import adamw

            _device, dtype, state_dtype = launch

            params = [p for p, _state, _scalars in members]
            avgs = [state["exp_avg"] for _p, state, _scalars in members]
            squares = [state["exp_avg_sq"] for _p, state, _scalars in members]
            key = tuple((p.data_ptr(), a.data_ptr(), q.data_ptr(), p.numel()) for p, a, q in zip(params, avgs, squares))
            if launch not in self._tables or self._tables[launch][0] != key:
                self._tables[launch] = (key, adamw.table(params, avgs, squares))
            adamw.step(self._tables[launch][1], [p.grad for p in params], [x for *_rest, scalars in members for x in scalars], dtype, state_dtype)
        return loss


@torch.no_grad()
def _step_eager(p: Tensor, avg: Tensor, square: Tensor, scalars: tuple[float, ...]) -> None:
    """One tensor's AdamW step, its moments and update accumulated in fp32 and rounded once at the end."""
    beta1, beta2, eps, decay, step_size, correction = scalars
    grad = p.grad.float()
    exp_avg = avg.float().mul_(beta1).add_(grad, alpha=1.0 - beta1)
    exp_avg_sq = square.float().mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    denom = (exp_avg_sq.sqrt() * correction).add_(eps)
    work = p.float().mul_(decay).addcdiv_(exp_avg, denom, value=-step_size)
    round_into(p, work)
    round_into(avg, exp_avg)
    round_into(square, exp_avg_sq)
