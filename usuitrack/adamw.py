"""`StochasticAdamW.step` on CUDA: every parameter of one dtype in one Triton launch, through tables of
pointers, so a step over many small tensors (biases, gains) is one launch rather than a dozen per tensor.

Each program updates one `BLOCK` of one tensor; `chunk_member` maps a program to its tensor. Weights and
moments are read once and written once, all arithmetic in fp32; bf16 storage rounds stochastically from
philox bits under a seed drawn from torch's CUDA generator, as the weight write does (`write.py`).
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import triton
import triton.language as tl
from torch import Tensor

BLOCK = 1024
# Per tensor: beta1, beta2, eps, decay (1 - lr wd), step size lr / bc1, 1 / sqrt(bc2).
SCALARS = 6
_TL_DTYPES = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}


@triton.jit
def _round(value, bits):
    # bf16 keeps fp32's top 16 bits: adding 16 random low bits before truncating
    # carries into bit 16 with the discarded fraction's probability.
    return ((value.to(tl.int32, bitcast=True) + (bits & 0xFFFF).to(tl.int32)) & -65536).to(tl.float32, bitcast=True)


# Not `params`: Triton 3.8's launcher cannot bind an argument of that name.
@triton.jit
def _adamw(weights, grads, avgs, squares, scalars, chunk_member, member_first, numels, seed,
           P_DTYPE: tl.constexpr, S_DTYPE: tl.constexpr, BLOCK: tl.constexpr, STOCHASTIC: tl.constexpr):
    chunk = tl.program_id(0)
    member = tl.load(chunk_member + chunk)
    offsets = (chunk - tl.load(member_first + member)).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    inside = offsets < tl.load(numels + member)
    param = tl.load(weights + member).to(tl.pointer_type(P_DTYPE))
    grad = tl.load(grads + member).to(tl.pointer_type(P_DTYPE))
    avg = tl.load(avgs + member).to(tl.pointer_type(S_DTYPE))
    square = tl.load(squares + member).to(tl.pointer_type(S_DTYPE))
    beta1 = tl.load(scalars + member * 6)
    beta2 = tl.load(scalars + member * 6 + 1)
    eps = tl.load(scalars + member * 6 + 2)
    decay = tl.load(scalars + member * 6 + 3)
    step_size = tl.load(scalars + member * 6 + 4)
    correction = tl.load(scalars + member * 6 + 5)
    g = tl.load(grad + offsets, mask=inside, other=0.0).to(tl.float32)
    m = beta1 * tl.load(avg + offsets, mask=inside, other=0.0).to(tl.float32) + (1.0 - beta1) * g
    v = beta2 * tl.load(square + offsets, mask=inside, other=0.0).to(tl.float32) + (1.0 - beta2) * g * g
    w = decay * tl.load(param + offsets, mask=inside, other=0.0).to(tl.float32) - step_size * m / (tl.sqrt(v) * correction + eps)
    if STOCHASTIC:
        b0, b1, b2, _b3 = tl.randint4x(tl.load(seed), (chunk * BLOCK + tl.arange(0, BLOCK)).to(tl.uint32))
        w, m, v = _round(w, b0), _round(m, b1), _round(v, b2)
    tl.store(param + offsets, w.to(P_DTYPE), mask=inside)
    tl.store(avg + offsets, m.to(S_DTYPE), mask=inside)
    tl.store(square + offsets, v.to(S_DTYPE), mask=inside)


class Table(NamedTuple):
    """A fixed set of tensors' addresses and their programs, built once and reused while the addresses hold."""
    params: Tensor
    avgs: Tensor
    squares: Tensor
    chunk_member: Tensor
    member_first: Tensor
    numels: Tensor


def table(params: list[Tensor], avgs: list[Tensor], squares: list[Tensor]) -> Table:
    device = params[0].device
    numels = [p.numel() for p in params]
    chunks = [triton.cdiv(n, BLOCK) for n in numels]
    first = torch.tensor(chunks, dtype=torch.int64).cumsum(0) - torch.tensor(chunks, dtype=torch.int64)
    member = torch.repeat_interleave(torch.arange(len(params), dtype=torch.int32), torch.tensor(chunks))

    def addresses(tensors: list[Tensor]) -> Tensor:
        return torch.tensor([t.data_ptr() for t in tensors], dtype=torch.int64, device=device)

    return Table(addresses(params), addresses(avgs), addresses(squares), member.to(device),
                 first.to(device=device, dtype=torch.int32), torch.tensor(numels, dtype=torch.int64, device=device))


def step(fixed: Table, grads: list[Tensor], scalars: list[float], dtype: torch.dtype, state_dtype: torch.dtype) -> None:
    """One AdamW step over `fixed`'s contiguous tensors with `grads` (same dtype and layout) and `scalars`,
    `SCALARS` per tensor. The per-step tables go up through pinned memory, so nothing waits on the device."""

    device = grads[0].device
    pointers = torch.tensor([g.data_ptr() for g in grads], dtype=torch.int64).pin_memory().to(device, non_blocking=True)
    values = torch.tensor(scalars, dtype=torch.float32).pin_memory().to(device, non_blocking=True)
    seed = torch.randint(0, 2**31 - 1, (1,), device=device, dtype=torch.int32)
    _adamw[(fixed.chunk_member.numel(),)](fixed.params, pointers, fixed.avgs, fixed.squares, values, fixed.chunk_member,
                                         fixed.member_first, fixed.numels, seed, P_DTYPE=_TL_DTYPES[dtype],
                                         S_DTYPE=_TL_DTYPES[state_dtype], BLOCK=BLOCK, STOCHASTIC=dtype is torch.bfloat16)
