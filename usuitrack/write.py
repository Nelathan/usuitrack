"""SPEC step 8 on CUDA: a block's weight write in two Triton passes, through a
table of weight pointers, so nothing is stacked or copied back (SPEC "Blocks").

The measure pass reads `W` for `||W||^2` and, under norm bounds, `<W, L>` and
`||L||^2` per tile, `L` the lift; the write pass reads and writes `W` once.
Both form `L = rows cols^T` in registers from the rank-sized factors, in TF32.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

TILE = 64
RANK_CHUNK = 32
_TL_DTYPES = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}


@triton.jit
def _write_pass(pointers, rows, cols, alpha, sigma, seed, partials, M, N, R,
                row_b, row_m, row_r, col_b, col_n, col_r,
                DTYPE: tl.constexpr, TILE: tl.constexpr, RANK_CHUNK: tl.constexpr,
                WRITE: tl.constexpr, LIFT: tl.constexpr, FULL: tl.constexpr, STOCHASTIC: tl.constexpr):
    tile_m = tl.program_id(0)
    tile_n = tl.program_id(1)
    member = tl.program_id(2)
    weight = tl.load(pointers + member).to(tl.pointer_type(DTYPE))
    rm = tile_m * TILE + tl.arange(0, TILE)
    rn = tile_n * TILE + tl.arange(0, TILE)
    inside = (rm[:, None] < M) & (rn[None, :] < N)
    offsets = rm[:, None].to(tl.int64) * N + rn[None, :]
    w = tl.load(weight + offsets, mask=inside, other=0.0).to(tl.float32)
    lift = tl.zeros((TILE, TILE), dtype=tl.float32)
    if LIFT and FULL:
        # The full space: `rows` is the direction itself, laid out as `W` by its strides.
        lift = tl.load(rows + member * row_b + rm[:, None] * row_m + rn[None, :] * row_r, mask=inside, other=0.0).to(tl.float32)
    elif LIFT:
        for k in range(0, R, RANK_CHUNK):
            rk = k + tl.arange(0, RANK_CHUNK)
            a = tl.load(rows + member * row_b + rm[:, None] * row_m + rk[None, :] * row_r, mask=(rm[:, None] < M) & (rk[None, :] < R), other=0.0)
            c = tl.load(cols + member * col_b + rn[:, None] * col_n + rk[None, :] * col_r, mask=(rn[:, None] < N) & (rk[None, :] < R), other=0.0)
            lift += tl.dot(a.to(tl.float32), tl.trans(c.to(tl.float32)), input_precision="tf32")
    if WRITE:
        written = tl.load(alpha + member) * w - tl.load(sigma + member) * lift
        if STOCHASTIC:
            # bf16 keeps fp32's top 16 bits: adding 16 random low bits before truncating
            # carries into bit 16 with the discarded fraction's probability.
            noise = tl.randint(tl.load(seed), (member.to(tl.int64) * M * N + offsets).to(tl.uint32)) & 0xFFFF
            bits = (written.to(tl.int32, bitcast=True) + noise.to(tl.int32)) & -65536
            written = bits.to(tl.float32, bitcast=True)
        tl.store(weight + offsets, written.to(DTYPE), mask=inside)
    else:
        tile = (member * tl.num_programs(0) + tile_m) * tl.num_programs(1) + tile_n
        tl.store(partials + tile * 3 + 0, tl.sum(w * w))
        tl.store(partials + tile * 3 + 1, tl.sum(w * lift))
        tl.store(partials + tile * 3 + 2, tl.sum(lift * lift))


def write_block(params: list[Tensor], pointers: Tensor, side: str, direction: Tensor, frames: Tensor | None, decay: float,
                lr: float, fixed_scale: float | None, bounds: tuple, reference: Tensor | None) -> Tensor:
    """Writes `W <- decay W - scale lift(O)` into each of `params`, clamped into `bounds` times `reference` and
    rounded once; returns each member's scale, `[b]`. `frames` `None` is the full space, `O` itself oriented back."""

    if not all(param.is_contiguous() for param in params):
        raise ValueError("UsuiTrack: the CUDA weight write needs contiguous weights")
    count = len(params)
    M, N = params[0].shape
    # W = rows cols^T: on the input side the moment is [out, r] and the frame [in, r]. In the full
    # space `rows` is the direction as `W`: [out, in] on the input side, transposed on the output side.
    if frames is None:
        rows = cols = direction if side == "right" else direction.mT
    else:
        rows, cols = (direction, frames) if side == "right" else (frames, direction)
    grid = (triton.cdiv(M, TILE), triton.cdiv(N, TILE), count)
    shape = dict(M=M, N=N, R=direction.shape[-1], row_b=rows.stride(0), row_m=rows.stride(1), row_r=rows.stride(2),
                 col_b=cols.stride(0), col_n=cols.stride(1), col_r=cols.stride(2))
    kind = dict(DTYPE=_TL_DTYPES[params[0].dtype], TILE=TILE, RANK_CHUNK=RANK_CHUNK, FULL=frames is None, STOCHASTIC=params[0].dtype is torch.bfloat16)
    low, high = bounds
    bounded = (low, high) != (None, None)
    if fixed_scale is None or bounded:
        partials = torch.empty(count, grid[0] * grid[1], 3, device=direction.device, dtype=torch.float32)
        _write_pass[grid](pointers, rows, cols, partials, partials, partials, partials, **shape, **kind, WRITE=False, LIFT=bounded)
        weight_sq, cross, lift_sq = partials.sum(dim=1).unbind(-1)
    scale = lr * weight_sq.sqrt() if fixed_scale is None else torch.full((count,), fixed_scale, device=direction.device)
    alpha = torch.full_like(scale, decay)
    sigma = scale
    if bounded:
        # ||a W - s L||^2 from the partials: the clamp's factor without a pass over the result.
        norm = (decay * decay * weight_sq - 2.0 * decay * scale * cross + scale * scale * lift_sq).clamp_min(0.0).sqrt()
        clamped = norm.clamp(None if low is None else low * reference, None if high is None else high * reference)
        factor = clamped / norm.clamp_min(1e-12)
        alpha, sigma = alpha * factor, sigma * factor
    seed = torch.randint(0, 2**31 - 1, (1,), device=direction.device, dtype=torch.int32)
    _write_pass[grid](pointers, rows, cols, alpha, sigma, seed, seed, **shape, **kind, WRITE=True, LIFT=True)
    return scale
