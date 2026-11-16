"""SPEC step 7 on CUDA: the polar map's `r x r` iterations (`polar.gram_polynomial`) in one
Triton program per matrix, so a schedule is one launch rather than a chain of small GEMMs.

The program's matrices live in a scratch in global memory, L2-resident at these sizes, and
every product runs in `TILE x TILE` tiles, so the width is padded to a multiple of `TILE`
only; a CTA barrier separates the phases. The dots are tf32x3, three TF32 passes, within
fp32's rounding of the eager loop.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

TILE = 32
_COEFFICIENTS: dict[tuple, Tensor] = {}


@triton.jit
def _product(left, right, out, WIDTH: tl.constexpr, TILE: tl.constexpr):
    """`out = left @ right`, each `[WIDTH, WIDTH]` row-major."""
    rows = tl.arange(0, TILE)
    for i in range(0, WIDTH, TILE):
        for j in range(0, WIDTH, TILE):
            acc = tl.zeros((TILE, TILE), dtype=tl.float32)
            for k in range(0, WIDTH, TILE):
                a = tl.load(left + (i + rows)[:, None] * WIDTH + (k + rows)[None, :])
                b = tl.load(right + (k + rows)[:, None] * WIDTH + (j + rows)[None, :])
                acc = tl.dot(a, b, acc, input_precision="tf32x3")
            tl.store(out + (i + rows)[:, None] * WIDTH + (j + rows)[None, :], acc)
    tl.debug_barrier()


@triton.jit
def _polynomial(gram, out, scratch, coefficients, R, STEPS: tl.constexpr, WIDTH: tl.constexpr, TILE: tl.constexpr):
    matrix = tl.program_id(0).to(tl.int64)
    size = WIDTH * WIDTH
    m = scratch + matrix * 4 * size
    c, p, t = m + size, m + 2 * size, m + 3 * size
    rows = tl.arange(0, TILE)
    source = gram + matrix * R * R
    trace = 0.0
    for i in range(0, R, TILE):
        diagonal = i + rows
        trace += tl.sum(tl.load(source + diagonal * (R + 1), mask=diagonal < R, other=0.0), axis=0)
    scale = 1.0 / tl.maximum(tl.sqrt(trace), 1e-7)
    # The padding is zero in M and C, so it stays zero through every product.
    for i in range(0, WIDTH, TILE):
        for j in range(0, WIDTH, TILE):
            ri, rj = (i + rows)[:, None], (j + rows)[None, :]
            inside = (ri < R) & (rj < R)
            tl.store(m + ri * WIDTH + rj, tl.load(source + ri * R + rj, mask=inside, other=0.0) * (scale * scale))
            tl.store(c + ri * WIDTH + rj, tl.where((ri == rj) & inside, scale, 0.0))
    tl.debug_barrier()
    for step in range(STEPS):
        a = tl.load(coefficients + 3 * step)
        b = tl.load(coefficients + 3 * step + 1)
        coefficient = tl.load(coefficients + 3 * step + 2)
        _product(m, m, t, WIDTH, TILE)
        for i in range(0, WIDTH, TILE):
            for j in range(0, WIDTH, TILE):
                ri, rj = (i + rows)[:, None], (j + rows)[None, :]
                eye = tl.where((ri == rj) & (ri < R), a, 0.0)
                tl.store(p + ri * WIDTH + rj, eye + b * tl.load(m + ri * WIDTH + rj) + coefficient * tl.load(t + ri * WIDTH + rj))
        tl.debug_barrier()
        _product(p, m, t, WIDTH, TILE)
        _product(p, t, m, WIDTH, TILE)
        _product(p, c, t, WIDTH, TILE)
        c, t = t, c
    target = out + matrix * R * R
    for i in range(0, R, TILE):
        for j in range(0, R, TILE):
            ri, rj = (i + rows)[:, None], (j + rows)[None, :]
            tl.store(target + ri * R + rj, tl.load(c + ri * WIDTH + rj), mask=(ri < R) & (rj < R))


def polynomial(gram: Tensor, schedule: tuple[tuple[float, float, float], ...]) -> Tensor:
    """`polar.gram_polynomial` for a CUDA fp32 `gram` `[..., r, r]`."""

    r = gram.shape[-1]
    stack = gram.reshape(-1, r, r).contiguous()
    width = triton.cdiv(r, TILE) * TILE
    key = (schedule, gram.device)
    if key not in _COEFFICIENTS:
        _COEFFICIENTS[key] = torch.tensor(schedule, device=gram.device, dtype=torch.float32)
    out = torch.empty_like(stack)
    scratch = torch.empty(stack.shape[0], 4, width, width, device=gram.device, dtype=torch.float32)
    with torch.cuda.device(gram.device):
        _polynomial[(stack.shape[0],)](stack, out, scratch, _COEFFICIENTS[key], r, STEPS=len(schedule), WIDTH=width, TILE=TILE, num_warps=4)
    return out.reshape(gram.shape)
