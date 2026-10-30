# Projected activation, second pass

Design notes for PLAN S2: UsuiTrack's weight gradients without ever forming
`G`. Notes, not SPEC: nothing here is built. The first pass is
`../optimizers/PROJECTED_ACTIVATION.md` (evidence only).

## What it is for

A FLOP win, not a VRAM win (user, 2026-10-30). Activation checkpointing stays:
the block's internals are recomputed in backward, never kept. What goes is the
full weight-gradient GEMM `dW = delta^T X` (`2 T d_out d_in` FLOPs per matrix),
the projection of `G` into the frame, and the release hook that frees `G`.

## The exact form

The frame `Q` turns only in `step()`, so during a step's forward and backward
it is fixed and known. Wherever autograd would form `dW`, both `X` (recomputed
under checkpointing) and `delta` are in hand:

| side | `Z` | Oja action (the tangent before projection) |
|---|---|---|
| input (`Q` is `[d_in, r]`) | `delta^T (X Q)` | `G^T G Q = X^T (delta Z)` |
| output (`Q` is `[d_out, r]`) | `(delta Q)^T X` | `G G^T Q = delta^T (X Z^T)` |

Four skinny GEMMs, `~4 T r (d_in + d_out)` FLOPs against `2 T d_in d_out`: a
ratio of `2 r (d_in + d_out) / (d_in d_out)`, about 0.1 at `rank_fraction`
0.05. Exact: same `Z`, same tangent, same geometry as today, both sides. `dX =
delta W` is untouched and never needed `X`.

Tracking from `X` (S3) is not required by this form; it buys only not holding
`X`, which recompute already avoids. S3 and S4 stay as questions about the
tracker.

## Wiring candidates, under `torch.compile` and checkpointing

1. **Module streams.** `W` frozen for autograd (no `dW`, no null gradients);
   a forward pre-hook takes `X` on the recompute pass, a tap on the module
   output takes `delta` in backward, and the four GEMMs run there. The S3
   instrument already does the first half. Risk: each hook is a graph break.
2. **Zero adapter.** `y = x W^T + (x Q) A^T` with `A = 0` and `W` frozen:
   autograd yields `A.grad = Z` natively, compile-clean. It gives `Z` but not
   the tangent (that needs `X` and `Z` together), and wastes the `delta A`
   products against a zero `A`.
3. **`torch.library` custom op** with registered autograd: opaque to compile,
   one op per projected Linear; `Z` and the tangent leave through mutated
   outputs.
4. **Fold `Q` into the forward GEMM.** Stack `Q^T` under `W`:
   `x [W; Q^T]^T = [y, x Q]`, one GEMM producing the projected input for free
   (`r / d_out` more columns), no extra launch. With one frame per norm site,
   q/k/v (often one fused qkv GEMM already) and gate/up append it once.

The first pass used eager custom-autograd wrappers and ran 9% slower at 350M;
the FLOP saving was never collected. The bet now is that compile plus fewer,
fused launches collects it.

## Sharing

q/k/v and gate/up read one normed `X`, so `X Q` is one per norm site when the
frame is the site's (S3's aimed frames should converge onto it; unmeasured).
Exact `Z` per matrix still uses each matrix's own `delta`. A frame that mixes
`delta` in is per matrix again.

## Speed levers from inference compilers

Moondream's Photon (user's pointer, 2026-10-30): the whole forward compiled
into one megakernel, schedules searched under a CPU cost model, independent
work overlapped across what would be kernel boundaries (an MLP projection
starting while attention scans the KV cache). Forward and inference only, the
compiler proprietary. What transfers here:

- **Launch overhead is our optimizer's bottleneck at 350M**, not FLOPs: a solo
  `eigh` ~0.8 ms and a solo polar ~0.4 ms per matrix, launch-bound (FACTS).
  CUDA-graph capture of the step (static shapes, per-matrix loop) is the
  cheap version of "one program".
- **Overlap.** The skinny `Z` and tangent GEMMs are independent of `dX`; a side
  stream can run them under the `dX` GEMM.
- **Fusion across the boundary**, candidate 4 above: the projection rides the
  forward GEMM.

## Order of work

1. Microbench on one LFM3 decoder layer, compiled, checkpointed: today's
   `dW` + release against candidates 1, 2 and 4, on time and on `Z` and the
   tangent matching `project_and_aim(G, Q)` at the suite's tolerance.
2. The winner end to end in the harness against the standard arm: loss inside
   the noise floor, s/step lower.
