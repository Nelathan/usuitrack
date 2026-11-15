# A fast step for small, deep models

Opened 2026-11-15. PLAN Group V links here. This is the trip log for the arc. Compact the session after each
chapter is committed, and read this doc first after a compaction.

## The question

The step costs too much on small, deep models, and that cost is launch overhead: the host waits, not the GPU. The
target is 32 layers at small `d`, where the per-matrix calls multiply. Both facts are known; do not re-profile to
rediscover them (FACTS, "On a small deep model the step costs 38% of training").

The baseline is nanodit d768x12, 77 matrices, as tiny-dit wires it. The training step takes 357 ms, of which
`step()` is 137 ms. Stepped alone, the 77 matrices make about 14.3k CUDA events per step, 51 ms of GPU work in
127 ms. The runner's moment polar plus turn, stacked by shape, takes the step to 308 ms. The optimizer's
diagnostics accumulation costs at least another 20-22 ms. Bench: `outputs/tools/nanodit_step_bench.py`
(run from tiny-dit) and `outputs/tools/nanodit_step_profile.py`.

## Decisions (user, 2026-11-15)

- **Defaults.** The twin tracker, xside, Molegrad on, and the diagonal aim.
  - Twin: Newton-Schulz with one retraction, every step, `beta_f` 0.95, `rho` 1 (FACTS, "In replay under the diagonal aim,
    the twin's polar from Newton-Schulz reads as `eigh`'s").
  - xside: every matrix on the input side. It was not worse on nanodit and won at 2e-3 over 6k steps, but the user
    calls that a wash. This reverses the finetuning closure in ARCHIVE (writers keep their output side) by the
    user's choice (ARCHIVE, "every matrix on the input side").
  - Molegrad on: "it really works great". The cost is that some `G` statistics are missing and the probe is wonky.
  - The diagonal aim "would work for both `G` and Mole" (user). Worked out: it is built from `X` and `delta`, which only
    Molegrad's backward holds; the plain-`G` path has `G` alone, and its aim is `G`'s covariance action. The twin
    takes either, so the diagonal aim is the default wherever Molegrad is.
- **Does the twin need `G`?** No. It takes the same action today's turn takes, `frame.aim(action, basis)` from
  Molegrad's leaves. The diagonal-aim replays were exactly that.
- **Rules relaxed.** Triton is allowed. The rule against parallel compiled and eager implementations was KISS
  scaffolding from months of the design changing; a piece that is fixed may become a kernel. A CUDA graph is
  preferred over torch.compile: no long compiles, and it must ship.
- **Normalizing `Z` stays.** `M = beta M + (1-beta) Z/||Z||` is the easy cure. A burst of noise magnitude would
  otherwise infect the moment for longer than `beta` suggests. The user is uneasy: clipping would be cleaner, but
  clip to what? (Open, below.) Stability first.

## Chapters, in order

Each chapter is one or more reviewed commits, then a compaction.

1. **Defaults.** Twin as the tracker, with Newton-Schulz and retraction; the geodesic turn, `tangent_spectra` and
   the liveness mask leave. xside, Molegrad on, the diagonal aim. SPEC, README, tests, the harness defaults and the standard arm
   follow. Then verify: an LFM3 500-step arm against the standard arm (loss veto), and the nanodit bench for speed.
   The release's first-frame `frame.fit` is a range finder of two QRs, with no `eigh`.
   *Done 2026-11-16.* LFM3: target +8e-4 and source +1.6e-3 against the turn on the same side at 500, at the
   floor. nanodit: `step()` about 9 ms slower. The twin removes the sync, not launches (FACTS, "The new defaults
   on LFM3").
2. **The three issues the timing showed.**
   - `||Z||` is computed three times. Molegrad's `prepare` computes it only for a finiteness test, `_blend_moment`
     computes it again, and `diagnostics.moment` takes `unit_grad.norm()`, which is 1 by construction. Carry it once
     in `Prepared`.
   - Molegrad's `prepare` runs serially inside `step()`. It goes into backward in chapter 4. `prepare` dates from
     `release_matrix_grads`, gradient early release before Molegrad worked.
   - Every matrix's fp32 moment is held across the write, turn and rotation loops.
3. **Blockwise state.** Frames, moments and `Y` live in one `[b, n, r]` buffer per (block, shape), and each matrix's
   state is a view into it. Batched ops then need no `torch.stack` copy and no transient, and memory is bounded by
   the block. The earlier OOM came from stacking the whole model's same-shape matrices (FACTS, "bs16 fits once the
   polar runs per matrix"). The weights belong to the model, so they are written with `torch._foreach_*` or the
   chapter 6 kernel. Diagnostics become batched reductions per block, nearly free, so they keep every step.
4. **Weights first, and the preamble.** The weight write needs only the held moment and frame. In the last
   micro-batch's backward hooks, each block's matrices prepare and write as their gradients land, so the launches
   hide behind backward's GPU work. The tracker work (twin `Y`, Newton-Schulz, retraction, moment rotation) is
   needed only by the next backward's `Z = G Q`. It runs deferred on a side stream during the next forward, ordered
   by events. The training loop needs a seam for this; design it with the user. tiny-dit accumulates 4
   micro-batches, so only the last backward steps.
5. **CUDA graph.** The graph covers the step only. Forward and backward are compiled blocks already, and graphing
   them is a separate matter. The deferred tracker comes first: it is pure optimizer state at static addresses once
   chapter 3 is done. Prerequisites:
   - no host sync: the twin removes the `eigh` status readback, and `init_norm`'s `.item()` happens once, before
     capture;
   - lr as a device tensor, since a Python float is baked into the graph;
   - Molegrad's leaves copied into static buffers.

   Capture costs one step. If chapters 3 and 4 make it easy, it may come earlier.
6. **Triton.** The weight write first: lift, decay, norm bound and stochastic rounding in one pass over `W`. Eager
   makes about 10 passes; the compiled write measured -0.22 s/step on LFM3 (FACTS, "The compiled weight write
   saves 0.22 s/step"). Then the moment blend, which is less gain but easy. Newton-Schulz and the other matmuls stay
   on cuBLAS, batched. A Triton kernel is the only implementation, so the release becomes CUDA-only: say so in
   README.

## Open nuance (do not lose)

- **Clip instead of normalize `Z`, and clip to what?** One candidate is `||Z||` capped at `c` times a running
  (log-space) EMA of `||Z||`. That keeps magnitude information and caps bursts. Unmeasured; PLAN V1,
  not a chapter. The user once logged `||Z|| / ||M||`: strongly `beta`-dependent, about 4-5 at `beta` 0.90.
  That is the white stream's value: the EMA of independent unit vectors has norm `sqrt((1-beta)/(1+beta))`,
  0.229 at 0.90, so the ratio is 4.36. The ratio read the incomplete-cancellation floor, not bursts. So the
  clip's reference is `||Z||`'s own running scale, not `||M||`, which carries that floor.
- **Diagnostics granularity.** Accumulating only on logging steps samples 1 in `k` steps rather than averaging all of
  them. That is nearly free for slow means, but it would miss events in the totals (`nonfinite_grads`). The aim is
  instead to make the reads cheap enough, by chapter 3, to keep every step.
- **The `release_matrix_grads` / `prepare` path.** It is needed only without Molegrad. If Molegrad is the default,
  decide whether the `.grad` path stays as a fallback for models Molegrad cannot wrap.
- **`G` statistics missing under Molegrad**, and the probe's estimate is wonky (`grad_capture_probe`). Known; not in
  this arc.
- **The twin on the plain-`G` path under xside** was replayed under `G` (FACTS, "A warm frame matches"). Its loss
  run was on LFM3's native sides.
- **tiny-dit pins usuitrack by git rev**, so a default change reaches nanodit only when the pin moves. That is the
  tiny-dit session's call.
- **The user's working mode.** The user is on Papamonat: review each change with a Sonnet subagent, then commit and
  push without waiting. Design questions still go to the user. Stop after asking a question.
