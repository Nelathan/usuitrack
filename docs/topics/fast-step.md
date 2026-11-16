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
   *Done 2026-11-16* for the first and third: `Prepared` carries `||Z||` once, and phase two runs per matrix. The
   second is chapter 4's. nanodit, paired against chapter 1: 396 against 398 ms/step, a tie (spread ~10 ms); the
   gain is no fp32 moment held past its matrix, not time.
3. **Blockwise state.** Frames, moments and `Y` live in one `[b, n, r]` buffer per (block, shape), and each matrix's
   state is a view into it. Batched ops then need no `torch.stack` copy and no transient, and memory is bounded by
   the block. The earlier OOM came from stacking the whole model's same-shape matrices (FACTS, "bs16 fits once the
   polar runs per matrix"). The weights belong to the model, so they are written with `torch._foreach_*` or the
   chapter 6 kernel. Diagnostics become batched reductions per block, nearly free, so they keep every step.
   *Done 2026-11-16.* The optimizer sees params, not model blocks, so a block is a run of same-shaped matrices
   (and side, device, dtype) in param order, closed at `BLOCK_BYTES`, 256 MiB of transient, the prototype's cap,
   unmeasured as a peak on Anima. Weights are stacked for the write and copied back with `_foreach_copy_`.
   nanodit `step()` 140 to about 62 ms (FACTS, "Blocks take nanodit's `step()` from 140 to about 62 ms").
4. **Weights first, and the preamble.** The weight write needs only the held moment and frame. In the last
   micro-batch's backward hooks, each block's matrices prepare and write as their gradients land, so the launches
   hide behind backward's GPU work. The tracker work (twin `Y`, Newton-Schulz, retraction, moment rotation) is
   needed only by the next backward's `Z = G Q`. It runs deferred on a side stream during the next forward, ordered
   by events. The training loop needs a seam for this; design it with the user. tiny-dit accumulates 4
   micro-batches, so only the last backward steps.
   *Proposal, 2026-11-16, unbuilt (the user designs the seam):*
   - A block is complete when its earliest layer's backward has returned `Z`: its members are consecutive
     layers of one shape, and backward runs from the last layer down. A hook on the leaves of that member writes
     the block's weights; each `W` is then past its last use in this backward (`grad_x` is taken in the same
     call), and a checkpointed layer's recompute runs before its backward, so nothing reads `W` after the write.
   - Only the last micro-batch's backward may write, and nothing inside autograd knows which one that is. That is
     the seam: the loop must say so before the backward. Candidates: a context manager around the last backward
     (`with optimizer.final_backward(): loss.backward()`); a call before it (`optimizer.arm()`); or a micro-batch
     count given to Molegrad at build, which a loop that skips a batch silently breaks. `step()` stays the call
     that closes the step, finishing any block that had no hook.
   - The deferred tracker must finish before the next backward reaches the block, ordered by an event its
     hooks wait on. Molegrad reads a held frame at backward, not at forward (SPEC, "Molegrad"), so a frame
     moved in place during the next forward is the one its backward projects on.
   - The user (2026-11-16): the harness can tell which backward is the last, so the seam is no wall; shape
     it around tiny-dit's loop, which can be adapted to fit.
   - The side stream's transients run beside the forward's activations, so the step's peak moves into the
     forward's. Unmeasured.
   *Read after chapter 6, 2026-11-16, the go/no-go:* nanodit's training step, every group on the input side,
   synced per phase without the profiler: forward 77 ms, backward 178, UsuiTrack's `step()` 49, the fallback
   AdamW 12 (`outputs/tools/nanodit_phase_wall.py`, `outputs/nanodit_phase_wall.log`). Under the profiler
   (`nanodit_phase_idle.py`, its log) the forward leaves about 19 ms of GPU idle over four micro-batches and the
   backward about 15, so the last backward and the first forward hold about 4-5 ms each; `step()` is GPU work
   now, not launches. Hiding it there buys at most about 10-15 ms of a 316 ms step unless a side stream shares
   SMs with the forward's kernels, unmeasured. The fallback AdamW, a per-tensor loop over 70 tensors, spends
   about 10 of its 12 ms on the host.
5. **CUDA graph.** The graph covers the step only. Forward and backward are compiled blocks already, and graphing
   them is a separate matter. The deferred tracker comes first: it is pure optimizer state at static addresses once
   chapter 3 is done. Prerequisites:
   - no host sync: the twin removes the `eigh` status readback, and `init_norm`'s `.item()` happens once, before
     capture;
   - lr as a device tensor, since a Python float is baked into the graph;
   - Molegrad's leaves copied into static buffers.

   Capture costs one step. If chapters 3 and 4 make it easy, it may come earlier.
   *Read after chapter 3, 2026-11-16:* nanodit's 77 matrices stepped alone on synthetic `.grad` (the `G`
   path), every group on the input side: 70-76 ms of summed GPU time against 68-78 ms of unsynced wall, over
   4.3k CUDA events per step, against 14.3k and 60% idle before (`outputs/nanodit_step_profile_ch5.log`). The
   step no longer waits on the host, so a graph buys little there; the GPU time is fp32 SGEMM (~35 ms, mostly
   `prepare`'s projection and aim and the lift) and the write's elementwise chain (~18 ms), chapter 6's target.
   *Dropped (user, 2026-11-16):* blockwise is fast on eager, and a graph is not needed here.
6. **Triton.** The weight write first: lift, decay, norm bound and stochastic rounding in one pass over `W`. Eager
   makes about 10 passes; the compiled write measured -0.22 s/step on LFM3 (FACTS, "The compiled weight write
   saves 0.22 s/step"). Then the moment blend, which is less gain but easy. Newton-Schulz and the other matmuls stay
   on cuBLAS, batched. A Triton kernel is the only implementation, so the release becomes CUDA-only: say so in
   README.
   *Done 2026-11-16.* The write is a Triton kernel on CUDA: two passes per block through weight pointers,
   the lift in registers in TF32, philox rounding bits seeded from torch's CUDA generator. Off CUDA the eager
   write stays, and a test holds the kernel to it (user: the eager write as the CPU path, TF32, philox).
   Larger blocks follow, since the CUDA write holds no transient (FACTS, "The CUDA write takes nanodit's `step()` from 79 to 45 ms").
   The moment blend is not fused yet.
   *After chapter 6, 2026-11-16, the polar on the Gram.* Newton-Schulz runs its schedule on the short-side `r x r`
   Gram for both maps, the frame's with the retraction as a sixth iteration, the `r x r` rounds one Triton
   program per matrix up to `r = 128` and batched GEMMs above (SPEC step 7; user: "I think it's a win"). nanodit
   `step()` 39 to 29 ms at rank 0.05, 69-72 to 60 ms at 32 layers (FACTS, "One polar step cannot follow `Q+Y`").
   Molegrad's backward on nanodit is bandwidth-bound; a fused Triton body never beat cuBLAS, and its GEMMs are
   padded to a multiple of 16 instead (FACTS, "On a small model Molegrad's backward is bandwidth").

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
