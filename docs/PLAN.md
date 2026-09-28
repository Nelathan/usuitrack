# Open design questions

What we are unsure about, how each would be settled, and the evidence that has
moved each answer. `SPEC.md` says what the optimizer does; `FACTS.md` holds the
numbers; `ARCHIVE.md` holds everything closed (the full pre-2026-09-14 text of
this file is `PLAN.md` at commit `e533bdf`). See `AGENTS.md` for how this file is
worked.

Questions are grouped so a session can take **one group** and leave the rest
alone. Inside a group the items share runs, reads, or code.

---

## Group A. Weight precision: the rounding walk

### A1. Does the bf16 stochastic-rounding walk harm the model?

**Established (`FACTS.md`).** Every Anima write is stochastically rounded into
bf16, and at a per-step relative update of `3.7e-6` the weights walk: checkpoint
increments are uncorrelated with prior motion (cosine 0.002) and grow as
`sqrt(steps)` (ratio 0.378 against the walk's 0.378), for matrices and fallback
alike. A toy on a real Anima weight reproduces the magnitude and puts rounding
error at ~7x the fp32 travel. So distance from base on this lane reads the walk.

**Open: is the walk harmful, or only noisy telemetry?** It is isotropic noise of
relative size ~2e-3 on every weight, growing `sqrt(N)`, with nothing pulling it
back. Two user observations fit a harmful walk and are not yet evidence for one:
long runs "drowned in noise", and run 12 went stable but bland. Unbiased rounding
also means the expected path is intact, so the harm could be zero.

**How to settle.**
- *Walk-only control* -- **run 2026-09-14.** Base weights plus a simulated walk
  matched per role to run 13 @550, sampled with run 13's prompts and seed. The
  walk alone moves samples 0.018-0.041 pixel RMS from base on four prompts
  against run 13 @550's 0.116-0.211; on the armored-warrior prompt 0.146 against
  0.175. Looked at: it shifts composition on sensitive prompts, keeps base's style,
  shows no visible degradation; run 13's style and lighting shift is not in it.
  **So at run 13's scale the visible learning is not the walk**, and the walk is
  a composition perturbation rather than obvious damage. Not ruled out: harm that
  accumulates over longer runs (walk grows `sqrt(N)`), or subtler quality loss
  the user's eye would catch. Samples:
  `/mnt/luna/ai/output/anima_walk_only_control/samples` (step 0) against
  `anima_usuitrack_13_acc4/samples` steps 0 and 550.
- *Walk-free arm*: run 13 again with the walk removed. Options, none built:
  fp32 master weights on the CPU (the 2B model is 8 GB in fp32 against 31 GB
  RAM; the update is already formed in fp32 on the GPU, so per step it costs an
  8 GB download, a CPU add and a 4 GB bf16 upload -- ~1.5 s at the measured
  12 GB/s pinned PCIe and ~0.06 s/GB CPU add, +8% at `k=4`, no VRAM); an error-feedback residual (a bf16 residual is +4 GB and does not fit
  at bs4 x k4, an int8 one is +2 GB and might); or simply a larger step, since
  walk grows as `sqrt(lr)` and travel as `lr`.

**Prior:** P14's old question "does stochastic rounding on the weight write
cause source degradation" (on LFM) is the same question and was never run.

### A2. Displacement needs a walk-aware read

`scripts/usuitrack_base_distance.py` is still the only read of travel and it now
measures mostly noise at this step size. Candidates: the increment-correlation
read used above (`cos(w_t2 - w_t1, w_t1 - base)`), which separates coherent
travel from walk with two checkpoints; or the fp32 expectation, if a walk-free
arm exists. Decide which becomes the standard read before the next run is judged
by distance.

---

## Group B. Accumulation: what is built, and what is still assumed

Built, validated end to end on run 13, and closed in `ARCHIVE.md`: held frame,
per-micro-batch polar map, moment once per step, `sqrt(a + (1-a)/k)` step shrink,
bf16 running sums with an unrounded final fold, token-exact weights in the
ai-toolkit trainer, fallback grads divided into the group mean before the clip.

### B1. Are the planes `k=4` makes live signal, or span?

`k`, not the table, sets liveness, concentration and persistence (closed in
`ARCHIVE.md`, run 14).

**Still open: what the extra live planes are.** The user's reading: averaging
lets signal clear the noise floor, and the polar tangent turns it. The competing
reading: the mean of `k` independent tangents spans more directions. Capture
does not separate them cleanly -- an `r`-plane frame catches `sqrt(r/d)` of
isotropic noise too -- though capture rose far less than that noise baseline
would. A discriminating read: at `k=4`, project each *held-out* micro-batch's
gradient onto the planes that are live at `k=4` but not at `k=1`, against random
planes of the same count in the frame's complement. Signal planes catch more
than random; span planes do not.

Mechanism to keep in view: the step's tangent is the *mean of the per-micro-batch
quadratic tangents*, so its expectation is `(G_bar^T G_bar + Cov) Q` at the
micro-batch's own noise level whatever `k` is. `k` cuts the aim's variance, not
its bias toward the noise covariance; the extra live planes may be the aim
settling onto Cov's stable structure rather than onto signal. That split is
false: the signal lives inside Cov's leading directions (`ARCHIVE.md`, "the energy
aim is the right target"), so Cov structure and signal are one target.

Run 14's samples (user, 2026-09-28) side with `k=4`: the warrior is slightly
broken and nothing reads better than run 13; run 14 also has the worst loss after
heavy smoothing, and the lowest liveness, capture and overshoot cosine from the
start. At `k=1` the frame holds fewer planes, catches less, and what it steps
along carries less into the next batch. That is the frame's side of
accumulation, which a higher `beta` cannot reach -- the frame has no `beta`; its
time analog is a smaller `eta` (B2).

### B2. Orthogonalize each micro-batch, or sum and orthogonalize once?

Autograd's batch gradient is a sum, so sum-then-polar is the standard estimator;
polar-then-mean gives each micro-batch an equally loud vote. The temporal
version of this choice (polar before the moment) was measured on LFM and won on
invariance rather than loss. The across-micro-batch version was adopted by that
analogy and **never measured**. Arm: run 13 with raw projected gradients summed
before one polar map (a lab-side patch, not a release option). Read the same four
mechanism reads as B1, then samples.

**B2 is also the simplification question.** Accumulation brought machinery
(held frame, per-micro-batch folds, the trainer hook); a real batch would not
need it. Activation offloading is cheap under gradient checkpointing -- only
block boundaries are saved, and their transfer overlaps compute -- so a real
bs16 on the 12 GB card is reachable without accumulation. It buys no throughput
(bs1 is already within 5% of bs4 per sample, B3); what it changes is the
estimator: sum-then-polar over the whole batch. If that matches polar-first
across micro-batches, accumulation can go.

**Any batch comparison must move `eta` with the batch.** A larger batch cut lag
but read worse at an `eta` never raised with it (user, 2026-09-28): the turn a
frame can take is set by how far its aim can be trusted and how fast the model
moves (`ARCHIVE.md`, "the turn matters only while the aim moves"). The time
analog runs the other way: `k=1` with `eta/4` and a `beta` whose memory is `k`
times longer (0.975 for 0.9 at `k=4`) should match `k=4` if accumulation is only
a throughput choice for the frame as well as the update.

### B3. The limit of B2: per-sample polar map, and what it costs

Taken to the end, polar-first means bs1 x k16: every sample orthogonalized before
it is averaged. Throughput: accumulation is free per sample (run 12 1.17 s,
run 13 1.11 s), so the unknown is bs1's per-sample cost against bs4's batching,
plus 16 polar maps per step. Gradient checkpointing might turn off at bs1 and buy
time back. The opposite end -- a real bs16 batch through weight streaming or
activation offload -- is sum-then-polar over 16 samples, and pays PCIe transfers
of the 4 GB model per pass. B3 is therefore B2's question at the sample level
first and a speed question second.

**Throughput measured (2026-09-14): bs1 x k16 costs ~5% per sample** -- 0.97
s/sample against bs4 x k4's 0.92 (median step intervals, sampling excluded; the
bs1 figure rests on two logged intervals). Gradient checkpointing cannot come
off: bs1 without it OOMs. So per-sample polar-first is affordable; what it would
test is B2's question.

### B4. `micro_batch_agreement` falls with the learning rate

2.5-3.1e-3 on run 13's plateau, 1.3e-3 and then 3.0e-4 as lr decays to `2e-5`.
Every micro-batch in a step sees the same weights, so a pure data signal-to-noise
read should not care about lr. Candidate: part of the shared gradient is the
model being pulled along its own motion, so the meter partly reads the optimizer.
Confounded with the schedule. Parked with P-arrival (Group E): the user judges it
not worth much here.

### B5. Not tried

Stratified timestep sampling across the group (one micro-batch per quarter of
the `t` range): free, cuts the group's `t` variance.

---

## Group D. The step budget: learning rate, fallback, beta

### D1. The matrix learning rate is probably low

- `grad_moment_cosine` stayed positive through run 13 (+3.4e-3 to +6.6e-3): by
  P14's rule, under-travel. Run 12 hovered near zero.
- Anima's per-step relative update is ~20x smaller than LFM's best arm
  (`FACTS.md`), at roughly twice the tokens per step.
- The user reads run 13 as undertrained and wants more.

Against it: the cosine's zero is the descent optimum, not the finetune optimum
(forgetting is invisible to it), and there is no forgetting instrument on this
lane. The user's napkin math: bs4 at 768 px is noisier than LFM at bs16 -- about
LFM bs2 x 1024 -- so accumulation might want about half the LFM-equivalent lr.

**Levers, not yet chosen between:** raise lr at `k=4`; `k=2` or bs2 to get more
optimizer steps per wall-clock hour; a longer plateau (run 13's decay started at
432 of 576); higher `beta`. A1 bears on all of them: a larger step raises travel
against walk as `sqrt(lr)`.

### D4. Beta, and whether the moment damps too much

`beta 0.9` is the LFM optimum at two step sizes. With polar-first, the moment
cancels whatever disagrees between steps, and `moment_persistence` reads ~1e-3.
Either that is healthy -- a short geodesic in a low-rank frame -- or the method
works partly because it damps very hard. Re-orthogonalizing the moment (the
extrinsic mean, first-order Karcher mean of the directions) helped marginally on
LFM and was left out for flops. Arms: higher `beta` at matched
`update_to_param_ratio` (a beta change is an lr change); `REORTHOGONALIZE_MOMENT`
on Anima.

### D5. One unit for the step

The step a module takes is today an accident of four factors nobody chose
together: `lr`, the Muon aspect factor (fan-in and fan-out), the rank (with the
aspect factor, `||U||_F = sqrt(m) sqrt(r / min(m,n))`, so a rank change is an lr
change -- which also confounds every rank comparison), and the agreement shrink
after the polar map (the moment averages unit-norm directions, so disagreement
between steps shrinks the step and forces lr to be re-fixed). The fallback runs
in a fifth unit (AdamW's per-coordinate `lr`); on Anima it now carries only the
qk-norm gains. `update_to_param_ratio` is one fleet scalar: it cannot say which
module class took the change.

**Candidate law (user's direction, 2026-09-28, not yet chosen):** `lr` is the
relative change per step, `||dW|| / ||W|| = lr` per module (LAMB's trust ratio in
spirit), and the fallback is the minimum that lets the rest of the model follow
the matrices' change -- `fallback_ratio x lr` in the same relative unit, far
below 1. Axes to settle before any arm:
- agreement inside the step (a sensor of how far to go) or out of it (direction
  only; the moment's norm discarded);
- reference norm: Frobenius (total energy) or spectral (largest gain);
- fallback form: AdamW with a relative cap, or normalized momentum;
- zero-initialized and near-zero tensors need a floor on `||W||`;
- a relative step compounds on a growing weight, a finetune's weights barely
  grow, so this is likely moot here and not in pretraining.
Moonlight's `0.2 sqrt(max(m,n))` is a different distribution again (w2 x2.1
against today's). **Instrument first:** per-role-class relative change per step,
on both lanes, so the current distribution is seen before a law replaces it.

---

## Group E. Rank and the frame

### E7. Anima's `rank_fraction`

`rank_fraction 0.1` is the LFM measurement (`ARCHIVE.md`, "rank is a fraction of
the matrix's size"); Anima has never run it. At 0.1 its roles get ranks 205
(2048²), 145 (2048x1024), 410 (the ff pair), 34 and 32 (the patch projections),
and the per-step update rises 1.4-3x over run 17's table at the same lr, because
rank sets the step (SPEC).

The AdaLN linears now train here too, at 72 (`linear_1`, 256x2048) and 125
(`linear_2`, 6144x256), against a per-step gradient of rank at most the batch
(16): their input is one timestep embedding per sample. Most of their planes
carry no signal on a given step, and Newton-Schulz lifts whatever the moment holds
there to unit scale. Whether that matters has no read yet: the telemetry is
fleet-wide, so it waits on D5's per-role instrument, and on the samples. The
verdict so far is 19 finite steps (`ARCHIVE.md`). Arm: 0.1 against a lower
fraction under a real bs16 (B2), against run 17, with lr lowered to match run
17's per-module step so the comparison reads the subspace and not the step. The
verdict is samples. Frame and moment in bf16 grow from 0.11 to 0.47 GiB at 0.1;
the eigh and polar transients at rank 410 are unmeasured.

### E4. Parked: an arrival sensor

A converged frame orbits on batch noise and a constant turn keeps integrating it.
No sensor exists; `micro_batch_agreement` is published, wired to nothing, and
tracks lr (B4). The user judges this not worth pursuing on this lane now.
Its premise that orbiting costs found no cost on LFM at `0.04` (`ARCHIVE.md`,
"the turn matters only while the aim moves").

### E5. The `min(m,n)/2` cap

Binds only on matrices whose short side is under `4 c^2` of their long side
(1/25 at `rank_fraction 0.1`; Anima's patch_embed asks 37 and runs at 34).
Nothing acts on it until a capped matrix starves.

---

## Group F. Engineering, parked

- **F1. Per-matrix lr from the overshoot meter** (was P15). Decide by first
  logging `grad_moment_cosine` spread across matrices by role; if flat, it is a
  global schedule in costume. Family prior is poor (closed P2).
- **F2. Magic numbers** (was P5). Rank cap (E5); `beta`/`eps` (D4);
  `AURORA_PP_*` inherited and cheap; Newton-Schulz coefficients (5 steps, the
  largest cost, invisible end to end); `1e-12` floors never audited against
  their dtype -- the lesson of the `1e-6 sigma_max` floor that never fired.
- **F3. Five device syncs per step** (was P12). One is `eigh`'s convergence
  check; four unnamed. Low priority: optimizer time is invisible end to end.

---

## Already tried. Do not repeat.

- **Normalizing the rotation angle by `sigma_max`**, or bare ortho of the same
  family. Forces a constant top angle, re-inflates the residual tail.
- **Per-step Frobenius normalization of the gradient inside the basis update.**
  A good basis and a garbage basis read identically.
- **Clamping the rotation angle to a fixed ceiling.** Eats acquisition.
- **A magnitude cap on the tangent, and a full-rank zero-tangent branch.** Added
  to a path no default configuration reached.
- **Overlap reprojection of the moment through frame motion.** Charges a cosine
  tax on rotated directions; identity transport instead.
- **Per-matrix agreement ceilings.** Worse spread than no per-matrix term.
- **Stochastic rounding on the basis write.** Worse by `sqrt(2)`: the geodesic
  moves the frame by a real angle every step. Scoped to the basis only.
- **Turning only the top planes.** Worse on target; the leading plane's
  magnitude was the problem, which the polar tangent fixes.
- **A fresh eigendecomposition mid-training** in place of the tracked frame.
