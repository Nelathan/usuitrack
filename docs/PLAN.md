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
`ARCHIVE.md`, run 14). Size a table at the `k` it runs at.

**Still open: what the extra live planes are.** The user's reading: averaging
lets signal clear the noise floor, and the polar tangent turns it. The competing
reading: the mean of `k` independent tangents spans more directions. Capture
does not separate them cleanly -- an `r`-plane frame catches `sqrt(r/d)` of
isotropic noise too -- though capture rose far less than that noise baseline
would. A discriminating read: at `k=4`, project each *held-out* micro-batch's
gradient onto the planes that are live at `k=4` but not at `k=1`, against random
planes of the same count in the frame's complement. Signal planes catch more
than random; span planes do not.

### B2. Orthogonalize each micro-batch, or sum and orthogonalize once?

Autograd's batch gradient is a sum, so sum-then-polar is the standard estimator;
polar-then-mean gives each micro-batch an equally loud vote. The temporal
version of this choice (polar before the moment) was measured on LFM and won on
invariance rather than loss. The across-micro-batch version was adopted by that
analogy and **never measured**. Arm: run 13 with raw projected gradients summed
before one polar map (a lab-side patch, not a release option). Read the same four
mechanism reads as B1, then samples.

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
test is B2's question, and a table for it would need calibrating at bs1 x k16,
since liveness depends on `k` (B1).

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

## Group C. What the loss hands the optimizer

### C1. Discard per-token loss magnitude before the sum

The polar map discards magnitude after the sum; inside a micro-batch the sum is
still magnitude-weighted, so the loudest samples and patches own the direction
before the polar map ever sees it. Flow-matching loss magnitude varies strongly
with timestep. The user's proposal: normalize every image patch's error to unit
norm before summing -- numerically cleaner, and plausibly less dead tail at low
batch size. Timestep weighting goes with it (magnitude is what it sets).

**Built as an arm, choices made overnight and open to revision:** ai-toolkit
`loss_type: patch_equalized_mse` weights each 2x2 latent patch (the DiT's
patchify) by `mean(n) / n_p`, `n_p` its detached error norm across channels, so
every patch's gradient has the same norm and the micro-batch keeps MSE's summed
patch-gradient norm. That is "equal norm" rather than literally unit norm, so the
fallback and the clip still see MSE-sized gradients. Masks are not counted.
`timestep_type: linear` samples timesteps exactly as `weighted` does and drops
the loss weight.

**Prediction was: flatter tangent spectrum. Measured: no spectrum change.** Run 16
(`patch_equalized300`) against run 15 (`rebaseline300`), both `k=4`, 300 steps,
matched step (`FACTS.md`). After step 50, `tangent_live_fraction`,
`tangent_concentration` and every per-role live fraction are indistinguishable.
Within the first 50 steps the equalized run is slightly flatter. Also moved:
`grad_capture` a little higher (+0.017 -> +0.003, shrinking), and
`micro_batch_agreement` higher early (3.3e-3 vs 1.9e-3), converging by step 225
-- equal-voice patches agree more across micro-batches while the model is near
base. No speed cost.

So within-micro-batch magnitude is not what sets liveness at `k=4`; `k` is (B1).
**Open: the samples**, run 16 against run 15 at matched steps -- the one read
that can still separate the arms. If they match too, the loss stays MSE for
simplicity; per-sample equalization at bs1 (B3) is the other form of the same
idea.

Token-exact weighting across micro-batches is built (`ARCHIVE.md`); C1 is the
within-micro-batch half of the same concern.

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

### D2. The fallback should nudge, not train

Measured in the unit that compares optimizers (relative update per step), the
AdamW fallback at `2e-6` has ~20x the matrices' per-step budget, and it moved
further than any matrix role in run 13. The earlier derivation that picked `2e-6`
assumed unit weight rms and is withdrawn (`ARCHIVE.md`). The user's target: the
fallback's step motion at **1/4 to 1/10 of the matrices'**, so the gates follow
the changing attention and MLP rather than being retrained.

If AdamW realized its full `lr / rms` ceiling, that target is `fallback_lr
~1-2.5e-8` at `adaln`'s rms; it realizes only part of the ceiling, so the right
number is higher by that fraction, which has only been read from walk-dominated
displacement (A2). It feels like freezing only in nominal lr. **And it couples
to A1:** a coordinate update that small is ~1e-4 of a bf16 ulp, so on
stochastically rounded weights such a fallback would be almost entirely walk. A
nudge-sized fallback may need fp32 weights before it means anything. Also open:
whether the gates should be on AdamW at all (D3).

### D3. Should AdaLN modulation train under UsuiTrack?

`adaln` and `other` are 178M 2D linears, excluded as multiplicative gates because
one went non-finite one step into training (`SPEC.md`, parameter eligibility).
That failure predates dead-plane masking, the constant turn, and ortho-first. The
user's samples say the class matters: frozen (run 12) was stable and bland;
trained at `5e-6` (runs 9-11) moved most and came with instability. Arm: put the
gates under UsuiTrack with their own rank and a reduced lr, watch
`nonfinite_grads` from step one.

### D4. Beta, and whether the moment damps too much

`beta 0.9` is the LFM optimum at two step sizes. With polar-first, the moment
cancels whatever disagrees between steps, and `moment_persistence` reads ~1e-3.
Either that is healthy -- a short geodesic in a low-rank frame -- or the method
works partly because it damps very hard. Re-orthogonalizing the moment (the
extrinsic mean, first-order Karcher mean of the directions) helped marginally on
LFM and was left out for flops. Arms: higher `beta` at matched
`update_to_param_ratio` (a beta change is an lr change); `REORTHOGONALIZE_MOMENT`
on Anima.

---

## Group E. Rank and the frame

### E1. The `live_fraction` target

Run 13 settled at 0.90-0.91 fleet with the most skewed roles at 0.81-0.82. Buying
fraction costs live planes (`live ~ r^0.6` measured between runs 11 and 12, which
does not extrapolate from a calibration rank). What decides it is the evidence
that over-provisioning costs about twice under-provisioning; what is missing is
whether that holds this deep. Needs the user's call before the next table.

### E2. The sizing estimator is second order; the within-role spread is first

Per-role mean is the simple default. `geomean / mean` from the calibration
predicted run 13's per-role live fraction (correlation 0.92): one rank per role
leaves low-demand matrices with dead planes whatever average picked it. Geomean
is not proven better by maths or data. What would move the fraction is splitting
skewed roles (attn2 k/v/out) by depth band -- depth-dependent rank has been
observed before. Read first: per-matrix live counts by block index from a
calibration, to see whether the spread is depth.

### E3. `eta` is unswept and the only handle on frame motion

`0.05` ran clean at bs1 where `0.02` once diverged; it read worse loss with a
better projected grad norm on LFM. Never swept downward or in `0.01-0.03` since
the gain was removed. Cheap on LFM: bs1, 300 steps, a minute an arm. Do not make
the aim hot to simulate higher rank.

### E4. Parked: an arrival sensor

A converged frame orbits on batch noise and a constant turn keeps integrating it.
No sensor exists; `micro_batch_agreement` is published, wired to nothing, and
tracks lr (B4). The user judges this not worth pursuing on this lane now.

### E5. Calibration cannot probe past the `min(m,n)/2` cap

Capped roles read as lower bounds. Nothing acts on it until a role starves at
its cap.

---

## Group F. Engineering, parked

- **F1. Per-matrix lr from the overshoot meter** (was P15). Decide by first
  logging `grad_moment_cosine` spread across matrices by role; if flat, it is a
  global schedule in costume. Family prior is poor (closed P2).
- **F2. Magic numbers** (was P5). `eta` (E3); rank cap (E5); `beta`/`eps` (D4);
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
