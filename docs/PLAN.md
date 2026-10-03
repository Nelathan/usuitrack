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

## Group D. The step budget: learning rate, fallback, beta

### D1. The global matrix lr, and whether cosine should set it

Under the trust-ratio law (`SPEC.md` step 8) `lr` is the relative step of an
agreed direction, the same unit in every module. Across models, batches and beta
it is not measured. Read so far (`FACTS.md`):

- LFM2.5-350M, bs16 x 1024 tokens, beta 0.98: `6.2e-4` is the law's best target;
  across 10x, lower lr trades target for source, and cosine crosses zero
  between `2e-4` and `6.2e-4`. At bs32 the `sqrt(tokens)` lr under-steps.
- Anima, bs8, beta 0.98, `rank_fraction 0.025`: `3.2e-4` (run 21) and
  `1.5e-4` (run 22) ran hot. At beta 0.95, `5e-5` (run 23) was good but
  changed little. At bs16, `7e-5` (run 24) held cosine to ~60% of the epoch.
- The constructor's `1e-3` is a placeholder, and README says so.

Transfer: `sqrt(tokens)` held from bs16 to bs8 on LFM (D4) and undershoots at
bs32. Across beta a matched per-step ratio does not transfer -- at 0.98 the
moment hides ~2.3x the travel of 0.9 -- and the travel unit that would carry it
is unreviewed (HANDOVER).

**The user's controller, not built.** lr is a ceiling, set at the start.
When fleet `grad_moment_cosine` falls below zero (over-travel), lr is cut
slowly; it may rise again, never past the start. Its controls are fixed
schedules from the same peak: constant, cosine decay, linear, power. All
harness code, not optimizer arguments.
- For: cosine falls through every run even at a tenth of the lr, as the data
  is used up, so holding it at zero anneals the run on its own evidence; a
  decay restored cosine on both lanes (LFM's low arm, run 24's tail). Late
  runs wanted less lr.
- Against or unknown: at bs32 beta 0.95 cosine ended at -0.0037 with no loss
  cost, so zero is not shown to be the set point. Fleet cosine scatters about
  0.002 per 30-step bin on Anima, which bounds how fast a cut may react;
  `moment_persistence` crosses zero with it at a tenth of the size and is the
  noisier input; `caution_kept` does not see lr.
- How it is settled: LFM 1k, the controller against the four schedules at the
  same peak lr; then a long Anima run, samples deciding. The user decides
  when the controller is ready to test; a long run is the judge, not a probe.

Withdraws "the matrix lr is probably low" (runs 12-13, under the old step
rule): under the law, runs 20-22 ran hot.

### D4. Does beta 0.98 with caution hold on Anima?

Settled on LFM (`ARCHIVE.md`, "beta 0.98 with caution"). Anima has run neither.
Run 20 is the test: bs8, `rank_fraction` 0.025, warmup only, verdict on the
samples. lr is a transfer, not a measurement: the `sqrt(tokens)` rule held on
LFM (bs8 at the same lr lost on both losses), and the unrescaled mask wants
~1.4x the rescaled lr there, so ~9e-4 against the 6.5e-4 rule. `caution_kept`
per role is unread on Anima; on LFM it is uniform, so the unrescaled mask only
redistributes step where it is not.

Why bs8 and not bs16 on this lane: the dataset was curated for bs4, so bs16
pads partial buckets with duplicates (same image, new timestep and noise),
and bs8 avoids nested attention. LFM's 0.012 for doubling the batch is an upper
bound here. Reopens with the 1024 px dataset.

Runs 22 and 23 (`FACTS.md` ledger) moved the bs8 answer. At `5e-5`, a third of
run 22's lr, run 23 held `grad_moment_cosine` at or above zero to ~768, where
run 22 had leaned negative from ~128. It then leaned negative too. Beta moved
with lr, so beta is not separated on Anima. The user's read: run 23 is good
quality but changed little, so the earlier visible advances were the model
breaking and repairing, not learning; at bs8 this data wants a larger batch.
Run 24 then ran a real bs16 at 0.95 and `7e-5` for one epoch; beta is still
not separated on Anima (D7).

### D6. LFM run-to-run spread

A same-seed replicate differs by 1.6e-3 target and 5.6e-3 source (`FACTS.md`),
5x and 3x the floors `AGENTS.md` states. One pair. Settle with three seeds of
the default arm (~40 min) before another LFM claim below ~2e-3 target.

### D7. Should the default beta move to 0.95?

The user's lean: yes. 0.95 won at bs32 because its moment is 2.5x less stale
(time constant 20 steps against 50), not because it averages noise better;
staleness costs step size times memory length, so a large relative step wants
a short memory. Evidence (`FACTS.md`): at bs32 and one lr, 0.95 stepped 1.64x
further, tied target and forgot 0.015 less. At a small held step on bs16
(pre-law, rescaled caution) 0.95 lost 0.004 target to 0.98. A cosine
controller (D1) may let beta go lower still.

At bs16 and a matched step (`FACTS.md`), 0.95 traded 6.8e-3 target for
0.020 source: the beta trade, not a win. The default stays 0.98 (user,
2026-10-03) until beta is re-read under the controller (D1), which cuts lr
on over-travel and may change which memory pays. If 0.95 moves, SPEC, README
and the constructor default move together.

The user's hypothesis for Anima: its smaller relative step tolerates the
longer memory, since staleness is step times memory. Loss cannot show it;
`caution_kept` against its floor, now that the mask reads `O_t`, and
`moment_persistence` can.

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
there to unit scale. Whether that matters has no read yet beyond the per-role cosine: runs 24 and
25 put AdaLN `linear_1` at 5-10x attention's (`FACTS.md`); the verdict is the
samples. The
verdict so far is 19 finite steps (`ARCHIVE.md`). Arm: 0.1 against a lower
fraction under a real bs16, against run 17, with lr lowered to match run
17's per-module step so the comparison reads the subspace and not the step. The
verdict is samples. Frame and moment in bf16 grow from 0.11 to 0.47 GiB at 0.1;
the eigh and polar transients at rank 410 are unmeasured.

### E4. Parked: an arrival sensor

A converged frame orbits on batch noise and a constant turn keeps integrating it.
No sensor exists; the one candidate, `micro_batch_agreement`, left with
accumulation. The user judges this not worth pursuing on this lane now.
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
- **F2. Magic numbers** (was P5). Rank cap (E5); `eps`;
  `AURORA_PP_*` inherited and cheap; Newton-Schulz coefficients (5 steps, the
  largest cost, invisible end to end); `1e-12` floors never audited against
  their dtype -- the lesson of the `1e-6 sigma_max` floor that never fired.
- **F3. Five device syncs per step** (was P12). One is `eigh`'s convergence
  check, one per rank bucket per step, at step end (layerwise release leaves
  the basis update bucketed); `torch._linalg_eigh` checks too. Four unnamed.
  Cost unmeasured; profile first. The user wants a sync-free variant that
  keeps `eigh`'s stationarity, judged by maths, then on real tangents (no
  synthetic). Newton-Schulz fails both on paper: the polar tangent needs an
  exact unit turn on every live plane and a hard cliff at
  `sqrt(r eps) sigma_max` (`_polar_tangent`), and a fixed polynomial gives a
  soft ramp in `sigma / ||Delta||`, so the per-plane turn would follow the
  tangent spectrum as it flattens and half-turn the noise planes that once
  broke Anima's `eigh`. Candidates, unmeasured: batched Jacobi (same
  decomposition, many kernels); Newton inverse square root with `inv_ex` plus
  a matrix-sign cliff (sync-free, exact to roundoff once converged, more
  machinery, no eigenvalues for the spectrum reads); `eigh` one step late off
  the main thread (frame lags a step: a design change).
- **F4. Stratified timesteps within a batch** (was B5). One sample per slice
  of the `t` range; free, and cuts the batch's `t` variance. Not tried.

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
- **Re-polaring the moment before the lift.** Trades target for source at the
  held step and fails at twice the step: noise planes get full length.
- **Rescaling caution by `numel / kept`.** A uniform lr multiplier on LFM, and
  worse source at equal target.
