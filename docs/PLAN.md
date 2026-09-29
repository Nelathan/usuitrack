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

On Anima the moment is at the white floor in every block role (`FACTS.md`,
per-role step), so at `beta 0.9` most of each step cancels over the next ten.
The user's direction (2026-09-29): higher `beta` rather than larger batches to
buy signal, at bs8 (VRAM headroom, twice the steps per epoch), `rank_fraction`
0.025 with `lr 4e-4` (the same step as 0.1 at `2e-4`), warmup only. `step_gain`
reads whether the longer memory turns coherent: it rises above `w(beta)` only
if it does. On LFM `beta 0.99` lost to 0.9 on target at the same lr, i.e. at
0.31x the step when the moment is white, so a held-step arm is unmeasured.

Chosen 2026-09-29: `beta 0.98`, which should cancel noise and not persistence,
and with less thrash may carry a larger lr than the white floor's 2.29x
(`w` 0.229 -> 0.100); Anima at ~1e-3 (held step would be 9.2e-4). Increments
on LFM (bs16, `rank_fraction` 0.1): `beta 0.98` at held step, then caution on
top; then one Anima run with everything, not one Anima arm per lever.

Caution in the frame: mask the update where its sign disagrees with the current
gradient (Liang et al. 2024, arXiv 2411.16085; HeavyBall `_compilable_cautioning`,
which rescales by `numel / kept`), applied to the moment and this step's
gradient in frame coordinates rather than full space, since the full gradient is
freed at `prepare()`. It suppresses the part of the step the current batch
contradicts, which grows with `beta`. Not the bf16 walk (A2).

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
against today's). The instrument exists (`step_gain/<role>`). On Anima the
agreement factor is not a per-role variable at `beta 0.9`: every non-AdaLN role
sits at the white floor `w`, so today's distribution is `w aspect sqrt(r)/||W||`,
known at init, 160x end to end with the stream's two ends hottest (`FACTS.md`).
A law would mostly correct aspect, rank and weight norm. `w` moves with `beta`,
so D4's arms change the step unless `lr` follows `1/w`.

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
- **F2. Magic numbers** (was P5). Rank cap (E5); `beta`/`eps` (D4);
  `AURORA_PP_*` inherited and cheap; Newton-Schulz coefficients (5 steps, the
  largest cost, invisible end to end); `1e-12` floors never audited against
  their dtype -- the lesson of the `1e-6 sigma_max` floor that never fired.
- **F3. Five device syncs per step** (was P12). One is `eigh`'s convergence
  check; four unnamed. Low priority: optimizer time is invisible end to end.
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
