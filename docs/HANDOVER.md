# Handover

Written 2026-09-06. Orientation, traps, what to do next; `PLAN.md` carries the
reasoning. Harness invocations live in `AGENTS.md`.

## Where things stand

Branch `constant-basis-step`, 44 tests green. Run 11 is training on Anima and was
launched unattended; nobody has looked at it past step 20.

Three meters changed in the release this session, all of them because a read we
were quoting could not answer the question we were asking it:

- **`tangent_participation` is gone; `tangent_effective_planes` replaces it.**
  Same spectrum, published as a count in `[1, r]` instead of a fraction of `r`.
  The fraction is not comparable across rank settings, and that cost a real
  conclusion: run 10 looked like it had "a third more planes carrying the aim"
  when the count had moved 6% and the rest was a leaner denominator.
- **`raw_grad_norm` and `grad_capture` are new.** `projected_grad_norm` alone
  cannot say whether the frame catches the gradient -- it falls with a leaner
  rank even when every kept plane catches as much as before. Run 10's 19% drop
  against run 9 was exactly its 23% fewer planes. Run 11 reads `grad_capture`
  0.556 at step 20.
- **`track_live_planes`** (ai-toolkit) attaches the `RankCalibrator` at the
  table's own ranks with no rank override, so a *training* run publishes
  `rankcal/<role>/live_fraction` every logging window. The meter never needed the
  oversized rank. This was asked for two sessions ago and argued down; the run-10
  miss is the bill for that.

Also in ai-toolkit: `warmup_stable_cosine_decay` now decays from the end of
warmup when `decay_start_ratio` is omitted. The plateau is the option, not the
default.

## Run 11, unattended

`anima_usuitrack_11_lr1e4_cosine`, config
`config/train_full_fine_tune_anima_usuitrack_11.yaml`, output on `/mnt/luna`,
~2h35 at 4.05 s/it. Against run 10 it changes **two** things and adds two meters:

| | run 10 | run 11 |
|---|---|---|
| lr | `5e-5`, plateau to 75%, floor `1e-5` | `1e-4`, **cosine from warmup**, floor `2e-5` |
| `fallback_lr` | `5e-6` | `5e-6` |
| `beta`, `eta`, table, side map | held | held |
| diagnostics | `core` | `full` + `track_live_planes` |

The lr doubling rests on `grad_moment_cosine` being flat at `+0.0035` across runs
9 and 10 -- a 2.5x lr change moved it by `1e-4` -- so nothing says this lane is
near overshoot. The schedule change rests on run 10's samples reading
perturbed-then-settled. **The displacement read below says that reasoning was
half wrong, and the run is worth having anyway** because the peak lr is the term
that dominates either way.

**A late correction, so the config does not match what was discussed.**
`fallback_lr` was going to rise to `2.5e-5` on the argument that the AdaLN and
norm class was being left behind at a ratio nobody chose. Measuring run 10's
checkpoints refuted the premise -- that class is the *furthest* travelled role in
the model -- so the change was pulled and the run restarted at `5e-6` before it
reached step 30. `fallback_lr` is now `PLAN.md` P18 with the numbers.

## What the checkpoints said, and it matters more than the run

Distance from base weights, run 10, from the five saved checkpoints:

| | 1400 | 1600 | 1800 | 2000 | 2200 |
|---|---:|---:|---:|---:|---:|
| whole model | 7.226e-3 | 7.760e-3 | 8.251e-3 | 8.627e-3 | 8.800e-3 |

**Monotone through the entire anneal, every role, no reversal.** The model is
never dragged home; it only slows. So "it broke free and then fell back" is not
what happened at the weight level -- 0.88% total displacement is simply not far
enough for the change to hold, and the good intermediate samples were transient
points on a short path. The tail shape is a second-order knob.

**And the same read on run 9 says something worse.** At step 2200 the two runs
are 2.7% apart in whole-model displacement, after a 2.5x lr change. The matrix
roles are 48-56% apart; the AdamW fallback class -- AdaLN modulation, norm gains,
embeddings -- is **0.6%** apart, because `fallback_lr` was `5e-6` in both, and
that class is the largest mover in the model. AdamW's step is normalized and
travels at about its lr regardless of gradient scale; the tracker's does not. So
every whole-model displacement number on this lane is mostly a read of AdamW,
and matrix displacement is sublinear in lr: 2.5x the lr bought ~1.5x the travel.
If that exponent holds, run 11 lands ~1.3x run 10 on the matrix roles. That is
`PLAN.md` P18, and it is now the most load-bearing unswept knob here.

The script is `scripts/usuitrack_base_distance.py` in ai-toolkit: it
is the only read that distinguishes travel from churn at the weight level, the
same distinction `transport_curve` makes for the basis, and it costs nothing but
disk. **Take it on every Anima run from here.**

## What to do next

1. **Read run 11.** Samples first -- they are the verdict on this lane and
   nothing else has ever ranked a design here. Then
   `rankcal/<role>/live_fraction`, which closes `PLAN.md` P13 item 4, and
   `transport_curve`, which is the first honest read of whether this basis
   travels or churns.
2. **Run `dist.py` over run 11's checkpoints** against run 10's numbers above.
   If doubling the lr did not roughly double the displacement, the step size is
   not the lever and the run count is.
3. **P17 is the sharpest open question and it is new.** The controller's divisor
   is computed from a magnitude-weighted spectrum that the geodesic discards,
   and the head it measures agreement on is 1.26 effective planes wide. Settling
   it costs one diagnostic and no behaviour change.
4. Then P18 (`fallback_lr`, never swept), P15 (cosine spread), P16 (`eta`).

## Traps

Harness invocations and the operational traps (stale lab fork, `sys.path`,
`nohup`, reading ai-toolkit's sqlite, output on `/mnt/luna`, end-of-job saves, LR
anchoring, noise floors) are in `AGENTS.md`, "Running things". What follows is
research judgement.

**`tangent_concentration` and `agreement_ceiling` are one number.** They
anti-correlate at `-0.95` over both Anima runs, because both are read from the
same tangent spectrum -- concentration is its head, the ceiling its bulk. A story
in which one causes the other is a story about one number. Neither ranks runs 9
and 10 either: their level difference is the rank table, and the drop each shows
at the anneal is present in *both* runs at the same size.

**A metric divided by `r` cannot be compared across rank settings.** This is what
`tangent_participation` got wrong and it is a general hazard on this lane, where
every interesting comparison changes the table.

**A norm is not a capture.** `projected_grad_norm` moves with the rank, the
gradient scale and the basis quality at once. Use `grad_capture`.

**Under ortho-first, `beta` is also an LR.** The step carries the agreement
`||M||/||O|| = sqrt(p(1-w^2) + w^2)`, `w^2 = (1-beta)/(1+beta)`: `0.230` at beta
0.9 and `0.160` at 0.95, so moving to 0.95 is a **0.70x lr cut** and holding the
step needs `1.43x` the nominal lr. This is the trap that has caught four
comparisons.

**A norm read cannot detect a small mean.** `moment_persistence` bounds the
mean's *energy* share, and a mean far too small to move it still decides where a
thousand accumulated steps land. Reading "persistence ~ 0" as "the memory holds
nothing" produced a confident, wrong conclusion; the `beta 0` arm refuted it.

**`grad_moment_cosine` does not respond to lr on Anima.** `+0.00357` and
`+0.00347` across a 2.5x change, no reaction to the anneal either. P14's rule
picked the LR on LFM and is mute here. Stop budgeting runs to protect a clean
read of it until something makes it move.

**Window means on a noisy series are not the series.** Two `grad_moment_cosine`
readings quoted last session as a rise were sub-windows of a flat run; the
step-to-step range is `-0.006..+0.017` against a full-run mean of `0.0035`.
Compute the full-run mean or say the window.

**Loss does not rank tracker designs, and neither does geometry.** The
`side=right` arm maximized a geometry read while loss degraded; the calibrated
LFM table hit `live_fraction` 0.946 exactly and moved loss not at all;
ortho-first moved every mechanism metric the right way and moved loss not at all;
Anima's table raised live fraction by two thirds at less than half the optimizer
state for `2e-3` of loss in the wrong direction. The one hedge: a
better-conditioned basis has shown as subjective quality on Anima where loss does
not move.

**Role structure does not transfer between architectures.** On LFM the MLP roles
were the rank hogs (`w1` 210, `w3` 195) and attention was lean. On Anima it
inverts: attention carries the rank (self-attention K at 52 planes against V's
19) and both feed-forward sides sit near 20 despite being the largest weights.
Calibrate per model; do not port a table's shape.

**The measured live count lands `live_fraction` at ~0.85, not 0.95, and the miss
is per-role.** Run 11's first window already spreads from `attn2_v` at 0.763 to
`ff_up` at 0.900 -- 14 points, on a rule that was being applied as one number.
Read the per-role series before touching the table.

**Take the lower of mean and median.** Under-provisioning is the safe
direction -- a plane the basis lacks is a direction the tracker declines to move,
not an error -- so take whichever statistic runs lower for that role. LFM's
right-skewed roles had mean above median; several Anima roles run mean below it.

**Price optimizer changes end to end, not in the kernel.** A 1.5-2x faster polar
map produced a 0.8% *slower* run. Forward and backward dominate. Runs 9-11 all
carry `compile_tensor_kernels: false` on that evidence.

**Grad norm does not track loss quality here.** Across the `beta` sweep it fell
monotonically (6.24 / 5.94 / 5.12) while target loss got monotonically worse.
Smoother is not better.

**Per-matrix step allocation is loss-invisible so far.** Damping each matrix by
its own agreement changes contested steps 20x and loss by less than a floor, and
what difference there is favours *not* damping. Weigh that against any new
per-matrix scaling proposal.

**Rank calibration needs a long enough run to settle.** Under some side
configurations the high-`frac` roles were still drifting up at 200 steps; 500
settled them on LFM. Anima's 200 steps at `r_cal 256` drifted under 10% and was
accepted deliberately, since under-provisioning is safe. The first window is
always dropped as the acquisition transient -- which is also why a
`track_live_planes` reading at step 20 is a wiring check, not a measurement.
