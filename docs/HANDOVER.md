# Handover

Written 2026-09-07, updated 2026-09-12 with run 13. Orientation, traps, what to
do next; `PLAN.md` carries the reasoning. Harness invocations live in
`AGENTS.md`.

## Where things stand

Branch `constant-basis-step`, 50 tests green, committed. Accumulation is built
and validated end to end: calibrated at `k=4`, sized a table from the reading,
and run 13 confirms it on Anima -- see "Run 13, landed" below.

**Gradient accumulation is built.** `accumulate(weight)` folds one micro-batch
into the pending step; `k` calls then one `step()` is accumulation, and a loop
that never calls it behaves as before because `step()` calls it for whatever is
left pending. It lives in projected space, so `release_matrix_grads` is intact and
the effective batch rises without another full gradient resident. The frame is
held across the group and each micro-batch is orthogonalized before the average;
the moment moves exactly once per step, so the telemetry describes the step that
was applied. `PLAN.md` P20 carries the reasoning and the numbers, `SPEC.md` step 5
the semantics.

**The step is self-anchoring in `lr`, measured.** The mean of `k` orthogonalized
directions is shorter than each of them by `sqrt(a + (1-a)/k)`, `a` being
`micro_batch_agreement` -- within 3% over `k` in 1-8. At low `a` that is
`1/sqrt(k)`, the `sqrt`-batch rule arriving through the mechanism. **So `k=4`
without a 2x nominal `lr` is a colder run, not an equal one.**

**`micro_batch_agreement` is the arrival sensor P19 said did not exist.** Mean
pairwise cosine between `k` independently drawn micro-batch directions, which is
the aim's SNR with an analytic zero for pure noise. Published, wired to nothing,
and for the same reason the agreement gain was deleted.

**The rank calibrator publishes a geometric mean** beside mean and median, because
`live ~ r^alpha` is log-linear and the arithmetic mean is pulled by its tail.

**The agreement gain is gone. Every live plane turns by `eta`, and nothing scales
it.** `_anneal_tangent` is now `_polar_tangent`; `AGREEMENT_PLANES`, the `[d,16]`
buffer per matrix, the fleet ceiling, the cold-start zero, and the
`turn_fraction` / `agreement_ceiling` diagnostics are deleted. The full reasoning
is in `ARCHIVE.md` under "the agreement gain is removed"; the short form:

- The correlation test that P17 was built on **was invalid** -- `corr(excess,
  ceiling) ~ 0` is what a *correct* normalizer gives when the numerator's signal
  falls while the divisor rises, which is what runs 9-11 do. Do not cite it.
- What removes the gain is the principle, which never needed that test: the
  ceiling is `effective_rank / k`, a magnitude functional gating a step whose
  entire purpose is that magnitude decides nothing.
- And it did not deliver: 1.21x anneal over 2100 steps on Anima against 3.2x on
  LFM. It may work at a batch that gives contrast. Anima cannot have one.

Its removal is **unfalsified rather than validated**: run 12 turned every frame at
1.8x run 10's rate and nothing downstream moved -- displacement tracked `sqrt(r)`
to 3% per role, `grad_capture` `0.640 -> 0.647`, `live_fraction` 0.893. That is
one real fact, that doubling frame motion is free and the gain was protecting
nothing, and it leaves `eta` as the only handle on frame motion (P16).

`test_every_live_plane_turns_by_exactly_eta` asserts the chordal residual *equals*
`sin(eta)` rather than bounding it. That equality is the whole guarantee now.

**Run 12 is finished, read, and its samples are judged**: progressed, slightly
harmed, one steady trajectory, and the mid-run degradation during the anneal is
gone. The freeze did what the instability hypothesis predicted. It also removed
the creative and aesthetic advancement earlier runs showed, so `fallback_lr`
returns at `2e-6` -- derived, see P18, not guessed.

## What run 12 said

Against run 10 it changed three things -- `fallback_lr: 0.0`, the resized rank
table, the constant turn -- at run 10's lr and schedule.

**The freeze took.** `adaln`, embeddings and `time` read exactly `0.000e+00`
displacement at every checkpoint.

**UsuiTrack alone moves this model a quarter as far.** Whole-model displacement
`2.314e-3` against run 10's `8.800e-3`. Three quarters of every earlier Anima
result's weight motion was AdamW walking the fallback class at `5e-6`.

**Matrix displacement is `lr` and `sqrt(r)` and nothing else.** Every role's
run-12/run-10 ratio equals `sqrt(r_12/r_10)` within 3%, and `ff_up` -- the one
role whose rank did not change -- lands at `0.999`. Freezing three quarters of
the model's motion did not move it; nor did doubling the frame's turn rate
(`transport_speed` `0.00511 -> 0.00931`, as predicted from removing a gain that
averaged 0.505). **Displacement is a budget, not a quality read.** The full table
is in `PLAN.md` P18.

**The rank demand contracts as `live ~ r^0.6`.** Every trimmed role rose about
half as far as a fixed demand predicted; `ff_up`, untouched, moved `+1.5%`, so
there is no run-level drift. Therefore `frac ~ r^-0.4`, `rank <- live / 0.95`
under-corrects by ~2.5x, and the converging rule is
`r_new = r * (target / frac)^-2.5`. Fraction is bought with live planes: the
0.95 target costs 8-18% of the directions the tracker actually turns. **The
target is now a decision, not a default -- discuss it before the next table**
(`PLAN.md` P13 item 4 carries the per-role numbers).

**`grad_moment_cosine` moved for the first time on Anima**: `+0.00357`,
`+0.00347`, `+0.00336` across runs 9-11 (a 5x lr span, no response) and
`+0.00094` in run 12, with buckets crossing zero mid-run. Run 12 changed no lr.
Three candidate causes are confounded in it; the freeze is the one whose
mechanism predicts this sign. **Isolating it is the sharpest cheap run
available** -- run 12's config with `fallback_lr` back at `5e-6`, or run 10's
with only the freeze. If the crossing is real, P14's rule says `5e-5` with a
frozen fallback is near critically damped here.

**Unchanged and healthy:** `grad_capture` 0.647 (0.640 in run 11 -- the leaner
table cost no capture, which is the first evidence the trim was in the right
direction), `nonfinite_grads` 0 throughout, `tangent_live_fraction` 0.893, loss
0.14-0.16 with no trend, 9.7 GB.

Note `transport_curve` and `transport_lag` are absent: run 12 ran at
`diagnostics: core`. Read `transport_speed` on its own or ask for `full` next
time.

## Run 13, landed

`anima_usuitrack_13_acc4`: `k=4`, bs4, 576 steps (run 12's three epochs), `lr
1e-4`, the k=4-calibration table (sum 595 against run 12's 256), `fallback_lr
2e-6`, `diagnostics: core`, `track_live_planes: true`. Clean run: 0 non-finite
gradients, no OOM, 6.6 GB peak. Samples judged good by the user, wants more --
either lr, more steps, or `acc2`.

**Displacement beat every prediction.** Whole-model distance from base (same
script as run 12): **2.960e-3 against run 12's 2.327e-3, in a quarter of the
steps.** More total travel, not a fraction of it. The per-step shrink identity
correctly predicted the *step size* (plateau `update_to_param_ratio` 3.70e-6
against run 12's 2.33e-6 -- the table's `sqrt(r)` ratio, 1.59x) but multiplying
by step count underestimates total travel under accumulation; not yet
explained, see P20.

**The run's own schedule is the direct answer to "more lr or more steps."**
`decay_start_ratio: 0.75` starts cooling at step 432; growth rate fell 5.4x by
the end (3.31e-6/step at 350-450 vs 6.15e-7/step at 550-576). The last quarter
of this run was already spent at a shrinking step, by schedule, not by
saturation -- extending past 576 or holding the plateau lr longer both have
headroom this run's own trajectory shows unspent.

**`grad_capture` climbs through the run rather than sitting flat**: 0.650 at
step 50, 0.73-0.75 by step 300 onward -- above run 12's 0.647 and every prior
run. The mid-run note below calling it rank-insensitive was read too early,
before the basis adapted to the bigger table; superseded.

**`grad_moment_cosine` moved again**, 0.0056-0.0061 mean, above run 12's
0.00094 and near runs 9-11's ~0.0035, with 11 of 115 windows negative (min
-0.0046) -- the same crossing-zero shape run 12 showed. Confounded the same
way run 12 was (rank, lr and `k` all moved together); still unisolated.

**`tangent_concentration` reads 0.582**, below the 0.68-0.82 single-batch
range. Candidate story: averaging flattens the Gram's head, which is the same
mechanism that raised `tangent_live_fraction` -- more planes clear the floor,
fewer are concentrated at the top. Unmeasured in isolation; do not cite past
this session.

## What to do next

1. **A follow-up run spending the headroom P20 found**: hold the plateau lr
   longer (raise `decay_start_ratio`, or extend past 576 at constant `lr`) --
   the schedule, not the mechanism, capped run 13's last quarter. `acc2` is the
   other lever the user named; it halves the shrink to `sqrt(a + (1-a)/2)`
   against `k=4`'s `sqrt(a + (1-a)/4)`, so a hotter step at half the batch
   averaging, a different arm from raising `lr` at fixed `k=4`. Not yet chosen
   between; discuss before running either.
2. **Decide the `live_fraction` target** (P13 item 4). Run 13 settled at
   0.90-0.91 fleet, above the 0.85-0.90 the anchor rule predicted. The
   converging rule is still not the instrument -- P20 -- so this is read
   straight off run 13's per-role fractions, not derived.
3. **P16, `eta`.** Unswept, and with the gain gone it is the only handle on
   frame motion. Cheap on LFM: bs1, 300 steps, a minute an arm.
4. Then P15, P5. P19 has its sensor now and wants a run, not a design.

**There is more VRAM than assumed.** Run 13 sits at 6.6 GB of 12.3 with the bf16
accumulators and a sized table, no allocator warnings. Full diagnostics would
have fit, so the next run that wants `transport_curve` can have it. What does not
fit is a flat oversized rank: rank 256 at `k=4` needed
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to run at all, and then only
at 15.3 s/step against 7.7. Calibrate with a generous *table* (~2x measured
demand), not a flat rank. `bs2 x 8` is the next lever if one is needed, at the
cost of another `sqrt(2)` off the step.

