# Handover

Written 2026-09-04. Orientation, traps, what to do next; `PLAN.md` carries the
reasoning. Harness invocations live in `AGENTS.md`.

## Where things stand

Branch `constant-basis-step`. Working tree clean, 44 tests green. The lab repo
(`~/code/optimizers`) is deliberately dirty and stays that way.

**The update path was reordered: the polar map now runs before the average.**
`EMA(ortho(Z))` instead of `ortho(EMA(Z))`. Same cost -- one Newton-Schulz call
per bucket, same bytes -- and it changes two things that matter. Every step
enters the moment with a flat spectrum, so a burst cannot own the memory by
being large; and `||M||` becomes an agreement read the step size follows, where
the old exit polar map restored full magnitude however much the average had
cancelled.

**It is loss-neutral, and that is the honest headline.** Measured against the
old order at matched step, and against a control that re-orthogonalizes the
moment, all three sit within `1.5e-3` on target. It is kept for what it makes
true, not for a delta: the whole path is now scale-invariant, which retired
`grad_clip_norm` entirely, and `moment_persistence` became measurable.

**P2 is closed.** The projected gradient stream is white -- persistence within
`1e-3` of zero at 2-, 10- and 100-step windows. The moment is a variance
reducer, not a persistence extractor; `beta = 0.9` is the measured optimum of a
bias-variance trade; removing it costs `1.8e-2` on target. The cross-covariance
aim (`G^T M`) is dead on the same evidence. Full closure in `ARCHIVE.md`.

**The one real win of the session was the learning rate.** `grad_moment_cosine`
is an overshoot meter and its zero crossing picks the LR: `4e-4` read cosine
`-8.3e-5` and beat the old `2e-4` default on **both** heads (`-2.0e-3` target,
`-4.0e-2` source) at half the aggregate step. See `PLAN.md` P14.

## What to do next

1. **The cosine spread read** -- one extra accumulation, decides whether a
   per-matrix LR controller has anything to exploit. `PLAN.md` P15.
2. **Anima**: port `build_usuitrack_param_groups` and the `RankCalibrator` drain
   into ai-toolkit, then a bs4 calibration. Expect a *higher* `beta` to win
   there than on LFM -- more noise means more averaging is worth more staleness.
   `PLAN.md` P13.
3. Then `eta` (P16). The sync and performance pass (P12) dropped in priority:
   the polar map's dtype was measured at 1.5-2x in the kernel and **zero** end to
   end, so optimizer-side time is not where this lane spends it.

## Traps

Harness invocations and the operational traps (stale lab fork, `sys.path`,
`nohup`, reading ai-toolkit's sqlite, LR anchoring, noise floors) are in
`AGENTS.md`, "Running things". What follows is research judgement.

**Under ortho-first, `beta` is also an LR.** The step carries the agreement
`||M||/||O||` -- `0.23` at `beta 0.9`, stable across a 4.4x LR change because it
is a property of the gradients. So two arms at the same nominal LR and different
`beta` sit at different effective steps by up to 8x. Set each arm's LR to
`lr_ref * agreement_ref / agreement_arm` and check `update_to_param_ratio`
afterwards. This is the trap that has caught four separate comparisons.

**A norm read cannot detect a small mean.** `moment_persistence` bounds the
mean's *energy* share, and a mean far too small to move it still decides where a
thousand accumulated steps land. Reading "persistence ~ 0" as "the memory holds
nothing" produced a confident, wrong conclusion this session; the `beta 0` arm
refuted it by `1.8e-2`.

**Loss does not rank tracker designs, and neither does geometry.** The
`side=right` arm maximized a geometry read while loss degraded; the calibrated
table hit `live_fraction` 0.946 exactly and moved loss not at all; ortho-first
moved every mechanism metric the right way and moved loss not at all. The one
hedge: a better-conditioned basis has shown as subjective quality on Anima where
loss does not move.

**Price optimizer changes end to end, not in the kernel.** A 1.5-2x faster
polar map produced a 0.8% *slower* run. Forward and backward dominate; the
optimizer's largest term is invisible in the wall clock.

**Grad norm does not track loss quality here.** Across the `beta` sweep it fell
monotonically (6.24 / 5.94 / 5.12) while target loss got monotonically worse.
Smoother is not better.

**Per-matrix step allocation is loss-invisible so far.** Damping each matrix by
its own agreement changes contested steps 20x and loss by less than a floor, and
what difference there is favours *not* damping. Weigh that against any new
per-matrix scaling proposal.

**Rank calibration needs a long enough run to settle.** Under some side
configurations the high-`frac` roles were still drifting up at 200 steps; 500
settled them (`std` 2-9 planes). The first window is always dropped as the
acquisition transient, but that alone is not enough.

**mean vs median across a role is a real choice.** They agree for homogeneous
roles; for right-skewed ones (`w1`, `w3`, `w2` on LFM) the mean runs well above
the median because a few layers carry real high rank. Median as the base, lean
to mean there.
