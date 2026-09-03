# Handover

Written 2026-09-04. Orientation, traps, what to do next; `PLAN.md` carries the
reasoning and is short now. Harness invocations live in `AGENTS.md`.

## Where things stand

Branch `constant-basis-step`, HEAD `222c9b0`. Working tree clean, 40 tests green.
The lab repo (`~/code/optimizers`) is deliberately dirty and stays that way.

**The docs were rebuilt.** `PLAN.md` is ~370 lines of live questions only. The
former 2,600-line PLAN is frozen as the bottom half of `docs/ARCHIVE.md`, with
closed questions distilled above it. `AGENTS.md` now carries the working
contract, code style, and how to run both lanes. Nothing else should grow into
a diary again.

**Closed since the last handover.** The `eta` cliff was dead planes, not step
size -- `0.05` runs clean at bs1, 2.5x past the old `0.02` divergence point, so
`eta` is an ordinary unswept knob rather than a constant with a wall under it
(`0.05` is worse on target, so `0.01` stands). The `min(m,n)/2` cap is settled:
`r <= d` is linear algebra, the halving keeps energy in the residual so a tangent
exists, and `min(m,n)` failed on Anima with an empty residual. P6 closed -- the
full-spectrum rotation with dead planes masked took magnitude out of the turn,
which is what a directional burst used to exploit.

**Built this session: the moment now follows the frame's in-span rotation.**
Identity transport is exact for the geodesic and not for the frame that gets
*stored* -- bf16 rounding rotates the columns inside their own span, which
scrambles the moment one-for-one. The overlap is read from the stored frames,
its orthogonal polar factor applied to the moment, and the moment committed once
per step with stochastic rounding instead of twice with round-to-nearest.
Newton-Schulz gets an fp32 moment as a side effect. First measurement is a hint
at ~2x the noise floors on both heads, confounding three changes, and it is not
expected to pay at `beta = 0.9` anyway. See `PLAN.md` P2 lead 2.

## What to do next

1. **The `beta` sweep** -- `0.9 / 0.95 / 0.99`, with and without the rotation.
   This is the actual test of what was just built: the claim is that it raises
   the ceiling on memory, not that it helps at today's setting.
2. **Re-check the rotation under the per-role table.** The measurement ran at
   global `r128`; `r` varies per role under the table and so does the rotation.
3. **Anima**: port `build_usuitrack_param_groups` and the `RankCalibrator` drain
   into ai-toolkit, then a bs4 calibration at a raised LR. `PLAN.md` P13.
4. Then the LR re-sweep (P11) and the sync/performance pass (P12).

## Traps

Harness invocations and the operational traps (stale lab fork, `sys.path`,
`nohup`, reading ai-toolkit's sqlite, LR anchoring, noise floors) are in
`AGENTS.md`, "Running things". What follows is research judgement, not mechanics.

**Loss does not rank tracker designs, and neither does geometry.** The
`side=right` arm maximized a geometry read while loss degraded; the calibrated
table hit `live_fraction` 0.946 exactly and moved loss not at all. Any rank rule
must clear a loss bar, not point at the metric it optimizes. The one hedge: a
better-conditioned basis has shown as subjective quality on Anima where loss
does not move -- geometry is not worthless, it is just not a loss predictor.

**Rank calibration needs a long enough run to settle.** Under some side
configurations the high-`frac` roles were still drifting up at 200 steps; 500
settled them (`std` 2-9 planes). The first window is always dropped as the
acquisition transient, but that alone is not enough. Compare `mean` across
successive `report()` prints before trusting the number.

**mean vs median across a role is a real choice, not a default.** They agree for
homogeneous roles; for right-skewed ones (`w1`, `w3`, `w2` on LFM) the mean runs
well above the median because a few layers carry real high rank. Median as the
base, lean to mean there.

**`update_to_param_ratio` is the comparison axis, not nominal LR.** Anything
that changes rank changes the aggregate step, so two arms at the same LR can sit
at different effective step sizes and their losses will read off P11's
target-down/source-up axis rather than off the thing being tested.
