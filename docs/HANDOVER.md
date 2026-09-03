# Handover

Written at the end of the rank-calibration session, 2026-09-03, on top of the
cleaning session's handover. Read `PLAN.md` for the reasoning; this file is
orientation, traps, and what to do next. It is self-contained: the durable traps
and the reproduction recipe are folded in below, not linked.

## Where things stand

Branch `constant-basis-step`, last commit `bcceb63` "Add RankCalibrator, an
opt-in instrument for per-role rank tables". Release repo (`~/code/usuitrack`)
working tree clean, 38 tests green. The lab repo (`~/code/optimizers`) is
**deliberately dirty** and stays that way -- it is just a runner. It carries the
side heuristic, `--calibrate-rank`, and `LFM_RANK_TABLE` with the recalibrated
values; `--projection-side-policy` was deleted there.

**Carry-over #2 (the scale term) is CLOSED.** Three scale rules -- shipped
`sqrt(max(1, rows/cols))`, `unit` (1.0), `fullrank` (`sqrt(min(m,n)/r)`) -- at
LFM bs16, matched effective step. The shipped Muon aspect factor wins both heads.
It is **not** double duty with the per-role rank table: the table sets which
subspace and how many directions, the aspect factor sets how hard fan-out roles
push. Orthogonal levers. The `[m,n]`-shape formula transfers to the subspace
intact. Committed at `7718dd5`; `ARCHIVE.md`, "the Muon aspect factor is not
double duty", has the numbers.

**`RankCalibrator` is built and committed** (`bcceb63`, `usuitrack/diagnostics.py`).
Opt-in via `optimizer.rank_calibrator`, off costs one attribute read on the step
path. `_anneal_tangent` feeds it the `tangent_live_fraction` numerator per matrix
per basis update, tagged with the group's `calibration_label`. `roll()` closes a
window (per label: mean and median over its matrices, one host transfer);
`report()` gives `mean / median / std / frac` over the windows after the first.
**Measurements only** -- turning the stats into a table (mean or median per role,
rounding, clamp to `min(m,n)//2`) is a hand step, out of the tool. An earlier
`live_n / 0.95` rule padded on top of the slack the count already carries and
overprovisioned.

**Side resolution is now a heuristic, not a policy** (lab
`build_usuitrack_param_groups`). Order: an override entry
(`{name_substring: "in"|"out"}`); else the side whose dimension is `d_model`;
else square/ambiguous -> `out` in the name means the output side, else input.
Back-checked against LFM2.5-350M and Anima (Cosmos DiT): matches the hand-tuned
maps everywhere except Anima cross-attention `to_k`/`to_v` `(2048,1024)`, which
get an override to the input side. LFM needs none.

**`LFM_RANK_TABLE` recalibrated and validated.** From a 500-step `r_cal` 512
calibration (`std` 2-9 planes across 24 settled windows), hand-picked as median
with a lean to mean for the right-skewed roles (`w1`, `w3`, `w2`): `w1` 210,
`w3` 195, `conv.in` 185, `w2` 110, `q` 88, `v` 76, `k` 60, `out` 42,
`conv.out` 20 -- 11,886 planes. 1k run: target `1.66792` vs the prior
loss-validated table's `1.66730` (floor `3e-4`), source marginally better,
`tangent_live_fraction` `0.946`, 1.2% leaner. The calibrator reproduces a
hand-tuned, loss-validated table from scratch.

## Open work

`PLAN.md` was rebuilt this session and is now 300 lines of live questions only;
the former 2,600-line file is the bottom half of `docs/ARCHIVE.md`, frozen as an
investigation log, with the closures distilled above it. Read PLAN first -- it is
short now.

Next up is `PLAN.md` P13 items 1 and 2: port `build_usuitrack_param_groups` (side
heuristic, `side_overrides`, `calibration_label` stamping) and the
`RankCalibrator` drain into ai-toolkit's `toolkit/optimizers/usuitrack.py`, whose
`_param_side` is today's hand map, then run an Anima **bs4** calibration at a
raised LR. After that: the LR re-sweep (P11) and the sync/performance pass (P12).

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
