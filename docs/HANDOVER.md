# Handover

Written 2026-09-14, overnight, by an agent session the user handed over at 01:00.
Orientation only; `PLAN.md` carries the questions, `FACTS.md` the numbers,
`AGENTS.md` how to run things.

## Review first (committed without the user's eyes)

usuitrack, branch `constant-basis-step`:

- `5fa176f` **final fold unrounded.** `accumulate(weight, final)`; `step()` folds
  leftovers as final, so k=1 never rounds. Chosen form: a `final` flag rather
  than "trainer skips the last `accumulate()`", because token weights need the
  last micro-batch's weight too. Also rewords `micro_batch_agreement` (proxy, not
  the SNR formula) and `geomean` (skew read, not an estimator). New tests: bf16
  dtype of folds, bf16 k=4 vs k=1 on the update handed to the write.
- `272deaa` **PLAN rebuilt** into groups A-F (273 lines), `FACTS.md` created,
  ARCHIVE gained distilled closures and a CORRECTION section. Check the
  interpretations: A1/A2 (the walk), D2 (fallback target in `lr/rms` units), and
  that "tangent accumulation, do not repeat" is what you meant by "old reasoning
  is dead".
- `60400af` **AGENTS.md** lessons from the 2026-09-13 session.
- Later doc commits record runs 14-16, the walk control and the probes; the
  final one also carries this HANDOVER. Decisions still yours, recorded as open:
  the sizing estimator (mean default; geomean kept as a skew read, PLAN E2), the
  `live_fraction` target (E1), and whether the patch-equalized loss stays (C1,
  pending samples).

ai-toolkit, branch as found:

- `e3bfae4` **token-exact weighting + divide-before-clip**, timestep weight gone,
  the "no accumulate()" raise deleted. **This commit also contains
  `patch_equalized_mse`** (a botched split; `f6a5d2c`'s message records it).
- `f6a5d2c` `scripts/usuitrack_walk_read.py`, `scripts/usuitrack_walk_only_model.py`.
- Run configs are gitignored by the repo (`/config/*`): 14, 15, 16 and
  `sample_walk_only_control.yaml` exist on disk only.

The new hook (weights, final, group division) ran live in runs 15 and 16; see
Results. Nothing checks that token weights actually vary on this dataset -- at
equal bucket sizes every scale is 1.

## The finding that reorders everything: the weights walk

bf16 stochastic rounding on every write. At run 13's step (`3.7e-6` relative),
checkpoint increments are uncorrelated with prior motion and grow as
`sqrt(steps)` for every role, matrices and fallback alike; 88-96% of matrix
coordinates do not change in 50 steps. A toy on a real Anima weight reproduces
it and puts rounding error at ~7x the fp32 travel. Numbers: `FACTS.md`.

What it means, carefully:
- **Distance from base reads the walk**, not learning. Several earlier readings
  are withdrawn (`ARCHIVE.md`, CORRECTION).
- **The expected update is intact** -- rounding is unbiased -- so the walk may be
  harmless noise or may be what "drowned in noise / bland" was. Unknown. `PLAN.md`
  A1.
- It couples to everything with a small step: a nudge-sized fallback lr would be
  almost pure walk (D2); a larger matrix lr raises travel against walk as
  `sqrt(lr)` (D1).

Also corrected tonight in conversation, now in FACTS: `adaln`/`other` are 178M
**2D** linears (not lookups), rms 0.026/0.041, so AdamW's per-step budget at
`2e-6` is ~20x the matrices'. The GPU is at its limit at bs4 x k4 (76%
allocated), not at 6.6 GB.

## Overnight GPU queue

Launched from the scratchpad (`queue.sh`, logs beside it in
`/tmp/claude-1000/-home-djg-code-usuitrack/76ca4e9a-7594-4f2e-a805-dc20009e6310/scratchpad/`):

1. **Run 14 `k1_contrast`** -- run 13 at k=1, `lr 5e-5`, 576 steps, run 13's
   committed-at-the-time code path. Answers B1: did `k=4` or the table raise
   capture/liveness?
2. **Walk-only control** -- base + simulated walk matched to run 13 @550, sampled
   with run 13's prompts and seed. First attempt died (`lr 0` is rejected by
   UsuiTrack); retried at `lr 1e-12` *after* run 16 by `queue2.sh`. Output
   `/mnt/luna/ai/output/anima_walk_only_control/samples`. Compare against run 13's
   step-0 (base) and step-550 samples. Answers A1's first half.
3. **Run 15 `rebaseline300`** -- run 13's config on tonight's code, 300 steps.
   Checks the new hook against run 13's plateau and is run 16's baseline.
4. **Run 16 `patch_equalized300`** -- run 15 with per-patch gradient equalization
   and no timestep loss weight. PLAN C1.

Then, added overnight: **bs1 x k16 throughput probes** (PLAN B3's
prerequisite), gradient checkpointing on and off, 20 steps each, no sampling.

Queue mishaps, all fixed by config and rerun: the control died twice (`lr 0`
rejected; a 25-step warmup inside a 1-step job) and the probes once (same
warmup guard), all now on `lr_scheduler: constant`. Scripts `queue*.sh` in the
scratchpad show the order.

## Results

**Run 14 (B1): `k=4`, not the table, moved the spectrum.** Same table, matched
step. k=1 runs at live fraction 0.55 against 0.90, concentration 0.85 against
0.57, half the moment persistence; capture is mostly the table (0.640 -> 0.716
-> 0.747). Half of run 14's planes were dead: size a table at the `k` it runs
at. Its samples are unreviewed -- worth a look beside run 13's at matched steps.

**Run 15 validated the new trainer hook live** and tracks run 13 on capture,
liveness and concentration at matched steps. Its `moment_persistence` and
`grad_moment_cosine` read higher (1.7e-3 vs 1.3e-3; 7e-3 vs 5e-3) with no noise
floor to judge by.

**Run 16 (C1): patch equalization moved no spectrum read** after step 50 --
prediction not met. Slightly higher capture and early micro-batch agreement. The
samples (16 vs 15 at matched steps) are the remaining verdict.

**Walk-only control (A1, first half): the visible learning is not the walk.**
Walk alone moves samples 4-7x less than run 13 @550 on four prompts (pixel RMS),
shifts composition on the armored-warrior prompt, keeps base style, no visible
damage in the agent's look. Your eye is the verdict: `anima_walk_only_control`
samples vs run 13 step 0 and 550. Longer-run accumulation is not ruled out.

**bs1 x k16 costs ~5% per sample** (0.97 vs 0.92 s); bs1 without gradient
checkpointing OOMs. Per-sample polar-first (PLAN B3) is affordable.

**Run 12's plateau is a pure walk too** (1400 -> 1600): coherent straight-line
share of displacement <= ~7%.

## Samples to look at (nothing overnight was judged by your eye)

All under `/mnt/luna/ai/output/<run>/samples`, files `*__<step>_<prompt>.jpg`:
- walk-only control step 0 vs run 13 steps 0 and 550 -- does the walk harm?
- run 14 vs run 13 at matched steps -- what k=1 looks like on the same table.
- run 16 vs run 15 at matched steps (up to 299) -- the patch-equalized loss.

Disk: runs 14-16 keep five checkpoints each (~24 GB each); the control and the
two probes wrote throwaway end-of-job checkpoints. All deletable once judged.

## What to do next

Pick **one** PLAN group.

- **A (the walk)** is narrower than it looked at 02:00: the control says the
  visible learning is not the walk at run 13's scale. What stays open is whether
  it accumulates into harm over long runs, and a walk-aware displacement read
  (A2). If you want A1 closed, the walk-free fp32-master arm is priced (~+8%).
- **D (step budget)** is now the most actionable: under-travel evidence (D1),
  fallback target in `lr/rms` units (D2), AdaLN under UsuiTrack (D3). Note the
  table must be sized at the `k` a run uses (B1).
- **B2/B3** (polar-first across micro-batches vs sum-first; per-sample at bs1) is
  cheap to run now that bs1 is measured.
