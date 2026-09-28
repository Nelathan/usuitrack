# Handover

Written 2026-09-28 by an agent session the user released to commit finished
chapters for review that night. Orientation only; `PLAN.md` carries the
questions, `FACTS.md` the insights, `AGENTS.md` how to run things.

## Review first (committed without the user's eyes)

usuitrack, branch `constant-basis-step`, on `c6aba51`:

- `70dcf7f` AGENTS paths to `~/Projects`.
- `5d9d947` **C1 closed**: MSE stays; ledger carries your run 14-16 sample reads.
- `de24820` PLAN notes from the discussion (B1, B2, D2, E3's time analog).
- `d0b4d52` **first accumulated step no longer turns the frame.** The moment was
  allocated at the first micro-batch's fold; its existence marks "past the
  fitting step", so micro-batches 2..k aimed the just-fitted frame and turned it
  by eta where k=1 does not. Now allocated at the step's end. Every Anima k=4
  run so far, run 17 included, took that one extra turn at step 1.
- `4d3f4f8` **eta 0.04, constant** (optimizer, SPEC, test, FACTS, PLAN E3).
- `659f3a9` **E6 closed**: energy aim stays; E2 gets the write-role rank lesson.
- `50ab92c` **closes E3**: dropping to `0.01` once the
  frame settles buys nothing, so `0.04` stays constant. Run 17 recorded.

Each commit's tree passes the suite on its own (52 tests).

ai-toolkit, branch `main`: `c35ffe1` removes `patch_equalized_mse`. Run 16's
gitignored config still names it and would no longer run.

Rank as a fraction of shape (your decision; committed while you were on /rc):

- usuitrack `a10275a` code: `rank_fraction` replaces `rank`, RankCalibrator gone
  (50 tests). `a8dd60e` docs: closes E1/E2, opens E7 and D5, AGENTS commands.
- ai-toolkit `ee54d3d`: rank table, calibration and `rankcal/*` removed;
  `rank_fraction` through `optimizer_params`. Checked on a toy model only.
- lab `3c85ec6`: the fork's SUPERSEDED note (path fixed to `~/Projects`) and
  CLAUDE.md. `eda8687`: the harness follows the release -- side from `d_model`,
  clip only a reporting threshold, `--rank-fraction`. These were earlier
  sessions' uncommitted changes plus today's; the harness test module does not
  import, so the functions were checked by direct calls.

bs16 and AdaLN (your go-ahead: "you own it end to end"; committed on /rc):

- ai-toolkit `1f19ead` **block compile in place** (`nn.Module.compile`): the
  swapped-in wrappers saved every block key under `_orig_mod.`, so a
  block-compiled checkpoint did not load. Now 0 of 567 keys; diffusers loads it
  with nothing missing. Touches the shared compile section, all models.
- ai-toolkit `a620ebb` Anima's wrapper exposes `transformer_blocks`, so
  block_compile finds them (it had silently fallen back to whole-model).
- ai-toolkit `edf4380` `activation_offloading: true`: checkpointed block inputs
  in pinned host memory, attn1/attn2 checkpointed again inside the block. bs16
  fits at 0.718 s/sample against 0.621 for plain bs4.
- ai-toolkit `f9bcd12` + usuitrack `7e855dc` **D3 closed**: AdaLN pairs under
  UsuiTrack at the matrices' lr; SPEC/README drop the gates family; D2 closed.
  Evidence is 19 finite steps -- the samples of a real run are still owed.
- usuitrack `9e6a93d` FACTS: the bs16 memory and throughput read.

Accumulation removed (your call: "we dont accumulate, simplify our code"):

- usuitrack `fdfe41d` code + SPEC: `accumulate()`, its two-stage state and
  `micro_batch_agreement` gone; `step()` is sum-then-polar over the real batch,
  bitwise equal to the old un-accumulated path (44 tests).
- usuitrack `c8b4506` docs: **Group B closed** (ARCHIVE); B5 moved to PLAN F4;
  AGENTS drops the two accumulation traps. With `release_matrix_grads`,
  `gradient_accumulation > 1` now raises at the second `prepare()`.
- ai-toolkit `e2dc0ba`: the SDTrainer hook restored to upstream,
  `SplitOptimizer.accumulate` deleted.

D5 instrument and run 18 (your ask 2026-09-29: "build D5, then a 1k LFM, then
commit and push"; you were asleep for all of it):

- usuitrack `4e8eec0` **`relative_step/<role>`**: the caller sets
  `diagnostic_roles` (`{param: role}`); per class, the pre-rounding
  `||lr U||_F` against the class's `||W||_F` at read time. The release gains
  that one attribute. 45 tests, including one checking the read against the realised fp32
  change.
- lab `518f523`, ai-toolkit `a75c44e`: roles are the name minus the first
  numeric segment (the layer/block index), so AdaLN's `.1`/`.2` and
  `ff.net.0`/`.2` stay apart. ai-toolkit checked on a toy module through
  `build_usuitrack_optimizer`, not on Anima.
- usuitrack `da7918b` FACTS: run 18 in the ledger with your read, its step
  measured at 3.3x run 17's, live fraction 0.11 against 0.90, 14 OOM-skipped
  batches.
- usuitrack `35e4375` FACTS/PLAN D5: LFM's per-role distribution (2.1x spread,
  write projections smallest) and a third noise point for the 1k arm.

Pushed: usuitrack `constant-basis-step` only. ai-toolkit's only remote is
`ostris/ai-toolkit` and there is no Nelathan fork, so it was not pushed; the lab
was not named, so it was not pushed either.

## Environment changes

- ai-toolkit venv: `torchaudio==2.11.0+cpu` installed `--no-deps`; no torchaudio
  build matches torch 2.14 and ai-toolkit only calls `torchaudio.save`.
  Undo: `uv pip uninstall torchaudio`.
- The lab venv must run with `uv run --no-sync`, or it rebuilds causal-conv1d
  from source. The lab has uncommitted changes this session did not make.

## In flight

Rank is `rank_fraction` (default 0.1, `c sqrt(mn)`); the table and the live-plane
calibration are gone from the release, the lab harness (`--rank-fraction`) and
ai-toolkit (`rank_fraction` in `optimizer_params`). Anima configs that still
name `rank`, `rank_table`, `calibrate_rank` or `track_live_planes` no longer run.

Nothing is running. Run 18 finished (960 steps, 3.75 h, five epoch checkpoints on
`/mnt/luna/ai/output/anima_usuitrack_18_bs16_rf010`). Run 18 had no
`relative_step`: its config predates the instrument.

## Next

- Run 19: run 18 at higher lr (your lean), maybe more epochs. It will be the
  first Anima run with `relative_step/<role>` in `loss_log.db` -- read it for D5
  before judging the step. Mind the card: run 18 skipped 14 batches on OOM.
- E7: live fraction 0.11 at `rank_fraction` 0.1 on Anima against 0.81 on LFM.
  The rank question is now open with a number on it; lr matching can use
  `relative_step` instead of a derivation.
- D5: the law's axes still await you; the instrument now reads both lanes.
- D4 `beta` at eta 0.04.
