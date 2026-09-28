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

Running (launched 2026-09-28 ~19:10 via scratchpad `chain.sh`, LFM then Anima):

- LFM `lfm_1k_bs16_rf010_noaccum`: 1000 steps, standard arm, ~15 min.
- Anima `anima_usuitrack_18_bs16_rf010` (gitignored config): real bs16,
  `rank_fraction` 0.1, AdaLN under UsuiTrack, 960 steps = 5 epochs of 3072
  samples, a save per epoch, ~3.5 h. Output `/mnt/luna/ai/output/`; read
  `loss_log.db`. **lr is run 17's 1e-4 unchanged**, so the per-module step is
  not matched: derived ~2x from losing the accumulation shrink, times 1.4-3x
  from the larger rank. A worse sample set says "step too big" before it says
  anything about bs16 or rank.

## Next

- D5's per-role relative-change instrument first: without it, E7 and any
  bs16-vs-run-13 read compare unmatched steps.
- E7: Anima at `rank_fraction` 0.1 against a lower one, lr matched to run 17's
  per-module step.
- D5: one relative unit for the step, instrument first. Axes await the user.
- D4 `beta` at eta 0.04.
