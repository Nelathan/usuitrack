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

## Next

- Accumulation goes: activation offloading for a real bs16 on Anima (verify it
  fits at 768 first), then delete `accumulate()` and its two-stage state, the
  trainer hook, and close Group B.
- E7: Anima at `rank_fraction` 0.1 against a lower one, lr matched to run 17's
  per-module step.
- D5: one relative unit for the step, instrument first. Axes await the user.
- D4 `beta` at eta 0.04; D3 AdaLN under UsuiTrack.
