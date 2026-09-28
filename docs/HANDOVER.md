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

## Environment changes

- ai-toolkit venv: `torchaudio==2.11.0+cpu` installed `--no-deps`; no torchaudio
  build matches torch 2.14 and ai-toolkit only calls `torchaudio.save`.
  Undo: `uv pip uninstall torchaudio`.
- The lab venv must run with `uv run --no-sync`, or it rebuilds causal-conv1d
  from source. The lab has uncommitted changes this session did not make.

## In flight

Nothing. E3 closed: the turn matters only while the aim moves (ARCHIVE).

## Next

- E2 rebalanced table (write roles up, read roles to 128, equal planes).
- `beta` 0.95-0.97 at eta 0.04 on LFM (D4).
- D3: AdaLN under UsuiTrack. Its exclusion predates dead-plane masking.
- B2: a real bs16 via activation offloading, against k=4.
