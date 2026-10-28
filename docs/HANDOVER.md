# Handover

Written 2026-10-28, updated overnight to 2026-10-29. Orientation only; `PLAN.md` carries the questions,
`FACTS.md` the insights, `AGENTS.md` how to run things.

## Where it stands

The release cleanup is on `lab`: one optimizer with a minimal surface, its
maths in `docs/SPEC.md`, history moved to `docs/NOTES.md` and `docs/ARCHIVE.md`.
PLAN holds only the questions you brought next (norm, subspace gradients, side)
and a one-line index of the parked ones.

- **Package:** `frame.py` (fit, Oja aim, constant-angle turn, retraction),
  `polar.py` (balanced Newton-Schulz), `optimizer.py`, `diagnostics.py`,
  `stochastic.py`. Left is right on the transpose; state per matrix is `frame`
  and `moment`. **Old checkpoints do not load** (a `left` moment is now
  `[n,r]`).
- **Surface:** `lr`, `beta`, `weight_decay`, `rank_fraction`, `side`,
  `release_matrix_grads`, `diagnostics`. Accumulation works without release,
  raises with it. The lr schedule is the trainer's.
- **Telemetry:** on/off, six self-describing reads (`READS`). Transport,
  concentration, persistence and role breakouts are `LabDiagnostics`, installed
  by the harness.
- **Fixed:** fp16 targets were stochastically rounded on bf16's grid; a stale
  group index after `load_state_dict`; hooks missing on later param groups;
  `pop()` making two transfers; a NaN gradient turning `grad_capture` NaN; lab
  frame-motion reads sampling only the first param group.
- **Verified:** 69 tests; golden trajectories from the old code reproduced to
  ~5e-6 relative; `release-clean-s1`, the standard arm: 1.652230 / 3.057482 against the
  reference 1.652199 / 3.058867, 1.133 s/step (FACTS).
  `examples/train_pytorch.py` and `train_hf_trainer.py` run in both modes.
  **Not verified:** `examples/train_unsloth.py` (unsloth not installed).

## Release

`main` is reset to `origin/main`. The release is one squashed commit of `lab`'s
release files onto it, prepared locally after your review; you push.

## Next

- Packages upgraded around torch, which stays `2.14.*`: `causal-conv1d` has no
  torch 2.15 wheel yet (2026-10-28). Revisit when astral publishes one.
- PLAN L1, lr transfer to LFM3-1.7B: the 350M half ran and matched the
  predictions. The LFM3 half did not run: bs16 peaks in the optimizer's
  batched polar and fit once in three tries, as the desktop's share moved. Your
  call between freeing the card, bounding the polar's batch in the release, or
  a smaller batch (PLAN L1).
- The harness never trained in train mode, so HF checkpointing was off and a
  repair covered LFM2.5. Fixed; the 350M lane is now 0.84 s/step, not 1.13.

## To review

On `lab`, oldest first, each read by a review subagent:
- `8e226b1` release surface: package, tests, lab diagnostics, harness.
- `fcb0562` README and examples.
- `58ef76b` SPEC corrected to the code; NOTES.
- `cb012ab` PLAN emptied into ARCHIVE (two closures, the parked block), FACTS
  row for the arm, AGENTS throughput, this handover.
- `30e6767` dependencies upgraded around the `torch 2.14.*` hold.
- `f47a91a` harness: `o_proj` writes the stream; `--model-revision`.
- `214b78f` PLAN L1 with predictions, written before the runs.

Overnight, under your handover, each read by a review subagent:
- `b0aade0` harness: `model.train()` after load, the LFM2 checkpointing repair
  deleted, a behaviour test on a tiny real LFM2. Verified by
  `release-trainmode-s1` against `release-clean-s1` (FACTS).
- `f7296a5` FACTS for the 350M constant-lr arms, the re-check, the
  LFM3 lane; AGENTS throughput.
- the commit carrying this file: PLAN L1 evidence, this handover.

Run logs: wandb `pink-marker/usuitrack` (`lfm25-c5e5-s1`, `lfm25-c2e5-s1`,
`release-trainmode-s1`, `lfm3-probe3`, `lfm3-c5e5-s1`); the local logs are in the
session scratchpad and will not survive it.

ai-toolkit, branch `usuitrack-release-surface`, `d1eb42f` (its `main`
untouched): `toolkit/optimizers/usuitrack.py` ported (bool
diagnostics, no role classes, always `StochasticAdamW` for the fallback, reads
`_pending`). Its config is gitignored, so the record is here:
`config/train_full_fine_tune_anima_usuitrack.yaml` lost `rank: 64` and
`compile_tensor_kernels: true`, and `diagnostics: full` became `true`. It now runs
at the default `rank_fraction` 0.1; runs 27 and 28 used 0.025. **Set it before
the next Anima run.** The numbered run configs keep their dead keys as records.

## Open, not in PLAN

- Run 28 read `relative_step` 12% under `lr` on its plateau, where LFM reads
  `lr`. That is what the read exists to catch; not looked into.
- The Anima lane has no lab reads now (PLAN T1).

## Environment

- Python 3.14 (gitignored `.python-version`), torch `2.14.1+cu132`,
  `causal-conv1d 1.7.0+cu.13.2.torch.2.14` from `wheels.astral.sh/simple/cu132`;
  torch is held to `2.14.*` by that wheel.
- ai-toolkit installs `-e ../usuitrack`: it runs whatever this tree holds.
- **GPU power limit 198 W.** Undo: `pkexec nvidia-smi -pl 220`.
- ai-toolkit venv: `torchaudio==2.11.0+cpu` installed `--no-deps`. Undo:
  `uv pip uninstall torchaudio`.
