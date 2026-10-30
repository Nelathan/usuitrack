# Handover

Written 2026-10-28, updated 2026-10-30, 19:20. Orientation only; `PLAN.md` carries the questions,
`FACTS.md` the insights, `AGENTS.md` how to run things.

## Where it stands

The release cleanup is on `lab`: one optimizer with a minimal surface, its
maths in `docs/SPEC.md`, history moved to `docs/NOTES.md` and `docs/ARCHIVE.md`.
PLAN holds N (norm), S (subspace gradients: S1 accumulation, S2 weight
gradients from both streams), M1 (where magnitude is dropped), T (side), and a
one-line index of the parked ones. Closed: L1, `lr` does not transfer across
sizes, the release defaults to `4e-5`, LFM3's; S3 and S4, frames track from
the streams, the input side from `X` weighted by `delta`'s token norms; the
residual writers keep their output side; a live aim does not make rank 0.1 pay.

**S2 is built**, on branch `projected-grads` (was `x-side`; no upstream):
`--projected-grads` trains with no `dW`, ties `G`'s aim on loss and runs 3.49
s/step against 3.89. A runner with the weight write compiled reaches 3.285
(FACTS, "Where the S2 step's time goes"). Past that the optimizer step is
host-bound.

- **Package:** `frame.py` (fit, Oja aim, constant-angle turn, retraction),
  `polar.py` (balanced Newton-Schulz), `optimizer.py`, `diagnostics.py`,
  `stochastic.py`. Left is right on the transpose; state per matrix is `frame`
  and `moment`. **Old checkpoints do not load** (a `left` moment is now
  `[n,r]`). Polar and turn run per matrix; only the `[r,r]` tangent-Gram `eigh`
  is stacked, by rank (FACTS: 0.887 s/step at 350M; LFM3's peak is backward).
- **Surface:** `lr`, `beta`, `weight_decay`, `rank_fraction`, `side`,
  `release_matrix_grads`, `diagnostics`. Accumulation works without release,
  raises with it. The lr schedule is the trainer's.
- **Telemetry:** on/off, six self-describing reads (`READS`). Transport,
  concentration, persistence and role breakouts are `LabDiagnostics`, installed
  by the harness; `transport_spin` is gone.
- **Harness:** LFM3-1.7B pinned at `545fc8f` is the default model, at the
  standard arm (constant `4e-5` after a 50-step warmup, rank 0.05, 500
  steps, eval every 100); a shorter run is a prefix of a longer one, 1k only
  for the late plateau. The LFM2.5 lane's
  arm is in AGENTS. Lag reads only for tracker diagnostics.
  `--activation-frames {off,shadow,aim}` is the S3 instrument,
  `--stream-frames` the S4 one.
- **Fixed:** fp16 targets were stochastically rounded on bf16's grid; a stale
  group index after `load_state_dict`; hooks missing on later param groups;
  `pop()` making two transfers; a NaN gradient turning `grad_capture` NaN; lab
  frame-motion reads sampling only the first param group.
- **Verified:** 75 tests; golden trajectories from the old code reproduced to
  ~5e-6 relative; `release-clean-s1`, the standard arm: 1.652230 / 3.057482 against the
  reference 1.652199 / 3.058867, 1.133 s/step (FACTS).
  `examples/train_pytorch.py` and `train_hf_trainer.py` run in both modes.
  **Not verified:** `examples/train_unsloth.py` (unsloth not installed).

## Release

`main` is reset to `origin/main`. The release is one squashed commit of `lab`'s
release files onto it, prepared locally after your review; you push.

## Next

Agreed with the user, 2026-10-30 evening, in order:
- The compiled weight write onto `lab` (not released yet); prototype
  `outputs/tools/fused_write.py`, equivalence test in expectation.
- S2's leaves as expanded zeros (most of its +0.29 / +0.88 GB).
- Accumulation under S2: the leaves sum exactly; the first step's ordinary
  path and the harness's release/accumulation check need reworking.
- Norms: normalizing `G` or `Z`, and acting on the weights (norm or polar).
- Triton, after those: the weight write as one pass with the rank-`r` lift
  in-tile (63 -> ~15 ms); S2's per-matrix backward reading `X` and `delta`
  once (~50-100 ms).
- `ProjectedLinear` is a lab feature, not the optimizer: it swaps the user's
  module class and freezes weight grads, an integration shape most users
  would not want. The CUDA graph is optional; clean code first.
- The release `rank_fraction` default is still 0.1; the harness runs 0.05.
- M1 has no arm; its open item (a token-count weight) is predicted below noise.

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

Overnight 2026-10-28, under your handover, each read by a review subagent:
- `b0aade0` harness: `model.train()` after load, the LFM2 checkpointing repair
  deleted, a behaviour test on a tiny real LFM2. Verified by
  `release-trainmode-s1` against `release-clean-s1` (FACTS).
- `f7296a5` FACTS for the 350M constant-lr arms, the re-check, the
  LFM3 lane; AGENTS throughput.
- `5da844e` PLAN L1 evidence, this handover.
- `6103a5b` AGENTS: home-lab work, not frontier; full effort.
- `f183ff7` harness: `--lr-decay {linear,cosine}`, linear default.

Under your handover of 2026-10-29 night, oldest first:
- `f6ae127`, `a6eb4c5` the step: polar and turn per matrix, only the eigh
  stacked; one pass to write, one to turn and commit.
- `fb59777` run conventions (eval at the quarter marks, short runs, lag reads
  opt-in).
- `0791bea` lab reads: `transport_spin` dropped.
- `a634402` the S3 instrument (`experiments/activation_frames.py`, its tests,
  harness wiring).
- `b754b74` ledger: L1 on LFM3, the per-matrix step's speed.
- `622aba8` PLAN: M1 and S3 opened, before their runs.

2026-10-30 day, oldest first:
- `49f36dc` harness: LFM3 the default model at the standard arm; AGENTS lanes.
- `f0f6ee4` release: `lr` defaults to `4e-5`.
- `4f6b132` FACTS: 350M at cosine `4e-5`.
- `b9457e6` L1 closed into ARCHIVE.
- `5a82246` PLAN: S2 as exact weight gradients from both streams; S4 opened.
- `e451d00` `docs/PROJECTED_ACTIVATION_2.md`.
- `88c9bad` `frame.aim` split off `project_and_aim` (release, behaviour-identical).
- `fae7b82` the S4 instrument (`experiments/stream_frames.py`, tests, flag).
- `0803e40` FACTS: the standard arm at 500 steps.
- `7c162d7` FACTS: S4, the weighted `X` frame holds G's capture.
- `ebfa1b5` S3 and S4 closed into ARCHIVE; PLAN S2 carries the frame and the
  side question.
- `209ab14` harness: constant lr, eval every 100.

2026-10-30 evening, branch `projected-grads`, oldest first:
- `68b6973` FACTS: the constant reference (on `lab` as `db4349c`).
- `f1e0626` harness: `--projected-grads` (`experiments/projected_grads.py`,
  tests, flag).
- `825d31c` FACTS: S2's aim x side 2x2.
- `e1b5221`, `85e9782` PLAN S2 open items.
- `aa97ec7` FACTS: the 16-probe `||G||` estimate.
- `17f5de9` ARCHIVE: the residual writers keep their output side.
- `553b527` AGENTS: own the goal (on `lab` as `156e1f6`).
- `3f35de3` FACTS: frame motion under the stream aim; where the step's time
  goes.
- `3dc126d` FACTS/ARCHIVE: rank 0.1 buys capture and no loss.

Run logs: wandb `pink-marker/usuitrack` (`lfm25-c5e5-s1`, `lfm25-c2e5-s1`,
`release-trainmode-s1`, `lfm3-*`, ...); local logs, queue tools (`queue.sh`;
`trajectory.py` for bin means, `window.py` for a step window, `tokens.py`), golden trajectories (`outputs/tools/golden.py check`) and the ai-toolkit
undo record (`outputs/ai-toolkit-freeze-2026-10-28.txt`) live in gitignored
`outputs/`.

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
- What made `lfm3-cos2e5-r05-s1`'s one 21x gradient burst (FACTS); the redo at
  the same seed has none.

## Environment

- Python 3.14 (gitignored `.python-version`), torch `2.15.0+cu132`. No
  published `causal-conv1d` wheel covers torch 2.15 (astral stops at 2.14,
  upstream 1.7.0 has no cp314), so `pyproject.toml` points at a **local build**:
  `outputs/wheels/causal_conv1d-1.7.0-cp314-cp314-linux_x86_64.whl`, sm_89 only,
  this machine only; recipe `outputs/torch215/build.sh` (nvcc 13.3, gcc-15,
  ~10 min). Only LFM2.5's conv layers use it. Verified on 2.15: 72 tests,
  golden <= 6.2e-6, `torch215-std-s1` 1.651945 / 3.057546 at 0.895 s/step
  against `release-eighstack-s1` 1.652025 / 3.059937 at 0.887, LFM3 50 steps
  equal to 2.14 to 6e-5. Undo: `git revert` the switch commit, `uv sync`.
- Discord, zapzap and Zed were closed for the card (allowed; never the desktop
  or ghostty).
- ai-toolkit installs `-e ../usuitrack`: it runs whatever this tree holds.
- **GPU power limit 198 W.** Undo: `pkexec nvidia-smi -pl 220`.
- ai-toolkit venv: `torchaudio==2.11.0+cpu` installed `--no-deps`. Undo:
  `uv pip uninstall torchaudio`.
