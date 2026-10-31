# Handover

Written 2026-10-28, updated 2026-10-31, night. Orientation only; `PLAN.md` carries the questions,
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

**Molegrad is in the package**, on branch `projected-grads` (rebased onto
`lab`; no upstream): `Molegrad(model, optimizer)`, opt-in beside the `G` path
(user's choice A, 2026-10-31), SPEC "Molegrad", README. No `dW`, the diagonal
aim, no step ever holds `G`, accumulation works; ties `G` through 1k steps on
LFM3 at 3.25 against 3.71 s/step. Resume is tested; a review subagent's five
silent bugs are fixed, each with a test. M1 closed: the magnitude goes at `Z`.
The weight write is compiled on `lab` (-0.22 s/step), not released (PLAN R1).

- **Package:** `frame.py` (fit, Oja aim, constant-angle turn, retraction),
  `polar.py` (balanced Newton-Schulz), `optimizer.py`, `molegrad.py`,
  `diagnostics.py`, `stochastic.py`. Left is right on the transpose; state per matrix is `frame`
  and `moment`. **Old checkpoints do not load** (a `left` moment is now
  `[n,r]`). Polar and turn run per matrix; only the `[r,r]` tangent-Gram `eigh`
  is stacked, by rank (FACTS: 0.887 s/step at 350M; LFM3's peak is backward).
- **Surface:** `lr`, `beta`, `weight_decay`, `rank_fraction`, `side`,
  `release_matrix_grads`, `diagnostics`; `Molegrad` wraps a model and the
  optimizer. Accumulation works without release, raises with it, works under
  molegrad. The lr schedule is the trainer's.
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
- **Verified:** 93 tests on `projected-grads`, 79 on `lab`; golden trajectories from the old code reproduced to
  ~5e-6 relative; `release-clean-s1`, the standard arm: 1.652230 / 3.057482 against the
  reference 1.652199 / 3.058867, 1.133 s/step (FACTS).
  `examples/train_pytorch.py` and `train_hf_trainer.py` run in both modes.
  **Not verified:** `examples/train_unsloth.py` (unsloth not installed).

## Release

`main` is reset to `origin/main`. The release is one squashed commit of `lab`'s
release files onto it, prepared locally after your review; you push.

Before it (2026-10-31):
- Your calls: R1 (the weight write is `@torch.compile`d in the package, while
  README says pure PyTorch); the `rank_fraction` default (0.1; LFM3 and Anima
  run 0.05); `grad_capture` under molegrad (an estimate; dropping it leaves
  the probes no consumer but the non-finite count, which `Z` can carry).
- README's headline (LoRA rank 44 against rank 128 on LFM2.5) is the
  prerelease optimizer's (`docs/prerelease/PLAN.md`): re-measure on today's,
  LFM3 standard arm, LoRA, `G` and molegrad (the harness has the LoRA arm).
- The squash takes a curated file set: `lab`'s `pyproject.toml` (torch
  2.15 pin, local wheel, lab group) and the harness tests stay behind; `main`'s
  suite must pass on its own.
- Run 29's end and the accumulation probe; the suite on the GPU (today's runs
  were CPU-only, the card was busy); `examples/train_unsloth.py` run or cut.

## Next

Molegrad toward release, in order:
1. Your review of the commits below (2026-10-31).
2. `projected-grads` onto `lab`, after your go.
3. Run 29, Anima under molegrad (PLAN S2): samples at 192 good, some better
   than run 28's (you). Launched 2026-10-31 ~19:55 through
   `outputs/anima-mole/run29.sh` (the seeded runner, `NESTED=0`), ends ~22:40;
   output `/mnt/luna/ai/output/anima_usuitrack_29_molegrad`, log
   `outputs/anima-mole/run29.log`. Probes in FACTS, "Molegrad on Anima".
   Queued behind it (`outputs/anima-mole/queue4.sh`): accumulation without
   offloading, bs8 x 2 and bs4 x 4, 20 steps each, against run 29's bs16
   offloaded steps. Your call: drop the nested attention checkpoint from
   `anima.py` for the `G` path too; until then the runner patches it out.
4. The early lead on LFM3, if you want it: a second seed of both arms.

Still open:
- PLAN N2 needs `||W||` per role; the harness logs only the whole model's
  norm. Not a decay arm (user).
- Triton and the CUDA graph: optional, clean code first.
- The release `rank_fraction` default is still 0.1; the harness runs 0.05.

## To review

ai-toolkit, branch `usuitrack-molegrad` (2026-10-31): our 19 commits rebased
onto upstream `e9c72b7`; `usuitrack-release-surface` is the pre-rebase branch.
Conflicts: the vae guard (ours handles None), an import, and `SDTrainer.py`
taken whole from upstream, since our branch's net change there was empty (it
brings upstream's 1/n accumulation scaling). Then:
- `7d94c45` `molegrad` in `optimizer_params`, attached after the training
  placement; the raw-norm check collects first.
- `fc46f38` `warmup_stable_linear_decay`.
- Config `config/anima_usuitrack_29_molegrad.yaml` (untracked, like the others).

usuitrack: `ebf58b7` FACTS "Molegrad on Anima", `e6e3fa6` PLAN S2.

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

2026-10-30 evening, on `lab`, oldest first:
- `db4349c` FACTS: the constant reference.
- `156e1f6` AGENTS: own the goal.
- `df5f3c0` the compiled weight write; `tests/conftest.py` makes dynamo's
  recompile limit a failure.
- `44f601c` FACTS/PLAN: its saving; R1.
- `068d85c` its prose trimmed.

2026-10-30 evening, branch `projected-grads` on top of `lab`, oldest first:
- `52ca890` harness: `--projected-grads` (`experiments/projected_grads.py`,
  tests, flag).
- `11a9819` FACTS: S2's aim x side 2x2.
- `85f4eaa`, `6ff48ce` PLAN S2 open items.
- `68e6cf0` FACTS: the 16-probe `||G||` estimate.
- `86ee3a3` ARCHIVE: the residual writers keep their output side.
- `ce73df1` FACTS: frame motion under the stream aim; where the step's time
  goes.
- `7fb6d49` FACTS/ARCHIVE: rank 0.1 buys capture and no loss.
- `da0d1f6`, `946527f` this handover; `749d778` AGENTS: S2 built.
- `c802f37` PLAN S2: the compiled write is on `lab`.
- `82972eb` S2's leaves as expanded zeros; `0dc6803`, `b5087f1` their
  write-up.
- `472e14d` the probe leaf carries `G P`, so accumulation keeps the estimate.
- `9bfd37f` the frame fitted in its first backward; accumulation works.
- `7c87d61` FACTS/PLAN: S2 accumulates.
- `37c1bc9` PLAN/ARCHIVE: M1 closed.
- `7bc2422` harness: accumulation splits the step's batch, every token
  weighted the same.
- `f07ebd4` FACTS/PLAN: accumulation on matched rows.

2026-10-31, `projected-grads`, oldest first:
- `f7e64e2` S2 moves into the package, behaviour unchanged.
- `1240cfa` resume: the frame lives only in optimizer state; test.
- `ab512be` `step()` collects the projections; closure raises.
- `300e961` bias support.
- `5ada349` refusals for what it cannot project.
- `cc4d745` SPEC/README section.
- `63467b3` review fixes: zero_grad discards leaves, a Linear called twice at
  its fit raises, the fit sanitizes, `step(None)`, ties with any module,
  probes per dtype. Bundled as one review response; each has its own test.
- `6845808` rename: `Molegrad`, `MoleLinear`, `--molegrad`, the diagonal aim.
- `5bf13a5` FACTS correction: the 1k tie figure.
- `e4024fd` FACTS/AGENTS: run-to-run noise (D6) from 273 runs.
- this handover and PLAN S2.
- Runs: `s2-pkg-100`, `s2-review-100` match `s2-join-100` to 1e-4.

2026-10-31 night, `projected-grads`, yours to own (you waived review):
- `efa4e82` SPEC/READS: a moment that fills k of r planes steps at sqrt(k/r).
- `3c5b51f` FACTS: AdaLN's empty planes set `relative_step`; run 29 interim.
- `1848f8e` PLAN/ARCHIVE: S2's Anima bullet; the short step closed as ignored.
- this handover.

Overnight 2026-10-30/31, `projected-grads`:
- `447313d`, `dd53c2f` this handover.
- `1a96068` S2: the `[Q | P]` join; `52a6d61` its FACTS.
- `5c7f185` PLAN N2: no decay arm.
- `553eaa2` FACTS/PLAN: S2 ties `G` through 1k.

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
