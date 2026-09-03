# Handover

Written at the end of the rank session, 2026-09-03. Read `PLAN.md` for the
reasoning; this file is orientation, traps, and what to do next.

## Where things stand

Branch `constant-basis-step`, last commit `e7b450d` "Derive the liveness floor,
and stop turning dead planes". Working tree clean, 35 tests green.

**What shipped in the optimizer** is 30 lines. The liveness threshold in
`_anneal_tangent` is now `sqrt(r * eps)` relative to `sigma_max` instead of a
fitted `1e-6`; dead planes get a zero angle rather than merely a zero tangent
column; and `tangent_live_fraction` joins the core diagnostics. That is the
whole code change. Everything else this session was measurement.

**Anima is unblocked.** The run that died twice in `eigh` within twenty steps
now completes 2304 steps (`anima_usuitrack_8_live_floor`, wandb `9elbwps6`,
final model saved). The user reviewed the samples: first non-destructive run,
anatomy and fingers survive noisy batches, reads as one trajectory.

## What the session established, in one paragraph each

**The floor.** No plane was ever dead under the old threshold, so `1/sigma`
promoted rounding artifacts to unit-norm directions that the polar step turned
at full angle. Those fed back into the next tangent Gram, which is what `eigh`
decomposes -- a rank-poor aim manufactured its own ill-conditioning. Closes the
bs1 crash in PLAN as rank collapse, with mechanism.

**Per-role rank beats one global rank.** LFM bs16 1k, equal budget (12,032
planes against 11,776): target 1.667302 against 1.669330, source 3.079098
against 3.084276. Both heads, so it is off P11's effective-LR axis. A global
rank gives attention roughly twice the rank its gradients can fill and starves
the MLP by a third.

**One over-provisioned calibration is enough.** 150 steps at the training batch,
`rank = live_n / 0.95` floored to a multiple of 8 and capped at the calibration
rank. A role is settled when its calibration frac is below ~0.25. Drift is nil:
suggested ranks are identical across windows 2-12 of a 3600-step run, so no
periodic reporter is needed.

**Rank is also a step-size knob.** `update_to_param_ratio` tracks `sqrt(r/128)`
to four figures across the rank sweep. Desirable across roles within a model,
contaminating across a global rank sweep.

**Geometry does not rank designs, demonstrated.** The `side=right` arm has the
best capture and the best spectrum of any arm this session, and loses on both
losses by 27x and 2.8x the noise floors. Tracking the residual side is harder
precisely because that axis is coupled to the whole model, and that is the axis
worth tracking.

## Open work

`PLAN.md` has "Carry-over: open tasks from the rank session" with nine numbered
leads and the full reasoning. The three that matter first:

1. **Fix the compiled/uncompiled divergence, then collapse
   `ORTHOGONALIZATION_SCALE_MODE`.** The compiled
   `_orthogonalize_aurora_muon_tensor` hardcodes the muon scale and never reads
   the constant, while the uncompiled `_orthogonalize_aurora` honours a
   four-way branch. Two paths that disagree is the defect; the dead modes are
   the symptom. `graft` should not survive.
2. **Per-role rank needs a home so Anima can use it.** `rank` is already a
   per-param-group key, so the optimizer supports it. Missing: group
   construction by role (a monkeypatch in the lab today, absent from
   ai-toolkit), and a production per-matrix liveness report that accumulates on
   device and syncs once per interval rather than per step.
3. **Anima with a calibrated table.** The second operating point for whether
   0.95 transfers, whether the role table transfers to a DiT, and whether the
   source gain shows up as the sample quality the user judges on.

## Traps

**The lab fork is stale.** `~/code/optimizers/usuitrack/` is an old copy that
gets shadowed; the harness inserts `usuitrack-release` (a symlink to this repo)
at `sys.path[0]` and raises if the import missed. Never edit the fork.

**Running a script by path puts its own directory on `sys.path`, not the lab
root.** A runner in the scratchpad must insert `/home/djg/code/optimizers`
itself or `import experiments.llm_synth_smoke` fails.

**A `nohup` wrapper's completion is not the run's completion.** The tool
notification fires when the launcher exits, seconds in. Check the log, not the
notification.

**wandb's live `.wandb` file is unreadable mid-run** and `wandb-summary.json`
does not exist until the end. For ai-toolkit runs read the sqlite at
`<output>/loss_log.db` (tables `steps`, `metric_keys`, `metrics`, column
`value_real`); it carries every `usuitrack/*` metric live.

**LR is not anchored across batch changes.** Scale by `sqrt(tokens per step)`:
at 1/16 the tokens, use 1/4 the LR. Every bs1 arm this session inherited bs16's
`2e-4` and so ran at 4x too high. Relative comparisons within the sweep survive
because all arms shared it; the absolute losses mean nothing.

**Noise floors are 3e-4 on target and 2e-3 on source.** Below that, do not
claim a result.

**Loss does not rank tracker designs, and neither does geometry.** Capture,
`transport_*` and the spectrum reads explain results. The `side=right` arm is
the proof that a geometry read can be maximized while loss degrades, so any
rank rule must clear a loss bar rather than pointing at the metric it optimizes.

## Reproduction

Lab harness, from `/home/djg/code/optimizers`:

```
uv run python experiments/llm_synth_smoke.py \
  --max-steps 300 --batch-size 16 --eval-every 150 --wandb-log-every 25 \
  --seed 1 --rank 128 --usuitrack-lr 2e-4 --beta 0.9 \
  --basis-lag-diagnostic --basis-lag-interval 10 --no-final-sample \
  --wandb-run <name>
```

At bs1 an arm is ~1 minute, bs4 ~0.25 s/step, bs16 ~0.75 s/step. Anima config is
`~/code/ai-toolkit/config/train_full_fine_tune_anima_usuitrack.yaml`; it runs on
a 12GB card at 11/12GB, so batch cannot rise above 4 and `release_matrix_grads`
is what makes it fit at all -- gradient accumulation is incompatible with it.

## Method

The user steers; bring evidence, name the uncertainty, propose the cut, then let
the direction be chosen. State predictions before running. Do not launch a
follow-up sweep or write to docs on your own momentum while a question to the
user is open -- that reads as being overridden, and it happened in this session.
The user's contributions are load-bearing by default, not noteworthy exceptions.
