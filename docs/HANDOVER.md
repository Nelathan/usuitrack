# Handover

Written at the end of the cleaning session, 2026-09-03. Read `PLAN.md` for the
reasoning; this file is orientation, traps, and what to do next. It is
self-contained: the durable traps and the reproduction recipe from the rank
session are folded in below, not linked.

## Where things stand

Branch `constant-basis-step`, last commit `e354659` "Collapse the
orthogonalization scale dispatch to one implementation". Working tree clean, 38
tests green (was 35).

**What this session did, all cleanup, no behaviour change:** collapsed
`ORTHOGONALIZATION_SCALE_MODE`'s four-way branch and its second, compiled-only
implementation into one `_orthogonalize_update(update, scale)` that
`torch.compile` wraps directly, with the scale isolated in
`_muon_aspect_scale`. Deleted three dead paths in the same cut:
`_expected_projected_grad_shape` (never called -- PLAN's prior description of
the scale term cited this function and was wrong; corrected in place, doc was
the bug), `SubspaceProjector.project_and_back` (never called), and a
one-line Newton-Schulz alias. Verified bitwise-identical to the pre-session
optimizer over 12 steps on four shapes.

**One real win rode along:** the scale being a float let the update buckets
key on `(projected shape, scale)` instead of `(projected shape, original
shape)`, so matrices that share rows and rank but differ in cols -- and so
were needlessly split before -- now share one Newton-Schulz call when their
scale also matches. Not bitwise against running solo (batched vs. solo NS
round differently at the ulp level, same as every other batching already in
this file); verified at the tolerance the rest of the suite uses for that.

**Anima's config now sets `compile_tensor_kernels: true`** (the user's call:
anything running >10min should compile). Untested on this config -- verify the
first launch compiles clean under `release_matrix_grads=True` before trusting
an unattended run.

**Docs were carrying narration that belongs in PLAN, not in SPEC or code.**
Trimmed `_muon_aspect_scale`'s docstring to what it computes; the mismatch
reasoning and measured numbers moved to PLAN carry-over #2. SPEC.md's step 8
now states the current formula and its guarantee only, per the project's own
rule that SPEC never carries history.

**The exact-key-set assertion in the diagnostics test is gone**, at the user's
instruction: a metric added or removed at `core` should need no test update to
be noticed, only a look at what it reads.

## Open work

`PLAN.md`, "Carry-over: open tasks from the rank session," now has ten items.
What changed and what's next:

1. **DONE** -- see above.
2. **Test the scale term for real.** Unblocked, now a one-line patch to
   `_muon_aspect_scale` on both paths. Still not run.
3. CLOSED (`side=right` evidence), unchanged from last handover.
4. **Anima with a calibrated rank table.** Still the second operating point;
   still not run.
7. **Per-role rank needs a home.** Still blocking Anima's use of the table.
   `rank` is already a per-param-group key, so the optimizer supports it.
   Missing: group construction by role (a monkeypatch in the lab today, absent
   from ai-toolkit), and a production per-matrix liveness report that
   accumulates on device and syncs once per interval rather than per step.
10. **New this session.** Reopens whether the rank cap can move from
    `min(m,n)/2` to `min(m,n)` -- SubTrack (`~/code/SubTrack`) uses `min(m,n)`
    directly, and the `/2` here exists only for Oja-residual headroom, not
    because `min(m,n)` itself is wrong. Also records that Aurora
    (`~/code/aurora-release`) was built for exactly this shape: a rectangular
    matrix balanced before the polar map, which is what the projected moment
    already is, so item 1's `_balanced_polar_direction` is Aurora used as
    designed, not adapted. Whether a calibrated rank table or better tracking
    removes the need for the `/2` headroom is open and untested.

## Traps

**The lab fork is stale.** `~/code/optimizers/usuitrack/` is an old copy that
gets shadowed; the harness inserts `usuitrack-release` (a symlink to this repo)
at `sys.path[0]` and raises if the import missed. Never edit the fork.
Confirmed again this session -- it still carries the pre-cleanup names,
`_orthogonalize_aurora` and friends; that is expected, not new drift, and it
remains off-limits to edit.

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
at 1/16 the tokens, use 1/4 the LR. Every bs1 arm in the rank session inherited
bs16's `2e-4` and so ran at 4x too high. Relative comparisons within a sweep
survive because all arms share it; the absolute losses mean nothing.

**Noise floors are 3e-4 on target and 2e-3 on source.** Below that, do not
claim a result.

**Loss does not rank tracker designs, and neither does geometry.** Capture,
`transport_*` and the spectrum reads explain results. The `side=right` arm is
the proof that a geometry read can be maximized while loss degrades, so any
rank rule must clear a loss bar rather than pointing at the metric it optimizes.

**Batched vs. solo Newton-Schulz do not agree bitwise, only at ulp-level float
tolerance** -- true for the bucketing this session added and, unverified but
presumably, for every same-shape batch this optimizer already formed before.
Do not write a bitwise-identity test across a batching boundary; use
`torch.testing.assert_close` with the tolerance the rest of the suite already
carries.

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
user is open -- that reads as being overridden, and it happened in the rank
session. The user's contributions are load-bearing by default, not noteworthy
exceptions.

One instance from this session worth naming: a first cut claimed the new bucket
grouping was "exact, not approximate" because the scale each entry receives is
exact -- true of the scalar, false of the batched output. Caught by writing the
regression test rather than by inspection. The claim in PLAN and the code
comment are both corrected; the lesson is the standing one -- verify the claim,
don't just verify the reasoning that produced it.
