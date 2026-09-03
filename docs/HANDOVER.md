# Handover

Written at the end of the cleaning session, 2026-09-03, on top of the rank
session's handover. Read `PLAN.md` for the reasoning; this file is
orientation, traps, and what to do next.

## Where things stand

Branch `constant-basis-step`, on top of commit `3366265`. **Uncommitted** --
the user reviewed the diff and the handover this session produced but has not
yet said commit; do not commit without asking. 38 tests green (was 35).

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
7. **Per-role rank needs a home.** Still blocking Anima's use of the table;
   unchanged.
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

Everything in the rank-session handover still holds: the stale lab fork
(confirmed again this session -- it still carries the pre-cleanup names,
`_orthogonalize_aurora` and friends; that is expected, not new drift, and it
remains off-limits to edit), the `sys.path` issue running scripts by path,
`nohup` completion vs. run completion, wandb's unreadable live file, the LR
anchoring rule, the noise floors, and geometry-does-not-rank-designs. Not
repeated here; read the prior handover.

**One new trap from this session.** A batched Newton-Schulz call and a solo
one do not agree bitwise, only at ulp-level float tolerance -- true for the
bucketing this session added and, unverified but presumably, for every
same-shape batch this optimizer already formed before this session touched
anything. Do not write a bitwise-identity test across a batching boundary;
use `torch.testing.assert_close` with the tolerance the rest of the suite
already carries.

## Reproduction

Unchanged from the rank-session handover; see there for the harness command
and per-batch-size timings.

## Method

Unchanged. This session added one instance worth naming: a first cut claimed
the new bucket grouping was "exact, not approximate" because the scale each
entry receives is exact -- true of the scalar, false of the batched output.
Caught by writing the regression test rather than by inspection. The claim in
PLAN and the code comment are both corrected; the general lesson is the
existing one -- verify the claim, don't just verify the reasoning that
produced it.
