# Handover

Written 2026-10-03. Orientation only; `PLAN.md` carries the questions,
`FACTS.md` the insights, `AGENTS.md` how to run things. The 2026-10-01 review
list is in this file's history at `0a4d75c`.

## The repo split

`lab` is the root for experiments, cut from `trust-ratio` (which already
contained `constant-basis-step`). It carries the LFM harness in `experiments/`,
the old lab repo's docs frozen in `docs/prerelease/`, and a `lab` dependency
group that `uv sync` installs by default. `main` takes one squashed release
commit from `lab` at a time. Once `lab` is pushed, `constant-basis-step`, `trust-ratio`,
`remove-adafactor` and `study/rotation-clamp-eligibility` close.

From the old lab repo, only the harness and its test came over. The fork's
optimizer, projector and diagnostics tests target the prerelease API (`rank=`)
and the release suite covers each; its state-bytes check moved into
`tests/test_optimizer.py`. `test_llm_harness.py` lost the test of
`FP32StateAdamW`, a class the harness had already deleted, and one stale
assertion; a stub gained the `reduction` argument the harness now passes. `smoke_usuitrack.py` was a toy on the
fork's API and did not come over.

## Environment

- This repo's venv: torch `2.12.0+cu132` and `causal-conv1d 1.6.2.post1`, both
  pinned to the old lab venv's build. With torch 2.13 or `causal-conv1d` 1.7.0
  the CUDA extension failed to load (undefined `c10` symbol). `uv run` no
  longer rebuilds anything; the old `--no-sync` trap is gone.
- ai-toolkit installs `-e ../usuitrack`, so it runs whichever branch is checked
  out. **Keep `lab` checked out**; build `main` in a worktree.
- **GPU power limit 198 W**, set at your request. Undo: `pkexec nvidia-smi -pl 220`.
- ai-toolkit venv: `torchaudio==2.11.0+cpu` installed `--no-deps`. Undo:
  `uv pip uninstall torchaudio`.

## Anima: runs 22 and 23

Both under D5's law, bs8, `rank_fraction 0.025`. Run 22 (`1.5e-4`) was liked
but still hot and slowly degrading; stopped at 1130. Run 23 (`beta 0.95`,
`5e-5`, warmup 100) ran the full 1920: good quality, little change, cosine fine
to ~768 and negative after. Your conclusion: at bs8 this data wants a larger
batch (`PLAN.md` D4). Run 20's sample verdict is still owed.

**Withdrawn:** the previous handover's anchor, "run 18's relative step is
`lr ~1.6e-4` under the law, `~1.1e-4` at bs8", matched per-step ratios across a
beta change. A moment at beta 0.98 hides ~2.3x the travel per step that beta 0.9
shows at the same ratio, so it overstated the transfer. The travel derivation
that replaces it has not been through your review and is not in FACTS.

## To review

Committed while you were away, on your release. On `lab`, oldest first:

- `ec47769` the old lab repo's docs frozen in `docs/prerelease/`.
- `8bc49f9` the LFM harness in `experiments/`, the `lab` dependency group and
  the torch/causal-conv1d pins; `AGENTS.md`'s lanes now run from this repo.
- `7457781` `cautious` as a constructor argument, default `True`.
- `38a6263` core telemetry without a full-matrix pass of its own: the update
  norm read in frame coordinates (the lift isometry, checked on bf16 frames
  to 5.8e-5), `nonfinite_grads` from the raw norm. +6 ms instead of +11.
- `36c993f` release references say "on `lab`".
- `c1e4f3e` runs 22 and 23 in FACTS and PLAN D4/D5.
- `24a9850` `tangent_effective_planes` removed, as you asked.
- `14ed31a` PLAN D1 rewritten as "the global lr is open"; README says the
  constructor's `1e-3` is a placeholder.
- `937919a` per-role reads by name: `diagnostic_role_reads` (from
  `ROLE_READS`, empty by default) breaks any per-matrix read out as
  `<read>/<role>`. `step_gain/<role>` is gone; `update_to_param_ratio` is now
  `relative_step`. The harness takes `--role-reads`.

In ai-toolkit: `dab36c3` passes `optimizer_params.diagnostic_role_reads`
through and gives every `<read>/<role>` its own wandb section. **A config that
wants per-role keys must now name them**, e.g.
`diagnostic_role_reads: [caution_kept, grad_moment_cosine]`.

On `main`: `d44b1a2`, one squashed commit, package, tests, README and SPEC
identical to `lab` at `937919a`, plus a release-only `AGENTS.md`. The
message lists what breaks for existing callers. Nothing is pushed.

## Next, in your order

The batch on Anima. Then rank per role on a step-free footing (E7) and single
overrides. Parked: the travel unit for lr transfer across beta, an AdaLN
`linear_1` override, D6's noise floor.
