# Handover

Written 2026-10-03, evening. Orientation only; `PLAN.md` carries the questions,
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

## Where things stand

D5 is closed: agreement stays inside the step (`ARCHIVE.md`). Runs 24 and 25
were the arm only Anima could run; at a matched fleet step outside lost
agreement twice as early, and you favour inside on the samples.

What is left on the step is your controller (PLAN D1): cosine cuts lr
slowly on over-travel, lr may rise again but never past the start, with
constant, cosine, linear and power schedules as controls. Not built; you
decide when to test it, on a long run. Open beside it: D7 (beta 0.95 as the
default, because staleness scales with step), D8 (the mask against `polar(g)`),
E8 (eta on trial; `0.02` ties `0.04` on loss).

**Withdrawn:** "the travel unit" anchor of the previous handover (run 18's
relative step transferred across a beta change) still stands withdrawn; the
derivation that would replace it is unreviewed. Also withdrawn this session,
never written to the docs: "target and source trade along total travel" --
bs32 at beta 0.95 travelled further and forgot less.

## Queue, 2026-10-03 evening

Scratchpad scripts, each waiting for the previous `.done`:
- `queue29.sh`: the D8 arm, standard LFM arm with the mask against `direction`
  (`polar_mask_runner.py`, runner patch verified to hand `_cautious` the
  direction). wandb `polarmask-lr6.2e-4`.
- `queue30.sh`: the D7 arm, bs16 beta 0.95 at `3.8e-4`, eta 0.04. wandb
  `b095-lr3.8e-4`. Compare against `d5-inside-6.2e-4`.
- Overnight after 22:00: a long Anima run, to be planned with you.

## To review

On `lab`, this session, oldest first:
- `9cbdc3f` FACTS: the LFM lr bracket and the bs32 arms.
- `deede3c` FACTS: Anima runs 24 and 25, and `caution_kept` under its floor.
- `c2a138e` D5 closed into ARCHIVE; PLAN references redirected.
- `36bc436` PLAN: D1 rewritten around the controller; D7, D8, E8 added.
- `5c24907` FACTS and E8: eta 0.02.

Earlier and still unreviewed: `ec47769` .. `1700c3c` (this file at `1700c3c`
lists them). In ai-toolkit: `dab36c3` and `6a6c0c2` (the outside runner). The
run 24/25 configs are local; ai-toolkit ignores `config/`.

On `main`: `d44b1a2`, the squashed release. Nothing is pushed. After you push
`lab`, the branches `constant-basis-step`, `trust-ratio`, `remove-adafactor`
and `study/rotation-clamp-eligibility` close.
