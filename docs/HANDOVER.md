# Handover

Written 2026-10-03, late evening. Orientation only; `PLAN.md` carries the questions,
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

D5 closed (agreement inside), E8 closed (eta 0.02), D8 closed: the cautious
mask reads the step's direction `O_t`; loss-neutral, and `caution_kept` now
sits on its independence floor. D7: beta stays 0.98 until re-read under the
controller.

The controller (PLAN D1) lives on branch `cosine-controller`, worktree
`~/Projects/usuitrack-controller`, rebased on `lab`. Runs reach it through
`PYTHONPATH=/home/djg/Projects/usuitrack-controller`, so `lab` stays checked
out. Current form: vote EMA at `(1+beta)/2`, `lr_scale = exp(10 min(s, 0))`.
You find the formula too complex; parked for tomorrow.

**Withdrawn:** "the travel unit" anchor (run 18's relative step across a beta
change) still stands withdrawn. Also withdrawn, never written to the docs:
"target and source trade along total travel".

## Running overnight, 2026-10-03

Run 26 `anima_usuitrack_26_controller` (config local; ai-toolkit ignores
`config/`): run 24 with the controller, ceiling `1.4e-4`, constant schedule,
no warmup, beta 0.95, polar mask, 960 steps = 5 epochs, samples and a
checkpoint every epoch, all kept, on `/mnt/luna`. Read it from `loss_log.db`.
Started 22:45, ends about 02:30.

## To review

On `lab`, this session, oldest first: `9cbdc3f`, `deede3c`, `c2a138e`,
`36bc436`, `5c24907`, `c187f8c`, `80c2587` (eta 0.02 default), `446354c`,
`5d29a7a`, `ba15e51` (mask against `O_t`: code, SPEC, README), `89c7e6d`,
`85a4237`, and the commit carrying this file. On `cosine-controller`:
`8e8a209`, `2602806`, `0ef0a8b`, the code to review before the controller
can merge.

Earlier and still unreviewed: `ec47769` .. `1700c3c` (this file at `1700c3c`
lists them). In ai-toolkit: `dab36c3` and `6a6c0c2` (the outside runner). The
run 24/25 configs are local; ai-toolkit ignores `config/`.

On `main`: `d44b1a2`, the squashed release. Nothing is pushed. After you push
`lab`, the branches `constant-basis-step`, `trust-ratio`, `remove-adafactor`
and `study/rotation-clamp-eligibility` close.
