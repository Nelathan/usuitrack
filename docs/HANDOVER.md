# Handover

Written 2026-10-04, midday. Orientation only; `PLAN.md` carries the questions,
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

- This repo's venv: Python 3.14 (a local, gitignored `.python-version`), torch `2.14.1+cu132`, and
  `causal-conv1d 1.7.0+cu.13.2.torch.2.14`, a prebuilt wheel from
  `wheels.astral.sh/simple/cu132`. The wheel names its torch minor, so torch is
  held to `2.14.*`; moving torch means picking the matching wheel. Python 3.15
  waits on a cp315 pyarrow wheel. flash-attn is not installed: SDPA already
  dispatches to its flash backend here; the gain, unmeasured, would be packed
  varlen batches.
- ai-toolkit installs `-e ../usuitrack`, so it runs whichever branch is checked
  out. **Keep `lab` checked out**; build `main` in a worktree.
- **GPU power limit 198 W**, set at your request. Undo: `pkexec nvidia-smi -pl 220`.
- ai-toolkit venv: `torchaudio==2.11.0+cpu` installed `--no-deps`. Undo:
  `uv pip uninstall torchaudio`.

## Where things stand

D5 closed (agreement inside), E8 closed (eta 0.02), D8 closed: the cautious
mask reads the step's direction `O_t`. The cosine lr controller is closed
(`ARCHIVE.md`): it did not free the ceiling, and what stays is that a decaying
lr beat constant on target. D1 is now the schedule; D7 waits on it.

Branch `cosine-controller` (`f9c50fd`, rebased on `lab` at `92fc559`) keeps
the code, the four forms and every arm's numbers. Not merged and not to be
merged; worktree `~/Projects/usuitrack-controller`. Next candidates: the
schedule controls (D1), and E9 (per-role step, from run 26's reads).

**Withdrawn:** "the travel unit" anchor (run 18's relative step across a beta
change) still stands withdrawn. Also withdrawn, never written to the docs:
"target and source trade along total travel".

## Runs since the last handover

Run 26 (Anima, 5 epochs) and the LFM controller arms are done and read.
Run 26's five checkpoints are on `/mnt/luna`, 20 GB, kept for E9.

## To review

On `lab`, this session, oldest first: `9cbdc3f`, `deede3c`, `c2a138e`,
`36bc436`, `5c24907`, `c187f8c`, `80c2587` (eta 0.02 default), `446354c`,
`5d29a7a`, `ba15e51` (mask against `O_t`: code, SPEC, README), `89c7e6d`,
`85a4237`, `e4ebe36`, `fc4b30b`, `92fc559`, `f85ed10` (the controller's closure) and
the commit carrying this file. On `cosine-controller`, for the record only:
`5099784` .. `f9c50fd`.

Earlier and still unreviewed: `ec47769` .. `1700c3c` (this file at `1700c3c`
lists them). In ai-toolkit: `dab36c3` and `6a6c0c2` (the outside runner). The
run 24/25 configs are local; ai-toolkit ignores `config/`.

On `main`: `d44b1a2`, the squashed release. `lab` is pushed (2026-10-04);
`main` and `cosine-controller` are not. With `lab` pushed, the branches `constant-basis-step`, `trust-ratio`, `remove-adafactor`
and `study/rotation-clamp-eligibility` close.
