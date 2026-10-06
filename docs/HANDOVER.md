# Handover

Written 2026-10-05, late. Orientation only; `PLAN.md` carries the questions,
`FACTS.md` the insights, `AGENTS.md` how to run things. The 2026-10-04 review
list is in this file's history at `b7b2cd5`.

## Environment

- This repo's venv: Python 3.14 (a local, gitignored `.python-version`), torch `2.14.1+cu132`, and
  `causal-conv1d 1.7.0+cu.13.2.torch.2.14`, a prebuilt wheel from
  `wheels.astral.sh/simple/cu132`. The wheel names its torch minor, so torch is
  held to `2.14.*`; moving torch means picking the matching wheel. Python 3.15
  waits on a cp315 pyarrow wheel. flash-attn is not installed: SDPA already
  dispatches to its flash backend here; the gain, unmeasured, would be packed
  varlen batches (PLAN F5).
- ai-toolkit installs `-e ../usuitrack`, so it runs whichever branch is checked
  out. **Keep `lab` checked out**; build elsewhere in a
  worktree.
- **GPU power limit 198 W**, set at your request. Undo: `pkexec nvidia-smi -pl 220`.
- ai-toolkit venv: `torchaudio==2.11.0+cpu` installed `--no-deps`. Undo:
  `uv pip uninstall torchaudio`.

## Where things stand

**Moment-first is the design on `lab`** (`ARCHIVE.md`, "the moment averages
gradients"). The moment averages unit-Frobenius projected gradients, the step
is `polar(M)` at unit size, `lr` is the relative step itself. No mask, no
`||M||` scaling, `cautious` and `caution_kept` deleted. Defaults `lr 1e-4`,
`beta 0.95`; the harness decays linearly to a fifth (`--lr-final-ratio 0.2`).
D7 closed. On LFM it ties the old design on target under the same decay and,
at 0.95, keeps 0.028 more source; that gain is beta's (FACTS, "Moment first").

Prediction stated before the controls, and how it came out: lab under the
decay at 1.650-1.655 target / 3.08-3.10 source (got 1.6525 / 3.0862 and
1.6537 / 3.0845); moment-first seed 2 within ±2e-3 of seed 1 (1.0e-3).

The branch commit reproduces the runner arm it replaced: `mf-branch-verify-s1`
1.652199 / 3.058867 against 1.652291 / 3.060419.

## Run 28, done

**Anima run 28** (`config/anima_usuitrack_28_moment_first.yaml`, gitignored)
finished 2026-10-06 02:58: run 27's data and shape on moment-first, beta 0.95,
`lr 1.3e-5` flat after a 50-step warmup, cosine over the last epoch to 10%.
Five checkpoints and the samples under
`/mnt/luna/ai/output/anima_usuitrack_28_moment_first/`, ~23 GB. **Verdict is
yours, on the samples against run 27.** Reads in FACTS' ledger.

Open from it: on the plateau `relative_step` is 1.15e-5, 12% under `lr`,
where on LFM it equals `lr`. Some ~12% of the fleet's norm is not stepping,
or the read aggregates differently here; not looked into. The lr was a
transfer of run 27's peak, so with the deficit run 28 travelled ~1.45x run
27, not the ~1.6x planned.

## To review

On `lab`, oldest first:
- `0b3aa99`, `39e6ca6`: transport correction and spin reads.
- `d5e2e58`: torch 2.14 / CUDA 13.2 / causal-conv1d 1.7.0. Reference arm
  inside the noise floor of the torch 2.11 one.
- `ac68bdc`: PLAN F5, BFD packing.
- `27e09de`: **the design change** -- optimizer, tests, SPEC, README, harness
  `--cautious` removal, the `lr`/`beta` defaults. Read SPEC steps 5, 7, 8 and
  Decision 7 first.
- `647b382`: harness linear decay, `--lr-final-ratio`, AGENTS standard arm.
- `6332c7b`: FACTS, "Moment first".
- `1c279b1`: D7's closure; superseded markers on three earlier closures; D1
  and D4 rewritten; the "Already tried" re-polar entry rescoped.
- `5f745d3` and the commit carrying this file: HANDOVER, run 28's ledger row.

Not pushed. `main` is untouched.

## Found, not done

- `examples/*.py` call `UsuiTrack(..., rank=...)` at `lr=4e-4`: the prerelease
  API, broken before this session, and the lr is now 4x the default's meaning.
- Worktree `~/Projects/usuitrack-moment-first` (branch `moment-first`, merged
  into `lab`) can go: `git worktree remove ../usuitrack-moment-first && git
  branch -d moment-first`. Same for `~/Projects/usuitrack-controller` if
  `cosine-controller` is done with.
- LFM throughput is ~1.12 s/step at bs16 on torch 2.14 with the basis-lag
  diagnostic, against the 0.8 AGENTS states; not investigated.
