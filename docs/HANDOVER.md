# Handover

Written 2026-10-01 by a session the user released to commit ("you decide on
commit structure and timing; I can't review today"). Orientation only;
`PLAN.md` carries the questions, `FACTS.md` the insights, `AGENTS.md` how to run
things. The 2026-09-28 review list this file used to carry is in its history at
`d334fa2`.

## Review first (committed without the user's eyes)

usuitrack, branch `constant-basis-step`, on `d334fa2`:

- `94bec58` deletes `REORTHOGONALIZE_MOMENT`, the losing control arm.
- `f74be22` **new defaults**: `beta 0.98`, caution on, no `numel / kept`
  rescale. `CAUTIOUS` stays a module constant (your call) for the control arm.
  SPEC steps 5 and 8, README (example lr 4e-4 -> 7e-4, a caution sentence).
  One test changed for a reason other than the design: `step_gain`'s pair
  group at lr 1e-1, since beta 0.98 left lr 1e-3's change near fp32
  resolution. 47 tests.
- `e2f814a` FACTS: the ten arms of 2026-10-01.
- `8b70037` D4's LFM half to ARCHIVE; PLAN D4 is Anima's half; D6 (noise floor)
  opens.
- `22dce39` AGENTS: the standard LFM arm is now 1k steps at the new defaults.

Not pushed.

## Environment changes

- **rtk hook removed** from `~/.claude/settings.json` at your request (it
  swallowed `grep -h` across several files). Undo: restore the `PreToolUse`
  `Bash` hook running `rtk hook claude`.
- ai-toolkit venv: `torchaudio==2.11.0+cpu` installed `--no-deps`; no torchaudio
  build matches torch 2.14 and ai-toolkit only calls `torchaudio.save`.
  Undo: `uv pip uninstall torchaudio`.
- The lab venv must run with `uv run --no-sync`, or it rebuilds causal-conv1d
  from source.

## In flight

Nothing is running. Run 20 is discussed before launch.

## Run 20, before launch

- **Copying run 19's config carries `beta: 0.9` into `optimizer_params`**,
  which overrides the new default. Drop it.
- lr, derived and not measured on Anima: run 19 was bs16, `rank_fraction` 0.1,
  `beta 0.9`, `2e-4`. Per unit lr, run 20's step is `0.5` (rank 0.025,
  `sqrt(r)`) x `0.44` (`w` at 0.98) x `0.77` (the mask's norm on LFM) = 0.17 of
  run 19's. Holding run 19's step at bs16 is 1.2e-3; the `sqrt(tokens)` rule
  for bs8 gives 8.5e-4; LFM's no-rescale optimum sat ~10% above the held step,
  so ~9e-4.
- `usuitrack/caution_kept` is fleet-wide. Whether Anima's roles disagree with
  the batch unequally -- the case where the missing rescale matters -- has no
  per-role read.
