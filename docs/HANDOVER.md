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

**Run 20 is running**, launched 2026-10-01 ~13:50 from
`ai-toolkit/config/anima_usuitrack_20_b098_caution_bs8.yaml` (gitignored): run
18's config at bs8, `rank_fraction` 0.025, `lr 9e-4`, release defaults (the
`beta: 0.9` pin is gone), warmup 50, cosine over the last 10% to 0.2x,
`activation_offloading` off (plain checkpointing; fits past step 2 where the
unnested bs16 arm OOMed). 1920 steps = 5 epochs, samples every 100, saves every
384, ~5 s/step (0.64 s/sample against bs16-nested's 0.78), about 3 h plus
sampling. Output `/mnt/luna/ai/output/anima_usuitrack_20_b098_caution_bs8`; log
in the session scratchpad `run20.log`.

Read from `loss_log.db` (`metrics(step, key, value_real)`):
`caution_kept/<role>` (new, usuitrack `Read caution_kept per role class`,
ai-toolkit `8f2e624`), `step_gain/<role>`, `usuitrack/*`. Steps 1-5 read
`caution_kept` 0.61 (AdaLN) to 0.75 (`patch_embed`), fleet 0.65 -- inside the
moment's fill, not yet a read. Verdict on samples.

## Run 20's lr

Per unit lr, run 20's step is `0.5` (rank 0.025, `sqrt(r)`) x `0.44` (`w` at
0.98) x `0.77` (the mask's norm on LFM) = 0.17 of run 19's. Holding run 19's
step (bs16, `2e-4`) is 1.2e-3; the `sqrt(tokens)` rule for bs8 gives 8.5e-4;
LFM's no-rescale optimum sat ~10% above the held step, so 9e-4. Derived, not
measured on Anima. Run 18's and 19's configs both pin `beta: 0.9`; any config
copied from them must drop it.
