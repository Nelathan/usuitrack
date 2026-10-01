# Handover

Written 2026-10-01 by a session the user released twice: to commit ("you decide
on commit structure and timing; I can't review today"), and after run 20 to
carry PLAN D5 through ("complete this quest faithfully"). Orientation only;
`PLAN.md` carries the questions, `FACTS.md` the insights, `AGENTS.md` how to run
things. The 2026-09-28 review list is in this file's history at `d334fa2`.

## Review first (committed without the user's eyes)

usuitrack, branch `constant-basis-step`, on `d334fa2`:

- `94bec58` deletes `REORTHOGONALIZE_MOMENT`, the losing control arm.
- `f74be22` **new defaults**: `beta 0.98`, caution on, no `numel / kept`
  rescale. `CAUTIOUS` stays a module constant (your call) for the control arm.
  One test changed for a reason other than the design: `step_gain`'s pair
  group at lr 1e-1, since beta 0.98 left lr 1e-3's change near fp32 resolution.
- `e2f814a` FACTS: the ten arms of 2026-10-01.
- `8b70037` D4's LFM half to ARCHIVE; PLAN D4 is Anima's half; D6 opens.
- `22dce39` AGENTS: the standard LFM arm is 1k steps at the new defaults.
- `cd9913d` `caution_kept/<role>`, with a test.

Branch **`trust-ratio`**, off `constant-basis-step` at `064e89c` -- the D5 law,
kept on its own branch so it can be dropped whole:

- `679f68c` **design change**: the step is sized against `||W||_F`. Directions
  are unit (`1/sqrt(r)`), the moment's norm is the agreement, the lift is
  scaled by the weight's norm, so `||dW||/||W|| = lr x agreement x caution` in
  every module. Muon's aspect factor and `sqrt(r)` are gone, and with them
  `original_shape`. SPEC step 8 rewritten (LAMB citation, why weight norm and
  not shape, the zero-norm limit), README magnitude sentence. Tests: the
  first-step test now sizes by `||W||`; the aspect-pinning test is replaced by
  one that fails on the old code (30x the weight must be 30x the step, and
  four shapes land at `lr (1-beta)`); the bucket test now keys on projected
  shape. 48 pass. **Do not resume an old run's optimizer state across it** --
  the moment is in a different unit.
- `6b851f2` PLAN D5: the law, inside vs outside, the criteria fixed before any arm.
- `9386ce8` FACTS: run 20's per-role decomposition and fleet reads; ledger row.
- `a677283` FACTS/PLAN/ARCHIVE: the LFM veto passed; the aspect closure gets a
  "superseded" pointer.
- `1fd11f0` AGENTS standard arm `6.2e-4`, README example `6e-4` (the law's unit).

ai-toolkit `8f2e624`: `caution_kept/<role>` gets its own logging section.

Not pushed. Two decisions here are mine, not yours, and want your eyes:
**agreement inside** the step (you asked; I chose inside, outside was run as the
control and does not separate on LFM), and **no floor on `||W||`** (a
zero-norm matrix never moves; documented as a limit).

## Environment changes

- **rtk hook removed** from `~/.claude/settings.json` at your request. Undo:
  restore the `PreToolUse` `Bash` hook running `rtk hook claude`.
- ai-toolkit venv: `torchaudio==2.11.0+cpu` installed `--no-deps`. Undo:
  `uv pip uninstall torchaudio`.
- The lab venv must run with `uv run --no-sync`, or it rebuilds causal-conv1d.
- ai-toolkit and the lab import this working tree, so **they run whichever
  branch is checked out** -- `trust-ratio` right now.

## Run 20 -- done, your sample verdict owed

`anima_usuitrack_20_b098_caution_bs8`: bs8, beta 0.98, caution,
`rank_fraction 0.025`, `lr 9e-4`, 1920 steps. Its relative step was 1.67x run
18's at half the batch -- I derived its lr from sensor run 19, never
sample-judged, instead of from run 18, which was; that is my error. Fleet
`grad_moment_cosine` leaned negative from step 200 and turned positive only in
the anneal's last bucket (`FACTS.md`, "Run 20 against run 18"). Ledger row
says "verdict pending".

## In flight: run 21, the law on Anima

Launched 2026-10-01 ~17:50 from `ai-toolkit/config/anima_usuitrack_21_trust_ratio.yaml`
(gitignored): run 20's config, only name and `lr 3.2e-4` changed -- run 20's
bulk relative step (2.4e-5) over attention's agreement x caution (0.075).
Output `/mnt/luna/ai/output/anima_usuitrack_21_trust_ratio`, log
`run21.log` in the session scratchpad. ~3 h.

At steps 20-40 the law reads as designed: every non-AdaLN role at 0.061-0.068
per unit lr (run 20: 0.017-0.175), `patch_embed` 0.97x the bulk (was 6.5x),
`proj_out` 1.0x (was 4.5x), AdaLN 0.022-0.033, held at ~0.4x by its own
agreement.

How it is judged (PLAN D5): no role's `caution_kept` falling toward 0.5, and
fleet `grad_moment_cosine` not leaning negative as run 20's did; samples veto
(colour, composition, text), they do not rank. Note this arm matches run 20's
bulk step, which the fleet reads called too high -- it isolates the law, so a
cosine lean that persists here is lr, not the law.

## Next, in your order

Rank per role on a step-free footing (E7) -- the law makes rank a pure subspace
knob, so rank arms no longer move the step. Then global lr: run 18's relative
step (1.2e-5) is `lr ~1.6e-4` under the law, `~1.1e-4` scaled to bs8 by
`sqrt(tokens)` -- derived, not measured.
Then single overrides; `patch_embed` may need none.
