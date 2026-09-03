# Open design questions

Working notes, not a specification. `SPEC.md` describes what the optimizer does
today; this file describes what we are still unsure about and what would settle
each thing. See `AGENTS.md` for how this file is worked.

Everything closed lives in `ARCHIVE.md` -- distilled conclusions in its top half,
the frozen former PLAN as an investigation log below them. Where a line here says
*(archive)* the evidence is there.

---

## P2. The aim ignores persistence

The standing question, and the last one with real design surface. It started as a
step-size question; the step size is now derived and the aim is what is left.

**Goal.** An aim that prefers persistent structure to single-batch bursts.

**Hard constraint.** Frame motion must be able to anneal as the frame reaches
equilibrium. Reweighting the aim's spectrum is legal; discarding its magnitude is
not, because magnitude is the only restoring force in the aim. That is why bare
ortho beats the raw aim on loss and still must not ship: a constant angle per
refresh cannot settle. The shipped raw `sigma` does not fully satisfy the
constraint either -- it anneals for ~20 steps and then plateaus.

### What is established

1. **`sigma` is a contrast ratio, scale-free in the gradient.** `A` and `R` are
   both quadratic in `G`; dividing by `mean(diag R)` cancels scale exactly across
   a 1000x range. So `grad_clip_norm` does not set the tracker's scale.
2. **`sigma` is not rank-free.** A nuclear norm over `r` planes, growing roughly
   linearly in rank. Rank is the whole of the scale-freeness problem.
3. **`sigma` anneals for ~20 steps and is then flat for 980.** Any "constant
   `eta`, let `sigma` carry the annealing" proposal is falsified before it is
   built.
4. **Rank 128 is not a bottleneck.** Capture reads `0.64`-`0.74`; two thirds of
   the gradient's energy is inside the frame.
5. **The tracker sits at an equilibrium, not at convergence and not starved.**
   Stable contrast, stable capture, concentration ~0.4 against an isotropic floor
   of `1/128`. The residual is structured, not noise. Whether that equilibrium is
   the *right* one is the open question.
6. **The aim is bad at small batch and stable there.** At bs1 on LFM and bs4 on
   Anima the aim carries ~1.5 effective planes, flat across three epochs of a
   2304-step run. That is a property of the gradients at that batch size, not
   something training improves and not something that decays.
7. **We cannot currently tell whether the frame has converged.** Every released
   metric is floored by target noise. `basis_lag_angle` -- principal angles
   against the frame's own snapshot N refreshes back -- is the only read that
   goes to zero iff the frame stops. Approved as a sampled, opt-in diagnostic
   over ~32 matrices, off the hot path.

### Open leads

1. **Cross-covariance aim, `G^T M`.** The one candidate that adds persistence
   without adding state: replace the second `G` in Oja's `G^T (G Q)` with the
   moment already stored. If the frame has settled and `M`'s EMA converged, the
   target becomes `E[G]^T E[G]` where today's is `E[G]^T E[G] + Cov(G)` -- signal
   alone instead of signal plus per-batch noise. A direction can carry large
   energy every batch while averaging toward zero across them; it contributes to
   the first term and not the second, which is the persistence weighting the
   current aim structurally lacks.

   *Horizontality must be re-derived, and the naive construction breaks it.*
   Today's `R = sym(Q^T A)` works only because `A = SQ` for symmetric
   `S = G^T G`; `Q^T G^T M` has no reason to be symmetric, so symmetrizing would
   remove half the in-subspace component and leave the tangent contaminated with
   motion that corresponds to no subspace motion at all. The fix is exact and
   cheap: skip the Rayleigh step and project directly, `Delta = G^T M -
   Q(Q^T G^T M)`, horizontal by construction at one matmul of today's shape.
   Same cost class, zero new bytes. *(archive)*

2. **Rotation-coupled `beta`.** Damp the moment by how far the frame just turned,
   so memory shortens exactly when rotation makes it stale. Costs nothing new.
   Blocked behind lead 1 -- the optimum moves with tracking speed, and a
   persistence-weighted aim moves it again.

---

## P13. Rank: the table works, Anima has not seen it

1. **ai-toolkit wiring.** Port `build_usuitrack_param_groups` (the side
   heuristic, optional `side_overrides`, `calibration_label` stamping) and the
   `RankCalibrator` drain into `toolkit/optimizers/usuitrack.py`, whose
   `_param_side` is today's hand map. The heuristic reproduces that map
   everywhere except cross-attention `to_k`/`to_v`, which take an override to the
   input side; whether the heuristic's placement is actually better there is a
   question for the run, not a blind change. Blocks item 2.

2. **Anima with a calibrated table and a higher LR.** The second operating point
   for everything the rank work established: whether the `live_frac` ~0.9-0.95
   target transfers off LFM, whether role structure transfers to a DiT, and
   whether the source gain shows up as the sample quality this lane is actually
   judged on. Calibrate at **bs4** -- its training batch, since higher batch wants
   higher rank in ways calibration has to measure. The LR is due a raise on the
   same run: `1e-5` was set against a global rank, and a leaner table takes a
   smaller aggregate step.

3. **The `min(m,n)/2` cap is headroom for the Oja residual, and it also caps
   calibration.** The two hard ceilings are `r <= d` (orthonormal columns) and
   `r <= min(m,n)` (the gradient's own rank); the halving is neither, it exists
   to leave unfitted directions for the tangent to rotate into. SubTrack uses
   `min(m,n)` directly.

   The two open ends are the same end. On LFM's MLP the tracked side is 1024, so
   the cap is 512 -- which is exactly the `r_cal` a calibration run there can
   reach, and `w1`/`w3` still read high `frac` at 512, meaning their measured
   ranks are a lower bound. So "can the cap move to `min(m,n)`" and "calibrate
   deeper than 512" are one question: the cap is what stops the calibration going
   deeper, and a deeper calibration is what would say whether the headroom is
   still needed. Beyond that, usable rank scales with batch -- 101 live planes at
   bs16 against 60 at bs1 at the same rank -- so a ceiling from shape alone cannot
   be right across batch sizes, and `live_fraction` already measures what the
   gradient supports.

**The standing caveat on all of it.** A geometry read can improve while loss
degrades: the `side=right` arm captured 15% more gradient, spread the spectrum
best of any arm measured, and lost on both heads by 27x the noise floor.
Maximizing liveness is not intrinsically good. Any rank rule clears a loss bar or
it does not stand.

---

## P11. The learning rate has not been re-swept since the table

`rank` is no longer a sweep axis -- it is a calibrated table. The LR is, and it
has not moved since before the aim's shape, its magnitude rule, and the rank
allocation all changed. A substantial algorithm change invalidates the sweep
behind it.

**What is known.** Target finishes long before source stops degrading: on the
`4e-4` 1k baseline, steps 300-1000 improved target by `0.0086` while degrading
source by `0.0555`. Halving to `2e-4` was better on **both** axes at every
checkpoint and had still not found its floor at 1k, so `1e-4` is a live question
rather than a formality. Target down / source up is this harness's known
effective-LR axis; a Pareto move on both heads is the only kind that is off it.

**Two things the sweep must control for.**

*The rank/step coupling.* The orthogonalized update has `||.||_F = sqrt(r)`, so
`update_to_param_ratio` tracks `sqrt(r/128)` to four figures. Within a model
across roles that coupling is wanted -- the step scales with the number of
directions the gradient demonstrably supports. But it means **a table change is
also an LR change**: the calibrated table runs 1.2% leaner than its predecessor
and reads `1.651e-4` against a global r128's `1.437e-4`. Compare at matched
`update_to_param_ratio`, not at matched nominal LR. This is also why every rank
comparison in the archive conflates subspace and step, and why none of them can
be read as a clean rank result.

*Stochastic rounding.* SR delivers updates that round-to-nearest used to
discard, so the same nominal LR moves weights further -- more real progress per
step is also more forgetting per step. If that is what the source degradation is,
the fix is the LR and not a mechanism. Settled by running the sweep with
`stochastic_rounding` on and off, same rank table, batch and step count: same
shape with a horizontal offset confirms it, different shapes mean something else.
**Read at step 300** -- past 300 this lane measures forgetting, not learning.

**And there is no historical baseline to lean on.** The newest wandb directory in
the lab predates the lab's own final commit (a 1456-deletion tracking rewrite),
and not one of 197 stored run directories logs `basis_update_interval`. Every
archived number also carries a fallback that was barely training -- bf16 storage
updated through `torch.optim._functional.adamw`, exactly the round-to-nearest
loss `stochastic.py` documents. Fixed now, but the archive is not a clean control
for anything touching update magnitude.

---

## P5. Magic number census

| constant | where | status |
|---|---|---|
| `MIN_GEODESIC_STEPSIZE = 0.01` (`eta`) | step-size floor | calibratable, not derivable; cliff measured at `0.02` |
| `AGREEMENT_PLANES` `k = 16` | agreement meter | a second gain on `eta`'s quantity; accepted, not resolved |
| rank cap `min(m,n)/2` | `effective_rank` | P13 item 3; the fraction is still a choice |
| `grad_clip_norm = 1.0` | raw clip | mandatory; the threshold itself untested across models |
| `beta = 0.9`, `eps = 1e-8` | moment | `beta` measured by sweep; `eps` inherited |
| `AURORA_PP_ITERATIONS = 1`, `AURORA_PP_BETA = 0.5` | direction map | inherited from the method |
| `1e-12` floors | numerical | never read against the precision they run in |

**Goal.** Fewer constants, and the survivors derived or at least scale-free. The
controller half is done (`ARCHIVE.md`); the rows above are what did not yield.

**The lesson from the one that was wrong.** The annealer's liveness test sat at
`1e-6 * sigma_max`, six orders of magnitude below fp32's resolution, so no plane
was ever dead and the null tail was driven on rounding error -- it killed three
runs. It is now `sqrt(r * eps)` relative to `sigma_max`, derived from dtype and
rank. A threshold that never fires is not thereby harmless, and "probably fine"
in a census is a place to look first rather than a row to skip. The three
surviving numerical constants have still never been audited that way.

---

## P6. The frame has no guard against a directional burst

`grad_clip_norm` cannot protect the basis tracker at all: the clip is a uniform
rescale and the Oja tangent is exactly invariant to it. Its entire protective
effect lands on the projected moment, which is linear in `G`. Keep it there -- an
accumulator is hurt by a large contribution and a norm cap bounds exactly that.

The frame is **immune to magnitude bursts by construction and fully exposed to
directional ones.** A gradient of ordinary size pointing at a subspace one bad
batch invented moves the frame as far as a good one. Nothing resists that. So the
frame's guard, if it needs one, is a persistence test rather than a magnitude
test -- which makes this **the same question as P2's aim**, to be resolved with
it rather than separately.

*The trap.* "Already tried" contains the rotation-angle clamp, which ate the
large-`sigma` acquisition regime and made a good basis and a garbage basis read
identically. Any per-step bound on frame motion is adjacent to it. The escape is
that a persistence test bounds motion by *evidence* rather than by a constant --
a different object, but that has to be argued, not assumed.

*Also:* the clip fires on roughly 0.05% of tensors.

---

## P12. Five device syncs per step, and no performance pass yet

CUDA sync debug mode reports **25 synchronizing operations across 5 optimizer
steps**, eager, two matrices, bf16, telemetry off -- and 25 with telemetry on,
which was the question being asked and is settled. The 5-per-step baseline is
not. At least one is structural: `torch.linalg.eigh` checks its convergence info
on the host, once per bucket per basis update. The other four are unaccounted
for.

Finding them is cheap -- `torch.cuda.set_sync_debug_mode("error")` raises with a
traceback at each one. Neither a sync pass nor a broader performance pass has
been run on the release.

**Goal.** Name all five, decide which are load-bearing and which are accidents,
then look at the step path as a whole.

---

## Already tried. Do not repeat.

- **Normalizing the rotation angle by `sigma_max`.** Forces a constant top-angle
  per refresh, re-inflates the residual tail, prevents convergence. Bare ortho is
  the same family -- a different normalization, the same discarded magnitude --
  and reproduces the same failure. The mechanism, not the recipe, is what makes
  this rule transfer to normalizations it never saw.
- **Per-step Frobenius normalization of the gradient inside the basis update.**
  Made `sigma` a scale-free ratio, so it could not self-anneal and a good basis
  and a garbage basis read identically.
- **Clamping the rotation angle to a fixed ceiling.** Eats the large-`sigma`
  acquisition regime, and a clamped telemetry read reproduces the same
  good-basis/garbage-basis collapse.
- **A magnitude cap on the tangent, and a full-rank zero-tangent branch.** Both
  added to a code path no default configuration reached.
- **Overlap reprojection of the moment through frame motion.** Charges a cosine
  tax on rotated directions and hands Aurora weakened coordinates its polar map
  then amplifies. Identity transport in moving-frame coordinates instead.
- **Tangent accumulation** (average `n` Oja tangents in a held frame, one geodesic
  on the mean). Cost target loss monotonically at bs16; at bs4 the cost vanished
  and no gain replaced it. The controller absorbs single-batch noise without it.
- **Per-matrix agreement ceilings** from each matrix's own
  `tangent_participation`. Leaves a 3.63x spread between predicted ceiling and
  observed peak -- *worse* than the 2.36x with no per-matrix term at all. The
  same quantity works at the fleet level, which is what ships.
- **Stochastic rounding on the basis write.** Worse by `sqrt(2)` at every window.
  SR exists to stop a sub-ulp *update* vanishing; the geodesic moves the frame by
  a real angle every time, so bf16 costs the frame variance rather than bias and
  SR trades a half-ulp bound for a full-ulp uniform draw.
- **Turning only the top planes** (few-plane rotation). Worse than baseline on
  target, with lag and spin unmoved. The tail is not the problem; the leading
  plane's magnitude dominating the turn is -- orthogonalizing the tangent so every
  plane turns by the same angle is the version that works.
- **A fresh eigendecomposition mid-training** in place of the tracked frame.
  Worse than tracking. The redesign arc argued the reverse and was wrong.

---

## Reference points

Current-code numbers worth having at hand. LFM2.5-350M, broad-no-embeddings,
bs16 x seq1024, seed 1, `2e-4`, beta `0.9`, 1k steps. Noise floors: target
`3e-4`, source `2e-3`.

| arm | target | source | note |
|---|---:|---:|---|
| global r128 | 1.669330 | 3.084276 | one rank for the fleet |
| per-role table (12,032 planes) | 1.667302 | 3.079098 | Pareto over global |
| calibrated table (11,886 planes) | 1.667920 | ~= | reproduced from `RankCalibrator` |
| `side=right`, r128 | 1.677500 | 3.089930 | best geometry, worst loss |

**Anima**, full finetune, 2B DiT, rank 64, bs4 x 768px, `1e-5`, 2304 steps, under
the derived live floor: wandb `9elbwps6`, clean, model saved. Flow-matching loss
does not rank checkpoints on this lane; the verdict is the samples, and it is the
first non-destructive run -- one trajectory rather than a walk between sample
rounds. The aim does not converge over 2304 steps and does not need to:
participation `0.0234` to `0.0243`, concentration `0.850` to `0.838`, ~1.5
effective planes out of 64, flat across three epochs. Anima at bs4 sits where LFM
sits at bs1. The *frame* converges normally -- `transport_speed` `0.0028` to
`0.0018` while `transport_curve` rises `0.61` to `0.69`. The two models differ in
the aim, not in the frame's motion.

Older run tables, the batch ladder, and the cross-era consistency checks are in
`ARCHIVE.md`.
