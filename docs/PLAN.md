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

2. **`beta` is capped by smearing, and the frame's in-span rotation is never
   applied to the moment.**

   *Why `0.9` and not higher.* The moment lives in frame coordinates, so as the
   frame moves, older contributions describe directions the frame has partly
   left. `beta` is bounded by frame motion, not by anything about the moment
   itself. `0.9` ships because it is strong on target and stable enough; it can
   rise only once the basis settles or the moment moves with it.

   *The original construction was worse and is gone.* It projected the moment up
   through the old basis and back down through the new one, discarding whatever
   the new frame does not span -- under real frame motion that destroys
   persistence outright. Identity transport in moving-frame coordinates replaced
   it and loses nothing: a single Grassmann geodesic along a horizontal tangent
   gives `Q^T Q+ = V cos(theta) V^T`, exactly symmetric, so the coordinates carry
   over unchanged.

   ***`transport_spin` is exactly the part identity transport gets wrong, and
   nothing corrects it.*** Spin is the skew of `Q_old^T Q_now`: rotation of the
   frame's columns inside the span they already had, which moves the subspace not
   at all and scrambles the moment one-for-one. The optimizer measures it and
   acts on it nowhere. Two sources, behaving differently:

   - **genuine holonomy** -- fp32 reads `7e-8` at window 1 rising to `9e-6` per
     update, coherent. The composition of symmetric steps need not be symmetric,
     so this is a property of the path, not an error in any single step.
   - **bf16 rounding on the basis write** -- `5.6e-4` at window 1 where exact
     arithmetic gives zero, a pure random walk with no floor. Sixty times the
     holonomy per update.

   Orthogonalizing the tangent already removed the largest source. Raw `sigma`
   let one plane own the turn and read **7x the spin of any other arm** -- the
   frame twisting its own coordinates against the moment. (A 20x recollection is
   in circulation; the factor in the record is 7x.)

   ***The moment rotation is built*** (`SPEC.md`, step 6). The overlap is read
   from the frames as stored, its orthogonal polar factor applied to the moment,
   and the whole step now rounds the moment once instead of twice -- the moment
   stays fp32 from its accumulate through the polar map to a single stochastic
   commit after the rotation. An exact step returns the identity, pinned by
   `test_frame_rotation_is_the_identity_when_transport_is_exact`.

   *A frame-side version was considered and is dead.* Choosing the aligned
   representative of the new frame is a no-op in exact arithmetic, which reads
   as a safety property and is actually the disproof: the fp32 geodesic overlap
   is already symmetric to `7e-8`, so there is nothing there to correct, and the
   `5.6e-4` arrives at the bf16 *write*, downstream of any alignment. Writing
   again to fix it rounds again.

   **First measurement, LFM bs16, r128 global, 300 steps, seed 1, beta 0.9,
   against a matched control on the previous commit:**

   | read | control | rotation | delta |
   |---|---:|---:|---|
   | target | 1.707964 | 1.707229 | `-7.4e-4` (floor `3e-4`) |
   | source | 3.043938 | 3.040511 | `-3.4e-3` (floor `2e-3`) |
   | `grad_moment_cosine` | -0.018203 | -0.017809 | +2% relative |
   | `transport_spin` | 0.000976 | 0.000973 | flat |
   | `transport_lag` / `curve` | 0.008046 / 0.6759 | 0.008040 / 0.6758 | flat |
   | `tangent_live_fraction` | 0.7874 | 0.7858 | flat |
   | peak reserved | 2.78 GiB | 2.94 GiB | **+5.8%** |

   **The frame panel is flat and that is the control**, not a disappointment:
   the correction changes the moment's coordinates and touches no frame, so
   `transport_spin` was never the readout. `grad_moment_cosine` is, and it moved
   the right way by 2% relative.

   **Both heads improve at roughly twice their noise floors, which is a hint and
   not a result.** One seed, and the same-config spread at 300 steps is `2.9e-4`
   on target. It also **confounds three changes in one arm** -- the rotation,
   stochastic rounding on the moment, and an fp32 moment reaching Newton-Schulz
   where it used to get bf16. Nothing here attributes the delta to any of them.

   *Expected, and worth stating.* At `beta = 0.9` the memory is ten steps and the
   accumulated spin is ~`1.8e-3` rad, which P7 already computed as immaterial.
   The mechanism is not supposed to pay here. **The test is the `beta` sweep** --
   `0.9 / 0.95 / 0.99` with and without the correction -- because the claim is
   that it raises the ceiling, not that it helps at today's setting.

   *Still open on the same axis:* **couple `beta` to frame motion**, shortening
   memory when rotation makes the moment stale, costing nothing new. Which
   signal drives it -- speed, curve, or spin -- is untested.

   *The run was global `r128`, not the per-role table.* Both arms match so the
   comparison holds, but `r` varies per role under the table and so does the
   rotation, so a table re-check is owed before this is called settled.

   *The bar.* Lag, spin and curve are the right instruments for asking what the
   frame is doing and **none of them ranks a design** -- the arm with the lowest
   spin and lag measured all session had a mediocre target. Either mechanism
   clears a loss bar or it does not ship.

3. **Stochastic rounding on the projected moment** -- shipped with lead 2, and
   not yet separated from it. A floor under `beta`, where lead 2 is the ceiling.

   The moment is bf16 in storage. **The result that killed SR on the basis does
   not transfer.** The frame has no sub-ulp update to lose; an EMA is the
   opposite object. Its increment is `(1 - beta) * g` against an
   accumulator of order `|g|`, and bf16's eight mantissa bits put the relative
   ulp at ~`2^-8`: at `beta = 0.9` the increment is ~25 ulp and safe, at `0.99`
   ~2.6 ulp on average, so every element contributing less than the mean rounds
   away. Longer memory is where round-to-nearest starts eating the signal.

   **And P7's bf16 verdict was conditioned on `beta = 0.9`.** Spin contributes
   `0.0018` of smear against `0.0194` from legitimate motion, raising the total
   from 1.936% to 1.948% -- immaterial, at that memory length. Lengthen the
   memory and the spin random-walks over more updates, so the verdict does not
   automatically survive the change it is being asked to permit. The dtype
   question and the `beta` question are one question.

   *The arm.* `copy_stochastic_` on the moment accumulate, then re-sweep `beta`
   at `0.9 / 0.95 / 0.99`. SR moving nothing at `0.9` closes it. SR *changing the
   shape of the beta curve* says the sweep that recorded `0.95` as dominated was
   partly measuring its increment rounding away.

4. **`eta` is unswept, and nothing bounds it any more.** The `0.02` cliff was
   dead planes, not step size: retested at `0.05` -- 2.5x past the old
   divergence point, at bs1, the batch where it used to break -- 300 steps ran
   clean on a fixed seed (`eta5e-2-bs1-r128-300-s1`).

   | read | `eta` 0.01 | `eta` 0.05 |
   |---|---:|---:|
   | target | 1.737169 | 1.739179 |
   | source | 2.913690 | 2.915922 |
   | `transport_speed` | 0.001555 | 0.005780 |
   | `transport_curve` | 0.6981 | 0.5926 |
   | `turn_fraction` | 0.2524 | 0.1828 |
   | participation / concentration | 0.01228 / 0.8287 | 0.01236 / 0.8285 |

   **The agreement clamp is a real governor.** 5x `eta` bought 3.7x speed,
   because `turn_fraction` *fell* -- a larger proposed turn puts more matrices on
   the ceiling, and the controller absorbed a quarter of the increase itself.
   The aim panel is flat to four decimals, which is the control: `eta` moves the
   response, not the aim.

   **`0.05` is worse, so `0.01` still stands.** Target `+2.0e-3` against a `3e-4`
   floor is a real regression; source `+2.2e-3` is marginal. But `eta` has
   stopped being a constant with a stability wall under it and become an ordinary
   tuning parameter that has never been swept downward or in the `0.01`-`0.03`
   interior. Cheap: bs1, 300 steps, a minute an arm.

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

3. **Calibration cannot probe past the cap, so the deep roles read as lower
   bounds.** The cap itself is settled (`ARCHIVE.md`): `min(m,n)/2` exists so the
   residual always carries energy for a tangent to be built from, `min(m,n)`
   failed on Anima with an empty residual, and half is the honest design point
   for a subspace optimizer.

   What is left is a measurement limit, not a design question. On LFM's MLP the
   tracked side is 1024, so `r_cal` cannot exceed 512 -- and `w1`/`w3` still read
   high `frac` there, meaning their calibrated ranks are lower bounds and the
   measured reallocation toward the MLP is understated. Nothing acts on this
   until a model shows a role starving at its capped rank; recorded so a high
   `frac` on a deep role is read as "clipped by the cap" rather than as a
   settled number.

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

*Stochastic rounding on the weight write* (`stochastic.py`, the bf16 parameter
update -- nothing to do with the basis update, which was tested separately and is
worse with it). Stochastic rounding delivers updates that round-to-nearest used
to discard, so the same nominal LR moves weights further -- more real progress
per step is also more forgetting per step. If that is what the source degradation
is, the fix is the LR and not a mechanism. Settled by running the sweep with
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
| `GEODESIC_STEPSIZE = 0.01` (`eta`) | geodesic step | calibratable, not derivable, and no longer cliff-bounded -- unswept, P2 lead 3 |
| `AGREEMENT_PLANES` `k = 16` | agreement meter | a second gain on `eta`'s quantity; accepted, not resolved |
| rank cap `min(m,n)/2` | `effective_rank` | P13 item 3; the fraction is still a choice |
| `grad_clip_norm = 1.0` | raw clip | mandatory; the threshold itself untested across models |
| `beta = 0.9`, `eps = 1e-8` | moment | the `beta` sweep carries a bf16 confound, P2 lead 3; `eps` inherited |
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
  SR trades a half-ulp bound for a full-ulp uniform draw. **Scoped to the basis.**
  The moment is an EMA with a shrinking increment and does have a sub-ulp update
  to lose, so this result says nothing about it -- P2 lead 3.
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
