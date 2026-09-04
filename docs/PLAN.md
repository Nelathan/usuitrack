# Open design questions

Working notes, not a specification. `SPEC.md` describes what the optimizer does
today; this file describes what we are still unsure about and what would settle
each thing. See `AGENTS.md` for how this file is worked.

Everything closed lives in `ARCHIVE.md` -- distilled conclusions in its top half,
the frozen former PLAN as an investigation log below them. Where a line here says
*(archive)* the evidence is there.

---

## P14. The learning rate has a rule now, and it has not been used

**`grad_moment_cosine` is a calibrated LR controller.** Positive means the step
under-travels and the next gradient still points where the last one did;
negative means it overshoots the valley and the gradient has flipped behind it;
zero is critically damped. Interpolating two reads (`+0.0024` at `2e-4`,
`-0.0026` at `8.7e-4`) put the crossing at `4.05e-4`; the arm run there read
`-8.3e-5` and beat the old default on **both** heads at half the aggregate step:

| arm | target | source | ratio |
|---|---:|---:|---:|
| old default `2e-4` | 1.667731 | 3.080167 | 0.000166 |
| cosine-zero `4e-4` | **1.665708** | **3.040611** | 0.000076 |

`-2.0e-3` target and `-4.0e-2` source, 7x and 20x their floors. The first
Pareto move on this lane that was not bought with distance.

**Open.** The rule has been confirmed at one point on one model. It should be
checked where it matters: does the cosine-zero LR track the noise floor across
batch size, and does it hold on Anima at bs4, where a sweep is expensive and a
rule that needs no sweep is worth the most? Until then it is a strong
coincidence with a mechanism, not a law.

**Two things any LR comparison must still control for.**

*The rank/step coupling.* The step's norm carries `sqrt(r)`, so
`update_to_param_ratio` tracks `sqrt(r/128)` to four figures and **a table
change is also an LR change** -- the calibrated table runs 1.2% leaner than its
predecessor. Compare at matched `update_to_param_ratio`, never at matched
nominal LR. This is also why every rank comparison in the archive conflates
subspace and step. Under the ortho-first order there is a second term: the step
also carries the agreement `||M||/||O||`, `0.23` at `beta = 0.9` and stable
across a 4.4x LR change, so a `beta` change is an LR change too.

*Stochastic rounding on the weight write* (`stochastic.py`; nothing to do with
the basis update, which is worse with it). It delivers updates that
round-to-nearest discarded, so the same nominal LR moves weights further -- more
real progress per step is also more forgetting per step. If that is what source
degradation is, the fix is the LR and not a mechanism. Settled by running with
it on and off at matched rank table, batch and step count: same shape with a
horizontal offset confirms it.

**And there is no historical baseline to lean on.** The newest wandb directory in
the lab predates the lab's own final commit, and not one of 197 stored run
directories logs `basis_update_interval`. Every archived number also carries a
fallback that was barely training -- bf16 storage through
`torch.optim._functional.adamw`, exactly the round-to-nearest loss
`stochastic.py` documents. Fixed now, but the archive is not a clean control for
anything touching update magnitude.

**Also open: `beta` at the calm step.** `0.9` remains the optimum at both step
sizes measured (`0.5` and `0.99` both worse, matched step), but `0.99`'s penalty
shrank sixfold when the step halved. Whether the optimum actually moves at a
calmer step than we have run is untested, and it is the same axis as the LR
question.

---

## P15. A per-matrix learning rate from the overshoot meter

**The idea.** Smooth `grad_moment_cosine` per matrix; where it is negative,
damp that matrix's LR; where it is not, decay the damping back toward the
default. Asymmetric on purpose -- the configured LR stays a ceiling, so the
worst case is a slower run rather than a diverged one. It can run entirely
on-device: the damping factor is a 0-dim tensor multiplying the update, so it
adds no sync. State is two scalars per matrix.

**Why it is not obvious.** The global version of this knob produced P14's win,
which is the argument for it. Against it: per-matrix step allocation has already
been measured loss-invisible once (`ARCHIVE.md`, the closed P2 -- damping by
agreement changed nothing at 20x spread in contested-step size, and what
difference there was favoured *not* damping). Cosine is a different signal --
feedback on overshoot rather than feedforward on confidence -- so it is not
refuted, but the family's prior is poor.

**The measurement that decides whether to build it, and it is nearly free.**
Accumulate `cosine^2` alongside `cosine` and report the spread across matrices,
split by role. If every matrix sits at the same cosine, the controller is a
global LR schedule in costume and P14's rule already delivers it. Real spread
structured by role is the case worth building; real spread that is unstructured
step-to-step makes the EMA time constant the whole design problem.

**Two traps if it gets built.** A damping-only controller lowers the aggregate
step, so it must be compared against a global LR tuned to the same
`update_to_param_ratio` or it will re-measure the LR axis. And damping cuts
overshoot, which raises cosine, which decays the damping, which restores
overshoot -- a limit cycle unless the decay is slower than the response. Rprop's
bounded multiplicative factors are the place to steal from, with the caveat that
that lineage was built for low-noise regimes and this is the opposite.

---

## P16. `eta` is unswept, and nothing bounds it any more

The `0.02` cliff was dead planes, not step size: retested at `0.05` -- 2.5x past
the old divergence point, at bs1, the batch where it used to break -- 300 steps
ran clean on a fixed seed (`eta5e-2-bs1-r128-300-s1`).

| read | `eta` 0.01 | `eta` 0.05 |
|---|---:|---:|
| target | 1.737169 | 1.739179 |
| source | 2.913690 | 2.915922 |
| `transport_speed` | 0.001555 | 0.005780 |
| `transport_curve` | 0.6981 | 0.5926 |
| `turn_fraction` | 0.2524 | 0.1828 |

**The agreement clamp is a real governor.** 5x `eta` bought 3.7x speed, because
`turn_fraction` *fell* -- a larger proposed turn puts more matrices on the
ceiling and the controller absorbed a quarter of the increase itself. The aim
panel is flat to four decimals, which is the control: `eta` moves the response,
not the aim.

**`0.05` is worse, so `0.01` stands**, but `eta` has stopped being a constant
with a stability wall under it and become an ordinary tuning parameter never
swept downward or in the `0.01`-`0.03` interior. Cheap: bs1, 300 steps, a minute
an arm. **Do not make the aim hot to simulate higher rank** -- better loss from a
hotter `eta` would not mean a better basis, and the basis is the track, not the
goal.

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

## P5. Magic number census

| constant | where | status |
|---|---|---|
| `GEODESIC_STEPSIZE = 0.01` (`eta`) | geodesic step | calibratable, not derivable, and no longer cliff-bounded -- unswept, P16 |
| `AGREEMENT_PLANES` `k = 16` | agreement meter | a second gain on `eta`'s quantity; accepted, not resolved |
| rank cap `min(m,n)/2` | `effective_rank` | P13 item 3; the fraction is still a choice |
| `beta = 0.9`, `eps = 1e-8` | moment | measured optimum at two step sizes, but it moves with noise and with LR (P14); `eps` inherited |
| `AURORA_PP_ITERATIONS = 1`, `AURORA_PP_BETA = 0.5` | direction map | inherited from the method; measured at 0-3% of the polar map's cost, so nothing to save by dropping it |
| `NEWTON_SCHULZ_COEFFICIENTS`, 5 steps | polar map | the largest single cost in the optimizer; P12 |
| `1e-12` floors | numerical | never read against the precision they run in |

**Goal.** Fewer constants, and the survivors derived or at least scale-free. The
controller half is done (`ARCHIVE.md`); the rows above are what did not yield.
`grad_clip_norm` left the census by deletion rather than by calibration -- see
`SPEC.md` step 1.

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

**The largest lever is the polar map's dtype, and it is a faithfulness gap.**
Both references run the Newton-Schulz iteration in low precision -- the Polar
Express reference casts to bf16 "for speed", HeavyBall stochastically rounds
into bf16. UsuiTrack runs it in fp32. Measured on a 4070 SUPER, batched:

| shape | fp32 | bf16 | speedup | orthogonality residual |
|---|---:|---:|---:|---|
| (32, 2048, 128) | 1.836 ms | 0.750 ms | 2.45x | 1.7e-2 -> 3.1e-2 |
| (32, 2048, 512) | 22.043 ms | 6.251 ms | 3.53x | 9.0e-3 -> 2.4e-2 |
| (8, 4096, 256) | 3.066 ms | 1.034 ms | 2.97x | 1.3e-2 -> 3.0e-2 |

Note what fp32 buys: at five iterations the map is only ~1% orthogonal anyway,
so this is 1% versus 3% on a quantity the design already treats as approximate,
for 2.5-3.5x. The right shape is HeavyBall's -- stochastic rounding into bf16,
not a plain cast. Needs a loss arm, not an assertion.

**Second lever: the iteration count.** 5 -> 3 steps saves 37%, 5 -> 1 saves 74%.
Polar Express coefficients are fitted for a planned iteration count, so a 3-step
run needs its own fitted triple rather than the first three rows of the 5-step
schedule.

**Provenance, checked against both references.** Our schedule is HeavyBall's
`zeropower_via_newtonschulz5` -- the Muon-lineage speedrun coefficients credited
to `@scottjmaddox` / `@YouJiacheng` -- not the Polar Express series, whose first
triple is `(8.2373, -23.1577, 16.6806)`. The iteration body is algebraically
identical to both references; normalization matches HeavyBall (the Polar Express
reference additionally multiplies the norm by 1.01). `SPEC.md` step 7 states
this correctly; `README.md` claimed Polar Express and has been fixed. The
projector's single-step Stiefel retraction *does* use the genuine Polar Express
fixed point `(1.875, -1.25, 0.375)`.

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

Current code, per-role table, same lane. `beta` arms are at matched
`update_to_param_ratio`, which under ortho-first means each carries its own LR.

| arm | target | source | ratio |
|---|---:|---:|---:|
| `beta 0.9`, `lr 4e-4` (cosine-zero) | **1.665708** | **3.040611** | 0.000076 |
| `beta 0.9`, `lr 8.7e-4` (old default step) | 1.669194 | 3.086723 | 0.000165 |
| `beta 0.5`, `lr 1.6e-4` | 1.690248 | 3.013536 | 0.000077 |
| `beta 0.99`, `lr 1.29e-3` | 1.681707 | 3.138621 | 0.000075 |
| `beta 0`, `lr 2e-4` | 1.685989 | 3.020878 | 0.000166 |

`moment_persistence` reads within `1e-3` of zero in every one of them.

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
