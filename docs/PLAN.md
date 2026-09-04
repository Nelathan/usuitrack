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


## P13. Rank: Anima has a measured table, and a run to judge it

1. **ai-toolkit wiring -- done.** `toolkit/optimizers/usuitrack.py` now stamps a
   role label per weight, groups by `(lr, side, rank, role)`, takes a
   `rank_table` and a `calibrate_rank`, and drains the `RankCalibrator` through
   the `pop_diagnostics` the trainer already calls on its logging cadence -- one
   logging interval is one window, and the report reaches both the log and
   `loss_log.db` under `usuitrack/rankcal/*`. The side map stayed the hand map;
   the lab's `d_model` heuristic was **not** ported. Cross-attention `to_k`/`to_v`
   therefore tracked their 1024-wide text-context side through the calibration and
   run 9 -- held fixed on purpose, since there was no capacity to retest it there.

   It is being tested now, because the calibration made it interesting. Per 1000
   tracked dimensions `attn2.to_k` asked for 20.9 live planes and `attn2.to_v` for
   14.2 -- second and fourth densest of all roles, behind only self-attention K at
   25.6 and above every feed-forward and self-attention V role. A side that
   expensive per dimension is one the tracker is failing to compress.
   `anima_rankcal_r256_xattn_inner` repeated the first calibration with exactly one
   change, those two roles moved to the attention inner side. **The inner side
   won and is now the shipped map.** It asks for fewer planes -- `attn2.to_k`
   21.4 -> 20.4, `attn2.to_v` 14.5 -> 13.6 on the mean, 7-9% on the median -- and
   measures steadier, window `std` halving on both. On planes alone that margin is
   at the edge of what the comparison resolves, since the whole run shifted ~1-2%
   between calibrations and the single-matrix roles moved 3-4%. The density read
   is what makes it decisive: per 1000 tracked dimensions `attn2.to_k` falls
   20.9 -> 10.0 and `attn2.to_v` 14.2 -> 6.6, out of the top of the model and into
   the ordinary range below `ff_up`. Same structure, fewer planes, wider side --
   the tracker compressing it instead of straining. It is also free: basis plus
   projected moment is `3072r` elements either way.

   Note that for cross-attention K/V neither side is the residual stream: the
   input is the text context and the output is the attention inner space, 2048
   only by coincidence of width. The old override was not wrong about the
   semantics; the inner side is simply the better-conditioned thing to track.

2. **Anima with a calibrated table and a higher LR -- measured, running.**
   Calibration at `r_cal 256`, 200 steps, bs4/768, lr `2e-5` fit in 11.1 GB of
   12.3 and settled: windows drift under 10% and `std` is 1-6 planes.

   **The rank is the measured live count, unrounded** -- that is what puts
   `live_fraction` at ~0.95, since a basis sized to the planes that were live
   keeps almost all of them live. Where mean and median disagree, take the lower:
   under-provisioning is safe, a missing plane is a direction the tracker declines
   to move rather than an error. Run 9 shipped the median rounded *up* to
   multiples of 8 and paid for it -- it ran at `live_fraction` 0.79 with a fifth
   of every basis idle. The measured medians, and what run 9 actually used:

   | role | median | table | | role | median | table |
   |---|---|---|---|---|---|---|
   | attn1_k | 51.7 | 56 | | attn2_k | 24.7 | 32 |
   | attn1_q | 37.5 | 40 | | attn2_q | 17.9 | 24 |
   | attn1_out | 27.9 | 32 | | attn2_v | 16.5 | 24 |
   | attn1_v | 19.8 | 24 | | attn2_out | 9.4 | 16 |
   | ff_up | 21.2 | 24 | | patch_embed | 27.1 | 32 |
   | ff_down | 20.1 | 24 | | proj_out | 27.5 | 32 |
   | | | | | time_embed | 1.6 | 8 |

   **Role structure transfers to a DiT, and it is not the shape LFM had.**
   Attention carries the rank and is lopsided within itself: self-attention K
   wants 52 planes against V's 19, so the keys span a far richer subspace than the
   values they select. Cross-attention is uniformly leaner than self-attention.
   Both feed-forward sides sit near 20 despite being the largest weights in the
   model -- the opposite of LFM, where the MLP roles (`w1` 210, `w3` 195) were the
   rank hogs and attention was lean. `time_embed` reads 1.6 planes: the timestep
   path is nearly a single direction, and its 8 is a floor, not a measurement.

   The skew also inverts. On LFM the mean ran *above* the median for `w1`/`w3`/`w2`
   because a few layers carried real high rank; here the mean runs *below* the
   median on several roles, so the tail is on the low side and the median is the
   safe base rather than the conservative one.

   Optimizer state falls to 0.082 GB from 0.185 GB at the uniform rank 64 of run
   8. Two weights are capped by shape rather than measured (`patch_embed` at 34,
   `proj_out` at 32) and read saturated by construction.

   **Run 9 is a success on the read that decides this lane**: highly aesthetic,
   coherent samples, some bad ones. Its failure is displacement, not design --
   `update_to_param_ratio` ran at `1.12e-6` against run 8's `3.56e-6` and the LFM
   cosine-zero arm's `7.6e-5`, and the cosine decay took the last quarter to
   `5.3e-7`, so the model ended close to base with a minor style shift instead of
   a creative one. Part of that drop is the leaner table and is not a debt: less
   rank is less signal to act on, not a step to normalize back up. The lr rise
   this lane owes is to ortho-first -- the moment cancels *after*
   orthogonalization -- and is worth 2-4x.

   **Run 10** (`anima_usuitrack_10_inner_lr5e5`, wandb `7ffli01x`) is that run,
   clean over 2304 steps: lr `5e-5` annealed from 75% to `1e-5`, the unrounded
   table on the inner-side map, and the time module frozen -- its two linears
   asked for 1.7 planes of 256 and modulate every block through AdaLN, so what
   little they move is spent everywhere at once. Steps 1000-1700, against the two
   runs before it:

   | | run 8 | run 9 | run 10 |
   |---|---:|---:|---:|
   | rank | 64 uniform | table, rounded up | table, measured |
   | lr | `1e-5` | `2e-5` | `5e-5` |
   | `update_to_param_ratio` | 3.56e-6 | 1.12e-6 | **2.47e-6** |
   | ... in the last 200 steps | 5.41e-7 | 1.71e-7 | **7.06e-7** |
   | `tangent_live_fraction` | 0.472 | 0.787 | **0.854** |
   | `turn_fraction` | 0.367 | 0.491 | 0.510 |
   | `transport_speed` | 0.0020 | 0.0050 | 0.0051 |
   | loss (last 200 steps) | 0.1582 | 0.1499 | 0.1482 |

   The shallow anneal did its job: run 10 ends still moving at 7.1e-7 per step
   where run 9 had frozen at 1.7e-7. Loss remains what it always is on this lane,
   flat within noise across three quite different configurations. **The samples
   are the verdict and are unreviewed.**

4. **The sizing rule is off by the rank it was measured at.** Rank set to the
   live count measured at `r_cal 256` should put `live_fraction` at ~0.95, and
   run 10 landed at **0.854**. So the live count is not rank-invariant at the low
   end: at ranks of 20-51 roughly 15% fewer planes clear the Gram floor than the
   same roles showed at 256. To sit at 0.95 the rank has to go slightly *below*
   the measured count, and by how much is unmeasured. Cheap probe, 12 minutes: run
   a calibration at `r_cal` equal to the table itself and read what the same roles
   report there. Until then the rule is "the measured count lands ~0.85".

   Free on the same run: `grad_moment_cosine` is core-tier, so P14's overshoot
   meter got its first Anima read. At `r_cal 256`/lr `2e-5` it sits at
   `+0.0009..+0.0014` -- at the zero crossing, marginally under-travelling, which
   says `2e-5` is close to right on this lane. Rank changes the aggregate step, so
   it must be re-read at the table's rank before that is a claim.

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

**Goal.** Name all five and decide which are load-bearing. Lower priority than
it looks: see the dtype result below for what optimizer-side time is worth on
this lane.

**The optimizer step is not where the time goes. CLOSED, by measurement.**
The polar map's dtype looked like the largest lever and is worth nothing end to
end. Both references run the Newton-Schulz iteration in low precision -- the
Polar Express reference casts to bf16 "for speed", HeavyBall stochastically
rounds into bf16 -- and UsuiTrack runs it in fp32. Measured on a 4070 SUPER,
batched:

| shape | fp32 | bf16 | speedup | orthogonality residual |
|---|---:|---:|---:|---|
| (32, 2048, 128) | 1.836 ms | 0.750 ms | 2.45x | 1.7e-2 -> 3.1e-2 |
| (32, 2048, 512) | 22.043 ms | 6.251 ms | 3.53x | 9.0e-3 -> 2.4e-2 |
| (8, 4096, 256) | 3.066 ms | 1.034 ms | 2.97x | 1.3e-2 -> 3.0e-2 |

Through the real `_balanced_polar_direction` the win is smaller, 1.5-2.0x,
because Aurora's row work stays fp32 and the rounding cast is real work.

**End to end it is zero.** A 1k-step arm at `beta 0.9`, `lr 4e-4` on the table
took **854 s** in bf16 against **847 s** in fp32 -- slower, inside wall-clock
noise -- while landing `1.665305 / 3.043632` against `1.665956 / 3.039360`, both
deltas at twice their floors in opposite directions. So the accuracy cost is
nothing and the speed benefit is nothing: at 350M and bs16 x 1024, forward and
backward dominate so thoroughly that halving the optimizer's largest term
vanishes. **The kernel was the wrong unit to price.** Anima does not rescue it
either -- its basis is the same size class (2k x 64 against 1k x 128).

This retires the iteration count with it (5 -> 3 saves 37%, 5 -> 1 saves 74% of
something invisible), and it lowers the priority of the sync hunt: five syncs
per step are five syncs in a term that does not show up in the wall clock. Note
for whoever reopens this: Polar Express coefficients are fitted for a planned
iteration count, so a 3-step run needs its own fitted triple, not the first three
rows of the 5-step schedule.

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
