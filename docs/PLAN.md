# Open design questions

Working notes, not a specification. `SPEC.md` describes what the optimizer does
today; this file describes what we are still unsure about and what would settle
each thing. See `AGENTS.md` for how this file is worked.

Everything closed lives in `ARCHIVE.md` -- distilled conclusions in its top half,
the frozen former PLAN as an investigation log below them. Where a line here says
*(archive)* the evidence is there.

---

## P20. Gradient accumulation is built, and it prices its own learning rate

`accumulate(weight)` folds one micro-batch into the pending step; `k` calls then
one `step()` is gradient accumulation. It lives in projected space, so
`release_matrix_grads` is untouched and the effective batch rises at no VRAM
cost beyond two accumulator buffers per matrix -- which is what makes it the only
way Anima gets past bs4 on a 12 GB card.

The frame is held across the group and every micro-batch is orthogonalized on its
own before the average. Both are forced rather than chosen: a held frame is what
makes the micro-batch tangents vectors in one tangent space, so their arithmetic
mean is the geometrically correct one and no spherical mean is called for;
averaging raw projected gradients and orthogonalizing once would hand the step to
whichever micro-batch was loudest, which is the failure ortho-first exists to
prevent. `SPEC.md` step 5 carries the result.

**Measured, and it changes the learning-rate arithmetic.** `||O^{(i)}||` is fixed
by shape, so the mean of `k` directions is shorter than each of them in
proportion to their disagreement. On a toy at `lr 1e-3`, rank 32, 40 steps:

| k | `update_to_param_ratio` | `micro_batch_agreement` | measured shrink | `sqrt(a + (1-a)/k)` |
|---:|---:|---:|---:|---:|
| 1 | 4.650e-5 | -- | 1.000 | 1.000 |
| 2 | 3.310e-5 | 0.0011 | 0.712 | 0.707 |
| 4 | 2.365e-5 | 0.0017 | 0.509 | 0.501 |
| 8 | 1.708e-5 | 0.0011 | 0.367 | 0.355 |

Within 3% across an 8x span. So **under ortho-first the `sqrt`-of-batch-size
learning-rate rule is already in the mechanism**, and a run that raises `k` to 4
without raising nominal `lr` by 2x is running at half its predecessor's step, not
at the same one. This is the reverse of the usual trap: here the anchoring is
automatic and it is *holding* `lr` fixed that changes the step.

Two things follow for the next Anima run, and they are not free.

1. **Travel.** 576 steps at `k=4` consumes run 12's three epochs exactly, with a
   quarter of the weight updates. `lr 1e-4` holds the per-step size where run 12
   had it and accepts a quarter of the travel; matching run 12's *travel* in 576
   steps would need `8e-4`, which is not a serious proposal. Either the run is
   read as a mechanism arm at a quarter the displacement, or it buys steps.
2. **`a` on this lane is unmeasured**, and it is the term that sets the shrink.
   The toy reads 0.001; real gradients carry step-to-step `tangent_concentration`
   of 0.68-0.82, so Anima's `a` could be far higher and the shrink far milder. **A
   calibration run measures it at no cost and it is the number the training run's
   `lr` should be chosen from**, which is the argument for calibrating first.

**Answered on both counts, by `anima_rankcal_r256_acc4`** -- rank 256, `k=4`,
bs4/768, `lr 4e-5` holding the k=1 calibration's step, against
`anima_rankcal_r256_xattn_inner` at the same rank.

**`a` is 0.003 on Anima, so the shrink is the full `1/sqrt(k)`.** Windows read
0.0027, 0.0040, 0.0024 -- the toy's low-agreement regime, not the milder one
`tangent_concentration` suggested. Micro-batches of 4 samples agree essentially
not at all here, so `lr 1e-4` at `k=4` holds run 12's step exactly and there is
no correction to make.

**Liveness rises, sharply, and the table was under-sized.** At equal rank every
role reads 2.1-3.6x its k=1 live-plane count and the fleet fraction goes 0.094 ->
0.235. The flattened-head mechanism is the one that fits: a single batch's noise
is spiky and sits near the top of the spectrum, averaging flattens it, the
relative floor falls and planes that were clipped now turn. `live ~ r^0.6` is not
a noise artifact to be relaxed away -- demand rose rather than fell.

| role | k=1 | k=4 | ratio |
|---|---:|---:|---:|
| attn1_k | 53.2 | 109.8 | 2.07 |
| attn1_q | 36.9 | 78.6 | 2.13 |
| attn1_out | 25.8 | 78.3 | 3.03 |
| attn2_k | 19.4 | 56.0 | 2.89 |
| ff_down | 18.7 | 55.3 | 2.96 |
| attn2_q | 17.6 | 45.9 | 2.62 |
| attn2_v | 13.3 | 44.8 | 3.38 |
| ff_up | 19.3 | 43.7 | 2.27 |
| attn2_out | 11.3 | 40.3 | 3.57 |
| attn1_v | 18.0 | 39.4 | 2.19 |
| time_embed | 1.7 | 1.8 | 1.11 |

`patch_embed` and `proj_out` are censored by their shape caps of 34 and 32 planes
and read saturated, so their ratios are floors. `time_embed` does not move; its
aim has been rank-one on every read.

**A rank table cannot be derived by the converging rule from a calibration
rank.** `r * (target/frac)^-2.5`, fitted over ranks 16-34, puts most roles at 1-4
planes when fed `frac` measured at 256, and applied to the k=1 data it predicts
`ff_up` at 0.21 live fraction where run 11 measured 0.94. The exponent does not
carry across that span. Run 13's table is therefore sized by the rule with an
anchor -- rank equals the live count at the calibration rank, geometric mean over
windows -- which is what run 12 used and what delivered a fleet 0.893. Sum 595
against run 12's 256. **The `live_fraction` target (P13 item 4) is still open and
this table does not depend on it**; the per-role fractions run 13 reports at these
ranks are the read that settles it.

**Two costs the accumulation path carries, both paid.** The accumulators are the
only `[d, r]` buffers resident *across* a group, so they are live alongside each
backward's activations; at rank 256 they took `k=4` over the 12 GB card and the
job aborted on three consecutive OOM retries. They now follow the gradient's
storage dtype, as the moment does, which is bf16 on this lane and halves that
resident cost, with the write stochastically rounded because increment `i` of `k`
is about `1/k` of the total and round-to-nearest drops it outright. Separately,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is what let rank 256 run at
all, at 15.3 s/step against 7.7 without it.

**Run 13 landed, and displacement beat every prediction made from the shrink
identity alone.** `anima_usuitrack_13_acc4`: `k=4`, 576 steps, `lr 1e-4`, the
measured table (sum 595, geometric-mean live counts at rank 256/k=4), `core`
diagnostics. Whole-model distance from base, same script as run 12:

| | steps | displacement |
|---|---:|---:|
| run 12 | 2304 | 2.327e-3 |
| run 13 | 576 | 2.960e-3 |

**More travel in a quarter of the steps, not a quarter of the travel.** The
per-step shrink identity (measured `a=0.0018-0.0021` in situ, matching the
calibration) predicted the *step size* correctly -- plateau
`update_to_param_ratio` 3.70e-6 against run 12's 2.33e-6, the ratio the table's
`sqrt(r)` (1.59x) predicts -- but multiplying that by a quarter of the steps
underestimates total displacement, because per-step travel does not sum
linearly against a moving basis the way the arithmetic implied. Not yet
explained; flag for whoever next reasons about a step-size proxy for total
travel under accumulation.

**The run's own schedule throttled its last quarter.**
`warmup_stable_cosine_decay` with `decay_start_ratio: 0.75` starts decaying at
step 432; growth rate fell 5.4x by the end (3.31e-6/step at 350-450 against
6.15e-7/step at 550-576). This is the direct answer to "would more steps or a
higher lr buy more": yes, on both counts -- the run spent its last 144 steps at
a shrinking step size by schedule, not by mechanism saturation. A rerun that
holds the plateau lr longer, or extends past 576, has headroom that this run's
own numbers already show being spent down.

**`grad_capture` climbed through the run, not held flat.** 0.650 at step 50 (the
number the mid-run "insensitive to rank" read was drawn from), rising to a
0.73-0.75 plateau by step 300 -- above run 12's 0.647 and every prior run.
Reading it early, before the basis has adapted to the larger table, understated
it; the mid-run note in HANDOVER is superseded.

**`grad_moment_cosine` averages 0.0056-0.0061, above run 12's 0.00094 and near
runs 9-11's ~0.0035, with periodic negative excursions** (11 of 115 windows,
min -0.0046) -- the same crossing-zero behavior run 12 showed. Three
confounded changes (rank, lr, `k`) moved together here, same as run 12's freeze
was confounded with two others; still unisolated, see the standing item below.

**`tangent_concentration` reads 0.582**, below the 0.68-0.82 range single-batch
gradients showed and closer to the synthetic-gradient floor's neighborhood
without approaching it. Candidate mechanism: the same head-flattening that
raised `tangent_live_fraction` also lowers concentration, since both read the
same averaged Gram spectrum from opposite ends -- a flatter head is simultaneously
more planes clearing the floor and less concentration in the top ones. Not
measured in isolation; do not cite past this session without the calibration
that pins it down.

Not yet tried, recorded so it is not mistaken for settled: **stratified timestep
sampling across the group**, one micro-batch per quarter of the `t` range instead
of `k` independent uniform draws. Free, and it cuts the group's `t` variance below
what independent draws give. Deliberately not bundled into the first accumulation
run.

---

## P19. There is no sensor for arrival, and the turn is constant until there is

The agreement gain is removed (`ARCHIVE.md`). Every live plane turns by `eta`,
and nothing scales it. What the gain was for is still real: **a converged frame
orbits on batch noise, and a controller that keeps turning at full `eta` through
that is integrating noise into the basis.** Nothing measures it.

**Why the two obvious sensors are already spent.** The top-`k` agreement gain
needed a level reference, and the only anchor-free form -- the aim's two-lag
autocorrelation -- is falsified: persistence *shape* is stationary while its level
falls ~4x, so the ratio holds near `0.81` and anneals nothing. The ceiling that
replaced the anchor is a magnitude functional gating a magnitude-free step. Do
not re-derive either; both are written up in `ARCHIVE.md`.

**The hard part was batch size, and accumulation is how we buy our way out.** At bs4 an aim
spreading toward isotropy is what batch noise alone produces, so "arrived" and
"drowned in noise" have the same signature in every spread statistic we publish
(`agreement_ceiling` rose 4.6% over run 11 while `tangent_concentration` fell --
that pair says nothing about arrival). The gain looked like it worked on LFM at
bs16 and was close to inert on Anima at bs4, which is consistent with the meter
needing contrast the batch does not supply. **Anima's batch cannot rise above 4**
-- 11 of 12 GB, and `release_matrix_grads` is what makes it fit. What the card
cannot give, `accumulate()` can: the effective batch rises in projected space, at
the price of a weight update per `k` backwards (P20).

**Gradient accumulation supplies the sensor, and it is built.** The missing
quantity was a level reference, and `micro_batch_agreement` (P20, `SPEC.md`) is
one that needs no reference run: the mean pairwise cosine between the
orthogonalized directions of `k` independently drawn micro-batches estimates the
aim's signal-to-noise ratio `|s|^2/(|s|^2 + |n|^2)`, and a pure-noise aim reads
exactly zero because the null is analytic rather than measured. It compares
independent samples instead of inferring a level from one batch's spectrum, which
is what every spent sensor here could not do. Costs one norm of a tensor already
in hand, on device, no sync.

It is published and **wired to nothing, on purpose**. A controller driven off it
would be the agreement gain again, and the objection to that gain was never the
sensor's quality -- it was that a magnitude functional has no business gating a
step whose entire point is that magnitude decides nothing. What is now open is
the empirical question the gain never reached: does `a` fall over a run on a real
model, and does it fall in a way that separates arrival from noise. Unmeasured on
either lane; it needs one accumulated run to say anything.

**Meanwhile `eta` is the only handle**, and scheduling it on the prior that
aiming gets harder as the loss surface smooths is a knob, not a sensor -- worth
having as an arm, worth naming as a prior. See P16: `eta` has never been swept
downward or in the `0.01`-`0.03` interior, and 5x `eta` measured worse loss with
a better projected grad norm.

---

## P18. `fallback_lr` has never been swept, and the reason to raise it was wrong

The AdamW fallback covers biases, norm gains, the AdaLN modulation linears and
the positional embedding tables. It has sat at `5e-6` on Anima through runs 9,
10 and 11 while the matrix lr went `2e-5` -> `5e-5` -> `1e-4`, so the ratio drifted
from 4x to 20x without anyone choosing it.

**The obvious inference from that drift is false.** "This class is being left
behind" predicts it moves least; measured against the base weights it moves
*most*, and it is the only part of the model that does not care what the
optimizer is doing. Relative Frobenius distance from base at step 2200, run 9
against run 10 -- 2.5x apart in matrix lr, and the fallback identical at `5e-6`:

| role | run 9 | run 10 | delta |
|---|---:|---:|---:|
| AdaLN + norms (fallback) | 1.549e-2 | 1.559e-2 | **+0.6%** |
| other fallback (embeddings) | 1.068e-2 | 1.075e-2 | +0.7% |
| whole model | 8.571e-3 | 8.800e-3 | +2.7% |
| `patch_embed` | 8.727e-3 | 1.315e-2 | +51% |
| `attn1_k` | 3.101e-3 | 4.847e-3 | +56% |
| `ff_down` | 1.655e-3 | 2.455e-3 | +48% |
| `time_embed` | 2.132e-3 | 0 | frozen |

Three things fall out at once.

1. **The fallback class is the largest mover in the model** -- twice the fleet
   and four times the leanest matrix role -- at a fifth of the matrix lr. AdamW's
   step is normalized, so it travels at about its lr per step regardless of how
   small its gradients are, while the tracker's step is not. A lr ratio is not a
   motion ratio, and on this pair it is not even the right sign.
2. **It dominates the whole-model number.** Total displacement moved 2.7%
   between two runs whose matrix roles all moved ~50%, because the part that did
   not change is the part that carries most of the distance. Every whole-model
   displacement figure on this lane is mostly a read of AdamW at `5e-6`.
3. **Matrix displacement is sublinear in lr.** 2.5x the lr bought ~1.5x the
   travel -- some of that is the leaner table cutting the step, the rest is that
   a hotter step in a noisier direction cancels more. Run 11 doubles the lr
   again; if the exponent holds it should land ~1.3x run 10's matrix roles, and
   if it lands at 2x something other than the step size changed.

The uncomfortable reading: if sample character is carried by the AdaLN
modulation, three runs of tuning the tracker have been tuning the minority of
the model's motion.

**Still open, because far is not the same as right.** Displacement says the
class is not frozen; it says nothing about whether it is over- or under-trained
relative to the matrices. Three arms, and they are not the same experiment:

- **`5e-6` (current).** The status quo, and the only one with samples behind it.
- **`2e-5`.** More AdamW motion on the class that already dominates. Careful, not
  bold: this is the class that went non-finite on this model before, so
  `nonfinite_grads` is the first read to check.
- **Frozen.** The one that answers a different and arguably better question:
  what does UsuiTrack alone do to this model? Today every Anima result is a mix
  of a tracked matrix update and an AdamW step that carries most of the
  displacement, so no sample has ever been attributable to the optimizer under
  test. Freezing gives the first clean read of it. The cost is real and should
  be stated before running it -- AdaLN modulation is where a DiT adapts
  conditioning strength, and freezing it may simply mean the model cannot move
  far enough to show anything. That failure is itself informative, and cheap.

The frozen arm is the interesting one and it is not obviously the safe one. It
goes on the list as a deliberate arm, not as a default.

**Run 12 was that arm, and the freeze is verified.** `adaln`, the embeddings and
`time` all read exactly `0.000e+00` displacement from base at every checkpoint,
so `fallback_lr: 0.0` took. Two results, and the second is the larger one.

**1. UsuiTrack alone moves this model a quarter as far.** Whole-model
displacement at step 2200: `2.314e-3` against run 10's `8.800e-3`, `0.263x`. So
roughly three quarters of every previous Anima result's weight motion was AdamW
walking the fallback class at `5e-6`. Three runs of tuning the tracker were
tuning the minority of the model's motion, and now we know the fraction.

**2. Nothing we changed moved the matrix roles at all.** Per role at step 2200,
run 12 over run 10, against `sqrt(r_12/r_10)` -- the step-norm identity, the only
term the rank table should contribute:

| role | run 10 | run 12 | ratio | `sqrt(r)` | residual |
|---|---:|---:|---:|---:|---:|
| patch_embed | 1.315e-2 | 1.254e-2 | 0.954 | 0.961 | 0.993 |
| proj_out | 1.174e-2 | 1.147e-2 | 0.977 | 0.981 | 0.996 |
| attn2_k | 4.930e-3 | 4.533e-3 | 0.919 | 0.894 | 1.028 |
| attn1_k | 4.847e-3 | 4.711e-3 | 0.972 | 0.950 | 1.023 |
| attn1_q | 4.393e-3 | 4.276e-3 | 0.973 | 0.959 | 1.015 |
| attn2_v | 4.151e-3 | 3.793e-3 | 0.914 | 0.886 | 1.031 |
| attn1_out | 3.923e-3 | 3.761e-3 | 0.959 | 0.938 | 1.022 |
| attn2_q | 3.658e-3 | 3.574e-3 | 0.977 | 0.972 | 1.005 |
| ff_up | 3.555e-3 | 3.552e-3 | **0.999** | **1.000** | 0.999 |
| attn1_v | 3.377e-3 | 3.295e-3 | 0.976 | 0.972 | 1.004 |
| attn2_out | 2.915e-3 | 2.729e-3 | 0.936 | 0.943 | 0.993 |
| ff_down | 2.455e-3 | 2.357e-3 | 0.960 | 0.943 | 1.018 |

Every residual is within 3% of one, and `ff_up` -- rank unchanged -- lands at
`0.999`. **Over 2300 steps, matrix displacement is `lr` and `sqrt(r)` and
nothing else.** Freezing the class that carried three quarters of the model's
motion did not change it. Doubling the frame's turn rate (the constant turn,
`transport_speed` `0.00511 -> 0.00931`) did not change it. Neither did tracking
a differently-conditioned set of planes. Displacement is a budget, not a quality
read, and it cannot rank a tracker design -- a fact worth as much as the arm it
came from.

**Run 12's samples are read: progressed, slightly harmed, and the mid-run
degradation during the anneal is gone.** The freeze did what the instability
hypothesis predicted. What it also removed is the creative and aesthetic
advancement earlier runs showed, which is the cost this arm was warned would be
real. So the class is not optional, and the question becomes what it should
travel.

**`fallback_lr` is a travel budget, and that makes it derivable.** AdamW's step
is per-coordinate normalized, so it moves about `lr` per coordinate per step
wherever the gradient's sign persists, with no dependence on gradient magnitude:
predicted relative displacement is `lr * steps * mean_schedule / w_rms`. For run
10 that is `5e-6 * 2304 * 0.879 = 1.0e-2` against a measured `1.559e-2` on a class
whose weights sit near unit rms -- the same quantity within 1.5x, so the motion is
ballistic and the model holds.

| `fallback_lr` | predicted displacement, 2304 steps | vs run 12's matrix median `3.8e-3` |
|---|---:|---:|
| `5e-6` (runs 9-11) | 1.0e-2 | 2.7x |
| **`2e-6` (next)** | 4.1e-3 | 1.1x |
| `1e-6` | 2.0e-3 | 0.53x |
| `0` (run 12) | 0 | 0 |

`2e-6` matches the class's travel to the matrix fleet, which is the one ratio
defensible without another sample review, and it is still a 2.5x cut from the
regime that was unstable. `1e-6` would put the largest natural mover in the model
below every matrix role.

**Ruled out, so it is not re-derived: the anneal degradation was not a schedule
artifact.** The suspicion was that the fallback's lr floor did not track the
matrices', so its share grew through the decay. It does track:
`warmup_stable_cosine_decay` is a `LambdaLR` with a multiplicative
`min_lr_ratio`, and the split optimizer exposes both sub-optimizers through one
`param_groups`, so run 10 held the fallback at exactly 10% of the matrix lr from
step 100 to the end. Whatever the anneal degradation was, it was not the ratio
drifting.

**Also live and previously unnoticed:** `accelerator.clip_grad_norm_` runs at
`max_grad_norm` 1.0 before every step, and under `release_matrix_grads` the matrix
gradients are already freed, so `None` grads are skipped and the clip applies to
the fallback class alone. It did nothing in run 12 at `fallback_lr: 0.0`; at
`2e-6` it is live again, and it is part of why that class has not gone non-finite
since.

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

**It was flat on Anima through three runs, and run 12 moved it.** Full-run means:
`+0.00357` (run 9), `+0.00347` (run 10), `+0.00336` (run 11) -- unmoved across a
5x span in peak lr -- then **`+0.00094`** in run 12, with the 400-step buckets
crossing zero (`-1e-5` at 1200-1600) and coming back. That is a 3.5x move on a
meter that had refused to respond to the one knob it is supposed to calibrate,
and run 12 did not change the lr at all.

So the meter is not dead here; it was reading something the lr could not reach.
Three candidate causes, confounded in one run: the frozen fallback (a class
walking at `5e-6` on normalized steps is a systematically moving target the
matrices keep agreeing with), the constant turn (the frame now follows at 1.8x
the speed, so it under-travels less), and the leaner table. **The freeze is the
one with a mechanism that predicts this sign**, and it is the user's standing
hypothesis -- the fallback injects instability the tracker then chases. Isolating
it costs one run: run 10's config with the freeze and nothing else, or run 12's
with the fallback restored.

If the crossing is real, P14's rule now says `5e-5` with a frozen fallback is
approximately critically damped on this lane -- the first time the rule has had
anything to say about Anima.

One caution survives from when it was flat: the window means quoted in earlier
notes (`+0.0038` for run 9, `+0.008..0.010` for run 10) were window artifacts on
a noisy series whose step-to-step range is `-0.006..+0.017`. Quote the full-run
mean or name the window.

**And on this lane it is measuring the wrong failure.** Run 12 read `+0.00094`,
which by this rule says `5e-5` is near critically damped, while the samples read
as slightly hot. Both hold, because the cosine measures overshoot of the valley it
is descending and has nothing to say about forgetting. Noise reaches it only
through the step -- the wrong motion is undone by the next gradient, a negative
contribution proportional to noise energy times `lr` times curvature -- so on
noise-dominated batches the crossing sits *below* the true descent optimum and the
meter is already the conservative one. The part that harms a finetune is invisible
to it: motion driven by batch noise is uncorrelated with the gradient, so it moves
the cosine neither way while erasing whatever the base model held in those
directions. **The zero crossing is the descent optimum, not the finetune
optimum**, and an lr cut justified by retention has to be argued as one. This is
also the sharpest argument for accumulation (P20): noise-driven motion is exactly
what averaging micro-batches removes.

There is no forgetting instrument on the Anima lane at all. Flow-matching loss
does not rank checkpoints, displacement is a budget, and the samples are the only
read. That gap is worth naming as a gap.

**Open.** The rule is confirmed at one point on LFM and has now, on Anima, moved
for the first time -- but under three simultaneous changes, so what it responds
to here is unknown. Whether it can pick an lr on this lane is the question the
isolating run answers -- and the answer may be that it can pick a descent lr and
never a finetune one.

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

**`0.05` is worse on loss, so `0.01` stands** -- though it read a *better*
`projected_grad_norm`, which is the pair to remember: a hotter frame catches more
gradient and loses. `eta` has stopped being a constant with a stability wall
under it and become an ordinary tuning parameter never swept downward or in the
`0.01`-`0.03` interior. **With the agreement gain gone (P19) it is the only
handle on frame motion there is**, so this moved up: the turn is `eta` and
nothing else, and the table above is the whole of what we know about it. Cheap: bs1, 300 steps, a minute
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
   `loss_log.db` under `rankcal/*`. The side map stayed the hand map;
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

   **Run 10** (`anima_usuitrack_10_inner_lr5e5`, wandb `7ffli01x`) was that run,
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
   flat within noise across three quite different configurations.

   **Run 10's samples are reviewed and read as perturbed-then-settled**: good
   intermediate states, anatomical breakage in places, and a final anneal that
   comes back to roughly the concepts training started from. The natural reading
   is that the anneal dragged the model home. **It did not.** Distance from the
   base weights, computed from the five checkpoints, rises monotonically through
   the entire anneal -- 7.23e-3 at step 1400 to 8.80e-3 at 2200, with no role
   anywhere in the model reversing:

   | | 1400 | 1600 | 1800 | 2000 | 2200 |
   |---|---:|---:|---:|---:|---:|
   | whole model | 7.226e-3 | 7.760e-3 | 8.251e-3 | 8.627e-3 | 8.800e-3 |

   Nothing is pulled back; the model only slows down. So the tail shape is not
   the disease and the total is: **0.88% displacement over 2304 steps is simply
   not far enough to change the model durably**, and the good intermediate states
   were transient points along a short path. That argues the peak lr dominates
   and the anneal floor barely matters -- which is the opposite of what run 10's
   samples looked like they were saying.

   This read exists because per-step telemetry cannot answer it. Like
   `transport_speed` against `transport_curve` one level down,
   `update_to_param_ratio` cannot distinguish a thousand steps that add from a
   thousand that cancel. **Checkpoints against base can, they cost nothing but
   disk reads, and this lane should take that read on every run.**

4. **The sizing rule is measured, per role, and it has no fixed point.** Run 11
   reported `live_fraction` at its own table ranks over 2100 steps:

   | role | rank | mean live | frac |
   |---|---:|---:|---:|
   | attn2_v | 14 | 10.84 | 0.774 |
   | attn2_k | 20 | 15.67 | 0.783 |
   | attn2_out | 9 | 7.35 | 0.817 |
   | attn1_out | 25 | 20.87 | 0.835 |
   | attn1_k | 51 | 43.82 | 0.859 |
   | ff_down | 18 | 15.57 | 0.865 |
   | attn1_q | 37 | 32.19 | 0.870 |
   | attn1_v | 18 | 15.96 | 0.887 |
   | attn2_q | 18 | 16.01 | 0.889 |
   | patch_embed | 26 | 23.27 | 0.895 |
   | proj_out | 26 | 23.43 | 0.901 |
   | ff_up | 20 | 18.79 | 0.940 |

   **Liveness is a clip, not slack.** Planes below the Gram floor have their
   rotation zeroed, so `attn2_v` at 0.774 has a fifth of its frame that never
   turns -- a basis partly frozen, tracking a subspace it cannot follow. The
   target is `0.90-0.95`; over-provisioning has measured twice as costly as
   under-provisioning, so the correction is downward and biased lean.

   **The demand is a function of the rank it is measured at, and run 12
   measured the exponent.** Run 12 applied `mean / 0.95` per role and every role
   moved up -- by about half of what a fixed demand predicted:

   | role | r 11 -> 12 | live 11 -> 12 | frac predicted | frac measured |
   |---|---|---|---:|---:|
   | attn2_v | 14 -> 11 | 10.84 -> 9.16 | 0.985 | 0.833 |
   | attn2_k | 20 -> 16 | 15.67 -> 13.42 | 0.979 | 0.839 |
   | attn2_out | 9 -> 8 | 7.35 -> 6.85 | 0.919 | 0.856 |
   | attn1_out | 25 -> 22 | 20.87 -> 19.34 | 0.949 | 0.879 |
   | attn1_k | 51 -> 46 | 43.82 -> 40.92 | 0.953 | 0.890 |
   | attn1_q | 37 -> 34 | 32.19 -> 30.46 | 0.947 | 0.896 |
   | ff_down | 18 -> 16 | 15.57 -> 14.61 | 0.973 | 0.913 |
   | attn1_v | 18 -> 17 | 15.96 -> 15.54 | 0.939 | 0.914 |
   | attn2_q | 18 -> 17 | 16.01 -> 15.65 | 0.942 | 0.921 |
   | proj_out | 26 -> 25 | 23.43 -> 23.09 | 0.937 | 0.924 |
   | patch_embed | 26 -> 24 | 23.27 -> 22.30 | 0.970 | 0.929 |
   | **ff_up (control)** | 20 -> 20 | 18.79 -> 19.07 | 0.940 | **0.954** |

   **The demand contracts as `live ~ r^0.6`.** Per-role exponents span
   `0.37-0.70` with a median of `0.596`, and `ff_up` -- the one role whose rank
   did not change -- moved `+1.5%`, so there is no run-level drift hiding in the
   others. That fixes the whole shape of the rule:

   - `frac = live / r ~ r^-0.4`. Fraction is bought by cutting rank, and slowly:
     a 10% rank cut buys 4% of fraction.
   - **`rank <- live / 0.95` is the wrong rule** at any target above what the
     current rank already gives; it under-corrects by about 2.5x. The rule that
     converges in one step is `r_new = r * (target / frac)^(1/(alpha - 1))`,
     i.e. `(target/frac)^-2.5` at `alpha = 0.6`.
   - **Fraction costs live planes.** Taking `attn2_v` from 0.833 to 0.95 means
     rank 11 -> 8 and live 9.2 -> 7.5: a fifth of the directions the tracker
     actually turns, spent to stop carrying planes it cannot. Across the fleet
     the 0.95 target costs roughly 8-18% of live planes per role.

   **So the target is now a real decision, not a default.** 0.95 came from LFM,
   where it was free -- the calibrated table landed at 0.946 with nothing to
   contract. Here it is a trade: dead planes are clipped and never turn, but
   buying their removal spends planes that do. What decides it is the evidence
   that over-provisioning costs about twice what under-provisioning costs; what
   is missing is whether that still holds when the cut reaches this deep.
   **Discuss the target before the next table.**

   Note this is also a per-role step change -- the step carries `sqrt(r)`, so
   `attn2_v` drops 11% and `attn1_v` 3%. Deliberate, but it means run 12 is not
   a clean read of the freeze alone on any magnitude-sensitive metric.

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
