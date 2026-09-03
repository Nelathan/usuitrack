# Open design questions

Working notes, not a specification. `SPEC.md` describes what the optimizer does
today; this describes what we are unsure about and what would settle it.

**Two ground rules.**

*Synthetic gradients cannot answer any of these.* Random or planted-low-rank
data has a clean spectral cliff, iid noise, and no correlation between steps.
Real gradients have none of those properties. Synthetic tests are for finding a
steering rack connected backward, not for deciding a design.

*Release quality means honest about limits, not bulletproof.* We are not going
to build a fortress against every failure a stranger's model might produce. The
target is good enough to finish, clear enough that an experienced ML engineer
can see what it does, and honest enough that when they hit something we did not
anticipate, the docs told them where the edges were.

**If something is parked, it is in this file or it will not happen.**

---

## P1. CLOSED -- Adafactor is gone

**Answer: it was not needed, by either consumer.** Deleted from the release --
state, FLOPs, both `adafactor_*` constructor arguments, the two fused kernels
that carried it, and the `row_mute_fraction` diagnostic that existed to watch it.

Measured on LFM2.5-350M, rank 128, bs16 x seq1024, 1k steps, LR `2e-4`, one
variable apart across three arms: conditioning bought `0.0035` of target loss for
2.5% of walltime and 1.5% of matrix state. Removing it from the moment as well
was *better* than keeping it there once the aim read raw -- there was no
configuration in which conditioning the moment alone paid. (Numbers in wandb;
this file carries the reasoning.)

**The reason it is a good deletion is not the walltime.** The Oja action is
`(G^T G) B^T`, and a frame fitted on `G` **is** the leading eigenspace of
`G^T G`, so the action lands entirely inside the frame and the horizontal
residual is exactly zero (measured `3e-7`, numerical noise). Conditioning is a
congruence, not a similarity: it does not preserve eigenvectors, the frame stops
being the eigenspace of what the tangent sees, and a tangent survives on a
perfectly fitted frame (`1.08e-5`, 35x larger). **With Adafactor the tracker
could not converge at any step size** -- P2's plateau had a floor under it that
no schedule could reach. Without it, the tracker stops turning when the
gradient's principal subspace stops moving. That property is now locked in by
`test_a_fitted_frame_is_a_fixed_point_of_its_own_gradient`.

**Two readings that did not survive contact.** First, `tangent_concentration`
cannot be compared across these arms -- one measures the spectrum of the
conditioned gradient's tangent and the other of the raw gradient's, so the
observation "the flatter aim trains better" was comparing different quantities
and has been withdrawn. It remains a health metric *within* a configuration.
Second, Adafactor's contribution was real -- SNR weighting and a fuller spectrum,
which is why moment effective rank was historically its health proxy. The finding
is not that it did nothing. It is that the loss barely noticed.

**Confirmed on the image side.** A 2304-step Anima full finetune (2B DiT, rank
64, lr `1e-5`, wandb `7puon3ub`) ran clean on the deleted path: no failures, no
non-finite gradients, final checkpoint and 65 samples across 13 rounds. Judged on
the samples by the user, training is **subjectively better and more directional**
-- with Adafactor the model wandered more between sample rounds; without it, the
run pulls in one direction. That is the aim's fixed point showing up as image
quality rather than as a number, and it is the second model and second dtype to
agree.

**What this hands to P2.** The aim now converges, which is the property the step
rule needs and never had. The open question is no longer "does the aim need
conditioning" but **can an aim be designed that aims better on the raw gradient
than the conditioned one did** -- see P2's candidate and its trap.

*If a second moment turns out to be needed*, the shape is a projected one at
`[m,r]`, preconditioning where the update actually lives. Not a rescale of the
full matrix. That is a separate memory-versus-quality decision, not a
resurrection of this one.

---

## P2. The aim ignores persistence, and the step size reads a clock

**What is actually wrong is the aim, not only the step size.** This section
started as a step-size question. Measurement moved it.

### Established, with the measurement that settles each

1. **`sigma` is a contrast ratio and is already scale-free in the gradient.**
   `A` and `R` are both quadratic in `G`, so dividing by `mean(diag R)` cancels
   scale exactly -- rescaling the gradient 1000x leaves `sigma` identical to
   four decimals. Earlier prose here claimed `sigma` "carries the gradient's
   scale"; that was **wrong** and is withdrawn. `grad_clip_norm` does not set the
   tracker's scale, and the clip fires on ~0.05% of tensors, so it is a safety
   rail and not a tracking hyperparameter.
2. **`sigma` is not rank-free.** It is a nuclear norm over `r` planes and grows
   roughly linearly in rank (~37.6 at rank 64 against ~80 at rank 128). **Rank is
   the whole of the scale-freeness problem**, which is a much smaller problem
   than this section used to state.
3. **`sigma` anneals for ~20 steps and is then flat for 980.** Backed out of a
   1k run as `rotation_rad_sum / eta`: 170.6 at step 10, 92.0 by step 20, ~85 by
   step 50, then 86 -> 81 -> 79 from step 100 to step 1000. The `rotation`
   plateau has been blamed on the `eta` floor; **`sigma` itself plateaus**, and a
   step rule reading `sigma` would plateau with it. Any "constant `eta`, let
   `sigma` carry the annealing" proposal is falsified by this before it is built.
4. **Rank 128 is not a bottleneck.** Capture reads `0.64`-`0.74` across the
   archive at rank 128 (Oja lane ~0.66, projected-adamw 0.74). Two thirds of the
   gradient's energy is inside the frame. The idea that `sigma` plateaus because
   most energy is irreducibly outside is **wrong** and withdrawn. Note every one
   of those numbers was measured on the *conditioned* gradient; Adafactor flattens
   spectra, so raw-gradient capture is plausibly higher and is unmeasured.
5. **The tracker sits at an equilibrium, not at convergence and not starved.**
   Stable contrast (`sigma` ~80), stable capture (~0.66), concentration ~0.4
   against an isotropic floor of `1/128`. The residual is structured, not noise.
   Whether that equilibrium is the *right* one is the open question.

### The reframing

**Oja aims at this batch's covariance.** It is magnitude-weighted and
burst-sensitive by construction; persistence is exactly what it does not
measure. Three things make that expensive here rather than merely imperfect:

- Aurora/Newton-Schulz **orthogonalizes whatever is in the subspace**, so a burst
  direction that gets into the frame is amplified to full strength. The cost of
  admitting noise is asymmetric.
- The batch is ~16k tokens, against the ~256k a Muon-lineage pretraining run
  would use. Single-batch covariance is a noisy thing to aim by, and the whole
  design premise is to keep the tracked subspace small and safe *because* the
  batches are noisy.
- **Adafactor's SNR weighting was a persistence proxy** -- downweight
  high-variance directions -- and P1 deleted it. The honest statement is that we
  removed the only persistence mechanism in the aim and replaced it with nothing.
  P1's numbers say the loss barely noticed; they do not say persistence is
  worthless, only that Adafactor's expensive version of it was not paying.

**Persistence-weighted aiming was orphaned, not rejected.** The arc built and
measured it: the accumulation lattice, and C1 (sum-of-tangents) which it recorded
as *removing the noise wall*. C1 was deleted when position control won on
2026-07-11; position control was itself replaced by direct Oja three days later
on a measured head-to-head. Persistence was never tested against the mechanism
that won. That is a dropped thread, not a closed one.

### Inward and outward signals do different jobs

This resolves the self-preservation worry that blocks the obvious ideas.

The projected moment is built *from projections onto the current basis*, so it is
structurally blind to everything outside it: a direction outside the frame has
never had the chance to earn moment mass. Weighting basis directions by moment
strength therefore carries an incumbency bias. It is a sound **eviction** signal
and a useless **recruitment** signal.

**Only the Oja residual looks outward.** So: inward signals (moment magnitude,
capture, gradient-moment alignment) may rank what is already in the frame;
nomination of what should enter it can only come from the residual. Conflating
the two is the self-preserving loop.

Corollary: **capture is not an objective.** It measures fit to the current
gradient, so a slow frame on a slow gradient scores well while learning nothing.
It explains a result; it does not rank designs. Loss does.

### Candidates, none verified

- **Agreement as a step-size signal, not as an aim.** The cosine between the
  projected gradient and the moment asks "does this batch confirm accumulated
  history" -- a persistence question, not a preference, so it does not
  self-preserve. We log `grad_to_moment_ratio` (magnitudes); the cosine is a
  different and cheaper-to-read quantity we do not have. Paired with
  concentration it may disambiguate the case that has always been ambiguous:
  disagreement with high concentration is structured drift (turn); disagreement
  with a flat spectrum is noise (do not).
- **Cross-covariance aim.** The Oja action is `G^T (G B^T)` -- the gradient
  twice. `G^T M` has identical shape and cost and aims where the current gradient
  agrees with accumulated history: full-rank contact for the outside residual,
  persistence weighting from state we already store, zero new bytes. **Unverified
  in every respect** -- it is not a symmetric covariance action, so the fixed
  point and the horizontality invariant both need re-deriving before it means
  anything.
- **Revisit eigh aim**, now that Oja has a fixed point too (P1). Port exists at
  `865947e` in the lab repo; must live on a local-only branch, since the release
  keeps a minimal argument surface.

### What the first instrumented run showed, and what it means

Two instruments landed (`basis_lag_rms_sin`, `grad_moment_cosine`) and both paid
immediately. Reasoning, not the evidence trail -- the numbers are in wandb.

**The frame orbits, and the orbit is slowly shrinking.** Per-step rotation is
flat while the net displacement over five refreshes keeps falling. Same motion
per step, less distance covered, so cancellation is rising. The tracker is
neither converged nor stuck; it is settling, slowly. No previous metric could
tell those three apart.

**Inside the subspace the gradient is almost pure per-batch noise.** For an EMA
of iid noise the ratio `||Z||/||M||` is exactly `sqrt((1+beta)/(1-beta))` = 6.25
at `beta=0.95`; measured `grad_to_moment_ratio` sits at 6.1-6.5 for the whole
back half, and `grad_moment_cosine` collapses to zero by step ~110. The
persistent mean inside the frame is of order 2% of the per-batch noise.

**This is the strongest evidence yet *for* the first moment, and it explains the
optimizer.** The moment lifts signal-to-noise from ~0.02 to ~0.125 -- exactly the
6.25x the bound allows, so the averager is working at spec -- and Newton-Schulz
then orthogonalizes that to full strength. **UsuiTrack is an amplifier for a small
consistent component.** Remove the moment and the update becomes a 2%-signal
vector orthogonalized to unit norm with total confidence, which is the failure
mode the design exists to prevent. `cos(Z,M) ~ 0` is what a *working* averager
looks like when its input is noise-dominated; it is not the moment failing.

**It also deflates P1's result rather than contradicting it.** Adafactor's SNR
weighting was a persistence filter, and there is very little in-subspace
persistence to filter. The 0.0035 it bought reads less as "the mechanism was
inefficient" and more as "there was little there to weight."

**And it is a caution for the cross-covariance aim (A1).** That branch targets
`E[G]^T E[G]`, and if the mean gradient is 2% of noise, `G^T M` risks aiming at
noise against noise. The measurement is *inside* the frame and A1 aims at the
residual *outside* it, so this is not a refutation -- but whether persistent
structure exists outside the frame is now the thing to measure before A1 is
built, not after.

### The beta sweep: 0.95 was two to four times too long

Five arms, rank 128, LR `2e-4`, 1k steps, one variable apart. Run-to-run noise
was measured at the same time -- two independent `beta=0.95` runs on *different
code revisions* agreed to `3e-4` on target and `2e-3` on source, which also
confirms the guard deletion and the diagnostics are trajectory-neutral in a real
run, not just in unit tests. Every difference below is 10-60x that floor.

| beta | target | source | memory | rot | lag | `g/m` | iid `g/m` | excess |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 1.6889 | **3.0233** | 1 | 0.7428 | 0.0788 | 1.000 | 1.000 | -- |
| 0.5 | 1.6735 | 3.0388 | 2 | 0.7488 | 0.0779 | 1.754 | 1.732 | +1.3% |
| 0.9 | **1.6712** | 3.0826 | 10 | 0.7608 | 0.0763 | 4.490 | 4.359 | +3.0% |
| 0.95 | 1.6843 | 3.1137 | 20 | 0.7685 | 0.0747 | 6.440 | 6.245 | +3.1% |
| 0.98 | 1.7182 | 3.1899 | 50 | 0.7904 | 0.0743 | 10.442 | 9.950 | +4.9% |

**The shipped default was dominated on both axes.** Target bottoms at `0.9` and
degrades at an accelerating rate past it; source is monotone in `beta`
throughout, so `beta=0` owns retention. `0.5` is the better trade -- it gives up
`0.0023` of target for `0.0438` of retention against `0.9`, and beats `0.95` by
`0.011` and `0.075`.

**The moment earns its place, but barely.** `beta=0` is 59 sigma worse on target
than `0.9` -- so it is not deletable on a technicality -- yet against `0.5` it is
a near-perfect swap: `0.0154` worse target for `0.0155` better source. In
aggregate a wash. Whether that is worth a state tensor is a values question, not
a measurement one. Note the state is small by design, so this is not a VRAM
argument in either direction.

**Frame smear shows up as a measurable residue, which is the real result.**
`g/m` sits *above* the iid prediction `sqrt((1+beta)/(1-beta))` at every `beta`,
and the excess grows monotonically with memory length: +1.3%, +3.0%, +3.1%,
+4.9%. A persistent mean gradient would push the ratio the other way. What pushes
it this way is the frame rotating under the moment, so old coordinates partially
cancel and `||M||` falls below what iid noise alone predicts. Two further
signatures agree: rotation *rises* with `beta` while lag *falls*, so a long
memory makes the frame churn more and travel less. Churn, as rotation per unit
net displacement, is lowest with no moment at all: 9.43 at `beta=0` against 10.64
at `0.98`.

**Consequence for method, not just defaults.** Tracker work now runs with
`beta=0`. It removes the only other stateful component, eliminates smear by
construction, and gives the least-churning configuration available -- so what the
lag instrument reads is the tracker and nothing else.

### Moment memory is capped by frame rotation, not by beta

The moment's coordinates live in a frame that rotates under them. Transport is
the identity, which is exact for a rigid rotation -- but coordinate `i` twenty
steps ago names a *different direction in weight space* than coordinate `i` now.
The lag instrument measures that smear directly for the first time.

So `beta` faces three forces, not two: variance (longer is better, the 6.25x),
staleness of a moving mean (shorter is better), and **frame smear (shorter is
better, and its strength is set by the tracker's speed, not by `beta`)**. Pushing
`beta` past the point where the frame has rotated through the window buys smear,
not averaging. That is a candidate explanation for `beta=0.95` only barely
winning: it may be an artifact of the tracking rate it was measured at, in which
case `beta` and tracker speed must be tuned together rather than separately.

**Candidate that follows directly: couple the moment's decay to measured frame
motion** rather than fixing it. Damp the moment by how far the frame just turned,
so memory shortens exactly when rotation makes it stale and lengthens when the
frame is still. Costs nothing new -- the rotation is already computed for the
geodesic. Not yet derived or tested.

### A second moment would be largely redundant here

Suggested as a replacement for the first. It is not one: a second moment rescales
magnitudes per coordinate and cannot average *direction*, which is what the
measurement says the problem is. Newton-Schulz also discards magnitude entirely
-- orthogonalization is the Muon lineage's replacement for per-coordinate
adaptivity, which is why Muon carries momentum and no second moment. In front of
NS a second moment can only reweight directions before they are normalized, which
is second order. P1's note that "if a second moment is needed the shape is a
projected one at `[m,r]`" stands as a fallback, but the case for needing one is
weaker after this measurement, not stronger.

### Where the tracker actually stands

A step-size sweep (constant `eta`, no moment, seeded, 1k steps, frozen through
`0.3`) plus random-init and periodic-refit controls. Reasoning only; numbers are
in wandb.

**Oja tracking works** EIGH initialization hands the tracker `sigma ~ 50` and
the frame settles at 74, which looked like the aim degrading a good subspace.
But refitting by EIGH *mid-training*  
raised `sigma` from 79 to 90 every time, lowered projected-grad norm and left it
lower, and cost target loss. A single-batch eigendecomposition cannot find a
better frame once training is underway.

**`sigma ~ 50` was a property of the data, not of EIGH.** Early in training the
gradient is nearly low-rank, so any method fits it easily. As the model moves the gradient's effective rank grows and more energy necessarily sits outside a
rank-128 frame, so `sigma` rises. **The climb from 50 to 74 is the problem
getting harder, not the tracker getting worse.** Oja's frame is an implicit
average over hundreds of batches; a fresh fit is one noisy sample. After the
early phase, averaging wins, and that is what the refit control measures.

**The equilibrium at 74 is the aim's own fixed point.** Random initialization
starts at `sigma = 116` and converges *up* to 75; EIGH starts at 50 and converges
*down* to 74. Every equilibrium quantity matches from both directions --
projected-grad norm, lag, concentration. So the aim acquires correctly and is not
broken; 74-75 is simply where this formula lives.

**No step size improves it.** Across a 100x range `sigma` never goes below ~74.
`0.003` and `0.01` are indistinguishable; above `0.01` alignment degrades
monotonically and motion efficiency collapses 7.5x (net displacement per unit
turning). The aim, not the step size, sets the equilibrium -- which is why the
open work is a better formula and not a better schedule.

**Update quality is robust to subspace quality, so loss is a blunt instrument
here.** The entire 100x step-size sweep spans `0.0057` of target loss, and a
frozen basis costs only `0.0038`. Tracker changes must be judged on `sigma`,
lag, capture and concentration; loss deltas at this scale sit within a few
multiples of the noise floor and cannot resolve them.

**Tracking is not deletable regardless of what loss says.** A frozen basis
confines every update to one rank-128 subspace forever, so the rest of the
parameter space is unreachable. For continual pretraining -- the product target
-- that is disqualifying at any loss. Periodic EIGH refitting was the assumed
fallback and is now measured as *worse than tracking*, so that safety net is
gone: the tracker has to be good because there is nothing to fall back to.

**Early subspace quality matters more than late.** Random init recovers fully by
step 500 -- identical geometry to the EIGH arm -- and still finishes `0.0038`
worse on target, the same cost as never tracking at all. Damage done in the
first couple hundred steps is not repaid by later alignment. Whatever a new aim
does, it must not be slow to acquire.

**Ordering fix: correct, and null at the operating point.** The geodesic now runs
after the parameter update so one frame serves the projection, the tangent and
the lift. At `eta=0.01` the frame turns about a degree per step, so the effect is
within noise; it mattered mainly as a confound on the high-`eta` arms, where
per-step rotation was 10-30x larger. It also removes the last alibi for the
plateau: 74 with the frame held fixed through the whole step.

### The frame's motion is three quantities, not one

`rotation_rad_sum` is deleted. It was `sum_i eta sigma_i`, the tangent's nuclear
norm -- a real quantity, but not a distance, and in a unit no other metric here
shared. Displacement follows `||sigma||_2` and it followed `sum_i sigma_i`, so
the two differ by `sqrt(participation)` and do not move together; it also grew
with rank, which made it unreadable against the lag it was supposed to be
compared with. Every arm measured before this was ranked with two motion metrics
in incompatible units -- `basis_lag_rms_sin` divided by `sqrt(r)` and the lab
probe's `frame_displacement` did not.

What replaces it is a triple, all per-plane RMS sines so they divide by each
other. **`transport_speed`** is how far the subspace moved in one geodesic,
`sqrt(sum_i sin^2(eta sigma_i) / r)`, free from the eigenvalues already in hand.
**`transport_curve`** is `1 - lag/path` over the lag window: the fraction of the
travel that cancelled. **`transport_spin`** is the skew part of `Q_old^T Q_now`,
rotation of the frame's columns inside the span they already had.

How they read together, which is the only way they read at all. High speed with
low curve is a frame travelling, and it will slow as the aim converges. High
curve with low speed is a frame on its fixed point. High speed *and* high curve
is churn -- working hard, going nowhere, integrating batch noise into the basis
while it does. Low speed with low curve is ambiguous between settled and
starved, and spin separates them.

**Curve is what the step-size sweep could not see.** Halving `eta` halves a
productive drift and a useless orbit identically, which is why a 100x sweep
moved `sigma` by nothing: speed is not a quality. Curve is the first read that
is scale-free in `eta`.

**Spin is the identity-transport audit, and it came free.** Transport is the
identity in frame coordinates, which is exact for subspace motion and wrong for
in-span rotation -- that renames every coordinate the moment is stored in while
moving the subspace not at all. A horizontal-tangent geodesic gives
`Q^T Q+ = V cos(theta) V^T`, exactly symmetric, so the skew part is zero for one
ideal step and everything that accumulates over a window is holonomy plus
rounding. That makes "is identity transport still valid" a measurement rather
than an argument, and P7 answers it: yes, at a cost of 0.6% of the moment's
total smear.

**`transport_lag` is the moment's smear, read directly.** Under identity
transport a contribution from `k` updates ago has been misaligned by exactly the
principal angles over those `k` updates, so the lag interval is not a free knob:
at `1/(1 - beta)` it reads the fraction of the moment's own history that no
longer names the direction it was accumulated in. Default is now `10`, tracking
the new `beta` default of `0.9`.

**A withdrawal.** An earlier reading in this session put "directness" at 0.72 for
the raw baseline against 0.45 for ortho, concluding ortho cancels more. That was
arithmetic on interval-means of per-matrix norms with `r = 128` assumed uniform,
not a measurement. It is not evidence and is withdrawn; the instrument now
computes the ratio per matrix, against the path of the same matrix the snapshot
follows, and the 300-step arms will settle it.

### Diagnostics are three tiers, split by cost and not by usefulness

`off` is one attribute read. **`core`** is every read derivable from tensors the
step already formed -- no state, no decomposition, no extra pass -- so it can be
left on for a whole run without a decision: `transport_speed`,
`tangent_concentration`, `tangent_participation`, `projected_grad_norm`,
`grad_to_moment_ratio`, `grad_moment_cosine`, `update_to_param_ratio`,
`nonfinite_grads`. **`full`** adds the three that need a frame snapshot --
`transport_lag`, `transport_curve`, `transport_spin` -- at `[d,r]` over 32
sampled matrices. That is 16 MB at rank 128 on a 1024-wide model and a fixed
cost that does not grow with the model, since the sample count is fixed. The
VRAM-first rule did not have to adjudicate this; the state rule did, and `full`
ships in the release rather than living in the lab.

`tangent_participation` is new and free: `(sum lambda)^2 / (r sum lambda^2)`, the
effective plane count, in the same `[1/r, 1]` range as concentration. The pair is
head and bulk, and on a power law with no edge those genuinely separate. It
replaces the entire ten-metric spectrum-shape family.

**Twenty-six lab metrics deleted, not demoted.** The `sigma_energy_top*`,
`sigma_p*_over_max` and `plane_persist_*` families were built for two questions
that are now answered -- where a rank threshold would cut (nowhere; smooth power
law across four decades) and whether the aim repeats (31-38x floor raw, ~3x
under ortho). Keeping them "for study" is how the count gets back to 41. The
study tier is currently empty and that is correct; it refills with the
moment-weighted plane smear when that is built.

### Annealing needs the last tangent, not the last basis

The controller question, settled on the geometry rather than by trying things.
**The previous basis alone carries nothing**: at window 1 the net displacement
*is* the path, so curve is identically zero, and spin is exactly zero because the
overlap is symmetric. Curve does not exist below window 2.

The window *could* be had in production -- one `[d,r]` snapshot refreshed every
`n` steps plus a scalar path accumulator, one basis-sized buffer. It is dead for
a different reason: **curve at window `n` cannot respond faster than `n` steps.**

So the signal is the previous tangent, and the **polar** one. Curve is an
autocorrelation statement -- net-over-path is determined entirely by how
correlated successive tangent directions are -- so tangent agreement is a
low-latency estimator of the same quantity, at window 1 and readable before the
step rather than after it. Polar over raw on two counts: raw inner products are
dominated by `sigma_1` at concentration ~0.68, so the correlation measures
whether two consecutive batches were loud in the same plane, which is magnitude
leaking into a persistence question; and top-`k` polar costs `[d,k]`, an eighth
of a frame snapshot at `k=16`.

That gives a division the project could not state before: **the cheap proxy runs
in the update, the expensive truth runs in `full`, and the proxy can now be
checked against it.** Running agreement annealing at `diagnostics = "full"` asks
directly whether its meter tracks curve.

### `beta` default is `0.9`

Was `0.95`, which the sweep recorded as dominated on both axes. `0.9` is the best
target and the better short-run choice.

**And the `beta = 0` method rule is retired.** It existed because frame rotation
under the moment was a confound that could not be separated from the tracker's
own motion. Speed, curve and spin read the frame and nothing else at any `beta`,
while `beta = 0` makes `grad_moment_cosine` return nothing, pins
`grad_to_moment_ratio` at 1, and stops `transport_lag` meaning smear because
there is no memory to smear. The cost is that the 500-step `beta = 0` arms stop
being like-for-like; they remain a self-consistent family of their own.

### First read under the transport metrics: three arms at 300 steps

`beta=0.9`, rank 128, LR `2e-4`, seed 1, one variable apart. Reasoning only; the
numbers are in wandb.

**Ortho beats the raw aim by `0.0018` on target at neutral source**, 6x the noise
floor and the same sign as the 500-step read, at about half the size. Agreement
annealing adds `0.0003` over ortho, which is one sigma and is not a result. So
the ordering survives a step-count change and a `beta` change, and the margin is
small enough that mechanism reads have to carry the decision -- which is what
this section has said about loss all along.

**Moment smear falls 9.5x across the three arms.** `transport_lag` at the
moment's own memory reads `0.114`, `0.035`, `0.012`: the projected moment goes
from losing 11.4% of its direction to frame motion down to 1.2%. That is large,
clean and monotone in the arm ordering, and it buys `0.0018` of loss. Both
halves of that belong in the record. The mechanism the instrument was built to
see is real, and the loss barely notices -- the same shape P1 found for
Adafactor.

**The raw aim generates holonomy, and that is the strongest thing in the run.**
Bench measurement puts bf16 rounding at `5.6e-4` per update, random-walking to
`1.77e-3` over a window of ten. The raw arm reads `0.01232` against the ortho arms' `0.00167`
and `0.00122`, a 7x separation. *An earlier claim that the ortho arms sit "on the
bf16 floor" is withdrawn*: `top16 + anneal16` later read `0.00071`, less than
half the supposed floor, and the estimate came from extrapolating a `d=256, r=32`
synthetic to a `d=1024, r=128` model when spin is normalized by `sqrt(r)`. The 7x
separation stands; "at the floor" was never established. So
the raw aim does not merely move the frame further; it twists the frame's basis
against the moment stored in it, for no subspace progress. Orthogonalizing the
tangent removes essentially all of it. No metric in the previous set could have
produced this, and it is a mechanism argument for ortho independent of the loss.

**Corrected `transport_curve` runs the other way from the buggy read**, and this
matters because the buggy version made ortho look like it bought quiet by
cancelling more. It does not. Raw reads `0.702`, ortho `0.660`, annealed `0.630`
-- the raw aim is the churniest frame of the three, and every geometric read is
monotone in the same direction: the frame gets slower, straighter, less twisted
and slightly better on loss, all at once. The arithmetic checks independently,
since ortho's true path is `10 sin(eta) = 0.1000`, giving `0.653` against `0.660`
measured.

**An instrument correction, made mid-session.** `transport_speed` is read from
the eigenvalues *before* the geodesic runs, so it reports what the aim proposed.
In the release that is also what was followed, because nothing stands between
the two. Under the harness's orthogonalized tangent it is not: every live plane
turns by `eta` regardless of `sigma`, so the proposed speed ran about 2x the
followed one and every `transport_curve` built on it was inflated by that
factor. Speed and curve were discarded for both ortho arms and the arms re-run.
The cause was this session's own cull -- the lab probe measured *proposed* and
*followed* separately and the deleted comment said exactly why; collapsing them
into the proposed one lost the denominator. `full` now accumulates the path from
the frames actually written, so curve's numerator and denominator are the same
kind of measurement. Lag, spin and loss were never affected: they are read from
frames, not from eigenvalues.

### The reframing: ortho has no fixed point, and that is the whole story

This supersedes the reading that ortho is simply the better aim. It is a better
*shape* and a broken *magnitude*, and separating those two is the design
decision this section has actually been circling.

**`polar(Delta)` has unit singular values for every live plane, and `live` is a
relative threshold.** So a frame with a tiny residual turns by exactly the same
`eta` per plane as a frame with a huge one -- the motion is independent of how
well the frame fits. Measured directly: across a 10,000x range of residual
magnitude, raw displacement moves `8.46e-1 -> 8.46e-3 -> 8.51e-5` while bare
ortho reads `5.657e-2` in every case. At the smallest residual it turns 665x
harder than the residual justifies.

The fixed point survives only at exactly `Delta = 0`, where `live` is empty. That
is a measure-zero basin, not an attractor, so
`test_a_fitted_frame_is_a_fixed_point_of_its_own_gradient` still passes while the
property it was written to protect is gone in practice.

**This explains three separate observations that were being treated as
unrelated.** Metrics being flat through an ortho run is not the tracker settling
into an equilibrium -- it is a constant angle per refresh, forced by
construction, and it could not have been anything else. The stability cliff
between `eta = 0.01` and `0.02`, with `0.005` merely worse rather than better, is
what marginal stability looks like: `eta` alone sets the motion and nothing pulls
it back. And the reason annealing improves `transport_lag` and almost nothing
else is that it is supplying, from outside, the restoring force the aim gave up.

**Bare ortho therefore violates this section's own hard constraint** -- *frame
motion must be able to anneal as the frame reaches equilibrium* -- and it should
not ship on a loss delta of `0.0018`, whatever the loss says. The earlier note
that "the mechanism was never the constraint; settling is" was right, and bare
ortho is the mechanism that cannot settle.

### Scaled ortho: the arm that makes the annealing question moot, or does not

Restore `||sigma||_2 / sqrt(r)`, the RMS singular value, as a per-matrix scalar
on the polar factor. Every plane still turns by the same angle *as every other
plane*, so the leading plane stops owning the turn -- but the overall size of the
turn tracks the residual again, so a converging frame slows down on its own.

**It is a pure shape intervention against the raw aim.** A raw geodesic moves
`eta ||sigma||_2` to first order; this moves `sqrt(r sin^2(eta ||sigma||_2 /
sqrt(r)))`, the same number. Verified to four significant figures across four
decades of residual. So raw and scaled-ortho differ in exactly one thing -- how
the same total displacement is distributed across planes -- and any difference
between them is attributable to the spectrum alone. That is a cleaner contrast
than any arm run so far, all of which changed shape and magnitude together.

**And it costs nothing.** No `[d,k]` agreement buffer, no frame snapshot, no
previous tangent. It is two extra reductions on eigenvalues the geodesic already
has. If it works, the "do we keep the n-1 tangent for annealing" question is not
decided -- it is deleted, which is the better outcome.

**It diverged, twice, and the reason retires the magnitude framing entirely.**
First in a small-angle form -- `theta = eta ||sigma||_2 / sqrt(r)` -- which asks
for 4.95 rad per plane when `sigma_max` is large, because raw's displacement is
only safe while it is concentrated in ~2 planes where `sin` saturates and bounds
their contribution. Then in an exact form, `theta = arcsin(sqrt(sum_i sin^2(eta
sigma_i) / n))`, which is bounded by construction and reproduces raw's chordal
distance at any magnitude. That one crashed too, alone on the GPU, in 36 seconds.

**And the exact form's angle *is* `transport_speed`.** `arcsin` of the RMS sine
is the RMS angle, so bare scaled ortho is precisely bare ortho at an effective
`eta` equal to the raw aim's measured speed, `0.0232` -- and bare ortho is known
to diverge at `0.02`. The arm was doomed before it ran and the session's own
metric said so in one line. Check the instrument before spending the GPU.

### The stability cliff is set by how motion is distributed, not by how much

The comparison that settles it needs no further runs:

| arm | frame speed | distribution | outcome |
|---|---:|---|---|
| raw `sigma`, `eta=0.01` | **0.0232** | `sigma`-weighted, ~2 effective planes | stable |
| bare ortho, `eta=0.02` | 0.0200 | equal across 128 | diverges |
| scaled ortho (`eta_eff` 0.0232) | 0.0232 | equal across 128 | diverges |

**The raw aim moves the frame further per step than the ortho arm that dies, and
survives.** So total displacement is not the stability criterion; its
distribution across planes is. Bare ortho does not survive because it discards
magnitude -- it survives because `eta = 0.01` happens to sit under a ceiling of
roughly `0.015` rad per plane. That ceiling exists because equal angles drive all
126 noise-tail planes at full rate, and above some rate the frame decorrelates
from the gradient faster than the aim can re-fit it. This is P2 lead 1's "a
mechanism for integrating noise into the frame", now with a failure mode attached
rather than an argument.

It also explains the `eta` sweep that started this: `0.02` explodes, `0.005` is
merely under-tracking, and the window is narrow because 126 of 128 planes are
being turned on noise. `tangent_participation` reads `0.018` -- 2.3 effective
planes of 128 -- which is the same fact from the spectrum's side.

**The fix that follows is a ceiling, not a bound on magnitude.** `theta =
min(arcsin(...), eta)` makes `eta` the *maximum* angle rather than the only one:
bare ortho during acquisition, where both sit at the cap, and annealing below it
once the residual shrinks. With the clip in place the exact and small-angle forms
agree to three decimals, since they differ only at angles the clip removes -- so
chordal becomes a correctness choice rather than a behavioural one. Worth keeping
anyway: it is free, bounded by `pi/2` where the small-angle form is unbounded,
and it leaves the clip as the only nonlinearity in the path.

### The three candidates are one family, differing only in where magnitude comes from

Reading `install_agreement_annealing` carefully: every plane leaves it with the
same singular value and the tangent is `polar(Delta) * scale`. That is *exactly*
the shape of scaled ortho. These are not rival families:

| | shape | magnitude | anneals on | state |
|---|---|---|---|---|
| bare ortho | equal angles | none, fixed `eta` | nothing | zero |
| scaled ortho | equal angles | `||sigma||_2 / sqrt(r)` | how well the frame fits | zero |
| agreement anneal | equal angles | cross-step agreement | whether the aim repeats | `[d,k]` |

So the question was never "do we keep the `n-1` tangent". It is: **once the shape
is orthogonalized, which signal restores the magnitude?** That also explains why
raw and bare ortho fail in complementary ways -- raw carries magnitude but lets
one plane own the shape (7x the spin floor, holonomy); bare ortho fixes the shape
and throws the magnitude away (no fixed point, marginal stability). Each fixes
the other's defect and reintroduces its own.

*A trap recorded so it is not walked into again.* `scaled-ortho + anneal` and
`bare-ortho + anneal` measure identical to four decimals on every metric, and
that is a **composition artifact, not a result**. The harness installs ortho last
so it is outermost, and the first thing `annealed` does is renormalize its
incoming tangent to unit directions -- discarding whatever magnitude ortho just
computed. They agree because they are the same computation. Scaling has never
actually been tested alongside annealing.

### Two closures from the same session

**The Polar-Express retraction stays at one step.** Measured across angles
`0.005` to `0.1` rad and across 1 to 128 turning planes, a single degree-5 step
lands orthonormality error at `2-4e-7`; two steps and five steps buy nothing.
The memory of a 15-step retraction is of the lab fork, and going back to 5 would
be pure weight. Nothing to change.

**The bare-`eigh` deletion holds, but P5's evidence base was narrower than it
reads.** It was deleted because it "never fired once" across four LFM runs and
Anima. It has since fired in four configurations: unannealed `agree16`,
`eta=0.02`, and both forms of bare scaled ortho -- and a jittered retry does not
rescue any of them, it fails again on the retry. So the deletion note is right
that a silently rescued decomposition would be worse. The honest restatement is
that it never fires *on arms that are already stable*, and it now serves as the
project's de-facto divergence detector.

### `grad_moment_cosine` reads the step, not the subspace

It converges to `-0.019` and it is the same `-0.019` on all three of raw, ortho
and annealed ortho, to three decimals. The projected moment is 131k elements, so
a single sample's cosine has a noise floor near `0.003` and this is a mean over
hundreds of matrices and 300 steps. It is structural, stable, and **independent
of the frame** -- which is the finding.

A systematically negative cosine means the batch gradient anti-correlates with
the direction just stepped along. Moving downhill along a direction reduces the
gradient's component on it; overshooting flips the sign. So this reads the step
size against local curvature, and it says the step lands slightly past where the
averaged direction stops being downhill.

It agrees with P11 from an independent direction -- P11 found `2e-4` beat `4e-4`
on both axes and had not found its floor -- and it is the first metric here that
reads the *step* rather than the subspace. Worth remembering when the learning
rate lane reopens: this is a cheaper signal than a sweep.

### The design decision, settled on ten arms at 300 steps

All at `beta=0.9`, rank 128, LR `2e-4`, seed 1, one variable apart. Noise floor
`3e-4` on target, `2e-3` on source.

| arm | planes | `eta` | target | lag | spin | curve |
|---|---:|---:|---:|---:|---:|---:|
| ortho + anneal16 | 128 | 0.02 | **1.7074** | 0.0197 | 0.00142 | 0.630 |
| ortho + anneal16 | 128 | 0.01 | 1.7076 | 0.0119 | 0.00122 | 0.630 |
| bare ortho | 128 | 0.01 | 1.7081 | 0.0347 | 0.00167 | 0.660 |
| bare ortho | 128 | 0.00354 | 1.7085 | 0.0151 | 0.00149 | 0.598 |
| top16 + anneal16 | 16 | 0.01 | 1.7090 | **0.0046** | **0.00071** | 0.730 |
| top16 | 16 | 0.01 | 1.7092 | 0.0123 | 0.00128 | 0.665 |
| top16 | 16 | 0.0283 | 1.7094 | 0.0323 | 0.00148 | 0.679 |
| raw `sigma` | 128 | 0.01 | 1.7097 | 0.1136 | 0.01235 | 0.702 |

**Full-spectrum rotation is the operative variable, and it survives a transport
control in both directions.** Turning only the top 16 planes was matched to
full-rank's displacement (`eta = 0.0283`) and full-rank was matched down to
top-16's (`eta = 0.00354`); full rank wins at both rates, by `1.3e-3` and
`7e-4`. Within top-16, transport changes nothing at all. The sharpest single
comparison: `ortho` at `eta=0.00354` has *worse* frame geometry than `top16` at
`eta=0.01` -- lag `0.0151` against `0.0123` -- and a better target. Matched
transport, similar lag, opposite loss ordering.

**Top-`k` rotation is partial basis freezing, which is why it loses.** Freezing
112 of 128 planes costs `1.1e-3` against full-rank ortho, roughly a third of the
`0.0038` a fully frozen basis costs, which is about the right proportion. The
headroom it buys is real -- `top16` survives at `eta=0.0283` where full-rank
diverges at `0.02`, confirming the tail sets the stability cliff -- but the
planes you stop driving are the planes you stop tracking, and that is
disqualifying for the same reason a frozen basis is.

**`transport_lag` is not an objective, and treating it as one was this section's
main error today.** `top16 + anneal16` posts the best geometry of any arm on
every axis -- lag `0.0046`, 2.6x better than the best-target arm, and the lowest
spin measured -- with a mediocre target. **A frozen basis has zero lag.** Its
curve gives it away: `0.730`, the highest cancellation in the set, sitting beside
the lowest lag, which is what "barely moving net while still turning" means. Lag
falls into exactly the trap already recorded for capture -- it explains a result,
it does not rank designs. The same caution now applies to spin and curve. They
are the right instruments for asking *what the frame is doing*; none of them
ranks a design.

**What each arm contributes, restated as one mechanism.** Raw `sigma` lets one
plane own the turn, which shows up as 7x the spin of any other arm -- genuine
holonomy, the frame twisting its own coordinates against the moment. Orthogonal-
izing the shape removes that and buys `1.6e-3`. Orthogonalizing alone then has no
magnitude and cannot settle, and sits close enough to the stability cliff that a
float-level perturbation crossed it inside 300 steps. Agreement annealing
supplies the magnitude from a time signal, and its lag falls 4.3x over the run --
`0.0506 -> 0.0119`, still descending at step 300 -- where every non-annealed arm
is flat from step 75. That is the only arm that actually settles.

**Doubling `eta` under annealing is safe and marginally better.** `0.02` posts
`1.7074` against `0.01`'s `1.7076` -- inside noise, so read it as "no worse and
stable", not as a gain. It matters because `0.02` is the rate at which bare ortho
diverges: annealing's damping outruns the instability at an opening rate that
kills the unannealed aim.

**So the design is full-rank orthogonalized rotation with an agreement-annealed
magnitude**, at a cost of one `[d,16]` bf16 buffer per matrix. The zero-state
alternatives each fail a different half, and the reason is structural rather than
empirical: **every inward state signal plateaus.** `sigma`, capture,
concentration and participation are all flat after ~20 steps, so no aim reading
its own fit can anneal -- which is P2 fact 3, written long before scaled ortho
was built and sufficient to falsify it without a run. Only a cross-step signal
declines. Annealing must come from the time axis, and the `n-1` polar tangent is
the cheapest access to it.

*A trap for the `k` sweep.* The annealing meter's `k` and the rotation's plane
count are separate knobs and were briefly conflated. Rotation stays full-rank;
`k` selects only how many plane directions the agreement meter compares. Sweeping
`k` in `{4, 16, 32}` is open and cheap; sweeping the *rotation* rank is closed.

### The annealing meter's normalizer is derived now, and `2.87` is gone

**What the fitted divisor actually was.** `install_agreement_annealing` divided
the meter's floor-relative excess by `reference = 2.87`, LFM's measured
acquisition value less its floor. Unfolding the arithmetic: the agreement is
`||head^T stored||_F^2 / k`, the mean squared cosine of the principal angles
between this step's top-`k` plane subspace and last step's, so it is already
bounded in `[0,1]` and already scale-free. Both ends of its range are known
without measuring anything -- two random `k`-subspaces inside the `(d-r)`
horizontal complement agree at `k/(d-r)`, which the code already computed as
`floor`, and an aim that repeats exactly agrees at 1.

With `N = (d-r)/k`, the shipped scale was `(N*a - 1) / 2.87` and the
anchor-at-perfect form is `(N*a - 1) / (N - 1)`. **The same function up to a
constant**, measured at `17.2` against a predicted `19.2` at step 50, the gap
being the cap clipping the fitted values. So `reference` never carried the
annealing *shape* -- only the units, and therefore where `cap` bit. It is
redundant with `eta`, and it does not transfer: Anima at rank 64 lands near
`N = 92` against LFM's `56`.

**Anchoring at perfect agreement does not work, and the reason is useful.** Real
agreement lives in `[chance, ~0.07]`, so anchoring at 1 leaves 93% of the range
unused and forces `eta` up ~19x. That throws away the property the design
depends on -- `turn <= eta` is a safety bound only while `eta` is itself under
the bare-ortho cliff. It also has higher across-matrix variance, because the
fitted `cap` had been silently clipping a tail of high-agreement matrices.

**The first replacement -- each matrix's own running peak -- worked and is now
superseded.** `scale = excess / peak`, with `peak` the largest excess that matrix
had produced: chance agreement still mapping to zero turn, the top anchor
measured per *run* rather than fitted per *model*, acquisition setting new peaks
so `scale = 1` at the safe upper bound. Everything below about it holds as
measurement. It is replaced not because it measured badly but because a peak is a
single-sample estimator of a level that has to govern a whole run -- see "One
gain for the fleet".

Measured at 300 steps, `beta=0.9`, rank 128, `eta=0.01`, one variable apart:

| arm | target | source | lag | spin | capture |
|---|---:|---:|---:|---:|---:|
| fitted `2.87` (was shipped) | 1.70757 | 3.0440 | 0.01191 | 0.00122 | 0.4243 |
| derived running-max | 1.70783 | 3.0432 | **0.00572** | **0.00084** | 0.4158 |

Lag halves, spin falls 31%, target and source unchanged, speed unchanged. **A
control run with the fitted divisor and the cold-start fix reproduced the shipped
arm to five decimals on every axis**, so the anchor is the entire effect.

**Cold start was a real bug.** With no history the meter returned `scale = 1`,
making the first basis update a full bare-ortho turn at `eta`. Harmless at
`0.01`; at the 19x `eta` the perfect-anchor form needs, it failed `eigh`
immediately. It is zero now -- no evidence the aim repeats, no turn -- at a cost
of one basis update.

### The decay anchor is falsified, and its failure is a fact about the aim

The running peak is still a *remembered* number, so the obvious next move was to
delete it by estimating the attainable ceiling instead. Writing the lag-`L`
excess as `E0 * rho^L`, the ceiling is `E0 = excess1^2 / excess2` and the scale
`excess1 / E0` collapses to `excess2 / excess1` -- the ceiling cancels, leaving
the aim's own two-lag autocorrelation and no anchor at all.

**It does not anneal.** The scale sat at `0.818 -> 0.803` across 300 steps and
the arm landed beside bare ortho (lag `0.0296` against bare ortho's `0.0347`), a
constant-magnitude controller wearing a meter.

The reason is the useful part: **the agreement's decay shape is stationary while
its level is not.** `excess1` falls ~4x over a run while `excess2/excess1` holds
at ~0.81 throughout. The aim's persistence *timescale* does not change; only its
amplitude does. So annealing has to read amplitude, and reading amplitude
requires a level reference. An anchor is not an implementation detail waiting to
be eliminated -- it is structural, which is also why `2.87` worked.

### `k` is a monotone dial, and the anchor's settling picks it

All at `eta=0.01`, full-rank turn, derived anchor, 300 steps:

| `k` | target | lag | spin | capture | anneal depth | intervals still re-peaking |
|---:|---:|---:|---:|---:|---:|---|
| 4 | 1.70782 | **0.00385** | **0.00052** | 0.4112 | 4.6x | **8 of 12** |
| 8 | 1.70786 | 0.00431 | 0.00062 | 0.4126 | 3.9x | **8 of 12** |
| 16 | 1.70783 | 0.00572 | 0.00084 | 0.4158 | 3.2x | 4 of 12 |
| 32 | 1.70795 | 0.00800 | 0.00110 | **0.4195** | 2.7x | 2 of 12 |

Target spans `1.3e-4` against a `3e-4` floor, so **loss cannot choose here** and
there is no interior optimum -- narrower `k` anneals deeper, settles harder, and
captures less, monotonically. Note `k=4` beats `top16+anneal16` on both lag and
spin while turning all 128 planes, so that arm's geometry never required freezing
anything.

What chooses is the anchor. `k=4` and `k=8` are still setting new peaks in the
*final* logged interval, so their annealing depth is not a number -- it is
whatever the run length made it. **`k = 16` is the narrowest meter whose
normalizer stops moving inside the run**, and that is the selection rule, not a
preference. Widening `k` is a weaker question, not merely a noisier one: the
Frobenius norm of `head^T stored` sums squared cosines of *principal angles*, so
it asks whether the subspace persists and forgives rotation inside it.

*Prediction on the record that came out right:* `a_k` rises with `k`, because
relaxing the constraint outruns diluting it. Measured `0.544 / 0.615 / 0.664`
early for `k = 4 / 16 / 32`.

### At 1k the loss says nothing, and that is the finding

| arm (1k) | target | source | lag | spin | capture |
|---|---:|---:|---:|---:|---:|
| fitted `2.87` | **1.66868** | 3.0840 | 0.0083 | 0.0011 | 0.4044 |
| derived running-max | 1.66920 | **3.0828** | **0.0037** | **0.0007** | 0.3964 |

**A prediction failed here and it is worth recording as failed.** The argument
was that a 300-step comparison is biased toward fast trackers -- the first ~100
steps reward a frame that chases -- so the derived arm, travelling half as far
net, should pull ahead by 1k. It did not. The fitted arm's target edge *grew*
slightly, from `2.6e-4` at 300 to `5.2e-4` at 1k.

By this project's own noise rule that is still not a result: `5.2e-4` is `1.7x`
the `3e-4` floor on one seed, below the "few multiples" bar, and source moves the
other way by `1.2e-3` against a `2e-3` floor. **The two are indistinguishable on
loss at 1k**, which means the decision cannot be made on loss and has to rest on
geometry, honesty and the absence of fitted constants -- where the derived arm
wins on every axis: 2.2x the lag, 1.5x the spin, and no model-fitted number.

**The horizon worry about the running max is answered for 1k.** The derived
anchor stops setting new peaks after ~step 150 and never sets another through
step 1000, so it does not drift with run length at this horizon. Its scale floors
near `0.15` from ~step 400 and holds, and lag flattens at `0.0037` from ~step
600: the frame settles and stays settled. Longer horizons remain untested, and
`turn_fraction_max` is the instrument that would show it.

**The periodic bumps are the data.** Both arms dip and bump at *identical* steps
(~475, ~700) under two different controllers on the same seed and data order.
That is the corpus, not the tracker.

### `transport_speed` was measuring the aim, not the frame

It was computed from the tangent's eigenvalues *before* the geodesic ran --
`sqrt(sum_i sin^2(eta sigma_i) / r)`, the displacement the aim proposed. The
ortho hook replaces those eigenvalues inside the geodesic, so the frame turned by
`eta` while the metric reported the raw spectrum's proposal. Measured, that is
`2.2x` the motion actually followed: logged `0.0217` for bare ortho at
`eta = 0.01`, where a float64 check of the geodesic gives exactly `sin(eta)`, and
a direct probe of a live frame gives `0.0102` per optimizer step.

The docstring on `_record_followed_step` had **predicted this exact factor** and
the reading was left in place anyway; `transport_curve` was migrated to the
followed distance and the published scalar was not. It now reads from the frames
at every diagnostics tier, since `frames` and `new_frames` are both already in
hand -- one matmul, no snapshot, no state. `_record_basis_motion_diagnostics` is
renamed `_record_tangent_spectrum_diagnostics`, because concentration and
participation are all it reads now.

**Scope of the damage: none to conclusions.** Every arm was measured through the
same instrument, so all ratios hold, and every ranking here rests on lag, spin,
capture and target, which read the written frames. What it invalidates is
absolute speed readings, and specifically any claim that an ortho arm's logged
speed described its frame. *One thing to re-check rather than assume:* the
ten-arm table matched `top16` to full-rank displacement at `eta = 0.0283`. If
that matching used logged speed, it matched two proposals.

A probe also settled the question the discrepancy raised: the frame receives
**one** geodesic per optimizer step, moving exactly `eta`. The three geodesic
calls per step are the three shape buckets. Dynamics were never wrong -- only the
reading.

### `eta` changed jobs and its justification did not follow

`eta` was documented as a trust radius on a *linearized* step: accurate where you
stand, degrading as you walk. That justification is dead. The geodesic here is
exact -- verified in float64 that displacement is precisely `sin(eta)`, on a true
Stiefel geodesic whose polar retraction holds orthonormality at `2-4e-7`. There
is no linearization error left to bound.

What is uncertain is the *heading*, estimated from one noisy batch. So `eta` is
now "how far to walk on a direction we only half believe" -- a trust radius on
the aim, not on the curve. That reframing is what makes `eta * scale` coherent
rather than a category error: the meter is a confidence estimate on the aim and
`eta` is the full-confidence distance, so the product is the right composition.

**The binding constraint is statistical, not geometric.** A horizontal tangent
can sign off 90 degrees of rotation and roughly 45 is the largest defensible
single turn; `eta = 0.01 rad` is `0.57` degrees and the measured cliff is
`0.02 rad`, `1.15` degrees. We die **40x below** where geometry would stop us.
What kills the frame is integrating batch noise into it, step after step, so
`eta` is a statistical parameter wearing a geometric name.

`eta = 0.01` is therefore *calibratable* rather than derived -- find the cliff,
take half. It is the last bare constant in the controller: `floor` is derived
from `k`, `d` and `r`, the `[0,1]` clamp is structural, `peak` is measured per
run, and `k` is chosen by the settling rule above. Whether something cleaner
exists for `eta` is open, and the shape of an answer would tie it to the signal
the frame can resolve above batch noise, which the meter already measures.
(Since superseded in part: the peak became a derived ceiling, and `k` turned out
to be a second gain rather than a free choice. See the `k` sweep below.)

*A stated property, not a bug:* the meter reads persistence, and persistence is
confounded with batch noise. At larger batches the aim repeats better, so the
turn scale sits near its ceiling longer and the frame turns harder for longer.
That is arguably correct -- a quieter aim deserves more trust.

**Measured since, and the dependence is weaker than this predicted.** Quartering
the batch (16 to 4) moved `turn_fraction` by 5%, not by the factor this paragraph
implies, because the ceiling is derived from participation and participation falls
with the batch. Both ends of the meter move together. See "Tangent accumulation:
measured, then removed" for the table. This was written when the divisor was a
fitted constant, where the dependence would have been real.

Anima is `bs4 x 768px` on a 2B DiT, so its aim is built from far less data per
step than LFM's, not more. Low batch is the regime to check, and it now has been
on the LLM side.

### Capture tracks lag, so it cannot rank this family either

Sorting the arms by lag sorts them by `projected_grad_norm` too, monotonically:
`0.0347 -> 0.4350`, `0.0119 -> 0.4243`, `0.0057 -> 0.4158`, `0.0046 -> 0.4072`.
A frame that keeps moving captures more of *this batch's* gradient, noise
included. So capture falls into the same trap already recorded for lag and for
`tangent_concentration`: it explains a result, it does not rank a design.

This weakens the case against `top16 + anneal16`, whose only remaining defect was
the lowest capture in the set. **The "churn" reading of that arm is withdrawn** --
churn requires high speed *and* high curve, and its speed was ordinary. Low
speed with high curve is the doc's own description of a frame sitting on its
fixed point. What it actually is: a clean, quiet frame on a subspace that catches
less gradient, which is a position on the settling/capture axis rather than a
failure. It is still not worth freezing 112 planes for, because `k = 4` reaches
better geometry with a full-rank turn.

### One gain for the fleet, derived from the aim's effective rank

**The running peak had to go, and the objection was never empirical.** A peak is
one sample deciding a level that governs the rest of training. It settled by step
150 at 1k and did not drift, which is evidence it had not yet broken -- not
evidence it was right.

**What the per-matrix dump showed.** 92 matrices, 400 steps, `k=16`, logging
every matrix's raw `excess` each step rather than the fleet mean that had been
hiding this:

| quantity | p10 | p50 | p90 | max | p90/p10 |
|---|---:|---:|---:|---:|---:|
| peak `excess` | 0.0855 | 0.1495 | 0.2018 | 0.2581 | **2.36x** |
| settled `excess` (steps >= 300) | 0.0108 | 0.0215 | 0.0458 | 0.0584 | 4.24x |
| ratio peak/settled | 3.57 | 5.41 | 11.35 | **23.33** | 3.18x |

Two readings, and they point opposite ways. **The peaks are tight** -- 2.36x
across the middle 80% -- so the matrices are alike enough that a single global
gain is a defensible approximation. **The depths are not:** the running peak
hands the median matrix a 5.4x anneal, the p90 matrix 11.4x, and one matrix 23x,
none of it chosen. That is the concrete form of "a one-time max is bad".

**A candidate died here, cleanly.** Giving each matrix its own ceiling from its
own `tangent_participation` -- `participation * r / k`, the aim's effective rank
over the meter width -- leaves a **3.63x** spread between predicted ceiling and
observed peak, *worse* than the 2.36x you get with no per-matrix term at all.
Per-matrix participation does not predict per-matrix peak. It added noise.

**But the same quantity works at the fleet level, and that is the result.**

```
median participation * r / k  =  0.01766 * 128 / 16  =  0.1413
median observed peak                                 =  0.1495
```

6% apart. So the gain is not a constant to fit -- it is the fleet's effective aim
rank divided by the meter width, computed every step from eigenvalues the
geodesic already has. `scale = clamp(excess / G, 0, 1)`, one yardstick for
everyone, so matrices that hold their aim longer keep turning rather than having
their differences normalized away.

Measured at 300 steps against the arm it replaces:

| arm | target | source | lag | spin | curve | capture |
|---|---:|---:|---:|---:|---:|---:|
| fitted `2.87` | 1.70757 | 3.0440 | 0.01191 | 0.00122 | 0.630 | 0.4243 |
| derived running-max | 1.70783 | 3.0432 | 0.00572 | 0.00084 | 0.687 | 0.4158 |
| fleet gain | 1.70803 | 3.0456 | 0.00593 | 0.00086 | 0.683 | 0.4170 |

**Confirmed at 1k, all three arms on the same metric code:**

| arm (1k) | target | source | lag | spin | capture |
|---|---:|---:|---:|---:|---:|
| fitted `2.87` | 1.66868 | 3.0840 | 0.00829 | 0.00106 | 0.4044 |
| derived running-max | 1.66906 | 3.0832 | 0.00375 | 0.00066 | 0.3962 |
| fleet gain | 1.66893 | 3.0843 | **0.00362** | **0.00064** | 0.3967 |

Fleet and running-max are indistinguishable -- `1.3e-4` on target against a
`2.9e-4` same-config spread, `1.1e-3` on source against a `2e-3` floor, which
also clears the one number that was marginal at 300 steps. Both beat the fitted
divisor by 2.3x on lag and 1.6x on spin at equal loss.

**And the gain moves, which is the whole argument.** Logged over the run it reads
`0.092` at step 25, climbing to `0.13-0.15` and holding there in a +-7% band --
no smoothing needed. It rises because `tangent_participation` rises (`0.0136` to
`0.0195` over 1k) as the gradient's principal subspace flattens and the aim
spreads over more planes. The attainable agreement ceiling genuinely **increases
by ~47% across a run**.

That is what separates the two anchors even though they produce the same numbers
today. The running peak froze this level at ~step 150 and never revisited it; the
fleet gain tracks it. At 1k the drift is small enough not to show. On Anima's
2400 steps, or any longer run, an anchor frozen during acquisition is describing
a spectrum the model has since left behind.

**It matches rather than beats, which is the point.** Every axis lands inside the
`2.9e-4` spread measured between two runs of the *same* config, except source at
`2.4e-3` against a `2e-3` floor -- the one number not clearly inside noise and the
reason a 1k pair is running. What it buys is subtraction: no per-matrix state, no
remembered extremum, no horizon dependence, and no fitted constant anywhere in
the controller.

*Two properties to carry forward.* The `r / k` factor assumes one rank across the
fleet; mixed ranks need it per bucket. And `turn_fraction_max` changes meaning
between the two anchors -- under the peak it flagged a new record, under the
fleet gain it only means some matrix clamped -- so it stops being a drift
instrument here.

*Where the constants stand now.* `floor` derived from `k`, `d`, `r`. The `[0,1]`
clamp structural. The ceiling derived per step from the fleet's participation,
and measured invariant across a sixteenfold batch range. `k = min(16, r)`, which
the code enforces with `min(AGREEMENT_PLANES, rank)`.

That reads as one bare number, `eta`, but the `k` sweep says it is really
**one-and-a-half**: `k` is a second gain on the same quantity, `agreement_ceiling
x k` being flat within 7% across an eightfold range. The two are redundant in the
way `2.87` was redundant with `eta`, and the honest count will not be one until
that is resolved. P2's history is a story of deleting these one at a time --
`2.87` to a running peak, the peak to a fleet reading -- and this is the next one
in the queue, not a settled state.

### Open leads, ranked by what they would settle

1. **CLOSED -- few-plane rotation lost; orthogonalizing the tangent won.** The
   premise was right and the conclusion inverted. Concentration is `0.69`-`0.71`
   and `tangent_participation` reads `0.017`-`0.019`, so in energy terms the
   tangent is effectively rank 2.3 out of 128 -- the tail really is near-isotropic
   noise. But turning *only* the top planes (`rank1-plane-500`) was worse than
   baseline on target and left lag and spin at baseline levels, while
   orthogonalizing the tangent so every plane turns by the *same* angle was better
   than baseline on both. The tail is not the problem; the leading plane's
   magnitude dominating the turn is. Both are ways of distrusting the spectrum and
   only one of them works.
2. **CLOSED -- the live set saturates, and rank is a target/source dial.**
   Asked whether rank 64 is oversized for Anima, on the reading that
   `tangent_live_fraction` of `0.45` means half the planes are wasted. Tested on
   LFM at **bs1**, which is Anima's operating point in aim terms -- the ladder
   above reads concentration `0.829` at bs1 against Anima's `0.838`, effective
   rank `1.6` against `1.55`. 300 steps, seed 1, `2e-4`, beta `0.9`, everything
   else held.

   | read | r128 | r64 | r32 |
   |---|---:|---:|---:|
   | `tangent_live_fraction` | 0.4703 | 0.7998 | 0.9744 |
   | **live planes** | **60.2** | **51.2** | **31.2** |
   | `tangent_participation` | 0.01228 | 0.02414 | 0.04701 |
   | **effective planes** | 1.572 | 1.545 | 1.504 |
   | `tangent_concentration` | 0.8287 | 0.8354 | 0.8419 |
   | `turn_fraction` | 0.2524 | 0.2883 | 0.3514 |
   | `transport_speed` | 0.001555 | 0.002128 | 0.002757 |
   | `transport_curve` | 0.6981 | 0.6293 | 0.5974 |
   | optimizer peak reserved | 1325 MB | 1082 MB | 996 MB |
   | target | 1.737169 | **1.731236** | 1.733350 |
   | source | 2.913690 | 2.907390 | **2.889890** |

   *Two predictions were on the table and both were wrong.* The assistant's was
   that the live set is a fixed top-N slice, so `live_count` would hold flat and
   the fraction rise as `N/r` -- roughly `0.94` at r64. The user's was that live
   planes grow with rank, so the fraction would rise only a little. Measured:
   `live_count` goes `31.2 -> 51.2 -> 60.2`, which is **near-proportional from 32
   to 64 (+64% for 2x rank) and saturating from 64 to 128 (+18%)**. Neither a
   fixed set nor proportional growth. There is a knee, and at bs1 it sits near
   rank 64.

   **Effective planes are invariant to tracked rank** -- `1.50`, `1.55`, `1.57`
   across a 4x range. The aim carries about one and a half planes whatever the
   frame's size. Rank does not change what the gradient offers; it changes how
   many planes are positioned to see it, and past the knee it buys planes that
   cannot.

   **Rank is a target/source dial here, and the two heads disagree.** Target has
   an interior optimum at r64; r128 costs `5.9e-3` and r32 costs `2.1e-3`, both
   above the `3e-4` floor. Source improves monotonically as rank falls, `2.9137
   -> 2.8899`, a `2.4e-2` spread against a `2e-2` floor... on the boundary, but
   monotone across three arms rather than a single jump. So on this lane extra
   rank costs the prior and buys nothing after the knee.

   *The frame reads agree.* Lower rank is faster, longer-lagged and less
   cancelling -- `transport_curve` falls `0.698 -> 0.597` as rank drops. High
   rank is the churning end. That is the same conclusion from the motion side:
   the planes past the knee contribute travel that cancels.

   *Memory is not the reason to do it.* `1325 -> 996 MB` across a 4x rank cut.
   Activations dominate the budget on the 12GB card, so rank cannot buy batch.

   **What this implies for Anima, as a prediction and not a result.** Anima at
   rank 64 reads `live_fraction 0.45`, so `live_count 28.8` -- essentially LFM
   bs1's r32 arm (`31.2`). Its live set already fits in 32 planes. Rank 32 on
   Anima should therefore raise `live_fraction` to roughly `0.9`, hold effective
   planes near `1.5`, and -- if the LFM lane transfers -- improve prior
   preservation, which is the axis the samples are judged on. **The transfer is
   an assumption:** LFM source loss is not DiT sample quality, and nothing yet
   measures that link.

3. **CLOSED -- the floor is not load-bearing, and it is not a substitute for
   rank.** Asked whether the derived liveness floor should be raised, on the
   argument that an eigenvalue above the backward error does not imply a
   trustworthy *eigenvector*: for a near-degenerate cluster, eigenvector error
   scales as `eps * lambda_max / gap`, not with the eigenvalue, and the
   controller turns every live plane by one common angle -- so a plane whose
   direction is unreliable is turned as confidently as one that is not. Swept
   `LIVE_FLOOR_SIGMAS` at 1, 1.414, 2, 4 -- a 16x range in `eps` -- at ranks 128
   and 64, LFM bs1, everything else matched to the rank sweep above.

   | arm | `live_fraction` | live planes | `transport_curve` | `transport_speed` | target | source |
   |---|---:|---:|---:|---:|---:|---:|
   | r128 x1 | 0.470 | 60.2 | 0.698 | 0.00156 | 1.73717 | 2.91369 |
   | r128 x1.414 | 0.394 | 50.4 | 0.710 | 0.00146 | 1.73641 | 2.91155 |
   | r128 x2 | 0.323 | 41.4 | 0.723 | 0.00135 | 1.73629 | 2.91234 |
   | r128 x4 | 0.204 | 26.1 | 0.765 | 0.00106 | 1.73553 | 2.90889 |
   | r64 x1 | 0.800 | 51.2 | 0.629 | 0.00213 | 1.73124 | 2.90739 |
   | r64 x1.414 | 0.721 | 46.1 | 0.633 | 0.00206 | 1.73123 | 2.90567 |
   | r64 x2 | 0.631 | 40.4 | 0.647 | 0.00195 | 1.73118 | 2.90565 |
   | r64 x4 | 0.445 | 28.5 | 0.677 | 0.00166 | 1.73089 | 2.90436 |
   | r32 x1 | 0.974 | 31.2 | 0.597 | 0.00276 | 1.73335 | 2.88989 |

   **The floor is not load-bearing on loss.** At rank 64 target reads `1.73124`,
   `1.73123`, `1.73118`, `1.73089` across the whole 16x `eps` range -- a total
   span of `3.5e-4` against a `3e-4` floor. Source improves monotonically at both
   ranks but by `3.0e-3` against a `2e-3` floor. Both are hints, neither is a
   result. `tangent_participation` and `tangent_concentration` are invariant to
   five decimals across every arm, which is the control: the knob changes the
   response to the aim and not the aim.

   **So the constant stays at the derived `1.0`** -- not because moving it is
   risky, but because nothing justifies leaving an anchor that costs nothing to
   keep. The information below the floor really is too flimsy to matter, which
   was the user's read; it is also too flimsy to be worth the effort of
   excluding.

   **The floor cannot substitute for rank, and the failure is instructive.**
   r128 at x4 carries `26.1` live planes -- *fewer* than r32's `31.2` -- and
   still reads target `1.73553`, worse than r64 by `4.3e-3` and worse than r32.
   Matching the live count from above never reproduces the lower-rank result.

   **The two knobs move churn in opposite directions.** Dropping rank 128 to 32
   takes `transport_curve` `0.698 -> 0.597` with speed rising. Raising the floor
   at r128 takes it `0.698 -> 0.765` with speed *falling*. Muting a plane does
   not make the frame travel more purposefully; it makes it travel less and
   cancel more.

   *The mechanism, restated.* **Rank is the subspace the update is
   orthogonalized in; the floor is only the subspace the turn acts on.** Muting
   a plane stops it rotating but leaves it in the space the weight update lives
   in, so its ballast is still carried -- just held still, which is why churn
   rises. An earlier reading in this session had the ballast in the turn. It is
   in the update.

   *Consequence for the "keep `live_fraction` above 0.9" rule.* Raising the floor
   pushes the fraction **down** (`0.470 -> 0.204` at r128). The only instrument
   that raises it is lower rank, which is also the low-churn and source-optimal
   end. The rule is sound and it selects rank.

   *The arm this opens, not yet run.* Mask the **update** to the live planes
   rather than only the turn. If rank's advantage is entirely "do not
   orthogonalize in directions you cannot resolve," a fixed rank with a
   live-masked update should recover r32's behaviour without a rank decision.
   This is a design change rather than a knob and has not been authorized.

   *Still unexplained.* Target prefers rank 64 over both 128 and 32 at **every**
   floor setting, so its preference is not about flimsy planes. Something about
   having ~50 planes in the update subspace helps target specifically, and
   nothing measured so far says what.

4. **Rank is tunable from a 200-step diagnostic, and one global rank is not
   optimal.** Filling the interior of the rank sweep (LFM bs1, floor 1, 300
   steps) puts the target optimum at rank 48, not 64 -- the earlier "target
   prefers 64 and nothing explains it" was an artifact of not sampling between
   32 and 128.

   | rank | `live_fraction` | live planes | effective planes | `transport_curve` | target | source |
   |---:|---:|---:|---:|---:|---:|---:|
   | 128 | 0.4703 | 60.2 | 1.572 | 0.6981 | 1.73717 | 2.91369 |
   | 64 | 0.7998 | 51.2 | 1.545 | 0.6293 | 1.73124 | 2.90739 |
   | **48** | **0.9009** | 43.2 | 1.530 | 0.6114 | **1.73094** | 2.90375 |
   | 40 | 0.9425 | 37.7 | 1.517 | 0.6028 | 1.73191 | 2.89448 |
   | 32 | 0.9744 | 31.2 | 1.504 | 0.5974 | 1.73335 | 2.88989 |

   **`live_fraction` 0.90 is the target optimum and 0.95 is the balanced
   setting.** Two independent routes land on the same point: the user's rule
   that a frame which cannot be turned as one is ballast, and 300 steps of loss.
   Source keeps improving monotonically all the way down to rank 32, so rank
   stays a target/source dial -- but its target end is now located, and it is
   neither fully live (1.0) nor the `0.80` the old default sat at.

   *And better source is spendable.* A rank that buys source can be traded back
   for target by raising LR, which is the axis P11 shows this harness walks
   cleanly. The pair (rank, LR) should be tuned together rather than rank alone.

   **The tuning procedure is cheap enough to be routine.** A 200-300 step run at
   the target batch reads `tangent_live_fraction` directly; pick the rank that
   puts it near 0.90-0.95. At bs1 on LFM each arm costs about a minute. This is
   a diagnostic, not a sweep over loss, which is what makes it affordable on
   models where loss cannot rank anything.

   **One global rank is measurably wrong.** Per-matrix liveness over 92 matrices
   x 300 updates gives an ICC of `0.776` at rank 48 and `0.734` at rank 40 --
   roughly three quarters of the variance is genuine between-matrix difference
   rather than step noise. Per-matrix means run `0.446` to `0.999` at rank 48
   (median `0.971`, p25 `0.878`): most of the fleet is saturated while a
   minority is starved. Shape explains only 48% of that spread, so the reading
   carries real per-module information beyond `min(m, n)` -- and equally, shape
   alone would capture half the benefit for free.

   *What sets the usable rank is basis coverage of the matrix volume, and that
   scales with batch.* At rank 128 the live count is `101` at bs16 and `60` at
   bs1. So the current cap of `min(m, n) / 2` is geometry only and cannot be
   right across batch sizes; a derived cap has to be a function of what the
   gradient actually supports, which is what `live_fraction` already measures.

   *Pruning needs no refit and does not reset the moment.* The frame is **not**
   sorted -- the geodesic rotates into the tangent's eigenbasis and back out
   via `eigenvectors.mT`, so column `j` is not plane `j`. To prune, rotate
   `Q -> Q V` into that eigenbasis (orthogonal, so the subspace does not move
   and orthonormality holds), which sorts planes by sigma, apply the same `V` to
   the moment, then truncate both. That rotation is exactly what
   `transport_spin` measures and transport is the identity in those coordinates,
   so nothing is lost. The decomposition is already computed every basis update,
   making prune free at any step; only *growing* needs new orthogonal directions.
   `grad_to_moment_ratio` is invariant to the rotation and **not** invariant to
   the truncation, so it steps at each prune event.

   *Dynamic rank is parked, not abandoned.* The hoped-for shape was expand and
   reduce in one motion at a refit interval, adding no new state or windows.
   It does not close: new rows initialized to zero or noise cannot be read for
   liveness, because the reading needs a frame difference. Parked pending more
   statistical reads.

5. **CLOSED -- a per-role rank table beats one global rank on both heads at
   equal budget.** LFM bs16, LR `2e-4`, 1k steps, seed 1, beta `0.9`, both arms
   on current code so the live floor is common to them. The archived
   `usuitrack-release-lr2e-4-1k` (`1.681140 / 3.094790`) is **not** the control:
   it predates the live floor and was unseeded.

   | read | global r128 | per-role table | delta |
   |---|---:|---:|---:|
   | target | 1.669330 | **1.667302** | -2.0e-3 (floor `3e-4`) |
   | source | 3.084276 | **3.079098** | -5.2e-3 (floor `2e-3`) |
   | `tangent_live_fraction` | 0.8073 | 0.9419 | +0.135 |
   | `tangent_participation` | 0.01910 | 0.02322 | +21% |
   | `tangent_concentration` | 0.6826 | 0.6749 | -0.008 |
   | `update_to_param_ratio` | 1.437e-4 | 1.651e-4 | +15% |
   | `transport_speed` | 0.00118 | 0.00137 | +16% |
   | tracked planes | 11,776 | 12,032 | +2.2% |

   **Both heads improve, which is the part that matters.** A pure effective-LR
   increase walks P11's axis -- target down, source up. This is a Pareto gain,
   so it is off that axis, and `tangent_participation` up 21% says what changed:
   the tracker resolves a better-conditioned subspace, which no learning rate
   buys.

   *The table, calibrated at rank 256 on 150 steps, and what it reallocates:*
   `w1` 208, `w3` 192, `conv.in_proj` 176, `w2` 120, `q_proj` 104, `v_proj` 72,
   `k_proj` 56, `out_proj` 40, `conv.out_proj` 32. Against a global 128 that is
   **attention 1,632 planes against 3,072 -- cut 47% -- and feed-forward 8,320
   against 6,144 -- up 35%**, for a total 2.2% larger. A global rank gives
   attention roughly twice the rank its gradients can fill while starving the
   MLP by a third.

   *The calibration was itself rank-limited.* Only 3 of 9 roles cleared
   `frac < 0.25` at rank 256; `w1` read `0.777`, still climbing. The MLP
   matrices cap at 512 on the tracked side, so the table **under-provisions**
   them and the true reallocation is larger than measured.

   *Procedure, as it stands.* One over-provisioned calibration at the training
   batch, 150 steps, then `rank = live_n / 0.95` floored to a multiple of 8 and
   capped at the calibration rank -- so the calibration rank is the ceiling and
   the diagnostic can only recommend downward. Validity is per role: a role is
   settled when its calibration frac is **below ~0.25**. An earlier 0.65 line
   was too loose -- on the bs4 long run the four roles under 0.22 transferred
   exactly while the five above 0.32 over-estimated by 6-14%.

   *Drift is nil, so the interval is not needed.* On a 3600-step bs4 run with a
   per-role table, the suggested ranks are identical across windows 2-12 (3,300
   steps), the single exception being `k_proj` stepping 48 to 56 at window 3.
   Per-role `live_fraction` moves +0.014 to +0.056 over the whole run, nearly
   all of it in the first 300 steps. A checkpoint-cadence reporter would be
   reprinting a constant table; the read is worth having once, early, to confirm
   convergence.

   *Role is the right grouping.* Variance in per-matrix mean `live_fraction` is
   explained 73.6% by role, 58.4% by shape, 19.1% by depth. Role beats shape
   rather than proxying for it: `k_proj` and `v_proj` are both (512,1024) and
   read `0.711` against `0.429`. Depth's share is mostly the conv/attention
   alternation, not depth. Per-role ranks also keep the `eigh` buckets, since
   buckets key on `(device, dtype, shape[1])` and same-role matrices share a
   shape -- though that saving does not survive `release_matrix_grads`, which
   consumes gradients as they arrive rather than grouped.

6. **Rank is a step-size knob, not only a subspace knob.** Found while reading
   why `update_to_param_ratio` moved when nothing but rank changed.

   The orthogonalized projected moment is a rank-`r` object whose singular
   values are all ~1, so its Frobenius norm is `sqrt(r)`.
   The scale then multiplies by `sqrt(max(1, rows/cols))`, read from the
   parameter's own two dimensions -- `tuple(p.shape)`, nothing else. So `w1`
   (4608,1024) gets `2.121` and `w2` (1024,4608) gets `1.000`. At full rank --
   `r = min(m, n)`, which is what Muon assumes -- those compose to exactly
   `sqrt(rows)`, the invariant the line exists to enforce. Under a rank-`r`
   projection they compose to `sqrt(rows) * sqrt(r / min(m, n))` instead.

   *A correction to an earlier version of this paragraph.* It claimed the scale
   read "a shape built out of the parameter **and the basis** --
   `(p.shape[0], basis.shape[0])` on the right side" and noted "the left-side
   case carries `r` in the numerator already, an unremarked rank dependence."
   That describes `_expected_projected_grad_shape`, which **nothing called** and
   which has since been deleted. The numbers above were computed from `p.shape`
   and are unaffected; the sentence and the claimed rank dependence are
   withdrawn. Reality wins; the doc was the bug.

   *The sharper statement of the mismatch.* The factor is a function of the
   parameter's two ambient dimensions alone. The object it multiplies is
   `[d, r]`. So `r` never enters it, and neither does which of the two ambient
   axes survived the projection -- on `w1` tracked right it divides by the axis
   that was projected away, on `w2` tracked left by the axis that was kept. A
   term keyed on a geometry the update no longer inhabits cannot be expressing
   an invariant about it.

   **So the line stopped enforcing its own invariant the moment the optimizer
   became low-rank**, and what it leaves behind is shape-dependent: at `r=128`
   the residual attenuation is `0.354` for a matrix with `min(m,n)=1024` and
   `0.500` for one with `512` -- a `sqrt(2)` spread in effective step across the
   fleet that nobody chose.

   *Two readings, and they are not in conflict.* **Within a model, across roles,
   the coupling is wanted:** step scaling as `sqrt(r)` means the step scales
   with the number of directions the gradient demonstrably supports, set by
   measurement rather than by a hyperparameter. That is more principled than a
   flat rate, and it is part of why the per-role table produced a Pareto gain.
   **Across a global rank sweep the coupling contaminates:** changing one number
   moves both the subspace and the step, so every rank comparison in this
   document -- the bs1 sweep at 32/40/48/64/128 included -- conflates the two.
   "Rank 32 protects source" may be partly "rank 32 takes half the step". A
   global LR sweep cannot undo it, because the distortion is per-matrix and
   shape-dependent.

   *The aspect term is aspect, not size.* `w1` at (4608,1024) gets `2.121`;
   `w2` at (1024,4608) gets `1.000` -- identical element counts, transposed
   storage, 2x different step. `w2` also reads the lowest liveness of the three
   MLP roles in every run measured (`0.429` against `0.526`/`0.481` at bs4/r256;
   `0.457` against `0.777`/`0.721` at bs16/r256). Consistent with the smaller
   step starving it, not proof -- orientation also changes which side is
   tracked, and the two are tangled.

   **`ORTHOGONALIZATION_SCALE_MODE` is dead in production, and measurably so.**
   An arm at `scale_mode="none"` (LFM bs16, r128, 1k, seed 1) returned
   `update_to_param_ratio` of `1.437e-04` against the control's `1.437e-04` --
   identical to four figures -- with per-role `live_fraction` moving `+0.0002`
   to `+0.0011` uniformly and the `w1`-`w2` gap unchanged at `+0.193`. Removing
   a term that should have altered `w1`'s step by 2.12x altered nothing.

   The cause: `_orthogonalize_update_runtime` dispatches to
   `self._compiled_orthogonalize_update`, which is
   `_orthogonalize_aurora_muon_tensor`, and **that function hardcodes the muon
   scale and never reads the constant**. `_scale_orthogonalized_update` and its
   four-way branch are reachable only on the uncompiled path. So the optimizer
   computes a *different update* depending on whether tensor kernels are
   compiled, for any setting other than `muon` -- a divergence between two code
   paths that are supposed to agree, not merely dead code.

   **The arm therefore did not test the scale term**, and the scale question is
   still open. `graft` was the most clearly wrong of the four for this
   optimizer -- it rescales the orthogonalized update back to the original
   update's norm, re-imposing the pre-orthogonalization gradient magnitude and
   discarding the reason for orthogonalizing.

   **Resolved in code, not by measurement.** The dispatch is gone. There is one
   implementation, `_orthogonalize_update(update, scale)`, and `torch.compile`
   wraps that same function, so the two paths cannot disagree; the scale is
   `_muon_aspect_scale(original_shape)`, one function carrying the caveats
   above in its docstring. The collapse is behaviour-preserving and was verified
   as such: bitwise-identical parameters against the previous commit over 12
   steps on four shapes (right side, left side, square, rank-capped). The scale
   question stays open and is now a one-line patch with a test pinning what the
   term does and does not guarantee.

   *A correction recorded because it was stated and is wrong.* It was argued
   in session that tracking the larger side is structurally wasteful -- that a
   `[1024, r]` frame on a (512,1024) matrix spans dimensions the gradient can
   never populate. **That is false.** The frame is fitted, so Oja and the initial
   eigh place it inside the gradient's own row space, a `<= 512`-dimensional
   subspace of `R^1024`; nothing is stranded while `r < 512`, and `r` is capped
   at 256 there regardless. What the side genuinely changes is frame **memory**
   (`[d, r]`, so the larger side costs twice) and **semantics** (input-space
   versus output-space subspace). The side policy may still deserve
   re-evaluation for a different reason: `residual-facing` was chosen when rank
   was global, where aligning the frame to the residual stream was the only
   lever for making one rank fit every matrix, and per-role rank now does that
   job directly.

7. **Cross-covariance aim, `G^T M`.** Salvaged in full from `TRACKER_REDESIGN.md`
   before that document was deleted, because it was the one derivation there
   worth keeping.

   *The proposal.* Today's Oja action is `G^T (G Q)` -- the gradient acting on its
   own projection, the gradient twice. Replace the second `G` with the moment the
   design already stores: `A' = G^T M` on the right, `M G^T` on the left. Same
   shape as today's action, same cost class, **zero new bytes**.

   *Horizontality has to be re-derived, and the naive construction breaks it.*
   Today's code forms `R = sym(Q^T A)` and subtracts `QR`, which works only
   because `A = SQ` for symmetric `S = G^T G`, making `Q^T A = Q^T S Q` exactly
   symmetric -- the `sym()` in the code is float hygiene, not conceptual work. But
   `Q^T G^T M` has no reason to be symmetric; its transpose is `M^T G Q`, and
   equality would need a relationship between `M` and `GQ` that does not hold.
   Symmetrizing first therefore removes only *half* the in-subspace component and
   leaves the tangent carrying a piece inside `span(Q)` -- contaminating `sigma`
   with a quantity that corresponds to no subspace motion at all.

   *The fix is exact and cheap:* skip the Rayleigh step and project directly.
   `(I - QQ^T)X` is horizontal for any `X`, symmetric or not, since
   `Q^T(I - QQ^T)X = 0` always. So `Delta = G^T M - Q(Q^T G^T M)` is horizontal by
   construction, at one matmul of today's shape.

   *What it would buy, as an identity rather than an analogy.* If the frame has
   settled and `M`'s EMA has converged under a fixed `Q`, then `M -> E[G] Q` and
   `E[G^T M] = (E[G]^T E[G]) Q`. Against today's target that is exactly

   ```
   E[G^T G] = E[G]^T E[G] + Cov(G)
   ```

   Today's aim tracks the leading eigenspace of **signal plus per-batch noise**;
   this one tracks **signal alone**. A direction can carry large energy every
   batch while averaging toward zero across them -- bursty but not persistent --
   and it contributes to the first term and not the second. That is the
   persistence weighting the current aim structurally lacks, arrived at by asking
   a different question of the same data rather than by adding state.

   *The tension since it was proposed:* tracker work now runs at `beta=0`, where
   there is no moment to read. It needs a moment reintroduced for the aim alone,
   or a separate cheap persistence estimate -- and note the agreement meter is now
   exactly such an estimate, arrived at from the other direction.
8. **Position vs velocity control, reopened -- and the evidence has flipped.** The
   arc's centerpiece argued Oja is open-loop velocity control that random-walks
   on noise, and that EIGH-aimed position control is strictly better. The refit
   control measures the opposite on the current mechanism: a fresh
   eigendecomposition mid-training is *worse* than the tracked frame. Whatever is
   wrong with the aim, "replace it with position control" is no longer the
   obvious fix, and the arc's argument should be treated as superseded rather
   than pending.
9. **Rotation-coupled `beta`.** Damp the moment by how far the frame just turned,
   so memory shortens exactly when rotation makes it stale. Costs nothing new.
   Blocked behind the tracker work, since the optimum moves with tracking speed.
10. **Agreement `cos(Z, M)` as a step-size input.** Instrumented and understood but
   not used. Same `beta=0` tension as lead 2.
11. **Does tracking's value grow with horizon? -- partly answered, and the
   answer was no.** The prediction was that a 300-step comparison favours fast
   trackers, so a slower-net-travelling frame should pull ahead by 1k. Measured,
   the gap moved the other way by `2.6e-4`, and both deltas are under this
   harness's noise bar. Frozen `sigma` saturates by step 200, so the *geometric*
   gap is bounded; whether a bounded geometric gap ever produces a loss gap is
   still open, and now needs a horizon past 1k to ask. The frozen-vs-tracked pair
   is the shape of the test.
12. **Re-sweep rank and LR.** The biggest levers on loss, and the current
   `rank=128` / `2e-4` pair is inherited from before the aim's shape and its
   magnitude rule both changed. A substantial algorithm change invalidates the
   sweep behind it, and this one has not been redone in a long time. Debt, not
   today's work.
13. **Is `eta` derivable, or only calibratable?** The last bare constant in the
   controller. Its cliff is measurable (`0.02`, where a fixed seed diverges one
   run in two), so "find the cliff, take half" is available and honest. A derived
   form would tie it to the signal the frame can resolve above batch noise, which
   the meter already measures as `excess` above `floor`. Nothing beyond a sketch.
14. **CLOSED -- `k` is a gain on the turn, not a measurement width.** Swept 4, 8,
   16, 32 under the derived ceiling. It moves the turn scale 4.3x and target loss
   by less than the same-config spread. It is a second knob doing `eta`'s job,
   which is a redundancy of the same kind as the fitted `2.87` was. See below.
15. **CLOSED -- the fleet gain is stable enough to divide by.** Logged across 1k
   it settles into a +-7% band after acquisition and needs no smoothing, while
   rising ~47% over the run as participation grows. Stable as a divisor, and
   non-stationary in exactly the way a frozen peak cannot follow.
16. **CLOSED -- `beta` default is now `0.9`.** It was `0.95`, which the sweep
   recorded as dominated on both axes; `0.9` is the best target and the better
   short-run choice. `0.5` remains the best trade and `0.0` the best retention if
   the values call ever changes. The `beta = 0` *method* rule is retired with it:
   it existed because frame rotation under the moment could not be separated from
   the tracker's own motion, and speed, curve and spin now read the frame at any
   `beta`, while `beta = 0` blinds both moment metrics and stops `transport_lag`
   meaning smear.
17. **P5's remaining constants, P6's frame guard, P12's five syncs per step.**
   Unchanged and independent of the above.
18. **CLOSED -- tangent accumulation removed.** Falsified at batch 16, where it
   cost target loss monotonically in the window, and at batch 4, where the cost
   disappeared but no gain replaced it. The controller absorbs single-batch noise
   without it. Numbers below.

### Tangent accumulation: measured, then removed

Averaged `n` Oja tangents in a frame held still, then took one geodesic on the
mean. The window was frozen by the patch swallowing `n - 1` geodesics at the
default cadence, not by `--basis-update-interval n`, which builds no tangent on
the skipped steps and would average one.

All runs: LFM2.5-350M, rank 128, seed 1, 300 steps.

**Batch 16.**

| read | baseline | `n = 4` | `n = 8` |
|---|---:|---:|---:|
| target loss | 1.70769 | 1.70848 | 1.70897 |
| source loss | 3.04537 | 3.04328 | 3.04470 |
| `turn_fraction` | 0.217 | 0.312 | 0.564 |
| `agreement_ceiling` | 0.128 | 0.219 | 0.273 |
| `tangent_participation` | 0.0180 | 0.0316 | 0.0421 |
| `tangent_concentration` | 0.699 | 0.490 | 0.449 |
| s/step | 0.765 | 0.753 | 0.753 |

Target degrades monotonically in the window; at `n = 8` the gap is `1.3e-3`
against a `2.9e-4` same-config spread. Source stays inside its `2e-3` floor.
Step time is unchanged -- the geodesic and its `eigh` were never the cost.

`transport_lag`, `transport_curve` and `transport_spin` were also logged and are
under-sampled: 12 window measurements for the baseline, 6 at `n = 8`, and 2 once
the geodesic counter was corrected. Curve read 0.683 for the baseline and 0.302
at `n = 8`. Recorded, not concluded from.

**Control for the meter's own smoothing.** Averaging `n` batches raises
consecutive-aim agreement by construction, so the turn scale can rise without the
frame needing to turn further. The arm averages the tangent but computes the whole
meter -- head comparison and attainable ceiling -- on unaveraged single-batch aims.

| read | baseline | `n = 8` | `n = 8`, unaveraged meter |
|---|---:|---:|---:|
| `turn_fraction` | 0.217 | 0.564 | 0.394 |
| `agreement_ceiling` | 0.128 | 0.273 | 0.128 |
| target loss | 1.70769 | 1.70897 | 1.70868 |
| s/step | 0.765 | 0.753 | 0.882 |

The ceiling returning to 0.128 confirms the arm meters what it claims.
`turn_fraction` falls 0.564 to 0.394, so 49% of the rise was the meter reading its
own input filter and the rest is the frame being more skewed under eight times
fewer turns. Target loss did not recover, so the cost is in averaging the tangent
rather than in over-turning. The unaveraged meter costs 18% step time: one extra
`[r,r]` eigh per matrix per step.

**Batch 4, without accumulation.** The regime accumulation exists for: the aim is
built from a quarter of the data, so it is the noisiest we run.

| read | batch 16 | batch 4 |
|---|---:|---:|
| `turn_fraction` | 0.2168 | 0.2052 |
| `agreement_ceiling` | 0.1276 | 0.1018 |
| `tangent_participation` | 0.01796 | 0.01470 |
| `tangent_concentration` | 0.699 | 0.761 |
| `transport_speed` | 0.001754 | 0.001739 |
| `transport_curve` | 0.683 | 0.683 |

The controller does not starve. The failure mode available to it was the meter
collapsing to its chance floor as consecutive aims stop agreeing; instead the turn
scale held within 5% and the frame's motion within 1%.

The mechanism is in the same table. A noisier aim is a more concentrated one:
participation falls 18% and concentration rises 9%, because one batch's leading
direction dominates. The ceiling is derived from participation, so it falls 20%
alongside. Numerator and denominator move together and the ratio holds. A fitted
divisor would have sat still while the aim's spread fell, and the turn scale would
have fallen with it -- this is the clearest evidence so far for the derived
ceiling over a fitted or frozen anchor.

**Batch 4, with accumulation.**

| read | plain | `n = 4` |
|---|---:|---:|
| target loss | 1.77368 | 1.77389 |
| `turn_fraction` | 0.205 | 0.287 |
| `agreement_ceiling` | 0.102 | 0.186 |
| `tangent_participation` | 0.0147 | 0.0283 |

The loss cost is `2.1e-4`, inside the same-config spread, against `1.3e-3` at
batch 16. It does not become a gain. The participation and ceiling rises are
measured on the averaged tangent and carry the same smoothing confound as above.

**Removed.** Accumulation exists to fix a noisy single-batch aim. Tested at the
noisiest batch we run, the aim was not the problem: the controller absorbed the
noise itself, the frame moved the same distance, and accumulation was loss-neutral
there and loss-costing at larger batch. Low batch is not a hypothetical regime --
a larger model at small batch is one we may run -- which is why this was tested
rather than assumed.

`--accumulate-polars` was never run and is removed as superseded: averaging each
batch's polar factor measured cross-batch persistence instead of magnitude, which
is what the agreement controller measures every step with no `[d,r]` accumulator.

**One bug to carry forward.** The patches swallowed `n - 1` geodesics without
telling the optimizer, so `group["basis_update_step"]` counted turns that never
happened, and the lag window -- measured in those ticks but recorded only on steps
that actually turn -- stopped aligning with its own recorder. Scaling the window
to compensate produced zero samples. Any patch that suppresses a geodesic must
decrement that counter.

### The `k` sweep: `k` turns out to be a gain

Swept the meter width under the derived ceiling, 300 steps, LFM, rank 128, seed 1.
Every figure is a run mean; the `transport_*` reads are 12 window samples each.

| read | `k = 4` | `k = 8` | `k = 16` | `k = 32` |
|---|---:|---:|---:|---:|
| target loss | 1.70820 | 1.70769 | 1.70769 | 1.70753 |
| source loss | 3.04407 | 3.04515 | 3.04537 | 3.04453 |
| `turn_fraction` | 0.1354 | 0.2003 | 0.3292 | 0.5845 |
| `agreement_ceiling` | 0.4664 | 0.2400 | 0.1227 | 0.0623 |
| `agreement_ceiling` x `k` | 1.866 | 1.920 | 1.964 | 1.994 |
| `transport_speed` | 0.00142 | 0.00189 | 0.00287 | 0.00489 |
| `transport_curve` | 0.753 | 0.701 | 0.626 | 0.582 |
| `tangent_participation` | 0.01623 | 0.01671 | 0.01714 | 0.01747 |

`agreement_ceiling x k` is flat within 7% across an eightfold range of `k`. That is
the derivation confirmed exactly: the ceiling is `effective_rank / k`, and the
aim's effective rank is a property of the gradient, not of the meter --
`tangent_participation` moves 8% across the same range.

The consequence is that `k` scales the turn. Halving it roughly halves the turn
scale and the frame's speed with it, so `k` and `eta` are two knobs setting one
quantity. That is the same kind of redundancy the fitted `2.87` had, and it should
be treated the same way.

**Why the `/ k` does not cancel, which is the part worth understanding.** The
ceiling is derived from the claim that a top-`k` meter can reproduce at most
`effective_rank` directions, so for `k` past that rank the extra planes should
contribute little and mean agreement should fall as `1/k` -- cancelling the
ceiling's own `1/k` and leaving the ratio flat. Measured, it does not. The turn
scale rises roughly as `k^0.7`, so agreement falls much more slowly than `1/k`:
planes well beyond the spectrum's effective rank still carry persistent signal.
**`tangent_participation` predicts how the aim's energy is spread; it does not
predict how deep the persistence goes, and the ceiling uses it for the second.**
That gap is the open design question, not which `k` to pick.

**And loss cannot pick between them.** Target moves `6.7e-4` across the whole
sweep, with three of the four arms inside `1.6e-4`, against a `2.9e-4`
same-config spread -- while the frame's speed changes 3.4x. At this horizon the
tracking rate is not visible in the loss, so neither `k` nor `eta` can be chosen
by it. `transport_curve` does separate them, and it favours the fast end: at
`k = 32` the frame cancels 58% of its travel, at `k = 4` it cancels 75%. Lag and
spin favour the slow end, but trivially so, because a frame that turns less has
less of both. `k = 32` also ran without incident at 1.7x the released frame speed,
which is headroom worth remembering against the `eta = 0.02` cliff.

### The controller is invariant to an 8x batch change, and the aim collapses at 16x

The meter reads whether consecutive aims agree, so the failure mode expected of it
was a noisy aim that never repeats: `excess` collapses to the chance floor and the
frame stops turning. Tested by shrinking the batch, 16 down to 1, everything else
held. Run means, 11-12 window samples per `transport_*` figure.

**bs1 did not finish.** It ran 275 of 300 steps and died in
`torch.linalg.eigh` on the tangent Gram -- error 107, "ill-conditioned or has too
many repeated eigenvalues". Its column below is what it read up to that point, not
a completed arm. Every other arm completed.

| read | bs16 | bs8 | bs4 | bs2 | bs1 |
|---|---:|---:|---:|---:|---:|
| `tangent_participation` | 0.01714 | 0.01603 | 0.01479 | 0.01357 | 0.01232 |
| `tangent_concentration` | 0.7150 | 0.7338 | 0.7593 | 0.7899 | 0.8293 |
| `agreement_ceiling` | 0.1227 | 0.1132 | 0.1018 | 0.0912 | 0.0809 |
| `turn_fraction` | 0.3292 | 0.3300 | 0.3146 | 0.3158 | 0.3356 |
| `transport_speed` | 0.00287 | 0.00275 | 0.00258 | 0.00261 | 0.00282 |
| `transport_curve` | 0.626 | 0.625 | 0.631 | 0.632 | 0.622 |

The aim degrades monotonically and measurably: participation falls 28% and
concentration rises 16%, because at smaller batch a single direction dominates.
The ceiling, derived from participation, falls 34% with it. Across bs16 to bs2 the
turn scale is flat within 6% and the frame's motion within 11% on speed and 2% on
curve.

Both ends of the meter move together, so the ratio holds. That is what the derived
ceiling exists for, and it is demonstrated over an eightfold range. The
counterfactual is quantifiable: held at bs16's ceiling of 0.1227, bs2's excess
would give a turn scale of 0.235 instead of 0.316 -- a fitted divisor would have
slowed the tracker by a quarter at bs2, for no reason other than being fitted
somewhere else.

Target loss is not comparable across this ladder -- 300 steps at bs1 sees a
sixteenth of the data -- and is not reported.

**What broke at bs1.** Two readings were available when this was written, and
the evidence did not separate them. It does now -- see *The null tail was being
driven* below, which settles it as rank collapse and supplies the mechanism.
The two readings are kept because the reasoning that narrowed them is the
reasoning that found the cause.

*Rank collapse.* At bs1 concentration reads 0.83 and participation 0.0123, an
effective rank of 1.6 out of 128, so the tangent Gram is one large eigenvalue and
a long tail of near-equal near-zero ones -- exactly the input `eigh` reports as
ill-conditioned or having too many repeated eigenvalues, which is the error
raised. On this reading the aim did not become uninformative, it became
rank-deficient, and the controller has no view of that: it reads whether the aim
repeats, and a single dominant direction repeats perfectly well.

*Divergence.* PLAN's standing reading of an `eigh` failure is that the frame
turned too hard, and that reading has been right before.

*What separates them, as far as the data goes.* Up to the last logged point the
frame shows no sign of turning hard: `turn_fraction` 0.336 against bs16's 0.329,
`transport_speed` 0.00282 against 0.00287, `transport_curve` 0.622 against 0.626
-- all flat. A frame being thrown would show as a speed spike and it does not.
Against that, the last log lands at step 275 and the crash is somewhere in the
following 25 steps, so a divergence confined to that window would not appear.
The spectrum evidence is direct and the divergence evidence is absent rather than
contradicted, which favours rank collapse without settling it. Re-running bs1 with
a tighter logging cadence would settle it and has not been done.

So the limit is upstream of the controller. `tangent_concentration` and
`tangent_participation` both saw it coming, moving monotonically across the whole
ladder, which makes them the reads to watch on a gradient that is genuinely
rank-poor rather than merely undersampled.

**This is also the first evidence against removing the `eigh` jitter.** The
relative Tikhonov shift was deleted on the grounds that `eigh_jitter_retries`
never fired once across four LFM runs, an Anima run, and ~3300 basis updates -- a
fair reading of the evidence available then. This is the case it was for. It does
not settle whether the guard should return: a jitter would have kept bs1 running,
but a frame tracking a rank-1.6 aim at rank 128 is tracking almost nothing, and
failing loudly may be the better behaviour. What it does settle is that the
failure is reachable, and the note in `optimizer.py` saying a failing `eigh` now
fails should be read as a deliberate choice with a known trigger rather than as a
condition never observed.

### The null tail was being driven, and that is what `eigh` was choking on

Closes the bs1 question above, and the Anima crashes with it. The trigger was
not the aim's rank on its own. It was a threshold in `_anneal_tangent` that let
the rank-poor aim manufacture the ill-conditioning `eigh` reported.

The annealer normalizes the tangent's plane directions by `1 / sigma`, gated on
a liveness test that read `sigma > 1e-6 * sigma_max` -- `1e-12` in eigenvalue
terms. A symmetric `[r, r]` decomposition carries backward error of order
`r * eps * lambda_max`, which is `sqrt(r * eps) * sigma_max` in sigma units:
`2.8e-3` at rank 64 in fp32, `3.9e-3` at rank 128. The old threshold sat six
orders of magnitude below what fp32 can resolve, so **no plane was ever dead**.
Every plane below the floor had its rounding artifact divided by its own
near-zero sigma, arrived as a unit-norm direction, and was then turned by the
polar step at the same angle as the plane carrying the signal. Those directions
land in the next tangent, whose Gram is the matrix handed to `eigh`. A
rank-collapsed aim therefore produced a Gram of one large eigenvalue and a tail
of near-equal near-zero ones that were *not* the aim's tail but the annealer's
own noise, amplified and fed back -- which is precisely the input `eigh` reports
as ill-conditioned or having too many repeated eigenvalues.

That closes the two readings. Divergence is out: it required a speed spike in
the unlogged window, and the mechanism needs no divergence. Rank collapse is in,
with the refinement that rank collapse alone was not sufficient -- it became a
crash only because the null tail was being driven.

**The fix is two lines and one deleted constant.** The liveness threshold is now
`sqrt(r * eps)` relative to `sigma_max`, derived from the dtype and the rank
rather than fitted. And a dead plane gets a *zero angle*, not merely a zero
tangent column: the geodesic reads `cos(eta * sigma)` on the frame's own
component along each eigenvector, so a dead plane handed the live angle
contracts that component while the tangent term it should rotate against is
zero. That is a contraction, not a rotation, and it violates the identity this
method claims when it says a zero singular plane does not move.

**A new core diagnostic, `tangent_live_fraction`,** reports the share of planes
clearing the floor. It is the read that makes the whole failure visible, and it
cost nothing -- the sigmas were already in hand.

*The LFM null check* (`live-floor-nullcheck-300`, seed 1, rank 128, bs16,
`2e-4`, 300 steps) against `released-controller-300`:

| read | baseline | live floor | delta |
|---|---:|---:|---:|
| target | 1.7076863 | 1.7076766 | **-9.7e-6** (floor `3e-4`) |
| source | 3.0453720 | 3.0452392 | -1.3e-4 (floor `2e-3`) |
| `turn_fraction` | 0.21677 | 0.21993 | +0.003 |
| `transport_lag` | 0.005978 | 0.005095 | -15% |
| `transport_spin` | 0.000859 | 0.000735 | -14% |
| `transport_speed` | 0.001754 | 0.001574 | -10% |
| `transport_curve` | 0.683 | 0.718 | +0.035 |
| `tangent_live_fraction` | -- | **0.788** | new |

Loss-neutral inside the noise floor on both heads, and the frame moves less to
get to the same place. **The prediction stated before the run was wrong, and the
way it was wrong is the finding.** A strict no-op was expected, on the argument
that LFM's spectrum has a real tail. It read `0.788`: 27 of LFM's 128 planes sit
below the fp32 floor, so LFM had been driving a fifth of its frame on rounding
error the whole time, at no measurable cost to loss. The tail runs deeper than
the fitted `k^-1.5` implies.

*Anima is the case where it was not free.* Two runs at rank 64 / bs4 died in
`eigh` inside twenty steps -- `8y7ez5zm` at step ~17, `f8qybz7v` at ~19, a
different batch element each time, so not one cursed tensor. Under the derived
floor the same config completed all 2304 steps. The control is exact: at step
10, `tangent_participation` reads `0.02361` against the crashed run's `0.02349`
and `tangent_concentration` `0.845` against `0.848`. **The aim is unchanged;
only which planes are acted on changed.** `transport_speed` at step 10 halved,
`0.00565` to `0.00255`, while `turn_fraction` held -- the frame stopped moving
in directions that carried nothing. `tangent_live_fraction` sat at `0.45`
throughout: Anima permanently resolves under half its planes.

**The user's read was half right, and the half matters.** The hypothesis was
that the last basis turn was too strong. The turn scale was not high -- Anima's
`turn_fraction` at matched phase was *lower* than LFM's, `0.654` against
`0.737`. The frame's *motion* was too strong, because a turn of ordinary size
was applied to planes carrying nothing.

**This also revises the jitter question.** The section above reads the bs1 crash
as the first evidence against removing the `eigh` jitter. That reading is now
weaker: the failure was manufactured downstream of the decomposition, and a
jitter would have masked it rather than fixed it. Three runs died of this and
each death was a correct report of a real defect. The bare `eigh` stands.

### Carry-over: open tasks from the rank session

Written down so none of it is lost. Ordered by what would be cheapest to settle.

1. **CLOSED -- `ORTHOGONALIZATION_SCALE_MODE` is collapsed.** One
   implementation, one scale, no branch; `torch.compile` wraps the same function
   the eager path calls, and a test asserts the two agree. `graft` and the two
   other dead modes are deleted, along with `_expected_projected_grad_shape`
   (never called), `SubspaceProjector.project_and_back` (never called), and a
   redundant Newton-Schulz alias. Names now say which lineage does what:
   `_balanced_polar_direction` (Aurora balances, Newton-Schulz orthogonalizes)
   and `_muon_aspect_scale`. Behaviour-preserving, verified bitwise against the
   previous commit.

   The scale being a float unlocked a free win in the same cut: the update
   buckets that share one Newton-Schulz call now key on `(projected shape,
   scale)` instead of `(projected shape, original parameter shape)`. Two
   matrices with equal `rows` and rank but different `cols` produce the same
   `(rows, r)` projected shape and, when both are tracked on their wider side,
   the same clamped `scale = 1.0` -- so they used to split into separate calls
   for no reason and now merge. Verified against per-matrix stepping at the
   float tolerance the rest of this file's batched-vs-solo comparisons use
   (`test_matrices_sharing_a_scale_but_not_a_shape_still_bucket_correctly`):
   not bitwise, since a batched and a solo Newton-Schulz call round
   differently at the ulp level, the same as any other batching already in
   this file.

2. **Test the scale term for real.** Now cheap: patch `_muon_aspect_scale`, one
   function, both paths. The open question is whether the aspect factor starves
   `w2`, which reads the lowest liveness of the three MLP roles in every run
   measured while receiving half `w1`'s step. Note what the arms should be --
   dropping the factor leaves `||U||_F = sqrt(r)` for every matrix (rank
   coupling only, no shape coupling), while restoring the full-rank invariant
   with `sqrt(min(m,n)/r)` would also delete the rank coupling, which is the
   part we want to keep.

3. **CLOSED -- the `side=right` arm has the best geometry in the session and
   the worst loss.** LFM bs16, r128, 1k, seed 1, against the residual-facing
   control and the per-role table:

   | read | residual-facing | per-role table | **side=right** |
   |---|---:|---:|---:|
   | target | 1.669330 | **1.667302** | 1.677500 |
   | source | 3.084276 | **3.079098** | 3.089930 |
   | `projected_grad_norm` (capture) | 0.392509 | 0.383710 | **0.451346** |
   | `tangent_live_fraction` | 0.807303 | **0.941929** | 0.901274 |
   | `tangent_participation` | 0.019102 | 0.023217 | **0.023381** |
   | `tangent_concentration` | 0.682636 | 0.674940 | **0.616423** |
   | `transport_speed` | 0.001181 | 0.001370 | 0.001935 |

   Tracking the non-residual side captures 15% more gradient, spreads the
   spectrum best of any arm, and loses on **both** heads -- target by `8.2e-3`,
   which is 27x the noise floor. **This is the cleanest demonstration yet that
   capture and spectrum reads explain results without ranking designs**, and it
   is worth more than the rule stated abstractly: the arm with the best
   subspace geometry measured all session is the worst arm on loss.

   *Why, and it vindicates the original design.* The residual side is the axis
   connected to the whole model, so it is harder to track -- it moves, because
   everything upstream and downstream moves it. The other side is more
   self-contained and therefore more stable, which is exactly why it tracks
   better. **Trackability and usefulness are anti-correlated here.** The
   optimizer should track the side that is harder to track, because that is
   where the model's coupling lives. `residual-facing` was a design intuition --
   both `w1` and `w2` get a `d x r` basis on the side facing the residual
   stream -- and it now has evidence.

   *The caveat this puts on the rank programme.* `tangent_live_fraction` is a
   geometry read, and this arm proves a geometry read can be improved while loss
   degrades. Maximizing liveness is therefore not intrinsically good. The
   per-role table stands because it improved **both losses** at equal budget,
   not because it raised liveness -- and any future rank rule has to clear the
   same bar rather than pointing at the metric it optimizes.

4. **Anima with a calibrated rank table.** The second operating point for
   everything in this session: whether `0.95` transfers off LFM, whether the
   role table transfers to a DiT, and whether the source gain shows up as the
   sample quality the user actually judges on. The bs4 calibration procedure is
   the one Anima needs. Run at night.

5. **Re-read the rank sweeps under the step-size coupling -- the coupling is
   measured and exact.** `update_to_param_ratio` across the bs1 sweep tracks
   `sqrt(r/128)` to four figures:

   | rank | `update_to_param_ratio` | ratio to r128 | `sqrt(r/128)` |
   |---:|---:|---:|---:|
   | 128 | 1.43463e-4 | 1.0000 | 1.000 |
   | 64 | 1.01535e-4 | 0.7078 | 0.707 |
   | 48 | 8.78494e-5 | 0.6124 | 0.612 |
   | 40 | 8.01740e-5 | 0.5589 | 0.559 |
   | 32 | 7.16512e-5 | 0.4994 | 0.500 |

   At **fixed** rank the ratio is flat -- `1.4346e-4` for both the bs16 control
   and `scale=none`, `1.4365e-4` for `side=right` -- and it moves only where
   rank moves (`1.651e-4` for the per-role table). So the coupling is not a
   suspicion, it is the arithmetic, and "rank 32 protects source" is partly
   "rank 32 takes half the step". An arm holding effective step fixed while
   varying rank is what separates them.

6. **Calibrate deeper than 256.** Only 3 of 9 roles cleared `frac < 0.25` at
   rank 256 on bs16; the MLP roles cap at 512 on the tracked side. The current
   table under-provisions them, so the measured reallocation is a lower bound.

7. **Per-role rank needs a home, and Anima needs it.** `rank` is already a
   per-param-group key, so the optimizer already supports the table -- what does
   not exist is the part that produces and consumes one:

   * group construction by role, today a monkeypatch over
     `build_usuitrack_param_groups` in the lab harness, and absent entirely from
     ai-toolkit, which is what Anima runs on
   * a per-matrix liveness report, today a device-syncing trace hook that was
     removed before commit; the production form accumulates per-matrix sums on
     device and syncs once per report interval
   * the calibration procedure, validity rule and table format, which live only
     in this document

   This is the blocking work for trying the table on Anima.

8. **`min(m, n) / 2` as the rank cap is geometry only, and the `/2` is the
   fitted part.** Two hard ceilings exist and neither is `/2`: the frame is
   `[d, r]` with orthonormal columns, so `r <= d`; and the gradient has rank at
   most `min(m, n)`, so directions past that can never be populated. For `w1`
   (4608,1024) tracked on the 1024 axis both give `1024`, and the cap of `512`
   is the halving alone. Calibration can therefore go to `512` there but not
   beyond -- the cap clamps it. Beyond that, usable rank scales with batch --
   101 live planes at bs16 against 60 at bs1, same rank 128 -- so a ceiling
   derived from shape alone cannot be right across batch sizes, and
   `live_fraction` already measures what the gradient actually supports.

9. **CLOSED -- dynamic rank is not reopened.** The static table is stable and
   predictable and that makes it strictly better: suggested ranks are identical
   across windows 2-12 of a 3600-step run, so a controller would be tracking a
   constant while adding state, a window, and a failure mode. The mechanism also
   never closed on its own terms -- expand-and-reduce in one motion needs rows
   initialized to zero or noise, which cannot be read for liveness because the
   reading needs a frame difference. Recorded for the record: pruning alone is
   cheap and needs no refit -- rotate `Q -> QV` into the tangent eigenbasis,
   apply `V` to the moment, truncate both.

10. **Reopen `min(m,n)` as the rank cap, now that item 3 has evidence and item
    7's table exists.** Aurora's own README (`~/code/aurora-release`) confirms
    the balancing step targets exactly our shape: a rectangular matrix, not a
    square one, oriented tall before the polar map -- the projected moment is
    that shape by construction, so item 7 (balanced polar direction) is Aurora
    used the way it was built to be used, not an adaptation.

    Basis-side selection (item 3's `residual-facing` hint) and the rank cap
    (item 8's `min(m,n)/2`) are the two places the current design costs VRAM
    and flops rather than tracking better: `residual-facing` was picked on
    early evidence, item 3 confirms it on loss but shows the *other* side
    tracks better on every geometry read, and the `/2` in the cap exists only
    to leave room for the Oja residual, not because `min(m,n)` itself is wrong
    -- SubTrack (`~/code/SubTrack`) uses `min(m,n)` directly. Whether a
    calibrated rank table (item 7) or better tracking removes the need for that
    headroom, and so lets the cap move to `min(m,n)`, is open and untested. Not
    started; no evidence either way yet.

### How we work here

Recorded so a fresh session inherits the method, not just the findings.

**No parallel compiled and uncompiled implementations on main.** They drift,
and the drift is silent because both paths typecheck, both run, and only one of
them is ever exercised. `ORTHOGONALIZATION_SCALE_MODE` is the case that proved
it (since collapsed, and now guarded by a compiled-versus-eager equivalence
test): the compiled `_orthogonalize_aurora_muon_tensor` hardcodes the muon scale
while the uncompiled `_orthogonalize_aurora` honours a four-way branch, so the
optimizer computes a different update depending on a compile flag, and an
experiment that changed the constant measured nothing while appearing to
succeed. A second implementation of the same maths is a second thing to keep
true, and nothing in the test suite was watching. Where a compiled kernel is
needed for speed, it must be the *only* implementation, with the uncompiled
path either deleted or reduced to a call into the same function. Any surviving
pair on main is a defect to remove, not a convenience to maintain.

**The user steers; the assistant reads terrain.** Bring evidence, name the
uncertainty, propose the cut -- then let the direction be chosen. Do not
disappear into autonomy on questions that are the user's to answer.

**Fail early: ask and verify.** Before building on a premise, find the cheapest
check that could falsify it. Several of today's best results came from a check
that cost minutes and overturned an hour's plan. State predictions *before*
running, so a wrong one is visible as wrong.

**Rules here are values or observations, and both are open to revision through
discussion.** Nothing in these docs is settled because it is written down. When a
measurement contradicts a rule, the rule moves.

**Plausibility is not correctness.** Verify claims from other agents and from
past documents, including ones this project wrote. Reality wins; the doc is the
bug. When a prediction fails, say so plainly and reason about why -- that is the
most informative thing that happens in a session.

**Synthetic gradients are inadmissible for design decisions.** They have a clean
spectral cliff and no step-to-step correlation; `tangent_concentration` reads
~0.03 on synthetic against 0.68-0.82 on real gradients. Use them only to check
that an implementation is wired correctly.

**Loss is not always the instrument.** For tracker work it is blunt -- a 100x
step-size sweep spans `0.006` -- so mechanism reads decide, and loss only vetoes.
Know which question a metric can actually answer before quoting it.

**Measure run-to-run noise before believing a delta.** It is `3e-4` on target and
`2e-3` on source for this harness with a seed. Deltas below a few multiples of
that are not results.

**Docs carry reasoning, not evidence trails.** `SPEC.md` describes only the
current state and never history. `PLAN.md` holds open questions, insights, and
the reasoning behind decisions; the numbers live in wandb. Both files are
committed -- an earlier version of this line claimed `PLAN.md` was not, which
was simply false and is withdrawn.

**The release keeps a minimal surface.** Experiment knobs live in the harness or
on local-only branches, never as optimizer arguments. Deleting a losing option is
part of finishing an experiment.

**One GPU, so experiments are serial.** Each 1k-step LFM run is ~13 minutes, which
makes breadth affordable and makes it worth queueing arms rather than guessing.

### What is missing before any of this can be judged

**We cannot currently tell whether the frame has converged.** The arc says it
outright: "neither rotation_angle nor target self-angle can show convergence --
both are floored by target noise by construction." Every metric in the release is
floored that way. `basis_lag_angle` -- principal angles against the frame's own
snapshot N refreshes back -- is the only read that goes to zero iff the frame
actually stops. Approved as a **sampled diagnostic** over ~32 matrices, opt-in
and off the hot path; that is not the same thing as per-matrix state in the
update rule, and the VRAM-first ban is on the latter.

**The clock is gone.** `eta` is now the constant `0.01`; the `1/t` schedule is
deleted. Not as a fix -- it does not make the step size read geometry -- but as a
measurement prerequisite. Frame motion was the product of two annealing terms
(`sigma` and the schedule), so no reading could separate "the tracker settled"
from "the clock ran out", and the convergence instrument would have inherited
that ambiguity. With a constant step, `rotation_rad_sum` is `sigma` up to a fixed
factor and `basis_lag_rms_sin` measures the tracker rather than the schedule.
**And its justification is gone independently.** The hot start answered a *moving
aim*: while Adafactor's variances warmed, the conditioned Gram whose eigenspace
the tracker targets was itself shifting, so the frame chased rather than fitted.
The schedule reached its floor at basis update 100 and the variance memory was
`1/(1 - 0.99) = 100`; P3 already flagged that agreement for the `sigma` hump and
it explains the schedule too. With conditioning deleted the aim is stationary
from step 1 up to the model's own motion, so the compensation has nothing left to
compensate for. The acquisition cost is therefore measured against the wrong
baseline if compared to the old hot start -- what matters is whether a constant
step reaches the same capture and lag the harmonic runs did, which the 400-step
read answers.

**Goal.** An aim that prefers persistent structure to single-batch bursts, and a
step size that reads geometry rather than a clock -- knowing on its own when to
turn hard and when to settle. Hard constraint, restated: **frame motion must be
able to anneal as the frame reaches equilibrium.** The old wording banned "any
normalization that forces a constant angle per refresh", which generalised a
result about two scalar rescalings into a ban on spectral reweighting. That
generalisation is still wrong -- reweighting the spectrum is legal -- but the
specific thing it banned turns out to have been banned correctly: **bare ortho
*is* a constant angle per refresh, and it cannot settle**, which is why it beats
the raw aim on loss and still should not ship. The distinction that matters is
between reweighting the spectrum, which is allowed, and discarding its magnitude,
which removes the only restoring force in the aim. Scaled ortho does the first
without the second. Note the shipped
raw `sigma` does not currently satisfy it either: it anneals for ~20 steps and
then plateaus. A constant *multiplier* remains allowed; a hot start measurably
warmed the basis sooner and improved early capture.

---

## P3. CLOSED -- the sigma hump was an Adafactor artifact, and Adafactor is gone

The hypothesis was that `sigma` rises to ~11 by step 100 and then anneals, with
the hump ending exactly at `1/(1 - adafactor_beta2) = 100`, which would make the
early inflation an artifact of the factored variance warming up. There is no
factored variance any more, so there is nothing left to verify and nothing that
would act on the answer. If a hump appears in the current tracker it is a new
observation about the current tracker, not this one.

---

## P4. Cheap diagnostics that plug into someone else's stack

**Context.** The lab repo carries every diagnostic; the release is meant to be
pristine. Debugging Anima forced them back in ad hoc -- which paid for itself,
because chasing the logging syncs surfaced two more CPU syncs in the training
loop that are now fixed. That telemetry has since been removed from the release
again.

But a user running this optimizer currently has no way to answer *is the tracker
actually tracking?*, and that question has an exact answer sitting in the
eigendecomposition the update already performs.

**Goal.** Diagnostics that plug straight into whatever stack the user already
has. Concretely that means: a plain dict of floats, no assumption of wandb or
any particular trainer, opt-in and provably free when off, no device syncs on
the hot path, and read at logging cadence rather than per step. Not a feature
that only works on the machine it was written on.

**Done.** `diagnostics_enabled` plus `pop_diagnostics()`; see `SPEC.md`,
"Diagnostics", for the metric table. Every accumulation site is guarded by one
attribute read, measurements stay on-device, and the single host read happens at
drain time -- so a drained point is the interval's mean, which gives readable
lines without an EMA laid over per-step noise. Verified with CUDA sync debug
mode: five steps cost 25 synchronizing operations with telemetry off and 25 with
it on, and `pop_diagnostics()` costs 2.

What was ported from the lab's battle-tested set, and what was not: `capture`
was dropped (a weak proxy, and measuring it on the raw gradient so the P1 arms
stay comparable would have cost an extra projection per matrix per step);
`aurora_erank`, `aurora_alignment` and the leverage statistics were dropped for
the reasons in P1 and on cost; raw `update_norm` was demoted to
`update_to_param_ratio`, the only form of it that ever said anything, since the
norm alone moved only when the learning-rate schedule moved.

Both consumers are wired: the `optimizers` lab harness now imports the release
package rather than its own fork of it (it had drifted -- no rank cap, no
stochastic rounding), and ai-toolkit's `pop_basis_rotation_log` became
`pop_optimizer_diagnostics`, which had been silently returning `{}` since the
release stripped the method it duck-typed for.

---

## P5. Magic number census

Most of these were ablated on a single model.

| constant | where | status |
|---|---|---|
| `MIN_GEODESIC_STEPSIZE = 0.01` | step-size floor | P2; the load-bearing one |
| rank cap `min(m,n)/2` | `effective_rank` | settled by structure and bottleneck stability, but the *fraction* is still a choice |
| `grad_clip_norm = 1.0` | raw clip | now mandatory; the threshold itself is untested across models |
| `beta = 0.9`, `eps = 1e-8` | moment | `beta` measured by sweep, see P2; `eps` inherited |
| `AURORA_PP_ITERATIONS = 1`, `AURORA_PP_BETA = 0.5` | direction map | inherited from the method |
| `1e-12` floors, `1e-7` sigma threshold | numerical | audited once, and one of them was a bug -- see below |

**Immediate sub-question with a cheap answer.** Two guards on the hot path are
applied *unconditionally* and nothing measures whether they were ever needed:

- the eigh Tikhonov jitter, `grams += 1e-6 · (trace/r) · I`, which shifts every
  eigenvalue on every basis update
- `nan_to_num` on the tangent

We do not carry weight on a racecar because it does not hurt. Instrument both
once on a real run, then delete or make conditional. The codebase already has
the right pattern twelve lines away: `_side_gram_eigh` does a bare `eigh`,
catches the failure, *then* jitters and retries.

**Half done.** The jitter now follows that pattern -- bare `eigh`, catch, jitter
and retry, counted as `eigh_jitter_retries`. It costs nothing extra: `eigh`
already checks its convergence info on the host, so the failure was always
visible there. `nan_to_num` on the tangent is still unconditional; what exists
now is the counter (`nonfinite_grads`) that can retire it, on evidence from a
real run rather than an argument.

*The evidence is now sufficient to delete both.* Across four LFM 1k runs and a
2304-step Anima full finetune -- two models, two architectures (LLM and 2B DiT),
two ranks, bf16 throughout -- `nonfinite_grads` is **0** and
`eigh_jitter_retries` **never fired once**. The Anima run is the one that
matters: that is the model where a real non-finite failure prompted the guard,
and under the current eligibility rules nothing fires. Both paths are pure
weight on every configuration we have measured.

**Both are now deleted.** The `nan_to_num` on the tangent and the jittered `eigh`
retry are gone, and nothing replaces them: a failing `eigh` fails, because its
result steers the frame and a silently rescued decomposition is worse than a
stopped run.

Two guards deliberately survive, and neither is covered by the evidence above.
The raw-gradient sanitize at the clip protects a different thing at a different
place -- it is what stops a bad batch reaching any consumer, and the counter that
would justify removing it (`nonfinite_grads`) is the only one still reported.
`_side_gram_eigh`'s try-then-jitter on the *initial fit* also stays: a different
matrix, decomposed once per parameter rather than once per step, so it is not a
hot-path toll and the tangent-Gram evidence does not transfer to it.

**The numerical row was not fine.** It was the only row carrying "unaudited"
and it was hiding the defect that killed three runs: the annealer's liveness
test at `1e-6 * sigma_max`, six orders of magnitude below fp32's resolution, so
no plane was ever dead and the null tail was driven on rounding error. It is now
`sqrt(r * eps)` relative to `sigma_max` -- derived from dtype and rank, scale
free, nothing fitted. Full account under the batch ladder in P2.

The lesson generalizes past the one constant. A threshold that never fires is
not thereby harmless, and "probably fine" in a census is a place to look first
rather than a row to skip. The three surviving numerical constants in this row
have still never been read against the precision they run in.

**Goal.** Fewer constants, and the survivors derived or at least scale-free.

---

## P11. The learning rate was too high, and there is no baseline to compare against

**The reading, from the 1k baseline** (`usuitrack-release-baseline-1k`, rank 128,
bs16, LR `4e-4`, `StochasticAdamW` fallback):

| step | target | source |
|---:|---:|---:|
| 0 | 2.297068 | 2.917333 |
| 100 | 1.785651 | 3.082585 |
| 300 | 1.749955 | 3.135733 |
| 500 | 1.746436 | 3.145438 |
| 700 | 1.743655 | 3.188740 |
| 1000 | **1.741403** | **3.191206** |

**Target is finished by step 300. Source is not.** From step 300 to step 1000 the
target improves by `0.0086` while the source degrades by `0.0555` -- seven
hundred steps that buy almost nothing and cost six times as much as they buy.
That is not a position on a target/source frontier; it is the run continuing to
damage the model after it has stopped learning the target.

This part needs no archive comparison and carries no confound: it is one run,
one optimizer, one set of settings, read against itself.

Target falling and source rising is the known axis on this harness -- the
archive's LR sweep at rank 64 / bs32 / 200 steps walks straight down it:

| LR | target | source |
|---:|---:|---:|
| `2e-4` | 1.842040 | 2.989561 |
| `4e-4` | 1.770638 | 3.011879 |
| `6e-4` | 1.748466 | 3.034460 |
| `8e-4` | 1.743787 | 3.062698 |

The new run is nominally at `4e-4`. Early on it behaves like a much higher LR:
at step 200 its target (`1.760688`) is already better than the rank-128 500-step
archive replay's (`1.783078`), with source `0.17` worse. That is the
effective-LR shape exactly.

**There is nothing to compare it against, and that is the first finding.** The
newest wandb directory in the lab is `dbwc1x5o` (`oja-r128-lr5e4-fallback1e4-1k`,
`1.673074 / 3.056915`), and it is tempting to read the release against it. Do not.
That run started at 14:31 on 2026-07-16; the lab's last two commits landed at
14:53 and **22:31** the same day, the second being `7a9283e "Simplify UsuiTrack
tracking path"` -- 1456 deletions across the optimizer and projector, collapsing
tracking to the harmonic geodesic and deleting the refresh and grassmann
controls. **The lab's final code was never run.**

The evidence is in the config `dbwc1x5o` logged: `basis_refresh_interval=10`,
`grassmann_step_size=0.25`, `oja_step_schedule=mature` -- arguments the harness
no longer has. All 197 local run directories were checked and **not one logs
`basis_update_interval`**, so every stored run predates the current argument set.
`dbwc1x5o` is the newest directory, not the newest code.

So the release baseline has no like-for-like predecessor. Any statement of the
form "the release is worse than the lab was" is comparing across an unmeasured
tracker rewrite plus the whole port, and cannot be made from what exists.

Two cross-era consistency checks are still worth having, because they say the
telemetry measures the same quantities across the rewrite: rotation reads `0.777`
in `dbwc1x5o` against `0.816` here, and `projected_grad_to_moment_ratio` reads
`6.612` against `6.239`. And `update_to_param_ratio` scales with LR exactly as
predicted -- `3.42e-4` at `5e-4` against `2.87e-4` at `4e-4`, a ratio of `0.839`
against the LR ratio of `0.8`.

**The learning rate was simply too high.** A half-LR run (`2e-4`, everything else
identical) is better on **both** axes at every checkpoint, and is still improving
where the `4e-4` run had flat-lined:

| step | `4e-4` target / source | `2e-4` target / source |
|---:|---|---|
| 100 | 1.785651 / 3.082585 | 1.782255 / 3.028644 |
| 200 | 1.760688 / 3.078969 | **1.730615 / 3.026879** |
| 300 | 1.749955 / 3.135733 | **1.711344 / 3.049384** |
| 500 | 1.746436 / 3.145438 | **1.697692 / 3.080269** |
| 1000 | 1.741403 / 3.191206 | **1.681140 / 3.094790** |

Better on both axes at every checkpoint. It passed the `4e-4` run's *final*
target before step 300, and its source delta over the whole run is `+0.177`
against `+0.274`. There is no tradeoff being navigated here -- the higher LR was
paying source for target it was not getting.

**`2e-4` has not found its floor either.** Target is still falling between steps
900 and 1000 (`1.684750` -> `1.681140`), where `4e-4` had been flat since step
300. Whatever the right learning rate is, 1k steps at `2e-4` does not reach the
end of it, so `1e-4` is a live question rather than a formality.

*The tracker does not care.* Rotation (`0.79`-`0.86`), concentration (`0.45`
falling to `0.39`), and row-muting (`0.060` to `0.050`) are within noise of the
`4e-4` run at every checkpoint. Only the step scale changed:
`update_to_param_ratio` reads `1.44e-4` against `2.87e-4`, exactly half, for a
learning rate exactly halved. The learning rate moves how far each step goes; it
does not move what the basis tracker is doing.

**The stochastic-rounding hypothesis.** Stochastic rounding is delivering updates that round-to-
nearest used to discard, so the same nominal LR now moves the weights further
per step. More real progress per step is also more forgetting per step. If that
is what this is, the fix is not a mechanism change -- it is that the LR was
calibrated against an optimizer that was throwing part of every update away, and
it should come down.

**The confound, stated plainly.** The archive numbers come from a different
optimizer: refresh-based rather than Oja-tracked, with projected-clipping rails
that no longer exist, at rank 64 and 256 rather than 128. Reading a `0.10` source
gap across that many changes as "stochastic rounding did it" is not evidence, it
is a guess with a plausible mechanism.

**What would settle it.** An LR sweep with `stochastic_rounding` on and off,
same rank, same batch, same step count, on the current code. If the two curves
are the same shape with a horizontal offset, the effective-LR story is right and
the number to change is the LR. If the shapes differ, something else is going on.
Read the sweep at **step 300**, not step 1000: past 300 this lane is measuring
forgetting, not learning.

One more thing every archive comparison in this section carries: the lab harness
updated fallback parameters through `torch.optim._functional.adamw` on bf16
storage -- exactly the round-to-nearest loss `stochastic.py` documents -- so
every historical number in this lane was produced with a fallback set that was
barely training. Fixed now (`StochasticAdamW` everywhere), but it means the
archive is not a clean control for anything touching update magnitude.

---

## P6. The clip does nothing for the frame, and that is provable

`grad_clip_norm` is a single Frobenius magnitude cap applied once, upstream of
everything, and it is mandatory.

**It cannot protect the basis tracker at all.** The clip is a uniform rescale
`G -> cG`, and the Oja tangent is exactly invariant to that (P2, verified across
a 1000x range: `A` and `R` are both quadratic, the division by `mean(diag R)`
cancels the scale). So clipping changes the frame update by nothing whatsoever.
Its entire protective effect lands on the projected moment, which is linear in
`G`. The prose that used to be here -- that the clip protects "the frame and the
projected moment" -- was **wrong** and is withdrawn; it appears in `SPEC.md`
too and should be corrected there.

That reframes the question. It is not "do the two consumers want the same
protection" but:

- The moment is an accumulator, hurt by a large contribution. A norm cap bounds
  exactly that. Keep it.
- The frame is **already immune to magnitude bursts** by construction, and
  **fully exposed to directional ones**. A gradient of perfectly ordinary size
  pointing at a subspace one bad batch invented moves the frame exactly as far
  as a good one. Nothing in the design currently resists that.

So the frame's guard, if it needs one, is a persistence or agreement test rather
than a magnitude test -- which makes this the same question as P2's aim, not a
separate one. **P6 is now a sub-question of P2** and should be resolved with it.

**The trap remains as recorded.** "Already tried" contains the rotation-angle
clamp, which ate the large-`sigma` acquisition regime and made a good basis and
a garbage basis read identically. Any per-step bound on frame motion is adjacent
to it. The escape is that a persistence test bounds motion by *evidence* rather
than by a constant, which is a different object -- but that has to be argued,
not assumed.

**Also worth noting:** the clip fires on roughly 0.05% of tensors. Whatever it is
protecting against, it is rare, and its threshold has never been tested across
models.

---

## P12. Five device syncs per step, and nobody knows what they are

Measured while proving the diagnostics add nothing: CUDA sync debug mode reports
**25 synchronizing operations across 5 optimizer steps** on a two-matrix bf16
case, eager, with telemetry off. Turning telemetry on leaves it at 25 -- that
part is settled, and it was the question being asked at the time.

The 5-per-step baseline was not the question and is not explained. At least one
is structural: `torch.linalg.eigh` checks its convergence info on the host, once
per bucket per basis update, and the try-then-jitter path in P5 relies on that
being true. The other four are unaccounted for.

This matters more than a number usually would, because the Anima debugging
session was largely about removing exactly this class of stall from a training
loop, and two syncs found there were in the trainer rather than the optimizer.
Finding them is cheap -- `torch.cuda.set_sync_debug_mode("error")` raises with a
traceback at each one instead of warning -- but it is a separate session's work
and does not block P1.

**Goal.** Name all five. Then decide which are load-bearing and which are
accidents.

---

## Parked

**P6. Parameter eligibility and routing.** The exclusion rules -- lookup tables,
multiplicative gates -- and the residual-stream `side` policy live in the
ai-toolkit adapter, which exists for good reasons. A library-side
`RoutingPolicy` was prototyped and parked on branch
`study/rotation-clamp-eligibility`. Show a concrete alternative before arguing
for one.

**P7. CLOSED -- bf16 basis storage costs 0.6% of the moment's smear.**
`transport_spin` is the instrument this was waiting for, and unlike a loss delta
it measures a numerical property, so a synthetic gradient is admissible: the
question is what rounding does to a frame, not what a gradient distribution
does to a tracker.

The frame's own geodesic has `Q^T Q+ = V cos(theta) V^T`, exactly symmetric, so
an ideal step has zero spin. Measured, fp32 reads `7e-8` at window 1 and grows
*linearly* with the window -- genuine holonomy, coherently accumulating at
`9e-6` per update. bf16 reads `5.6e-4` at window 1, where exact arithmetic gives
zero, and `spin^2` is exactly linear in the window with **zero intercept**: a
pure random walk at `5.6e-4` rad per basis update, with no measurement floor.
So bf16 rounding does inject real in-span rotation, and in-span rotation is the
one motion that scrambles the projected moment's coordinates for no subspace
progress at all.

It is also immaterial. Over the moment's memory at `beta=0.9`, spin is `0.0018`
against `0.0194` of smear from the frame's legitimate motion; in quadrature that
raises total smear from 1.936% to 1.948%. The other channels are weaker on
inspection: the projection is `grad @ basis.mT` on a bf16 gradient, so an fp32
frame is rounded into the same bf16 matmul regardless, and the only surviving
path is the lift, ~0.2% relative on a direction Newton-Schulz has already
stripped the magnitude from. Against `+50%` of matrix optimizer state, this is
not a trade worth making. **Keep bf16.**

*Stochastic rounding on the basis write was tested and is worse*, by `sqrt(2)`
at every window. Stochastic rounding exists to stop a sub-ulp *update* from
vanishing; the frame has no sub-ulp update to lose, since the geodesic moves it
by a real angle every time. What bf16 costs the frame is variance rather than
bias, and SR trades a half-ulp bound for a full-ulp uniform draw -- exactly the
`sqrt(2)` observed. Recorded at the write site so it is not re-proposed.

**P8. `nan_to_num` placement.** Moved from the tangent's Gram onto the tangent
itself so it also covers the geodesic's separate read. Strictly safer, same
cost, but decided without a ruling. Revert it or keep it; folded into P5's
"does it ever fire" question either way.

**P9. `basis_update_interval != 1`.** The default is per-gradient motion. A
larger interval is a distinct estimator-variance and systems trade: phase-one
conditioning stays intact but fewer geodesics run. Never measured directly, and
a larger harmonic move is not automatically equivalent to several small ones.

**P10. Full-spectrum rotation vs rank-1 drift.** Today every tracked plane
rotates on every basis update. The lab archive flags full-spectrum rotation as a
way to spin the frame on the near-isotropic noise tail, and rank-1 drift as the
original heading. This is the same axis P2's `dare` metric measures -- if
concentration turns out to be low, full-spectrum rotation is the reason.

---

## Already tried. Do not repeat.

- **Normalizing the rotation angle by `sigma_max`.** Forces a constant top-angle
  per refresh, re-inflates the residual tail, prevents convergence. Reverted.
  *Re-derived independently and the arc was right.* Bare ortho is the same family
  -- a different normalization, the same discarded magnitude -- and it reproduces
  the same failure: motion that does not depend on fit, so it cannot converge.
  The arc's wording is worth keeping because it named the mechanism ("prevents
  convergence") rather than the recipe, which is what let it apply to a
  normalization it never saw.
- **Per-step Frobenius normalization of the gradient inside the basis update.**
  Made `sigma` a scale-free ratio, so it could not self-anneal and a good basis
  and a garbage basis read identically. Removed.
- **Clamping the rotation angle to a fixed ceiling.** Not the same as
  normalizing, but eats the large-`sigma` acquisition regime the hot schedule
  exists for, and a clamped telemetry read reproduces the "good basis and
  garbage basis look the same" failure above. Reverted.
- **A magnitude cap on the tangent, and a full-rank zero-tangent branch.** Both
  were added to a code path that no default configuration reached. Deleted.
- **Overlap reprojection of the moment through frame motion.** Charges a cosine
  tax on rotated directions and hands Aurora weakened coordinates its polar map
  then amplifies. Identity transport in moving-frame coordinates instead.


---

## Run log

Everything below was measured on `LiquidAI/LFM2.5-350M-Base`,
broad-no-embeddings, bs16 x seq1024, `release_matrix_grads`, compiled,
`StochasticAdamW` fallback at `1e-4`, 1k steps unless noted. All on the released
optimizer, which is the point -- see the note on the lab fork below.

| run | rank | LR | conditioning | target | source | s/step |
|---|---:|---:|---|---:|---:|---:|
| `usuitrack-release-baseline-200` (200 steps) | 128 | `4e-4` | both | 1.759362 | 3.076731 | 0.7654 |
| `usuitrack-release-baseline-1k` | 128 | `4e-4` | both | 1.741403 | 3.191206 | 0.7489 |
| `usuitrack-release-lr2e-4-1k` | 128 | `2e-4` | both | **1.681140** | **3.094790** | 0.7483 |
| `usuitrack-p1-armB-rawtangent-lr2e-4-1k` | 128 | `2e-4` | moment | 1.686383 | 3.113945 | 0.7533 |
| `usuitrack-p1-armD-noadafactor-lr2e-4-1k` | 128 | `2e-4` | none | 1.684664 | 3.115753 | 0.7299 |

Initial losses are `2.297068` target and `2.917333` source throughout.

**Anima**, full finetune, 2B DiT, rank 64, bs4 x 768px, lr `1e-5`,
warmup-stable-cosine, 2304 steps, no Adafactor: wandb `7puon3ub`, 2h30m, clean.
This lane has no meaningful loss -- flow-matching loss on a diffusion finetune
does not rank checkpoints -- so the verdict is the samples, reviewed by the user:
better than the Adafactor-era runs, and notably more directional. The earlier
runs wandered between sample rounds; this one holds a heading. Telemetry:
`nonfinite_grads` 0, no `eigh` retries, `rotation_rad_sum` at the
`MIN_GEODESIC_STEPSIZE` floor within roughly three logging intervals,
`tangent_concentration` ~0.865 with no trend -- far above LFM's ~0.39, which is a
fact about this model's gradients rather than about the tracker, and a warning
that any step rule built on concentration has to survive both operating points.

**Anima under the derived live floor**, same config as `7puon3ub` (rank 64, bs4
x 768px, `1e-5`, 2304 steps): `anima_usuitrack_8_live_floor`, wandb `9elbwps6`,
completed, final model saved. This is the run the live-floor fix was made for --
the two attempts immediately before it died in `eigh` inside twenty steps.

Full-run telemetry, 200-step bins:

| bin | `turn_fraction` | `live_fraction` | participation | concentration | `curve` |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.5065 | 0.4432 | 0.02341 | 0.8496 | 0.6063 |
| 600 | 0.3996 | 0.4701 | 0.02377 | 0.8449 | 0.6541 |
| 1200 | 0.3712 | 0.4864 | 0.02392 | 0.8424 | 0.6666 |
| 1800 | 0.3539 | 0.4817 | 0.02391 | 0.8430 | 0.6792 |
| 2200 | 0.3409 | 0.4906 | 0.02428 | 0.8378 | 0.6855 |

**The aim does not converge over 2304 steps.** Participation moves `0.0234` to
`0.0243` and concentration `0.850` to `0.838` -- an effective rank of `1.50`
rising to `1.55` out of 64, flat across three epochs. Whatever sets this model's
aim rank is a property of the gradients at this batch size and this data, not
something training improves. Compare the bs1 column of the LFM ladder above:
concentration `0.829`, effective rank `1.6`. **Anima at bs4 sits where LFM sat
at bs1.**

The frame, by contrast, converges normally and lands where LFM lands:
`transport_speed` falls `0.0028` to `0.0018` while `transport_curve` rises
`0.61` to `0.69` -- travelling, then settling onto a fixed point, which is the
signature SPEC describes. LFM's 300-step baseline reads `0.00175` and `0.683`.
The two models differ in the aim, not in the frame's motion.

`turn_fraction` decays `0.507` to `0.341` and is still falling slowly at the
end. LFM reaches `0.217` by step 300. Anima's tracker holds a turn scale
roughly half again as large, indefinitely, because consecutive aims keep
disagreeing -- the sustained-motion equilibrium of a subspace that is genuinely
being resampled every step rather than converging.

Verdict is the samples, reviewed by the user: **the first non-destructive run.**
Fingers and anatomy survive noisy batches, and it reads as one trajectory rather
than a walk between sample rounds. Loss is not comparable and is not reported --
flow-matching loss at bs4 moved `0.143` (first 400 steps) to `0.154` (last 400),
which is noise on this lane.

**Two cautions about anything older than these.**

*The lab was measuring a fork.* Until this session the harness imported its own
copy of the optimizer, which had drifted: no rank cap, no stochastic rounding. It
now imports the release package and checks that it did.

*There is no run of the lab's final code.* The newest wandb directory,
`dbwc1x5o` (`oja-r128-lr5e4-fallback1e4-1k`, `1.673074 / 3.056915`), started at
14:31 on 2026-07-16; the lab's last two commits landed at 14:53 and 22:31 the
same day, the second being `7a9283e "Simplify UsuiTrack tracking path"` -- 1456
deletions collapsing tracking to the harmonic geodesic. All 197 local run
directories were checked and not one logs `basis_update_interval`, so every
stored run predates the current argument set. `dbwc1x5o` is the newest directory,
not the newest code, and no comparison against it is like-for-like.

*The fallback was broken in every archived run.* The lab updated fallback
parameters through `torch.optim._functional.adamw` on bf16 storage -- exactly the
round-to-nearest loss `usuitrack/stochastic.py` documents -- so every historical
number carries a fallback set that was barely training.
