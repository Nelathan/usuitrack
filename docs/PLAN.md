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
variable apart:

| conditioning | target | source | s/step | matrix state |
|---|---:|---:|---:|---:|
| both consumers (was default) | 1.681140 | 3.094790 | 0.7483 | 97.4 MB |
| moment only, raw aim | 1.686383 | 3.113945 | 0.7533 | 97.4 MB |
| neither | 1.684664 | 3.115753 | **0.7299** | **95.9 MB** |

`0.0035` of target loss for 2.5% of walltime and 1.5% of state. Note that
removing it from the moment as well was *better* than keeping it there once the
aim read raw -- there was no configuration in which conditioning the moment
alone paid.

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

### Where the tracker actually stands, and why it is not deletable

The step-size sweep (constant `eta`, no moment, seeded, 1k steps, `eta` from
frozen to `0.3`) settled several things and one of them is a correction.

**Tracking does real geometric work.** A frozen frame loses essentially all its
alignment within 200 steps -- `sigma` 50 -> 91 -> 98 -> **101, saturated** -- while
tracking at `0.01` holds 74 and is still improving at step 1000. That is a 26%
better residual, sustained.

**No step size recovers the quality of the initial fit.** EIGH hands the tracker
`sigma ~ 50`. Every arm, at every step size across a 100x range, rises to 80+ and
then settles at best to 74. `0.003` and `0.01` are indistinguishable; above `0.01`
alignment degrades monotonically and motion efficiency collapses (net displacement
per unit turning falls 7.5x from `eta=0.01` to `0.3`). **The aim, not the step
size, is what limits the tracker** -- a bad formula run slowly is still a bad
formula.

**Loss barely notices any of it.** The entire 100x sweep spans `0.0057` of target
loss. Frozen costs only `0.0038` against the best tracked arm -- the same
magnitude as the Adafactor effect P1 deleted. Source is flat to slightly better
frozen.

**That last number does not argue for deleting tracking, and reading it that way
was an error of criterion.** A frozen basis confines every weight update to one
rank-128 subspace forever; the rest of the parameter space is unreachable. For
continual pretraining -- the product target -- that is disqualifying whatever a
1k-step loss says. Loss at this horizon cannot see a capability constraint.
Periodic EIGH refitting is the fallback if Oja tracking cannot be fixed.

**What it does say is that update quality is robust to subspace quality.** Even a
badly aligned frame produces good updates, which is a real and useful property --
but it means loss is a blunt instrument for tracker work, and tracker changes
must be judged on `sigma`, lag, and capture rather than on loss deltas that sit
near the noise floor.

**Open, and the first thing to run when experiments resume: random basis
initialization.** If Oja from a random frame converges to the same `sigma ~ 74`
it settles at from an EIGH start, then 74 is Oja's own equilibrium and EIGH
initialization is simply a better place than its aim can hold. If random
converges somewhere worse, the aim cannot acquire at all. Either answer localizes
the defect.

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
turn hard and when to settle. Hard constraint unchanged: no normalization that
forces a constant angle per refresh. A constant *multiplier* is not that and
remains allowed; a hot start measurably warmed the basis sooner and improved
early capture. Returning to a constant is not the goal.

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
| `MIN_BASIS_UPDATE_STEP = 0.01` | step-size floor | P2; the load-bearing one |
| rank cap `min(m,n)/2` | `effective_rank` | settled by structure and bottleneck stability, but the *fraction* is still a choice |
| `grad_clip_norm = 1.0` | raw clip | now mandatory; the threshold itself is untested across models |
| `beta = 0.95`, `eps = 1e-8` | moment | inherited |
| `AURORA_PP_ITERATIONS = 1`, `AURORA_PP_BETA = 0.5` | direction map | inherited from the method |
| `1e-12` floors, `1e-7` sigma threshold | numerical | probably fine, unaudited |

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

**P7. bf16 basis storage.** The geodesic runs in fp32; the moved frame is
written back in the parameter dtype, so a bf16 model rounds the frame every
basis update. Only synthetic evidence exists, which is inadmissible here.
Whatever P4 settles is the instrument for looking at it.

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
`MIN_BASIS_UPDATE_STEP` floor within roughly three logging intervals,
`tangent_concentration` ~0.865 with no trend -- far above LFM's ~0.39, which is a
fact about this model's gradients rather than about the tracker, and a warning
that any step rule built on concentration has to survive both operating points.

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
