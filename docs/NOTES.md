# Notes

Prose cut from `docs/SPEC.md` at the 2026-10 release cleanup: history, the
experiments behind a rule, and lab-only reads. Kept for the archive; SPEC is
the current design and does not depend on anything here. Verbatim: present
tense is the time of writing, and "above" points into SPEC as of `d76ea99`.

## Anima's bottleneck projections and the rank cap

Observed on Anima's `(2048, 64)` input and output projections: at rank 64 on
their 64-wide data side -- full rank, 100% occupancy -- training errored. Two
things were wrong at once and both fixes were real. The `side` hint moved them
onto the 2048-wide residual stream, which is where their basis belongs and which
brought occupancy from 25% into line with every other module at ~3%. And the
rank still has to be capped, because those modules are bottlenecks: too much
change in them destabilized the model regardless of which side was tracked.

## Side: residual-facing against auto at the edges

The difference concentrates at the edges of the network, where the data side is
narrow. Shape-only `auto` tracked the 68- and 64-wide data sides of a model's
input and output projections and produced its two worst-conditioned bases
(orthonormality error 1.1e-03 and 1.0e-03 against ~5e-04 elsewhere); the hint
spends the same rank budget inside the 2048-wide stream, where the complement
the Oja tangent needs is ample.

## The constant turn: what was tried

The time axis was tried and did not deliver either. A gain scaled by how well
consecutive top-`k` aims agreed, divided by an attainable ceiling
`(\sum\lambda)^2/(\sum\lambda^2 k)`, shipped for several runs and is now
removed on two grounds: the ceiling is a magnitude functional gating a step whose
whole purpose is to be magnitude-free, and on Anima it annealed the turn by 1.21x
over 2100 steps while on LFM it looked like it worked. The anchor-free two-lag
form that would replace the ceiling is separately falsified -- the aim's
persistence *shape* is stationary while its level is not, so the ratio holds near
0.81 and anneals nothing (`ARCHIVE.md` on `lab`).

The shape of the idea remains right: a converged frame orbits on batch noise and
should stop turning. What does not exist is a sensor that can tell arrival from
batch noise, and at bs4 a spreading aim is what noise alone produces. Until one
exists the turn is constant, which at least makes frame motion a single term that
`transport_speed` and `transport_curve` read directly. `eta` is then the only
handle, and scheduling it on the prior that aiming gets harder as the surface
smooths is a knob, not a sensor.

There is no cold start. It existed because the gain had no history on the first
update and because a full turn at an `eta` chosen for a scale near 0.05 failed
`eigh`; masking dead planes removed that failure, whose cause was rounding
artifacts promoted to unit-norm directions by `1/sigma`, not step size.

## Raw sigma as the step

**Raw `sigma` is a contrast ratio, not a magnitude, and it is no longer the
step.** Both `A` and `R` are quadratic in the gradient, so the division by
`mean(diag R)` cancels gradient scale exactly: rescaling `G` by any constant
leaves `Delta` and every `sigma_i` bit-identical (verified across a 1000x range).
What `sigma` measures is out-of-frame coupling against mean in-frame energy per
plane. It is scale-free in the gradient, which is half of why there is no clip, but
*not* rank-free -- a nuclear norm over `r` planes, growing roughly linearly in
`r` (~37.6 at rank 64 against ~80 at rank 128 on the same problem). Driving the
frame with it directly was the released rule until the polar transform above
replaced it, and the reason is that its self-annealing is an acquisition transient only:
`sigma` falls sharply over roughly twenty steps, then flattens and declines a few
percent across the remaining ~900. Rescaling it by a constant (`sigma_max`, the
tangent's Frobenius norm) cannot fix that, because a ratio cannot shrink as its
numerator does; capping the angle fails at the other end, costing acquisition.
Both were tried and reverted. The constant turn above sidesteps the question:
`sigma` decides nothing, so its acquisition transient cannot leak into the
frame's motion.

## Diagnostics: the former tiers and the transport reads

Optional, off by default, and structurally incapable of costing anything when
off: every accumulation site is guarded by one attribute read. Set
`diagnostics` to `"core"` or `"full"` and drain with `pop_diagnostics()`, which
returns a plain dict of floats -- no wandb, no trainer, no assumptions about the
caller. The two live tiers are split by cost, not by usefulness: `"core"` is
every read derivable from tensors the step already formed, while `"full"` adds
the three that need a frame snapshot over `diagnostics_lag_matrices` sampled
matrices.

Measurements accumulate on-device every step; the single device-to-host read
happens inside `pop_diagnostics()`. A drained value is therefore the mean over
the interval since the last drain, which is why the intended cadence is the
caller's logging cadence rather than every step. `nonfinite_grads` is reported as
a total over the interval rather than a mean, because a rare event averaged
against a long quiet interval reads as zero.

No `"core"` read makes its own pass over a full matrix. Each is a reduction of
a rank-sized tensor or of a norm the step already took: the update's norm is
read in frame coordinates, `||U|| = ||O|| ||W||`, since the lift is an
isometry, and non-finite gradients are counted from the raw norm. With
telemetry off the frame-motion read is skipped too.

A read sampled once per matrix can also be published per role class, as
`<read>/<role>`: the caller labels matrices with `diagnostic_roles` (a
`{param: role}` map, since only the caller knows names) and names the reads to
break out in `diagnostic_role_reads`, from `ROLE_READS`. Nothing is broken out by
default; each selected read adds one key per role.

| key | what it measures |
|---|---|
| `transport_speed` | chordal distance the subspace moved in one geodesic, per plane, measured from the frames before and after: `||Q_now - Q_old (Q_old^T Q_now)||_F / sqrt(r)`. Every motion metric below shares this unit, so they can be divided by one another |
| `tangent_concentration` | `lambda_max / sum_i lambda_i` of the tangent Gram, in `[1/r, 1]`: the leading direction's share of the aim |
| `tangent_live_fraction` | the fraction of planes whose eigenvalue clears the Gram's numerical noise floor, `r * eps * lambda_max`: how many planes the decomposition can resolve at all. Below `1.0` the aim is rank-collapsed against the rank it was given, and the planes below the floor are held still rather than turned on rounding error |
| `projected_grad_norm` | norm of the sanitized gradient inside the held frame |
| `raw_grad_norm` | norm of the gradient as it arrived, before sanitizing and projection; non-finite when the gradient was |
| `grad_capture` | `projected_grad_norm / raw_grad_norm`: the share of the gradient's magnitude the frame catches. The only one of the three that reads the basis rather than the model -- the projected norm alone falls with a leaner rank even when every kept plane catches as much as before |
| `moment_persistence` | how much coherent signal the average holds once the floor of an independent stream is subtracted. An EMA of independent constant-norm directions already has norm `w = sqrt((1-beta)/(1+beta))` from incomplete cancellation, so with `||M||` as `rho` (each blended gradient is a unit), this reports `(rho^2 - w^2)/(1 - w^2)`: 0 is a white stream, 1 a direction held throughout, negative anti-correlated. The raw ratio is recoverable as `p(1-w^2) + w^2` |
| `grad_moment_cosine` | this step's unit gradient against the moment as it stood *before* the blend, reconstructed so the reading is not self-referential. Read it as an overshoot meter: positive means the step under-travels and the next gradient still points where the last one did, negative means it overshoots and the gradient has flipped behind it. Zero is critically damped, which makes it a learning-rate read -- the LR that zeroes it beat a 2.2x hotter one on both eval heads |
| `relative_step` | the whole step's `||dW||_F` over every matrix against their current `||W||_F`: the final ratio, `lr` itself whenever every matrix stepped (step 8), and the read that compares across models. The update before rounding, so the bf16 stochastic-rounding walk is not in it |
| `nonfinite_grads` | matrix gradients that arrived non-finite and were sanitized, counted from `raw_grad_norm` |
| `transport_lag` | net distance a sampled frame covered over `diagnostics_lag_interval` basis updates, same unit as the speed. Also the projected moment's smear: set the interval to `1/(1-beta)` and it reads how far the moment's own history has drifted from the coordinates it was accumulated in |
| `transport_curve` | `1 - lag / path`: the fraction of that window's travel which cancelled. Zero is a straight drift, approaching one is a frame churning in place |
| `transport_spin` | `||skew(Q_old^T Q_now)||_F / sqrt(r)`: rotation of the frame's columns *within* their own span over the window. Moves the subspace not at all: the path's holonomy plus storage rounding. The moment is turned with the stored frame every step, so spin does not scramble it |
| `transport_correction` | `||skew(R_s)||_F / sqrt(r)`, same unit as spin: the in-span rotation the moment is turned by on this step, from the stored frame's overlap (step 6's correction). Zero when the frame is stored exactly |

`transport_lag`, `transport_curve` and `transport_spin` need the frame snapshot
and appear only under `diagnostics = "full"`; the speed and the correction do
not, since both frames are already in hand when the geodesic runs. Read the four motion reads as a set. High speed with
low curve is a frame travelling, and it will slow as the aim converges. High
curve with low speed is a frame sitting on its fixed point. High speed *and*
high curve is churn: the tracker working hard, going nowhere, and integrating
batch noise into the frame while it does. Low speed with low curve is ambiguous
between settled and starved, and spin separates them -- a settled frame is still
in every sense, a spinning one is still rotating its own coordinates. A single
Grassmann geodesic along a horizontal tangent has `Q_old^T Q_now = V cos(theta)
V^T`, exactly symmetric, so spin is zero for one ideal step and what accumulates
over a window is holonomy plus retraction and rounding error.

`transport_speed` is measured from the written frames, not from the eigenvalues
the geodesic was handed. Those agree only while nothing stands between the aim
and the frame, and they are not the same quantity the moment anything transforms
the geodesic -- an orthogonalized tangent turns every live plane by `eta`
whatever `sigma` said. `tangent_concentration` does come free from those
eigenvalues. Read speed and concentration together: the same speed is a confident drift when
concentration is high and a frame spinning on its noise tail when it is low.

The tangent Gram is decomposed bare. There is no jitter and no retry: a failing
`eigh` fails, because its result steers the frame and a silently rescued
decomposition is worse than a stopped run. The initial fit in `_side_gram_eigh`
keeps its try-then-jitter fallback -- different matrix, once per parameter rather
than once per step.
