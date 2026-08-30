# Basis-tracking aim and step-size redesign: a solution tree

This is a design exploration, not a specification. It does not pick a winner
by fiat; it lays out the space, prices each branch against the project's
stated values, and ends with a recommendation the team can accept, reject,
or counter-propose against. Nothing here is committed to code.

**This is a revision, not a first draft, and it changes conclusions.** The
first draft of this document treated the step-size schedule `eta_t =
max(0.01, 1/t)` as the problem. Measurement that existed at the time, but
that this document had not yet been shown, moved the question upstream: the
step size is a real but secondary problem, and the primary one is the aim.
`docs/PLAN.md`'s P2 and P6 were rewritten around that measurement and are
the ground truth this revision works from. Where the first draft's branches
die on this evidence, this document says so by name, not by quiet omission.

## 1. The problem, stated precisely

The step-size rule is still `eta_t = max(0.01, 1/t)`, `t` counting basis
updates, and it is still a pure function of a step counter with no read of
the geometry. That much of the original framing survives. What does not
survive is the story about *why* the tracker plateaus, and what a fix
should target.

### What is established, with the measurement that settles each point

1. **`sigma` is a contrast ratio, already scale-free in the gradient.** The
   Oja action `A = G^T G Q` (right side) and the Rayleigh matrix `R =
   sym(Q^T A)` are both quadratic in `G`; dividing the tangent by
   `mean(diag R)` cancels that scale exactly. Verified across a 1000x
   gradient rescale: `sigma` identical to four decimals. **The first draft
   asserted `grad_clip_norm=1.0` gives `sigma` a new absolute scale ceiling
   and treated that as new leverage for a scale-free rule. That was wrong,
   and it is withdrawn here.** The clip does not touch the tracker's scale
   at all — see point 6 and Branch removal below.
2. **`sigma` is not rank-free.** It is a nuclear norm over `r` planes and
   grows roughly linearly with rank: `~37.6` at rank 64 against `~80` at
   rank 128. Rank is the entire remaining scale-freeness problem, and it is
   a much smaller, much more tractable problem than "does not mean the same
   thing across parameter size and rank," which is how the first draft
   stated it.
3. **`sigma` does not keep annealing.** Backed out of a real 1k-step run as
   `rotation_rad_sum / eta`: `170.6` at step 10, `92.0` at step 20, `~85` at
   step 50, then `86 -> 81 -> 79` from step 100 to step 1000. It anneals for
   roughly the first 20 basis updates and is flat for the remaining 980.
   **This is measured, real-gradient evidence, and it falsifies the first
   draft's Branch 1** ("hold `eta` constant and let `sigma`'s own decay
   carry self-annealing") **on data that already existed before that branch
   was written.** A constant `eta` times a flat `sigma` of `~80` is a
   constant nonzero rotation forever — the opposite of settling. §6 records
   this death explicitly, with the reasoning error named.
4. **Rank 128 is not a bottleneck.** Capture — the fraction of gradient
   energy the frame retains — reads `0.64`-`0.74` across the archive at rank
   128 (Oja lane `~0.66`, projected-AdamW lane `0.74`). Two thirds of the
   gradient's energy is already inside the frame. Every number here was
   measured on the *conditioned* (Adafactor) gradient — Adafactor flattens
   spectra, so raw-gradient capture is plausibly higher and is currently
   unmeasured — but the direction is unambiguous: "`sigma` plateaus because
   most of the gradient's energy is irreducibly outside the frame" is wrong
   and withdrawn.
5. **The tracker sits at an equilibrium, not at convergence and not
   starvation.** Stable contrast (`sigma ~80`), stable capture (`~0.66`),
   concentration `~0.4` against an isotropic floor of `1/128`. The residual
   the tracker keeps rotating toward is *structured*, not noise, and it is
   not shrinking. **Whether that equilibrium is the right one is the open
   question** — not "has the tracker converged," which was the first
   draft's framing, but "is the thing the tracker keeps chasing worth
   chasing."

### The mathematical picture these five facts assemble into

Point 5's language — "equilibrium, not convergence" — is not loose; it has
a precise reading once set against P1's fixed-point property, and stating
that reading plainly is the single most load-bearing move in this revision.

P1 proved a fixed point exists **for the deterministic map**: a frame `Q`
fitted on a *specific* gradient `G` is exactly the top-`r` eigenspace of
`G^T G`, so the horizontal residual on that exact `G` is zero. But no two
consecutive basis updates see the same `G` — each is one ~16k-token batch's
realization of a noisy process, not the population gradient. The fixed
point is a property of `S = E[G^T G]`, the *population* second moment;
what the tracker actually receives, every step, is one noisy sample `G_t`
of that population, and the Oja action it acts on is `S_t = G_t^T G_t`, not
`S`. Even a frame sitting exactly on `S`'s top eigenspace will see a
nonzero tangent on almost every individual `G_t`, because `G_t^T G_t` is
not `S` — it differs by sampling noise that does not vanish just because
the frame is correctly placed. The tracker therefore does not converge onto
the fixed point; it **orbits it, at a radius set by batch noise**. That
radius is exactly what `sigma ~80` (rank 128) is measuring once the
transient 20-step acquisition phase has passed: not "distance left to
travel," but "the standing noise floor of a single-batch covariance
estimate, as seen through a fitted frame." This reading is consistent with
every one of the five facts above without needing any of them to be an
anomaly: capture is stable and high because the frame really has found `S`'s
top eigenspace; concentration is stable and structured because the orbit
itself has shape (the noise is not isotropic — real gradients have
correlated batch-to-batch structure, not iid noise); and `sigma` is flat
because an orbit radius, unlike a distance-to-target, does not shrink on its
own once the target has been reached.

**This is this document's own synthesis, not a measured fact** — it is the
most economical explanation of all five points at once, and it is the
premise the whole solution tree below is organized around, but it should be
read as a strong hypothesis, falsifiable by Branch C below, not as
something already nailed down.

### What this hands to the aim question

If the tracker is orbiting a correctly-placed fixed point rather than
failing to reach one, no step-size rule reading only `(sigma,
concentration)` from a *single* batch's tangent can shrink that orbit —
those quantities are computed from exactly the noisy per-batch sample that
produces the orbit's radius in the first place. A step-size fix can change
how fast the frame moves around the orbit; it cannot make the orbit
smaller, because the orbit's size is set by what one batch's covariance
tells you, and no amount of clever weighting of that single measurement
changes how much information one batch carries. **Shrinking the orbit
requires more information per aim than one batch supplies — which is a
statement about the aim, not the step size.** That is the reframing this
document works from for the rest of its length.

### `grad_clip_norm` and the frame: a provable no-op

One more correction inherited from P6's rewrite, because it removes a
branch of reasoning the first draft leaned on. The clip is a uniform
rescale `G -> cG` for whatever scalar `c` the norm cap chooses that step.
Since the tangent construction is exactly scale-invariant (point 1 above),
**the clip changes the frame update by nothing whatsoever, for any batch it
fires on.** Its entire protective effect lands on the projected moment,
which is linear in `G` and therefore not scale-invariant. The frame is
*already* immune to magnitude bursts, by construction, with no clip needed
— and *fully exposed* to directional ones: an ordinarily-sized gradient
pointing at a subspace one bad batch invented moves the frame exactly as
far as a well-behaved one would. Nothing in the current design resists
that. This reframes P6 from "do the two consumers want the same guard" to
"the frame's only real guard is the aim itself" — which folds P6 into P2,
not beside it, and is why this document does not treat them as separate
questions below.

## 2. The new central thesis: it is the aim, not primarily the step size

**Oja aims at this batch's covariance.** It is magnitude-weighted and
burst-sensitive by construction — the tangent is built from `G_t^T G_t`
for whichever single batch arrived this step, with no mechanism that asks
whether this batch's direction has ever shown up before. Three things turn
that from an imperfection into a real cost, rather than a rounding error:

- **Aurora/Newton-Schulz orthogonalizes whatever is in the subspace.** A
  burst direction that gets *into* the frame is not merely present at its
  raw, small weight — it is amplified to full orthonormal strength by the
  polar map before it reaches the parameter update. The cost of admitting
  noise into the frame is asymmetric: a noisy direction that enters costs
  more, downstream, than a noisy direction that stays out.
- **The batch is small relative to the design's own reference point.** A
  training batch here is on the order of ~16k tokens, against the ~256k a
  Muon-lineage pretraining run typically uses. A single-batch covariance is
  a genuinely noisy quantity to aim by, and the entire design premise —
  keep the tracked subspace narrow, rank-capped well below `min(m,n)` — is
  itself a bet that the batches are noisy enough that a small, careful
  subspace beats a wide, credulous one. Aiming that narrow subspace at raw
  single-batch noise cuts against the reason the subspace is narrow.
- **Adafactor's SNR weighting was a persistence proxy, and P1 deleted it
  without replacement.** Adafactor downweighted high-variance directions —
  an approximate, expensive way of asking "has this direction shown up
  consistently." P1's numbers say the loss barely noticed its removal; they
  do not say persistence-weighting itself is worthless, only that
  Adafactor's specific, expensive implementation of it was not earning its
  keep. The honest statement is that the aim lost its only persistence
  mechanism and nothing took its place.

**Persistence-weighted aiming was orphaned, not rejected.** This is the
single most important correction this revision makes to the git
archaeology the first draft already did well. §4 below keeps that
archaeology largely intact and adds the piece it was missing: the
accumulation lattice the arc built (C1, "sum-of-tangents," measured and
recorded as *removing the noise wall*) was killed on 2026-07-11 when
position control (eigh aim) won the arc's centerpiece argument, and
position control was itself replaced by direct Oja tracking three days
later on a measured head-to-head. **Persistence was never tested against
the mechanism that ultimately shipped.** It lost to an intermediate design
that itself lost. That is a dropped thread, not a closed question, and it
is the thread this document picks back up in Branch A below.

## 3. The architectural constraint that resolves the self-preservation worry

Before building any candidate that reads the moment for anything, one
worry has to be named and settled, because it would otherwise contaminate
every branch below: doesn't weighting the aim by "what has persisted"
just entrench whatever the frame already captured, and starve anything new?

**The projected moment is built entirely from projections onto the
current basis.** `Z_t = G_t Q_t`, and `M` is an EMA of exactly that. A
direction living outside `span(Q)` has never had the chance to earn moment
mass — it is invisible to `M` by construction, not because it is
unimportant. So weighting a decision by moment strength carries a built-in
incumbency bias: it can only ever speak in favor of what the frame already
holds. **Only the Oja residual — the horizontal, `(I - QQ^T)(\cdot)` part of
the tangent — looks outward**, because it is the one quantity in this whole
design that is computed from the *full*, unprojected `[m,n]` gradient
before anything is thrown away.

The resolution: **inward signals may rank or evict; only the residual may
nominate.** Moment magnitude, capture, gradient-moment alignment — all of
these answer "how well is the frame doing at what it already tracks," and
are legitimate inputs to *how hard to push* an already-outward-pointing
signal, or to *which existing plane looks weakest*. None of them may decide
*what direction is worth entering the frame in the first place*; that
decision structurally requires a full-rank read, and the only full-rank
read this design keeps is the residual itself. Conflating the two — using
an inward signal to gate what enters — is precisely the self-preserving
loop the worry above is pointing at, and every candidate below is checked
against this line explicitly.

**Corollary: capture is not an objective.** It measures fit to the current
gradient, and a frame that has stopped moving on a gradient that has itself
stopped changing scores well on capture while having learned nothing new
about the model. Capture (and, by the same logic, agreement — see Branch
B1) explain a result after the fact. They do not rank designs. Loss does.
Every falsification experiment in §5 that claims to validate a design
change, not merely instrument it, has to eventually clear that bar.

## 4. Values and constraints, updated against this revision

In priority order, restated as binding tests, with the corrections this
revision requires:

1. **VRAM-first.** State today is `[d,r]` basis + `[?,r]` projected moment
   per matrix. A branch adding *permanent, universal* per-matrix state must
   be justified in bytes against that baseline. A branch adding *transient*
   state, or state applied only to a *sampled subset* of matrices, is
   cheaper and must say so explicitly with its actual scope — see the
   revised treatment of `basis_lag_angle` in Branch C, which the coordinator
   has now approved as a sampled, opt-in diagnostic and which changes its
   VRAM arithmetic entirely relative to the first draft's universal framing.
2. **Self-annealing must survive, correctly restated.** "Big residual, big
   step; well-fit, small step, settles" was written assuming residual shrinks
   to zero on convergence. §1's synthesis says real per-batch residual does
   not shrink to zero — it settles at a nonzero equilibrium. The corrected
   statement: a step rule must not apply a *fixed* per-refresh angle
   regardless of what the residual says, at any grain (whole-frame or
   per-plane) — it must remain a function of the residual's own size and
   shape. It is not required to, and on the evidence in §1 cannot, drive
   the residual itself to zero.
3. **Scale-free**, now narrowed correctly to rank alone, per §1 point 2.
   Parameter size and `grad_clip_norm` are not part of this problem;
   withdrawing that framing from the first draft is itself part of this
   value's correct statement now.
4. **No device syncs, no per-step host reads**, unchanged.
5. **Maintainable and honest.** Fewer constants; survivors derived or
   scale-free; no speculative flexibility.
6. **Cheap: reuse what's already computed.** This value gets sharper
   teeth in this revision than in the first draft. The moment `M` is
   *already stored, already updated every step, already free* — any branch
   that can build persistence out of `M` rather than a new buffer clears
   this bar by construction, which is exactly why Branch A1 below is this
   document's leading candidate rather than a buffer-based revival of C1.

## 5. Hard constraints carried over, and what changed about them

The "already tried" list from `PLAN.md` is unchanged and still binding —
quoted in the first draft, not repeated in full here, with one exception
worth restating because a reader might otherwise think it now conflicts
with something new below: **normalizing the rotation angle by `sigma_max`,
Frobenius-normalizing the gradient inside the update, and clamping the
rotation angle to a fixed ceiling all remain banned**, for the reasons
already on record (they force a constant angle regardless of residual size
or shape, and each was measured to prevent settling or to make a good basis
and a garbage basis read identically).

The `p`-flattened per-plane rule `eta * sigma_max^(1-p) * sigma_i^p` remains
a trap, not a free lunch, at any `p < 1`, for the reason given in the first
draft: it interpolates continuously toward the banned whole-frame
normalization, applied per-plane.

**One thing changed, and it resolves an apparent tension the reader would
otherwise trip over.** The coordinator has confirmed **a constant
*multiplier* on the step-size schedule is explicitly allowed** — this is
not the same object as normalizing the angle to a constant. A multiplier
scales `eta` by a fixed factor while `sigma` still varies per basis update
and per plane; the rotation `eta * sigma_i` still tracks the residual's own
size, just at a different overall rate. A hot start using exactly this —
scaling the early, transient part of the schedule — measurably warmed the
basis sooner and improved early capture. But §1 point 3 already shows why
this cannot be read as a fix for the plateau: `sigma` is flat for 980 of
1000 steps regardless of what multiplier is chosen, so a multiplier only
ever changes how fast the ~20-step transient acquisition phase runs. It
does not touch the steady-state orbit at all. **Returning to a constant
multiplier is not the goal**, and this document does not recommend one as
a substitute for addressing the aim.

## 6. The two threads the arc orphaned, not just one

This section keeps the previous revision's git archaeology intact — it
answers a real, still-open question about whether Oja tracking is even the
right control-loop structure — and adds the piece that connects it to this
revision's central thesis: **the arc orphaned two things in the same
sequence of commits, not one, and only one of them (position control) was
named as the reason for the reversal at the time.**

### The position-control vs velocity-control argument, and its reversal

The arc's centerpiece (2026-07-11, "Position control vs velocity control")
argued Oja tangent-tracking is open-loop velocity control: "each boundary
applies a displacement derived from a gradient — a velocity command. Noise
in velocity *integrates*: the basis random-walks with no restoring force
when a step was wrong. The only defense is tiny steps," which is why the
tangent step size had landed at milliradians and still lost to drift.
Directly refitting the frame by `eigh` on a fresh side-Gram each refresh is
closed-loop position control instead: "each boundary names a *position* ...
and the geodesic contracts toward it by factor α. Errors decay
geometrically instead of integrating ... Positions average to their mean;
velocities average to nothing." The measured contrast: eigh aim could
rotate 22.5° planes every 10 steps without thrashing, where tangent
tracking crawled at milliradians and still lost. That argument shipped as
the "approved design cut": `grassmann_aim="eigh"`, `grassmann_rotate_rank=
None` (all planes), `grassmann_step_size=0.25`, interval `10` — and the
tangent accumulator, C1, was deleted in the same commit as dead weight once
position control replaced it as the signal being smoothed.

Reconstructed from `/home/djg/code/optimizers` git history: `a5f3995`
(2026-07-11, 22:21) is the eigh-aim promotion above, validated on one
500-step open-loop run, with the step-schedule question explicitly left
open. `3f506be` (2026-07-14, 22:58), three days later, is a purpose-built
head-to-head — boundary EIGH, live-0.25-EIGH, a separate Oja-target
estimator, and direct one-state Oja on next-interval predictive capture
(`0.4588` / `0.5077` / `0.5299` / `0.5274` respectively — direct Oja and the
separate-Oja estimator essentially tied for best, both above EIGH), and a
matched 500-step LFM2.5-350M training replay where direct Oja led EIGH at
every checkpoint (step 500 loss: EIGH `1.895432`, separate-target
`1.891082`, direct Oja **`1.890727`**). `5f3d830` and `865947e`
(2026-07-15) promote Oja to default after a rank-128, 1k-step promotion
gate. `7a9283e` (2026-07-16, 22:31), "Simplify UsuiTrack tracking path," is
cleanup of an already-decided state, not a new argument.

**This was a reasoned reversal, not an unexplained one, but it was decided
entirely on the conditioned (Adafactor) tangent.** Every number in
`3f506be`, `5f3d830`, and `865947e` predates this week's Adafactor deletion.
The position/velocity argument was never rebutted in prose anywhere in that
history — it was outrun by a measurement that favored the theoretically
"wrong" arm, on a mechanism (conditioned Oja) that no longer exists.

**On whether the fixed-point property weakens the velocity-control
objection:** no evidence was found that anyone made this argument, in
either direction, at the time. This document's contribution here, held over
from the first draft, is the reasoning itself: the velocity-control
objection is a critique of integrating a noisy, *unbiased-but-untargeted*
displacement — it assumes there is nothing for the integrated quantity to
converge to, so every step's noise is permanent drift. With a fixed point,
that assumption fails: near the fixed point, the "velocity" is not free
noise on top of a moving target, it *is* the restoring force toward that
target, exactly analogous to how position control's error decays
geometrically. §1's synthesis this revision adds sharpens rather than
undercuts that argument — the fixed point explains *why* the frame settles
onto the right eigenspace in the first place, while the *orbit* around it
(the thing that does not shrink) is a separate, batch-noise phenomenon that
would equally afflict a naively-implemented eigh aim refit from a single
boundary batch, unless that refit is itself built from averaged, persistent
data. That distinction — control-loop structure (position vs velocity) is
one axis; persistence-of-the-aim is a second, largely orthogonal axis — did
not exist as a stated frame in the first draft and is one of this
revision's structural additions; it is why Branch A2 below (revisiting eigh
aim) is now explicitly scoped as answering only the first axis.

### What this document still cannot determine

Whether direct Oja would still beat eigh aim on the *current*, unconditioned
tangent was never run. The arc's argument was never rebutted on the
mechanism that actually shipped; it was rebutted on one that is now gone.
This remains the largest piece of first-axis uncertainty in this document,
and Branch A2 names the cheap experiment that would settle it.

## 7. The solution tree

Restructured, per the coordinator's direction, around the aim question
first and the step-size question second — the reverse of the first draft's
ordering, because §1's synthesis makes the aim the branch that can actually
change the orbit's radius, and the step size only how fast the frame moves
around whatever orbit the aim produces.

### Branch A — Change the aim to carry persistence

#### A1. Cross-covariance aim: `G^T M` in place of `G^T Z` (leading candidate)

The proposal, stated by the coordinator: the current Oja action is `G^T
(G Q)` — the gradient acting on its own projection, i.e. the gradient
twice. Replace the second `G` with the already-stored moment: `A' = G^T M`
(right side; `M G^T` on the left). Same shape as today's action (`[n,r]`),
same cost class, **zero new bytes** — it reuses state the design already
carries for a different purpose.

**Horizontality has to be re-derived, and the naive construction breaks
it.** Today's code forms `R = sym(Q^T A)` and subtracts `QR`. That works
because `A = SQ` for symmetric `S = G^T G`, so `Q^T A = Q^T S Q` is already
exactly symmetric — the `sym()` call in the code is defensive
floating-point hygiene, not doing conceptual work. For the cross term,
`Q^T A' = Q^T G^T M` has **no reason to be symmetric** — its transpose is
`M^T G Q`, and `Q^T G^T M = M^T G Q` would require a special relationship
between `M` and `GQ` that does not hold in general. Symmetrizing before
subtracting therefore only removes *half* of the in-subspace component,
leaving `Q^T(A' - QR') = \text{antisym}(Q^T A') \neq 0` — the tangent
retains a piece that lives inside `span(Q)`, contaminating `sigma` with a
quantity that has no Grassmannian meaning (it does not correspond to any
subspace motion) and breaking the invariant the geodesic retraction is built
on.

**The fix is simple and exact: skip the Rayleigh/symmetrize step and
project directly.** `(I - QQ^T)X` is horizontal for *any* `X`, symmetric
action or not — `Q^T(I - QQ^T)X = Q^T X - Q^T X = 0` always. So `Delta'' =
G^T M - Q(Q^T G^T M)` is horizontal by construction, no symmetry argument
required. This is a real, checkable derivation, not a guess: it costs one
matmul the same shape as today's, computed the same way.

**Does it have a fixed point? This document can offer a first-order
argument, not a proof.** Consider the idealized case: the gradient
distribution is locally stationary, the frame has already settled onto
today's fixed point (the top-`r` eigenspace of `S = E[G^T G]`), and enough
steps have passed at a fixed frame that `M`'s EMA has converged to its
limit under a *constant* `Q`. Under those conditions `Z_t = G_t Q` for every
recent `t`, so `M \to E[G] \, Q$ — the moment converges not to anything
about the gradient's *covariance*, but to its *mean*, projected through the
(now-fixed) frame. Then, in expectation over a fresh batch `G_t`:

```
E[G_t^T M] ≈ E[G_t]^T (E[G] Q) = (E[G]^T E[G]) Q
```

This is **a different target matrix than today's**: `E[G]^T E[G]` is the
outer product of the *mean* gradient with itself, not `E[G^T G]`, the raw
second moment. The relationship between them is an exact identity, and it
is the cleanest statement of what this branch buys:

```
E[G^T G] = E[G]^T E[G] + Cov(G)
```

Today's aim tracks the top-`r` eigenspace of **signal plus per-batch
noise**. The cross-covariance aim tracks the eigenspace of **signal
alone**. That is not a heuristic resemblance to Adafactor's SNR weighting;
it is a decomposition, and the `beta=0.95` moment supplies the ~20-step
averaging window that separates the two terms at no extra cost. A direction can carry large, real energy every batch
(contributing heavily to `E[G^T G]`) while averaging toward zero across
batches if its sign or orientation is inconsistent (contributing nothing to
`E[G]^T E[G]`) — which is exactly a mathematical description of "bursty but
not persistent." Conversely, a direction with a smaller but *consistent*
per-batch component survives the averaging and dominates `E[G]^T E[G]`
relative to its share of raw energy. **This is the mechanism by which the
cross-covariance construction would supply the persistence-weighting the
current second-moment aim structurally lacks — not an analogy to Adafactor's
SNR weighting, but a different, cheaper route to a similar effect, arrived
at by asking a genuinely different mathematical question of the same
data.** If `Q` sits at the top-`r` eigenspace of this *new* target matrix
and `M` has converged as above, the identical algebraic argument P1 used
for the current design applies here too: `(E[G]^T E[G]) Q = Q\Lambda'$ for
some `\Lambda'`, so `E[(I-QQ^T) G_t^T M] = 0` — a fixed point, of a
different, more persistence-filtered quantity.

**What this argument does not establish**, stated as plainly as the
coordinator asked: this is an expectation-level, idealized calculation. It
assumes the frame is stationary long enough for `M` to have converged to
its constant-frame limit, glosses over the finite EMA window (`beta=0.95`,
an effective memory of roughly 20 steps — notably the same order as the
measured ~20-step acquisition transient in §1, which may or may not be a
coincidence and is not analyzed further here) and the parallel-transport
bookkeeping that keeps `M`'s stored coordinates valid as the frame actually
keeps moving in practice. It does not establish that the realized process
(noisy `G_t`, a frame that never actually stops moving) reaches this fixed
point rather than orbiting a *different* nonzero equilibrium, the same way
today's design orbits its own. **Whether the cross-covariance aim's orbit
is smaller, larger, or differently shaped than today's is an open, real
question this document cannot settle from documents alone.**

- **Cost.** Zero new persistent bytes. FLOPs: one matmul of the shape
  already computed today (`G^T (\cdot)`), reusing `M` — cheaper than or
  equal to today's construction, since `M` is already resident and no
  additional projection of `G` is required beyond what the moment update
  already does. One implementation detail matters and should be gotten
  right, not hand-waved: whether the action uses `M` from *before* this
  step's blend-in of `Z_t`, or after. Using the pre-update `M` keeps the
  persistence signal genuinely distinct from the instantaneous one; using
  the post-update `M` (which already contains 5% of this step's `Z_t` at
  `beta=0.95`) partially reintroduces the thing this branch is trying to
  filter out. This should be an explicit, documented choice, not an
  accident of call order.
- **Self-annealing.** Structurally preserved by the same mechanism as
  today — the tangent still reads a residual, and the residual's own size
  and shape still set the rotation, at whatever step-size rule sits on top.
  Whether its *equilibrium* is smaller than today's is exactly what is
  unmeasured.
- **Which trap it neighbors.** None of the banned four directly. It is
  adjacent to nothing in the "already tried" list because it changes what
  is measured, not how the measured quantity is turned into an angle.
- **Which self-preservation risk it carries, checked against §3.** `M` is
  an inward signal (built from projections onto the current `Q`), so using
  it *inside* the residual computation looks, at first glance, like exactly
  the incumbency-bias loop §3 warns against. The resolution is that the
  *contact point with the outside world is still `G`* — full-rank,
  unprojected, read fresh every step — and `M` only supplies the *direction
  to correlate `G` against*, not a substitute for `G` itself. The residual
  `(I-QQ^T)G^T M` still requires the full-rank `G` to compute; it cannot be
  formed from `M` alone. So this construction still nominates from a
  full-rank read, exactly as §3 requires — it changes what the residual is
  asking the full-rank gradient (does it persist, rather than does it have
  energy), not whether it looks outward at all.
- **Falsified by.** A direct real-gradient analogue of
  `test_a_fitted_frame_is_a_fixed_point_of_its_own_gradient`, adapted for
  this construction: run a real model long enough for `M` to have settled
  under a slowly-moving frame, then check whether `(I-QQ^T)G_t^T M` shrinks
  toward the noise floor for a genuinely persistent `M`, the way the
  original test checks the horizontal residual against the exact gradient
  that fitted the frame. Synthetic data is inadmissible for this per the
  project's own rule — the whole premise being tested is about persistence
  across *correlated, real* batches, which synthetic iid data does not
  have by construction. This is the single most important experiment this
  document names, because it is the only one that could kill A1 outright
  on its own premise rather than on cost or elegance grounds.

#### A2. Revisit eigh aim (position control), now approved for a local-only branch

Carries the cost profile the first draft already worked out: a full
side-Gram `eigh` (`O(d^3)`) every refresh interval rather than today's
`O(r^3)` tangent-Gram `eigh` every step, plus the rigidity question — a raw
`eigh` refit is a *replacement* of the frame, not a rigid rotation of it,
so it needs its own construction (a geodesic step *toward* the freshly-fit
target via a log map, not a jump-replace) to keep moment transport's
identity-coordinates property intact, per the hard constraint in §5's
"already tried" list.

**What changes in this revision: A2 is explicitly scoped to the
position/velocity axis only.** §6 draws the distinction plainly — position
control and persistence-weighting are separate, largely orthogonal design
axes. A raw `eigh` refit from a single boundary batch is no more persistent
than direct Oja tracking from a single batch; it is simply a different
*way of converting* one noisy sample into a rotation. Reviving eigh aim by
itself, without also feeding it an averaged or otherwise persistence-
weighted side-Gram, does not address this document's central thesis. It
answers a real but different question (does the closed-loop contraction
argument still win now that Oja also has a fixed point, on the mechanism
that actually shipped) and should be pursued for that reason, on its
already-approved local-only branch, in parallel with A1 rather than as a
substitute for it.

- **Falsified by.** Rerun `3f506be`'s exact head-to-head (boundary EIGH vs.
  live-EIGH vs. direct Oja) on the current unconditioned tangent, same
  model, same rank, same step budget. This remains the cheapest single
  experiment named anywhere in this document, because the harness already
  exists.

#### A3. Revive C1-style tangent accumulation directly — closed on cost, formally

The arc's own accumulation lattice (C1 sum-of-tangents through C6
Riemannian rotation-averaging) already surveyed this space and measured
C1 as effective ("removes the noise wall") before killing it for the
position-control argument, not for cost. Reviving it *now* — an explicit
`[d,r]` or partial buffer, EMA'd or reset-and-summed, per matrix — runs
straight into the same VRAM-first ceiling the arc itself named for
SubTrack's full-rank buffer, one size class down: any persistent buffer the
same order as a second basis is a real, permanent memory cost this design
has otherwise held at exactly one frame plus one moment. **A1 is preferred
over A3 specifically because it gets the same kind of averaging benefit —
persistence weighting via an accumulated quantity — by reusing a buffer the
design already pays for, rather than adding a new one.** A3 is recorded here
to close it formally, not because it is expected to be revisited: if A1
turns out not to deliver the effect the derivation above predicts, A3 is
the fallback with a known, already-measured mechanism, at a known,
non-trivial cost.

### Branch B — Step-size signals, layered on whichever aim is chosen

#### B1. Agreement (`cos(Z, M)`) as a step-size signal, not an aim signal

The coordinator's proposal: the cosine between this step's projected
gradient `Z` and the moment `M` asks "does this batch confirm accumulated
history" — a persistence question about *how much to trust this step's
outward reading*, not a preference about *which direction to rotate
toward*. Checked against §3's rule explicitly, because this is exactly the
kind of candidate that rule exists to police: both `Z` and `M` are inward
signals (both are projections onto the current `Q`), so agreement between
them cannot nominate anything — it has no access to the residual, only to
what the frame already captures. What it legitimately *can* do is throttle
the step size applied to the (separately-computed, outward-derived)
tangent this step, which is a ranking/gating role, not a nomination role,
and therefore does not reopen the incumbency loop.

Paired with `tangent_concentration`, this gives a 2x2 read of the
always-ambiguous case P2 names directly:

| | high concentration | flat spectrum |
|---|---|---|
| **agree** | converged and steady — low/moderate step, nothing new | equilibrium noise — small step, default case |
| **disagree** | structured drift — the batch disagrees with history, but its own structure is real: **turn** | noise — this batch's disagreement has no shape: **do not turn** |

- **Cost.** Zero new state — `Z`, `M`, and the tangent-Gram eigenvalues are
  all already computed in the existing fused kernel and basis-update batch.
  One extra on-device reduction (a cosine) per matrix per step.
- **Self-annealing.** Refines, does not replace, today's mechanism — the
  rotation still reads the residual's own size, gated additionally by
  whether this reading looks like it should be trusted.
- **Falsified by.** Pure instrumentation first, no mechanism change: log
  agreement alongside `sigma`/`concentration`/`basis_lag_angle` (Branch C)
  on a real run, and check whether the disagree/high-concentration quadrant
  actually correlates with later improvement in `basis_lag_angle`'s
  settled-vs-churning read, and — per §3's corollary, since agreement is
  itself an inward-facing quantity and therefore cannot rank designs on its
  own — with **loss**, not with any diagnostic alone. This is a real
  training-run-level experiment, not a wiring check.

#### B2. Constant multiplier on the schedule — allowed, low-risk, not a fix

Restated from §5: this remains available and already partially validated
(the hot-start result), but §1 point 3 makes plain that it only rescales
the ~20-step transient acquisition phase, not the 980-step steady-state
orbit. It can ship independently of everything else in this document, at
any time, without waiting on A1's or B1's falsification results, and it
should not be mistaken for progress on the central question.

**Formal death notice for the first draft's Branch 1.** The prior version
of this document proposed deleting `eta`'s clock dependence entirely and
relying on `sigma`'s own geometric decay to carry self-annealing — reasoning
that subspace iteration under a stationary action converges geometrically
on its own once a fixed point exists. That argument is falsified directly
by the measurement in §1 point 3, which predates this document's original
draft: `sigma` does not decay past the ~20-step transient; it plateaus at
`~80` (rank 128) for 980 of 1000 steps. A constant `eta` against a flat
`sigma` produces a constant nonzero rotation forever, which is the opposite
of the "well-fit -> small step -> settles" behavior value 2 requires. The
reasoning error, named plainly: the fixed-point property (P1) guarantees
where the *expected*, noiseless action would settle; it says nothing about
whether the *realized*, per-batch process actually shrinks its distance to
that point over time, which is precisely the equilibrium-not-convergence
distinction §1 works out. That branch should have been checked against the
`sigma` backout data before being proposed, and was not; it is dead, and
this document does not attempt to rescue a variant of it.

### Branch C — `basis_lag_angle`: a prerequisite instrument, not a branch to evaluate

**This branch's status changes completely in this revision.** The first
draft closed it on VRAM-first grounds, reading the arc's proposal as a
per-matrix, universal, permanently-stored snapshot buffer. The coordinator
has since approved it as a **sampled diagnostic over ~32 matrices, opt-in,
off the hot path** — which changes the cost arithmetic entirely: this is
not "double the state of every tracked matrix," it is "hold one extra
`[d,r]` snapshot for a small, bounded, sampled subset, refreshed at a
logging cadence, discarded like every other diagnostic." That scope was
never priced in the first draft's rejection and reverses it.

More importantly, its role changes from "one branch among several" to
**prerequisite for judging any of the branches above.** Every experiment
proposed in Branch A and Branch B — A1's fixed-point check, A2's rerun of
`3f506be`, B1's correlation study — ultimately needs to answer some version
of "did this actually shrink the orbit, or just rearrange it," and
`rotation_rad_sum` and target self-angle are both, in the arc's own words,
"floored by target noise by construction" — they cannot distinguish a
frame that has settled onto a stable direction from one that keeps rotating
toward a different noisy target every refresh while producing the same
aggregate angle. `basis_lag_angle` — principal angles against the frame's
own snapshot from `N` refreshes back — is the only metric in this design
space that goes to zero **iff the frame's actual direction stops changing**,
independent of how large or small the per-step angle happens to be. Two
reads from one comparison: `basis_lag_mean_angle` (settling: `-> 0` iff
every plane stops) and `basis_lag_top_angle` (orbit radius: churn planes
hold it up even while the mean settles).

Concretely, this metric is what would tell apart the two readings §1's
synthesis leaves open: an equilibrium `sigma ~80` could mean a frame that
has locked onto a stable pair of directions and is harmlessly orbiting them
(lag angle small and flat), or a frame whose *aimed-at* direction keeps
sliding to a new, differently-noisy target every refresh while producing a
similar aggregate rotation magnitude (lag angle large and flat). Every
number currently in the release is blind to that distinction. Build this
before, or at latest alongside, the first real-gradient experiment in
Branch A or B — without it, none of those experiments' results are
interpretable as "the orbit got smaller" versus "the orbit moved."

### Branch D — Full-spectrum vs per-plane rotation (P10), revisited

The prior conclusion — concentration measured high (`~0.4`-`0.86`, far
above the isotropic floor `1/r`), so the near-isotropic-tail-churn worry
P10 raised does not describe what real gradients produce — survives this
revision's reframing intact. The "equilibrium, not convergence" picture
does not change this: concentration measured *at* the equilibrium is still
evidence the residual has real shape, whether or not that shape is fully
settled. Closed, as before, on the concentration data already in
`PLAN.md`; the arc's own parallel proposal, "R6, distinguish signal planes
from cutoff-churn planes," was scoped as a diagnostic-only lead and never
promoted — it is the same conclusion this document reaches independently,
not new evidence for it.

### Branch E — Adaptive cadence (`basis_update_interval`, P9), demoted

The first draft's cadence-gating idea (B5b: skip a matrix's basis update
when its tangent Frobenius norm falls below a threshold, on the theory
that a converged matrix's tangent shrinks toward zero) **relied on exactly
the premise §1 now shows is false for real gradients**: `sigma` does not
approach zero on settling, it approaches a nonzero equilibrium (`~80` at
rank 128). An absolute-threshold gate built on that premise either never
fires (threshold set above the real equilibrium value) or fires
immediately and permanently the moment the frame *first* reaches
equilibrium — which, per Branch C's exact distinction, could be either a
genuinely converged, harmless orbit or a still-live, still-informative one,
and an absolute tangent-norm gate cannot tell those apart. This branch is
demoted from independently pursuable to **a downstream consumer of Branch C
or Branch B1**: a cadence gate becomes sound once there is a signal
(`basis_lag_angle` settling, or a run of low-agreement-high-concentration
readings) that actually distinguishes "this matrix has nothing left to
learn" from "this matrix is still learning something, just at a stable
apparent rate." Static interval changes (the first draft's B5a) remain an
orthogonal systems question, unaffected by this reframing, and still
unmeasured per P9.

## 8. Recommendation

Sequenced, not ranked as independent picks — several of these depend on
each other's existence to even be interpretable, which the first draft's
flatter branch structure did not surface.

1. **Build Branch C first.** It is cheap (sampled, opt-in, already
   approved), it does not require choosing among A1/A2/B1 first, and every
   later experiment in this document needs it to distinguish "the orbit got
   smaller" from "the orbit moved." Shipping it first also means the team
   is not running expensive real-model experiments blind to the one
   question that actually validates them.
2. **In parallel, instrument B1 (agreement) as pure logging, no mechanism
   change**, alongside `sigma`, `concentration`, and (once built)
   `basis_lag_angle`, on an existing run. This is the cheapest possible
   check of whether the 2x2 partition in Branch B1 actually separates cases
   the way the hypothesis predicts, before committing to using it as a live
   step-size input.
3. **Develop A1 (cross-covariance aim) as the primary aim candidate.** It
   is the only candidate in this tree that structurally builds in
   persistence at zero new memory cost, by reusing state the design already
   carries; the expectation-level derivation in §7 gives a concrete,
   checkable reason to expect it targets a genuinely different, more
   persistence-filtered quantity than today's second-moment aim, rather
   than being a relabeled version of it; and it has a falsification
   experiment — a real-gradient analogue of the existing fixed-point test —
   that could kill it outright on its own premise, which is exactly the
   kind of test this project's standing rule ("plausibility is not
   correctness") asks for before anything ships.
4. **A2 (eigh aim revisit) proceeds independently, in parallel, on its
   already-approved local-only branch**, understood explicitly as answering
   the orthogonal position/velocity-control question rather than the
   persistence question this document centers on. Its cheap falsification
   experiment (rerunning `3f506be` on the unconditioned tangent) should run
   regardless of how A1 turns out, because it settles a question this
   document could not: whether direct Oja still wins on the mechanism that
   actually shipped.
5. **B2 (constant multiplier) can ship at any time, independently of
   everything above** — low-risk, already partially validated, and
   explicitly not a substitute for the aim work.
6. **Branch D stays closed. Branch E stays parked** behind Branch C or B1
   actually producing a signal it can safely gate on.

## 9. Risks, named plainly

- **A1's fixed-point argument is an idealized, expectation-level
  derivation — not a proof, and not a measurement.** The weak links, named
  explicitly: the assumption that `M` has converged to `E[G]Q` under a
  frame that has stopped moving long enough for that to happen; the glossed-
  over finite EMA window and its interaction with parallel transport under
  a frame that in practice never fully stops; and the total absence, so
  far, of any real-gradient check of whether the realized (not idealized)
  process actually approaches this fixed point rather than orbiting a
  different one. This document names the experiment that would settle it
  and stops short of claiming the result.
- **Whether a smaller orbit is even the right thing to optimize for is a
  separate, harder question this document cannot answer.** §3's corollary
  applies to the whole design, not just to individual candidates: capture,
  agreement, and orbit radius all explain results; only loss ranks
  designs. Nothing in this document substitutes for a loss-level comparison
  between the current aim and A1, once A1 exists to compare.
- **The equilibrium-not-convergence finding itself is this document's own
  synthesis of five separately-measured facts, not a sixth measured fact.**
  It is offered as the most economical explanation on hand, and Branch C is
  specifically the falsification path for it — if `basis_lag_angle` shows a
  settled, non-churning direction at the current equilibrium, the "orbiting
  a fixed point" picture holds; if it shows a direction that keeps sliding,
  something in this synthesis is wrong and needs revision before A1's
  results can be trusted to mean what this document says they mean.
- **`basis_lag_angle`'s exact per-model byte cost is left unstated
  on purpose** — it depends on model size and sampling cadence, both left
  to the implementation, and this document only asserts that "sampled over
  ~32 matrices" changes the VRAM arithmetic qualitatively relative to a
  universal buffer, not that it is free.
- **Synthetic evidence remains inadmissible for every claim in this
  document**, including the mathematical derivations in Branch A1 — those
  are checkable algebra, not empirical claims, and are labeled as such;
  every falsification experiment named above requires real, correlated,
  non-iid gradients, and none of them can be substituted with a synthetic
  run except to check that an implementation is wired correctly once one of
  these branches is built.
- **This document's own prior draft got one branch wrong on evidence that
  already existed**, and says so above rather than quietly dropping it.
  That is worth flagging as a standing risk for this whole exercise, not
  just a historical note: reasoning from the mathematics without checking
  it against the measurements already on file is exactly the failure mode
  that produced the dead Branch 1, and this revision's own new claims
  (§1's synthesis, A1's derivation) carry the same risk until they, too,
  are checked against real-gradient measurement.
