# UsuiTrack specification

Current matrix-update design. This document specifies only the selected
basis-tracking path; see the [README](../README.md) for results and usage.

## Read these traps first

- **Basis != subspace.** `Q` and `QH` span the same subspace, but coordinates in
  those frames differ.
- **Left is right on the transpose, not with renamed shapes.** Every frame
  operation takes the matrix oriented so the frame sits on its columns, `G` for
  `right` and `G^T` for `left`, and derives once.
- **Overlap reprojection != parallel transport.** Reprojection preserves the
  least-squares part of a fixed ambient vector. UsuiTrack carries momentum with
  its moving frame, so its stored coordinates do not change at refresh.
- **A healthy polar output can hide a sick input.** Newton--Schulz restores
  semi-orthogonal scale after weak directions have already become noise-dominated.
- **A tracked side is a rule, not architecture.** `right` puts every frame on
  its matrix's input and `auto` on its shorter side; neither knows a
  transformer's residual-facing axis, and a param group's `side` says it where
  it matters.
- **`ndim == 2` is a shape gate, not the precondition.** Lookup tables pass it
  and break the method, for reasons no numerical guard can see. See "Parameter
  eligibility".
- **Frame motion must be able to anneal.** What the tracker owes is the ability
  to settle -- motion that falls as the frame approaches its equilibrium. A
  normalization that fixes the move's size whatever the frame's distance from
  its target breaks that, however well it is conditioned.
- **Passing tests over a mechanism that never fires is not evidence.** Guards
  and code paths that no configuration reaches will pass every test they have.
- **Exact implementation of a formula does not validate the formula's premise.**

## Coordinates

Let a matrix parameter and gradient be

$$W,G\in\mathbb R^{m\times n},\qquad 1\le r\le\min(m,n).$$

Rank is not configured per matrix. One model-wide `rank_fraction` `c`
(default `0.05`) sizes every matrix by its shape:

$$r=\min\!\left(\max\!\left(1,\ \operatorname{round}(c\sqrt{mn})\right),\ \max\!\left(1,\ \lfloor\min(m,n)/2\rfloor\right)\right).$$

The rank sets the subspace, not the step: the step is sized against the
weight's own norm (step 8), so `rank_fraction` decides which directions a matrix
moves in and, where the moment fills its rank, never how far (step 7). On LFM the rule matches or beats a
per-role rank sized from measured live planes, with no measurement of its own
(`FACTS.md` on `lab`, "Rank is a fraction of the matrix's size"). Matrices of one shape
share one rank.

The cap has two structural reasons, neither of them numerical. A gradient of shape `[m,n]`
has rank at most `min(m,n)`, so a wider basis tracks directions the gradient
cannot populate; half leaves an orthogonal complement for the aim's horizontal part at
every step. And narrow modules are bottlenecks -- they carry the residual
stream through a small waist, so a large update there destabilizes every block
downstream, independently of how well-conditioned the step was.

At `c = 0.05` the cap binds on no shape of LFM's or Anima's; at `0.1`, on
Anima's `(2048, 68)` patch embedding, which asks for 37 planes and runs at 34.
The default is the smaller: on LFM3 `0.1` buys capture and no loss under
either aim (`ARCHIVE.md` on `lab`, "a live aim does not make a larger rank pay").

`Q:[d,r]`, `Q^T Q = I`, is a frame on one side of the matrix, `d = n` for
`right` and `d = m` for `left`. Orient the gradient so that side is its columns,
`G_o = G` (right) or `G^T` (left), `[p,d]`; then for both sides

$$\Pi_Q(G)=G_oQ:[p,r],\qquad \Lambda_Q(Z)=\text{orient}^{-1}(ZQ^\top):[m,n].$$

The moment lives in `[p,r]`, the frame in `[d,r]`; every formula below is the
right case, and left is it on the transpose.

`side="right"`, the default, tracks every matrix's input side, where Molegrad's
aim is the diagonal aim from `X` and `delta` ("Molegrad"). On a residual
writer it gives up source in a finetune, 3-9e-3 on LFM3 (`FACTS.md` on `lab`,
"Residual writers on the input side"). `side="auto"` tracks the shorter side, `right`
for `m >= n`: a shape guess, not an architectural one (see the trap above). A
param group can set any of the three; tracking the residual stream on every
weight is one such assignment, measured on LFM and Anima
(`docs/notes/2026-10-28-side-residual-facing-against-auto.md` on `lab`).

Every Gram, aim and frame computation upcasts fp16/bf16 inputs to
fp32 first. The basis is the one piece of state everything else is built
on; a half-precision frame computation doesn't just cost this step's
accuracy, it corrupts the tracked frame for every step after it.

## One matrix step

```text
raw gradient G
  -> sanitize
  -> range-finder initialization on the first gradient, or a random frame
  -> held-frame projected gradient Z
  -> Rayleigh-normalized covariance action, the aim a
  -> release full gradient; retain rank-sized pending work
  -> unit Frobenius norm Z/||Z||_F
  -> projected EMA M of the unit gradient
  -> Aurora leverage balance of M (normalization)
  -> Newton-Schulz polar map of M (orthogonalization)
  -> unit scale 1/sqrt(r)
  -> lift through the held frame Q, scaled by ||W||_F
  -> parameter update
  -> frame moment Y of the aims; frame to polar(Q + Y)
  -> rotate M by the stored-frame overlap; commit it once
```

The moment is the last thing written, and everything above it reads the held
frame `Q`. The lift is through `Q`, not `Q_+`: the frame moves after the
parameter update, so no step lifts through a frame it did not measure in.

### 1. Sanitize

Let `G` replace each non-finite entry with zero. The incoming gradient's
Frobenius norm, taken before the replacement, is measured for telemetry and
applied to nothing.

**There is no gradient clipping, because no consumer can see the magnitude.**
The frame's aim is exactly invariant to a uniform rescale (step 4: the action and
its Rayleigh quotient are both quadratic in `G` and divide). The update normalizes each
step's projected gradient to unit Frobenius norm *before* the average (step 5),
so a per-step rescale -- which is exactly what a clip is -- cannot reach the
moment either. A blip batch contributes one unit of weight like every other
step. Pinned by `test_the_update_is_invariant_to_gradient_scale`.

### 2. Initialize the frame

Normalize `A = G_o / ||G_o||_F` and form the side Gram `K = A^T A`. `Q` is a
randomized range finder's frame on its leading `r`-dim eigenspace, with one power
step: `Q = qr(K qr(K Omega).Q).Q`, `Omega` an `[n,r]` Gaussian drawn from a fixed
seed by a private generator, so the fit is a function of the gradient alone and
the training run's random stream never moves. A gradient of rank at most `r`
lies wholly in `Q`; past that, `Q` holds the leading eigenspace approximately,
the shortfall growing as the spectrum flattens beyond `r`. No eigendecomposition
runs, so a Gram whose eigenvalues fall below fp32 resolution a few hundred deep
-- a nearly rank-one gradient, as a zero-initialized model's upstream matrices
see first -- fits like any other. The frame needs the leading eigenspace, not
its eigenbasis, and not exactly: the tracker moves it from the next step on.
The fitting step moves no frame; the moment and weight step as usual.

**The fit takes the first gradient as it is.** A gradient that is exactly zero
after step 1's sanitizing, which includes one that was entirely non-finite, has
no leading eigenspace, and the range finder returns an arbitrary orthonormal
frame from it (the QR of zero); its `Z` is zero, so the moment holds and the
weight holds but for its decay, and the tracker moves the frame from the next
step on. The fit reads nothing back to the host. Molegrad draws the sketch when it
binds, so its fit traces inside a compiled backward; the optimizer's own fit draws
the sketch at the fit and runs eager. Molegrad's once-per-matrix check that the fit
is the first backward is the one read it keeps. A model trained
from a zero-initialized output (adaLN-zero) sees exactly such gradients upstream
until the output moves, and gains nothing from a fit: its param groups take
`"frame_init": "random"`, which draws each frame when the group is added, on the param's device, as the
QR of an `[n,r]` Gaussian from a private generator seeded once per optimizer and
device, so the draws follow the param order and the training run's random stream
never moves. The frame then tracks from step one; a matrix whose gradient
is still zero has a zero aim, and its frame holds.
`frame_init="range_finder"`, the default, is meant for finetuning, where the first gradient
already carries the subspace and a short run needs the frame on it from step one
(unmeasured).

### 3. Project in the held frame

Using the held frame before the current frame move,

$$Z_t=\Pi_{Q_t}(G_t).$$

### 4. Track with the frame's moment

On every step after the fit, reuse the held-frame projection to form the
covariance action and normalize it by the frame's mean Rayleigh quotient:

$$A=G_o^\top Z,\qquad a=\frac{A}{\operatorname{mean}\operatorname{diag}(Q^\top A)}.$$

The denominator is floored at `1e-30`, which guards only an exactly zero
gradient. `A` and its Rayleigh quotient are both quadratic in `G`, so the aim is
blind to the gradient's scale, and `Q^T a` has mean diagonal one. The frame
keeps a moment of its aims, `Y:[d,r]`, starting at zero, and moves to the polar
factor of the held frame plus that moment:

$$Y_t=\beta_f Y_{t-1}+(1-\beta_f)\,a_t,\qquad \beta_f=0.95,\qquad
Q_+=\operatorname{polar}(Q+Y_t).$$

**The held frame is the anchor.** For an aim taken at the held frame,
`Q^T a = Q^T A / mu` is symmetric positive semidefinite (`Z^T Z` on the `G`
path, `(XQ)^T D (XQ)` for Molegrad's diagonal aim, `D` the diagonal of
`||delta_t||^2`). `Y` averages aims taken at
the frames of the last `~1/(1 - beta_f)` updates and is not transported, so its
in-span part is that only approximately, as close as the frame stayed over the
window. Where it holds, `T = Q + Y` has its singular values at or above one, and
no plane collapses, whatever the aim leaves empty. A moment inside the span,
`Y = QS`, gives `T = Q(I + S)` and `polar(T) = Q`: a frame on the gradient's
leading eigenspace holds, so on a persistent gradient the tracker settles as
the frame arrives. On a white stream it does not: the EMA of independent aims
keeps a horizontal part of about `sqrt((1 - beta_f)/(1 + beta_f))` of one aim's,
and the frame walks at that floor. Motion comes from the moment's horizontal part, measured
against an anchor of one plus the energy the frame already holds, so a frame
that holds much of the gradient moves little.

**`beta_f` sets the trade between lag and noise.** The frame is in effect the
polar factor of the aims of the last `~1/(1 - beta_f)` updates. In replay on
LFM3's recorded gradients, input side under the diagonal aim, it captures as much of the
next batch as a constant 0.02 rad geodesic turn does, within `6e-3`, while
moving `0.007-0.010` per step against that turn's `0.020` (`FACTS.md` on `lab`,
"In replay under the diagonal aim, the twin's polar from Newton-Schulz reads as
`eigh`'s").

**The polar factor is Newton-Schulz, then one retraction.** Step 7's
Newton-Schulz on `T` lands within a few percent of orthonormal, and one
near-identity step finishes it: with `S = Q_raw^T Q_raw` and the converged
steady-state coefficient triple from Amsel, Persson, Musco, and Gower, ["The
Polar Express"](https://arxiv.org/abs/2505.16932) (2025),

$$Q_+=Q_{raw}(aI+bS+cS^2),$$

`1.2-1.6e-4` off orthonormal in replay on LFM3's gradients under either aim,
and the same frame as the exact polar factor to three digits (same FACTS
entry). A top plane holding ~30x the mean in-span energy leaves it `2e-2` off;
the next move re-polarizes `Q + Y`, so the error does not accumulate. The retraction runs as one
more iteration of step 7's loop on the same Gram. No
eigendecomposition runs and nothing is read back to the host.

The frame is stored round-to-nearest; stochastic rounding measured `sqrt(2)`
more in-span rotation. `Y` is held against the stored frame, `Y = QS + H`, with
`S = Q^T Y` `[r,r]` in fp32 and `H` `[d,r]` in the parameter's dtype, split again
after each move. `Y`'s columns are large along the planes the frame holds and
its horizontal part, the signal, is small: a bf16 `Y` rounded whole, unbiased but
noisy at a fraction of an ulp of its large part every step, walks the frame,
which on LFM3 cost `6e-2` in training loss by step 50 (`FACTS.md` on `lab`, "The
frame moment's in-span part needs fp32"). Held as `QS + H`, the large part is
exact in `S`, on the frame the anchor uses, and `H` rounds only its own small
size. A load takes `S` back from the checkpoint in fp32, where torch would cast
it to the parameter's dtype. `Y` is not carried by the frame's in-span rotation.
A matrix with no gradient on a step is
skipped: its frame, both moments and its weight hold.

### 5. Accumulate momentum

The moment integrates the projected gradient at unit Frobenius norm:

$$M_t=\beta M_{t-1}+(1-\beta)\frac{Z_t}{\|Z_t\|_F},\qquad \beta=0.95.$$

There is no EMA bias correction, and none is needed: the polar map of step 7
discards the moment's size, so an undersized early moment steps as far as a
full one.

**The average runs on gradients, and the polar map runs once, on the average.**
A batch's spectrum is information the average needs: the planes a batch
carries strongly are the planes the expected gradient carries, and its noise
planes are small. Averaging first keeps that weighting, so `M` estimates the
expected gradient; a polar map ahead of the average would hand every noise
plane of every batch full length, and the average of polar factors is not the
polar factor of the average. The normalization takes out only the batch's
overall size, so no one batch owns the memory by being large -- a clip's job,
without a threshold -- and `||M||` stays a persistence read on a constant-norm
stream (`moment_persistence`, a lab read). The step's size does not follow it (step 8).

**What the moment is for.** Late in an LFM run at bs16 the stream is white --
`moment_persistence` within `1e-3` of zero -- so there the moment is a variance
reducer, not a persistence extractor: it buys `sqrt((1+beta)/(1-beta))` in
signal fraction, 6.2x at `beta = 0.95`, and pays in staleness as the mean
drifts under the moving parameters. Early in a run it is not white: the
dominant planes persist while the loss falls fast, and the moment holds them.
Source damage grows with `beta`: forgetting follows memory length and travel.
`0.95` holds target against `0.98` under a decaying `lr` on LFM and forgets
less; `docs/FACTS.md` on `lab` carries the arms. The staleness half is set by
how far the parameters move, so `beta` and the schedule are read together.

The moment is no substitute for batch. A batch averages its samples at one
parameter point; the moment averages gradients taken at points the parameters
have since left. At equal samples the larger batch wins, and raising `beta` to
match a smaller batch's sample memory makes it worse, not better.

### 6. Transport momentum through frame motion

The whole step reads one frame. The held frame `Q_t` supplies the projection,
the aim, and the lift back to parameter space; the frame moves after
the parameter update, so `Q_{t+1}` is never used to lift an update that `Q_t`
measured. Moving the frame first would rotate every step's own gradient by
however far the tracker had just moved -- a systematic error scaling with the
step size, and one that falls entirely on the current gradient rather than on
the accumulated history it is correct for.

The moment moves with the frame. In the subspace the two frames share, the
moment's coordinates are carried column for column onto the new frame:

$$MQ^\top\mapsto M_+Q_+^\top,\qquad M_+=MR_s,$$

`R_s` the orthogonal polar factor of the overlap `Q^T Q_+` of the frames **as
stored**, from one retraction, exact at a step's small angles. Only the rotation
is applied, never the full overlap: its symmetric part is the principal-angle
contraction that re-projecting a fixed ambient vector would apply, and
transport is right to discard it. So the stored coordinates and the moment's
singular spectrum carry through the move unchanged.

The overlap is symmetric, and `R_s = I`, when the frame moves along a
horizontal `Y` alone, since then `Q^T Q_+ = (I + Y^T Y)^{-1/2}`. Two things turn
the columns inside their span: the moment's in-span part, where it is not
symmetric, and rounding the new frame into bf16, which moves the subspace not at
all while scrambling the coordinates one-for-one. `R_s` takes both.

The correction is why the moment is committed last. This step's projected
gradient is measured in `Q_t`, blends into a moment in `Q_t`, and the parameter
update lifts through `Q_t`; the moment moves to `Q_{t+1}` only once nothing is
left to read in the old coordinates. Rotating before the blend would add a `Q_t`
gradient onto a `Q_{t+1}` moment. Because the rotation must come last, the moment
stays fp32 from its accumulation through the polar map to that commit, and the
step rounds it exactly once -- stochastically in bf16, since a rotation is a
rearrangement whose systematic rounding would accumulate against the coordinates
it exists to preserve; to nearest in any other dtype.

### 7. Balanced polar direction

Two operations from two lineages, applied in order and worth keeping apart by
name. The **leverage balancing** is
[Aurora](https://github.com/tilde-research/aurora-release) (Tilde Research): a
diagonal row rescale that changes how force is distributed across rows and
orthogonalizes nothing. The **orthogonalization** is the Newton-Schulz polar
map of the Muon lineage, and it is the step that discards `M`'s singular values
so that the step's size is `lr`'s alone. Both are reimplemented from the
methods; neither is a runtime dependency. UsuiTrack takes only the rectangular
direction map from Aurora, per the decision note at the end of this section.

Balancing acts on `M_t`, the moment after this step's blend. For a rectangular tensor, orient it as
`A:[p,q]`, `p >= q`, transposing if needed. With the row scaling
`D_ii = 1/||A_{i,:}||_2` (floored at `1e-7`), the direction is

$$P=\operatorname{NS}(DA),$$

transposed back: one leverage-balancing iteration, whose diagonal update is
never applied. Square tensors skip balancing and use `NS(M)` directly.

`NS` first divides by Frobenius norm and orients its input with rows no greater
than columns. Five default Newton-Schulz polynomial steps apply, using the
per-step-tuned quintic schedule from the Muon optimizer lineage (the batched
implementation credited by [HeavyBall](https://github.com/HomebrewML/HeavyBall)'s
source to GitHub users `@scottjmaddox` and `@YouJiacheng`, descending from
[Keller Jordan's Muon](https://github.com/KellerJordan/Muon)). Values ported
directly into this optimizer; not a runtime dependency:

$$H_k=X_kX_k^\top,\qquad
X_{k+1}=(a_kI+b_kH_k+c_kH_k^2)X_k.$$

with

```text
k    a         b         c
0    4.0848   -6.8946    2.9270
1    3.9505   -6.3029    2.6377
2    3.7418   -5.5913    2.3037
3    2.8769   -3.1427    1.2046
4    2.8366   -3.0525    1.2012
```

Every iterate is `X_0 C_k` with `C_k` a polynomial in the short-side Gram `G = X_0^T X_0`
(the tall orientation), so the schedule runs on `G`: with `M = C^T G C`, the iterate's Gram, and
`P = aI + bM + cM^2`, an iteration is `C <- PC`, `M <- P^2 M`, every factor commuting. `G` and the
closing `X_0 C` are the only passes over `d r^2`, against two per iteration on `X`. On CUDA, up to
`r = 128`, the `r x r` iterations are one Triton program per matrix (`usuitrack/gram.py`), its
products in tf32x3; above that and off CUDA, batched GEMMs run the same loop, and a test holds the
two together. The Gram squares `X`'s condition number; LFM3's moments keep their smallest singular
value at `5e-3` of the largest or more, and the route reads as the iteration on `X` to three digits
(`docs/FACTS.md` on `lab`, "One polar step cannot follow `Q+Y`").

The result `O_t` is an approximate leverage-balanced polar direction, not an
exact SVD polar factor: at five iterations its orthogonality residual is ~1e-2.
A moment that fills its `r` planes has a polar factor of norm `sqrt(r)`;
scaled by `1/sqrt(r)`, its step is a unit. One that fills only `k` of them,
a frame wider than the gradient's rank, has norm about `sqrt(k)`: five
iterations leave a plane far below the top near zero.

**Decision (projected balancing):** Aurora's balancing chooses direction inside
the retained update space. UsuiTrack, not Aurora, owns momentum, basis motion,
scale, and LR.

### 8. Scale, lift, and update

The step is sized against the weight it lands on. `O_t` is a unit and the lift
is an isometry, so

$$U_t=\|W_{t-1}\|_F\,\Lambda_{Q_t}(O_t),\qquad
\frac{\|U_t\|_F}{\|W_{t-1}\|_F}=1$$

for a moment that fills its rank, and `sqrt(k/r)` for one that fills `k` of
`r` planes (step 7).

The learning rate is the relative change per step -- one unit in every module,
whatever its shape, rank or size: LAMB's trust ratio (You et al. 2019, arXiv
1904.00962), taken against a unit direction rather than the update's own norm.
The moment decides where each module steps and never how far, so `lr` reads
directly as `relative_step`, and LoRA-scale learning rates (`1e-5` to `2e-4`)
are in the right range. A run that should slow down gets that from the
trainer's `lr` schedule, which the optimizer does not own; a linear decay to a fifth of the peak beat a constant `lr` on LFM
(`docs/FACTS.md` on `lab`).

**Why the weight's norm and not its shape.** A scale read from shape -- Muon's
`sqrt(max(1, m/n))`, or the polar factor's own `sqrt(r)` -- fixes the update's
norm whatever the weight's, so the update's RMS per element grows as the matrix
shrinks: the module with fewest parameters takes the largest relative step. In
a diffusion transformer those are the input and output projections, the
bottlenecks the rank cap exists for (above). Against `||W||`, the update's RMS
is a fixed fraction of the weight's own. The shape-scaled step is the `fixed`
rule below, for matrices that have no norm to be relative to.

**The step rule.** A param group's `step_rule` is one of two.

- `"trust"` (default): the rule above, `U_t = ||W_{t-1}||_F Lambda_{Q_t}(O_t)`.
  A matrix with `||W|| = 0` never moves, since the step is sized against zero.
- `"fixed"`: the step is sized by the matrix's shape, not its norm. For
  `W:[m,n]` as stored, whichever side the frame is on,

$$U_t=\sqrt{\max(1,m/n)}\,\sqrt r\;\Lambda_{Q_t}(O_t),\qquad
\|U_t\|_F=\sqrt{\max(1,m/n)\,r}$$

  for a moment that fills its rank, and `sqrt(k/r)` of that for `k` of `r` planes.
  It is Muon's aspect factor times `sqrt r`, the norm of the polar factor that
  `O_t` is the unit of. `lr` is then an absolute step, so a zero-initialized matrix moves from its
  first nonzero gradient, and the relative step `||U||/||W||` follows from the
  weight rather than being `lr`. The rank enters the size here, so `rank_fraction`
  is no longer a pure subspace knob for a fixed group.

**Norm bounds.** A param group's `norm_bounds = (low, high)` (default
`(None, None)`) bounds each matrix's Frobenius norm in multiples of `R`, its norm
before its first write (state `init_norm`, fp32 also for a bf16 matrix, and loaded
from the checkpoint unrounded). The written weight is clamped radially,

$$W_t=\tilde W_t\,\frac{\operatorname{clamp}(\|\tilde W_t\|_F,\ \mathrm{low}\,R,\ \mathrm{high}\,R)}{\|\tilde W_t\|_F},$$

$\tilde W_t$ the right-hand side of the update below, inside the same accumulator and the
same single rounding; `None` opens a side. A write inside the bounds is unchanged.
`(1, 1)` pins the norm: under the trust rule `lr` is then the relative step at every
step, not only the first, and weight decay only rescales the step by
`1/(1 - lr lambda)`. A ceiling alone lets the norm move below it, so decay acts
until the floor, if any, holds it. `R` is read to the host once, at a matrix's first write; a matrix
whose `R` is zero cannot be bounded relative to it and raises. A checkpoint without
`init_norm` (saved without bounds) takes `R` from the weights it resumes with. The diagnostics read
the update before the clamp.

Weight decay, the lift and the rounding are the same under both rules, and a
param group chooses its own, so one optimizer can hold both. `relative_step`
stays `||dW||/||W||` under either.

With decoupled weight decay `lambda` (default `0`), and `U_t` sized from the
weight before the decay,

$$W_t=(1-\alpha\lambda)W_{t-1}-\alpha U_t.$$

Matrix parameters retain no full-size first or second moment.

**Rounding.** `W`, the lift and the update are accumulated in fp32 and written
back once: stochastically for bf16, to nearest for fp16, exactly for fp32. A
LoRA-scale `alpha` puts `alpha * U` well under a bf16 ulp -- measured at ~1/76
ulp on a 2B DiT at a relative step below `2e-5` -- and round-to-nearest then
discards every step. `U` is never cast to `W`'s dtype on the way in; a single
rounding happens at the write, with weight decay folded into the same
accumulator so it is not rounded twice. See `usuitrack/stochastic.py`.

**The write on CUDA** is one Triton kernel in two passes per block (`usuitrack/write.py`); off
CUDA the same maths runs in eager PyTorch, and a test holds the two together. The measure pass
reads `W` for `||W||^2`, and under norm bounds also `<W, L>` and `||L||^2`, so the clamp's factor
comes from `||aW - sL||^2 = a^2||W||^2 - 2as<W, L> + s^2||L||^2` with no pass over the result;
the write pass reads and writes `W` once, through a table of the members' addresses. Both form
the lift `L` from the rank-sized factors in registers, in TF32, whose error does not show
above stochastic rounding's (`docs/FACTS.md` on `lab`, "The CUDA write takes"). The rounding bits come from philox under a seed drawn from torch's CUDA generator per
block, so a seeded run reproduces, with draws other than the eager path's.

Stochastic rounding keeps the expectation and pays in variance. A coordinate
whose update `u` is below its ulp moves by a whole ulp with probability
`|u| / ulp`, so after `N` steps the weights sit a random walk of variance
`N |u| ulp` per coordinate away from the path the updates describe. The walk
grows as `sqrt(N)` and the path at most as `N`, but at a small step the walk
wins for a long time: at Anima's `relative_step` of `3.7e-6` it is
several times the fp32 travel after 350 steps, and ~90% of matrix coordinates do
not move at all in a 50-step window. **Distance from base weights on a bf16 model is
therefore a read of the walk, not of learning**, until the update approaches a
ulp. `docs/FACTS.md` on `lab` carries the measurement.

## Two-phase execution

For each matrix, phase one runs exactly once after its full gradient is complete:

1. sanitize `G`;
2. read `Z_t` and its norm and, after initialization, form the aim `a_t` in the held frame `Q_t`;
3. retain `Z_t`, its norm and the optional `a_t`, then release `G`.

Phase one does not move the basis, update the moment, or update the parameter:
the moment is rotated and committed after the frame moves (step 6), so it is
written only in `step()`. Phase two runs in `step()`, one block at a time
(below), so no fp32 moment outlives its block: it blends the unit `Z_t` into `M_t`,
applies the balanced polar map, lifts `O_t` through `Q_t`, updates `W`, then
blends the aim into `Y`, moves the frame to `Q_{t+}` and commits the moment.

Without `release_matrix_grads`, both phases run in `step()`, on whatever
gradient has accumulated, so gradient accumulation works. With it, the
optimizer registers a `register_post_accumulate_grad_hook` per matrix that runs
phase one as soon as that matrix's gradient is complete and frees it. That
takes exactly one backward per `step()`: a second gradient for a pending matrix
raises rather than overwrite or sum into rank-sized work it can no longer
reconstruct. Gradient clipping and norm logging in a trainer see no matrix
gradients in this mode.

The split is deliberately not a transaction, and cannot be made one at this
memory budget: phase one exists so that `G` is freed the moment it is consumed,
and holding it until `step()` is the cost the design avoids. So state commits
in two places: phase one commits the projected work and, on a first step, the
fitted frame; phase two commits the moment, the frame and the weight. Normal
completion of a step still requires `step()`, not `zero_grad()`. But a caller
that must bail out mid-step regardless (an OOM-retry loop, say) is allowed to:
`zero_grad()` drops any pending prepared update rather than rejecting the
call, with a warning naming how many. Nothing needs rolling back: neither the
moment nor the frame moves before `step()`, so what is dropped is the gradients
themselves, plus a frame fitted on a first step.

## Blocks

A group's matrices step in blocks: matrices of one side, shape, device and dtype,
taken in param order, a block closing once its members' batched work would pass
`BLOCK_BYTES` (256 MiB) or its members `BLOCK_MEMBERS` (65535: the CUDA write
launches a grid row per member, and that axis holds 65535). A member's work is about four fp32 copies of
its moment and frame, plus, off CUDA, its share of the eager write -- the stacked
weight and four fp32 temporaries at the write's peak, 18 bytes an entry for bf16;
the CUDA write holds none. A group forms its blocks at its first frame, from its
side and `rank_fraction` then, and a block allocates its buffers whole at its first
frame, so a member that never trains holds zeroed state.

A block holds each state entry (Persistent state) as one `[b, ...]` buffer, and each
member's state is a view into it, so phase two runs as batched calls on the buffers
themselves: a step's launch count follows its blocks, not its matrices. The weights
belong to the model: the CUDA write reaches them through their addresses, and the
eager write stacks them and copies them back in one call, a block of one in place.
A step that reaches only some members gathers their state and scatters it back; the
others keep their weights and state. A block steps the members that fitted their
frame this step apart from those that aim, since only the second move their frame.

Under Molegrad a fit made in the backward waits in the module's own tensor and enters
its block at `prepare()`: the block's frames share one buffer, and so one autograd
version counter, which a write during the backward would bump under another member's
saved frame.

A block steps each member as that member's own optimizer would; the batched kernels
sum in another order, so fp32 results agree to about `1e-5` relative, not bitwise.

## Molegrad

`Molegrad(model, optimizer)` is an opt-in path that never forms `G`.
Every tracked matrix must be the weight of a plain `nn.Linear`, bias allowed;
the weight is frozen for autograd, so no `dW` GEMM runs. With `X:[T,n]` the
Linear's input and `delta:[T,m]` the gradient at its output, `G = delta^T X`,
and the Linear's backward returns, as gradients of storage-free leaves, what
phase one would have read from `G`:

| | right | left |
|---|---|---|
| `Z` (step 3), exact | `delta^T (X Q)` | `X^T (delta Q)` |
| the aim, replacing step 4's `A` | the token diagonal, `X^T (||delta_t||^2 * X Q)` | `delta` alone, `delta^T (delta Q)` |
| `G P`, 16 Gaussian probes | `delta^T (X P)` | `delta^T (X P)` |

**The diagonal aim.** Step 4 aims at `G`'s own Gram, `G^T G = X^T (delta
delta^T) X` on the right; its cross-token terms need `G`, and the token
diagonal does not. Each token's row of `X` weighted by `||delta_t||^2` keeps
that diagonal, which is the selection the loss makes between tokens: rows past
the last supervised token have `delta_t = 0`, so it is also the mask. On the
left the aim is `delta^T delta Q`, which captures as much as `G`'s Gram; it is
`delta` unweighted, not the token diagonal of `G G^T`, which would weight each
row by `||x_t||^2`. That weighting tracks the same frame (`FACTS.md` on `lab`,
"The left aim's row weighting changes nothing"), so the left side does without
`X`'s row norms. On
LFM3 the two aims tie on loss through 1k steps (`FACTS.md` on `lab`, "S2 ties
`G` through 1k steps"); the `G` path keeps step 4's, since `G` alone does not
separate `X` from `delta`. Either is the action step 4 normalizes into its aim.

`||G||_F^2` is estimated as `||G P||_F^2 / 16`, unbiased; across a model's
matrices it reads `||G||` to within 1%, but 4-11% apart per matrix and step
(`FACTS.md` on `lab`, "Training without `dW`"). Capture over it is published
as `grad_capture_probe`, never as `grad_capture`: it can read above 1.
Like step 1's norm it is telemetry and the non-finite flag, and applied to
nothing. `Z` and the probes are read from the same `X` and `delta`, so a
non-finite stream reaches the estimate too, and the estimate alone is the flag.
All three leaves are token sums, so micro-batches accumulate exactly.

A matrix with no frame fits it in its first backward, formed and freed inside
the backward; under accumulation that is the first micro-batch's gradient, not
the step's, sanitized as in step 1. A held frame is read by the backward, not saved
at the forward, so a frame moved in place between the two is the one `Z` is taken in,
checkpointed or not. The range finder's Gaussian is drawn when
the waiting basis is bound, outside the backward, so a compiled backward traces
the fit whole. The fitting step has no aim. The
frame lives only in optimizer state, which the modules read again after every
load. `step()` collects the leaves itself; a closure would run its backward
after that, so it raises. `zero_grad()` and a load drop leaves no step
consumed, with the same warning as pending work.

No `weight.grad` ever exists for a tracked matrix: gradient clipping,
`GradScaler`, gradient-norm logging and anything else that reads it see only
the fallback parameters. A tracked weight used outside its Linear, as a
functional call, gets no gradient there and is not detected. The leaves are
placed with the weights when it is built, so the model has its device and
dtype by then. Refused with the reason, rather than trained wrong: a Linear
subclass (its own forward would be replaced), a weight shared with another
module, a tracked matrix that is not a Linear's weight, a Linear called more
than once in the backward that fits its frame (each call fits its own), and
more than one process (the leaves are not all-reduced).

## Parameter eligibility

`ndim == 2` is the structural gate `add_param_group` enforces, and it is not the
precondition the method has. UsuiTrack tracks a subspace of a **shared linear
map**: it assumes a weight's rows (or columns) are coordinates in one common
space, so a basis fitted from a few batches means something for all of them.
One family passes the shape test and breaks that assumption. The failure is not
numerical, so no guard in the update path can catch it; it was found as a run
going non-finite thousands of steps in.

**Lookup tables.** `nn.Embedding` weights, and any matrix whose rows are
independent per-token vectors. There is no shared map, and the gradient is
row-sparse: most rows go untouched for thousands of steps, while the basis only
ever sees the rows a batch lights up. Observed on a 32128x1024 vocabulary table: non-finite mid-run
with a finite gradient and a healthy loss, while the updates it did receive were
~3e-7 per element against weights of order 1e-2..1e2 -- below bf16 resolution,
so it carried the risk without training. Muon-lineage optimizers exclude
embedding tables for the same reason.

AdaLN/FiLM modulation linears are shared linear maps and are eligible, though
their output scales another layer's. Their input is one conditioning vector per
sample rather than per token, so a step's gradient has rank at most the batch
size.

Lookup tables are named consistently within an architecture and are not cheaply
detectable at runtime, so the caller names them. The integration owns that
policy; the library owns the structural rule.

## Fallback path

Callers own a separate optimizer for everything UsuiTrack does not take --
non-2D parameters and the excluded families above -- matching the standard Muon
split. The tested
reference configuration uses AdamW with fp32 first/second moments, LR `1e-4`,
`betas=(0.9,0.99)`, `eps=1e-8`, and zero weight decay; nothing about UsuiTrack
requires that specific choice. For bf16 models, `StochasticAdamW` is provided
as a drop-in: the fallback set holds the smallest-stepping weights in a model
(norm gains, biases), so leaving it on round-to-nearest cancels out what
stochastic rounding buys on the matrix side. It is not a memory optimization --
`torch.optim.AdamW` already stores moments in the parameter dtype -- it moves
the moment recurrences and the parameter update into fp32 accumulators that are
rounded once, instead of rounding on every in-place bf16 op. Sparse gradients
raise.

## Persistent state

| matrix state | right shape | left shape | dtype |
|---|---|---|---|
| `frame` | `[n,r]` | `[m,r]` | parameter dtype |
| `frame_moment` | `[n,r]` | `[m,r]` | parameter dtype |
| `frame_moment_span` | `[r,r]` | `[r,r]` | fp32 |
| `moment` | `[m,r]` | `[n,r]` | parameter dtype |

That is the whole per-matrix state, with `init_norm` (fp32, 0-dim) under norm bounds;
each entry is a view into its block's buffer (Blocks), and a load packs a checkpoint
into fresh blocks. A checkpoint without the frame moment starts it at zero. Projections and
aims are transient pending work. Fallback parameters retain AdamW's fp32 first moment, second
moment, and step tensor in their separate optimizer.


## Diagnostics

Optional, off by default, and inert when off: the optimizer calls no event and
takes no reduction beyond the raw gradient norm phase one always records. `diagnostics=True` installs a `Diagnostics`; drain
with `pop_diagnostics()`, which returns a plain dict of floats. Reads accumulate
on-device as 0-dim tensors; `pop` makes the one device-to-host transfer, so a
drained value is the mean over every matrix and step since the last drain, and
the intended cadence is the caller's logging cadence. `nonfinite_grads` is a
total, because a rare event averaged against a quiet window reads as zero.

No read makes its own pass over a full matrix. Each is a reduction of a
rank-sized tensor or of a norm the step already took: the update's norm is read
in frame coordinates, `||U|| = ||O|| S`, `S` the step rule's size, since the lift is an isometry, and
non-finite gradients are counted from the raw norm.

| key | what it measures |
|---|---|
| `relative_step` | `sqrt(sum ||alpha U||^2) / sqrt(sum ||W||^2)` over every matrix of the step, against the weights after it, before rounding. `lr` whenever every matrix stepped under the trust rule and every moment fills its rank (step 8); below it by the rank the moments leave empty. A `fixed` group contributes its shape-sized step over its own `||W||`, so the read is `lr` only by coincidence there. The update is read before a norm bound's clamp (step 8) |
| `grad_capture` | `||G_o Q|| / ||G||`: the share of the gradient's magnitude the frame catches. Reads the basis rather than the model |
| `grad_capture_probe` | under molegrad, `||G_o Q||` over the probe estimate of `||G||` in place of `grad_capture`: a guide for sizing `rank_fraction`, noisy per matrix and able to exceed 1 |
| `grad_moment_cosine` | this step's unit projected gradient against the moment *before* the blend, so the reading is not self-referential. Positive: the step under-travels and the gradient keeps its direction; negative: it overshoots |
| `projected_grad_norm` | `||G_o Q||`. The step is blind to it; bursts mark outlier batches the fallback optimizer is not blind to |
| `nonfinite_grads` | matrix gradients whose raw norm was non-finite and were sanitized. |

`usuitrack.diagnostics.READS` carries a one-line reading guide per key. A
subclass adds reads by overriding the event methods, which run once per block
step with the members' tensors stacked on the leading dimension; the lab's
`experiments/lab_diagnostics.py` is one, with frame-motion, persistence and
per-role reads.

## Decisions and reasons

These choices define the current design; they are redesignable.

1. **One-sided projection:** retains a rank-limited update without a square core
   or second basis.
2. **The input side by default, overridable:** one rule for every matrix, the
   side Molegrad's diagonal aim lives on; a param group sets `side` where the
   architecture knows better.
3. **Side-Gram range-finder initialization:** fits the one-sided target's
   leading eigenspace in fp32 from the sanitized gradient, with no
   eigendecomposition to fail on a nearly rank-one Gram.
4. **The frame as the polar factor of a moment of aims:** the frame follows
   the normalized covariance action through `Y` and `polar(Q + Y)` every step,
   anchored by itself, with no second basis and no host sync.
5. **Moving-frame momentum:** the overlap's rotation carries the projected
   moment's coordinates, and its spectrum, through the frame's move.
6. **No second moment on the full gradient:** the aim and the moment both
   read the sanitized gradient directly. A frame on the leading eigenspace of
   `G^T G` holds its aim inside its span, so the tracker
   converges when the gradient's principal subspace stops moving; any two-sided
   rescale is a congruence rather than a similarity and destroys that property.
7. **Average first, orthogonalize once, and no gradient clip:** the moment
   averages unit-norm projected gradients, so a batch's spectrum weights the
   average while its overall size never reaches it, and every consumer is
   scale-invariant. That retires the clip. The polar map runs once, on the
   moment, so the step is the expected gradient's direction at unit size. On
   LFM under the same decaying lr it ties averaging polar factors with a
   cautious mask on target (`FACTS.md`): the order is chosen for the estimate
   it makes, not for loss.
8. **Balanced polar direction, step relative to the weight:** direction belongs
   to projected geometry; size belongs to the weight it lands on, so `lr` is one
   relative unit across the fleet, and only the schedule changes it (see
   step 8).
9. **Separate AdamW fallback:** non-matrix tensors remain trainable under a
   separate optimizer, with their full state exposed rather than hidden in the
   matrix claim.

## Identities and edge behavior

| condition | consequence required by this design |
|---|---|
| gauge change `Q -> QH` with matching coordinate change | same ambient update |
| transpose problem and swap left/right | transposed projected and ambient update |
| the frame moment lies inside the current subspace, symmetric | zero frame motion |
| the frame moment leaves `k` directions outside the span | at most `k` planes move |
| full rank | projection/lift loses no component; balancing can still alter direction |
| any shape, rank or weight scale, every step agreeing, trust rule | `||U||_F / ||W||_F = 1`, so the relative step is `alpha` |
| gradient rescaling away from epsilon floors | aim, moment and update unchanged |
| frame move | stored moment carried by the overlap's rotation; its singular values unchanged |
