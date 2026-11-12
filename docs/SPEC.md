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
- **`side="auto"` knows shape, not architecture.** It cannot infer a transformer's
  residual-facing axis; a param group's `side` says it where it matters.
- **`ndim == 2` is a shape gate, not the precondition.** Lookup tables pass it
  and break the method, for reasons no numerical guard can see. See "Parameter
  eligibility".
- **The requirement is that frame motion can anneal; raw sigma is one mechanism
  for that, not the rule itself.** What the tracker owes is the ability to settle
  -- motion that falls as the frame approaches its equilibrium. Two specific
  scalar normalizations (by `sigma_max`, by the tangent's Frobenius norm) were
  tried and reverted for failing that. Do not read those two results as a ban on
  every reweighting of the tangent's spectrum, and do not read raw `sigma` as the
  goal.
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
cannot populate; half leaves an orthogonal complement for the Oja residual at
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

`side="auto"` tracks the shorter side, `right` for `m >= n`: a shape guess, not
an architectural one (see the trap above). A param group can fix `side` to
`left` or `right`; tracking the residual stream on every weight is one such
assignment, measured on LFM and Anima (`docs/NOTES.md` on `lab`).

Every `eigh`, Gram, and tangent computation upcasts fp16/bf16 inputs to
fp32 first. The basis is the one piece of state everything else is built
on; a half-precision eigendecomposition doesn't just cost this step's
accuracy, it corrupts the tracked frame for every step after it.

## One matrix step

```text
raw gradient G
  -> sanitize
  -> range-finder initialization on the first gradient, or a random frame
  -> held-frame projected gradient Z
  -> Rayleigh-normalized one-state Oja tangent
  -> release full gradient; retain rank-sized pending work
  -> unit Frobenius norm Z/||Z||_F
  -> projected EMA M of the unit gradient
  -> Aurora leverage balance of M (normalization)
  -> Newton-Schulz polar map of M (orthogonalization)
  -> unit scale 1/sqrt(r)
  -> lift through the held frame Q, scaled by ||W||_F
  -> parameter update
  -> exact Oja frame move
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
The Oja tangent is exactly invariant to a uniform rescale (step 4: `action` and
`rayleigh` are both quadratic in `G` and divide). The update normalizes each
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
its eigenbasis, and not exactly: the tracker turns it from the next step on.
The fitting step moves no frame; the moment and weight step as usual.

**The fit takes the first gradient as it is.** A gradient that is exactly zero
after step 1's sanitizing, which includes one that was entirely non-finite, has
no leading eigenspace, and the range finder returns an arbitrary orthonormal
frame from it (the QR of zero); its `Z` is zero, so the moment holds and the
weight holds but for its decay, and the tracker turns the frame from the next
step on. The optimizer's fit reads nothing back to the host, so it compiles with
the rest of the step; Molegrad's once-per-matrix check that the fit is the first
backward is the one read it keeps. A model trained
from a zero-initialized output (adaLN-zero) sees exactly such gradients upstream
until the output moves, and gains nothing from a fit: its param groups take
`"frame_init": "random"`, which draws each frame when the group is added, on the param's device, as the
QR of an `[n,r]` Gaussian from a private generator seeded once per optimizer and
device, so the draws follow the param order and the training run's random stream
never moves. The frame then tracks from step one at the usual `eta`; a
matrix whose gradient is still zero has a zero tangent and turns no plane.
`frame_init="range_finder"`, the default, is meant for finetuning, where the first gradient
already carries the subspace and a short run needs the frame on it from step one
(unmeasured).

### 3. Project in the held frame

Using the held frame before the current Oja move,

$$Z_t=\Pi_{Q_t}(G_t).$$

### 4. Track with one-state Oja

On every step after the fit, reuse the held-frame projection to form the
covariance action

$$A=G_o^\top Z,$$

and form the symmetrized Rayleigh matrix and horizontal tangent

$$R=\operatorname{sym}(Q^\top A),\qquad
\Delta=\frac{A-QR}{\operatorname{mean}(\operatorname{diag}R)}.$$

The denominator is floored at `1e-12`. With

$$\Delta^\top\Delta=V\operatorname{diag}(\sigma_i^2)V^\top,$$

the frame does **not** move along `Delta` itself. Two transforms sit between the
aim and the geodesic, and they are one design: every live plane turns by the
same constant angle, and the spectrum decides nothing about the motion.

**Polar tangent.** `Delta` is replaced by its polar factor

$$\operatorname{polar}(\Delta)=\Delta V\operatorname{diag}(1/\sigma_i)V^\top,$$

with numerically dead planes -- `sigma_i <= sqrt(r eps) sigma_max`, the Gram's
own backward error -- held at zero rather than divided by, preserving the
identity that a zero-singular plane does not move. The zero reaches the angle,
not merely the tangent column: a dead plane handed a live angle would contract
the frame's component along that eigenvector against a tangent term that is
zero, which is not a rotation. This is the same distrust of magnitude the weight update already applies:
Newton-Schulz discards the projected moment's singular values because direction
survives a noisy batch and magnitude does not, and the tangent's singular values
only decide how motion is divided between planes. It costs two `[r,r]` matmuls
and reuses the eigendecomposition the geodesic already needs.

**The turn is constant.** Every live plane turns by exactly `eta`, and nothing
scales it. Bare polar has no fixed point -- it turns by `eta` forever, whether or
not the frame has arrived -- and no measurement has ever supplied one. The
spectrum cannot: under this transform it is a smooth power law, `sigma^2 ~
k^{-1.5}` across four decades, with no edge separating signal from noise. It is
not scaled by time either; `docs/NOTES.md` on `lab` records the sensors that
were tried and why none is in.

$$\eta=0.02,$$

not scheduled. The frame is in effect an average of its aims over the last
`~1/eta` basis updates, and the aim -- the leading eigenspace of `G^T G` --
moves as the model does, fastest early. `eta` sets the trade between lag (a long
memory holds where the gradient was) and noise (a short one follows each
batch). The energy aim is a low-variance target, so lag costs only while it
moves: on LFM `0.02` and `0.04` tie on both losses at 1k, the smaller turn lags
early (capture 0.750 against 0.760 after warmup) and captures the same from
mid-run, and it moves the frame half as far per update, so it follows half as
much batch noise (`FACTS.md` on `lab`). Once the aim
settles the turn stops mattering -- a smaller late turn buys nothing -- so a
schedule would add a term and no gain. The initial fit already places the
frame near the first gradient's leading eigenspace rather than at random, so the
geodesic maintains a fit rather than searching for one. With `ell_i = 1` on
live planes and `0` on dead ones, the frame update is the exact geodesic along
the polar tangent:

$$Q_{raw}=\left[(QV)\operatorname{diag}(\cos(\eta\,\ell_i))
+(\Delta V)\operatorname{diag}
\left(\frac{\ell_i\sin\eta}{\sigma_i}\right)\right]V^\top.$$

A step's tangent-Gram eigendecompositions share one batched `eigh`: on a
small `[r,r]` matrix it is launch-overhead-bound, and each call reads its
status back to the host. Grams below the largest rank are padded with `-1` on
the padded diagonal, a separate block whose eigenpairs sort below the Gram's
(a scaled Gram is PSD) and are dropped. Nothing larger is stacked across
matrices. Each Gram is divided by its mean diagonal for the `eigh`, and
the eigenvalues are multiplied back: on a Gram of tiny entries, as behind a gate
still near zero, fp32 `eigh` on CUDA returns vectors that are not orthonormal,
and the turn would leave the manifold.

With `S=Q_raw^T Q_raw`, one near-identity step using
the converged steady-state coefficient triple from Amsel, Persson, Musco, and
Gower, ["The Polar Express"](https://arxiv.org/abs/2505.16932) (2025),
retracts before storage:

$$Q_+=Q_{raw}(aI+bS+cS^2).$$

The frame is stored round-to-nearest: it moves by a real angle every update,
so there is no sub-ulp step to lose, and stochastic rounding measured `sqrt(2)`
more in-span rotation. UsuiTrack stores no target frame or second tracker
state. A matrix with no gradient on a step is skipped: its frame, moment and
weight hold.

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
the Oja tangent, and the lift back to parameter space; the geodesic runs after
the parameter update, so `Q_{t+1}` is never used to lift an update that `Q_t`
measured. Ordering the geodesic first would rotate every step's own gradient by
however far the tracker had just turned -- a systematic error scaling with the
step size, and one that falls entirely on the current gradient rather than on
the accumulated history it is correct for.

The geodesic chooses an ambient rotation `R` with `Q_+ = RQ`. UsuiTrack defines
momentum as moving with that frame:

$$MQ^\top\mapsto MQ_+^\top.$$

Thus its stored coordinates and singular spectrum are unchanged:

$$M_+=M.$$

This is parallel transport along the selected lifted Oja path. Multiplication by the
old/new frame overlap would answer a different question: represent the surviving
projection of a fixed old ambient vector. It contracts each rotated plane by a
principal-angle cosine before the balanced polar map.

`M_+ = M` holds for the geodesic, which gives `Q^T Q_+ = V\cos\theta V^\top`,
exactly symmetric. It does not hold for the frame that is *stored*: rounding the
geodesic's result into bf16 rotates the frame's columns inside the span they
already had, and that rotation moves the subspace not at all while scrambling the
moment one-for-one. So the overlap is read back from the frames **as stored**,
its orthogonal polar factor `R_s` taken, and the moment corrected by it:

$$M_+=MR_s.$$

Only the rotation is applied, never the full overlap -- the symmetric part is the
principal-angle contraction above, which parallel transport is right to discard.
An exact step returns `R_s = I`, so the correction fires only on the deviation
between the frame computed and the frame kept.

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
2. read `Z_t` and, after initialization, form `Delta_t` in the held frame `Q_t`;
3. retain `Z_t` and the optional `Delta_t`, then release `G`.

Phase one does not move the basis, update the moment, or update the parameter:
the moment is rotated and committed after the frame moves (step 6), so it is
written only in `step()`. Phase two runs in `step()`: it blends each unit `Z_t`
into `M_t`, applies the balanced polar map to each `M_t`, lifts each `O_t`
through `Q_t`, updates `W`, then takes the tangent-Gram eigendecompositions,
batched by rank, and moves each frame to `Q_{t+}`.

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
| diagonal aim, replacing step 4's `A` | `X^T (||delta_t||^2 * X Q)` | `delta^T (delta Q)` |
| `G P`, 16 Gaussian probes | `delta^T (X P)` | `delta^T (X P)` |

**The diagonal aim.** Step 4 aims at `G`'s own Gram, `G^T G = X^T (delta
delta^T) X` on the right; its cross-token terms need `G`, and the token
diagonal does not. Each token's row of `X` weighted by `||delta_t||^2` keeps
that diagonal, which is the selection the loss makes between tokens: rows past
the last supervised token have `delta_t = 0`, so it is also the mask. On the
left the aim is `delta^T delta Q`, which captures as much as `G`'s Gram. On
LFM3 the two aims tie on loss through 1k steps (`FACTS.md` on `lab`, "S2 ties
`G` through 1k steps"); the `G` path keeps step 4's. The diagonal aim has
energy in every frame direction, so more planes are live and more of the frame
turns: 0.996 against 0.845 on LFM3.

`||G||_F^2` is estimated as `||G P||_F^2 / 16`, unbiased; across a model's
matrices it reads `||G||` to within 1% (`FACTS.md` on `lab`, "Training without
`dW`"), so `grad_capture` reads the same under molegrad as on the `G` path.
Like step 1's norm it is telemetry and the non-finite flag, and applied to
nothing. All three leaves are token sums, so micro-batches accumulate exactly.

A matrix with no frame fits it in its first backward, formed and freed inside
the backward; under accumulation that is the first micro-batch's gradient, not
the step's, sanitized as in step 1. The range finder's Gaussian is drawn when
the waiting basis is bound, outside the backward, so a compiled backward traces
the fit whole. The fitting step has no tangent. The
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
| `moment` | `[m,r]` | `[n,r]` | parameter dtype |

That is the whole per-matrix state; the side is resolved from the group each
step. Projections and Oja tangents are transient pending work. Fallback parameters retain AdamW's fp32 first moment, second
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
| `relative_step` | `sqrt(sum ||alpha U||^2) / sqrt(sum ||W||^2)` over every matrix of the step, against the weights after it, before rounding. `lr` whenever every matrix stepped under the trust rule and every moment fills its rank (step 8); below it by the rank the moments leave empty. A `fixed` group contributes its shape-sized step over its own `||W||`, so the read is `lr` only by coincidence there |
| `grad_capture` | `||G_o Q|| / ||G||`: the share of the gradient's magnitude the frame catches. Reads the basis rather than the model |
| `grad_moment_cosine` | this step's unit projected gradient against the moment *before* the blend, so the reading is not self-referential. Positive: the step under-travels and the gradient keeps its direction; negative: it overshoots |
| `tangent_live_fraction` | the share of planes with `sigma_i > sqrt(r eps) sigma_max` (step 4). Below `1` the aim is rank-collapsed against the rank it was given: the rank is too large for what this batch size resolves |
| `projected_grad_norm` | `||G_o Q||`. The step is blind to it; bursts mark outlier batches the fallback optimizer is not blind to |
| `nonfinite_grads` | matrix gradients whose raw norm was non-finite and were sanitized. |

`usuitrack.diagnostics.READS` carries a one-line reading guide per key. A
subclass adds reads by overriding the event methods; the lab's
`experiments/lab_diagnostics.py` is one, with frame-motion, spectrum and
per-role reads.

## Decisions and reasons

These choices define the current design; they are redesignable.

1. **One-sided projection:** retains a rank-limited update without a square core
   or second basis.
2. **Side by shape, overridable:** `auto` tracks the shorter side; a param
   group fixes `side` where the architecture knows better.
3. **Side-Gram range-finder initialization:** fits the one-sided target's
   leading eigenspace in fp32 from the sanitized gradient, with no
   eigendecomposition to fail on a nearly rank-one Gram.
4. **One-state full-gradient basis tracking:** the live frame follows the Oja
   covariance action on every step, with a constant geodesic step
   `eta = 0.02` and no second basis.
5. **Moving-frame momentum:** identity coordinates preserve the projected
   moment's spectrum through the chosen frame rotation.
6. **No second moment on the full gradient:** the tangent and the moment both
   read the sanitized gradient directly. A frame on the leading eigenspace of
   `G^T G` has zero Oja residual, so the tracker
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
| Oja covariance action lies inside the current subspace | zero frame motion |
| Oja tangent is rank-deficient | zero singular planes remain fixed |
| full rank | projection/lift loses no component; balancing can still alter direction |
| any shape, rank or weight scale, every step agreeing, trust rule | `||U||_F / ||W||_F = 1`, so the relative step is `alpha` |
| gradient rescaling away from epsilon floors | Oja tangent, moment and update unchanged |
| geodesic frame rotation | stored moment coordinates and singular values unchanged, up to the stored-frame rotation correction |
