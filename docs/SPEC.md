# UsuiTrack specification

Current matrix-update design. This document specifies only the selected
basis-tracking path; see the [README](../README.md) for results and usage.

## Read these traps first

- **Basis != subspace.** `Q` and `QH` span the same subspace, but coordinates in
  those frames differ.
- **Left != right with renamed shapes.** Derive each projection and lift.
- **Overlap reprojection != parallel transport.** Reprojection preserves the
  least-squares part of a fixed ambient vector. UsuiTrack carries momentum with
  its moving frame, so its stored coordinates do not change at refresh.
- **A healthy polar output can hide a sick input.** Newton--Schulz restores
  semi-orthogonal scale after weak directions have already become noise-dominated.
- **A downstream operation cannot protect upstream state.** Raw clipping must
  precede held-frame projection and Oja tangent construction.
- **`side="auto"` knows shape, not architecture.** It cannot infer a transformer's
  residual-facing axis.
- **`ndim == 2` is a shape gate, not the precondition.** Lookup tables and
  multiplicative gates pass it and break the method, for reasons no numerical
  guard can see. See "Parameter eligibility".
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

The configured rank is a ceiling, not a promise. Each matrix runs at

$$r_{\text{eff}}=\min\!\left(r,\ \max\!\left(1,\ \lfloor\min(m,n)/2\rfloor\right)\right).$$

Two structural reasons, neither of them numerical. A gradient of shape `[m,n]`
has rank at most `min(m,n)`, so a wider basis tracks directions the gradient
cannot populate; half leaves an orthogonal complement for the Oja residual at
every step. And narrow modules are bottlenecks -- they carry the residual
stream through a small waist, so a large update there destabilizes every block
downstream, independently of how well-conditioned the step was.

Observed on Anima's `(2048, 64)` input and output projections: at rank 64 on
their 64-wide data side -- full rank, 100% occupancy -- training errored. Two
things were wrong at once and both fixes were real. The `side` hint moved them
onto the 2048-wide residual stream, which is where their basis belongs and which
brought occupancy from 25% into line with every other module at ~3%. And the
rank still has to be capped, because those modules are bottlenecks: too much
change in them destabilized the model regardless of which side was tracked.

Rank itself is a configured hyperparameter picked by parameter size, the same
way a LoRA rank is; this is a per-parameter cap on that choice. A model trained
at rank 256 runs a `(2048, 64)` module at 32. Exceeding the cap is not an
error, but UsuiTrack reports it once at startup rather than silently giving the
caller something other than what was asked for.

**Migration:** this cap changed stored basis shapes. Checkpoints written before
it will fail the basis-shape check on resume.

`Q` is always the canonical column frame, `Q^T Q = I`:

| side | canonical frame | stored basis | project $\Pi_Q(G)$ | lift $\Lambda_Q(Z)$ |
|---|---|---|---|---|
| right | $Q:[n,r]$ | $Q^\top:[r,n]$ | $GQ:[m,r]$ | $ZQ^\top:[m,n]$ |
| left | $Q:[m,r]$ | $Q:[m,r]$ | $Q^\top G:[r,n]$ | $QZ:[m,n]$ |

`auto` chooses right for `m >= n`, otherwise left: a shape guess, not an
architectural one (see the trap above). On transformer weights, set `side`
explicitly per parameter group instead. The rule is **track the residual
stream**: weights that *read* the stream (q/k/v, the feed-forward
up/gate-projection, the patch or token embedder's output) track their input
side, `right`; weights that *write* it (the attention out-projection, the
feed-forward down-projection, the final projection) track their output side,
`left`. That puts every tracker's basis in the one space the whole network
shares, where the gradient really is low-rank, rather than in a per-layer data
space that happens to be narrow. This measurably beat shape-only `auto` and is
the recommended production setting; decision 2 below has the reasoning.

The difference concentrates at the edges of the network, where the data side is
narrow. Shape-only `auto` tracked the 68- and 64-wide data sides of a model's
input and output projections and produced its two worst-conditioned bases
(orthonormality error 1.1e-03 and 1.0e-03 against ~5e-04 elsewhere); the hint
spends the same rank budget inside the 2048-wide stream, where the complement
the Oja tangent needs is ample.

Every `eigh`, Gram, and tangent computation upcasts fp16/bf16 inputs to
fp32 first. The basis is the one piece of state everything else is built
on; a half-precision eigendecomposition doesn't just cost this step's
accuracy, it corrupts the tracked frame for every step after it.

## One matrix step

```text
raw gradient G
  -> sanitize and raw clip
  -> stable EIGH initialization on the first step
  -> held-frame projected gradient Z
  -> Rayleigh-normalized one-state Oja tangent
  -> projected EMA M
  -> release full gradient; retain rank-sized pending work
  -> exact Oja frame move with identity moment coordinates
  -> Aurora leverage balance + Newton-Schulz polar map
  -> full-parameter Muon scale
  -> lift through the moved frame Q+
  -> parameter update
```

### 1. Sanitize and raw clip

Let `S(G)` replace each non-finite entry with zero. With threshold `c=1` by
default,

$$G_c=S(G)\min\left(1,\frac{c}{\max(\|S(G)\|_F,10^{-12})}\right).$$

This occurs before all matrix consumers, and there is no off switch: `c` has no
`None`. Clipping the raw gradient protects the projected moment, which is linear
in `G` and accumulates, and a guard installed later cannot protect an earlier
memory. It does **not** protect the frame: the Oja tangent is exactly invariant
to a uniform rescale of the gradient (see step 4), so the clip changes basis
motion by nothing at all. A single blip batch (grad norm spiking
~180x) otherwise reaches both at full size, and a frame that has turned toward
one is wrong for as long as it takes the tracker to turn back.

Making it mandatory also collapses the update to a single implementation.
Once `c` is always present, the only thing that varies between steps is whether
the frame exists yet and whether a basis update is due, so the tangent is built
in exactly one place: the fused per-side kernel. There is no second readable
copy to drift out of sync with it -- that is what this document is for.

### 2. Initialize the frame

Normalize `A = G_c / ||G_c||_F` and form the symmetrized side Gram:

$$K=A^\top A\quad\text{(right)},\qquad K=AA^\top\quad\text{(left)}.$$

`Q` is the top-`r` eigenvector frame. On backend failure, retry `eigh` with

$$K\leftarrow K+10^{-6}\max(\operatorname{tr}(K)/d,10^{-12})I.$$

An exactly zero gradient produces the deterministic EIGH frame of the zero side
Gram.

### 3. Project in the held frame

Using the held frame before the current Oja move,

$$Z_t=\Pi_{Q_t}(G_{c,t}).$$

### 4. Track with one-state Oja

After EIGH initialization, the live frame updates on the configured basis-update
cadence (default every full gradient). Reuse the held-frame projection to form
the covariance action:

$$A=G_c^\top Z\quad\text{(right)},\qquad
A=G_cZ^\top\quad\text{(left)},$$

and form the symmetrized Rayleigh matrix and horizontal tangent

$$R=\operatorname{sym}(Q^\top A),\qquad
\Delta=\frac{A-QR}{\operatorname{mean}(\operatorname{diag}R)}.$$

The denominator is floored at `1e-12`. With

$$\Delta^\top\Delta=V\operatorname{diag}(\sigma_i^2)V^\top,$$

the frame does **not** move along `Delta` itself. Two transforms sit between the
aim and the geodesic, and they are one design: every plane turns by the same
angle, and how large that angle is comes from time rather than from the spectrum.

**Polar tangent.** `Delta` is replaced by its polar factor

$$\operatorname{polar}(\Delta)=\Delta V\operatorname{diag}(1/\sigma_i)V^\top,$$

with numerically dead planes (`sigma_i <= 1e-6 sigma_max`) held at zero rather
than divided by, preserving the identity that a zero-singular plane does not
move. This is the same distrust of magnitude the weight update already applies:
Newton-Schulz discards the projected moment's singular values because direction
survives a noisy batch and magnitude does not, and the tangent's singular values
only decide how motion is divided between planes. It costs two `[r,r]` matmuls
and reuses the eigendecomposition the geodesic already needs.

**Agreement annealing.** Bare polar has no fixed point -- it turns by `eta`
forever -- and the spectrum cannot supply one: measured under this transform it
is a smooth power law, `sigma^2 ~ k^{-1.5}` across four decades, with no edge
separating signal from noise. The time axis can. With `H_t` the leading
`k = min(16, r)` normalized plane directions of `Delta_t`,

$$a_t=\frac{\lVert H_t^\top H_{t-1}\rVert_F^2}{k},\qquad
s_t=\operatorname{clamp}\!\left(\frac{a_t-k/(d-r)}{G_t},\,0,\,1\right),$$

and the geodesic runs on `s_t polar(Delta)`, so every live plane turns by exactly
`eta s_t`. `a_t` is the mean squared cosine of the principal angles between
consecutive top-`k` aims: it asks whether a `k`-dimensional aim persists and
forgives rotation inside it. A skewed frame re-measures its own lag every step
and reads high; an aligned frame emits uncorrelated batch noise and reads low.

Neither anchor is fitted. `k/(d-r)` is the agreement of two random `k`-subspaces
of the horizontal complement, so an aim agreeing only by chance stops the frame.
The divisor `G_t` is the fleet median of `(\sum\lambda)^2/(\sum\lambda^2 k)` --
each matrix's own effective aim rank over the meter width -- computed every step
from eigenvalues already in hand, with a one-step lag so the median spans the
whole model rather than one bucket. It is a fleet quantity because per matrix it
does not work: a matrix's own effective rank predicts its own attainable
agreement with a 3.6x spread, worse than no per-matrix term, while the fleet
median lands within 6% of the fleet median of the ceilings observed. Nothing is
remembered; the attainable ceiling rises ~47% over a run as the aim spreads over
more planes, so any frozen anchor describes a spectrum the model has left.

`s_t <= 1` is structural, so the frame can never turn harder than bare polar at
the same `eta` -- a bound, not a measurement. With no stored aim the scale is
zero and the frame is held for one basis update.

$$\eta=0.01,$$

not scheduled. A `max(0.01, 1/t)` anneal previously sat here, answering a problem
that no longer exists: while an upstream factored second moment warmed up, the
Gram whose eigenspace the tracker targets was itself shifting, so the frame
chased a moving target and a hot start was the correct compensation. The target
is now the leading eigenspace of `G_c^T G_c` from the first step and moves only
as the model does. Removing the schedule also makes the tracker observable: with
a constant step and a measured `transport_speed`, frame motion is one annealing
term rather than the product of two, so "the tracker settled" is separable from
"the clock ran out". EIGH initialization already places the frame on the first
gradient's leading eigenspace rather than at random, so the geodesic maintains a
fit rather than searching for one. With `basis_update_interval=k`, phase one
still runs for every matrix gradient while geodesics occur only on matrix steps
divisible by `k`. The frame update is

$$Q_{raw}=\left[(QV)\operatorname{diag}(\cos(\eta\sigma_i))
+(\Delta V)\operatorname{diag}
\left(\frac{\sin(\eta\sigma_i)}{\sigma_i}\right)\right]V^\top,$$

which is exact for whatever horizontal tangent it is handed; the `sigma_i` it
sees are the per-plane turns already decided above, not the aim's raw spectrum.
The zero-singular-value limit is `sin(eta sigma) / sigma -> eta`.

**Raw `sigma` is a contrast ratio, not a magnitude, and it is no longer the
step.** Both `A` and `R` are quadratic in the gradient, so the division by
`mean(diag R)` cancels gradient scale exactly: rescaling `G_c` by any constant
leaves `Delta` and every `sigma_i` bit-identical (verified across a 1000x range).
What `sigma` measures is out-of-frame coupling against mean in-frame energy per
plane. It is scale-free in the gradient and unaffected by `grad_clip_norm`, but
*not* rank-free -- a nuclear norm over `r` planes, growing roughly linearly in
`r` (~37.6 at rank 64 against ~80 at rank 128 on the same problem). Driving the
frame with it directly was the released rule until the controller above replaced
it, and the reason is that its self-annealing is an acquisition transient only:
`sigma` falls sharply over roughly twenty steps, then flattens and declines a few
percent across the remaining ~900. Rescaling it by a constant (`sigma_max`, the
tangent's Frobenius norm) cannot fix that, because a ratio cannot shrink as its
numerator does; capping the angle fails at the other end, costing acquisition.
Both were tried and reverted, and neither is what the agreement controller does:
it reads a different axis entirely.

Equal-rank tangent-Gram eigendecompositions are batched. With
`S=Q_raw^T Q_raw`, one near-identity step using the converged steady-state
coefficient triple from Amsel, Persson, Musco, and Gower, ["The Polar
Express"](https://arxiv.org/abs/2505.16932) (2025), retracts before storage:

$$Q_+=Q_{raw}(aI+bS+cS^2).$$

The Oja tangent rotates every tracked plane. UsuiTrack stores no target frame or
second tracker state and requires a full matrix gradient on every step.

### 5. Accumulate momentum

$$M_t=\beta M_{t-1}+(1-\beta)Z_t,\qquad \beta=0.9.$$

There is no EMA bias correction.

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

$$MQ^\top\mapsto MQ_+^\top\quad\text{(right)},$$
$$QM\mapsto Q_+M\quad\text{(left)}.$$

Thus its stored coordinates and singular spectrum are unchanged:

$$M_+=M.$$

This is parallel transport along the selected lifted Oja path. Multiplication by the
old/new frame overlap would answer a different question: represent the surviving
projection of a fixed old ambient vector. It contracts each rotated plane by a
principal-angle cosine before Aurora.

### 7. Aurora direction

The leverage-balancing scheme below is
[Aurora](https://github.com/tilde-research/aurora-release) (Tilde Research).
Reimplemented here from the method, not a runtime dependency; UsuiTrack takes
only the rectangular direction map, per the decision note at the end of this
section.

Aurora acts only on `M`. For a rectangular tensor, orient it as
`A:[p,q]`, `p >= q`, transposing if needed. Initialize row scaling

$$D_{0,ii}=1/\|A_{i,:}\|_2.$$

Run one leverage-balancing/polar iteration:

$$P_k=\operatorname{NS}(D_kA),$$
$$D_{k+1,ii}=D_{k,ii}
\left(\frac{q/p}{\|P_{k,i,:}\|_2^2}\right)^{1/2}$$

with the diagonal update omitted after the iteration. Transpose back. Square
tensors skip leverage balancing and use `NS(M)` directly.

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
exact SVD polar factor.

**Decision (projected Aurora):** Aurora chooses direction inside the retained
update space. UsuiTrack, not Aurora, owns momentum, basis motion, scale, and LR.

### 8. Scale, lift, and update

Muon scale uses the original parameter shape, not the projected shape:

$$\widehat U_t=O_t\sqrt{\max(1,m/n)},
U_t=\Lambda_{Q_{t+}}(\widehat U_t).$$

The selected contract has zero weight decay. Apply the learning rate:

$$W_t=W_{t-1}-\alpha U_t.$$

Matrix parameters retain no full-size first or second moment.

**Rounding.** For bf16 `W`, the lift and the update are accumulated in fp32 and
written back once through stochastic rounding (`stochastic_rounding=True`,
default). Orthogonalization means `alpha` alone sets the step size, so a
well-conditioned run puts `alpha * U` well under a bf16 ulp -- measured at
~1/76 ulp on a 2B DiT at `alpha=2e-5` -- and round-to-nearest then discards
every step. `U` is never cast to `W`'s dtype on the way in; a single rounding
happens at the write, with weight decay folded into the same accumulator so it
is not rounded twice. See `usuitrack/stochastic.py`.

## Two-phase execution

For each matrix, phase one runs exactly once after its full gradient is complete:

1. sanitize and clip `G` (always);
2. form the clipped gradient `G_c`;
3. read `Z_t` and, after initialization, form `Delta_t` in the held frame `Q_t`;
4. update `M_t`;
5. retain `M_t`, the optional `Delta_t`, and the frame reference, then release `G`.

Phase one does not move the basis or update the parameter. Phase two runs in
`step()`: it batches tangent-Gram eigendecompositions, moves each frame to
`Q_{t+}`, applies Aurora to `M_t`, lifts through `Q_{t+}`, and updates `W`.

`prepare(param)` exposes phase one explicitly. A no-accumulation training loop
can invoke it from a `register_post_accumulate_grad_hook` callback to release
each full matrix gradient as soon as it is consumed (`release_matrix_grads=True`
at construction).

The split is deliberately not a transaction, and cannot be made one at this
memory budget. Phase one exists so that `G` can be freed the moment it is
consumed; deferring the moment update to `step()` would mean
holding a matrix-sized gradient per parameter until then, which is
the cost the two-phase design exists to avoid. So state commits in two places:
phase one commits the moving averages, phase two commits the weight. Normal
completion of a step still requires `step()`, not `zero_grad()`. But a caller
that must bail out mid-step regardless (an OOM-retry loop, say) is allowed to:
`zero_grad()` drops any pending prepared update rather than rejecting the
call. This does not roll back the moving-average state prepare() already
advanced -- that one step's contribution to projected-moment state
is unrecoverable -- it only clears the retained tangent so the next
`prepare()`/`step()` starts clean. Repeated preparation, gradient accumulation
before `step()`, and optimizer closures with pending work remain unsupported.

## Parameter eligibility

`ndim == 2` is the structural gate `add_param_group` enforces, and it is not the
precondition the method has. UsuiTrack tracks a subspace of a **shared linear
map**: it assumes a weight's rows (or columns) are coordinates in one common
space, so a basis fitted from a few batches means something for all of them. Two
families pass the shape test and break that assumption. Neither failure is
numerical, so no guard in the update path can catch them; both were found as a
run going non-finite thousands of steps in.

**Lookup tables.** `nn.Embedding` weights, and any matrix whose rows are
independent per-token vectors. There is no shared map, and the gradient is
row-sparse: most rows go untouched for thousands of steps, while the basis only
ever sees the rows a batch lights up. Observed on a 32128x1024 vocabulary table: non-finite mid-run
with a finite gradient and a healthy loss, while the updates it did receive were
~3e-7 per element against weights of order 1e-2..1e2 -- below bf16 resolution,
so it carried the risk without training. Muon-lineage optimizers exclude
embedding tables for the same reason.

**Multiplicative gates.** AdaLN/FiLM modulation linears, and anything whose
output scales or shifts another layer's output instead of feeding forward. These
are ordinary, well-conditioned matrices; the problem is that a tracking error in
a gate is multiplied through everything downstream instead of staying local.
Observed on a `(6144, 256)` AdaLN modulation linear, clear of any rank or shape
degeneracy: non-finite one step into training.

Both families are named consistently within an architecture and are not cheaply
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
rounded once, instead of rounding on every in-place bf16 op. `optimizer_state_bytes_by_category` accounts
matrix and fallback state separately when both optimizers are inspected.
Sparse gradients are unsupported.

## Persistent state

| matrix state | right shape | left shape | dtype |
|---|---|---|---|
| basis | `[r,n]` | `[m,r]` | parameter dtype |
| projected EMA | `[m,r]` | `[r,n]` | gradient dtype |
| update counts and resolved side | scalar | scalar | Python |

Oja tangents are transient pending work. No target frame, second basis, or lag
snapshot is stored. Fallback parameters retain AdamW's fp32 first moment, second
moment, and step tensor in their separate optimizer.


## Diagnostics

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

| key | what it measures |
|---|---|
| `transport_speed` | chordal distance the subspace moved in one geodesic, per plane, measured from the frames before and after: `||Q_now - Q_old (Q_old^T Q_now)||_F / sqrt(r)`. Every motion metric below shares this unit, so they can be divided by one another |
| `tangent_concentration` | `lambda_max / sum_i lambda_i` of the tangent Gram, in `[1/r, 1]`: the leading direction's share of the aim |
| `turn_fraction` | the controller's own output: mean turn scale in `[0,1]`, so a logged point says how much of `eta` the frame is actually taking. Without it an `eta` ladder is blind, since `eta` and the scale multiply |
| `agreement_ceiling` | the fleet divisor `G`, one scalar per step. Rises as `tangent_participation` rises; a flat or collapsing gain means the controller has stopped tracking the aim's spread |
| `tangent_participation` | `(sum_i lambda_i)^2 / (r sum_i lambda_i^2)`, in `[1/r, 1]`: the effective number of planes carrying the aim, as a fraction of `r`. The bulk of the same spectrum concentration reads the head of |
| `projected_grad_norm` | norm of the clipped gradient inside the held frame |
| `grad_to_moment_ratio` | that norm against the projected moment *after* this step's update: `1/(1-beta)` on the first step, lower once the moment has history |
| `update_to_param_ratio` | mean per-step weight motion against current weight norm, over matrix parameters only |
| `nonfinite_grads` | matrix gradients that arrived non-finite and were sanitized |
| `transport_lag` | net distance a sampled frame covered over `diagnostics_lag_interval` basis updates, same unit as the speed. Also the projected moment's smear: set the interval to `1/(1-beta)` and it reads how far the moment's own history has drifted from the coordinates it was accumulated in |
| `transport_curve` | `1 - lag / path`: the fraction of that window's travel which cancelled. Zero is a straight drift, approaching one is a frame churning in place |
| `transport_spin` | `||skew(Q_old^T Q_now)||_F / sqrt(r)`: rotation of the frame's columns *within* their own span. Moves the subspace not at all; scrambles the projected moment one-for-one, because transport is the identity in these coordinates |

`transport_lag`, `transport_curve` and `transport_spin` need the frame snapshot
and appear only under `diagnostics = "full"`; the speed does not, since both
frames are already in hand when the geodesic runs. Read the four motion reads as a set. High speed with
low curve is a frame travelling, and it will slow as the aim converges. High
curve with low speed is a frame sitting on its fixed point. High speed *and*
high curve is churn: the tracker working hard, going nowhere, and integrating
batch noise into the frame while it does. Low speed with low curve is ambiguous
between settled and starved, and spin separates them -- a settled frame is still
in every sense, a spinning one is renaming the moment's coordinates underneath
it. A single Grassmann geodesic along a horizontal tangent has `Q_old^T Q_now =
V cos(theta) V^T`, exactly symmetric, so spin is zero for one ideal step and
what accumulates over a window is holonomy plus retraction and rounding error.

`transport_speed` is measured from the written frames, not from the eigenvalues
the geodesic was handed. Those agree only while nothing stands between the aim
and the frame, and they are not the same quantity the moment anything transforms
the geodesic -- an orthogonalized tangent turns every live plane by `eta`
whatever `sigma` said. `tangent_concentration` and `tangent_participation` do
come free from those eigenvalues. Read speed and concentration together: the same
speed is a confident drift when concentration is high and a frame spinning on its
noise tail when it is low.

The tangent Gram is decomposed bare. There is no jitter and no retry: a failing
`eigh` fails, because its result steers the frame and a silently rescued
decomposition is worse than a stopped run. The initial fit in `_side_gram_eigh`
keeps its try-then-jitter fallback -- different matrix, once per parameter rather
than once per step.


## Decisions and reasons

These choices define the current design; they are redesignable.

1. **One-sided projection:** retains a rank-limited update without a square core
   or second basis.
2. **Residual-facing side policy:** up/gate and q/k/v use storage-right; down
   and attention-output use storage-left. Generic `auto` remains shape-only.
3. **Side-Gram `eigh` initialization:** directly solves the one-sided target and
   has explicit fp32, finite-input, symmetrization, and jitter behavior.
4. **One-state full-gradient basis tracking:** the live frame follows the Oja
   covariance action on its configured cadence, with a constant geodesic step
   `eta = 0.01` and no second basis.
5. **Moving-frame momentum:** identity coordinates preserve the projected
   moment's spectrum through the chosen frame rotation.
6. **No second moment on the full gradient:** the tangent and the moment both
   read the clipped gradient directly. A frame fitted on `G` is the leading
   eigenspace of `G^T G`, so the Oja residual is zero there and the tracker
   converges when the gradient's principal subspace stops moving; any two-sided
   rescale is a congruence rather than a similarity and destroys that property.
7. **Raw clipping before all consumers:** bounds what one batch can write into
   the projected moment. It has no effect on the frame, which is scale-invariant
   by construction.
8. **Aurora plus full-shape Muon scale:** direction belongs to projected geometry;
   scale remains tied to parameter geometry.
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
| full rank | projection/lift loses no component; Aurora can still alter direction |
| gradient rescaling below raw clipping and away from epsilon floors | Rayleigh-normalized Oja tangent unchanged; pre-Aurora magnitude follows the stated conditioning |
| geodesic frame rotation | stored moment coordinates and singular values unchanged |
