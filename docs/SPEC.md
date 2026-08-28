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
  precede Adafactor, held-frame projection, and Oja tangent construction.
- **`side="auto"` knows shape, not architecture.** It cannot infer a transformer's
  residual-facing axis.
- **`ndim == 2` is a shape gate, not the precondition.** Lookup tables and
  multiplicative gates pass it and break the method, for reasons no numerical
  guard can see. See "Parameter eligibility".
- **Bound the rotation, not the tangent.** Every quantity feeding the frame
  update is a proxy for one thing: how far the basis turned. Guard that, in
  radians, where the tangent producers converge -- not each proxy where it
  happens to be convenient.
- **Rounding that rescues an accumulator ruins a converger.** `W` should
  integrate small steps and wants stochastic rounding; `Q` should settle and
  wants precision. Applying the first fix to the second makes it worse.
- **The Oja tangent is written twice.** Once readably on `SubspaceProjector`,
  once inlined into the fused per-side kernels, and only the kernels run under
  the default configuration. Changes go in both; `tests/test_oja_consistency.py`
  is what keeps them honest.
- **Exact implementation of a formula does not validate the formula's premise.**

## Coordinates

Let a matrix parameter and gradient be

$$W,G\in\mathbb R^{m\times n},\qquad 1\le r\le\min(m,n).$$

The configured rank is a ceiling, not a promise. Each matrix resolves its own

$$r_{\text{eff}}=\min\!\left(r,\ \max\!\left(1,\ \operatorname{round}(d/4)\right)\right),$$

where `d` is the *tracked* dimension -- the width of the space the basis lives
in, which is `n` for a right-side tracker and `m` for a left-side one, not
`min(m, n)`. The quarter is a contrast requirement: Oja rotates the frame using
the part of the gradient lying *outside* it, so a basis that spans most of its
ambient space drives itself on numerical noise. Keeping three quarters of the
space in the complement guarantees there is always real signal there. At exactly
full rank this produced non-finite loss within a couple of steps; at a
half-dimension cap, loss spikes.

Measuring against the tracked side rather than `min(m, n)` matters only where a
side hint deliberately tracks the larger dimension -- and that is precisely
where `min(m, n)` was the wrong yardstick. A `(2048, 68)` input projection
tracking its 2048-wide output space was being clamped to rank 17 by its 68-wide
data side, a space its basis does not live in. For square-ish weights and for
`auto` (which always picks the smaller side) the two coincide, so LLM-shaped
layers are unaffected: a 1024x4096 MLP still resolves to rank 256.

A configured rank above this bound is not an error. UsuiTrack warns once per
parameter group, naming how many parameters were clamped and to what, and
proceeds. **Migration:** this bound changed the shape of stored bases;
checkpoints written before it will fail the basis-shape check on resume.

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
  -> Adafactor SNR conditioning
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

This occurs before all matrix consumers. Disabling the threshold leaves only
sanitization.

### 2. Adafactor SNR conditioning

The selected path maintains row and column means of squared
full-gradient entries (`beta2=0.99`, `epsilon_a=1e-30`):

$$R_t=\beta_2R_{t-1}+(1-\beta_2)\operatorname{mean}_j(G_{c,ij}^2+\epsilon_a),$$
$$C_t=\beta_2C_{t-1}+(1-\beta_2)\operatorname{mean}_i(G_{c,ij}^2+\epsilon_a).$$

Bias-correct both, then reconstruct

$$\widehat V_{ij}=\frac{\widehat R_i\widehat C_j}
{\operatorname{mean}(\widehat R)}.$$

The gradient consumed by tracking and projection is

$$\widetilde G=\frac{G_c}{\sqrt{\widehat V}}\operatorname{RMS}(G_c).$$

RMS restoration retains relative SNR weighting without feeding an RMS-one matrix
into the tracker. Reconstruction denominators and square roots are floored by
`epsilon_a`.

### 3. Initialize the frame

Normalize `A = G_tilde / ||G_tilde||_F` and form the symmetrized side Gram:

$$K=A^\top A\quad\text{(right)},\qquad K=AA^\top\quad\text{(left)}.$$

`Q` is the top-`r` eigenvector frame. On backend failure, retry `eigh` with

$$K\leftarrow K+10^{-6}\max(\operatorname{tr}(K)/d,10^{-12})I.$$

An exactly zero gradient produces the deterministic EIGH frame of the zero side
Gram.

### 4. Project in the held frame

Using the held frame before the current Oja move,

$$Z_t=\Pi_{Q_t}(\widetilde G_t).$$

### 5. Track with one-state Oja

After EIGH initialization, the live frame updates on the configured basis-update
cadence (default every conditioned full gradient). Reuse the held-frame
projection to form the covariance action:

$$A=\widetilde G^\top Z\quad\text{(right)},\qquad
A=\widetilde GZ^\top\quad\text{(left)},$$

and form the symmetrized Rayleigh matrix and horizontal tangent

$$R=\operatorname{sym}(Q^\top A),\qquad
\Delta=\frac{A-QR}{\operatorname{mean}(\operatorname{diag}R)}.$$

The denominator is floored at `1e-12`. With

$$\Delta^\top\Delta=V\operatorname{diag}(\sigma_i^2)V^\top,$$

the exact full-rank Grassmann step uses

$$\eta_t=\max(0.01, 1/t),$$

where `t` counts basis updates, including EIGH initialization. Thus the first
geodesic uses `1/2`. With `basis_update_interval=k`, phase one still runs for
every matrix gradient while geodesics occur only on matrix steps divisible by
`k`; `t` still counts only basis updates. The released harmonic schedule reaches
its steady `.01` floor at basis update 100. The frame update is

$$Q_{raw}=\left[(QV)\operatorname{diag}(\cos\theta_i)
+(\Delta V)\operatorname{diag}
\left(\frac{\sin\theta_i}{\sigma_i}\right)\right]V^\top,
\qquad
\theta_i=\min(\eta_t\sigma_i,\ \theta_{\max}),\quad
\theta_{\max}=\tfrac{\pi}{4}.$$

The zero-singular-value limit is `sin(theta) / sigma -> eta_t`.

**Why the clamp lives here.** The $\theta_i$ are exactly the principal angles
between the old frame and the moved one, so this is the only place in the update
where "how far did the basis turn" is a quantity with units. Bounding it is
therefore one clamp in meaningful coordinates, in the single spot where every
tangent producer converges, instead of a scatter of proxies -- a rank ratio, a
tangent magnitude, a Gram condition number -- each of which merely correlates
with rotation. It is exact rather than approximate: $(\Delta V)/\sigma$ is
already the unit-norm geodesic direction, so scaling it by $\sin\theta$ lands
on the honest geodesic point at $\theta$, and $Q_{raw}$ stays as orthonormal as
it was before.

$\theta_{\max}$ is a pathology bound, not a tuning knob. A frame that turns
more than a quarter-circle in one step is thrashing, not tracking, and the
eigenvectors that produced such a tangent are not trustworthy anyway. Healthy
runs sit orders of magnitude below it: on a synthetic low-rank-plus-noise
bench the requested rotation peaks at 1.4e-2 rad during basis acquisition and
settles to ~2e-4, four orders below the ceiling. It bites where the gradient
has no low-rank structure to track at all, which is the case that has nothing
useful to do with a subspace tracker anyway.
`UsuiTrack.pop_basis_rotation_angle()` returns the mean per-matrix RMS $\theta$
*after* clamping -- the rotation the frame actually took -- so a reading pinned
near $\theta_{\max}$ is itself the saturation signal, and a run can be checked
against the ceiling rather than assumed safe. That accumulator lives on
device and is the only host sync, so read it at logging cadence, never per step.

$\Delta$ itself is left unbounded. Its magnitude is not the quantity that has to
stay sane, and it is sanitized (`nan_to_num`) once where the buckets are formed,
covering both its Gram and the geodesic.
Equal-rank tangent-Gram eigendecompositions are batched. With
`S=Q_raw^T Q_raw`, one near-identity step using the converged steady-state
coefficient triple from Amsel, Persson, Musco, and Gower, ["The Polar
Express"](https://arxiv.org/abs/2505.16932) (2025), retracts before storage:

$$Q_+=Q_{raw}(aI+bS+cS^2).$$

The Oja tangent rotates every tracked plane. UsuiTrack stores no target frame or
second tracker state and requires a full matrix gradient on every step.

### 6. Accumulate momentum

$$M_t=\beta M_{t-1}+(1-\beta)Z_t,\qquad \beta=0.95.$$

There is no EMA bias correction.

### 7. Transport momentum through frame motion

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

### 8. Aurora direction

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

### 9. Scale, lift, and update

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

1. sanitize and clip `G`;
2. update Adafactor state and form `G_tilde`;
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
consumed; deferring the Adafactor and moment updates to `step()` would mean
holding a matrix-sized conditioned gradient per parameter until then, which is
the cost the two-phase design exists to avoid. So state commits in two places:
phase one commits the moving averages, phase two commits the weight. Normal
completion of a step still requires `step()`, not `zero_grad()`. But a caller
that must bail out mid-step regardless (an OOM-retry loop, say) is allowed to:
`zero_grad()` drops any pending prepared update rather than rejecting the
call. This does not roll back the moving-average state prepare() already
advanced -- that one step's contribution to Adafactor/projected-moment state
is unrecoverable -- it only clears the retained tangent so the next
`prepare()`/`step()` starts clean. Repeated preparation, gradient accumulation
before `step()`, and optimizer closures with pending work remain unsupported.

## Parameter eligibility

`ndim == 2` is the structural gate `add_param_group` enforces, and it is *not*
the precondition the method has. UsuiTrack tracks a low-rank subspace of a
**shared linear map**: it assumes a weight's rows (or columns) are coordinates
in one common space, so that a basis fitted from a few batches means something
for all of them. Two families of tensor pass the shape test and break that
assumption. Neither failure is numerical, so no guard in the update path can
catch them -- both were found as a run going non-finite thousands of steps in.

**Lookup tables.** `nn.Embedding` weights, and any matrix whose rows are
independent per-token vectors. There is no shared map to track, and the
gradient is row-sparse: most rows go untouched for thousands of steps with
their Adafactor row variance pinned at the eps floor, while the basis only ever
sees the rows a batch lights up. Observed: a 32128x1024 T5 vocabulary table
went non-finite mid-run with a finite gradient and a healthy loss, while the
updates it did receive were ~3e-7 per element against weights of order
1e-2..1e2 -- below bf16 resolution, so it carried the risk without training.
Muon-lineage optimizers exclude embedding tables for the same reason.

**Multiplicative gates.** AdaLN/FiLM modulation linears, and anything whose
output scales or shifts another layer's output instead of feeding forward.
These are ordinary, well-conditioned matrices; the problem is that a small
tracking error in a gate is multiplied through everything downstream instead of
staying additive and local. Observed: a `(6144, 256)` AdaLN modulation linear,
well clear of any rank or shape degeneracy, went non-finite one step into
training.

Both families are named consistently within an architecture and are not
cheaply detectable at runtime, so the split is: the library owns the structural
rule and the routing mechanics, the caller supplies the names.
`usuitrack.routing` provides `RoutingPolicy(exclude, track_right, track_left)`
and `route_parameters(named_parameters, policy) -> Routing`, which splits a
model's named parameters into UsuiTrack's per-side groups and a fallback list.
`Routing.describe()` summarizes what a policy actually matched -- worth logging
once at startup, since a policy is a list of substrings and its failure mode is
a hint that silently matches nothing after an upstream rename.

The same policy carries the `side` hint, for the same reason: side is
architecture semantics, not shape. See the Coordinates section for the rule.

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
| basis | `[r,n]` | `[m,r]` | parameter dtype (see note) |
| projected EMA | `[m,r]` | `[r,n]` | gradient dtype |
| Adafactor row variance | `[m]` | `[m]` | fp32 |
| Adafactor column variance | `[n]` | `[n]` | fp32 |
| update counts and resolved side | scalar | scalar | Python |

Oja tangents are transient pending work. No target frame, second basis, or lag
snapshot is stored. Fallback parameters retain AdamW's fp32 first moment, second
moment, and step tensor in their separate optimizer.

**The basis is stored in the parameter dtype, and for bf16 that costs
convergence.** The geodesic runs in fp32 but the moved frame is written back in
the parameter dtype, so a bf16 model rounds every frame update to nearest.
Measured on a synthetic low-rank-plus-noise bench (256x128, r=16, 140 steps
after warmup), comparing the rotation the geodesic *commanded* against the net
subspace drift it *achieved*:

| basis storage | commanded | net drift | ratio |
|---|---|---|---|
| fp32 | 0.0309 rad | 0.0036 rad | 12% |
| bf16, round-to-nearest | 0.0320 rad | 0.0300 rad | 94% |
| bf16, stochastic rounding | 0.0328 rad | 0.0449 rad | 137% |

The commanded rotation is the same in all three, so the tracking signal is
intact; what differs is what the frame does with it. In fp32 the per-step
rotations largely cancel -- the basis has converged and is oscillating around a
fixed subspace, which is the desired steady state. In bf16 it random-walks:
nearly all of the commanded motion turns into net displacement, because
per-entry rounding noise at ~3.9e-3 relative is the same size as a steady-state
rotation of ~2e-4 rad. The failure mode is drift, not freezing.

**Stochastic rounding is the wrong remedy here, and the reason generalizes.**
It makes the drift worse, not better, because it works by injecting noise to
preserve the expectation of a sum. That is exactly right for `W`, which should
*integrate* many small same-signed steps (step 9), and exactly wrong for `Q`,
which should *converge* and then stop moving -- a rounding scheme that
guarantees motion every step guarantees a frame that never settles. Storage
that should integrate wants stochastic rounding; storage that should converge
wants precision.

So the remedy, if this holds on a real run, is an fp32 basis (`r * d` fp32 per
matrix instead of bf16). Not yet applied: the numbers above are a synthetic
bench, and the confirming signal is a real training run where
`pop_basis_rotation_angle()` sits near the bf16 noise floor while loss
plateaus. Do not treat the bench as the verdict.

## Decisions and reasons

These choices define the current design; they are redesignable.

1. **One-sided projection:** retains a rank-limited update without a square core
   or second basis.
2. **Residual-facing side policy:** up/gate and q/k/v use storage-right; down
   and attention-output use storage-left. Generic `auto` remains shape-only.
3. **Side-Gram `eigh` initialization:** directly solves the one-sided target and
   has explicit fp32, finite-input, symmetrization, and jitter behavior.
4. **One-state full-gradient basis tracking:** the live frame follows the Oja
   covariance action on its configured cadence, with harmonic geodesic motion
   `1/2, 1/3, ...` down to `0.01` and no second basis.
5. **Moving-frame momentum:** identity coordinates preserve the projected
   moment's spectrum through the chosen frame rotation.
6. **Adafactor before tracking and projection:** both consumers see the same
   SNR-conditioned signal; RMS restoration preserves tracker scale units.
7. **Raw clipping before all consumers:** protects persistent Adafactor state and
   Oja tangent construction, not just momentum.
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
