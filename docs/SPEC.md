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
- **Raw sigma is the controller, not a raw number to be tamed.** Normalizing or
  capping the geodesic angle removes the self-annealing that makes the tracker
  settle. Both have been tried and reverted.
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

This occurs before all matrix consumers, and there is no off switch: `c` has no
`None`. Clipping the raw gradient is what protects every persistent memory
downstream of it -- Adafactor's factored variance, the frame, the projected
moment -- and a guard installed later cannot protect an earlier one. A single
blip batch (grad norm spiking ~180x) otherwise writes a ~30000x spike into the
factored second moment, which at `beta2=0.99` over-dampens whole gradient
directions for ~100 steps while the basis tracks the starved residual and every
later refresh adapts to the corruption.

Making it mandatory also collapses the update to a single implementation.
Once `c` is always present, the only thing that varies between steps is whether
the frame exists yet and whether a basis update is due, so the tangent is built
in exactly one place: the fused per-side kernel. There is no second readable
copy to drift out of sync with it -- that is what this document is for.

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

$$Q_{raw}=\left[(QV)\operatorname{diag}(\cos(\eta_t\sigma_i))
+(\Delta V)\operatorname{diag}
\left(\frac{\sin(\eta_t\sigma_i)}{\sigma_i}\right)\right]V^\top.$$

The zero-singular-value limit is `sin(eta_t sigma) / sigma -> eta_t`.

`sigma` is raw: neither normalized nor clamped. That is the mechanism, not an
oversight. A poorly fitted basis produces large `sigma` and turns hard; a
well-fitted one produces small `sigma` and settles. Normalizing the angle
(by `sigma_max` or by the tangent's Frobenius norm) forces a constant turn every
refresh, re-inflates the residual tail, and prevents convergence -- tried,
rejected. Capping the angle costs the other end, the large-`sigma` acquisition
the schedule is hot for. Measured on a 2B DiT, the per-step angle anneals
`0.7 -> 0.11 -> 0.038 -> 0.035` rad over a full run without any bound applied.

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

1. sanitize and clip `G` (always);
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

`ndim == 2` is the structural gate `add_param_group` enforces, and it is not the
precondition the method has. UsuiTrack tracks a subspace of a **shared linear
map**: it assumes a weight's rows (or columns) are coordinates in one common
space, so a basis fitted from a few batches means something for all of them. Two
families pass the shape test and break that assumption. Neither failure is
numerical, so no guard in the update path can catch them; both were found as a
run going non-finite thousands of steps in.

**Lookup tables.** `nn.Embedding` weights, and any matrix whose rows are
independent per-token vectors. There is no shared map, and the gradient is
row-sparse: most rows go untouched for thousands of steps with their Adafactor
row variance pinned at the eps floor, while the basis only ever sees the rows a
batch lights up. Observed on a 32128x1024 vocabulary table: non-finite mid-run
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
| Adafactor row variance | `[m]` | `[m]` | fp32 |
| Adafactor column variance | `[n]` | `[n]` | fp32 |
| update counts and resolved side | scalar | scalar | Python |

Oja tangents are transient pending work. No target frame, second basis, or lag
snapshot is stored. Fallback parameters retain AdamW's fp32 first moment, second
moment, and step tensor in their separate optimizer.


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
