# UsuiTrack

A PyTorch optimizer for matrix parameters that keeps a rank-limited, moving
low-rank basis instead of a full-size first or second moment. It targets full
fine-tuning and continued pretraining on consumer GPUs, where AdamW's two
full-size moments per matrix are often the thing standing between you and a
bigger model or a bigger batch.

UsuiTrack is not a LoRA replacement in the usual sense. It doesn't freeze a
low-rank parameterization of the weight; it tracks a moving rank-`r` subspace
of the *gradient* and updates the full weight through it, every step. The
subspace itself adapts as training proceeds.

## Result

Single controlled comparison, matched matrix-parameter set, matched optimizer
state budget, `LiquidAI/LFM2.5-350M-Base`, 1k steps:

| | target loss | source retention loss | peak allocated VRAM | matrix state |
|---|---|---|---|---|
| LoRA rank 44 + AdamW | 1.674265 | 3.084880 | 3.81 GB | 98.9 MB |
| UsuiTrack rank 128 | **1.672156** | **3.037444** | **2.88 GB** | 97.4 MB |

LoRA leads target loss early (step 100: 1.770 vs 1.876), UsuiTrack overtakes
around step 900, and LoRA runs about 1% faster per step. Full methodology and
the evidence trail this survived are in [`docs/SPEC.md`](docs/SPEC.md).

## Install

Not yet on PyPI:

```bash
uv add git+https://github.com/Nelathan/usuitrack
```

Requires `torch>=2.1`, Python `>=3.11`. Pure PyTorch: no other runtime
dependency.

## Use

UsuiTrack only accepts 2D matrix parameters. Give it every `Linear`-style
weight and route everything else to a separate optimizer, the same split Muon
and other orthogonalized optimizers use. "Everything else" is wider than the
non-2D tensors: lookup tables (embeddings) are 2D but are not shared linear
maps, and belong in the fallback. `docs/SPEC.md` has the reasoning.

Rank is not configured per matrix: `rank_fraction` (default `0.1`) sizes each
matrix at `rank_fraction * sqrt(m * n)`, capped at half its smaller side, so a
1024x1024 projection runs at 102 and a 4608x1024 MLP matrix at 217. The tracked
side defaults to `side="auto"`, the shorter one; a param group can fix it with
`side="left"` or `"right"` where you know better.

`lr` is the relative step, `||dW|| / ||W||` per matrix, so the default `4e-5`
means the same in every module. Schedule it like any optimizer's: a trainer's or
torch's scheduler drives it, and the optimizer has no schedule of its own.

```python
import torch
from torch import nn
from usuitrack import UsuiTrack

embedding_params = {id(p) for m in model.modules() if isinstance(m, nn.Embedding) for p in m.parameters()}
matrix_params = [p for p in model.parameters() if p.ndim == 2 and id(p) not in embedding_params]
other_params = [p for p in model.parameters() if p.ndim != 2 or id(p) in embedding_params]

matrix_opt = UsuiTrack(matrix_params, lr=1e-4)
other_opt = torch.optim.AdamW(other_params, lr=1e-4, betas=(0.9, 0.99))

loss = model(**batch).loss
loss.backward()
matrix_opt.step()
other_opt.step()
matrix_opt.zero_grad()
other_opt.zero_grad()
```

An optional layerwise-release mode drops each matrix gradient right after
it's consumed instead of holding the whole backward's gradients in memory at
once:

```python
matrix_opt = UsuiTrack(matrix_params, lr=1e-4, release_matrix_grads=True)
```

This takes one backward per `step()`: a second gradient before `step()` raises,
so accumulate with it off. Since the matrix gradients are gone by the time
`backward()` returns, a trainer's gradient clipping and logged gradient norm see
only the fallback parameters. `docs/SPEC.md`, "Two-phase execution", has the
contract.

With activation checkpointing, the weight-gradient GEMMs are a quarter of the
linear layers' matmul FLOPs. `ProjectedGrads` skips them: each tracked `Linear`'s
backward returns the rank-`r` projection the optimizer consumes, straight from
the layer's input and output gradient, and the full gradient is never formed.
On LFM3-1.7B it ties the ordinary path's loss through 1k steps at 12% less
time per step, and gradient accumulation works without holding any gradient.

```python
from usuitrack import ProjectedGrads, UsuiTrack

matrix_opt = UsuiTrack(matrix_params, lr=1e-4)
ProjectedGrads(model, matrix_opt)  # after building the optimizer, before training
```

The loop is unchanged; `step()` collects the projections. Every matrix must be
a plain `nn.Linear`'s weight, and the tracked weights have no `.grad`, so
clipping, `GradScaler` and gradient-norm logging see only the fallback
parameters. A setup it cannot handle raises with the reason. `docs/SPEC.md`,
"Projected gradients", has the contract.

See [`examples/`](examples) for a plain-PyTorch loop, an HF `Trainer`
integration, and a full-finetune run through Unsloth's `FastModel` loader
(no LoRA, no quantization: UsuiTrack drives the full weights directly).

There is no gradient clipping, and nothing to configure: every consumer of the
gradient is insensitive to its magnitude. The basis tracker divides one
quadratic in `G` by another, and the update normalizes each step's gradient to
unit norm before it enters the moment, so a bad batch contributes one unit of
weight like any other step. Non-finite entries are still zeroed.

### Diagnostics

Off by default. With `diagnostics=True` the optimizer accumulates a few reads
on-device and hands them back as a plain dict of floats, for whatever logger you
already have:

```python
matrix_opt = UsuiTrack(matrix_params, diagnostics=True)
...
if step % 25 == 0:
    for key, value in matrix_opt.pop_diagnostics().items():
        my_logger.log(f"usuitrack/{key}", value, step=step)
```

Each value is the mean since the last call (`nonfinite_grads` is a count), so call it on your logging cadence:
it is the one device-to-host transfer. `usuitrack.diagnostics.READS` says how to
read each key. The two that tell you the most about a new setup are
`tangent_live_fraction` (does the rank fit what the gradient carries at this batch
size?) and `grad_capture` (how much of the gradient does the frame hold?).

UsuiTrack is built for full-parameter training; LoRA is the comparison point
above, not a mode of operation. Because the step is relative to `||W||`, **a
matrix initialized to zero never moves**, and that includes LoRA's `B` matrix.
Give such weights a non-zero init or the fallback optimizer.

## How it works

[`docs/SPEC.md`](docs/SPEC.md) is the precise mathematical specification:
coordinates, the per-step pipeline, and the design decisions with their
reasons. [`docs/LEGEND.md`](docs/LEGEND.md) is a long-form narrative account
of the same mechanism, written for intuition rather than reference. The two
are meant to be read together and are checked against each other.

In short: each matrix keeps a small orthonormal basis (`r` columns, sized by `rank_fraction`) and a
projected first moment in that basis. The basis moves via an exact
Grassmann-geodesic step driven by an Oja covariance tangent, so momentum
transports through basis motion as a rigid rotation instead of being
re-projected and losing energy. Every tracked plane the gradient resolves turns
by the same constant angle per step -- the tangent's magnitudes would only decide
how motion is split between planes, and on a single batch they are the least
trustworthy thing in it. Direction comes from two steps kept apart:
a leverage-balancing row rescale
([Aurora](https://github.com/tilde-research/aurora-release), from Tilde
Research), then a Newton-Schulz polar map that does the orthogonalizing, using
the per-step-tuned quintic schedule from the Muon lineage (see `docs/SPEC.md`
step 7 for provenance). The moment averages each step's projected gradient at
unit norm, and the polar map runs once, on the moment, so a batch's strong
planes weight the average and its noise planes stay small. Magnitude is
relative to the weight it lands on: `lr` is the fraction of `||W||` a matrix
moves per step, the same in every module whatever its shape or rank (LAMB's
trust ratio); see `docs/SPEC.md` step 8. That puts it on LoRA's scale. The
default `4e-5` was set on a 1.7B LLM at 16 x 1024 tokens per step, under a
cosine decay to a fifth of it over the run, which beat a constant `lr`; a 350M
LLM at the same batch takes more, `1e-4`, so the relative step does not
transfer across sizes one-to-one. On a 2B
image DiT at batch 8 the best relative step measured so far is about `1.3e-5`.
For bf16 weights the update is accumulated in fp32 and written back with
stochastic rounding, because an orthogonalized step is often a fraction of a
bf16 ulp and round-to-nearest would discard it outright; `StochasticAdamW` is
provided for the fallback group for the same reason.

The design question behind all of this -- what a small-batch, noisy-gradient
run actually needs from an optimizer -- was framed for me by Martin Marek's
[batch-size study](https://github.com/martin-marek/batch-size). No code taken.

## Status and limitations

This is an early release. The core algorithm is measured (see above); the
generic-PyTorch integration surface is not yet fully swept:

- Tested on a single GPU. No DDP, FSDP, or other distributed testing.
- Gradient accumulation works without `release_matrix_grads`, not with it;
  under `ProjectedGrads` it works and holds no gradient.
- Not tested under `autocast`/`GradScaler`, `torch.compile` of the optimizer
  step, or exotic tensor subclasses.
- Checkpoints from before this release's state layout (`frame`, `moment`) do not
  load.
- Minimum PyTorch version (`2.1`) is inferred from the API surface used
  (`register_post_accumulate_grad_hook`), not empirically swept across
  versions.

If you hit one of these, open an issue. This list is the honest starting
point, not a promise every box will stay unchecked.

## License

MIT. See [`LICENSE`](LICENSE).
