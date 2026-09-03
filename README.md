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
non-2D tensors: lookup tables (embeddings) and multiplicative gates (AdaLN-style
modulation linears) are 2D but are not shared linear maps, and belong in the
fallback. `docs/SPEC.md` has the reasoning.

Rank is a ceiling, not a promise: each parameter runs at no more than half its
smaller side, so a model configured at rank 128 runs its tall narrow modules
lower. UsuiTrack says so once at startup.

```python
import torch
from torch import nn
from usuitrack import UsuiTrack

embedding_params = {id(p) for m in model.modules() if isinstance(m, nn.Embedding) for p in m.parameters()}
matrix_params = [p for p in model.parameters() if p.ndim == 2 and id(p) not in embedding_params]
other_params = [p for p in model.parameters() if p.ndim != 2 or id(p) in embedding_params]

matrix_opt = UsuiTrack(matrix_params, lr=4e-4, rank=128)
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
matrix_opt = UsuiTrack(matrix_params, lr=4e-4, rank=128, release_matrix_grads=True)
```

This requires one backward per `step()` call, no gradient accumulation, and
it isn't transactional: see `examples/` and `docs/SPEC.md` for the exact
contract.

See [`examples/`](examples) for a plain-PyTorch loop, an HF `Trainer`
integration, and a full-finetune run through Unsloth's `FastModel` loader
(no LoRA, no quantization: UsuiTrack drives the full weights directly).

Gradient clipping is always on (`grad_clip_norm`, default `1.0`) and cannot be
disabled: it is what protects the basis and the projected moment from a single
bad batch.

### Is the tracker actually tracking?

Optional telemetry, off by default, that answers that without assuming any
particular trainer -- it is a plain dict of floats, so it drops straight into
whatever logging you already have:

```python
matrix_opt.diagnostics = "core"   # or "full"; "off" is the default
...
if step % 10 == 0:
    for key, value in matrix_opt.pop_diagnostics().items():
        my_logger.log(f"usuitrack/{key}", value, step=step)
```

Measurements accumulate on-device every step and are read back once per
`pop_diagnostics()` call, so a logged point is the mean over the interval since
the last read rather than a sample of one step, and the hot path stays free of
device syncs. Call it on your logging cadence, not every step. An empty dict
means nothing has happened since the last read and is safe to skip.

Two tiers, split by cost rather than by usefulness. `"core"` is everything that
falls out of tensors the step already formed, so it can stay on for a whole run
without a decision. `"full"` adds the three reads that need a frame snapshot,
which is a fixed cost over a fixed sample of matrices.

The one that answers the title question is `transport_speed`: how far the frame
moved in one update, as an RMS angle per tracked plane. Switch to `"full"` and
`transport_curve` answers the follow-up -- how much of that motion actually went
anywhere, rather than cancelling.

`tangent_concentration` answers a different question, and it is the more
interesting one over a long run: how structured the space being tracked still
is. It is the leading direction's share of the tangent's energy, so it falls as
the gradient's principal subspace flattens and the easy structure gets used up.
Measured on a 1k finetune it reads `0.79` early and `0.68` by the end, while
`tangent_participation` climbs from `0.014` to `0.020` -- the same story from the
other side, with more planes carrying comparable energy. Read together they say
how much anisotropy is left to track, not whether the frame is chasing noise;
speed, curve and spin are what answer that.

`pop_diagnostics()`'s docstring documents the rest.

UsuiTrack is built for full-parameter training; LoRA is the comparison
point above, not a mode of operation. It also runs fine on LoRA adapter
matrices themselves (they're plain 2D tensors), but expect it unoptimized
there: dedicated orthogonalized-update work on LoRA already exists and
does that job with less machinery than UsuiTrack carries. Whether tracking
a moving subspace helps a LoRA adapter stay useful over a much longer run,
where a fixed low-rank parameterization tends to overfit, is an open
question we haven't tested. If you try it, the residual-facing `side`
policy was tuned for full weight storage conventions and we have no idea
whether it transfers to adapter matrices; `side="auto"` is the safer
starting point there.

## How it works

[`docs/SPEC.md`](docs/SPEC.md) is the precise mathematical specification:
coordinates, the per-step pipeline, and the design decisions with their
reasons. [`docs/LEGEND.md`](docs/LEGEND.md) is a long-form narrative account
of the same mechanism, written for intuition rather than reference. The two
are meant to be read together and are checked against each other.

In short: each matrix keeps a small orthonormal basis (`rank` columns) and a
projected first moment in that basis. The basis moves via an exact
Grassmann-geodesic step driven by an Oja covariance tangent, so momentum
transports through basis motion as a rigid rotation instead of being
re-projected and losing energy. Every tracked plane turns by the same angle --
the tangent's magnitudes decide only how motion is split between planes, and on
a single batch they are the least trustworthy thing in it -- and how large that
angle is comes from how much this step's aim agrees with the previous step's. A
frame that is still skewed keeps re-measuring its own lag, so its aim repeats and
it turns; a fitted frame sees uncorrelated batch noise and slows. Both ends of
that meter are geometry rather than tuning, so the tracker carries no fitted
constant beyond the step size itself. Direction comes from two steps kept apart:
a leverage-balancing row rescale
([Aurora](https://github.com/tilde-research/aurora-release), from Tilde
Research), then a Newton-Schulz polar map that does the orthogonalizing, using
the optimal coefficient schedule from Amsel, Persson, Musco, and Gower's ["The
Polar Express"](https://arxiv.org/abs/2505.16932). Magnitude comes from Muon's
aspect factor on the original parameter shape -- which enforces a
storage-invariant update norm at full rank and, under a rank-`r` projection,
leaves a `sqrt(r/min(m,n))` residual instead; see `docs/SPEC.md` step 8.
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
- `release_matrix_grads` requires exactly one backward per `step()`. Plain
  gradient accumulation (no release, multiple `backward()` calls before one
  `step()`) is tested and works.
- Not tested under `autocast`/`GradScaler`, `torch.compile`'s `capturable`/
  differentiable modes, or exotic tensor subclasses.
- Minimum PyTorch version (`2.1`) is inferred from the API surface used
  (`register_post_accumulate_grad_hook`), not empirically swept across
  versions.

If you hit one of these, open an issue. This list is the honest starting
point, not a promise every box will stay unchecked.

## License

MIT. See [`LICENSE`](LICENSE).
