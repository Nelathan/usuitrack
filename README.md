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
weight and route everything else to a separate optimizer: the same split Muon
and other orthogonalized optimizers use.

"Everything else" is more than the non-2D tensors. UsuiTrack tracks a subspace
of a *shared linear map*, and two families of weight pass the shape test while
breaking that assumption: **lookup tables** (embeddings, whose rows are
independent vectors with row-sparse gradients) and **multiplicative gates**
(AdaLN/FiLM modulation linears, whose error compounds through everything they
scale rather than staying local). Both are named consistently within an
architecture and neither is cheaply detectable at runtime, so you name them:

```python
import torch
from usuitrack import RoutingPolicy, UsuiTrack, route_parameters

policy = RoutingPolicy(
    exclude=("embed.weight", "pos_emb", "norm1.linear", "norm_out.linear"),
    # side is architecture semantics, not shape: track the residual stream.
    track_right=("to_q", "to_k", "to_v", "net.0.proj."),   # read the stream
    track_left=("to_out.", "net.2."),                      # write the stream
)
routing = route_parameters(model.named_parameters(), policy)
print(routing.describe())  # check the hints actually matched something

matrix_opt = UsuiTrack(
    [{"params": [p for _, p in entries], "side": side} for side, entries in routing.matrix.items()],
    lr=4e-4,
    rank=128,
)
other_opt = torch.optim.AdamW([p for _, p in routing.fallback], lr=1e-4, betas=(0.9, 0.99))

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
re-projected and losing energy. Direction comes from a leverage-balanced
Newton-Schulz polar map ([Aurora](https://github.com/tilde-research/aurora-release),
from Tilde Research), using the optimal coefficient schedule from
Amsel, Persson, Musco, and Gower's ["The Polar
Express"](https://arxiv.org/abs/2505.16932); scale comes from the original
parameter shape, Muon-style. Each geodesic is capped at a quarter-turn, and
`pop_basis_rotation_angle()` reports how far the basis actually moved, so
"is the tracker tracking?" is a number you can log rather than infer.
For bf16 weights the update is accumulated in
fp32 and written back with stochastic rounding, because an orthogonalized step
is often a fraction of a bf16 ulp and round-to-nearest would discard it
outright; `StochasticAdamW` is provided for the fallback group for the same
reason.

The design question behind all of this -- what a small-batch, noisy-gradient
run actually needs from an optimizer -- was framed for me by Martin Marek's
[batch-size study](https://github.com/martin-marek/batch-size). No code taken.

## Status and limitations

This is an early release. The core algorithm is measured (see above); the
generic-PyTorch integration surface is not yet fully swept:

- Tested on a single GPU. No DDP, FSDP, or other distributed testing.
- The tracked basis is stored in the parameter dtype. On a synthetic bench, a
  bf16 basis random-walks on rounding noise instead of settling once converged
  (`docs/SPEC.md`, "Persistent state"). Unconfirmed on a real run; watch
  `pop_basis_rotation_angle()` if you are training in bf16.
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
