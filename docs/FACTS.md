# Facts

Measured numbers that neither state a rule (`SPEC.md`) nor frame an open question
(`PLAN.md`). Each carries where it came from. A fact here is a measurement at a
date, not a law; when a new read contradicts one, replace it and say so in PLAN.

## Anima lane

### What the model is made of

Base `circlestone-labs/Anima-Base-v1.0-Diffusers` transformer, by the roles
`scripts/usuitrack_base_distance.py` assigns (2026-09-14):

| role | tensors | params | base rms | trained by |
|---|---|---:|---:|---|
| attn1 q/k/v/out, attn2 q/out | 2D | 117M each | 0.0099-0.0116 | UsuiTrack |
| attn2 k/v | 2D | 58.7M each | 0.0094 / 0.0146 | UsuiTrack |
| ff_up, ff_down | 2D | 470M each | 0.0111 / 0.0114 | UsuiTrack |
| patch_embed, proj_out | 2D | 0.14M / 0.13M | 0.0158 / 0.0055 | UsuiTrack |
| adaln (`linear_1`/`linear_2`) | **2D** | 58.7M | 0.0259 | AdamW fallback |
| other (`linear_1`/`linear_2`) | **2D** | 119M | 0.0406 | AdamW fallback |
| other 1D (norm_q/norm_k gains) | 1D | 14k | 0.774 | AdamW fallback |
| time (linears / norm) | 2D / 1D | 16.8M / 2k | 0.0088 / 0.192 | frozen (`ignore_if_contains`) |

The fallback class is not "small modules": ~178M 2D parameters, ~9% of the
transformer, excluded from UsuiTrack as multiplicative gates (`SPEC.md`,
parameter eligibility). Which modules `other`'s `linear_1`/`linear_2` belong to
has not been checked by name.

### Per-step motion, in one unit

Relative update per step (update norm over weight norm), 2026-09-14:

| class | per-step relative motion |
|---|---:|
| UsuiTrack matrices, run 13 plateau (`update_to_param_ratio`) | 3.7e-6 |
| UsuiTrack matrices, run 12 plateau | 2.3e-6 |
| AdamW `adaln` ceiling at `fallback_lr 2e-6` (`lr / rms`) | 7.7e-5 |
| AdamW `other` ceiling at `2e-6` | 4.9e-5 |
| LFM best arm (cosine-zero, `lr 4e-4`, bs16 x 1024 tokens) | 7.6e-5 |

AdamW moves each coordinate by up to `lr` per step regardless of gradient scale,
so its relative motion is `lr / rms`: at `2e-6` the fallback's per-step budget is
~20x the matrices' realized step. A nominal lr ratio between the two optimizers
means nothing without the rms. Anima's matrix step is ~20x smaller than LFM's
best arm at roughly twice the tokens per step (bs4 x k4 x ~2304 latent tokens
at 768 px, assuming 8x VAE and 2x2 patches -- not checked).

### bf16 stochastic rounding walks the weights

All weights on this lane are bf16 and every write is stochastically rounded.
Run 13 checkpoints (plateau ends at step 432), per role, 2026-09-14:

| role | `|w400-w350| / |w350-base|` | cos(increment, net) | coords unchanged 350->400 |
|---|---:|---:|---:|
| every matrix role | 0.374-0.382 | 0.001-0.004 | 88-96% |
| adaln | 0.377 | 0.003 | 67% |
| other | 0.374 | 0.001 | 68% |
| patch_embed / proj_out | 0.437 / 0.404 | -0.062 / -0.016 | 84% / 53% |

A random walk predicts `sqrt(50/350) = 0.378` and cosine 0; coherent travel
predicts ~0.14 and a positive cosine. Toy on a real `attn1.to_q` (2048x2048),
rank-78 update stream at relative `3.7e-6` per step, `beta 0.9`:

| update stream | fp32 `|net|` at 350 | bf16+SR `|net|` | bf16 ratio / cos / unchanged |
|---|---:|---:|---|
| incoherent | 3.0e-4 | 2.13e-3 | 0.380 / 0.001 / 91% |
| 10% coherent | 6.2e-4 | 2.20e-3 | 0.366 / 0.026 / 91% |
| fully coherent | 1.30e-3 | 2.47e-3 | 0.331 / 0.119 / 91% |

The incoherent toy reproduces the real run's magnitude, ratio, cosine and
unchanged fraction. Rounding error is 7x the fp32 travel there, 1.6x even for a
perfectly straight path. A small toy (256x128, rms-1 weights, `lr 0.05`) moves
its weights 4.9x the intended update per step through the same mechanism.

Consequence: every distance-from-base figure on this lane -- the P13 run-10
monotone rise, the P18 fallback tables, the run 12/13 comparison below -- is
dominated by the walk. It remains the right read for "how far did the weights
go"; it is not a read of how far learning went.

### Distance from base, run 12 @2200 vs run 13 @550

| role | run 12 | run 13 |
|---|---:|---:|
| ALL | 2.314e-3 | 2.944e-3 |
| adaln / other | 0 (frozen) | 4.691e-3 / 3.173e-3 |
| attn1 k / q / v / out | 4.71 / 4.28 / 3.30 / 3.76 e-3 | 2.99 / 2.73 / 2.15 / 2.54 e-3 |
| attn2 k / q / v / out | 4.53 / 3.57 / 3.79 / 2.73 e-3 | 3.31 / 2.41 / 2.60 / 1.96 e-3 |
| ff_up / ff_down | 3.55 / 2.36 e-3 | 2.29 / 1.57 e-3 |
| patch_embed / proj_out | 12.5 / 11.5 e-3 | 7.17 / 6.90 e-3 |

Every matrix role moved less in run 13; the whole-model excess is the fallback
class, frozen in run 12. Both columns are walk-dominated (above).

### Run 12 and run 13 telemetry

Bucket means over each run (steps are optimizer steps):

| read | run 12 (k=1, 2304 steps, lr 5e-5) | run 13 (k=4, 576 steps, lr 1e-4) |
|---|---|---|
| `update_to_param_ratio` plateau | 2.33e-6 | 3.70e-6 |
| `moment_persistence` | +5e-4 -> -3.5e-5 (mid) -> +5.5e-4 | -6.7e-4 (start) -> 1.1-1.7e-3 |
| `grad_moment_cosine` | +4.6e-3 -> -1.8e-4 (mid) -> +1.4e-3 | +3.4e-3 .. +6.6e-3, +1.0e-3 at the end |
| `micro_batch_agreement` | -- | 2.5-3.1e-3 plateau, 1.3e-3 then 3.0e-4 as lr decays to 2e-5 |
| `grad_capture` | 0.606 -> 0.655 | 0.646 -> 0.748 |
| `tangent_live_fraction` | 0.884 -> 0.902 | 0.857 -> 0.906 |
| `tangent_concentration` | 0.85 flat | 0.613 -> 0.573 |

Run 13 changed `k`, lr, rank table (sum 595 vs 256) and `fallback_lr` together,
so no row isolates `k`. Run 14 (`k=1`, same table and lr-equivalent step) is the
contrast.

### Rank calibration at `k=4`, and what the sized table reached

`anima_rankcal_r256_acc4` (rank 256, bs4 x k4, `lr 4e-5`, bf16 accumulators, 9
windows) against run 13's per-role `live_fraction` at its table, steps >= 200:

| role | mean | median | geomean | geomean/mean | table | run 13 frac |
|---|---:|---:|---:|---:|---:|---:|
| attn1_k | 114.4 | 120.1 | 105.6 | 0.92 | 106 | 0.890 |
| attn1_out | 76.5 | 84.3 | 65.7 | 0.86 | 66 | 0.871 |
| attn1_q | 82.4 | 84.0 | 78.4 | 0.95 | 78 | 0.908 |
| attn1_v | 42.2 | 43.9 | 38.5 | 0.91 | 39 | 0.918 |
| attn2_k | 56.4 | 66.1 | 45.1 | 0.80 | 45 | 0.818 |
| attn2_out | 38.7 | 32.9 | 27.6 | 0.71 | 28 | 0.812 |
| attn2_q | 47.7 | 50.5 | 45.2 | 0.95 | 45 | 0.920 |
| attn2_v | 43.8 | 49.1 | 34.2 | 0.78 | 34 | 0.821 |
| ff_down | 53.7 | 63.8 | 45.4 | 0.85 | 45 | 0.898 |
| ff_up | 46.9 | 47.7 | 45.1 | 0.96 | 45 | 0.955 |
| patch_embed / proj_out | shape-capped | | | | 34 / 32 | 0.988 / 0.999 |

`corr(geomean/mean, frac) = 0.92` over the ten uncapped roles. At equal rank 256,
k=4 live counts were 2.1-3.6x the k=1 calibration's (`PLAN.md` history in
`ARCHIVE.md`). Bf16 rounding cannot account for that: stochastically rounding a
2048x256 tangent with a power-law spectrum moves its live count by at most one
plane. No fp32 `k=4` calibration exists to compare -- that attempt OOMed at step
3, before its first window.

### Memory and throughput

| run | shape | s/step (wall, incl. sampling) | s/sample | memory |
|---|---|---:|---:|---|
| run 12 | bs4, k=1, table 256 | 4.67 | 1.17 | -- |
| run 13 | bs4, k=4, table 595 | 17.76 | 1.11 | wandb: 76% of 12.3 GB allocated |
| run 14 | bs4, k=1, table 595 | ~3.2-3.7 (progress bar) | | |
| rankcal r256 k=4, fp32 accumulators | | | | OOM at step 3 |
| rankcal r256 k=4, bf16 accumulators | | 13.9-15.3 | | needed `expandable_segments` |

Accumulation costs nothing per sample; it costs optimizer steps per hour. The
card is at its limit at bs4 x k4 with the sized table -- an earlier "6.6 GB
peak" note was a different counter and should not be used for headroom.
Sampling: ~23 s per 1024x1024 image at 30 steps.

## LFM2.5-350M lane

bs16 x seq1024, seed 1, 1k steps, broad-no-embeddings. Noise floors: target
`3e-4`, source `2e-3`.

| arm | target | source | `update_to_param_ratio` |
|---|---:|---:|---:|
| global r128, `2e-4` | 1.669330 | 3.084276 | |
| per-role table (12,032 planes), `2e-4` | 1.667302 | 3.079098 | 1.66e-4 |
| calibrated table (11,886 planes) | 1.667920 | ~= | |
| `side=right`, r128 | 1.677500 | 3.089930 | |
| `beta 0.9`, `lr 4e-4` (cosine-zero) | **1.665708** | **3.040611** | 7.6e-5 |
| `beta 0.9`, `lr 8.7e-4` | 1.669194 | 3.086723 | 1.65e-4 |
| `beta 0.5`, `lr 1.6e-4` | 1.690248 | 3.013536 | 7.7e-5 |
| `beta 0.99`, `lr 1.29e-3` | 1.681707 | 3.138621 | 7.5e-5 |
| `beta 0`, `lr 2e-4` | 1.685989 | 3.020878 | 1.66e-4 |

`moment_persistence` within `1e-3` of zero in every arm. With no moment
(`beta 0`) target loss is far worse: the moment's temporal cancellation is doing
real work there.

## Run ledger (Anima)

| run | what changed | outcome |
|---|---|---|
| 8 `live_floor` | rank 64 uniform, `1e-5`, derived live floor | first non-destructive run |
| 9 | calibrated table rounded up, `2e-5` | aesthetic, coherent; ended close to base |
| 10 `inner_lr5e5` | measured table, inner-side cross-attn K/V, `5e-5`, fallback `5e-6`, time frozen | perturbed-then-settled; anatomical breakage in places |
| 11 `lr1e4_cosine` | `1e-4`, cosine from warmup | cooler for longer |
| 12 `freeze_fallback` | fallback frozen, resized table, constant turn, `5e-5` | stable, slightly harmed, bland: no creative/aesthetic advance |
| 13 `acc4` | k=4, table from k=4 calibration (sum 595), `1e-4`, fallback `2e-6` | samples good, wants more; undertrained |
| 14 `k1_contrast` | run 13 with k=1 and `5e-5` | running 2026-09-14 |
