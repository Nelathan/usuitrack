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
predicts ~0.14 and a positive cosine.

Run 12 over a longer horizon (1400 -> 1600, plateau until 1728; k=1, step
`2.3e-6`): every matrix role reads ratio 0.369-0.383 against the walk's 0.378,
cosine 0.000-0.005, 80-93% of coordinates unchanged in 200 steps. For a
straight coherent component carrying fraction `f` of the displacement norm the
cosine is `f^2 * (t2-t1)/t1 / ratio`, so cosine <= 0.002 bounds `f` at ~7%.
The toy's fp32 *incoherent* stream (gradient noise through the moment) reads
cosine 0.045 on its own; the real 0.001 is below even that, so rounding noise
buries the moment's short-range correlation too. Read with
`scripts/usuitrack_walk_read.py` (ai-toolkit).

Toy on a real `attn1.to_q` (2048x2048), rank-78 update stream at relative
`3.7e-6` per step, `beta 0.9`:

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

### Walk-only control: what the walk does to samples

`anima_walk_only_control`: base weights plus a simulated walk
(`scripts/usuitrack_walk_only_model.py`) matched per role to run 13 @550 (whole
model 2.943e-3 against run 13's 2.944e-3), sampled at 1024 px, 30 steps, seed 42,
run 13's five prompts. Pixel RMS on [0,1] RGB against base (base samples are
bit-identical across runs, checked):

| prompt | walk-only | run 13 @50 | run 13 @550 |
|---|---:|---:|---:|
| 0 whale landscape | 0.018 | 0.078 | 0.128 |
| 1 armored warrior | 0.146 | 0.222 | 0.175 |
| 2 gown | 0.041 | 0.148 | 0.211 |
| 3 creature and bird | 0.036 | 0.138 | 0.181 |
| 4 ethereal figure | 0.018 | 0.050 | 0.116 |

Pixel RMS is crude and composition-sensitive; the images were also looked at
(agent's read, not the user's): the walk shifts composition on prompt 1, is near
identical to base on prompt 2, and carries none of run 13's style shift.

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

Run 13 changed `k`, lr, rank table (sum 595 vs 256) and `fallback_lr` together.

### Run 14: `k` against the table

`anima_usuitrack_14_k1_contrast` is run 13 with `k=1` and `lr 5e-5`, 576 steps.
Bucket means by optimizer step (runs 12 and 13 at the same step indices):

| read, steps 200-431 | run 12 (k=1, table 256) | run 14 (k=1, table 595) | run 13 (k=4, table 595) |
|---|---:|---:|---:|
| `update_to_param_ratio` | 2.34e-6 | 3.64e-6 | 3.71e-6 |
| `grad_capture` | 0.625 | 0.701 | 0.738 |
| `tangent_live_fraction` | 0.889 | 0.552 | 0.891 |
| `tangent_concentration` | 0.847 | 0.849 | 0.577 |
| `moment_persistence` | 6.3e-4 | 6.1e-4 | 1.16e-3 |
| `grad_moment_cosine` | 2.6e-3 | 2.4e-3 | 4.5e-3 |

Per-role live fraction, steps >= 200, run 14 against run 13: attn1 k/q/v/out
0.552/0.569/0.636/0.493 (run 13 0.890/0.908/0.918/0.871); attn2 k/q/v/out
0.500/0.504/0.467/0.508 (0.818/0.920/0.821/0.812); ff up/down 0.659/0.557
(0.955/0.898); patch_embed/proj_out 0.797/0.840 (0.988/0.999). The k=4/k=1
ratio at table rank is 1.4-1.8 per role, against 2.1-3.6 at calibration rank 256.

### Runs 15 and 16: current code, and patch-equalized loss

Both run 13's config for 300 steps (decay from 225), `k=4`. Run 15 on the
2026-09-14 trainer (token-exact weights, final fold unrounded, fallback divided
before the clip); run 16 adds `loss_type: patch_equalized_mse` and
`timestep_type: linear`. Bucket means by optimizer step:

| read, steps 150-224 | run 13 | run 15 | run 16 |
|---|---:|---:|---:|
| `update_to_param_ratio` | 3.71e-6 | 3.69e-6 | 3.70e-6 |
| `grad_capture` | 0.717 | 0.719 | 0.723 |
| `tangent_live_fraction` | 0.883 | 0.885 | 0.888 |
| `tangent_concentration` | 0.585 | 0.578 | 0.572 |
| `moment_persistence` | 1.33e-3 | 1.71e-3 | 1.81e-3 |
| `grad_moment_cosine` | 5.3e-3 | 7.1e-3 | 6.7e-3 |
| `micro_batch_agreement` | 2.70e-3 | 2.88e-3 | 2.48e-3 |
| `micro_batch_agreement`, steps 0-49 | 1.93e-3 | 1.87e-3 | 3.28e-3 |

Run 15 tracks run 13 on capture, liveness and concentration. Its persistence and
overshoot cosine read higher; no noise floor exists for those reads on this lane,
so that is unresolved, not an effect of the code change. Wall clock 17.4 (run 15)
and 17.7 s/step (run 16), both including sampling.

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
| run 15 | bs4, k=4, table 595, median interval | 14.78 | 0.92 | |
| run 12 | bs4, k=1, table 256, median interval | 3.93 | 0.98 | |
| probe | bs1, k=16, checkpointing, median of 2 intervals | 15.46 | 0.97 | |
| probe | bs1, k=16, no gradient checkpointing | | | OOM (3 retries) |
| rankcal r256 k=4, fp32 accumulators | | | | OOM at step 3 |
| rankcal r256 k=4, bf16 accumulators | | 13.9-15.3 | | needed `expandable_segments` |

Pinned PCIe on this box: 12.2 GB/s host-to-device, 12.6 GB/s device-to-host;
a CPU fp32 add runs ~0.06 s/GB (2026-09-14, measured beside a training run).

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

### The turn matters while the aim moves, and not once it has settled

The frame is in effect an average of its aims over the last `~1/eta` updates, so
the turn it can take is set by two things only: how far its aim can be trusted
and how fast the model moves the aim. Early, the aim moves fast and lag
dominates: on LFM (bs16, r128, `lr 2e-4`, `beta 0.9`, seed 1) capture and target
loss improve steadily from `0.005` to `0.04` and are flat to `0.08` (target at
300 steps 1.7468 at `0.005`, 1.7448 at `0.01`, 1.7426 at `0.04` and `0.08`).
The harmonic start `max(0.01, 1/t)` lands between `0.02` and `0.04`: its old win
was the size of the turn, not the decay.

Once the aim settles, the turn stops mattering. Dropping `0.04` to `0.01` at
step 200 or 500 never beats constant `0.04` at 1k (1.68330 and 1.68301 against
1.68288; `0.01` throughout 1.68391), and capture does not rise after the drop.
The switch at 200 keeps ~60% of `0.04`'s lead, all of it banked before the
switch. A settled frame orbits -- net travel over 10 updates is ~30% of its path
at both `eta` -- and the energy aim is steady enough that the wider orbit at
`0.04` costs no measurable capture. Every late gap is at or below the `3e-4`
floor; one seed per arm.

On Anima (run 17 against run 15, `k=4`) the same shape: capture +0.057 and live
fraction +0.016 over the first 100 steps, gaps nearly gone by 200, and neither
loss nor samples separate the runs at 300. `--seed` seeds only the rounding
draws; the LFM seed pair differs by `1.6e-4`. Runs `e3_*` (lab wandb) and
`anima_usuitrack_17_eta040_300`, 2026-09-28.

### The energy aim is the right target; the write roles lack rank

At fixed weights, 1024 bs1 samples, 27 matrices: the mean gradient's subspace
(`M`, pair-estimated) against the top-`r` eigenspace of what the aim targets at
batch `B`, each scored on how much of the *held-out* half's mean-gradient
energy it captures.

- **The signal collapses as the model trains.** At base the mean gradient is
  3.5-10% of a sample's energy; after 300 steps it is 0.1-0.4%, and its
  subspace no longer replicates between two halves of 512 samples (overlap
  ~0.2-0.3).
- **So no signal-only target is estimable, and the energy aim beats one.** The
  mean gradient lives inside the directions where per-sample energy lives, so
  the Cov-dominated aim is a low-variance estimate of it. At the trained state
  the energy frame captures more held-out signal than the estimated signal
  frame in eight of nine roles, and the gap widens with rank, because the signal
  frame saturates at its estimation noise (0.75-0.93) while the energy frame
  keeps climbing.
- **Rank splits by what a role does to the residual stream.** Roles that read it
  (conv.in_proj, w1, w3, q, k, v) saturate by rank 64-128 at 0.94-0.99 capture.
  Roles that write it (conv.out_proj, w2, out_proj) keep climbing through 256:
  0.71 -> 0.89, 0.77 -> 0.86, 0.75 -> 0.89 from table rank to 256. The
  liveness-sized table gives exactly these roles the smallest ranks (20, 110,
  42). Liveness reads the steepness of the tangent spectrum, not the breadth of
  the signal.
- **A real batch aims better than accumulated micro-batches, in the write roles.**
  The per-micro-batch aim's target keeps the micro-batch's noise weight
  (`Cov/4` at bs4); one bs16 batch's target has `Cov/16`. At table rank that is
  0.71 against 0.77 on w2, and within a point on the read roles.

Caveats that remain: these score ideal top-`r` eigenspaces of the targets, not
the tracked `Q`; the trained state is one 300-step run at `eta 0.01`.
Scratchpad `subspace_read.py`, 2026-09-28.

## Run ledger (Anima)

| run | what changed | outcome |
|---|---|---|
| 8 `live_floor` | rank 64 uniform, `1e-5`, derived live floor | first non-destructive run |
| 9 | calibrated table rounded up, `2e-5` | aesthetic, coherent; ended close to base |
| 10 `inner_lr5e5` | measured table, inner-side cross-attn K/V, `5e-5`, fallback `5e-6`, time frozen | perturbed-then-settled; anatomical breakage in places |
| 11 `lr1e4_cosine` | `1e-4`, cosine from warmup | cooler for longer |
| 12 `freeze_fallback` | fallback frozen, resized table, constant turn, `5e-5` | stable, slightly harmed, bland: no creative/aesthetic advance |
| 13 `acc4` | k=4, table from k=4 calibration (sum 595), `1e-4`, fallback `2e-6` | samples good, wants more; undertrained |
| 14 `k1_contrast` | run 13 with k=1 and `5e-5` | half the table dead at k=1; worst smoothed loss of 13-16; warrior slightly broken |
| 15 `rebaseline300` | run 13's config on 2026-09-14 code, 300 steps | tracks run 13; best bird; at step ~300 it moved warrior and bird where run 13 @300 barely had |
| 16 `patch_equalized300` | run 15 + patch-equalized loss, no timestep weight | spectrum unchanged vs 15; samples moved less than 15 (C1 closed) |
| 17 `eta040_300` | run 15 at `eta 0.04` | much faster first 100 steps, then level with 15; slightly rougher early samples that recover |

Sample reads above are the user's, from wandb, 2026-09-28. Run 15 against run 13
at step ~300 is not matched on schedule: run 15 decays from 225 and run 13 from
432, so run 15 moved more *at a lower lr*. Whether the 2026-09-14 trainer
(token-exact weights, unrounded final fold, fallback divided before the clip)
or something else did that is not separated. Whale, gown and ethereal barely
moved in any run.
