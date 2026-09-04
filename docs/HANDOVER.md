# Handover

Written 2026-09-05. Orientation, traps, what to do next; `PLAN.md` carries the
reasoning. Harness invocations live in `AGENTS.md`.

## Where things stand

Branch `constant-basis-step`, 44 tests green. The release package was **not
touched** this session: the work was two Anima training runs, two calibrations,
the ai-toolkit integration that made them possible, and these docs. Both repos
are committed.

The ai-toolkit side now carries role labels, per-role rank (`rank_table`), a
`calibrate_rank` mode with the `RankCalibrator` draining through
`pop_diagnostics`, and the cross-attention side map below. Its configs live in
`config/` and are gitignored there, so the run configurations survive only on
disk: `calibrate_rank_anima_usuitrack.yaml`,
`train_full_fine_tune_anima_usuitrack_9.yaml` and `..._10.yaml`.

**Anima has a measured rank table and a full run behind it.** Calibration at
`r_cal 256` (200 steps, bs4/768, lr `2e-5`) settled -- under 10% drift, `std` 1-6
planes. **The table is the measured live count itself**, which is what lands
`live_fraction` at ~0.95: rank equals the planes that were live, so almost all of
them stay live. Run 9 shipped a table rounded *up* to multiples of 8 instead,
which is why it ran at 0.79 with a fifth of every basis idle. Do not round up,
and where a role's mean sits below its median take the mean -- low is safe.

Run 9 (`anima_usuitrack_9_rank_table`, wandb `fvehhrs1`) ran all 2304 steps clean
on that rounded table at lr `2e-5` / fallback `5e-6`, beta `0.9`. The table, the
role structure it revealed, and the memory numbers are in `PLAN.md` P13.

**The mechanism reads moved; the loss did not.** Against run 8 (uniform rank 64,
lr `1e-5`), over the last 500 steps:

| steps 1000-1700 | run 8 | run 9 | run 10 |
|---|---:|---:|---:|
| loss | 0.1534 | 0.1521 | 0.1533 |
| loss, last 200 steps | 0.1582 | 0.1499 | 0.1482 |
| `tangent_live_fraction` | 0.472 | 0.787 | 0.854 |
| `turn_fraction` | 0.367 | 0.491 | 0.510 |
| `transport_speed` | 0.0020 | 0.0050 | 0.0051 |
| `update_to_param_ratio` | 3.56e-6 | 1.12e-6 | 2.47e-6 |
| ... last 200 steps | 5.41e-7 | 1.71e-7 | 7.06e-7 |

The table bought exactly what it was built to buy -- a basis that is two-thirds
live instead of half, turning and transporting more -- at 0.082 GB of optimizer
state against 0.185 GB. Loss sits within noise of the baseline, which is the
expected outcome on this lane and not a verdict.

**Run 9's samples are reviewed and it is a success**: highly aesthetic, coherent
images, with some bad ones. Its failure was that it barely moved -- the anneal
left the model close to base with a minor style shift rather than a creative one.
The displacement numbers said the same thing, which is what run 10 answers.

**Run 10 is the current head of this lane** (`anima_usuitrack_10_inner_lr5e5`,
wandb `7ffli01x`), clean over 2304 steps: lr `5e-5` annealed from 75% down to
`1e-5`, the unrounded table, cross-attention K/V on the inner side, and the time
module frozen. It moves 2.2x more per step than run 9 and, in its last 200 steps,
4x more -- the shallow anneal leaves it still learning where run 9 had stopped.
Its 65 samples are **unreviewed and are the next thing to look at.**

**The step is far too small, and that is the whole finding.**
`update_to_param_ratio`, the honest "how far did the weights move" read:

Anima runs 30-70x below the LFM arm the cosine rule was calibrated on
(`7.6e-5`), and run 9 sat 68x below it with its last quarter near-frozen. Run 10
recovered most of that: `2.47e-6` mid-run, `7.06e-7` at the end.

Part of the drop is the leaner table and that part is fine -- **less rank is less
signal to act on, not a step-size debt to normalize away.** The lr rise this lane
needs is owed to ortho-first instead: the moment now cancels *after*
orthogonalization, which costs 2-4x in effective step. `3e-5` is the safe setting
for the next run.

## What to do next

1. **Review run 10's samples** against run 9's. Everything else here is a
   mechanism read, and mechanism reads have never ranked designs on this lane.
   `/mnt/luna/ai/output/anima_usuitrack_10_inner_lr5e5/samples`, wandb `7ffli01x`.
2. **Pin the sizing rule** with a calibration at `r_cal` = the table itself, 12
   minutes, `PLAN.md` P13 item 4. It converts "the measured count lands ~0.85"
   into a rule that hits the target it aims at.
3. **`fallback_lr` has never been swept on this lane.** It sat at `5e-6` through
   runs 9 and 10 while the matrix lr moved 2.5x, so the AdaLN modulation linears
   and every norm gain are training at a ratio nobody chose. They are also the
   weights that went non-finite on this model before, so this is a careful sweep,
   not a bold one.
4. Unchanged from before: the cosine spread read (P15), then `eta` (P16).

## Traps

Harness invocations and the operational traps (stale lab fork, `sys.path`,
`nohup`, reading ai-toolkit's sqlite, output on `/mnt/luna`, end-of-job saves, LR
anchoring, noise floors) are in `AGENTS.md`, "Running things". What follows is
research judgement.

**Under ortho-first, `beta` is also an LR.** The step carries the agreement
`||M||/||O||` -- `0.23` at `beta 0.9`, stable across a 4.4x LR change because it
is a property of the gradients. So two arms at the same nominal LR and different
`beta` sit at different effective steps by up to 8x. Set each arm's LR to
`lr_ref * agreement_ref / agreement_arm` and check `update_to_param_ratio`
afterwards. This is the trap that has caught four separate comparisons.

**A norm read cannot detect a small mean.** `moment_persistence` bounds the
mean's *energy* share, and a mean far too small to move it still decides where a
thousand accumulated steps land. Reading "persistence ~ 0" as "the memory holds
nothing" produced a confident, wrong conclusion; the `beta 0` arm refuted it by
`1.8e-2`.

**Loss does not rank tracker designs, and neither does geometry.** The
`side=right` arm maximized a geometry read while loss degraded; the calibrated
LFM table hit `live_fraction` 0.946 exactly and moved loss not at all;
ortho-first moved every mechanism metric the right way and moved loss not at all;
and now Anima's table raised live fraction by two thirds, at less than half the
optimizer state, for `2e-3` of loss in the wrong direction. The one hedge: a
better-conditioned basis has shown as subjective quality on Anima where loss does
not move.

**Role structure does not transfer between architectures.** On LFM the MLP roles
were the rank hogs (`w1` 210, `w3` 195) and attention was lean. On Anima it
inverts: attention carries the rank (self-attention K at 52 planes against V's
19) and both feed-forward sides sit near 20 despite being the largest weights.
Calibrate per model; do not port a table's shape.

**The measured live count lands `live_fraction` at ~0.85, not 0.95.** Rank set
to the count measured at `r_cal 256` gave run 10 `0.854`, because the count is not
rank-invariant: at ranks of 20-51 about 15% fewer planes clear the Gram floor than
the same roles show at 256. To sit at 0.95 the rank goes slightly below the
measured count. `PLAN.md` P13 item 4 has the 12-minute probe that would pin it.

**Take the lower of mean and median.** Under-provisioning is the safe
direction -- a plane the basis lacks is a direction the tracker declines to move,
not an error -- so the statistic to take is whichever runs lower for that role.
LFM's right-skewed roles had mean above median; several Anima roles run mean
below it, and there the mean is the base.

**Price optimizer changes end to end, not in the kernel.** A 1.5-2x faster polar
map produced a 0.8% *slower* run. Forward and backward dominate; the optimizer's
largest term is invisible in the wall clock. Run 9 was launched with
`compile_tensor_kernels: false` on that evidence -- thirteen role shapes to
compile, a path never verified under `release_matrix_grads`, and no measurable
upside.

**The overshoot meter has not been read cleanly on Anima.** Run 8 reads
`-0.0130`, run 9 `+0.0030`, run 10 `+0.0040` -- but run 8 predates ortho-first,
and run 10 changed lr, table, cross-attention side and the frozen time module at
once. Every Anima cosine delta so far is confounded. Its zero crossing picked the
LR on LFM (P14) and that has never been reproduced here; a clean read needs a run
that changes lr alone.

**Grad norm does not track loss quality here.** Across the `beta` sweep it fell
monotonically (6.24 / 5.94 / 5.12) while target loss got monotonically worse.
Smoother is not better.

**Per-matrix step allocation is loss-invisible so far.** Damping each matrix by
its own agreement changes contested steps 20x and loss by less than a floor, and
what difference there is favours *not* damping. Weigh that against any new
per-matrix scaling proposal.

**Rank calibration needs a long enough run to settle.** Under some side
configurations the high-`frac` roles were still drifting up at 200 steps; 500
settled them on LFM (`std` 2-9 planes). Anima's 200 steps at `r_cal 256` drifted
under 10% and was accepted deliberately, since under-provisioning is the safe
direction. The first window is always dropped as the acquisition transient.

**mean vs median across a role is a real choice.** They agree for homogeneous
roles; where a role is skewed they do not, and which one is conservative depends
on the skew's direction -- see the two entries above.
