# Open design questions

What we are unsure about, how each would be settled, and the evidence that has
moved each answer. `SPEC.md` says what the optimizer does; `FACTS.md` holds the
numbers; `ARCHIVE.md` holds everything closed. See `AGENTS.md` for how this file
is worked.

Questions are grouped so a session can take **one group** and leave the rest
alone. Inside a group the items share runs, reads, or code. Nothing here has a
run yet; each "how to settle" is a proposal, not an agreed arm.

---

## Group L. Scale

### L1. Does `lr` transfer from LFM2.5-350M to LFM3-1.7B?

`lr` is a relative step (SPEC step 8), so the claim is that one value means the
same dynamics at any size. The 1e-4 default pushed for loss and overshot:
`grad_moment_cosine` stayed negative until late in the run. Arms (user,
2026-10-28): constant `5e-5` and `2e-5` at both sizes, the standard arm
otherwise, LFM3 pinned to revision `545fc8f`, `o_proj` on the output side like
LFM2.5's `out_proj`. A 20-step LFM3 probe reads step time and peak memory
(headroom).

Read: mechanism decides, at equal `lr` across sizes -- `relative_step`,
`grad_moment_cosine`'s level and sign, `grad_capture` and
`tangent_live_fraction` (does `rank_fraction 0.1` transfer too). Loss vetoes:
target gain against forgetting, each from that model's own initial losses.

Predictions, written before the runs: `relative_step` equals `lr` at both
sizes. Cosine sits nearer zero at `5e-5` and positive at `2e-5`, and the same
`lr` gives the same sign at both sizes if transfer holds. Capture and live
fraction land within a few points of the 350M's 0.77 and 0.83, since rank is
the same fraction of each square side. `2e-5` gains less target and forgets
less than `5e-5` at both sizes. Unknowns: whether bs16 fits at 1.7B (estimate
~6-8 GB peak), and its step time (estimate ~5 s).

Evidence (FACTS, 2026-10-28/29). 350M half as predicted: the plateau cosine is
negative from step ~500 at `5e-5`, positive throughout at `2e-5`; `2e-5` gains
less and forgets less; `relative_step` reads `lr`; capture and live fraction
do not move with lr. LFM3 half not run: the peak is set by the batched polar in
`optimizer.step`, not by activations, and bs16 ran once and failed twice since,
as the desktop's share of the card moved. Ways through, none taken: free the
card (close Discord and zapzap); bound the polar's batch in the release, which
caps a transient that grows with depth (a release change); or a smaller batch,
which changes the arm and its lr.

---

## Group N. Weight norm

### N1. Can the step force an ideal weight norm?

The step is relative to `||W||` (SPEC step 8), so a zero matrix never moves
and a small-norm init moves slowly in absolute terms, however much its gradient
says. Fine-tuning starts from trained norms; pretraining inits are non-zero but
far from the final norm. The user's question: since the design favours geometry,
can it target a norm rather than inherit one -- size the step against an ideal
`||W||` for the shape, or move the norm toward it explicitly? Open: what "ideal"
means (an init-scale rule, a measured trained-model norm per role), and whether
that is the optimizer's job at all. How to settle: a small model trained from
scratch, where the init is a free variable.

### N2. Rising weight norms, and weight decay

Weight norms rise over runs (observed, unmeasured as a table). A step at a fixed
fraction of `||W||` compounds when it correlates with `W`, so the rise may be
N1's issue seen from the other end. `weight_decay` is in the release and
untested on either lane. How to settle: `||W||` per role over a standard LFM arm
first (cheap, a lab read), then a decay arm against it.

---

## Group S. Subspace gradients

### S1. Accumulation under `release_matrix_grads`

Release frees each gradient in its hook, so it takes one backward per step; a
model whose weights plus gradients do not fit needs both release and
accumulation. Projecting per micro-batch is rank-sized, but the Oja tangent is
quadratic in `G` (`A = G^T G Q`), so summed per-micro-batch tangents are not the
summed gradient's tangent. The in-optimizer fold that did this was removed
(`ARCHIVE.md`, "accumulation is removed"); its open questions (B1-B4 there) are
this one's prior.

### S2. Projected gradients from projected activations (molegrad)

Project activations into the subspace in the forward pass so that backward
produces `Z` directly and the full gradient never exists. Proven to work and
slower; it needs an autograd replacement, since PyTorch has no hook for a
projected weight gradient. The question is whether it can be made fast. A
prerequisite read: tracking the frame from activations rather than from `G`,
evaluated with the lab's transport reads.

---

## Group T. Side

### T1. Anima under `auto`, with a few exceptions

ai-toolkit carries a hardcoded side table (residual-facing, plus cross-attention
K/V). Since liveness masking, `auto` may do as well with one or two exceptions.
Settling it needs the per-role reads the release no longer has: open is how the
ai-toolkit lane gets `LabDiagnostics`, which lives in this repo's `experiments/`.

---

## Parked at release

Moved whole to `ARCHIVE.md`, "PARKED at release"; reopen by moving the text
back.

- A1. Does the bf16 stochastic-rounding walk harm the model?
- A2. A walk-aware displacement read.
- D1. The matrix lr's peak and decay shape on each lane.
- D6. LFM run-to-run spread: three seeds of the default arm.
- E4. An arrival sensor for the constant turn.
- E5. The `min(m,n)/2` cap.
- E7. Anima's `rank_fraction`.
- E9. Which roles move, and should step or rank follow them?
- F1-F5. Per-matrix lr from the overshoot meter; magic numbers; device syncs;
  stratified timesteps; BFD packing on LFM.
- "Already tried. Do not repeat." is there too.
