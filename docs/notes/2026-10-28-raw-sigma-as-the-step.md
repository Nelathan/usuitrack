# Raw sigma as the step

Prose cut from `docs/SPEC.md` at the 2026-10 release cleanup. Verbatim: present tense is the time of writing, and "above" points into SPEC as of `d76ea99`.

**Raw `sigma` is a contrast ratio, not a magnitude, and it is no longer the
step.** Both `A` and `R` are quadratic in the gradient, so the division by
`mean(diag R)` cancels gradient scale exactly: rescaling `G` by any constant
leaves `Delta` and every `sigma_i` bit-identical (verified across a 1000x range).
What `sigma` measures is out-of-frame coupling against mean in-frame energy per
plane. It is scale-free in the gradient, which is half of why there is no clip, but
*not* rank-free -- a nuclear norm over `r` planes, growing roughly linearly in
`r` (~37.6 at rank 64 against ~80 at rank 128 on the same problem). Driving the
frame with it directly was the released rule until the polar transform above
replaced it, and the reason is that its self-annealing is an acquisition transient only:
`sigma` falls sharply over roughly twenty steps, then flattens and declines a few
percent across the remaining ~900. Rescaling it by a constant (`sigma_max`, the
tangent's Frobenius norm) cannot fix that, because a ratio cannot shrink as its
numerator does; capping the angle fails at the other end, costing acquisition.
Both were tried and reverted. The constant turn above sidesteps the question:
`sigma` decides nothing, so its acquisition transient cannot leak into the
frame's motion.
