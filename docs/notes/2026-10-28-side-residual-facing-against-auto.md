# Side: residual-facing against auto at the edges

Prose cut from `docs/SPEC.md` at the 2026-10 release cleanup. Verbatim: present tense is the time of writing, and "above" points into SPEC as of `d76ea99`.

The difference concentrates at the edges of the network, where the data side is
narrow. Shape-only `auto` tracked the 68- and 64-wide data sides of a model's
input and output projections and produced its two worst-conditioned bases
(orthonormality error 1.1e-03 and 1.0e-03 against ~5e-04 elsewhere); the hint
spends the same rank budget inside the 2048-wide stream, where the complement
the Oja tangent needs is ample.
