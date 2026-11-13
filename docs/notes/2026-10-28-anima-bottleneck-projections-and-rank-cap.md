# Anima's bottleneck projections and the rank cap

Prose cut from `docs/SPEC.md` at the 2026-10 release cleanup. Verbatim: present tense is the time of writing, and "above" points into SPEC as of `d76ea99`.

Observed on Anima's `(2048, 64)` input and output projections: at rank 64 on
their 64-wide data side -- full rank, 100% occupancy -- training errored. Two
things were wrong at once and both fixes were real. The `side` hint moved them
onto the 2048-wide residual stream, which is where their basis belongs and which
brought occupancy from 25% into line with every other module at ~3%. And the
rank still has to be capped, because those modules are bottlenecks: too much
change in them destabilized the model regardless of which side was tracked.
