# Modeling

`interference.c` contains the empirical interference and SM-scaling formulas
extracted from the original default scheduler without changing coefficients.
The default scheduler calls these functions. `lrmodel.c` and `lrmodel.h`
contain the original multivariate regression implementation linked into the
runtime. These coefficients have not been refitted for new hardware.

The original `predict.c` always returned 1.0; that placeholder is deliberately
excluded. Full training data and offline fitting scripts are not part of this
snapshot. See [the run guide](../../docs/RUNNING.md).

Archived raw colocated-task duration captures are published in
[`data/interference/`](../../data/interference/), with source paths, checksums,
and notes on repeated headers and identifier encoding. The six models' kernel
profiles are available in [`data/profiling/`](../../data/profiling/).
