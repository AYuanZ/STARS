# STARS source code

- `runtime/core/`: CUDA interception, client protocol, CLI, and utilities.
- `profiling/`: kernel-profile loading, six actual inference workloads in
  `workloads/`, and optional vector-add execution checks.
- `modeling/`: linear regression and the scheduler's empirical interference model.
- `scheduler/`: STARS and FIFO/EDF/EDF-noP/SJF/SRTF scheduling algorithms.

Build with `make` at the repository root. See [the run guide](../docs/RUNNING.md)
for dependencies, source provenance, and validation scope.
