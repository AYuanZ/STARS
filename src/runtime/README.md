# Runtime

[`core/`](core/) contains CUDA interception, scheduler communication, the
runtime client, and the CLI. Build from the repository root with `make`;
artifacts appear in `build/`.

See [the run guide](../../docs/RUNNING.md) for environment variables and workload
integration requirements. The independent NV/MPSE examples do not preload this
library or require the STARS scheduler.
