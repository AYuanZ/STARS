# Runtime core

The complete host build now produces `build/libstars.so`,
`build/stars-scheduler`, and `build/starsctl` from the repository root.
`make` and `make check` in this directory forward to the root Makefile.

The runtime links the client, communication utilities, kernel-profile loader,
regression code, and CUDA interception layer. It retains STARS symbol names;
CUDA and NVML are resolved dynamically. The original copyright notices and
third-party licenses remain in their source files.

See [the run guide](../../../docs/RUNNING.md) for configuration and the limits
of the validation. Set `STARS_LOG_PATH` to choose the runtime log destination;
its default is `/tmp/stars-predict.log`.
