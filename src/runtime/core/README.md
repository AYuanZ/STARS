# Runtime Core

This directory contains the first staged release of the STARS runtime core.

## Released components

- `cli.c`: command-line control interface for the runtime scheduler.
- `comm.c` and `comm.h`: Unix-domain socket communication primitives and
  runtime message definitions.
- `common.c` and `common.h`: shared logging, I/O, and utility functions.
- `cuda_defs.h`: minimal CUDA Driver API and NVML declarations used by the
  interception layer.
- `hook.c`: CUDA call interception and MPS execution-control logic.
- `Makefile`: validation and partial-build rules for this staged release.

## Release scope

This is a source snapshot, not yet a standalone runtime distribution. Some
internal dependencies referenced by `cli.c`, `comm.h`, and `hook.c` are not
part of this release. The Makefile therefore builds only the self-contained
`common.o` module by default. The remaining build dependencies and complete
runtime integration will be released incrementally.

Set `STARS_LOG_PATH` to select the runtime log destination. If unset, the
runtime uses `/tmp/stars-predict.log`.

Run the available checks with:

```bash
make check
```

The released files retain their existing copyright and license notices.
