# Scheduling

The default `scheduler.c` implements the STARS research policy. `queuing.c`
provides the task queue algorithms; the other scheduler source files preserve
the FIFO, EDF, EDF-noP, SJF, and SRTF baselines.

```bash
make SCHEDULER=stars   # from the repository root
make SCHEDULER=FIFO
make schedulers       # build all six variants
```

All variants share the runtime protocol. Host compilation is verified for all
six; GPU inference scheduling has not been validated end to end. The independent
NV/MPSE GPU baselines do not use these daemons. See [the run guide](../../docs/RUNNING.md).
