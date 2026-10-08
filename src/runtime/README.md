# Runtime

This component provides the GPU sharing runtime, execution control, and the
interfaces used to apply scheduling and resource-allocation decisions.

The first staged source release is available in [`core/`](core/). It contains
the command-line interface, communication primitives, common utilities,
minimal CUDA declarations, and CUDA interception layer. Dependencies needed
for a complete runtime build will be released incrementally.
