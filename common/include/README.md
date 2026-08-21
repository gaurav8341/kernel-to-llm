# common/include/

Generic, kernel-agnostic helpers meant to be reused from multiple `projects/pXX/`.

Candidates (per `ARCHITECTURE.md`), to be added as they're actually needed by a project rather than pre-built:
- a `cudaEvent_t` timing wrapper
- a correctness-check helper (compare device output vs a numpy/PyTorch reference within tolerance)

Nothing here should implement a kernel — only plumbing shared *around* kernels.
