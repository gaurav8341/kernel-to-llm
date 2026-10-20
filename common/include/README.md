# common/include/

Generic, kernel-agnostic helpers meant to be reused from multiple `projects/pXX/`.

Added as they're actually needed by a project rather than pre-built (per `ARCHITECTURE.md`):
- `utils.cuh` — `CUDA_CHECK` error macro and `GpuTimer`, a `cudaEvent_t` timing wrapper (P00)
- `check.cuh` — tolerance comparison of a device buffer against a host reference, with a
  one-line pass/fail report (P01)

Nothing here should implement a kernel — only plumbing shared *around* kernels.
