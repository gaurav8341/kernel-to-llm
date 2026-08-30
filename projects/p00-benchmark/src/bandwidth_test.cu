// P00 — Bandwidth test: time cudaMemcpy of a large host<->device transfer, compute GB/s.
#include "../../../common/include/timing.cuh"

int main() {
    // TODO: allocate a ~1GB host buffer and a matching device buffer.
    // TODO: time cudaMemcpy(H2D) with GpuTimer, repeat a few times, take the best/median.
    // TODO: compute achieved GB/s = bytes / (ms / 1000) / 1e9.
    // TODO: repeat for D2H, and optionally D2D.
    // TODO: print achieved vs the 3050's theoretical VRAM bandwidth (see docs/hardware-spec-sheet.md).
    return 0;
}
