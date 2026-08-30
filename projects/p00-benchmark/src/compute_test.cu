// P00 — Compute test: a max-throughput fp32 kernel, compute achieved TFLOPS.
#include "../../../common/include/timing.cuh"

// TODO: write a kernel that is compute-bound, not memory-bound — e.g. many FMAs
// per loaded value (a tight loop of fused multiply-adds on register-resident data),
// so the measurement reflects ALU throughput rather than memory bandwidth.
__global__ void computeKernel(/* TODO: params */) {
    // TODO
}

int main() {
    // TODO: launch computeKernel with a grid/block sized to saturate all SMs.
    // TODO: time it with GpuTimer.
    // TODO: compute achieved TFLOPS = total_FLOPs / (ms / 1000) / 1e12.
    // TODO: compute theoretical peak FP32 TFLOPS = SMs * CUDA_cores_per_SM * clock_GHz * 2 (FMA).
    // TODO: print achieved vs theoretical, and the % of peak reached.
    return 0;
}
