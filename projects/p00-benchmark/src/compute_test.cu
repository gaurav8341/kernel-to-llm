// P00 — Compute test: a max-throughput fp32 kernel, compute achieved TFLOPS.
#include "utils.cuh"

// TODO: write a kernel that is compute-bound, not memory-bound — e.g. many FMAs
// per loaded value (a tight loop of fused multiply-adds on register-resident data),
// so the measurement reflects ALU throughput rather than memory bandwidth.
__global__ void computeKernel(float *d_out, float initial_val, int iterations) {
    // TODO
    // idx
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float acc0 = initial_val + float(tid);
    float acc1 = initial_val - float(tid);
    float acc2 = initial_val * 0.5f;
    float acc3 = initial_val * 0.5f;

    float factor = 1.00001f;

    // Main compute loop
    // #pragma unroll 16
    for(int i = 0; i < iterations; i++){
        float old_acc0 = acc0;
        acc0 = acc0 * factor + acc1;
        acc1 = acc1 * factor + acc2;
        acc2 = acc2 * factor + acc3;
        acc3 = acc3 * factor + old_acc0;
    }

    // this was also not required but to avoid compiler optimizations
    if (tid  == 0){
        d_out[0] = acc0 + acc1 + acc2 + acc3;  
    }
}

int main() {
    // TODO: launch computeKernel with a grid/block sized to saturate all SMs.

    float *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    int threads_per_block = 256;
    int num_blocks = prop.multiProcessorCount * 32;
    int iterations =100000;

    GpuTimer* fma_event = new GpuTimer("Compute Bound FMA");
    fma_event->start();
    computeKernel<<<num_blocks, threads_per_block>>>(d_out, 1.0f, iterations);
    fma_event->stop();

    float ms = fma_event->log();

    CUDA_CHECK(cudaFree(d_out));

    // compute the FLOPS
    uint fma_per_iter = 4; // 4 fma with all acc variable
    uint flops_per_fma = 2; // 2 FLOPs 1 multiplication 1 addition

    unsigned long long total_flops = (unsigned long long)threads_per_block * num_blocks
                                      * iterations * fma_per_iter * flops_per_fma;
    float achieved_tflops = total_flops / (ms / 1000) / 1e12;

    fprintf(stdout, "Achieved TFLOPS: %f\n", achieved_tflops);

    // ============================================================

    // CORES_PER_SM (utils.cuh) is architecture-specific and not queryable via
    // cudaDeviceProp — 128 FP32 cores/SM is correct for sm_86 (Ampere consumer,
    // GA10x); would need updating for other compute capabilities.

    int clockRate_kHz;
    CUDA_CHECK(cudaDeviceGetAttribute(&clockRate_kHz, cudaDevAttrClockRate, /*device=*/0));
    float clock_GHz = clockRate_kHz / 1e6f;

    unsigned long long theoretical_flops = (unsigned long long)prop.multiProcessorCount
                                            * CORES_PER_SM * clock_GHz * 1e9 * 2; // 2 is FMA factor
    float theoretical_tflops = theoretical_flops / 1e12;

    float pct_of_peak = achieved_tflops / theoretical_tflops * 100.0f;

    fprintf(stdout, "Theoretical TFLOPS: %f\n", theoretical_tflops);
    fprintf(stdout, "%% of peak reached: %.2f%%\n", pct_of_peak);

    return 0;
}
