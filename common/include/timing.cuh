// cudaEvent_t timing helpers — generic plumbing, reused from every projects/pXX/.
// Not a learning target: nothing here should ever contain kernel logic.
#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err__));                               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Wraps a pair of cudaEvent_t for GPU-side timing of a code region.
// Usage:
//   GpuTimer timer;
//   timer.start();
//   myKernel<<<grid, block>>>(...);
//   float ms = timer.stopMs();
struct GpuTimer {
    cudaEvent_t startEvent, stopEvent;

    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&startEvent));
        CUDA_CHECK(cudaEventCreate(&stopEvent));
    }

    ~GpuTimer() {
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
    }

    void start() { CUDA_CHECK(cudaEventRecord(startEvent)); }

    // Records the stop event, synchronizes, and returns elapsed milliseconds.
    float stopMs() {
        CUDA_CHECK(cudaEventRecord(stopEvent));
        CUDA_CHECK(cudaEventSynchronize(stopEvent));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
        return ms;
    }
};
