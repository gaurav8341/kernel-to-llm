#pragma once

#include<cuda.h>
#include<cuda_runtime.h>

#include<string>
#include<iostream>

#include<cstdio>
#include<cstdlib>

#define CORES_PER_SM 128 // cores per sm in the rtx 3050 laptop edition

// do/while(0) so the macro stays a single statement — a bare { } block would
// break `if (x) CUDA_CHECK(...); else ...`
#define CUDA_CHECK(call)                                                                        \
do {                                                                                            \
    cudaError_t cuda_err = (call);                                                              \
    if(cuda_err != cudaSuccess){                                                                \
        fprintf(stderr, "========================================\n");                          \
        fprintf(stderr, "CUDA ERROR: %s\n", cudaGetErrorName(cuda_err));                        \
        fprintf(stderr, "%s:%d:%s\n", __FILE__, __LINE__, cudaGetErrorString(cuda_err));        \
        fprintf(stderr, "========================================\n");                          \
        exit(EXIT_FAILURE);                                                                     \
    }                                                                                           \
} while(0)

struct GpuTimer{

    cudaEvent_t startEvent, stopEvent;
    float ms = 0.0f;
    std::string event;

    GpuTimer(std::string event){
        CUDA_CHECK(cudaEventCreate(&startEvent));
        CUDA_CHECK(cudaEventCreate(&stopEvent));
        this->event = event; 
    }

    ~GpuTimer(){
        CUDA_CHECK(cudaEventDestroy(startEvent));
        CUDA_CHECK(cudaEventDestroy(stopEvent));
    }

    void start(){
        CUDA_CHECK(cudaEventRecord(startEvent));
    }

    void stop(){
        CUDA_CHECK(cudaEventRecord(stopEvent));
    }

    void synchronize(){
        CUDA_CHECK(cudaEventSynchronize(startEvent));
        CUDA_CHECK(cudaEventSynchronize(stopEvent));
        CUDA_CHECK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    }

    // Elapsed ms, synchronized but SILENT. log() is the chatty version; this is
    // the one a caller wants when it is formatting its own output, or timing in
    // a loop where 100 printed lines would be noise instead of data.
    //
    // It calls synchronize() itself on purpose: reading the `ms` field directly
    // after stop() returns whatever was there before — 0.0f on the first use —
    // because stop() only RECORDS the event, it does not wait for it.
    float elapsed(){
        synchronize();
        return ms;
    }

    float log(){
        fprintf(stdout, "%s : %f ms\n", event.c_str(), elapsed());
        return ms;
    }

    // Get throughput per sec in GB/s
    float get_throughput(size_t size){
        synchronize(); // so ms is valid even if log() wasn't called first
        float throughput = (size / (ms / 1000)) / 1e9;
        fprintf(stdout, "Throughput GB/s: %f\n", throughput);
        return throughput;
    }

    // Compute rate in TFLOPS, given the number of USEFUL floating-point
    // operations in the work that was timed. The counterpart to
    // get_throughput(): GB/s is the figure of merit for a memory-bound kernel,
    // TFLOPS for a compute-bound one.
    //
    // `flops` is a double, not a size_t, because callers compute it as
    // 2*M*N*K and 2 * 4096^3 = 1.37e11 overflows a 32-bit int. Double carries
    // 53 bits of mantissa, so it is exact well past anything a GPU can run.
    //
    // Returns 0 on a zero elapsed time rather than inf: a kernel that failed to
    // launch times at exactly 0.000 ms, and inf prints as a spectacular,
    // completely fictional rate.
    float get_tflops(double flops){
        synchronize();
        if (ms <= 0.0f) {
            fprintf(stdout, "TFLOPS: n/a (elapsed time was 0 — did the kernel launch?)\n");
            return 0.0f;
        }
        float tflops = (flops / (ms / 1000.0)) / 1e12;
        fprintf(stdout, "TFLOPS: %f\n", tflops);
        return tflops;
    }

};