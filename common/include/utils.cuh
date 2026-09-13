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

    float log(){
        synchronize();
        fprintf(stdout, "%s : %f ms\n", event.c_str(), ms); 
        return ms;
    }

    // Get throughput per sec in GB/s
    float get_throughput(size_t size){
        synchronize(); // so ms is valid even if log() wasn't called first
        float throughput = (size / (ms / 1000)) / 1e9;
        fprintf(stdout, "Throughput GB/s: %f\n", throughput);
        return throughput;
    }

};