#include<cuda.h>
#include<cuda_runtime.h>

#include<string>
#include<iostream>

#include<cstdio>
#include<cstdlib>

using namespace std;


#define CUDA_CHECK(call)                                                                        \
{                                                                                               \
    cudaError_t cuda_err = (call);                                                              \
    if(cuda_err != cudaSuccess){                                                                \
        fprintf(stderr, "========================================\n");                          \
        fprintf(stderr, "CUDA ERROR: %s\n", cudaGetErrorName(cuda_err));                        \
        fprintf(stderr, "%s:%d:%s\n", __FILE__, __LINE__, cudaGetErrorString(cuda_err));        \
        fprintf(stderr, "========================================\n");                          \
        exit(EXIT_FAILURE);                                                                     \
    }                                                                                           \
}

struct GpuTimer{

    cudaEvent_t startEvent, stopEvent;
    float ms = 0.0f;
    string event;

    GpuTimer(string event){
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

};