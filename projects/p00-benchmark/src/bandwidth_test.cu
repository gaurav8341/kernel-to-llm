// P00 — Bandwidth test: time cudaMemcpy of a large host<->device transfer, compute GB/s.
#include<iostream>
#include<cstdlib>

#include <cuda.h>

#include "../../../common/include/utils.cuh"

int main() {
    // TODO: allocate a ~1GB host buffer and a matching device buffer.
    char* h_buffer_pinned = nullptr;
    char* h_buffer_unpinned = nullptr;
    char* d_buffer = nullptr;

    size_t size = 1024 * 1024 * 1024;

    CUDA_CHECK(cudaMallocHost((void **)&h_buffer_pinned, size));

    // h_buffer_unpinned = (char*)malloc(size); // the C way
    // h_buffer_pinned = new char[size]; // the CPP way
    
    // The first argument is alignment size, Size(2nd arg) must be integral multiple of alignment(1st argument)
    h_buffer_unpinned = (char*)aligned_alloc(64, size); // Aligned C allocation

    CUDA_CHECK(cudaMalloc((void **)&d_buffer, size));

    // TODO: time cudaMemcpy(H2D) with GpuTimer, repeat a few times, take the best/median.
    // Host to GPU copy -- Sync
    GpuTimer* h2d_sync = new GpuTimer();
    
    h2d_sync->start();
    CUDA_CHECK(
        cudaMemcpy(
            d_buffer,
            h_buffer_pinned,
            size,
            cudaMemcpyHostToDevice
        )
    );
    h2d_sync->stop();
    h2d_sync->log("H2D-SYNC");

    // Host to GPU copy -- async
    CUDA_CHECK(
        cudaMemcpyAsync(
            d_buffer,
            h_buffer_pinned,
            size,
            cudaMemcpyHostToDevice
        )
    );

    // Host to GPU copy unpinned host and sync
    CUDA_CHECK(
        cudaMemcpy(
            d_buffer,
            h_buffer_unpinned,
            size,
            cudaMemcpyHostToDevice
        )
    );

    // TODO: compute achieved GB/s = bytes / (ms / 1000) / 1e9.
    // TODO: repeat for D2H, and optionally D2D.
    // GPU to HOST copy -- sync
    CUDA_CHECK(
        cudaMemcpy(
            h_buffer_pinned,
            d_buffer,
            size,
            cudaMemcpyDeviceToHost
        )
    );

    // GPU to HOST copy -- async
    CUDA_CHECK(
        cudaMemcpyAsync(
            h_buffer_pinned,
            d_buffer,
            size,
            cudaMemcpyDeviceToHost
        )
    );

    // GPU to HOST copy -- unpinned host and sync
    CUDA_CHECK(
        cudaMemcpy(
            h_buffer_unpinned,
            d_buffer,
            size,
            cudaMemcpyDeviceToHost
        )
    );


    // TODO: print achieved vs the 3050's theoretical VRAM bandwidth (see docs/hardware-spec-sheet.md).

    // lets free our memory
    CUDA_CHECK(cudaFreeHost(h_buffer_pinned));
    CUDA_CHECK(cudaFree(d_buffer));
    free(h_buffer_unpinned); // for malloc and aligned_alloc
    // delete[] h_buffer_unpinned; // for CPP declaration
    return 0;
}
