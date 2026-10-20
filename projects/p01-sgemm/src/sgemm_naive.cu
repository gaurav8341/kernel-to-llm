// P01 Week 2 — Naive SGEMM: one thread per output element, global memory only.
//
// This is the baseline you profile and then beat. Expect it to be memory bound:
// each thread reads a full row of A and a full column of B from global memory,
// so the same data gets re-fetched by every thread in the row/column.
#include "sgemm.cuh"

// TODO: one thread computes one C[row][col].
//   - map threadIdx/blockIdx -> (row, col)
//   - guard against row >= M || col >= N (M, N need not be tile multiples)
//   - accumulate over k in a local float, then write C = alpha * acc + beta * C
// Remember row-major indexing: A[row * K + k], B[k * N + col], C[row * N + col].
__global__ void sgemmNaiveKernel(int M, int N, int K, float alpha,
                                 const float *A, const float *B,
                                 float beta, float *C) {
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if(col >= N || row >= M){
        return;
    }

    float psum = 0.0f;
    for (int i = 0; i < K; i++){
        psum += A[K * row + i] * B[i * N + col];
    }

    C[row * N + col] = alpha * psum + beta * C[row * N + col];
}

// No default argument here: C++ allows it on the DECLARATION only (sgemm.cuh),
// and repeating it in the definition is an error.
void sgemm_naive(int M, int N, int K, float alpha,
                 const float *dA, const float *dB,
                 float beta, float *dC, dim3 block) {
    // TODO: pick a 2D block (16x16 and 32x32 are the two obvious choices — the
    // difference in how they coalesce is worth measuring) and a grid that covers
    // M x N with a ceil-div, then launch.
    //
    // One thing to decide deliberately, because it decides whether this kernel
    // coalesces at all: which of threadIdx.x / threadIdx.y maps to the column.
    // Adjacent threads in a warp should touch adjacent addresses in C and B.

    // here x is COL and y is ROW
    //
    // A zero extent is the "no override" sentinel from sgemm.cuh, so an
    // unspecified block falls back to this kernel's own default and every
    // existing call site keeps its old behaviour.
    const dim3 blockSize = (block.x == 0 || block.y == 0) ? dim3(16, 16) : block;

    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (M + blockSize.y - 1) / blockSize.y);

    sgemmNaiveKernel<<<gridSize, blockSize>>>(M, N, K, alpha, dA, dB, beta, dC);
    CUDA_CHECK(cudaGetLastError());
}
