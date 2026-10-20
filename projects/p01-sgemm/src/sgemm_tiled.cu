// P01 Week 3 — Tiled SGEMM: stage A and B tiles in __shared__ memory.
//
// The point: a TILE_DIM x TILE_DIM tile of A loaded once is reused TILE_DIM
// times from shared memory instead of being re-read from global. That is the
// arithmetic-intensity win you computed for the naive kernel in Week 2.
#include "sgemm.cuh"

// TODO: tiled version.
//   - declare __shared__ float As[TILE_DIM][TILE_DIM], Bs[...]
//   - loop over the K dimension in ceil(K / TILE_DIM) tiles:
//       * cooperatively load one A tile and one B tile (zero-pad out-of-range
//         elements so ragged M/N/K stay correct)
//       * __syncthreads()
//       * accumulate the TILE_DIM partial products from shared memory
//       * __syncthreads() before overwriting the tiles on the next iteration
//   - write C = alpha * acc + beta * C, guarded by the M/N bounds
//
// Bank conflicts: shared memory has 32 banks, 4 bytes wide. For a
// [TILE_DIM][TILE_DIM] float array with TILE_DIM == 32, work out which bank
// element [r][c] lands in. Then: when the 32 lanes of a warp each read a
// different row of the SAME column, how many distinct banks do they hit?
// Find the fix yourself — but measure first with ncu
// (l1tex__data_bank_conflicts_pipe_lsu_mem_shared*) so the README records
// the before/after rather than just the fix.
__global__ void sgemmTiledKernel(int M, int N, int K, float alpha,
                                 const float *A, const float *B,
                                 float beta, float *C) {
    // TODO
}

// No default argument here: C++ allows it on the DECLARATION only (sgemm.cuh).
void sgemm_tiled(int M, int N, int K, float alpha,
                 const float *dA, const float *dB,
                 float beta, float *dC, dim3 block) {
    // The block shape is NOT free for this kernel: the shared tiles are
    // [TILE_DIM][TILE_DIM] and the cooperative load assumes one element per
    // thread, so anything else either leaves part of the tile unloaded or
    // indexes past it. A zero extent means "no override" and gets the only
    // legal shape; anything else is rejected loudly rather than silently
    // producing wrong answers.
    const dim3 blockSize = (block.x == 0 || block.y == 0)
                         ? dim3(TILE_DIM, TILE_DIM) : block;
    if (blockSize.x != TILE_DIM || blockSize.y != TILE_DIM) {
        fprintf(stderr,
                "sgemm_tiled: block must be %dx%d (shared tiles are sized by "
                "TILE_DIM), got %ux%u. Rebuild with TILE=%u to change it.\n",
                TILE_DIM, TILE_DIM, blockSize.x, blockSize.y, blockSize.x);
        exit(EXIT_FAILURE);
    }

    // TODO: grid covering M x N with a ceil-div, then launch sgemmTiledKernel
    // with <<<gridSize, blockSize>>> and CUDA_CHECK(cudaGetLastError()).
}
