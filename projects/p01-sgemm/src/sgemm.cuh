#pragma once

// P01 — shared contract between the kernels (src/), the benchmark harness (bench/)
// and the correctness checker (correctness/).
//
// ---------------------------------------------------------------------------
// LAYOUT: everything in this project is ROW-MAJOR.
//
//   A is M x K      A[m][k]  ->  A[m * K + k]      row stride K
//   B is K x N      B[k][n]  ->  B[k * N + n]      row stride N
//   C is M x N      C[m][n]  ->  C[m * N + n]      row stride N
//
// The operation is the BLAS one, not a plain product:
//
//   C = alpha * (A * B) + beta * C
//
// Two reasons for the BLAS form over C = A*B. It costs one extra multiply-add
// per output element, and in exchange (a) the signature lines up with
// cublasSgemm so the benchmark compares like with like, and (b) alpha/beta are
// how every real GEMM call site in P05/P08 will look.
//
// cuBLAS is COLUMN-major. That mismatch is dealt with entirely inside bench/,
// and it needs NO data conversion and no transpose kernel: a row-major M x N
// matrix is bit-for-bit a column-major N x M matrix, so only the argument
// order and the leading dimensions change. Your kernels never see this — they
// get plain row-major buffers.
// ---------------------------------------------------------------------------

#include "utils.cuh"

#include <random>

// Every SGEMM variant matches this signature, so bench/ and correctness/ can
// loop over {naive, tiled, cuBLAS, ...} instead of repeating a block per
// kernel. All pointers are DEVICE pointers.
//
// Contract for a launcher: it does the grid/block math and launches. It does
// NOT allocate, and it does NOT synchronize.
//
// Why no sync: a sync inside the launcher would put a full pipeline drain
// inside every timed iteration in bench/, so you would be measuring
// launch + drain rather than the kernel, and the caller would lose any say in
// when that cost is paid. The caller syncs — right before it reads the result,
// or right before it stops the timer.
using SgemmLauncher = void (*)(int M, int N, int K, float alpha,
                               const float *dA, const float *dB,
                               float beta, float *dC);

// Week 2 — one thread per output element, global memory only.
void sgemm_naive(int M, int N, int K, float alpha,
                 const float *dA, const float *dB,
                 float beta, float *dC);

// Week 3 — tile A and B through __shared__ memory.
void sgemm_tiled(int M, int N, int K, float alpha,
                 const float *dA, const float *dB,
                 float beta, float *dC);

// Tile edge for the tiled kernel, overridable so `make ptx TILE=32` can dump
// PTX for a different size without editing this file.
//
// The #ifndef is what makes -DTILE_DIM work. Without it, a command-line -D and
// this #define collide: the compiler warns "TILE_DIM redefined", this header
// wins, and the flag silently does nothing.
//
// 16 to start: 16x16 = 256 threads/block, and two 16x16 fp32 tiles = 2KB of
// shared memory per block. sm_86 allows 1024 threads/block and 100KB of shared
// memory per block, so 32 is also legal (1024 threads, 8KB) — it is just a
// different occupancy trade, which is the point of being able to override it.
#ifndef TILE_DIM
#define TILE_DIM 16
#endif

// ---------------------------------------------------------------------------
// Host helpers both mains need. `inline` because they are defined in a header
// that more than one translation unit includes — see the note in check.cuh.
// ---------------------------------------------------------------------------

// Fill n floats with deterministic values in [-1, 1).
//
// The seed is a parameter, not a constant inside, for two separate reasons:
//
//   1. A and B must get DIFFERENT seeds. If the seed were fixed internally,
//      every call would replay the same sequence and A would equal B — a much
//      weaker test, because a kernel that confuses its two operands would
//      still pass.
//   2. bench/ and correctness/ must get the SAME seeds, so the inputs behind a
//      timing run and a correctness run are identical and a benchmark number
//      always refers to a verified kernel.
//
// mt19937 rather than rand(): rand()'s sequence is implementation-defined, so
// two binaries could disagree. mt19937 is specified by the standard, and the
// mapping below is plain arithmetic on its raw output, so the values are
// identical everywhere — no dependence on uniform_real_distribution, which is
// NOT specified bit-for-bit across implementations.
inline void initialize_matrix(float *buf, size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    for (size_t i = 0; i < n; ++i) {
        const double u = (double)rng() / 4294967296.0;   // [0, 1)
        buf[i] = (float)(u * 2.0 - 1.0);                 // [-1, 1)
    }
}

// Useful FLOPs in an M x N x K GEMM: one multiply and one add per (m, n, k),
// so 2*M*N*K. (The alpha/beta epilogue is 2 more FLOPs per output element —
// O(M*N) against the O(M*N*K) inner work, so it is conventionally ignored.)
//
// double, not int: 2 * 4096^3 = 1.37e11, and INT_MAX is 2.15e9. In int this
// overflows and silently corrupts the TFLOPS column at exactly the largest
// size you care about most.
inline double gemm_flops(int M, int N, int K) {
    return 2.0 * (double)M * (double)N * (double)K;
}
