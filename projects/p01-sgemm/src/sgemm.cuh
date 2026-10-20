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

#include <cstring>
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
// The trailing `block` is an OPTIONAL launch-geometry override. A block with a
// zero extent means "launcher, use your own default" — zero is an unambiguous
// sentinel because a launch with 0 threads is never legal, so no real request
// can collide with it.
//
// It is a real parameter rather than a default argument on the typedef because
// default arguments are not part of a function's TYPE: they work when you call
// a function by name, not through a pointer. Harness code that loops over the
// variant table therefore passes the sentinel explicitly; the declarations
// below carry the default for direct calls.
using SgemmLauncher = void (*)(int M, int N, int K, float alpha,
                               const float *dA, const float *dB,
                               float beta, float *dC, dim3 block);

// Week 2 — one thread per output element, global memory only.
void sgemm_naive(int M, int N, int K, float alpha,
                 const float *dA, const float *dB,
                 float beta, float *dC, dim3 block = dim3(0, 0));

// Week 3 — tile A and B through __shared__ memory.
void sgemm_tiled(int M, int N, int K, float alpha,
                 const float *dA, const float *dB,
                 float beta, float *dC, dim3 block = dim3(0, 0));

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
// The variant table, defined ONCE here so that correctness/, bench/ and tools/
// all iterate the same list and a new kernel has to be registered in exactly
// one place.
//
// block_is_free records whether the launch geometry is actually tunable:
//   naive  one thread per output element, so any legal block works.
//   tiled  the shared tiles are declared [TILE_DIM][TILE_DIM] and the
//          cooperative load assumes one element per thread, so the block MUST
//          be TILE_DIM x TILE_DIM. Sweep that one by rebuilding with TILE=,
//          not by passing a block.
// ---------------------------------------------------------------------------
struct SgemmVariant {
    const char   *name;
    SgemmLauncher fn;
    bool          block_is_free;
};

inline const SgemmVariant *sgemm_variants(int *count) {
    static const SgemmVariant table[] = {
        {"naive", sgemm_naive, true},
        {"tiled", sgemm_tiled, false},
    };
    *count = (int)(sizeof(table) / sizeof(table[0]));
    return table;
}

// nullptr / "all" selects everything; otherwise an exact name match.
inline bool sgemm_variant_selected(const char *name, const char *only) {
    return (only == nullptr) || (strcmp(only, "all") == 0) || (strcmp(only, name) == 0);
}

// Parse "BXxBY" (e.g. "32x8") into a dim3 with z = 1. Returns false on anything
// it does not fully consume, so "16" and "16x16junk" are both rejected rather
// than silently half-read.
inline bool sgemm_parse_block(const char *text, dim3 *out) {
    int bx = 0, by = 0;
    char trailing = 0;
    if (sscanf(text, "%dx%d%c", &bx, &by, &trailing) != 2) return false;
    if (bx <= 0 || by <= 0) return false;
    *out = dim3((unsigned)bx, (unsigned)by);
    return true;
}

// Threads per block is a hard device limit (1024 on sm_86). Checking it here
// turns a bare cudaErrorInvalidConfiguration — which never says WHICH limit was
// hit — into a message naming the number and the cap.
inline bool sgemm_block_fits_device(dim3 block, int *limit_out) {
    int device = 0, max_threads = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&max_threads, cudaDevAttrMaxThreadsPerBlock, device));
    if (limit_out) *limit_out = max_threads;
    return (int)(block.x * block.y * (block.z ? block.z : 1)) <= max_threads;
}

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
