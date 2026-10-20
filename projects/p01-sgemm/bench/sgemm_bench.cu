// P01 — benchmark harness: naive vs tiled vs cuBLAS at 512^2, 1024^2, 4096^2.
//
// Output goal: a markdown table you paste into the project README:
//   | Kernel | Time (ms) | TFLOPS | % of cuBLAS | Correct |
//
// Usage you're aiming for:
//   ./bin/sgemm_bench                 # 512, 1024, 4096
//   ./bin/sgemm_bench 2048            # any sizes from argv
//
// YOUR TASK — write it in this order, compiling after each step.
#include "sgemm.cuh"
#include "check.cuh"

#include <cublas_v2.h>

// TODO 1: a CUBLAS_CHECK macro, same idea as CUDA_CHECK in utils.cuh.
//   cuBLAS returns cublasStatus_t, not cudaError_t — compare against
//   CUBLAS_STATUS_SUCCESS. Why the do { } while (0) wrapper?

// TODO 2: a cuBLAS baseline that matches your SgemmLauncher signature.
//   Create one cublasHandle_t in main() (creating it is slow — don't do it
//   per call) and call cublasSgemm.
//   THE PUZZLE: cuBLAS assumes column-major, your buffers are row-major.
//   You can solve it without transposing any data. Hints:
//     - a row-major M x N matrix has the same bytes as a column-major N x M
//       matrix — i.e. its transpose
//     - (A*B)^T = B^T * A^T
//   Work out which pointer goes in which argument, and what M/N/lda/ldb/ldc
//   become. Verify by checking cuBLAS's output against your CPU reference on a
//   NON-square shape (e.g. 64 x 128 x 32) — square shapes hide a swapped M/N.

// TODO 3: time one launcher.
//   - warm up first (a few untimed calls). What is the first call paying for
//     that later calls aren't?
//   - time with GpuTimer (utils.cuh) — cudaEvents, not std::chrono. Why?
//   - repeat N times and report the MIN, not the mean. Argue why min is the
//     right statistic for "how fast is this kernel".

// TODO 4: bench one size.
//   - allocate A, B, C on device, fill A/B with fill_random, copy over
//   - run cuBLAS once into a separate buffer and use it as the reference
//     (a CPU triple loop at 4096^3 is ~137 GFLOP — how long would that take?)
//   - for each variant: memset C to 0, time it, check C against the cuBLAS
//     result with check_close_device
//   - tolerance here should be LOOSER than correctness/ (1e-4). Why would two
//     correct fp32 GEMMs disagree, and why does it get worse as K grows?
//     (Hint: floating-point addition isn't associative.)
//   - TFLOPS = gemm_flops / seconds / 1e12
//   - print the table row by row; free everything

// TODO 5: main — create the handle, parse sizes from argv (default
//   {512, 1024, 4096}), bench each, destroy the handle.
//   Memory check: 3 x 4096^2 x 4 bytes = ? Does it fit your 4 GB?
int main(int argc, char **argv) {
    return 0;
}
