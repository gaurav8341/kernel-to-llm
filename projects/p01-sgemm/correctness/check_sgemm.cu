// P01 — correctness: naive and tiled SGEMM vs a reference, fp32, rel tol 1e-4.
//
// Usage:
//   ./bin/check_sgemm                         # CPU reference, 512^3
//   ./bin/check_sgemm 512 256 128             # CPU reference, M N K
//   ./bin/check_sgemm --data correctness/data # numpy reference (reference.py)
//
// Exit code: 0 = every variant passed, 1 = a variant failed, 2 = bad usage.
//
// The `ms` column is a COLD SINGLE RUN and is not a benchmark: no warmup, so it
// still carries first-launch costs (context setup, module load, JIT), and n=1,
// so it carries whatever the OS and the driver were doing at that instant. It is
// here to answer "did my change make this 10x slower" at a glance. Real numbers
// come from bench/, which warms up and reports the best of N.
//
// The TFLOPS line inherits every one of those caveats, and adds one more: at
// the small ragged shapes this file is most useful at (7x3x5), the kernel is
// pure launch overhead and TFLOPS is meaningless. Read it at 512^3 and up.
#include "sgemm.cuh"
#include "check.cuh"

#include <cfloat>
#include <cstring>
#include <vector>

static const float  kAlpha  = 1.0f;
static const float  kBeta   = 0.0f;
static const double kRelTol = 1e-4;

// The absolute floor cannot be a constant, because the error it has to absorb
// grows with K.
//
// Each output element is a sum of K products. The kernel accumulates in fp32,
// so every one of those K additions rounds, and the partial sums themselves
// grow like sqrt(K) for inputs in [-1, 1). The errors random-walk rather than
// cancel, which puts the expected absolute error at roughly
// FLT_EPSILON * K * |typical element| — about 2e-5 at K=512, measured at 3.8e-5.
//
// A fixed floor of 1e-5 therefore FAILS a correct kernel: an output element
// that happens to land near zero (|ref| ~ 1e-3 out of a typical 7.5) sees a
// relative error of several percent from an absolute difference that is pure
// rounding noise. That is a false alarm, and a checker that cries wolf on
// correct code is worse than useless — it trains you to ignore it.
//
// Scaling with K keeps the floor four to five orders of magnitude below what a
// REAL bug produces: a wrong index formula gives O(1) errors (measured 34.7),
// and a missing bounds guard leaves elements at zero, i.e. an error of |ref|
// itself. Neither hides under this floor.
static double abs_floor_for_k(int K) {
    return 4.0 * (double)FLT_EPSILON * (double)K;
}

// ---------------------------------------------------------------------------
// The oracle: C = alpha * (A * B) + beta * C, row-major, on the host.
//
// Accumulates in double while the kernels accumulate in float, so the
// reference is strictly MORE accurate than the thing it judges. If the oracle
// had the same rounding error as the kernel, a disagreement would not tell you
// which one was wrong — and worse, a kernel could match the oracle's error and
// look correct while both drifted. The whole value of a reference is that its
// error is negligible compared to the tolerance.
//
// Cost is O(M*N*K) scalar work: 512^3 takes about a second, 1024^3 about ten.
// For 4096^3 use --data (numpy), or lean on bench/, which checks against
// cuBLAS instead.
// ---------------------------------------------------------------------------
static void sgemm_cpu(int M, int N, int K, float alpha,
                      const float *A, const float *B,
                      float beta, float *C) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                acc += (double)A[m * K + k] * (double)B[k * N + n];
            }
            C[m * N + n] = alpha * (float)acc + beta * C[m * N + n];
        }
    }
}

// ---------------------------------------------------------------------------
// The --data path: raw float32 dumps written by correctness/reference.py.
// ---------------------------------------------------------------------------

// Reads exactly n floats, and treats a short read as an error rather than
// leaving the tail of the buffer as uninitialized garbage that would then be
// silently "compared".
static bool read_f32(const char *dir, const char *name, float *dst, size_t n) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", dir, name);

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "check: cannot open %s\n", path);
        return false;
    }
    const size_t got = fread(dst, sizeof(float), n, f);
    // A file LONGER than expected means the shape and the buffers disagree,
    // which is just as broken as a short read.
    const bool extra = (fgetc(f) != EOF);
    fclose(f);

    if (got != n || extra) {
        fprintf(stderr, "check: %s holds %s%zu floats, expected %zu\n",
                path, extra ? ">" : "", got, n);
        return false;
    }
    return true;
}

static bool read_shape(const char *dir, int *M, int *N, int *K) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/shape.txt", dir);

    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "check: cannot open %s — run `make reference` first\n", path);
        return false;
    }
    const bool ok = (fscanf(f, "%d %d %d", M, N, K) == 3);
    fclose(f);

    if (!ok || *M <= 0 || *N <= 0 || *K <= 0) {
        fprintf(stderr, "check: %s is not a valid \"M N K\"\n", path);
        return false;
    }
    return true;
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s [M N K]            CPU reference (default 512 512 512)\n"
            "       %s --data <dir>       numpy reference from reference.py\n",
            argv0, argv0);
}

int main(int argc, char **argv) {
    int M = 512, N = 512, K = 512;
    const char *data_dir = nullptr;

    // argc counts the program name, so "M N K" is argc == 4, not 3.
    if (argc == 1) {
        // defaults
    } else if (argc == 3 && strcmp(argv[1], "--data") == 0) {
        data_dir = argv[2];
    } else if (argc == 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
        if (M <= 0 || N <= 0 || K <= 0) {
            fprintf(stderr, "check: M, N, K must all be positive\n");
            return 2;
        }
    } else {
        usage(argv[0]);
        return 2;
    }

    if (data_dir && !read_shape(data_dir, &M, &N, &K)) return 2;

    const size_t nA = (size_t)M * K, nB = (size_t)K * N, nC = (size_t)M * N;

    const double kAbsFloor = abs_floor_for_k(K);

    fprintf(stdout, "SGEMM correctness  M=%d N=%d K=%d  alpha=%.1f beta=%.1f  "
                    "reference=%s  rel_tol=%.1e  abs_floor=%.1e\n",
            M, N, K, kAlpha, kBeta, data_dir ? data_dir : "cpu", kRelTol, kAbsFloor);

    std::vector<float> hA(nA), hB(nB), hRef(nC, 0.0f);

    if (data_dir) {
        if (!read_f32(data_dir, "A.bin",     hA.data(),   nA)) return 2;
        if (!read_f32(data_dir, "B.bin",     hB.data(),   nB)) return 2;
        if (!read_f32(data_dir, "C_ref.bin", hRef.data(), nC)) return 2;
    } else {
        // Different seeds, or A and B would be identical and a kernel that
        // confused its two operands would still pass. Same seeds as bench/,
        // so a timing number always refers to verified inputs.
        initialize_matrix(hA.data(), nA, /*seed=*/1234);
        initialize_matrix(hB.data(), nB, /*seed=*/5678);
        fprintf(stdout, "computing CPU reference (%.1f MFLOP)...\n",
                gemm_flops(M, N, K) / 1e6);
        sgemm_cpu(M, N, K, kAlpha, hA.data(), hB.data(), kBeta, hRef.data());
    }

    bool all_passed = true;

    // Check the checker before trusting it to judge anything: the reference
    // against itself must be a perfect PASS. If this line ever fails, the bug
    // is in check.cuh or in the loaded data, not in a kernel.
    {
        Results r = check_close_host(hRef.data(), hRef.data(), nC, kRelTol, kAbsFloor);
        all_passed &= report_check("oracle", r, kRelTol);
    }

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, nA * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, nB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, nC * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), nA * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), nB * sizeof(float), cudaMemcpyHostToDevice));

    struct Variant { const char *name; SgemmLauncher fn; };
    const Variant variants[] = {
        {"naive", sgemm_naive},
        {"tiled", sgemm_tiled},
    };

    for (const Variant &v : variants) {
        // Zero C before every variant, for two reasons that both matter:
        //   1. beta = 0 does not make C's prior contents irrelevant. The
        //      epilogue computes beta * C, and 0.0f * NaN is NaN, not 0 — so
        //      garbage in C can propagate into a correct kernel's output.
        //   2. Without it, C still holds the PREVIOUS variant's result. A
        //      kernel with a missing bounds guard leaves some elements
        //      unwritten, and those would still match the reference from the
        //      last run — hiding exactly the bug the ragged shapes are for.
        CUDA_CHECK(cudaMemset(dC, 0, nC * sizeof(float)));

        // cudaEvents, not std::chrono: a kernel launch is asynchronous, so the
        // host clock would stop as soon as the launch was *queued* and report a
        // few microseconds no matter how long the kernel actually ran. Events are
        // recorded in the stream, so they bracket the kernel on the device.
        GpuTimer timer(v.name);
        timer.start();
        v.fn(M, N, K, kAlpha, dA, dB, kBeta, dC);
        timer.stop();

        // The launcher deliberately does not synchronize (see sgemm.cuh), so the
        // caller does it. elapsed() waits on stopEvent, which was recorded after
        // the kernel, so this is also the sync we owe before reading the result
        // back — and it is where a fault inside the kernel surfaces.
        //
        // elapsed() rather than log() here: the timing lines are printed below,
        // after the PASS/FAIL verdict, so that correctness reads first.
        timer.elapsed();

        Results r = check_close_device(hRef.data(), dC, nC, kRelTol, kAbsFloor);
        all_passed &= report_check(v.name, r, kRelTol);

        // Correctness first, cost second — and the rate ONLY if the answer was
        // right. The speed of a kernel that computes the wrong result is not a
        // measurement of anything: the empty stub in src/ times at 0.002 ms and
        // "achieves" 131 TFLOPS on a card whose theoretical peak is 4.33. That
        // is how "my kernel beats cuBLAS 30x" gets reported.
        //
        // The elapsed time is still worth printing on a failure — it tells you
        // whether the kernel ran at all — so only get_tflops() is gated.
        timer.log();
        if (r.success) timer.get_tflops(gemm_flops(M, N, K));
    }

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    // Shapes worth running once both kernels pass 512^3:
    //   ./bin/check_sgemm 512 256 128   non-square — catches swapped row/col
    //   ./bin/check_sgemm 500 500 500   not a tile multiple — catches guards
    //   ./bin/check_sgemm 1 1 1         degenerate — catches tile-loop bounds
    //   ./bin/check_sgemm 7 3 5         tiny and ragged — same, harder
    return all_passed ? 0 : 1;
}
