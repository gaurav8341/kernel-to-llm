#pragma once

// Generic correctness checking: compare a device buffer against a host reference
// within a tolerance. Kernel-agnostic on purpose — see common/include/README.md.
//
// Used by P01's bench/ and correctness/, and meant to be reused from P02 on, so
// nothing in here knows what a GEMM is.
//
// ---------------------------------------------------------------------------
// The four questions, answered.
//
// 1. Why not `==`?
//    fp32 carries ~24 bits of mantissa and every intermediate result is
//    rounded. A GEMM sums K products, and the GPU sums them in a different
//    ORDER than a CPU triple loop does (different thread and register
//    schedule). Floating-point addition is not associative — (a+b)+c differs
//    from a+(b+c) — so two perfectly correct implementations disagree in the
//    low bits. `==` asks "did you round the same way I did", not "is this the
//    right answer".
//
// 2. Absolute error vs relative error — each breaks at an opposite extreme.
//    ABSOLUTE |ref - got| breaks for LARGE values: near |ref| ~ 1e6 the gap
//    between adjacent fp32 numbers is already ~0.06, so a correct result is
//    routinely off by far more than a 1e-4 absolute tolerance. Everything
//    fails.
//    RELATIVE |ref - got| / |ref| breaks when ref ~ 0: the denominator
//    vanishes, so the ratio explodes (or divides by zero) for a difference
//    that is numerically meaningless.
//
// 3. So: relative error is the primary test, with a FLOOR under the
//    denominator. An element is only counted bad when it exceeds BOTH the
//    relative tolerance AND the absolute floor — the floor keeps near-zero
//    outputs from failing on noise, while the relative test still catches a
//    genuinely wrong large value.
//
// 4. NaN and Inf: `nan > tol` is FALSE. In IEEE-754 every comparison involving
//    NaN is false except `!=`, so a tolerance test on its own silently PASSES
//    a kernel that wrote nothing but NaN. Non-finite values are rejected
//    explicitly, before any tolerance arithmetic runs. A non-finite value in
//    the REFERENCE is a failure too: it means the oracle is broken, and a
//    check against a broken oracle is worse than no check.
// ---------------------------------------------------------------------------

#include "utils.cuh"

#include <cmath>
#include <vector>

// What you want to see when a check fails — not just pass/fail.
//
// `bad` is the field that pays for itself: 3 bad elements out of 262144 says
// "missing boundary guard", while 262144/262144 says "wrong index formula".
// Those are completely different bugs, distinguished at a glance. `avg_error`
// on its own cannot tell them apart, because one catastrophically wrong
// element barely moves the mean.
//
// Errors are kept in double: the comparison arithmetic runs in double anyway
// (so that measuring the error does not itself round), and storing the result
// in float would throw that away again.
struct Results {
    bool success = false;

    double max_absolute_error = 0.0;
    double max_relative_error = 0.0;   // relative to |ref|, floored
    double avg_error = 0.0;            // mean |ref - got| over all n elements

    size_t worst_index = 0;            // where max_relative_error was found
    size_t bad = 0;                    // elements outside the tolerance
    size_t n = 0;                      // elements compared
};

// Compare two HOST buffers of n floats.
//   rel_tol    relative tolerance, the primary test
//   abs_floor  floor on the denominator, and the absolute escape hatch
//
// The defaults are a starting point for short reductions, NOT a universal
// truth. abs_floor in particular has to grow with however many fp32 additions
// the kernel chains together: 1e-5 is right for a handful and too tight for a
// GEMM at K=512, where it false-alarms on a correct kernel. See
// abs_floor_for_k() in p01-sgemm/correctness/check_sgemm.cu for the reasoning.
inline Results check_close_host(const float *ref, const float *got, size_t n,
                                double rel_tol = 1e-4, double abs_floor = 1e-5) {
    Results r;
    r.n = n;
    r.success = true;

    double abs_err_sum = 0.0;

    for (size_t i = 0; i < n; ++i) {
        const double a = (double)ref[i];
        const double b = (double)got[i];

        // Non-finite first: no tolerance test below would catch it (see 4).
        if (!std::isfinite(a) || !std::isfinite(b)) {
            ++r.bad;
            r.success = false;
            r.max_absolute_error = INFINITY;
            r.max_relative_error = INFINITY;
            r.worst_index = i;
            continue;
        }

        const double abs_err = fabs(a - b);
        const double rel_err = abs_err / fmax(fabs(a), abs_floor);

        abs_err_sum += abs_err;
        if (abs_err > r.max_absolute_error) r.max_absolute_error = abs_err;
        if (rel_err > r.max_relative_error) {
            r.max_relative_error = rel_err;
            r.worst_index = i;
        }

        // Both must be exceeded — see 3.
        if (rel_err > rel_tol && abs_err > abs_floor) {
            ++r.bad;
            r.success = false;
        }
    }

    r.avg_error = (n > 0) ? abs_err_sum / (double)n : 0.0;
    return r;
}

// Same comparison, but `got` lives on the DEVICE.
inline Results check_close_device(const float *h_ref, const float *d_got, size_t n,
                                  double rel_tol = 1e-4, double abs_floor = 1e-5) {
    std::vector<float> h_got(n);
    CUDA_CHECK(cudaMemcpy(h_got.data(), d_got, n * sizeof(float), cudaMemcpyDeviceToHost));
    return check_close_host(h_ref, h_got.data(), n, rel_tol, abs_floor);
}

// One line per check. Returns success, so main() can turn a run into an exit
// code — `make run-check` has to fail loudly in a script, not just print FAIL.
//
// Correctness ONLY — no timing, no throughput. Performance reporting lives in
// GpuTimer (utils.cuh), which owns the clock and knows how to turn elapsed time
// into GB/s or TFLOPS. Keeping them apart matters for a reason beyond tidiness:
// a figure of merit is not universal (a GEMM is compute bound and wants TFLOPS,
// a reduction or transpose is bandwidth bound and wants GB/s), so a correctness
// header that printed one would have to know what kind of kernel it is judging.
// This one must not know.
inline bool report_check(const char *label, const Results &r, double rel_tol = 1e-4) {
    fprintf(stdout, "%-12s %s  max_rel=%.3e (tol %.1e)  max_abs=%.3e  avg=%.3e  bad=%zu/%zu",
            label, r.success ? "PASS" : "FAIL",
            r.max_relative_error, rel_tol, r.max_absolute_error, r.avg_error,
            r.bad, r.n);
    if (!r.success) fprintf(stdout, "  worst_index=%zu", r.worst_index);
    fprintf(stdout, "\n");
    return r.success;
}
