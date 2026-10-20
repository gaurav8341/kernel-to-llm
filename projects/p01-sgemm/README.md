# P01 SGEMM

*Weeks 2–3 · Global mem → shared mem → beat cuBLAS — Difficulty: Starter*

**Project 01: Tiled SGEMM — Reach 60–70% of Your 3050's Peak TFLOPS**

Write a raw CUDA matrix multiply without cuBLAS. Naive version first — it will be 10× slower than it should be. Profile, understand why, fix it.

### Week 2 — Naive kernel + profiling baseline

- [ ] Write naive GEMM: one thread = one output element, global memory only → `src/sgemm_naive.cu`
- [ ] Test correctness vs numpy matmul (fp32, tolerance 1e-4) → `make run-check`
- [ ] Profile with ncu — identify it as "memory bound", record L2 cache hit rate → `reports/`
- [ ] Compute arithmetic intensity of your naive kernel (FLOPS / bytes moved)

### Week 3 — Tiled kernel + benchmark table

- [ ] Implement tiled GEMM: load tiles into `__shared__` memory, compute locally → `src/sgemm_tiled.cu`
- [ ] Fix bank conflicts: pad shared memory arrays, verify with ncu bank conflict counter
- [ ] Benchmark table: naive vs tiled vs cuBLAS at sizes 512², 1024², 4096² → `make run-bench`
- [ ] README: annotated diagram of the tiling strategy + your TFLOPS numbers

### 🔽 Week 3+ — Read the PTX your kernel generates *(lower-level, optional)*

- [ ] Dump PTX for both naive and tiled kernels → `make ptx` (dumps to `reports/ptx/`)
- [ ] Find `ld.global` vs `ld.shared` — confirm shared memory loads appear in inner loop
- [ ] Look for `st.local` / `ld.local` — if present, you have a register spill, fix it
- [ ] Check `bar.sync` count — matches your `__syncthreads()` calls
- [ ] Add `const __restrict__` to kernel args, redump PTX — spot `ld.global.nc` appearing (L1 texture cache for read-only)
- [ ] Use Godbolt with CUDA NVCC + flag `--ptx` to browse PTX interactively

`make ptx` / `make sass` print an opcode histogram for exactly these instructions, so
the before/after of each change is one line rather than a manual grep.

### Results

Fill in from `make run-bench` once the kernels are written. Target is 60–70% of peak;
for reference, cuBLAS itself reaches ~71% of this machine's measured peak at 4096².

| Size | Kernel | Time (ms) | TFLOPS | % of peak | % of cuBLAS |
|---|---|---|---|---|---|
| 512² | naive | | | | |
| 512² | tiled | | | | |
| 512² | cuBLAS | 0.123 | 2.19 | 53% | 100% |
| 1024² | naive | | | | |
| 1024² | tiled | | | | |
| 1024² | cuBLAS | 0.806 | 2.66 | 65% | 100% |
| 4096² | naive | | | | |
| 4096² | tiled | | | | |
| 4096² | cuBLAS | 46.748 | 2.94 | 71% | 100% |

Peak is 4.12 TFLOPS measured / 4.33 theoretical — see [`docs/hardware-spec-sheet.md`](../../docs/hardware-spec-sheet.md).
The cuBLAS row is a real measurement from this machine (best of 10 after 3 warmup) and is
the baseline the other two rows have to be read against. Note cuBLAS's own climb from 53%
to 71% as the matrices grow: at 512² there is not enough work to saturate 20 SMs, so a
naive kernel looking "close to cuBLAS" at that size means less than it appears.

### Layout

| Path | What |
|---|---|
| `src/sgemm.cuh` | the row-major convention, the shared `SgemmLauncher` signature, `TILE_DIM`, host helpers |
| `src/sgemm_naive.cu` | Week 2 kernel — **your** code |
| `src/sgemm_tiled.cu` | Week 3 kernel — **your** code |
| `bench/sgemm_bench.cu` | timing harness + cuBLAS baseline, emits the markdown table above |
| `correctness/check_sgemm.cu` | rel-tol 1e-4 check against a CPU or numpy reference |
| `correctness/reference.py` | numpy matmul reference → raw float32 dumps for `--data` |
| `reports/ptx/`, `reports/sass/` | `make ptx` / `make sass` output |

Both kernels compile into objects that the benchmark and the checker link against, so there
is one copy of each kernel and the same code is what gets timed and what gets verified.

### Commands

```sh
make                                  # build bench + checker
make run-check                        # CPU reference, 512³
make run-check ARGS="1024 1024 1024"  # bigger CPU reference (slower)
make reference SHAPE="1024 1024 1024" # numpy reference → correctness/data/
make run-check ARGS="--data correctness/data"
make run-bench                        # 512², 1024², 4096²
make run-bench SIZES="2048"           # one size
make ptx                              # PTX dump + opcode histogram
make sass                             # SASS dump + register/spill report
```

`reference.py` needs numpy, which is not currently installed in this machine's default
python — `pip install numpy` (P08+ will want it regardless). Until then `make run-check`
uses the built-in CPU reference, which needs nothing.

Correctness is checked at two tolerances on purpose: `correctness/` uses 1e-4 against a
double-accumulated reference (the roadmap's requirement), while `bench/` uses 1e-3 against
cuBLAS, because cuBLAS reorders the K accumulation and fp32 results legitimately diverge in
the last bits as K grows. The bench check is a guard against timing a kernel that computes
garbage, not the real correctness test.

**You'll need to figure out:** thread/block/grid hierarchy, `__shared__` memory lifetime, `__syncthreads()`, memory coalescing, shared memory bank conflicts, occupancy vs register count, roofline model, PTX `ld.global` vs `ld.shared`, register spill → `st.local`

**Resources:**
- [CUDA Guide — Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- [Simon Boehm — How to Optimize CUDA Matmul](https://siboehm.com/articles/22/CUDA-MMM)
- [NVIDIA Blog — Using Shared Memory](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
- [CUDA by Example — Ch3–5](https://developer.nvidia.com/cuda-example)
- [PTX ISA Reference (use as dictionary)](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- [Godbolt — Compiler Explorer (CUDA NVCC + --ptx)](https://godbolt.org/)
