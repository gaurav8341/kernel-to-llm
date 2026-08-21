# Code Structure

How this repo is organized as projects get built out, project-by-project per `Objective.md` / `README.md`.

## Layout

```
kernel-to-llm/
├── common/                      # shared infra only — never the learning content itself
│   ├── include/                 # timing macros, correctness-check helpers
│   └── ptx_sass/                # dump_ptx.sh, dump_sass.sh (used in P01, P04, P06)
├── projects/
│   ├── p00-benchmark/
│   ├── p01-sgemm/
│   ├── p02-reduction/
│   ├── p03-convolution/
│   ├── p04-wmma-gemm/
│   ├── p05-fused-softmax-layernorm/   # links p02's reduction kernel
│   ├── p06-flashattention/            # linkable lib target — reused downstream
│   ├── p07-int4-quant/                # linkable lib target — reused downstream
│   ├── p08-gpt2-scratch/              # links p06
│   ├── p09-inference-runtime/         # links p07
│   ├── p10-streams-pipeline/
│   └── p11-capstone/                  # links p06, p07, p09, p10
├── docs/                         # hardware spec sheet (P00), profiling reports, diagrams
└── README.md
```

Each `projects/pXX/` is self-contained: `src/`, `bench/`, its own `README.md` with the benchmark table, and its own build target.

`common/` stays deliberately thin — only generic infra (a `cudaEvent_t` timing wrapper, a numpy-diff correctness checker, PTX/SASS dump scripts) belongs there, never a kernel. The kernels are the point of the roadmap; infra plumbing shouldn't blur into that.

## Inside a project

Generic skeleton — not every project uses every folder (see variations below):

```
projects/p01-sgemm/
├── src/                  # the kernels — naive.cu, tiled.cu, etc.
├── bench/                # benchmark harness: your kernel vs baseline (cuBLAS/PyTorch/naive)
├── correctness/          # test vs numpy/PyTorch reference, tolerance checks
├── reports/              # ncu/nsys output, screenshots, PTX/SASS dumps, plots
├── README.md             # deliverable checklist + benchmark table (mirrors the Linear milestone)
└── CMakeLists.txt        # (or Makefile, per the open build-system decision)
```

**What varies by project:**

| Folder | Present in | Notes |
|---|---|---|
| `python/` (torch bindings) | P05, P08, P09, P11 | wraps the CUDA kernel as a `torch.utils.cpp_extension` custom op |
| `reports/ptx/`, `reports/sass/` | P01, P04, P06 | only the projects with an optional "Going Lower" PTX/SASS section |
| `assets/` (input data) | P03 (test images), P08/P11 (dataset — gitignored, not committed) | keep large/binary inputs out of git; note a download script instead |
| `checkpoints/` | P08, P11 | training checkpoints — gitignored |

**Gitignore candidates:** `*.o`, `*.ptx`, `*.cubin`, compiled binaries, `checkpoints/`, `assets/*.bin` or raw datasets, `reports/*.nsys-rep` / `*.ncu-rep` (keep the exported screenshots/tables, not the raw profiler captures, unless you want them versioned).

Every project's `README.md` should end up structurally identical to its Linear milestone description (objective, week breakdown, checklist) but with the benchmark numbers and links to `reports/` filled in once actually done — the milestone is the plan, the README is the record.

## Open decisions

Two things to settle before scaffolding for real — recorded here so the reasoning doesn't get lost, not because either is decided:

1. **Build system.** CMake with one top-level `CMakeLists.txt` and per-project subdirectories (standard for CUDA, integrates cleanly with `torch.utils.cpp_extension` once P05+ need Python bindings) vs. plain per-project Makefiles (simpler, more visible, but the P05→P02 / P08→P06 / P09→P07 / P11→(P06,P07,P09,P10) linking has to be hand-wired). Leaning CMake given how much cross-project linking starts at P05.

2. **How "reuse" actually happens.** When the roadmap says "swap in your FlashAttention kernel from P06," two options:
   - **True reuse** — P08/P11 `#include`/link P06's actual compiled kernel. Less repetition, but couples the projects together.
   - **Deliberate re-derivation** — copy/rewrite the kernel into the downstream project. More repetition, but each `projects/pXX/` stays a fully standalone artifact (useful if pointing someone at just P06 or just P11, e.g. for a resume).

   This changes how tightly `common/` and `projects/pXX/` should be coupled, so it's worth deciding once, up front, rather than per-project.

## Reuse map

Which projects consume which, per the roadmap:

| Project | Reuses |
|---|---|
| P04 | benchmarked against P01 (not linked, just compared) |
| P05 | P02 (warp reduction) |
| P08 | — (from-scratch architecture; P06 not yet swapped in) |
| P09 | P07 (INT4 weights), P08 (trained model) |
| P11 | P06 (FlashAttention), P07 (INT4), P09 (inference runtime), P10 (pipeline) |
