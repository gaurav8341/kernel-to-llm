# kernel-to-llm

CUDA × LLM Mastery — a project-first, week-by-week roadmap for learning GPU kernel programming from the ground up, on a single RTX 3050 4GB laptop GPU, ending in a self-trained, self-quantized, self-served LLM.

> Project-first. Week-by-week. Every deliverable is concrete.

**Hardware:** RTX 3050 4GB · GA106 Ampere · 80 Tensor Cores · 128-bit bus · 8GB System RAM

**12** projects · **24** weeks ideal / **32–36** weeks real · difficulty ramps Starter → Expert

## Files

- [`Objective.md`](Objective.md) — the roadmap source (Markdown)
- [`cuda-llm-master-roadmap.html`](cuda-llm-master-roadmap.html) — styled, readable version of the same roadmap
- [`cuda-llm-master-roadmap.pdf`](cuda-llm-master-roadmap.pdf) / [`CUDA × LLM — Master Roadmap.pdf`](<CUDA × LLM — Master Roadmap.pdf>) — printable exports

## The track

| # | Project | Weeks | Difficulty |
|---|---------|-------|------------|
| 00 | Speed of Light Benchmark — peak TFLOPS & bandwidth | 1 | Starter |
| 01 | The Memory Architect — tiled SGEMM, beat cuBLAS | 2–3 | Starter |
| 02 | The Warp Tactician — parallel reduction, warp shuffle | 4 | Starter → Mid |
| 03 | Warp Divergence in the Wild — Gaussian blur + Sobel | 5 | Starter → Mid |
| 04 | The Tensor Core Specialist — FP16 GEMM with WMMA | 6–7 | Mid |
| 05 | The Kernel Fusionist — fused Softmax + LayerNorm | 8–9 | Mid |
| 06 | The Most Important Kernel — FlashAttention | 10–11 | Hard |
| 07 | The Quantization Engineer — INT4 bitpacking | 12 | Mid |
| 08 | Train a Real Language Model — GPT-2 124M from scratch | 13–15 | Mid |
| 09 | Make Inference Fast — KV cache + INT4 + tok/s | 16–17 | Hard |
| 10 | The System Architect — CUDA streams & double buffering | 18 | Mid |
| 11 | Capstone — 50M LLM, train/quantize/deploy/report | 19–24 | Expert |

Each project builds on the last: kernels from earlier projects (tiled GEMM, warp reduction, WMMA, INT4 quant, FlashAttention) get reused directly in the GPT-2 training run (P08) and the final capstone (P11).

**Going lower:** three optional deep-dive tracks run alongside the main projects — reading PTX/SASS assembly (P01, P04), the CUDA Driver API below the Runtime, and GA106 microarchitecture / HDL. See the "Going Lower" section in the roadmap for entry points.

## How to use this

1. Start at P00, work top to bottom — each week has concrete deliverables, not just reading.
2. "You'll need to figure out" lists are concepts to look up *when you hit them*, not a reading list to front-load.
3. If you're tracking against a job search timeline, P01–P04 alone produce resume-ready CUDA artifacts (a cuBLAS-beating GEMM, a working Tensor Core kernel) — front-load those four.
4. Full detail, resources, and lower-level (PTX/SASS) call-outs per project live in [`Objective.md`](Objective.md) / the HTML roadmap.
