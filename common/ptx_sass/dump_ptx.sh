#!/usr/bin/env bash
# Dump PTX for a single .cu translation unit.
#
#   dump_ptx.sh <kernel.cu> [out_dir] [extra nvcc flags...]
#
# YOUR TASK — write this script. Steps:
#
#   1. Start with `set -euo pipefail` (look up what each of e / u / o pipefail
#      protects you from).
#   2. Parse args: $1 is the .cu file; an optional out_dir (default reports/ptx);
#      anything left over gets forwarded to nvcc (so `-DTILE_DIM=16` works).
#      How will you tell an out_dir apart from an nvcc flag?
#   3. Find the repo root relative to this script (not the caller's cwd) so you
#      can pass -I<repo>/common/include. Hint: ${BASH_SOURCE[0]} + dirname.
#   4. Run nvcc with --ptx, -O3, -arch=sm_86 (your 3050 is compute capability
#      8.6 — check `nvidia-smi --query-gpu=compute_cap --format=csv`).
#      Why -O3? Try dumping without it once and compare.
#   5. Print a count of each opcode below with grep -c. Careful: grep -c exits
#      non-zero when the count is 0 — what does that do under `set -e`?
#   6. If st.local / ld.local appear at all, print a loud warning.
#
# What to look for in the PTX (P01 "Going Lower"):
#   ld.global      global load     — should NOT be in a tiled kernel's inner loop
#   ld.shared      shared load     — SHOULD be in the inner loop
#   ld.global.nc   non-coherent    — appears once args are const __restrict__
#   fma.rn.f32     fused mul-add   — the useful work
#   st.local/ld.local              — register spill
#   bar.sync       barrier         — count should match your __syncthreads()
#
# Do it by hand once first (nvcc --ptx ... and open the file) before scripting it.
# Reference: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/
