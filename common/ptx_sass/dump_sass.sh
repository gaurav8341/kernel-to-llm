#!/usr/bin/env bash
# Dump SASS (real machine ISA, one step below PTX) for a single .cu.
#
#   dump_sass.sh <kernel.cu> [out_dir] [extra nvcc flags...]
#
# Write dump_ptx.sh first — this one is the same shape with two extra steps.
#
# First, understand the difference (write the answer here in your own words):
#   - What is PTX? Who turns it into SASS, and when (compile time vs load time)?
#   - Why can PTX look fine while the kernel is still slow?
#
# YOUR TASK — steps:
#   1. Same arg parsing / repo-root logic as dump_ptx.sh (default out_dir
#      reports/sass).
#   2. Compile to a .cubin with nvcc (which flag? see `nvcc --help | grep cubin`).
#   3. Disassemble the cubin with cuobjdump into a .sass file
#      (see `cuobjdump --help`).
#   4. Print opcode counts. SASS mnemonics to count:
#        LDG / STG     global load/store        LDS / STS   shared load/store
#        FFMA          fp32 fused multiply-add  BAR.SYNC    barrier
#        LDL / STL     local (spill) traffic
#      Match each one to its PTX counterpart in dump_ptx.sh.
#   5. Print registers-per-thread and spill bytes for each kernel. ptxas will
#      tell you if you ask it verbosely — find the -Xptxas flag that does this.
#      Why does register count matter for occupancy? (20 SMs, 64K regs/SM.)
#
# Reference: https://docs.nvidia.com/cuda/cuda-binary-utilities/
