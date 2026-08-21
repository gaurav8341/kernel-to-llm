# P00 Benchmark

*Week 1 · Know your ceiling before writing a single kernel — Difficulty: Starter*

**Project 00: Benchmark Your RTX 3050 — Peak TFLOPS & Memory Bandwidth**

Before writing any kernel, measure your hardware's physical ceiling. You'll reference these numbers in every future project to know if you're hitting the roof.

### Week 1 — Setup + measure hardware limits

- Install CUDA toolkit, NVCC, Nsight Systems + Compute on WSL2/Ubuntu
- Write a bandwidth test: time cudaMemcpy of 1GB host→device, compute GB/s
- Write a compute test: max-throughput fp32 kernel, compute TFLOPS achieved
- Record your 3050's actual numbers: theoretical vs achieved bandwidth and TFLOPS
- Write a one-page "Hardware Spec Sheet" — you'll reference this all roadmap

**You'll need to figure out:** NVCC compilation flags, cudaEvent_t timing, theoretical peak = SMs × clock × ops/cycle, PCIe bandwidth vs VRAM bandwidth, TGP throttling on laptop GPUs

**Resources:**
- [CUDA Install Guide (Linux/WSL2)](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [Nsight Systems](https://developer.nvidia.com/nsight-systems)
- [Nsight Compute](https://developer.nvidia.com/nsight-compute)
- [RTX 3050 Mobile Specs (TechPowerUp)](https://www.techpowerup.com/gpu-specs/geforce-rtx-3050-mobile.c3788)
