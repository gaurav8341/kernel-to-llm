# P00 Benchmark

*Week 1 · Know your ceiling before writing a single kernel — Difficulty: Starter*

**Project 00: Benchmark Your RTX 3050 — Peak TFLOPS & Memory Bandwidth**

Before writing any kernel, measure your hardware's physical ceiling. You'll reference these numbers in every future project to know if you're hitting the roof.

### Week 1 — Setup + measure hardware limits

- [x] Install CUDA toolkit, NVCC, Nsight Systems + Compute
- [x] Write a bandwidth test: time cudaMemcpy of 1GB host→device, compute GB/s → `src/bandwidth_test.cu`
- [x] Write a compute test: max-throughput fp32 kernel, compute TFLOPS achieved → `src/compute_test.cu`
- [x] Record actual bandwidth: theoretical vs achieved → see Results below and `docs/hardware-spec-sheet.md`
- [x] Record actual TFLOPS: theoretical vs achieved → see Results below
- [x] Write a one-page "Hardware Spec Sheet" → [`docs/hardware-spec-sheet.md`](../../docs/hardware-spec-sheet.md)

**P00 complete.**

### Results

| Metric | Theoretical | Achieved | % of peak |
|---|---|---|---|
| VRAM bandwidth | 176.03 GB/s | — (not exercised by a memcpy-only test) | — |
| PCIe bandwidth | 15.75 GB/s (Gen4 x8, not the slot's x16 max) | H2D 12.67 GB/s · D2H 13.22 GB/s (pinned, sync) | ~80-84% |
| FP32 TFLOPS | 4.33 TFLOPS | 3.99-4.12 TFLOPS across runs | 92-95% |

The PCIe figures above are *after* switching the IOMMU to passthrough. Before that, the same test measured H2D 7.22 GB/s · D2H 7.12 GB/s — about 45% of the ceiling.

The full writeup — raw tool output, and the investigation into *why* achieved bandwidth started at only ~45% of the PCIe ceiling — is in [`docs/hardware-spec-sheet.md`](../../docs/hardware-spec-sheet.md). Short version: GPU throttling, chipset uplink contention, ASPM idle-cycling and link errors were each ruled out with live evidence, leaving IOMMU translation overhead over a physically fragmented pinned buffer. Booting with `iommu=pt` confirmed it, at the cost of system-wide DMA isolation. The fragmentation itself is still unaddressed — that's where the remaining ~16-20% is likely hiding.

### Tooling

`src/` holds the two deliverables above. `tools/` holds Claude-generated diagnostic helpers (GPU/PCIe introspection, pinned-memory physical-layout check) used to investigate the bandwidth results — see [`tools/README.md`](tools/README.md).

**You'll need to figure out:** NVCC compilation flags, cudaEvent_t timing, theoretical peak = SMs × clock × ops/cycle, PCIe bandwidth vs VRAM bandwidth, TGP throttling on laptop GPUs

**Resources:**
- [CUDA Install Guide (Linux/WSL2)](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [Nsight Systems](https://developer.nvidia.com/nsight-systems)
- [Nsight Compute](https://developer.nvidia.com/nsight-compute)
- [RTX 3050 Mobile Specs (TechPowerUp)](https://www.techpowerup.com/gpu-specs/geforce-rtx-3050-mobile.c3788)
