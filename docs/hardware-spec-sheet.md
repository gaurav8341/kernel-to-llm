# Hardware Spec Sheet — RTX 3050 Laptop (GA107, sm_86)

*Produced by P00. Referenced by every later project to judge "% of peak" achieved.*

## Theoretical peak

| Metric | Value | How computed |
|---|---|---|
| FP32 TFLOPS (theoretical) | 4.33 TFLOPS | 16 SMs × 128 cores/SM × 1.057 GHz × 2 (FMA) = 4.329 TFLOPS |
| VRAM bandwidth (theoretical) | 176.03 GB/s | 5501 MHz × 2 (GDDR6 DDR factor) × 128-bit bus / 8 = 176.032 GB/s |
| PCIe bandwidth (theoretical) | 15.75 GB/s | Gen4 x8 (negotiated link, not the slot's Gen4 x16 max) × 1.969 GB/s/lane (16 GT/s, 128b/130b encoding) |

## Measured (this machine)

| Metric | Achieved | % of theoretical |
|---|---|---|
| H2D bandwidth (GB/s) | 7.22 (pinned, sync) | 45.8% of PCIe |
| D2H bandwidth (GB/s) | 7.12 (pinned, sync) | 45.2% of PCIe |
| FP32 compute (TFLOPS) | 3.99 | 92.16% of FP32 peak |

## Notes

- All theoretical numbers above come from `projects/p00-benchmark/tools/device_query.cu` (`make run-device-query`), run on this machine — not a generic spec sheet. Laptop RTX 3050 configs vary by OEM/TGP, and this laptop's PCIe width in particular is not what its slot's Gen4 x16 max would suggest, so the on-device reading is the only trustworthy reference point for "% of peak" here.
  - VRAM: `cudaDeviceGetAttribute(cudaDevAttrMemoryClockRate / cudaDevAttrGlobalMemoryBusWidth)` — memory clock 5501 MHz, bus 128-bit.
  - PCIe: NVML (`nvmlDeviceGetCurrPcieLinkGeneration/Width`) sampled *while `bandwidth_test` was actively running* — Gen4 x8. The idle reading briefly shows Gen1 (power-saving downclock), so idle NVML/`nvidia-smi` reads are not a reliable theoretical-peak source; sample under load.
- **PCIe link is x8, not x16**: `nvmlDeviceGetMaxPcieLinkWidth` reports Gen4 x16 as this GPU's max capability, but the negotiated link (both idle and under load) is consistently Gen4 x8 — this laptop wires the dGPU with half the lanes the die supports. That alone halves the PCIe ceiling used above (31.5 GB/s → 15.75 GB/s) and is worth remembering for any future PCIe-bound project on this machine.
- GB/s uses the decimal convention (1e9) throughout, matching the formula already used in `bandwidth_test.cu`.
- Measured H2D/D2H figures are a single run each (pinned, synchronous variant) from `bandwidth_test.cu`'s current output — the repeat/best-or-median TODO from that file is still open, so treat these as one sample, not a stable average. Unpinned transfers measured meaningfully slower (~5.5-6.1 GB/s H2D, ~2.6-2.9 GB/s D2H, both noisier run-to-run) and async variants tracked within noise of their sync counterparts on the default stream.
- Even against the corrected 15.75 GB/s PCIe ceiling, ~45% achieved is a real gap (not just an artifact of comparing against the wrong bus) — worth investigating later if a PCIe-bound project needs more headroom.
- **Likely cause of the remaining gap: IOMMU translation overhead on a physically fragmented pinned buffer.** GPU core throttling, chipset uplink contention, ASPM idle-cycling, and link errors were all ruled out with live evidence (see `nvidia-smi -q -d PERFORMANCE` during load, `lspci -t`, per-transfer gap timing from an `nsys` trace, and clean AER counters, respectively). This system's IOMMU (Intel VT-d) runs in `Translated` mode, not passthrough, and `projects/p00-benchmark/tools/pinned_memory_layout.cu` (`make run-pinned-layout`, needs `sudo` — unprivileged reads of `/proc/self/pagemap` return zeroed PFNs) shows the 1GB pinned buffer's physical pages are extremely fragmented: 36,932 separate contiguous runs averaging ~28KB (7 pages) each, with the single largest run only 1.28MB — nowhere close to huge-page-backed (`transparent_hugepage/enabled` is `madvise`, and nothing indicates `cudaMallocHost` requests it). With fragmentation this severe the IOMMU can't build large superpage mappings, so a linear DMA sweep crosses into a newly-mapped region roughly every 28KB — likely causing frequent IOTLB misses well beyond the hardware's cache capacity. This is strong corroborating evidence, not proof; the test that would confirm causation outright is booting with `iommu=pt` and re-measuring (not yet done — trades away IOMMU DMA isolation for this device, so left as a deliberate follow-up rather than something flipped casually).
- TGP/thermal throttling observed: TODO
- Any gap between theoretical and achieved worth flagging for later projects: PCIe link width (x8 vs x16) is the standout finding from P00 — see above.

- **IOMMU virtulization for memory addressing was bottleneck**:
So setup the iommu and set it to `pt` which means there will not be any virtualization of addressing. The devices can acces the MMU access without any iommu filter or virtualization. Setting it to `pt` mode will disable the virtulization and allow the devices to use the memory addresses directly without any virtualization or walkthrough bottleneck

This gave us a real perf boost: H2D sync went from 7.22 GB/s to 12.67 GB/s and D2H sync from 7.12 GB/s to 13.22 GB/s — roughly a 76-86% increase, pushing us from ~45% to ~80-84% of the 15.75 GB/s PCIe ceiling. But this is also a security issue as now any device can access the device memory directly without any virtulization which is unsafe memory management. 

The rest of the gap from ideal likely comes from memory fragmentation (see the pinned-layout measurement above — longest contiguous run was only 1.28 MB out of the 1GB buffer, pre-`iommu=pt`; not re-measured after the reboot).

## Raw output

`make run-device-query`:

```
=== VRAM (from CUDA runtime) ===
memory clock:      5501 MHz
bus width:         128-bit
theoretical BW:     176.03 GB/s

=== PCIe (from NVML) ===
max link:          Gen4 x16  -> 31.51 GB/s theoretical
current link:      Gen4 x8  -> 15.75 GB/s theoretical
(sample during an active cudaMemcpy to see the negotiated gen under load,
 not the idle/power-saving state)
```

`make run-bandwidth`:

```
H2D-SYNC : 148.732422 ms
Throughput GB/s: 7.219285
H2D-ASYNC : 148.424225 ms
Throughput GB/s: 7.234276
H2D-UNPINNED-SYNC : 178.896103 ms
Throughput GB/s: 6.002041
D2H-SYNC : 150.703842 ms
Throughput GB/s: 7.124846
D2H-ASYNC : 150.590820 ms
Throughput GB/s: 7.130195
D2H-UNPINNED-SYNC : 407.501526 ms
Throughput GB/s: 2.634939
```

`sudo make run-pinned-layout`:

```
page size:                 4096 bytes
total pages:                262144
pages with valid PFN:       262144
pages with unavailable PFN: 0
contiguous physical runs:  36932
longest run:               328 pages (1.28 MB)
average run length:        7.1 pages (0.0027% of buffer)
```

`make run-bandwidth`:

After setting `iommu=pt`

```
./bin/bandwidth_test
H2D-SYNC : 84.718430 ms
Throughput GB/s: 12.674241
H2D-ASYNC : 84.337852 ms
Throughput GB/s: 12.731434
H2D-UNPINNED-SYNC : 165.404770 ms
Throughput GB/s: 6.491601
D2H-SYNC : 81.233696 ms
Throughput GB/s: 13.217937
D2H-ASYNC : 81.223938 ms
Throughput GB/s: 13.219524
D2H-UNPINNED-SYNC : 475.717804 ms
Throughput GB/s: 2.257098
```

`make run-compute`:

```
Compute Bound FMA : 26.280895 ms
Achieved TFLOPS: 3.989879
Theoretical TFLOPS: 4.329472
% of peak reached: 92.16%
```