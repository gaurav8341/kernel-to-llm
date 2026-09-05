# Hardware Spec Sheet — RTX 3050 Laptop (GA106, sm_86)

*Produced by P00. Referenced by every later project to judge "% of peak" achieved.*

## Theoretical peak

| Metric | Value | How computed |
|---|---|---|
| FP32 TFLOPS (theoretical) | TODO | SMs × CUDA cores/SM × boost clock (GHz) × 2 (FMA) |
| VRAM bandwidth (theoretical) | 176.03 GB/s | 5501 MHz × 2 (GDDR6 DDR factor) × 128-bit bus / 8 = 176.032 GB/s |
| PCIe bandwidth (theoretical) | 15.75 GB/s | Gen4 x8 (negotiated link, not the slot's Gen4 x16 max) × 1.969 GB/s/lane (16 GT/s, 128b/130b encoding) |

## Measured (this machine)

| Metric | Achieved | % of theoretical |
|---|---|---|
| H2D bandwidth (GB/s) | 7.22 (pinned, sync) | 45.8% of PCIe |
| D2H bandwidth (GB/s) | 7.12 (pinned, sync) | 45.2% of PCIe |
| FP32 compute (TFLOPS) | TODO | TODO |

## Notes

- All theoretical numbers above come from `projects/p00-benchmark/src/device_query.cu` (`make run-device-query`), run on this machine — not a generic spec sheet. Laptop RTX 3050 configs vary by OEM/TGP, and this laptop's PCIe width in particular is not what its slot's Gen4 x16 max would suggest, so the on-device reading is the only trustworthy reference point for "% of peak" here.
  - VRAM: `cudaDeviceGetAttribute(cudaDevAttrMemoryClockRate / cudaDevAttrGlobalMemoryBusWidth)` — memory clock 5501 MHz, bus 128-bit.
  - PCIe: NVML (`nvmlDeviceGetCurrPcieLinkGeneration/Width`) sampled *while `bandwidth_test` was actively running* — Gen4 x8. The idle reading briefly shows Gen1 (power-saving downclock), so idle NVML/`nvidia-smi` reads are not a reliable theoretical-peak source; sample under load.
- **PCIe link is x8, not x16**: `nvmlDeviceGetMaxPcieLinkWidth` reports Gen4 x16 as this GPU's max capability, but the negotiated link (both idle and under load) is consistently Gen4 x8 — this laptop wires the dGPU with half the lanes the die supports. That alone halves the PCIe ceiling used above (31.5 GB/s → 15.75 GB/s) and is worth remembering for any future PCIe-bound project on this machine.
- GB/s uses the decimal convention (1e9) throughout, matching the formula already used in `bandwidth_test.cu`.
- Measured H2D/D2H figures are a single run each (pinned, synchronous variant) from `bandwidth_test.cu`'s current output — the repeat/best-or-median TODO from that file is still open, so treat these as one sample, not a stable average. Unpinned transfers measured meaningfully slower (~5.5-6.1 GB/s H2D, ~2.6-2.9 GB/s D2H, both noisier run-to-run) and async variants tracked within noise of their sync counterparts on the default stream.
- Even against the corrected 15.75 GB/s PCIe ceiling, ~45% achieved is a real gap (not just an artifact of comparing against the wrong bus) — worth investigating later if a PCIe-bound project needs more headroom.
- TGP/thermal throttling observed: TODO
- Any gap between theoretical and achieved worth flagging for later projects: PCIe link width (x8 vs x16) is the standout finding from P00 — see above.

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
