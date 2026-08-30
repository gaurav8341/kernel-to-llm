# Hardware Spec Sheet — RTX 3050 Laptop (GA106, sm_86)

*Produced by P00. Referenced by every later project to judge "% of peak" achieved.*

## Theoretical peak

| Metric | Value | How computed |
|---|---|---|
| FP32 TFLOPS (theoretical) | TODO | SMs × CUDA cores/SM × boost clock (GHz) × 2 (FMA) |
| VRAM bandwidth (theoretical) | TODO | memory clock × bus width (128-bit) × DDR factor / 8 |
| PCIe bandwidth (theoretical) | TODO | PCIe gen/lanes on this laptop |

## Measured (this machine)

| Metric | Achieved | % of theoretical |
|---|---|---|
| H2D bandwidth (GB/s) | TODO | TODO |
| D2H bandwidth (GB/s) | TODO | TODO |
| FP32 compute (TFLOPS) | TODO | TODO |

## Notes

- TGP/thermal throttling observed: TODO
- Any gap between theoretical and achieved worth flagging for later projects: TODO
