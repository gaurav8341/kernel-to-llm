# tools/

Diagnostic/instrumentation code written by Claude during P00, not part of the benchmark
deliverable itself. Kept separate from `src/` (which holds the project author's own
`bandwidth_test.cu` / `compute_test.cu`) so it's clear at a glance which code is the
learning exercise and which is supporting tooling.

- `device_query.cu` — reads this machine's actual GPU memory clock/bus width and negotiated
  PCIe link generation/width, and computes theoretical VRAM/PCIe bandwidth from them. Feeds
  the "Theoretical peak" table in `docs/hardware-spec-sheet.md`.
- `pinned_memory_layout.cu` — checks whether a `cudaMallocHost` buffer's physical pages are
  contiguous or fragmented, to test the IOMMU-translation-overhead hypothesis for why measured
  bandwidth falls short of the PCIe theoretical ceiling. Needs `sudo` (see file header).

Both are wired into the top-level `Makefile` (`make run-device-query`, `make run-pinned-layout`).
