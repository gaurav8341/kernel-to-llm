// P00 — Device query: reads this machine's actual GPU clocks/bus width (CUDA runtime)
// and negotiated PCIe link generation/width (NVML), then computes theoretical VRAM
// and PCIe bandwidth from those live values. Feeds the "Theoretical peak" table in
// docs/hardware-spec-sheet.md — numbers come from this specific GPU/laptop, not a
// generic spec sheet, since laptop RTX 3050 configs vary by OEM/TGP.
#include <cstdio>
#include <cuda_runtime.h>
#include <nvml.h>

// Useful (post-encoding) unidirectional throughput per lane, GB/s (decimal, 1e9),
// indexed by PCIe generation. Gen 1-2 use 8b/10b encoding, Gen 3+ use 128b/130b.
static const double PCIE_GBPS_PER_LANE[] = {
    0.0,      // no Gen 0
    0.250,    // Gen 1: 2.5 GT/s
    0.500,    // Gen 2: 5.0 GT/s
    0.9846,   // Gen 3: 8.0 GT/s
    1.9692,   // Gen 4: 16.0 GT/s
    3.9385,   // Gen 5: 32.0 GT/s
};

int main() {
    // ---- VRAM bandwidth, via CUDA runtime ----
    int memClockKHz = 0, busWidthBits = 0;
    cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&busWidthBits, cudaDevAttrGlobalMemoryBusWidth, 0);

    double memClockHz = memClockKHz * 1000.0;
    double ddrFactor = 2.0; // GDDR6 is DDR: 2 transfers/clock
    double vramBandwidthGBs = (memClockHz * ddrFactor * busWidthBits / 8.0) / 1e9;

    printf("=== VRAM (from CUDA runtime) ===\n");
    printf("memory clock:      %.0f MHz\n", memClockHz / 1e6);
    printf("bus width:         %d-bit\n", busWidthBits);
    printf("theoretical BW:     %.2f GB/s\n\n", vramBandwidthGBs);

    // ---- PCIe bandwidth, via NVML (link gen/width isn't exposed by the CUDA runtime API) ----
    nvmlInit();
    nvmlDevice_t dev;
    nvmlDeviceGetHandleByIndex(0, &dev);

    unsigned int maxGen = 0, maxWidth = 0, curGen = 0, curWidth = 0;
    nvmlDeviceGetMaxPcieLinkGeneration(dev, &maxGen);
    nvmlDeviceGetMaxPcieLinkWidth(dev, &maxWidth);
    nvmlDeviceGetCurrPcieLinkGeneration(dev, &curGen);
    nvmlDeviceGetCurrPcieLinkWidth(dev, &curWidth);

    double maxPcieGBs = PCIE_GBPS_PER_LANE[maxGen] * maxWidth;
    double curPcieGBs = PCIE_GBPS_PER_LANE[curGen] * curWidth;

    printf("=== PCIe (from NVML) ===\n");
    printf("max link:          Gen%u x%u  -> %.2f GB/s theoretical\n", maxGen, maxWidth, maxPcieGBs);
    printf("current link:      Gen%u x%u  -> %.2f GB/s theoretical\n", curGen, curWidth, curPcieGBs);
    printf("(sample during an active cudaMemcpy to see the negotiated gen under load,\n");
    printf(" not the idle/power-saving state)\n");

    nvmlShutdown();
    return 0;
}
