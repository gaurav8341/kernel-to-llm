// Claude-generated tooling (not hand-written by the project author — see tools/README.md).
// P00 — Device query: reads this machine's actual GPU clocks/bus width (CUDA runtime)
// and negotiated PCIe link generation/width (NVML), then computes theoretical VRAM
// and PCIe bandwidth from those live values. Feeds the "Theoretical peak" table in
// docs/hardware-spec-sheet.md — numbers come from this specific GPU/laptop, not a
// generic spec sheet, since laptop RTX 3050 configs vary by OEM/TGP.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <nvml.h>

#include "utils.cuh" // CUDA_CHECK

#define NVML_CHECK(call)                                                                        \
do {                                                                                            \
    nvmlReturn_t nvml_err = (call);                                                             \
    if(nvml_err != NVML_SUCCESS){                                                               \
        fprintf(stderr, "NVML ERROR at %s:%d: %s\n", __FILE__, __LINE__,                        \
                nvmlErrorString(nvml_err));                                                     \
        exit(EXIT_FAILURE);                                                                     \
    }                                                                                           \
} while(0)

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

static const unsigned int MAX_KNOWN_PCIE_GEN =
    sizeof(PCIE_GBPS_PER_LANE) / sizeof(PCIE_GBPS_PER_LANE[0]) - 1;

// NVML reports the generation as a raw number; don't index the table with it blindly.
static double pcie_gbps(unsigned int gen, unsigned int width) {
    if (gen == 0 || gen > MAX_KNOWN_PCIE_GEN) {
        fprintf(stderr, "warning: unknown PCIe generation %u (table covers Gen1-%u)\n",
                gen, MAX_KNOWN_PCIE_GEN);
        return 0.0;
    }
    return PCIE_GBPS_PER_LANE[gen] * width;
}

int main() {
    // ---- VRAM bandwidth, via CUDA runtime ----
    int memClockKHz = 0, busWidthBits = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&busWidthBits, cudaDevAttrGlobalMemoryBusWidth, 0));

    double memClockHz = memClockKHz * 1000.0;
    double ddrFactor = 2.0; // GDDR6 is DDR: 2 transfers/clock
    double vramBandwidthGBs = (memClockHz * ddrFactor * busWidthBits / 8.0) / 1e9;

    printf("=== VRAM (from CUDA runtime) ===\n");
    printf("memory clock:      %.0f MHz\n", memClockHz / 1e6);
    printf("bus width:         %d-bit\n", busWidthBits);
    printf("theoretical BW:     %.2f GB/s\n\n", vramBandwidthGBs);

    // ---- PCIe bandwidth, via NVML (link gen/width isn't exposed by the CUDA runtime API) ----
    NVML_CHECK(nvmlInit());
    nvmlDevice_t dev;
    NVML_CHECK(nvmlDeviceGetHandleByIndex(0, &dev));

    unsigned int maxGen = 0, maxWidth = 0, curGen = 0, curWidth = 0;
    NVML_CHECK(nvmlDeviceGetMaxPcieLinkGeneration(dev, &maxGen));
    NVML_CHECK(nvmlDeviceGetMaxPcieLinkWidth(dev, &maxWidth));
    NVML_CHECK(nvmlDeviceGetCurrPcieLinkGeneration(dev, &curGen));
    NVML_CHECK(nvmlDeviceGetCurrPcieLinkWidth(dev, &curWidth));

    double maxPcieGBs = pcie_gbps(maxGen, maxWidth);
    double curPcieGBs = pcie_gbps(curGen, curWidth);

    printf("=== PCIe (from NVML) ===\n");
    printf("max link:          Gen%u x%u  -> %.2f GB/s theoretical\n", maxGen, maxWidth, maxPcieGBs);
    printf("current link:      Gen%u x%u  -> %.2f GB/s theoretical\n", curGen, curWidth, curPcieGBs);
    printf("(sample during an active cudaMemcpy to see the negotiated gen under load,\n");
    printf(" not the idle/power-saving state)\n");

    nvmlShutdown();
    return 0;
}
