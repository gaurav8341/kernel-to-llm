// P00 — Pinned memory layout: checks whether a cudaMallocHost buffer's physical
// pages are actually contiguous, or scattered across many discontiguous 4KB
// frames. Feeds the IOMMU-overhead hypothesis in docs/hardware-spec-sheet.md —
// pinning only guarantees pages can't move/be swapped out, not that they're
// physically contiguous or huge-page-backed, and that distinction is what
// determines how much translation work the IOMMU does per transfer.
//
// Needs root: /proc/<pid>/pagemap has returned zeroed PFNs for unprivileged
// reads since Linux 4.0 (CVE-2015-1420 hardening), so run this with sudo or
// the "pages with valid PFN" count below will read ~0 and the rest of the
// output is meaningless.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <unistd.h>
#include <fcntl.h>
#include <cuda_runtime.h>

int main() {
    size_t size = 1024ULL * 1024 * 1024; // match bandwidth_test.cu's buffer size

    char *buf = nullptr;
    cudaError_t err = cudaMallocHost((void **)&buf, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallocHost failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    long page_size = sysconf(_SC_PAGESIZE);
    size_t num_pages = size / page_size;

    int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) {
        perror("open /proc/self/pagemap");
        cudaFreeHost(buf);
        return 1;
    }

    uint64_t prev_pfn = 0;
    bool have_prev = false;
    size_t current_run = 0, longest_run = 0;
    size_t num_runs = 0;
    size_t valid_pages = 0, unavailable_pages = 0;

    for (size_t i = 0; i < num_pages; i++) {
        uintptr_t vaddr = (uintptr_t)buf + i * page_size;
        uint64_t vpn = vaddr / (uint64_t)page_size;
        uint64_t entry = 0;
        off_t offset = (off_t)(vpn * sizeof(uint64_t));

        if (pread(fd, &entry, sizeof(entry), offset) != (ssize_t)sizeof(entry)) {
            fprintf(stderr, "pread failed at page %zu\n", i);
            break;
        }

        bool present = (entry >> 63) & 1;
        bool swapped = (entry >> 62) & 1;
        uint64_t pfn = entry & ((1ULL << 55) - 1);

        if (!present || swapped || pfn == 0) {
            unavailable_pages++;
            have_prev = false;
            if (current_run > longest_run) longest_run = current_run;
            current_run = 0;
            continue;
        }

        valid_pages++;
        if (have_prev && pfn == prev_pfn + 1) {
            current_run++;
        } else {
            if (current_run > longest_run) longest_run = current_run;
            if (have_prev) num_runs++;
            current_run = 1;
        }
        prev_pfn = pfn;
        have_prev = true;
    }
    if (current_run > longest_run) longest_run = current_run;
    if (have_prev) num_runs++;

    close(fd);
    cudaFreeHost(buf);

    printf("page size:                 %ld bytes\n", page_size);
    printf("total pages:                %zu\n", num_pages);
    printf("pages with valid PFN:       %zu\n", valid_pages);
    printf("pages with unavailable PFN: %zu%s\n", unavailable_pages,
           unavailable_pages == num_pages ? "  <-- all zero: rerun with sudo" : "");

    if (valid_pages > 0) {
        double avg_run = num_runs ? (double)valid_pages / (double)num_runs : 0.0;
        printf("contiguous physical runs:  %zu\n", num_runs);
        printf("longest run:               %zu pages (%.2f MB)\n",
               longest_run, longest_run * page_size / (1024.0 * 1024.0));
        printf("average run length:        %.1f pages (%.4f%% of buffer)\n",
               avg_run, 100.0 * avg_run * page_size / (double)size);
    }

    return 0;
}
