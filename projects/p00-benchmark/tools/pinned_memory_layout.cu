// Claude-generated tooling (not hand-written by the project author — see tools/README.md).
// P00 — Pinned memory layout: checks whether a pinned host buffer's physical
// pages are actually contiguous, or scattered across many discontiguous 4KB
// frames. Feeds the IOMMU-overhead hypothesis in docs/hardware-spec-sheet.md —
// pinning only guarantees pages can't move/be swapped out, not that they're
// physically contiguous or huge-page-backed, and that distinction is what
// determines how much translation work the IOMMU does per transfer.
//
// Compares two allocation strategies back to back:
//   1. cudaMallocHost          — CUDA allocates + pins in one step (baseline).
//   2. mmap + MADV_HUGEPAGE,
//      then cudaHostRegister   — THP-hinted anonymous mapping, pinned after
//                                the fact, to see whether asking for huge
//                                pages collapses the run count.
//
// Needs root: /proc/<pid>/pagemap has returned zeroed PFNs for unprivileged
// reads since Linux 4.0 (CVE-2015-1420 hardening), so run this with sudo or
// the "pages with valid PFN" count below will read ~0 and the rest of the
// output is meaningless.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <cuda_runtime.h>

static void report_layout(const char *label, char *buf, size_t size, int pagemap_fd) {
    long page_size = sysconf(_SC_PAGESIZE);
    size_t num_pages = size / page_size;

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

        if (pread(pagemap_fd, &entry, sizeof(entry), offset) != (ssize_t)sizeof(entry)) {
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

    printf("=== %s ===\n", label);
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
    printf("\n");
}

int main() {
    size_t size = 1024ULL * 1024 * 1024; // match bandwidth_test.cu's buffer size

    int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) {
        perror("open /proc/self/pagemap");
        return 1;
    }

    // --- Strategy 1: cudaMallocHost (baseline) ---
    char *cuda_buf = nullptr;
    cudaError_t err = cudaMallocHost((void **)&cuda_buf, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallocHost failed: %s\n", cudaGetErrorString(err));
        close(fd);
        return 1;
    }
    report_layout("cudaMallocHost", cuda_buf, size, fd);
    cudaFreeHost(cuda_buf);

    // --- Strategy 2: mmap + MADV_HUGEPAGE, then cudaHostRegister ---
    void *thp_buf = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (thp_buf == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }

    if (madvise(thp_buf, size, MADV_HUGEPAGE) != 0) {
        perror("madvise(MADV_HUGEPAGE)");
        // Not fatal — fall through and report whatever layout we actually get.
    }

    // Fault every page in before registering: cudaHostRegister pins the pages
    // that exist at call time, so unfaulted pages would just show up as
    // "unavailable" rather than reflecting the THP layout.
    memset(thp_buf, 0, size);

    err = cudaHostRegister(thp_buf, size, cudaHostRegisterDefault);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaHostRegister failed: %s\n", cudaGetErrorString(err));
        munmap(thp_buf, size);
        close(fd);
        return 1;
    }
    report_layout("mmap + MADV_HUGEPAGE + cudaHostRegister", (char *)thp_buf, size, fd);

    cudaHostUnregister(thp_buf);
    munmap(thp_buf, size);
    close(fd);

    return 0;
}
