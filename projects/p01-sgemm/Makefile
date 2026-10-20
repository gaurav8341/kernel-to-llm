# P01 — tiled SGEMM. Same shape as p00-benchmark/Makefile, two things added:
# kernels are compiled into objects and linked into both mains, and the bench
# links cuBLAS.
#
#   make                                  build bench + checker
#   make run-check                        CPU reference, 512^3
#   make run-check ARGS="512 256 128"     any M N K
#   make run-check ARGS="--data correctness/data"
#   make reference SHAPE="1024 1024 1024" numpy reference -> correctness/data/
#   make run-bench                        512^2, 1024^2, 4096^2
#   make run-bench SIZES="2048"           one size
#   make run-probe                        sweep block shapes x sizes (tools/)
#   make run-probe ARGS="--sizes 4096 --blocks 32x8,8x32"
#   make ptx / make sass                  dump + opcode histogram
#   make clean

NVCC   := nvcc
ARCH   := -arch=sm_86
COMMON := ../../common/include

# Optional tile-edge override: make TILE=32
#
# src/sgemm.cuh wraps its TILE_DIM #define in #ifndef precisely so this works —
# without the guard, nvcc warns "TILE_DIM redefined", the header wins, and the
# flag silently does nothing.
#
# CAVEAT: make decides what to rebuild from file timestamps, and it has no idea
# the compiler flags changed. Changing TILE therefore needs a `make clean` first,
# or you link new objects against stale ones. The ptx/sass targets are immune —
# they always recompile from source.
ifneq ($(TILE),)
TILE_FLAG := -DTILE_DIM=$(TILE)
endif

# -Isrc so correctness/ and bench/ can say #include "sgemm.cuh" without a path.
FLAGS := -O3 $(ARCH) -I$(COMMON) -Isrc $(TILE_FLAG)

BIN_DIR := bin
OBJ_DIR := $(BIN_DIR)/obj

# --- The design decision -----------------------------------------------------
# Each kernel is compiled ONCE into a .o and linked into both mains, rather than
# #included into each.
#
# The reason is not compile time, it is trust. #including the .cu files would
# compile the kernels twice, under whatever flags each main happened to be built
# with, into two independent binaries. Nothing would then guarantee that the
# machine code bench/ timed is the machine code correctness/ verified — and that
# equality is the one thing every number in this project rests on. One object
# linked into both makes it true by construction instead of by hope.
# -----------------------------------------------------------------------------
KERNEL_SRCS := src/sgemm_naive.cu src/sgemm_tiled.cu
KERNEL_OBJS := $(patsubst src/%.cu,$(OBJ_DIR)/%.o,$(KERNEL_SRCS))

# Headers are prerequisites too. Without sgemm.cuh listed, editing the shared
# contract (the launcher signature, the layout rules, TILE_DIM) would leave
# already-built objects untouched and silently stale.
HEADERS := src/sgemm.cuh $(COMMON)/check.cuh $(COMMON)/utils.cuh

PTX_SASS := ../../common/ptx_sass

# Overridable on the command line; empty means "use the program's own defaults".
ARGS  ?=
SIZES ?=
SHAPE ?=

# .PHONY tells make these names are commands, not files to produce. Without it,
# a file named `clean` in this directory would count as the target, make would
# find it already up to date, and `make clean` would do nothing at all.
.PHONY: all run-bench run-check run-probe reference ptx sass clean

all: $(BIN_DIR)/sgemm_bench $(BIN_DIR)/check_sgemm $(BIN_DIR)/sgemm_probe

# Order-only prerequisite (everything after the `|`): make requires the
# directory to EXIST but ignores its timestamp. That distinction matters —
# a directory's mtime changes every time a file is written into it, so a normal
# prerequisite would make every binary look out of date after any build.
$(BIN_DIR) $(OBJ_DIR):
	mkdir -p $@

$(OBJ_DIR)/%.o: src/%.cu $(HEADERS) | $(OBJ_DIR)
	$(NVCC) $(FLAGS) -c $< -o $@

$(BIN_DIR)/check_sgemm: correctness/check_sgemm.cu $(KERNEL_OBJS) $(HEADERS) | $(BIN_DIR)
	$(NVCC) $(FLAGS) $< $(KERNEL_OBJS) -o $@

# -lcublas only here: the bench needs cuBLAS as its baseline and its reference,
# the checker has its own CPU oracle and must not depend on the library it is
# meant to be independent of.
$(BIN_DIR)/sgemm_bench: bench/sgemm_bench.cu $(KERNEL_OBJS) $(HEADERS) | $(BIN_DIR)
	$(NVCC) $(FLAGS) $< $(KERNEL_OBJS) -o $@ -lcublas

# The probe launches sgemmNaiveKernel directly to vary the block shape, so it
# needs the kernel object and cuBLAS (its reference), same as the bench.
$(BIN_DIR)/sgemm_probe: tools/sgemm_probe.cu $(KERNEL_OBJS) $(HEADERS) | $(BIN_DIR)
	$(NVCC) $(FLAGS) $< $(KERNEL_OBJS) -o $@ -lcublas

run-probe: $(BIN_DIR)/sgemm_probe
	./$(BIN_DIR)/sgemm_probe $(ARGS)

run-bench: $(BIN_DIR)/sgemm_bench
	./$(BIN_DIR)/sgemm_bench $(SIZES)

run-check: $(BIN_DIR)/check_sgemm
	./$(BIN_DIR)/check_sgemm $(ARGS)

reference:
	python3 correctness/reference.py $(SHAPE)

# Not built from $(BIN_DIR): these read the .cu sources directly and always
# recompile, so they are the one safe way to sweep TILE without a clean.
ptx:
	$(PTX_SASS)/dump_ptx.sh src/sgemm_naive.cu reports/ptx $(TILE_FLAG)
	$(PTX_SASS)/dump_ptx.sh src/sgemm_tiled.cu reports/ptx $(TILE_FLAG)

sass:
	$(PTX_SASS)/dump_sass.sh src/sgemm_naive.cu reports/sass $(TILE_FLAG)
	$(PTX_SASS)/dump_sass.sh src/sgemm_tiled.cu reports/sass $(TILE_FLAG)

clean:
	rm -rf $(BIN_DIR)
