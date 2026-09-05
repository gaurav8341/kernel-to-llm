NVCC := nvcc
ARCH := -arch=sm_86
FLAGS := -O3 $(ARCH) -I../../common/include

BIN_DIR := bin

.PHONY: all clean run-bandwidth run-compute run-device-query

all: $(BIN_DIR)/bandwidth_test $(BIN_DIR)/compute_test $(BIN_DIR)/device_query

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/bandwidth_test: src/bandwidth_test.cu | $(BIN_DIR)
	$(NVCC) $(FLAGS) $< -o $@

$(BIN_DIR)/compute_test: src/compute_test.cu | $(BIN_DIR)
	$(NVCC) $(FLAGS) $< -o $@

$(BIN_DIR)/device_query: src/device_query.cu | $(BIN_DIR)
	$(NVCC) $(FLAGS) $< -o $@ -lnvidia-ml

run-bandwidth: $(BIN_DIR)/bandwidth_test
	./$(BIN_DIR)/bandwidth_test

run-compute: $(BIN_DIR)/compute_test
	./$(BIN_DIR)/compute_test

run-device-query: $(BIN_DIR)/device_query
	./$(BIN_DIR)/device_query

clean:
	rm -rf $(BIN_DIR)
