# ==== Compiler & Flags ====
NVCC     = nvcc
CFLAGS   = -O2 -std=c++20              # ✅ Enable C++17
INCLUDES = -Iinclude

# ==== Targets ====
TARGET     = build/main
BENCHMARK  = build/benchmark_runner

# ==== Source Files ====
COMMON_SRCS    = src/morton.cu src/utils.cu src/stream_compaction.cu
SRCS           = src/main.cu $(COMMON_SRCS)
BENCHMARK_SRCS = src/benchmark_runner.cu src/benchmark_utils.cu $(COMMON_SRCS)

# ==== Default build ====
all: $(TARGET) $(BENCHMARK)

# ==== Build main ====
$(TARGET): $(SRCS)
	$(NVCC) $(CFLAGS) $(INCLUDES) -o $@ $^

# ==== Build benchmark_runner ====
$(BENCHMARK): $(BENCHMARK_SRCS)
	$(NVCC) $(CFLAGS) $(INCLUDES) -o $@ $^

# ==== Clean ====
clean:
	rm -f $(TARGET) $(BENCHMARK)

.PHONY: all clean benchmark
