# ==== Compiler & Flags ====
NVCC     = nvcc
CFLAGS   = -O2 -std=c++20 --expt-extended-lambda              # ✅ Enable C++20
INCLUDES = -Iinclude

# ==== Targets ====
TARGET     = build/main
BENCHMARK  = build/benchmark_runner

# ==== Source Files ====
COMMON_SRCS = \
    src/morton.cu \
    src/utils.cu \
    src/stream_compaction.cu \
    src/stream_compaction_bin.cu   

SRCS            = src/main.cu $(COMMON_SRCS)
BENCHMARK_SRCS  = src/benchmark_runner.cu src/benchmark_utils.cu $(COMMON_SRCS)

# ==== Default build ====
all: $(TARGET) $(BENCHMARK)

# ==== Build main ====
$(TARGET): $(SRCS)
	@mkdir -p $(dir $@)
	$(NVCC) $(CFLAGS) $(INCLUDES) -o $@ $^

# ==== Build benchmark_runner ====
$(BENCHMARK): $(BENCHMARK_SRCS)
	@mkdir -p $(dir $@)
	$(NVCC) $(CFLAGS) $(INCLUDES) -o $@ $^

# ==== Clean ====
clean:
	rm -f $(TARGET) $(BENCHMARK)

.PHONY: all clean
