# ========= user-selectable build type =========
BUILD ?= release          # 可用: release / debug

# ========= common paths =========
NVCC      = nvcc
INCLUDES  = -Iinclude

# ========= flags per build type =========
ifeq ($(BUILD),debug)
  CXXFLAGS = -O0 -g -G --generate-line-info -std=c++20 --expt-extended-lambda
else
  CXXFLAGS = -O2          -std=c++20 --expt-extended-lambda
endif

# -dc 生成含 device code 的 .o
NVCC_COMPILE = $(NVCC) $(CXXFLAGS) -dc $(INCLUDES) -c $< -o $@

# ========= source groups =========
COMMON_SRCS := \
    src/morton.cu \
    src/utils.cu \
    src/stream_compaction.cu \
    src/stream_compaction_bin.cu

MAIN_SRCS       := src/main.cu $(COMMON_SRCS)
BENCHMARK_SRCS  := src/benchmark_runner.cu src/benchmark_utils.cu $(COMMON_SRCS)

# ========= object lists =========
MAIN_OBJS      := $(MAIN_SRCS:src/%.cu=build/%.o)
BENCHMARK_OBJS := $(BENCHMARK_SRCS:src/%.cu=build/%.o)

# ========= targets =========
TARGET     := build/main
BENCHMARK  := build/benchmark_runner

# default
all: $(TARGET) $(BENCHMARK)
	@echo "✔️  Build finished ($(BUILD))"

# ========= compile every .cu -> .o =========
build/%.o: src/%.cu
	@mkdir -p $(dir $@)
	$(NVCC_COMPILE)

# ========= link stage (no -dc) =========
$(TARGET): $(MAIN_OBJS)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -o $@ $^

$(BENCHMARK): $(BENCHMARK_OBJS)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -o $@ $^

# ========= clean =========
clean:
	rm -rf build

.PHONY: all clean
