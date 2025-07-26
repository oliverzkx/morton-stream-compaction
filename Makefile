###############################################################################
# Makefile — CUDA Stream-Compaction Project
#
# * Two build flavours are supported via the BUILD variable:
#       release (default) — optimised, no debug symbols
#       debug             — full symbols, -G for Nsight-Compute / Nsight-Systems
#
# * No build rules have been changed—only comments were rewritten in English.
###############################################################################

# ========= user-selectable build type ========================================
BUILD ?= release          # valid values: release | debug

# ========= common paths ======================================================
NVCC      = nvcc
INCLUDES  = -Iinclude      # add more -I switches here if needed

# ========= compiler flags per build type =====================================
ifeq ($(BUILD),debug)
  CXXFLAGS = -O0 -g -G --generate-line-info -std=c++20 --expt-extended-lambda
else
  CXXFLAGS = -O2           -std=c++20 --expt-extended-lambda
endif

# -dc produces device-code–containing object files required for separate link
NVCC_COMPILE = $(NVCC) $(CXXFLAGS) -dc $(INCLUDES) -c $< -o $@

# ========= source file groups ===============================================
COMMON_SRCS := \
    src/morton.cu \
    src/utils.cu \
    src/stream_compaction.cu \
    src/stream_compaction_bin.cu

MAIN_SRCS       := src/main.cu $(COMMON_SRCS)
BENCHMARK_SRCS  := src/benchmark_runner.cu src/benchmark_utils.cu $(COMMON_SRCS)

# ========= object file lists (generated automatically) =======================
MAIN_OBJS      := $(MAIN_SRCS:src/%.cu=build/%.o)
BENCHMARK_OBJS := $(BENCHMARK_SRCS:src/%.cu=build/%.o)

# ========= final binaries ====================================================
TARGET     := build/main
BENCHMARK  := build/benchmark_runner

# default target --------------------------------------------------------------
all: $(TARGET) $(BENCHMARK)
	@echo "✔️  Build finished ($(BUILD))"

# ========= compile step (.cu → .o) ===========================================
build/%.o: src/%.cu
	@mkdir -p $(dir $@)
	$(NVCC_COMPILE)

# ========= link step (no -dc here) ===========================================
$(TARGET): $(MAIN_OBJS)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -o $@ $^

$(BENCHMARK): $(BENCHMARK_OBJS)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -o $@ $^

# ========= cleanup ===========================================================
clean:
	rm -rf build

.PHONY: all clean
