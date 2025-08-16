###############################################################################
# Makefile — CUDA Stream-Compaction Project
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

# --- Q1 micro-benchmark ---
Q1_SRCS         := src/q1_microbench.cu src/benchmark_utils.cu $(COMMON_SRCS)

# ========= object file lists ================================================
MAIN_OBJS      := $(MAIN_SRCS:src/%.cu=build/%.o)
BENCHMARK_OBJS := $(BENCHMARK_SRCS:src/%.cu=build/%.o)
Q1_OBJS        := $(Q1_SRCS:src/%.cu=build/%.o)

# ========= final binaries ===================================================
TARGET     := build/main
BENCHMARK  := build/benchmark_runner
Q1_BIN     := build/q1_microbench

# ========= CSV output dir & file for Q1 =====================================
CSV_DIR    := csv
Q1_NAME    := q1_breakdown
Q1_CSV     := $(CSV_DIR)/$(Q1_NAME).csv
Q1_ARGS    ?= --k 8 --dist uniform --rates 0.05,0.50,0.95

# ========= default target ===================================================
all: $(TARGET) $(BENCHMARK) $(Q1_BIN)
	@echo "✔️  Build finished ($(BUILD))"

# ========= compile step (.cu → .o) ==========================================
build/%.o: src/%.cu
	@mkdir -p $(dir $@)
	$(NVCC_COMPILE)

# ========= link step ========================================================
$(TARGET): $(MAIN_OBJS)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -o $@ $^

$(BENCHMARK): $(BENCHMARK_OBJS)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -o $@ $^

$(Q1_BIN): $(Q1_OBJS)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -o $@ $^

# ========= run Q1 and dump CSV ==============================================
q1: $(Q1_BIN) | $(CSV_DIR)
	@echo "→ Running $(Q1_BIN) for multiple N values"
	@for N in 1000000 5000000 10000000 20000000; do \
		echo "Running with N=$$N"; \
		$(Q1_BIN) $(Q1_ARGS) --N $$N >> $(Q1_CSV) 2>/dev/null; \
	done
	@echo "✔️  CSV written: $(Q1_CSV)"

$(CSV_DIR):
	@mkdir -p $(CSV_DIR)

# ========= cleanup ==========================================================
clean:
	rm -rf build $(CSV_DIR)

.PHONY: all clean q1