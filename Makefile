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
  CXXFLAGS = -O2 -std=c++20 --expt-extended-lambda
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
Q1_SRCS         := src/q1_microbench.cu src/benchmark_utils.cu $(COMMON_SRCS)
Q3_SRCS         := src/q3_param_sweep.cu src/benchmark_utils.cu $(COMMON_SRCS)

# ========= object file lists ================================================
MAIN_OBJS      := $(MAIN_SRCS:src/%.cu=build/%.o)
BENCHMARK_OBJS := $(BENCHMARK_SRCS:src/%.cu=build/%.o)
Q1_OBJS        := $(Q1_SRCS:src/%.cu=build/%.o)
Q3_OBJS        := $(Q3_SRCS:src/%.cu=build/%.o)

# ========= final binaries ===================================================
TARGET     := build/main
BENCHMARK  := build/benchmark_runner
Q1_BIN     := build/q1_microbench
Q3_BIN     := build/q3_param_sweep

# ========= CSV output dir & files ===========================================
CSV_DIR        := csv
Q1_NAME        := q1_breakdown
Q1_CSV         := $(CSV_DIR)/$(Q1_NAME).csv
Q1_ARGS        ?= --k 8 --dist uniform --rates 0.05,0.50,0.95

# Q3 CSVs
Q3_BLOCK_CSV   := $(CSV_DIR)/q3_block.csv
Q3_KBITS_CSV   := $(CSV_DIR)/q3_kbits.csv
Q3_HIT_CSV     := $(CSV_DIR)/q3_hit.csv
# 可选重复次数（与 q3_param_sweep.cu 的 --repeats 对应）
Q3_REPEATS     ?= 3

# ========= default target ===================================================
all: $(TARGET) $(BENCHMARK) $(Q1_BIN) $(Q3_BIN)
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

$(Q3_BIN): $(Q3_OBJS)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -o $@ $^

# ========= run Q1 and dump CSV ==============================================
q1: $(Q1_BIN) | $(CSV_DIR)
	@echo "→ Running $(Q1_BIN) for multiple N values"
	@for N in 1000000 5000000 10000000 20000000; do \
		echo "Running with N=$$N"; \
		$(Q1_BIN) $(Q1_ARGS) --N $$N >> $(Q1_CSV) 2>/dev/null; \
	done
	@echo "✔️  CSV written: $(Q1_CSV)"

# ========= run Q3 sweeps and dump CSVs ======================================
# 仅 blockSize 扫描（你已跑通过的）
q3block: $(Q3_BIN) | $(CSV_DIR)
	@echo "→ Running $(Q3_BIN) for blockSize sweep"
	$(Q3_BIN) --sweep block --repeats $(Q3_REPEATS)
	@echo "✔️  CSV written: $(Q3_BLOCK_CSV)"

# 一键跑三种扫描：kBits、blockSize、hit
q3: $(Q3_BIN) | $(CSV_DIR)
	@mkdir -p $(CSV_DIR)
	@echo "→ Q3: kBits sweep"
	$(Q3_BIN) --sweep kbits --repeats $(Q3_REPEATS)
	@echo "→ Q3: blockSize sweep"
	$(Q3_BIN) --sweep block --repeats $(Q3_REPEATS)
	@echo "→ Q3: hit-rate sweep"
	$(Q3_BIN) --sweep hit --repeats $(Q3_REPEATS)
	@echo "✔️  CSVs written: $(Q3_KBITS_CSV), $(Q3_BLOCK_CSV), $(Q3_HIT_CSV)"

# ========= plotting (optional convenience) ==================================
plot_q3:
	python3 scripts/plot_q3.py
	@echo "✔️  Figures saved under figures/"
	
$(CSV_DIR):
	@mkdir -p $(CSV_DIR)

# ========= cleanup ==========================================================
clean:
	rm -rf build $(CSV_DIR)

.PHONY: all clean q1 q3block q3 plot_q3
