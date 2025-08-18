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

# ========= Q4: distribution study (uniform/clustered/skewed) ================
Q4_SRCS := src/q4_distribution.cu src/benchmark_utils.cu $(COMMON_SRCS)

# ========= Q5: scalability (dataset size sweep) =============================
Q5_SRCS := src/q5_scalability.cu src/benchmark_utils.cu $(COMMON_SRCS)

# ========= Q6: ablation study (gather/no-gather, binning/no-binning, force) =
Q6_SRCS := src/q6_ablation.cu src/benchmark_utils.cu $(COMMON_SRCS)

# ========= Q7: per-bin aggregation proxy (atomic vs slices) =================
Q7_SRCS := src/q7_proxy_bins_agg.cu $(COMMON_SRCS)

# ========= object file lists ================================================
MAIN_OBJS      := $(MAIN_SRCS:src/%.cu=build/%.o)
BENCHMARK_OBJS := $(BENCHMARK_SRCS:src/%.cu=build/%.o)
Q1_OBJS        := $(Q1_SRCS:src/%.cu=build/%.o)
Q3_OBJS        := $(Q3_SRCS:src/%.cu=build/%.o)
Q4_OBJS        := $(Q4_SRCS:src/%.cu=build/%.o)
Q5_OBJS        := $(Q5_SRCS:src/%.cu=build/%.o)
Q6_OBJS        := $(Q6_SRCS:src/%.cu=build/%.o)
Q7_OBJS        := $(Q7_SRCS:src/%.cu=build/%.o)

# ========= final binaries ====================================================
TARGET     := build/main
BENCHMARK  := build/benchmark_runner
Q1_BIN     := build/q1_microbench
Q3_BIN     := build/q3_param_sweep
Q4_BIN     := build/q4_distribution
Q5_BIN     := build/q5_scalability
Q6_BIN     := build/q6_ablation
Q7_BIN     := build/q7_proxy

# ========= CSV output dir & files ===========================================
CSV_DIR        := csv

# Q1 CSVs
Q1_NAME        := q1_breakdown
Q1_CSV         := $(CSV_DIR)/$(Q1_NAME).csv
Q1_ARGS        ?= --k 8 --dist uniform --rates 0.05,0.50,0.95

# Q3 CSVs
Q3_BLOCK_CSV   := $(CSV_DIR)/q3_block.csv
Q3_KBITS_CSV   := $(CSV_DIR)/q3_kbits.csv
Q3_HIT_CSV     := $(CSV_DIR)/q3_hit.csv
# 可选重复次数（与 q3_param_sweep.cu 的 --repeats 对应）
Q3_REPEATS     ?= 3

# Q4 CSVs (per-dist and merged)
Q4_UNIFORM_CSV   := $(CSV_DIR)/q4_uniform.csv
Q4_CLUSTERED_CSV := $(CSV_DIR)/q4_clustered.csv
Q4_SKEWED_CSV    := $(CSV_DIR)/q4_skewed.csv
Q4_MERGED_CSV    := $(CSV_DIR)/q4_all.csv

# 固定参数（可在命令行覆盖：make q4 Q4_N=20000000 ...）
Q4_N        ?= 10000000
Q4_KBITS    ?= 8
Q4_HIT      ?= 0.50
Q4_SEED     ?= 1234
# 需要跑的算法变体（逗号分隔，传给 q4_distribution 的 --variants）
Q4_VARIANTS ?= naive,planB,planA_shared,planA_warp,planA_bitmask,thrust

# Q5 CSVs (scalability)
Q5_CSV      := $(CSV_DIR)/q5_scaling.csv
# Defaults (override on CLI if needed)
Q5_NS       ?= 1M,5M,10M,20M,50M
Q5_KBITS    ?= 8
Q5_HIT      ?= 0.50
Q5_DIST     ?= uniform
Q5_SEED     ?= 1234
Q5_REPEAT   ?= 5
Q5_BLOCK    ?= 256
Q5_VARIANTS ?= planB,planA_shared,planA_warp,planA_bitmask
Q5_ROOF     ?= proxy_v1

# Q6 CSV (ablation)
Q6_CSV      := $(CSV_DIR)/q6_ablation.csv
Q6_N        ?= 20000000
Q6_KBITS    ?= 8
Q6_HIT      ?= 0.50
Q6_DIST     ?= uniform
Q6_SEED     ?= 1234
Q6_REPEATS  ?= 5
Q6_BLOCK    ?= 256

# Q7 CSV (per-bin aggregation proxy)
Q7_CSV      := $(CSV_DIR)/q7_proxy.csv
Q7_N        ?= 20000000
Q7_KBITS    ?= 8
Q7_HIT      ?= 0.50
Q7_DIST     ?= uniform          # uniform|clustered|skewed
Q7_SEED     ?= 1234
Q7_REPEAT   ?= 5
Q7_BLOCK    ?= 256

# ========= default target ====================================================
all: $(TARGET) $(BENCHMARK) $(Q1_BIN) $(Q3_BIN) $(Q4_BIN) $(Q5_BIN) $(Q6_BIN) $(Q7_BIN)
	@echo "✔️  Build finished ($(BUILD))"

# ========= compile step (.cu → .o) ==========================================
build/%.o: src/%.cu
	@mkdir -p $(dir $@)
	$(NVCC_COMPILE)

# ========= link step =========================================================
$(TARGET): $(MAIN_OBJS)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -o $@ $^

$(BENCHMARK): $(BENCHMARK_OBJS)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -o $@ $^

$(Q1_BIN): $(Q1_OBJS)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -o $@ $^

$(Q3_BIN): $(Q3_OBJS)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -o $@ $^

$(Q4_BIN): $(Q4_OBJS)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -o $@ $^

$(Q5_BIN): $(Q5_OBJS)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -o $@ $^

$(Q6_BIN): $(Q6_OBJS)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -o $@ $^

$(Q7_BIN): $(Q7_OBJS)
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

# ========= run Q4 and dump CSVs (per dist + merged) =========================
q4: $(Q4_BIN) | $(CSV_DIR)
	@echo "→ Running Q4: uniform / clustered / skewed"
	$(Q4_BIN) --N $(Q4_N) --k $(Q4_KBITS) --rates $(Q4_HIT) --seed $(Q4_SEED) \
	          --dist uniform   --variants $(Q4_VARIANTS) --csv $(Q4_UNIFORM_CSV)
	$(Q4_BIN) --N $(Q4_N) --k $(Q4_KBITS) --rates $(Q4_HIT) --seed $(Q4_SEED) \
	          --dist clustered --variants $(Q4_VARIANTS) --csv $(Q4_CLUSTERED_CSV)
	$(Q4_BIN) --N $(Q4_N) --k $(Q4_KBITS) --rates $(Q4_HIT) --seed $(Q4_SEED) \
	          --dist skewed    --variants $(Q4_VARIANTS) --csv $(Q4_SKEWED_CSV)
	@echo "→ Merging Q4 CSVs → $(Q4_MERGED_CSV)"
	@{ \
	  head -n 1 $(Q4_UNIFORM_CSV) > $(Q4_MERGED_CSV); \
	  tail -n +2 $(Q4_UNIFORM_CSV)   >> $(Q4_MERGED_CSV); \
	  tail -n +2 $(Q4_CLUSTERED_CSV) >> $(Q4_MERGED_CSV); \
	  tail -n +2 $(Q4_SKEWED_CSV)    >> $(Q4_MERGED_CSV); \
	}
	@echo "✔️  Q4 CSVs written: $(Q4_UNIFORM_CSV), $(Q4_CLUSTERED_CSV), $(Q4_SKEWED_CSV), $(Q4_MERGED_CSV)"

# ========= run Q5 scalability sweep and dump CSV ============================
q5: $(Q5_BIN) | $(CSV_DIR)
	@echo "→ Running Q5: scalability (N sweep)"
	$(Q5_BIN) \
	  --Ns $(Q5_NS) \
	  --k $(Q5_KBITS) --hit $(Q5_HIT) --dist $(Q5_DIST) --seed $(Q5_SEED) \
	  --variants $(Q5_VARIANTS) --block $(Q5_BLOCK) --repeat $(Q5_REPEAT) \
	  --csv $(Q5_CSV) --roofline $(Q5_ROOF)
	@echo "✔️  Q5 CSV written: $(Q5_CSV)"

# ========= run Q6 ablation set and dump CSV =================================
q6: $(Q6_BIN) | $(CSV_DIR)
	@echo "→ Running Q6: ablation study → $(Q6_CSV)"
	@rm -f $(Q6_CSV)
	# Baseline: Plan A + Auto + gather + binning
	$(Q6_BIN) --N $(Q6_N) --k $(Q6_KBITS) --hit $(Q6_HIT) --dist $(Q6_DIST) \
	          --seed $(Q6_SEED) --plan A --force-kernel auto --repeats $(Q6_REPEATS) \
	          --block $(Q6_BLOCK) --csv $(Q6_CSV)
	# Plan A: no-gather (需要你在源码里开启 scatterToBins 路径方可生效)
	@tmp=$$(mktemp); \
	$(Q6_BIN) --N $(Q6_N) --k $(Q6_KBITS) --hit $(Q6_HIT) --dist $(Q6_DIST) \
	          --seed $(Q6_SEED) --plan A --no-gather --force-kernel auto \
	          --repeats $(Q6_REPEATS) --block $(Q6_BLOCK) --csv $$tmp; \
	tail -n +2 $$tmp >> $(Q6_CSV); rm -f $$tmp
	# Plan A: no-binning
	@tmp=$$(mktemp); \
	$(Q6_BIN) --N $(Q6_N) --k $(Q6_KBITS) --hit $(Q6_HIT) --dist $(Q6_DIST) \
	          --seed $(Q6_SEED) --plan A --no-binning --force-kernel auto \
	          --repeats $(Q6_REPEATS) --block $(Q6_BLOCK) --csv $$tmp; \
	tail -n +2 $$tmp >> $(Q6_CSV); rm -f $$tmp
	# Plan A: force shared/warp/bitmask
	@for K in shared warp bitmask; do \
	  tmp=$$(mktemp); \
	  $(Q6_BIN) --N $(Q6_N) --k $(Q6_KBITS) --hit $(Q6_HIT) --dist $(Q6_DIST) \
	            --seed $(Q6_SEED) --plan A --force-kernel $$K \
	            --repeats $(Q6_REPEATS) --block $(Q6_BLOCK) --csv $$tmp; \
	  tail -n +2 $$tmp >> $(Q6_CSV); rm -f $$tmp; \
	done
	# Plan B: baseline
	@tmp=$$(mktemp); \
	$(Q6_BIN) --N $(Q6_N) --k $(Q6_KBITS) --hit $(Q6_HIT) --dist $(Q6_DIST) \
	          --seed $(Q6_SEED) --plan B --repeats $(Q6_REPEATS) \
	          --block $(Q6_BLOCK) --csv $$tmp; \
	tail -n +2 $$tmp >> $(Q6_CSV); rm -f $$tmp
	# Plan B: no-binning
	@tmp=$$(mktemp); \
	$(Q6_BIN) --N $(Q6_N) --k $(Q6_KBITS) --hit $(Q6_HIT) --dist $(Q6_DIST) \
	          --seed $(Q6_SEED) --plan B --no-binning --repeats $(Q6_REPEATS) \
	          --block $(Q6_BLOCK) --csv $$tmp; \
	tail -n +2 $$tmp >> $(Q6_CSV); rm -f $$tmp
	@echo "✔️  Q6 CSV written: $(Q6_CSV)"

# ========= run Q7 per-bin aggregation proxy and dump CSV ====================
q7: $(Q7_BIN) | $(CSV_DIR)
	@echo "→ Running Q7: per-bin aggregation proxy → $(Q7_CSV)"
	$(Q7_BIN) --N $(Q7_N) --k $(Q7_KBITS) --hit $(Q7_HIT) --dist $(Q7_DIST) \
	          --seed $(Q7_SEED) --block $(Q7_BLOCK) --repeat $(Q7_REPEAT) \
	          --csv $(Q7_CSV)
	@echo "✔️  Q7 CSV written: $(Q7_CSV)"

# ========= plotting (optional convenience) ==================================
plot_q3:
	python3 scripts/plot_q3.py
	@echo "✔️  Figures saved under figures/"

plot_q4:
	python3 scripts/plot_q4.py
	@echo "✔️  Q4 figures saved under figures/"

$(CSV_DIR):
	@mkdir -p $(CSV_DIR)

# ========= cleanup ===========================================================
clean:
	rm -rf build $(CSV_DIR)

.PHONY: all clean q1 q3block q3 q4 q5 q6 q7 plot_q3 plot_q4
