#!/usr/bin/env bash
###############################################################################
#  Minimal CPU ↔ GPU speed-up measurer for Morton Stream-Compaction project
#  把可执行文件 main 跑 5 次取平均，算 4 个 GPU kernel 的加速比
###############################################################################
PTS=${1:-10000000}   # 点数，缺省 1e7
RUNS=${2:-5}         # 每个实现跑几次，缺省 5
set -euo pipefail

echo "==> make -j"
make -j

#---- 找到可执行文件 -------------------------------------------------------#
if   [[ -x build/main ]]; then BIN=build/main
elif [[ -x ./main      ]]; then BIN=./main
else
  echo "❌  找不到可执行文件 build/main 或 ./main" >&2
  exit 1
fi
echo "Binary : $BIN"
echo "Points : $PTS   Runs/variant : $RUNS"
echo "----------------------------------------------------------------"

#---- 帮助函数：执行命令 N 次取平均 ---------------------------------------#
avg() {
  local cmd="$1" sum=0
  for _ in $(seq "$RUNS"); do
    ms=$( eval "$cmd" \
          | grep -Eo '[0-9]+\.[0-9]+ ms' | head -n1 | awk '{print $1}' )
    ms=${ms:-0}
    sum=$(echo "$sum + $ms" | bc -l)
  done
  echo "scale=6; $sum / $RUNS" | bc -l
}

#---- CPU 基线 ------------------------------------------------------------#
CPU_CMD="$BIN -n $PTS -c -t"
AVG_CPU=$(avg "$CPU_CMD")
printf "Avg CPU         : %.3f ms\n" "$AVG_CPU"

#---- GPU kernel 命令 ------------------------------------------------------#
SHARED_CMD="$BIN -n $PTS -g --mode bin   --kernel shared   -t"
WARP_CMD  ="$BIN -n $PTS -g --mode bin   --kernel warp     -t"
BMASK_CMD="$BIN -n $PTS -g --mode bin   --kernel bitmask  -t"
NAIVE_CMD="$BIN -n $PTS -g --mode naive                 -t"

AVG_SHARED=$(avg "$SHARED_CMD")
AVG_WARP=$(avg   "$WARP_CMD")
AVG_BMASK=$(avg  "$BMASK_CMD")
AVG_NAIVE=$(avg  "$NAIVE_CMD")

printf "Avg GPU-shared  : %.3f ms\n" "$AVG_SHARED"
printf "Avg GPU-warp    : %.3f ms\n" "$AVG_WARP"
printf "Avg GPU-bitmask : %.3f ms\n" "$AVG_BMASK"
printf "Avg GPU-naive   : %.3f ms\n" "$AVG_NAIVE"

spd () { echo "scale=2; $1 / $2" | bc -l; }

echo
printf "%-13s %-11s %-10s\n" "Kernel" "Avg(ms)" "Speed-Up×"
printf "%-13s %-11.3f --\n"   "CPU"        "$AVG_CPU"
printf "%-13s %-11.3f %-10.2f\n" "GPU-shared" "$AVG_SHARED"  "$(spd $AVG_CPU $AVG_SHARED)"
printf "%-13s %-11.3f %-10.2f\n" "GPU-warp"   "$AVG_WARP"    "$(spd $AVG_CPU $AVG_WARP)"
printf "%-13s %-11.3f %-10.2f\n" "GPU-bitmask""$AVG_BMASK"   "$(spd $AVG_CPU $AVG_BMASK)"
printf "%-13s %-11.3f %-10.2f\n" "GPU-naive"  "$AVG_NAIVE"   "$(spd $AVG_CPU $AVG_NAIVE)"
