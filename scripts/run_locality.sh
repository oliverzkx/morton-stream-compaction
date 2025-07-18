#!/usr/bin/env bash
set -euo pipefail

N=1048576
K=8
BIN="./build/main"
OUT="scripts/locality_data.csv"      # 建议显式写到 scripts/ 目录

# 写 CSV 表头
echo "mode,variant,k,N,kernel_ms,total_ms,out_pts" > "$OUT"

# 去掉 ANSI 颜色 + 非打印字符
clean() { sed -r 's/\x1B\[[0-9;]*[mK]//g' | tr -cd '\11\12\15\40-\176'; }

parse_bin_line() {      # $1 = 一整行文本
  awk '
    {
      for (i=1;i<=NF;i++) {
        if ($i=="pts," || $i=="pts")      PTS=$(i-1)
        if ($i=="kernel")                KMS=$(i+1)
        if ($i=="total")                 TMS=$(i+1)
      }
      printf "%s,%s,%s", PTS,KMS,TMS
    }' <<< "$1"
}

run() {                  # $1=mode  $2=variant
  local MODE=$1 VAR=$2
  echo "➜ running: $MODE / $VAR"

  # 捕获并清洗输出
  local TXT=$($BIN --mode "$MODE" --variant "$VAR" -k "$K" -n "$N" | clean)

  local PTS KMS TMS
  if [[ "$MODE" == "naive" ]]; then
    PTS=$(awk '/Compacted count:/ {print $3; exit}' <<< "$TXT")
    TMS=$(awk '/Naive GPU total time/ {print $(NF-1); exit}' <<< "$TXT")
    KMS=$TMS                         # Naive 模式 kernel_ms = total_ms
  else
    local LINE
    LINE=$(grep -i "\[$VAR\]" <<< "$TXT" | head -1)
    IFS=',' read -r PTS KMS TMS <<< "$(parse_bin_line "$LINE")"
  fi

  echo "$MODE,$VAR,$K,$N,$KMS,$TMS,$PTS" >> "$OUT"
}

run naive  atomic        # dummy variant 保持列一致
run bin    atomic
run bin    partition
