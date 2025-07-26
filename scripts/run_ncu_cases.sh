#!/usr/bin/env bash
###############################################################################
# run_ncu_cases.sh  ——  batch Nsight-Compute profiling for multiple kernels
# 放在项目根目录，`chmod +x run_ncu_cases.sh` 后执行： ./run_ncu_cases.sh
###############################################################################
set -u        # 未定义变量立即报错
# set -e      # 注释掉，遇错继续

# ----------- user config -----------------------------------------------------
NCU=/opt/nvidia/nsight-compute/2023.3.0/ncu   # Nsight Compute CLI
EXE=./build/main                              # 可执行文件
POINTS=1048576                                # 输入点数
KBITS=8                                       # bin k 位
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")            # 时间戳目录
OUTDIR=report/${TIMESTAMP}
mkdir -p "${OUTDIR}"

# ----------- case list:  <name> <args...> ------------------------------------
declare -a CASES=(
  "bin_atomic   --mode bin --variant atomic     -k ${KBITS} -n ${POINTS}"
  "bin_shared   --mode bin --variant partition  --kernel shared   -k ${KBITS} -n ${POINTS}"
  "bin_warp     --mode bin --variant partition  --kernel warp     -k ${KBITS} -n ${POINTS}"
  "bin_bitmask  --mode bin --variant partition  --kernel bitmask  -k ${KBITS} -n ${POINTS}"
)

# ----------- loop ------------------------------------------------------------
for entry in "${CASES[@]}"; do
  caseName=$(echo "${entry}" | awk '{print $1}')
  args=$(echo "${entry}" | cut -d' ' -f2-)

  echo -e "\n🔹 Profiling \033[1m${caseName}\033[0m ..."
  if "${NCU}" --set detailed \
              --replay-mode kernel \
              --target-processes all \
              --export "${OUTDIR}/${caseName}_%p" \
              ${EXE} ${args}; then
      echo "✅  Saved to ${OUTDIR}/${caseName}_<pid>.ncu-rep"
  else
      ec=$?
      echo "❌  ${caseName} failed (exit ${ec})" | tee -a "${OUTDIR}/error.log"
  fi
done

echo -e "\n🎉 All cases finished.  Reports ➜ ${OUTDIR}"
[ -f "${OUTDIR}/error.log" ] && echo "⚠️  Some cases failed – see error.log"

###############################################################################
# 可选：自动打包
# tar -czf "${OUTDIR}.tar.gz" -C report "${TIMESTAMP}"
###############################################################################
