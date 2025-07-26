#!/usr/bin/env bash
###############################################################################
# run_ncu_cases.sh — Batch-profile multiple kernel variants with Nsight Compute
#
# Usage from project root:
#   chmod +x run_ncu_cases.sh
#   ./run_ncu_cases.sh
#
# The script loops over several preset command-line variants, invokes
# `ncu` in replay-mode for each, and stores *.ncu-rep files in
#   report/<timestamp>/
###############################################################################

set -u        # abort on use of an undefined variable
# set -e      # uncomment to exit on the first non-zero status

# -------- user-configurable paths & settings ---------------------------------
NCU=/opt/nvidia/nsight-compute/2023.3.0/ncu   # Nsight Compute CLI
EXE=./build/main                              # application to profile
POINTS=1048576                                # total points generated
KBITS=8                                       # low Morton bits per bin
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")            # timestamped report dir
OUTDIR=report/${TIMESTAMP}
mkdir -p "${OUTDIR}"

# -------- case list:  <nickname> <CLI arguments …> ---------------------------
# The first token is used as the file prefix; the remaining tokens are passed
# verbatim to the executable.
declare -a CASES=(
  "bin_atomic   --mode bin --variant atomic     -k ${KBITS} -n ${POINTS}"
  "bin_shared   --mode bin --variant partition  --kernel shared   -k ${KBITS} -n ${POINTS}"
  "bin_warp     --mode bin --variant partition  --kernel warp     -k ${KBITS} -n ${POINTS}"
  "bin_bitmask  --mode bin --variant partition  --kernel bitmask  -k ${KBITS} -n ${POINTS}"
)

# -------- profiling loop -----------------------------------------------------
for entry in "${CASES[@]}"; do
  caseName=$(echo "${entry}" | awk '{print $1}')          # nickname
  args=$(echo "${entry}"   | cut -d' ' -f2-)              # CLI args

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

echo -e "\n🎉  All cases finished.  Reports ➜ ${OUTDIR}"
[ -f "${OUTDIR}/error.log" ] && echo "⚠️  Some cases failed — see error.log"

###############################################################################
# Optional: archive the report directory
# tar -czf "${OUTDIR}.tar.gz" -C report "${TIMESTAMP}"
###############################################################################
