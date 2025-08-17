#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal Q6 plotting for ONE CSV (e.g., q6_005.csv)
Outputs 2 figures:
  - Plan A breakdown (codes/hist/scan/scatter/compact)
  - Plan B breakdown (codes/count/(scan+reduce)/write)
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


PLAN_A_STACK = ["codes_ms", "hist_ms", "scan_ms", "scatter_ms", "compact_ms"]
PLAN_B_STACK = ["codes_ms", "count_ms", "scan_reduce_ms", "write_ms"]

LABELS_A = {
    "baseline": "baseline",
    "no-gather": "no-gather",
    "no-binning": "no-binning",
    "force-shared": "force-shared\n(shared)",
    "force-warp": "force-warp\n(warp)",
    "force-bitmask": "force-bitmask\n(bitmask)",
}
ORDER_A = ["baseline", "no-gather", "no-binning",
           "force-shared", "force-warp", "force-bitmask"]

LABELS_B = {
    "baseline": "baseline",
    "no-binning": "no-binning",
}
ORDER_B = ["baseline", "no-binning"]


def _to_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)


def load_one(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # 合并 Plan B 的 scan + reduce
    if "reduce_ms" in df.columns:
        df["scan_reduce_ms"] = df.get("scan_ms", 0.0) + df.get("reduce_ms", 0.0)
    else:
        df["scan_reduce_ms"] = df.get("scan_ms", 0.0)

    num_cols = ["hit_rate", "kernel_ms", "e2e_ms", "total"] + \
               ["codes_ms","hist_ms","scan_ms","scatter_ms","compact_ms",
                "count_ms","reduce_ms","write_ms","scan_reduce_ms"]
    _to_numeric(df, num_cols)

    # 只保留 Plan A / Plan B
    dfA = df[df["plan"].str.upper() == "A"].copy()
    dfB = df[df["plan"].str.upper() == "B"].copy()

    # 命名规整
    dfA["ablation"] = dfA["ablation"].map(lambda x: str(x))
    dfB["ablation"] = dfB["ablation"].map(lambda x: str(x))

    return dfA, dfB


def plot_planA_breakdown(dfA: pd.DataFrame, out_path: Path):
    if dfA.empty:
        print("[warn] No Plan A rows in CSV; skip Plan A plot.")
        return
    # 按预设顺序、只取有的
    variants = [v for v in ORDER_A if v in dfA["ablation"].unique()]
    dfA = dfA.set_index("ablation").loc[variants].reset_index()

    # 取唯一命中率用于标题
    hr = float(dfA["hit_rate"].iloc[0]) if "hit_rate" in dfA else None

    labels = [LABELS_A.get(v, v) for v in variants]
    x = np.arange(len(variants))
    width = 0.75

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    bottom = np.zeros(len(variants), dtype=float)

    # 为了图例稳定，按固定顺序堆叠
    colors = {
        "codes_ms":    "#4e79a7",
        "hist_ms":     "#f28e2b",
        "scan_ms":     "#59a14f",
        "scatter_ms":  "#e15759",
        "compact_ms":  "#b6992d",
    }

    for k in PLAN_A_STACK:
        vals = dfA.get(k, pd.Series([0.0]*len(variants))).to_numpy()
        ax.bar(x, vals, width, bottom=bottom, label=k, color=colors.get(k, None), edgecolor="none")
        bottom += vals

    ax.set_xticks(x, labels, rotation=0)
    ax.set_ylabel("time (ms)")
    title = f"Plan A breakdown @ hit={hr:.2f}" if hr is not None else "Plan A breakdown"
    ax.set_title(title)
    ax.legend(ncols=5, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


def plot_planB_breakdown(dfB: pd.DataFrame, out_path: Path):
    if dfB.empty:
        print("[warn] No Plan B rows in CSV; skip Plan B plot.")
        return
    variants = [v for v in ORDER_B if v in dfB["ablation"].unique()]
    dfB = dfB.set_index("ablation").loc[variants].reset_index()

    hr = float(dfB["hit_rate"].iloc[0]) if "hit_rate" in dfB else None

    labels = [LABELS_B.get(v, v) for v in variants]
    x = np.arange(len(variants))
    width = 0.55

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=150)
    bottom = np.zeros(len(variants), dtype=float)

    colors = {
        "codes_ms":       "#4e79a7",
        "count_ms":       "#f28e2b",
        "scan_reduce_ms": "#59a14f",
        "write_ms":       "#e15759",
    }

    for k in PLAN_B_STACK:
        vals = dfB.get(k, pd.Series([0.0]*len(variants))).to_numpy()
        ax.bar(x, vals, width, bottom=bottom, label=k, color=colors.get(k, None), edgecolor="none")
        bottom += vals

    ax.set_xticks(x, labels, rotation=0)
    ax.set_ylabel("time (ms)")
    title = f"Plan B breakdown @ hit={hr:.2f}" if hr is not None else "Plan B breakdown"
    ax.set_title(title)
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot minimal Q6 figures for one CSV")
    ap.add_argument("--csv", required=True, help="path to one Q6 CSV (e.g., csv/q6_005.csv)")
    ap.add_argument("--outdir", default="figures", help="output directory for PNGs")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfA, dfB = load_one(csv_path)

    # 命名里带上命中率值，稳妥起见再检查一次
    hr = None
    if not dfA.empty:
        hr = float(dfA["hit_rate"].iloc[0])
    elif not dfB.empty:
        hr = float(dfB["hit_rate"].iloc[0])
    suffix = f"hit{int(round((hr or 0.0) * 100)):03d}"

    plot_planA_breakdown(dfA, out_dir / f"q6_planA_breakdown_{suffix}.png")
    plot_planB_breakdown(dfB, out_dir / f"q6_planB_breakdown_{suffix}.png")


if __name__ == "__main__":
    main()
