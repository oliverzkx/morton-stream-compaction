#!/usr/bin/env python3
# scripts/plot_q4.py
# Read Q4 CSVs, aggregate (mean/std over repeats), and make grouped bar charts.

import os
import sys
import glob
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------- config ---------------------------
DEFAULT_FILES = [
    "csv/q4_uniform_high.csv",
    "csv/q4_clustered_high.csv",
    "csv/q4_skewed_high.csv",
]
OUT_DIR = "figures"
# 排除没有实际计时的变体
DROP_VARIANTS = {"naive", "thrust"}

# 分布的显示顺序 & 名称
DIST_ORDER = ["uniform", "clustered", "skewed"]
DIST_LABEL = {"uniform": "Uniform", "clustered": "Clustered", "skewed": "Skewed"}

# 变体显示顺序 & 名称（根据你项目习惯可调整）
VARIANT_ORDER = ["planB", "planA_shared", "planA_warp", "planA_bitmask"]
VARIANT_LABEL = {
    "planB": "PlanB",
    "planA_shared": "PlanA-Shared",
    "planA_warp": "PlanA-Warp",
    "planA_bitmask": "PlanA-Bitmask",
}

# --------------------------- utils ---------------------------
def load_csvs(paths):
    rows = []
    for p in paths:
        if not os.path.exists(p):
            print(f"[WARN] file not found: {p}")
            continue
        df = pd.read_csv(p)
        df["__src"] = os.path.basename(p)
        rows.append(df)
    if not rows:
        raise SystemExit("[ERR] no input CSVs found.")
    out = pd.concat(rows, ignore_index=True)
    return out

def tidy(df):
    # 只保留需要的列
    cols_keep = [
        "variant","dist","kBits","N","hit_rate",
        "active_bins","active_bins_ratio","max_over_mean","std_over_mean",
        "kernel_ms","e2e_ms","__src"
    ]
    miss = [c for c in cols_keep if c not in df.columns]
    if miss:
        raise SystemExit(f"[ERR] missing columns in CSV: {miss}")

    df = df[cols_keep].copy()
    # 统一小写 & 过滤
    df["variant"] = df["variant"].astype(str)
    df["dist"] = df["dist"].astype(str)
    df = df[~df["variant"].isin(DROP_VARIANTS)]

    # 只保留我们关心的分布 & 变体
    df = df[df["dist"].isin(DIST_ORDER)]
    df = df[df["variant"].isin(VARIANT_ORDER)]

    # 数字化
    num_cols = ["kBits","N","hit_rate","active_bins","active_bins_ratio",
                "max_over_mean","std_over_mean","kernel_ms","e2e_ms"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def aggregate(df):
    # 对相同 (dist, variant) 的多次实验（不同 seed/文件）求 mean & std
    grp = df.groupby(["dist","variant"], as_index=False).agg(
        kernel_ms_mean=("kernel_ms","mean"),
        kernel_ms_std =("kernel_ms","std"),
        e2e_ms_mean   =("e2e_ms","mean"),
        e2e_ms_std    =("e2e_ms","std"),
        max_over_mean_mean=("max_over_mean","mean"),
        max_over_mean_std =("max_over_mean","std"),
        std_over_mean_mean=("std_over_mean","mean"),
        std_over_mean_std =("std_over_mean","std"),
        active_bins_ratio_mean=("active_bins_ratio","mean"),
        active_bins_ratio_std =("active_bins_ratio","std"),
    )
    # 保证排序
    grp["dist"] = pd.Categorical(grp["dist"], DIST_ORDER)
    grp["variant"] = pd.Categorical(grp["variant"], VARIANT_ORDER)
    grp = grp.sort_values(["dist","variant"]).reset_index(drop=True)
    return grp

def _ensure_outdir():
    os.makedirs(OUT_DIR, exist_ok=True)

def _bar_group(ax, x_labels, series_dict, yerr_dict=None, title="", ylabel=""):
    """
    series_dict: {legend_name: list_of_values_aligned_with_x_labels}
    yerr_dict:   {legend_name: list_of_errs_aligned_with_x_labels} (optional)
    """
    n_groups = len(x_labels)
    legends = list(series_dict.keys())
    n_series = len(legends)
    idx = np.arange(n_groups, dtype=float)

    width = 0.8 / max(n_series,1)
    for i, name in enumerate(legends):
        vals = np.array(series_dict[name], dtype=float)
        offs = (i - (n_series-1)/2) * width
        if yerr_dict and name in yerr_dict and yerr_dict[name] is not None:
            ax.bar(idx + offs, vals, width, label=name, yerr=yerr_dict[name], capsize=3)
        else:
            ax.bar(idx + offs, vals, width, label=name)
    ax.set_xticks(idx)
    ax.set_xticklabels(x_labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

# --------------------------- plotting ---------------------------
def plot_kernel_ms(grp):
    _ensure_outdir()
    # X：dist；每个 dist 内画多个 variant 的 bar
    x_labels = [DIST_LABEL[d] for d in DIST_ORDER]
    series = {}
    yerr = {}
    for v in VARIANT_ORDER:
        sub = grp[grp["variant"]==v]
        series[VARIANT_LABEL[v]] = [sub[sub["dist"]==d]["kernel_ms_mean"].values[0] if not sub[sub["dist"]==d].empty else np.nan
                                    for d in DIST_ORDER]
        yerr[VARIANT_LABEL[v]]   = [sub[sub["dist"]==d]["kernel_ms_std"].values[0] if not sub[sub["dist"]==d].empty else 0.0
                                    for d in DIST_ORDER]

    fig, ax = plt.subplots(figsize=(8,4.5))
    _bar_group(ax, x_labels, series, yerr, title="Kernel Time vs. Spatial Distribution", ylabel="kernel_ms (mean ± std)")
    out = os.path.join(OUT_DIR, "q4_kernel_ms.png")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"[OK] saved {out}")

def plot_imbalance(grp):
    _ensure_outdir()
    # 画两张：max_over_mean 和 std_over_mean（每张以 dist 为 X）
    x_labels = [DIST_LABEL[d] for d in DIST_ORDER]

    # (A) max_over_mean —— 取所有 variant 的均值再平均一下（也可以只用 PlanB）
    # 这里我们以 PlanB 为主（更敏感），展示更直观：
    focus = "planB"
    sub = grp[grp["variant"]==focus]

    fig, ax = plt.subplots(figsize=(6.5,4))
    series = {"PlanB": [sub[sub["dist"]==d]["max_over_mean_mean"].values[0] if not sub[sub["dist"]==d].empty else np.nan
                        for d in DIST_ORDER]}
    yerr   = {"PlanB": [sub[sub["dist"]==d]["max_over_mean_std"].values[0] if not sub[sub["dist"]==d].empty else 0.0
                        for d in DIST_ORDER]}
    _bar_group(ax, x_labels, series, yerr, title="Load Imbalance (max/mean)", ylabel="max_over_mean")
    out = os.path.join(OUT_DIR, "q4_imbalance_max_over_mean.png")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"[OK] saved {out}")

    # (B) std_over_mean（同样用 PlanB）
    fig, ax = plt.subplots(figsize=(6.5,4))
    series = {"PlanB": [sub[sub["dist"]==d]["std_over_mean_mean"].values[0] if not sub[sub["dist"]==d].empty else np.nan
                        for d in DIST_ORDER]}
    yerr   = {"PlanB": [sub[sub["dist"]==d]["std_over_mean_std"].values[0] if not sub[sub["dist"]==d].empty else 0.0
                        for d in DIST_ORDER]}
    _bar_group(ax, x_labels, series, yerr, title="Load Imbalance (std/mean)", ylabel="std_over_mean")
    out = os.path.join(OUT_DIR, "q4_imbalance_std_over_mean.png")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"[OK] saved {out}")

# --------------------------- main ---------------------------
def main():
    # 支持：不传参数=用默认三份 high CSV；也可传多个路径（含通配符），例如：
    # python scripts/plot_q4.py csv/q4_*_high.csv
    args = sys.argv[1:]
    if not args:
        files = DEFAULT_FILES
    else:
        files = []
        for a in args:
            files.extend(glob.glob(a))
    print("[INFO] loading files:", files)

    df = load_csvs(files)
    df = tidy(df)
    grp = aggregate(df)

    # 打印一下聚合表，方便核对
    print("\n[AGGREGATED]")
    print(grp.to_string(index=False))

    plot_kernel_ms(grp)
    plot_imbalance(grp)

if __name__ == "__main__":
    main()
