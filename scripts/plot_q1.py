#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Figure X: Stacked bar charts of per-stage kernel times for different plans/variants.

Usage examples:
  python scripts/plot_q1.py \
      --csv csv/q1_breakdown.csv \
      --out-png figures/figure_q1.png \
      --out-svg figures/figure_q1.svg \
      --mode absolute

  # Proportional (each bar sums to 1.0)
  python scripts/plot_q1.py --csv csv/q1_breakdown.csv --mode proportion
"""

import argparse
import os
from io import StringIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PLAN_ORDER = ["PlanA-shared", "PlanA-warp", "PlanA-bitmask", "PlanB-atomic"]
STAGE_COLS = [
    "codes_ms",
    "hist_ms",
    "scan_ms",
    "scatter_ms",
    "count_ms",
    "reduce_ms",
    "write_ms",
    "compact_ms",
]

def parse_args():
    ap = argparse.ArgumentParser(description="Plot Figure X for Q1 stacked stage times.")
    ap.add_argument("--csv", type=str, default="csv/q1_breakdown.csv",
                    help="Input CSV path (default: csv/q1_breakdown.csv)")
    ap.add_argument("--out-png", type=str, default="figures/figure_q1.png",
                    help="Output PNG path (default: figures/figure_q1.png)")
    ap.add_argument("--out-svg", type=str, default="figures/figure_q1.svg",
                    help="Optional SVG output path (default: figures/figure_q1.svg)")
    ap.add_argument("--mode", type=str, choices=["absolute", "proportion"],
                    default="absolute",
                    help="Plot absolute ms or proportional share per bar (default: absolute)")
    ap.add_argument("--dpi", type=int, default=300, help="DPI for PNG (default: 300)")
    ap.add_argument("--title", type=str, default="Figure X — Stage Time Breakdown",
                    help="Figure title")
    # 可选：关闭统一数轴
    ap.add_argument("--no-same-y", action="store_true",
                    help="Disable unified y-axis for absolute mode")
    return ap.parse_args()

def load_and_clean(csv_path: str) -> pd.DataFrame:
    # Remove repeated header lines safely
    with open(csv_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    lines = raw.splitlines()
    cleaned_lines = []
    for i, ln in enumerate(lines):
        if i == 0 or not ln.lower().startswith("plan,variant"):
            cleaned_lines.append(ln)
    df = pd.read_csv(StringIO("\n".join(cleaned_lines)))

    # Ensure numeric dtypes
    num_cols = ["kBits","N","hit_rate","kernel_ms","e2e_ms","total"] + STAGE_COLS
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fill stage NaNs with 0 (stage not applicable / not recorded)
    for c in STAGE_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)
    return df

def group_by_hit(df: pd.DataFrame) -> dict:
    """Return dict: hit_rate -> grouped mean dataframe indexed by 'label' with stage cols + kernel_ms"""
    df = df.copy()
    df["label"] = df["plan"] + "-" + df["variant"]
    hits = sorted(x for x in df["hit_rate"].dropna().unique())
    out = {}
    for h in hits:
        sub = df[df["hit_rate"] == h]
        g = (sub.groupby("label")[STAGE_COLS + ["kernel_ms"]]
                  .mean()
                  .reindex(PLAN_ORDER)
                  .dropna(how="all"))
        out[h] = g
    return out

def compute_global_ylim(grouped_by_hit: dict) -> float:
    """Find a unified y-max across all subplots (absolute mode).
       Use the max of stack sum vs kernel_ms to avoid clipping when stages don't sum to kernel_ms."""
    global_max = 0.0
    for g in grouped_by_hit.values():
        if g.empty:
            continue
        stack_sum = g[STAGE_COLS].sum(axis=1).values
        km = g["kernel_ms"].values
        heights = np.maximum(stack_sum, km)
        m = float(np.nanmax(heights)) if heights.size else 0.0
        global_max = max(global_max, m)
    # add a small headroom
    return global_max * 1.10 if global_max > 0 else 1.0

def plot_figure(grouped_by_hit: dict, mode: str, out_png: str, out_svg: str,
                title: str, dpi: int, same_y: bool):
    # --- Aesthetics ---
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": dpi,
    })

    # sharey=True only makes sense when we unify y-range visually
    share_y = (mode == "proportion") or (mode == "absolute" and same_y)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=share_y)
    fig.suptitle(title, y=1.03, fontsize=13)

    # Pre-compute global y-limit for absolute mode (if enabled)
    y_global = None
    if mode == "absolute" and same_y:
        y_global = compute_global_ylim(grouped_by_hit)

    # Draw subplots
    for ax, (hit, g) in zip(axes, sorted(grouped_by_hit.items(), key=lambda kv: kv[0])):
        if g.empty:
            ax.set_visible(False)
            continue

        x = np.arange(len(g.index))
        bottom = np.zeros(len(g.index), dtype=float)

        if mode == "absolute":
            y_label = "Kernel time (ms)"
            for col in STAGE_COLS:
                vals = g[col].values
                ax.bar(x, vals, bottom=bottom, label=col)
                bottom += vals
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
            if y_global is not None:
                ax.set_ylim(0, y_global)
        else:
            y_label = "Share of kernel time"
            denom = g["kernel_ms"].replace(0, np.nan).values
            for col in STAGE_COLS:
                vals = g[col].values
                prop = np.divide(vals, denom, out=np.zeros_like(vals), where=~np.isnan(denom))
                ax.bar(x, prop, bottom=bottom, label=col)
                bottom += prop
            ax.set_ylim(0, 1.02)
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

        ax.set_title(f"hit_rate = {hit}")
        ax.set_ylabel(y_label)
        ax.set_xticks(x)
        ax.set_xticklabels(g.index, rotation=15, ha="right")

    # One shared legend outside
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.06))

    plt.tight_layout()
    # Ensure output dirs exist
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    os.makedirs(os.path.dirname(out_svg), exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", dpi=dpi)
    fig.savefig(out_svg, bbox_inches="tight")
    print(f"[OK] Saved: {out_png}\n[OK] Saved: {out_svg}")

def main():
    args = parse_args()
    df = load_and_clean(args.csv)

    # Sanity: keep only rows that have the stage columns; others will be 0-filled
    for c in STAGE_COLS + ["kernel_ms", "plan", "variant", "hit_rate"]:
        if c not in df.columns:
            raise ValueError(f"CSV missing required column: {c}")

    grouped = group_by_hit(df)
    plot_figure(grouped, args.mode, args.out_png, args.out_svg, args.title, args.dpi,
                same_y=(not args.no_same_y))

if __name__ == "__main__":
    main()