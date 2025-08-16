#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Q2: End-to-End vs. Kernel-Only Performance

Outputs:
  - figures/q2_throughput.png / .svg  (Figure Y)
  - figures/q2_speedup.png   / .svg   (Figure Z, only if baseline found)
  - figures/q2_summary.csv            (grouped averages used for plotting)

Usage examples:
  # 按 hit_rate 分 3 个子图（默认），统一 Y 轴
  python scripts/plot_q2.py --csv csv/q1_breakdown.csv

  # 按 N 分面
  python scripts/plot_q2.py --csv csv/q1_breakdown.csv --facet N

  # 指定 baseline（比如 Naive-baseline），找不到就自动跳过速度图
  python scripts/plot_q2.py --baseline-label Naive-baseline

  # 仅画 throughput，不画 speedup
  python scripts/plot_q2.py --no-speedup

  # 关闭统一 Y 轴
  python scripts/plot_q2.py --no-same-y
"""

import argparse
import os
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PLAN_ORDER = ["PlanA-shared", "PlanA-warp", "PlanA-bitmask", "PlanB-atomic"]
STAGE_FREE_COLS = ["kernel_ms", "e2e_ms", "N", "hit_rate", "plan", "variant"]

def parse_args():
    ap = argparse.ArgumentParser(description="Q2: End-to-End vs. Kernel-Only Performance")
    ap.add_argument("--csv", type=str, default="csv/q1_breakdown.csv",
                    help="Input CSV path")
    ap.add_argument("--facet", type=str, choices=["hit_rate", "N", "all"], default="hit_rate",
                    help="Facet the plots by 'hit_rate' (default), by 'N', or aggregate across all ('all').")
    ap.add_argument("--out-dir", type=str, default="figures",
                    help="Output directory for figures/summary (default: figures)")
    ap.add_argument("--png-name", type=str, default="q2_throughput.png",
                    help="Output PNG filename for throughput plot")
    ap.add_argument("--svg-name", type=str, default="q2_throughput.svg",
                    help="Output SVG filename for throughput plot")
    ap.add_argument("--png-speedup", type=str, default="q2_speedup.png",
                    help="Output PNG filename for speedup plot")
    ap.add_argument("--svg-speedup", type=str, default="q2_speedup.svg",
                    help="Output SVG filename for speedup plot")
    ap.add_argument("--summary-csv", type=str, default="q2_summary.csv",
                    help="CSV filename to save grouped averages")
    ap.add_argument("--baseline-label", type=str, default="Naive-baseline",
                    help="Label of baseline (e.g., 'Naive-baseline'); if not present, speedup plot is skipped.")
    ap.add_argument("--no-speedup", action="store_true",
                    help="Do not produce speedup plots")
    ap.add_argument("--dpi", type=int, default=300, help="DPI for PNG")
    ap.add_argument("--title", type=str, default="Figure Y — Kernel-only vs. End-to-End Throughput",
                    help="Figure Y title")
    ap.add_argument("--title-speedup", type=str, default="Figure Z — Speedup over Baseline",
                    help="Figure Z title")
    ap.add_argument("--no-same-y", action="store_true",
                    help="Disable unified y-axis for panels")
    return ap.parse_args()

def load_and_clean(csv_path: str) -> pd.DataFrame:
    with open(csv_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    lines = raw.splitlines()
    cleaned = []
    for i, ln in enumerate(lines):
        if i == 0 or not ln.lower().startswith("plan,variant"):
            cleaned.append(ln)
    df = pd.read_csv(StringIO("\n".join(cleaned)))

    # ensure numeric types
    for c in ["N","hit_rate","kernel_ms","e2e_ms"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Label = "Plan-variant"
    df["label"] = df["plan"].astype(str) + "-" + df["variant"].astype(str)

    # Throughput (M elems/s)
    # kernel_ms / e2e_ms 可能为 0/NaN，安全处理
    df["thr_kernel"] = np.divide(df["N"], df["kernel_ms"], out=np.zeros_like(df["N"], dtype=float), where=df["kernel_ms"]>0) / 1000.0
    df["thr_e2e"]    = np.divide(df["N"], df["e2e_ms"],    out=np.zeros_like(df["N"], dtype=float), where=df["e2e_ms"]>0)    / 1000.0

    return df

def group_for_plot(df: pd.DataFrame, facet: str) -> dict:
    """
    Returns dict: facet_value -> grouped dataframe with mean thr_kernel/thr_e2e per label.
    If facet == "all": one key 'ALL'.
    """
    out = {}
    if facet == "hit_rate":
        keys = sorted(df["hit_rate"].dropna().unique())
        for k in keys:
            sub = df[df["hit_rate"] == k]
            g = (sub.groupby("label")[["thr_kernel","thr_e2e"]]
                     .mean()
                     .reindex(PLAN_ORDER)
                     .dropna(how="all"))
            out[k] = g
    elif facet == "N":
        keys = sorted(df["N"].dropna().unique())
        for k in keys:
            sub = df[df["N"] == k]
            g = (sub.groupby("label")[["thr_kernel","thr_e2e"]]
                     .mean()
                     .reindex(PLAN_ORDER)
                     .dropna(how="all"))
            out[int(k)] = g
    else:  # "all"
        g = (df.groupby("label")[["thr_kernel","thr_e2e"]]
               .mean()
               .reindex(PLAN_ORDER)
               .dropna(how="all"))
        out["ALL"] = g
    return out

def compute_global_ylim(panels: dict) -> float:
    ymax = 0.0
    for g in panels.values():
        if g.empty:
            continue
        m = float(np.nanmax(g[["thr_kernel","thr_e2e"]].values))
        ymax = max(ymax, m)
    return ymax * 1.10 if ymax > 0 else 1.0

def plot_throughput(panels: dict, same_y: bool, out_dir: str, png_name: str, svg_name: str,
                    title: str, dpi: int):
    os.makedirs(out_dir, exist_ok=True)

    # subplot layout
    n = len(panels)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    sharey = same_y
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4.2*rows), sharey=sharey)
    axes = np.array(axes).reshape(rows, cols)

    if same_y:
        ymax = compute_global_ylim(panels)
    else:
        ymax = None

    fig.suptitle(title, y=1.02, fontsize=13)
    all_handles, all_labels = None, None

    for ax, (facet_val, g) in zip(axes.flatten(), sorted(panels.items(), key=lambda kv: str(kv[0]))):
        if g.empty:
            ax.set_visible(False)
            continue

        x = np.arange(len(g.index))
        width = 0.38
        bars1 = ax.bar(x - width/2, g["thr_kernel"].values, width=width, label="kernel-only")
        bars2 = ax.bar(x + width/2, g["thr_e2e"].values,    width=width, label="end-to-end")

        ax.set_xticks(x)
        ax.set_xticklabels(g.index, rotation=15, ha="right")
        ax.set_ylabel("Throughput (M elems/s)")
        ax.set_title(f"{facet_val}")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
        if ymax is not None:
            ax.set_ylim(0, ymax)

        # collect legend once
        handles, labels = ax.get_legend_handles_labels()
        if all_handles is None:
            all_handles, all_labels = handles, labels

    # hide unused axes
    for ax in axes.flatten()[len(panels):]:
        ax.set_visible(False)

    if all_handles:
        fig.legend(all_handles, all_labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, png_name), dpi=dpi, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, svg_name), bbox_inches="tight")
    print(f"[OK] Throughput saved → {os.path.join(out_dir, png_name)} / {svg_name}")

def build_speedup_panels(panels: dict, baseline_label: str) -> dict:
    """
    Compute speedup vs baseline label, per panel. Returns dict facet_val -> df with columns:
    sp_kernel, sp_e2e.
    If baseline not present in a panel, that panel is omitted.
    """
    out = {}
    for facet_val, g in panels.items():
        if baseline_label not in g.index:
            # skip this panel if no baseline row
            continue
        base_k = g.loc[baseline_label, "thr_kernel"]
        base_e = g.loc[baseline_label, "thr_e2e"]
        # avoid div-by-zero
        base_k = base_k if base_k > 0 else np.nan
        base_e = base_e if base_e > 0 else np.nan

        sp = pd.DataFrame(index=g.index)
        sp["sp_kernel"] = g["thr_kernel"] / base_k
        sp["sp_e2e"]    = g["thr_e2e"]    / base_e
        out[facet_val] = sp
    return out

def compute_global_ylim_speedup(panels: dict) -> float:
    ymax = 0.0
    for g in panels.values():
        if g.empty:
            continue
        m = float(np.nanmax(g[["sp_kernel","sp_e2e"]].values))
        ymax = max(ymax, m)
    return ymax * 1.15 if ymax > 0 else 2.0

def plot_speedup(panels: dict, same_y: bool, out_dir: str, png_name: str, svg_name: str,
                 title: str, dpi: int):
    if not panels:
        print("[WARN] No panels for speedup (baseline missing in all facets). Skipping Figure Z.")
        return

    os.makedirs(out_dir, exist_ok=True)

    n = len(panels)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    sharey = same_y
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4.2*rows), sharey=sharey)
    axes = np.array(axes).reshape(rows, cols)

    if same_y:
        ymax = compute_global_ylim_speedup(panels)
    else:
        ymax = None

    fig.suptitle(title, y=1.02, fontsize=13)
    all_handles, all_labels = None, None

    for ax, (facet_val, g) in zip(axes.flatten(), sorted(panels.items(), key=lambda kv: str(kv[0]))):
        if g.empty:
            ax.set_visible(False)
            continue
        x = np.arange(len(g.index))
        width = 0.38
        bars1 = ax.bar(x - width/2, g["sp_kernel"].values, width=width, label="kernel-only speedup")
        bars2 = ax.bar(x + width/2, g["sp_e2e"].values,    width=width, label="E2E speedup")

        ax.set_xticks(x)
        ax.set_xticklabels(g.index, rotation=15, ha="right")
        ax.set_ylabel("Speedup (×)")
        ax.set_title(f"{facet_val}")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
        if ymax is not None:
            ax.set_ylim(0, ymax)

        handles, labels = ax.get_legend_handles_labels()
        if all_handles is None:
            all_handles, all_labels = handles, labels

    for ax in axes.flatten()[len(panels):]:
        ax.set_visible(False)

    if all_handles:
        fig.legend(all_handles, all_labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, png_name), dpi=dpi, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, svg_name), bbox_inches="tight")
    print(f"[OK] Speedup saved → {os.path.join(out_dir, png_name)} / {svg_name}")

def main():
    args = parse_args()
    df = load_and_clean(args.csv)

    # Facet panels
    panels = group_for_plot(df, facet=args.facet)

    # 导出汇总 CSV（方便表格/复核）
    summary_rows = []
    for facet_val, g in panels.items():
        for lbl, row in g.iterrows():
            summary_rows.append({"facet": facet_val, "label": lbl,
                                 "thr_kernel": row["thr_kernel"], "thr_e2e": row["thr_e2e"]})
    summary = pd.DataFrame(summary_rows)
    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, args.summary_csv)
    summary.to_csv(summary_path, index=False)
    print(f"[OK] Summary saved → {summary_path}")

    # 画 Figure Y: throughput
    plot_throughput(
        panels=panels,
        same_y=(not args.no_same_y),
        out_dir=args.out_dir,
        png_name=args.png_name,
        svg_name=args.svg_name,
        title=args.title,
        dpi=args.dpi,
    )

    # 画 Figure Z: speedup（需要 baseline）
    if not args.no_speedup:
        sp_panels = build_speedup_panels(panels, baseline_label=args.baseline_label)
        plot_speedup(
            panels=sp_panels,
            same_y=(not args.no_same_y),
            out_dir=args.out_dir,
            png_name=args.png_speedup,
            svg_name=args.svg_speedup,
            title=args.title_speedup,
            dpi=args.dpi,
        )

if __name__ == "__main__":
    main()