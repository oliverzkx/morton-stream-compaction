#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q2 plotting: End-to-End vs Kernel-only throughput, and speedup vs a baseline.

Outputs (file names use --out-prefix):
  - <prefix>throughput.png / .svg                   (3 列：按 hit_rate，bar: kernel vs e2e；默认对 N 取均值)
  - <prefix>throughput_facetN.png / .svg            (可选：按 N 分面，不做均值)
  - <prefix>speedup_<mode>.png / .svg               (mode ∈ {kernel, e2e, both})
  - <prefix>speedup_<mode>_facetN.png / .svg        (可选：按 N 分面)

Speedup 计算：
  speedup_kernel = T_base_kernel / T_impl_kernel
  speedup_e2e    = T_base_e2e    / T_impl_e2e
Throughput：
  throughput = N / (ms/1000) / 1e6 = N / ms * 1e-3
"""

import argparse, os
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PLAN_ORDER = [
    "Naive-atomic",
    "Baseline-thrust",
    "PlanA-shared", "PlanA-warp", "PlanA-bitmask",
    "PlanB-atomic",
]
REQUIRED_COLS = ["plan","variant","kBits","N","hit_rate","kernel_ms","e2e_ms"]

def parse_args():
    ap = argparse.ArgumentParser(description="Plot Q2 figures.")
    ap.add_argument("--csv", required=True, help="csv/q2_breakdown.csv")
    ap.add_argument("--out-prefix", default="figures/q2_", help="output path prefix")
    ap.add_argument("--baseline-label", default="Baseline-thrust",
                    help="baseline label = plan-variant (e.g., Baseline-thrust or Naive-atomic)")
    ap.add_argument("--figure", choices=["throughput","speedup","both"], default="both",
                    help="which figures to produce")
    ap.add_argument("--speedup-mode", choices=["kernel","e2e","both"], default="both",
                    help="speedup type to draw")
    ap.add_argument("--facetN", action="store_true",
                    help="facet by N (no averaging over N)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--title-throughput", default="Figure Y — Throughput by Hit Rate")
    ap.add_argument("--title-speedup", default="Figure Z — Speedup vs Baseline by Hit Rate")
    return ap.parse_args()

def load_and_clean(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    lines = raw.splitlines()
    cleaned = []
    for i, ln in enumerate(lines):
        if i == 0 or not ln.lower().startswith("plan,variant"):
            cleaned.append(ln)
    df = pd.read_csv(StringIO("\n".join(cleaned)))

    for c in REQUIRED_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    for c in ["kBits","N","hit_rate","kernel_ms","e2e_ms"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["label"] = df["plan"].astype(str) + "-" + df["variant"].astype(str)
    df = df.dropna(subset=["N","hit_rate","kernel_ms","e2e_ms"])
    # 排序类别，缺失的也保留顺序位置
    df["label"] = pd.Categorical(df["label"], categories=PLAN_ORDER, ordered=True)
    return df

def compute_throughput_M(N_vals, ms_vals):
    ms = np.asarray(ms_vals, dtype=float)
    N  = np.asarray(N_vals,  dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        thr = np.where(ms > 0, N / ms * 1e-3, 0.0)
    return thr

def style(dpi):
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "xtick.labelsize": 9,  "ytick.labelsize": 10, "legend.fontsize": 9,
        "figure.dpi": dpi,
    })

def aggregate_over_N(df: pd.DataFrame) -> pd.DataFrame:
    # 对同 (hit_rate, label) 把 N、kernel_ms、e2e_ms 取均值
    g = (df.groupby(["hit_rate","label"], as_index=False)[["N","kernel_ms","e2e_ms"]]
           .mean())
    g["label"] = pd.Categorical(g["label"], categories=PLAN_ORDER, ordered=True)
    g = g.sort_values(["hit_rate","label"])
    return g

def facet_by_N(df: pd.DataFrame) -> pd.DataFrame:
    t = df.copy()
    t["label"] = pd.Categorical(t["label"], categories=PLAN_ORDER, ordered=True)
    t = t.sort_values(["N","hit_rate","label"])
    return t

# -------------------- THROUGHOUT PLOTS --------------------

def plot_throughput_mean(df_mean, out_prefix, title, dpi):
    style(dpi)
    hits = sorted(df_mean["hit_rate"].unique())
    fig, axes = plt.subplots(1, len(hits), figsize=(14, 4.8), sharey=True)
    if len(hits) == 1:
        axes = [axes]
    fig.suptitle(title + " (averaged over N)", y=1.02, fontsize=13)

    # 统一 y 轴
    ymax = 0.0
    for h in hits:
        sub = df_mean[df_mean["hit_rate"] == h].set_index("label").reindex(PLAN_ORDER)
        thr_k = compute_throughput_M(sub["N"], sub["kernel_ms"])
        thr_e = compute_throughput_M(sub["N"], sub["e2e_ms"])
        ymax = max(ymax, np.nanmax([thr_k, thr_e]))
    ymax = float(np.ceil((ymax * 1.10) / 5.0) * 5.0) if ymax > 0 else 1.0

    for ax, h in zip(axes, hits):
        sub = df_mean[df_mean["hit_rate"] == h].set_index("label").reindex(PLAN_ORDER)
        x = np.arange(len(sub.index)); width = 0.38
        thr_k = compute_throughput_M(sub["N"], sub["kernel_ms"])
        thr_e = compute_throughput_M(sub["N"], sub["e2e_ms"])
        ax.bar(x - width/2, thr_k, width=width, label="Kernel-only", alpha=0.95)
        ax.bar(x + width/2, thr_e, width=width, label="End-to-End", alpha=0.85)
        ax.set_title(f"hit_rate = {h}")
        ax.set_xticks(x); ax.set_xticklabels(sub.index, rotation=18, ha="right")
        ax.set_ylabel("Throughput (M elems/s)")
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.03))
    plt.tight_layout(rect=[0,0.06,1,0.97])

    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    fig.savefig(out_prefix + "throughput.png", bbox_inches="tight")
    fig.savefig(out_prefix + "throughput.svg", bbox_inches="tight")
    print(f"[OK] Saved: {out_prefix}throughput.(png|svg)")

def plot_throughput_facetN(df_facet, out_prefix, title, dpi):
    style(dpi)
    hits = sorted(df_facet["hit_rate"].unique())
    Ns = sorted(df_facet["N"].unique())
    fig, axes = plt.subplots(len(Ns), len(hits),
                             figsize=(4.9*len(hits), 3.8*len(Ns)), sharey=True)
    axes = np.atleast_2d(axes)
    fig.suptitle(title + " (facet by N)", y=1.02, fontsize=13)

    # 统一 y 轴
    ymax = 0.0
    for N in Ns:
        for h in hits:
            sub = df_facet[(df_facet["N"] == N) & (df_facet["hit_rate"] == h)].set_index("label").reindex(PLAN_ORDER)
            thr_k = compute_throughput_M(sub["N"], sub["kernel_ms"])
            thr_e = compute_throughput_M(sub["N"], sub["e2e_ms"])
            ymax = max(ymax, np.nanmax([thr_k, thr_e]))
    ymax = float(np.ceil((ymax * 1.10) / 5.0) * 5.0) if ymax > 0 else 1.0

    for i, N in enumerate(Ns):
        for j, h in enumerate(hits):
            ax = axes[i, j]
            sub = df_facet[(df_facet["N"] == N) & (df_facet["hit_rate"] == h)].set_index("label").reindex(PLAN_ORDER)
            x = np.arange(len(sub.index)); width = 0.38
            thr_k = compute_throughput_M(sub["N"], sub["kernel_ms"])
            thr_e = compute_throughput_M(sub["N"], sub["e2e_ms"])
            ax.bar(x - width/2, thr_k, width=width, label="Kernel-only", alpha=0.95)
            ax.bar(x + width/2, thr_e, width=width, label="End-to-End",  alpha=0.85)
            ax.set_title(f"N = {int(N):,}, hit = {h}")
            ax.set_xticks(x); ax.set_xticklabels(sub.index, rotation=18, ha="right")
            ax.set_ylabel("Throughput (M elems/s)")
            ax.set_ylim(0, ymax)
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.03))
    plt.tight_layout(rect=[0,0.06,1,0.97])

    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    fig.savefig(out_prefix + "throughput_facetN.png", bbox_inches="tight")
    fig.savefig(out_prefix + "throughput_facetN.svg", bbox_inches="tight")
    print(f"[OK] Saved: {out_prefix}throughput_facetN.(png|svg)")

# -------------------- SPEEDUP PLOTS --------------------

def _merge_with_baseline(df: pd.DataFrame, baseline_label: str, use_mean: bool, facetN: bool):
    base = df[df["label"] == baseline_label]
    if base.empty:
        raise ValueError(f"Baseline '{baseline_label}' not found in CSV.")

    if use_mean:
        dfm   = aggregate_over_N(df)
        basem = dfm[dfm["label"] == baseline_label]
        keys  = ["hit_rate"]
        merged = pd.merge(dfm, basem[keys + ["kernel_ms","e2e_ms"]],
                          on=keys, suffixes=("", "_base"))
        return merged
    else:
        # 不做均值时，要求 (hit_rate, N) 对齐
        keys  = ["hit_rate","N"] if facetN else ["hit_rate","N"]
        merged = pd.merge(df, base[keys + ["kernel_ms","e2e_ms"]],
                          on=keys, suffixes=("", "_base"))
        return merged

def plot_speedup(df, baseline_label, out_prefix, title, dpi, facetN=False, mode="both"):
    """
    mode: 'kernel' | 'e2e' | 'both'
    """
    style(dpi)

    if facetN:
        merged = _merge_with_baseline(df, baseline_label, use_mean=False, facetN=True)
        merged["speedup_kernel"] = merged["kernel_ms_base"] / merged["kernel_ms"]
        merged["speedup_e2e"]    = merged["e2e_ms_base"]    / merged["e2e_ms"]
        merged["label"] = pd.Categorical(merged["label"], categories=PLAN_ORDER, ordered=True)
        hits = sorted(merged["hit_rate"].unique())
        Ns   = sorted(merged["N"].unique())

        def _draw_one(ax, sub, title_text, ymax_pad=1.10):
            sub = sub.set_index("label").reindex(PLAN_ORDER)
            x = np.arange(len(sub.index)); width = 0.38
            yk = sub["speedup_kernel"].values
            ye = sub["speedup_e2e"].values
            ymax = np.nanmax(yk if mode=="kernel" else (ye if mode=="e2e" else np.maximum(yk,ye)))
            ymax = float(np.ceil(ymax * ymax_pad)) if ymax > 0 else 2.0

            if mode in ("kernel","both"):
                ax.bar(x - (width/2 if mode=="both" else 0.0), yk, width=width if mode=="both" else 0.7, label="Kernel-only", alpha=0.95)
            if mode in ("e2e","both"):
                ax.bar(x + (width/2 if mode=="both" else 0.0), ye, width=width if mode=="both" else 0.7, label="End-to-End", alpha=0.85)

            ax.set_title(title_text)
            ax.set_xticks(x); ax.set_xticklabels(sub.index, rotation=18, ha="right")
            ax.set_ylabel("Speedup vs baseline (×)")
            ax.set_ylim(0, ymax)
            ax.axhline(1.0, color="k", linewidth=0.8, linestyle="--", alpha=0.7)
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

        fig, axes = plt.subplots(len(Ns), len(hits), figsize=(4.9*len(hits), 3.8*len(Ns)), sharey=True)
        axes = np.atleast_2d(axes)
        fig.suptitle(title + f" (baseline = {baseline_label}, facet by N, mode={mode})", y=1.02, fontsize=13)

        for i, N in enumerate(Ns):
            for j, h in enumerate(hits):
                ax = axes[i, j]
                sub = merged[(merged["N"] == N) & (merged["hit_rate"] == h)]
                _draw_one(ax, sub, f"N = {int(N):,}, hit = {h}")

        handles, labels = axes[0,0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.03))
        plt.tight_layout(rect=[0,0.06,1,0.97])

        os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
        fig.savefig(out_prefix + f"speedup_{mode}_facetN.png", bbox_inches="tight")
        fig.savefig(out_prefix + f"speedup_{mode}_facetN.svg", bbox_inches="tight")
        print(f"[OK] Saved: {out_prefix}speedup_{mode}_facetN.(png|svg)")

    else:
        merged = _merge_with_baseline(df, baseline_label, use_mean=True, facetN=False)
        merged["speedup_kernel"] = merged["kernel_ms_base"] / merged["kernel_ms"]
        merged["speedup_e2e"]    = merged["e2e_ms_base"]    / merged["e2e_ms"]
        merged["label"] = pd.Categorical(merged["label"], categories=PLAN_ORDER, ordered=True)
        hits = sorted(merged["hit_rate"].unique())

        fig, axes = plt.subplots(1, len(hits), figsize=(14, 4.8), sharey=True)
        if len(hits) == 1:
            axes = [axes]
        fig.suptitle(title + f" (baseline = {baseline_label}, averaged over N, mode={mode})", y=1.02, fontsize=13)

        # 统一 y 轴
        ymax = 0.0
        for h in hits:
            sub = merged[merged["hit_rate"] == h]
            smax = np.nanmax(sub["speedup_kernel"] if mode=="kernel"
                             else (sub["speedup_e2e"] if mode=="e2e"
                                   else np.maximum(sub["speedup_kernel"], sub["speedup_e2e"])))
            ymax = max(ymax, smax)
        ymax = float(np.ceil(ymax * 1.10)) if ymax > 0 else 2.0

        for ax, h in zip(axes, hits):
            sub = merged[merged["hit_rate"] == h].set_index("label").reindex(PLAN_ORDER)
            x = np.arange(len(sub.index)); width = 0.38
            if mode in ("kernel","both"):
                ax.bar(x - (width/2 if mode=="both" else 0.0), sub["speedup_kernel"].values,
                       width=width if mode=="both" else 0.7, label="Kernel-only", alpha=0.95)
            if mode in ("e2e","both"):
                ax.bar(x + (width/2 if mode=="both" else 0.0), sub["speedup_e2e"].values,
                       width=width if mode=="both" else 0.7, label="End-to-End",  alpha=0.85)

            ax.set_title(f"hit_rate = {h}")
            ax.set_xticks(x); ax.set_xticklabels(sub.index, rotation=18, ha="right")
            ax.set_ylabel("Speedup vs baseline (×)")
            ax.set_ylim(0, ymax)
            ax.axhline(1.0, color="k", linewidth=0.8, linestyle="--", alpha=0.7)
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.03))
        plt.tight_layout(rect=[0,0.06,1,0.97])

        os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
        fig.savefig(out_prefix + f"speedup_{mode}.png", bbox_inches="tight")
        fig.savefig(out_prefix + f"speedup_{mode}.svg", bbox_inches="tight")
        print(f"[OK] Saved: {out_prefix}speedup_{mode}.(png|svg)")

# -------------------- MAIN --------------------

def main():
    args = parse_args()
    df = load_and_clean(args.csv)

    # Throughput
    if args.figure in ("throughput","both"):
        if args.facetN:
            plot_throughput_facetN(facet_by_N(df), args.out-prefix if hasattr(args, "out-prefix") else args.out_prefix,
                                   args.title_throughput, args.dpi)  # 防御：某些终端传参异常
        else:
            plot_throughput_mean(aggregate_over_N(df), args.out_prefix, args.title_throughput, args.dpi)

    # Speedup
    if args.figure in ("speedup","both"):
        modes = [args.speedup_mode] if args.speedup_mode != "both" else ["kernel","e2e"]
        for m in modes:
            plot_speedup(df, args.baseline_label, args.out_prefix, args.title_speedup, args.dpi,
                         facetN=args.facetN, mode=m)

if __name__ == "__main__":
    main()
