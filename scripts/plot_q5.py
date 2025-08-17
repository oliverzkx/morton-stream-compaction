#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q5 plotting script
Usage:
  python3 scripts/plot_q5.py --csv csv/q5_scaling.csv --outdir figures --pdf
Optional:
  --title_suffix "H=0.50, Uniform"
  --variants planB,planA_shared,planA_warp,planA_bitmask
  --xlog    # use log-scale on X (N)
Notes:
  - Uses matplotlib only (no seaborn), one chart per figure, no explicit colors.
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser(description="Plot Q5 scalability results")
    ap.add_argument("--csv", required=True, help="Path to q5_scaling.csv")
    ap.add_argument("--outdir", default="figures", help="Output directory for figures")
    ap.add_argument("--pdf", action="store_true", help="Also save .pdf versions")
    ap.add_argument("--title_suffix", default="", help="Suffix appended to figure titles")
    ap.add_argument("--variants", default="", help="Comma-separated variant filter "
                                                   "(e.g. planB,planA_shared)")
    ap.add_argument("--xlog", action="store_true", help="Use log-scale on X axis (N)")
    return ap.parse_args()


def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def label_variants(df):
    name_map = {
        "planB": "Plan B (atomic)",
        "planA_shared": "Plan A (shared)",
        "planA_warp": "Plan A (warp)",
        "planA_bitmask": "Plan A (bitmask)",
        "naive": "Naive",
        "thrust": "Thrust",
    }
    df["VariantLabel"] = df["variant"].map(name_map).fillna(df["variant"])
    return df


def maybe_filter_variants(df, variants_str: str):
    if not variants_str:
        return df
    keep = [v.strip() for v in variants_str.split(",") if v.strip()]
    return df[df["variant"].isin(keep)].copy()


def plot_lines(df, x_col, y_col, y_label, title, out_png, out_pdf=None, xlog=False):
    plt.figure(figsize=(7.5, 5.0), dpi=140)
    for label, sub in df.groupby("VariantLabel"):
        sub = sub.sort_values(x_col)
        plt.plot(sub[x_col], sub[y_col], marker="o", label=label)
    plt.xlabel("Dataset size N (millions)")
    plt.ylabel(y_label)
    plt.title(title)
    if xlog:
        plt.xscale("log")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    if out_pdf:
        plt.savefig(out_pdf)
    plt.close()


def main():
    args = parse_args()
    ensure_outdir(args.outdir)

    df = pd.read_csv(args.csv)
    required = {"variant", "N", "kernel_Melps", "e2e_Melps", "achieved_BW_GBps"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing columns: {sorted(missing)}")

    df = label_variants(df)
    df = maybe_filter_variants(df, args.variants)
    if df.empty:
        raise SystemExit("No rows after variant filter.")

    # X as millions
    df = df.copy()
    df["N_M"] = df["N"] / 1e6

    suffix = f" — {args.title_suffix}" if args.title_suffix else ""

    # 1) Kernel-only throughput vs N
    plot_lines(
        df, "N_M", "kernel_Melps",
        "Kernel-only throughput (M elems/s)",
        f"Q5: Kernel-only Throughput vs N{suffix}",
        os.path.join(args.outdir, "q5_kernel_throughput_vs_N.png"),
        os.path.join(args.outdir, "q5_kernel_throughput_vs_N.pdf") if args.pdf else None,
        xlog=args.xlog,
    )

    # 2) E2E throughput vs N
    plot_lines(
        df, "N_M", "e2e_Melps",
        "End-to-End throughput (M elems/s)",
        f"Q5: End-to-End Throughput vs N{suffix}",
        os.path.join(args.outdir, "q5_e2e_throughput_vs_N.png"),
        os.path.join(args.outdir, "q5_e2e_throughput_vs_N.pdf") if args.pdf else None,
        xlog=args.xlog,
    )

    # 3) Achieved bandwidth vs N
    plot_lines(
        df, "N_M", "achieved_BW_GBps",
        "Achieved write bandwidth (GB/s, proxy)",
        f"Q5: Achieved Bandwidth vs N (proxy_v1){suffix}",
        os.path.join(args.outdir, "q5_bandwidth_vs_N.png"),
        os.path.join(args.outdir, "q5_bandwidth_vs_N.pdf") if args.pdf else None,
        xlog=args.xlog,
    )

    print("✔ Wrote figures to", os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()
