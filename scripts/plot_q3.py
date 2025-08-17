#!/usr/bin/env python3
import csv, os, sys
import math
import matplotlib.pyplot as plt

CSV_DIR = "csv"
FIG_DIR = "figures"

FILES = [
    ("q3_block.csv",  "blockSize",  "Block size (threads)"),
    ("q3_kbits.csv",  "kBits",      "kBits (bins = 2^k)"),
    ("q3_hit.csv",    "hit_rate",   "Hit rate"),
]

def load_csv(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def to_float(s, default=None):
    try:
        return float(s)
    except Exception:
        return default

def to_int(s, default=None):
    try:
        return int(s)
    except Exception:
        return default

def check_counts(rows):
    """Warn if out_count deviates from N * hit_rate by > 0.5%"""
    bad = 0
    for row in rows:
        N = to_int(row.get("N", "0"), 0)
        hit = to_float(row.get("hit_rate", "0"), 0.0)
        out = to_int(row.get("out_count", "0"), 0)
        target = int(round(N * hit))
        if target == 0: 
            continue
        err = abs(out - target) / target
        if err > 0.005:  # >0.5%
            bad += 1
            print(f"[WARN] out_count mismatch: label=({row.get('plan')},{row.get('variant')},{row.get('kernel')}) "
                  f"x={row.get('blockSize') or row.get('kBits') or row.get('hit_rate')} "
                  f"out={out} vs N*hit≈{target} (err={err*100:.2f}%)")
    if bad == 0:
        print("[OK] out_count sanity check passed.")
    else:
        print(f"[WARN] {bad} rows failed out_count sanity check.")

def group_series(rows, xkey, ykey):
    """
    Return mapping: label -> list of (x, y) sorted by x.
    label = 'PlanB-atomic' or 'PlanA-partition-<kernel>'
    x parsed as float/int depending on field; y as float.
    """
    series = {}
    for row in rows:
        plan = row.get("plan", "")
        if plan == "PlanA":
            label = f"{plan}-{row.get('variant','')}-{row.get('kernel','')}"
        else:
            label = "PlanB-atomic"
        # parse x
        if xkey == "blockSize" or xkey == "kBits":
            x = to_int(row.get(xkey), None)
        else:
            x = to_float(row.get(xkey), None)
        y = to_float(row.get(ykey), None)
        if x is None or y is None:
            continue
        series.setdefault(label, []).append((x, y))
    # sort by x
    for k in series:
        series[k] = sorted(series[k], key=lambda p: p[0])
    return series

def plot_xy(series, xlabel, ylabel, title, outpath):
    plt.figure()
    for label, pts in series.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"[OK] saved {outpath}")

def handle_file(csv_name, xkey, xlabel):
    path = os.path.join(CSV_DIR, csv_name)
    rows = load_csv(path)
    if not rows:
        print(f"[SKIP] {path} not found.")
        return
    print(f"[LOAD] {path}: {len(rows)} rows")
    check_counts(rows)
    # kernel-only
    s1 = group_series(rows, xkey, "thrpt_kernel_Meps")
    plot_xy(s1, xlabel, "Throughput (M elems/s)",
            f"Throughput vs {xlabel} (Kernel-only)", 
            os.path.join(FIG_DIR, f"{csv_name.replace('.csv','')}_kernel.png"))
    # end-to-end
    s2 = group_series(rows, xkey, "thrpt_e2e_Meps")
    plot_xy(s2, xlabel, "Throughput (M elems/s)",
            f"Throughput vs {xlabel} (End-to-End)",  
            os.path.join(FIG_DIR, f"{csv_name.replace('.csv','')}_e2e.png"))

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    for name, xkey, xlabel in FILES:
        handle_file(name, xkey, xlabel)

if __name__ == "__main__":
    main()
