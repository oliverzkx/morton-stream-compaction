# scripts/plot_q7.py
# Usage:
#   python3 scripts/plot_q7.py \
#     --inputs csv/q7_uniform_8M.csv csv/q7_clustered_8M.csv csv/q7_skewed_8M.csv \
#     --outdir figures
#
# It will create:
#   figures/q7_agg_ms_bar.png
#   figures/q7_e2e_ms_bar.png
#   figures/q7_summary.txt   (numerical table for the paper)

import argparse, os, csv
from collections import defaultdict, OrderedDict
import numpy as np
import matplotlib.pyplot as plt

def read_rows(paths):
    rows = []
    for p in paths:
        with open(p, 'r', newline='') as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append(row)
    return rows

def to_float(x):
    try: return float(x)
    except: return float('nan')

def summarize(rows):
    # group by (dist, impl)
    groups = defaultdict(list)
    for row in rows:
        dist = row['dist']
        impl = row['impl']
        groups[(dist, impl)].append({
            'agg_ms': to_float(row['agg_ms']),
            'e2e_ms': to_float(row['e2e_ms'])
        })
    # compute mean/std
    summary = {}
    for (dist, impl), vals in groups.items():
        agg = np.array([v['agg_ms'] for v in vals], dtype=float)
        e2e = np.array([v['e2e_ms'] for v in vals], dtype=float)
        summary[(dist, impl)] = {
            'agg_mean': float(np.nanmean(agg)),
            'agg_std' : float(np.nanstd(agg, ddof=1)) if len(agg) > 1 else 0.0,
            'e2e_mean': float(np.nanmean(e2e)),
            'e2e_std' : float(np.nanstd(e2e, ddof=1)) if len(e2e) > 1 else 0.0,
            'n'       : len(vals)
        }
    return summary

def write_table(summary, out_txt):
    # fixed order of dists
    dists = ['uniform', 'clustered', 'skewed']
    impls = ['atomic', 'slices']
    with open(out_txt, 'w') as f:
        f.write("dist,impl,n,agg_ms_mean,agg_ms_std,e2e_ms_mean,e2e_ms_std\n")
        for d in dists:
            for im in impls:
                s = summary.get((d, im), None)
                if s is None: continue
                f.write(f"{d},{im},{s['n']},{s['agg_mean']:.3f},{s['agg_std']:.3f},{s['e2e_mean']:.3f},{s['e2e_std']:.3f}\n")

def plot_bar(summary, metric_key_mean, metric_key_std, title, out_png):
    dists = ['uniform', 'clustered', 'skewed']
    impls = ['atomic', 'slices']
    x = np.arange(len(dists))
    width = 0.35

    means_a = [summary.get((d,'atomic'),{}).get(metric_key_mean, np.nan) for d in dists]
    stds_a  = [summary.get((d,'atomic'),{}).get(metric_key_std, 0.0) for d in dists]
    means_s = [summary.get((d,'slices'),{}).get(metric_key_mean, np.nan) for d in dists]
    stds_s  = [summary.get((d,'slices'),{}).get(metric_key_std, 0.0) for d in dists]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - width/2, means_a, width, yerr=stds_a, capsize=3, label='atomic')
    ax.bar(x + width/2, means_s, width, yerr=stds_s, capsize=3, label='slices')

    ax.set_xticks(x)
    ax.set_xticklabels(dists)
    ax.set_ylabel('Milliseconds')
    ax.set_title(title)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+', required=True,
                    help='CSV files: e.g. csv/q7_uniform_8M.csv csv/q7_clustered_8M.csv csv/q7_skewed_8M.csv')
    ap.add_argument('--outdir', default='figures')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rows = read_rows(args.inputs)
    summary = summarize(rows)
    write_table(summary, os.path.join(args.outdir, 'q7_summary.txt'))

    # Plot agg_ms (the fair comparison of aggregation strategy)
    plot_bar(summary, 'agg_mean', 'agg_std',
             'Per-bin aggregation time (agg_ms): atomic vs slices',
             os.path.join(args.outdir, 'q7_agg_ms_bar.png'))

    # Plot e2e_ms (includes offsets/scatter cost for slices)
    plot_bar(summary, 'e2e_mean', 'e2e_std',
             'End-to-end time (includes preprocessing for slices)',
             os.path.join(args.outdir, 'q7_e2e_ms_bar.png'))

    print(f"[OK] Wrote figures to {args.outdir}/ and table to {args.outdir}/q7_summary.txt")

if __name__ == '__main__':
    main()
