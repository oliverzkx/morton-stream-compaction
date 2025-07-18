#!/usr/bin/env python3
"""
Publication-quality bar chart of kernel execution time.
Input : scripts/locality_data.csv
Output: images/fig_kernel_time.pdf  (vector)
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT  = Path(__file__).resolve().parent.parent
CSV   = ROOT / "scripts" / "locality_data.csv"
OUT   = ROOT / "images" / "fig_kernel_time.pdf"
OUT.parent.mkdir(exist_ok=True)

# ── 1. 读取数据 ──────────────────────────────────
df = pd.read_csv(CSV)
df = df.query("mode != 'mode'")
df["label"] = df["mode"] + "-" + df["variant"]
df = df.sort_values("kernel_ms", ascending=False)

# ── 2. Matplotlib 全局样式 ───────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.edgecolor": "0.15",
    "axes.linewidth": 0.6,
    "axes.grid": True,
    "grid.color": "0.85",
    "grid.linestyle": "--",
    "grid.linewidth": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "xtick.major.size": 3,
    "ytick.major.size": 3
})

# color-blind safe palette (blue / orange / green)
colors = ["#4E79A7", "#E7853D", "#54A24B"]

# ── 3. 绘图 ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.2, 2.0))   # 单栏期刊尺寸
bars = ax.bar(df["label"],
              df["kernel_ms"].astype(float),
              color=colors[:len(df)],
              width=0.55)

ax.set_xlabel("Variant", labelpad=2)
ax.set_ylabel("Kernel time (ms)", labelpad=2)
ax.set_title("Kernel Execution Time  (N = 1,048,576)",
             pad=4, fontsize=9)

# 预留 5 % 头部空隙
ymax = df["kernel_ms"].max() * 1.05
ax.set_ylim(0, ymax)

# 在柱顶标数值
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2,
            h + ymax*0.02,
            f"{h:.2f}",
            ha="center", va="bottom", fontsize=7)

plt.tight_layout(pad=0.4)
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"Saved vector figure → {OUT}")
