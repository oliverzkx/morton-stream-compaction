#!/usr/bin/env python3
# scripts/plot_q4_overview.py
# Stitch three existing Q4 figures into one overview image (no subplots).

import os
from PIL import Image

IN_DIR = "figures"
OUT = os.path.join(IN_DIR, "q4_overview.png")

# 输入文件名（与你刚生成的文件名一致）
IMG_MAX = os.path.join(IN_DIR, "q4_imbalance_max_over_mean.png")
IMG_STD = os.path.join(IN_DIR, "q4_imbalance_std_over_mean.png")
IMG_KER = os.path.join(IN_DIR, "q4_kernel_ms.png")

def load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    return Image.open(path).convert("RGBA")

def main():
    im_max = load(IMG_MAX)
    im_std = load(IMG_STD)
    im_ker = load(IMG_KER)

    # 统一上排高度
    target_h = max(im_max.height, im_std.height)
    def resize_to_height(im, h):
        w = int(im.width * (h / im.height))
        return im.resize((w, h), Image.LANCZOS)

    im_max_r = resize_to_height(im_max, target_h)
    im_std_r = resize_to_height(im_std, target_h)

    # 上排并排
    top_w = im_max_r.width + im_std_r.width
    top_h = target_h
    top = Image.new("RGBA", (top_w, top_h), (255, 255, 255, 255))
    top.paste(im_max_r, (0, 0))
    top.paste(im_std_r, (im_max_r.width, 0))

    # 让下排（kernel_ms）宽度等于上排宽度
    ker_scale = top_w / im_ker.width
    ker_new = im_ker.resize((top_w, int(im_ker.height * ker_scale)), Image.LANCZOS)

    # 竖向拼接：留一点间距
    gap = 20
    out = Image.new("RGBA", (top_w, top_h + gap + ker_new.height), (255, 255, 255, 255))
    out.paste(top, (0, 0))
    out.paste(ker_new, (0, top_h + gap))

    # 存成 PNG（去 alpha）
    out = out.convert("RGB")
    out.save(OUT, format="PNG", optimize=True)
    print(f"[OK] saved {OUT}")

if __name__ == "__main__":
    main()
