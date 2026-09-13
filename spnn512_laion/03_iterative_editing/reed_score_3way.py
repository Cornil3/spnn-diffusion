#!/usr/bin/env python3
"""Score 10 shared samples for three codecs -- SD-1.5 VAE, SPNN (both halves),
and the hybrid (SD-1.5 VAE encoder + SPNN decoder) -- against each chain's own x^1."""
import numpy as np, torch
from pathlib import Path
from PIL import Image
import lpips as lpips_lib
from skimage.metrics import structural_similarity as ssim

FULL = Path("/rg/shocher_prj/ron.libman/reed_table1/out/rollouts")
SIDS = ["100081_t1","108871_t2","129587_t3","148240_t2","177572_t1",
        "189893_t1","215135_t2","24629_t1","262161_t1","291614_t2"]
HYB = Path("/rg/shocher_prj/ron.libman/reed_table1/out_vaeenc_ids" + "-".join(SIDS)) / "rollouts"
EDS = ["IP2P","MagicBrush","DiffEdit","PbE","SDInpaint"]
ARMS = [("SD-1.5 VAE          ", FULL, "vanilla"),
        ("SPNN  (enc + dec)   ", FULL, "spnn"),
        ("SD-VAE enc + SPNN dec", HYB, "hybrid")]
ITERS = (5, 15, 25)
dev = "cuda" if torch.cuda.is_available() else "cpu"
lp = lpips_lib.LPIPS(net="alex", verbose=False).to(dev).eval()
rgb = lambda p: np.asarray(Image.open(p).convert("RGB"), np.float32) / 255

def score(root, codec):
    d = root / codec
    out = {}
    for it in ITERS:
        pairs = []
        for sid in SIDS:
            a, b = d / f"iter{it}" / f"{sid}.png", d / "all" / f"{sid}_step01.jpg"
            if a.exists() and b.exists():
                pairs.append((rgb(a), rgb(b)))
        if not pairs:
            continue
        mse = [float(((u - v) ** 2).mean()) for u, v in pairs]
        ps  = [10 * np.log10(1 / max(m, 1e-10)) for m in mse]
        ss  = [ssim(u, v, channel_axis=2, data_range=1.0) for u, v in pairs]
        t = lambda x: torch.from_numpy(np.stack(x)).permute(0, 3, 1, 2).to(dev) * 2 - 1
        with torch.no_grad():
            ls = lp(t([p[0] for p in pairs]), t([p[1] for p in pairs])).flatten().cpu().tolist()
        out[it] = (np.mean(mse), np.mean(ls), np.mean(ss), np.mean(ps), len(pairs))
    return out

hdr = f"{'codec':22s}" + "".join(f"{m:>21s}" for m in ("MSE ↓", "LPIPS ↓", "SSIM ↑", "PSNR ↑"))
sub = f"{'':22s}" + "".join(f"{k:>7s}" for _ in range(4) for k in ("5", "15", "25"))
print(f"{len(SIDS)} shared samples · scored against each chain's own x¹\n")
for e in EDS:
    print(f"\n{e}"); print(hdr); print(sub); print("-" * len(sub))
    rows = []
    for lab, root, codec in ARMS:
        s = score(root / e, codec)
        if not s:
            print(f"{lab:22s}  (not run)"); continue
        cells = "".join(f"{s[i][0]:7.3f}" if i in s else " " * 7 for i in ITERS) \
              + "".join(f"{s[i][1]:7.3f}" if i in s else " " * 7 for i in ITERS) \
              + "".join(f"{s[i][2]:7.3f}" if i in s else " " * 7 for i in ITERS) \
              + "".join(f"{s[i][3]:7.2f}" if i in s else " " * 7 for i in ITERS)
        n = "/".join(str(s[i][4]) for i in ITERS if i in s)
        print(f"{lab:22s}{cells}   n={n}")
