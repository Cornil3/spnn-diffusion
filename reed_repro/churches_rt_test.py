"""
Paired significance test: does lambda_roundtrip=1.0 actually change anything?

The aggregate table shows 2blk and 2blk_rt separated by very little. Means alone
cannot say whether that is a real effect or sampling noise, so this scores every
image under both arms and runs a Wilcoxon signed-rank test on the 100 paired
differences -- plus a plain count of how many images each arm wins, which is
robust to outliers dominating the mean.

  python -m reed_repro.churches_rt_test --tag churches_img2img
"""

import argparse
from pathlib import Path

import torch

from .data import load_png_safe, pil_to_tensor01
from .metrics import PairwiseMetrics

KS = [5, 15, 25]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="churches_img2img")
    ap.add_argument("--out_root", default="reed_repro/results")
    ap.add_argument("--a", default="2blk")
    ap.add_argument("--b", default="2blk_rt")
    args = ap.parse_args()

    from scipy.stats import wilcoxon

    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.out_root) / args.tag
    pair = PairwiseMetrics(device=device)

    keys = sorted({p.stem for p in (root / args.a / "iter_25").glob("*.png")} &
                  {p.stem for p in (root / args.b / "iter_25").glob("*.png")})
    print(f"paired on {len(keys)} images: {args.a} vs {args.b}\n")

    hdr = (f"{'ref':<5}{'k':<5}{'metric':<8}{args.a+' mean':>13}{args.b+' mean':>14}"
           f"{'delta':>10}{'wins '+args.b:>13}{'p':>10}   verdict")
    print(hdr); print("-" * len(hdr))

    for ref in ("x1", "x0"):
        for k in KS:
            rows = {m: ([], []) for m in ("mse", "psnr", "lpips", "ssim")}
            for key in keys:
                ims = {}
                for arm in (args.a, args.b):
                    tgt = load_png_safe(root / arm / f"iter_{k:02d}" / f"{key}.png")
                    r = (load_png_safe(root / arm / "iter_01" / f"{key}.png")
                         if ref == "x1" else load_png_safe(root / "source" / f"{key}.png"))
                    if tgt is None or r is None:
                        ims = None; break
                    ims[arm] = pair(pil_to_tensor01(tgt), pil_to_tensor01(r))
                if not ims:
                    continue
                for m in rows:
                    rows[m][0].append(ims[args.a][m])
                    rows[m][1].append(ims[args.b][m])

            for m, (va, vb) in rows.items():
                ta = torch.tensor(va); tb = torch.tensor(vb)
                d = (tb - ta)
                lower_better = m in ("mse", "lpips")
                # count images where arm b is the better one
                wins_b = int((d < 0).sum() if lower_better else (d > 0).sum())
                try:
                    p = float(wilcoxon(va, vb).pvalue)
                except Exception:
                    p = float("nan")
                if p < 0.05:
                    better = args.b if (d.mean() < 0) == lower_better else args.a
                    verdict = f"{better} significantly better"
                else:
                    verdict = "no significant difference"
                print(f"{ref:<5}{k:<5}{m:<8}{ta.mean():>13.4f}{tb.mean():>14.4f}"
                      f"{d.mean():>+10.4f}{wins_b:>9}/{len(va):<3}{p:>10.4f}   {verdict}")
        print()


if __name__ == "__main__":
    main()
