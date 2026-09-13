"""
Metrics for the churches img2img sweep: MSE / PSNR / LPIPS / SSIM / FID at k=5,15,25.

Reported against BOTH references:
  vs x^1  the first cycle's output — the big table's convention (REED Appendix A)
  vs x^0  the original image — the natural reading when no edit is applied, since every
          cycle here is just noise-and-denoise

Scored on the paired intersection of images completed by EVERY arm, so a partially
finished arm cannot be compared against a fuller one on a different image set.

  python -m reed_repro.churches_eval --tag churches_img2img
"""

import argparse
import json
import os
from pathlib import Path

import torch

from .data import load_png_safe, pil_to_tensor01
from .metrics import PairwiseMetrics, compute_fid_paired

KS = [5, 15, 25]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="churches_img2img")
    ap.add_argument("--out_root", default="reed_repro/results")
    ap.add_argument("--arms", nargs="+", default=["compvis", "2blk", "2blk_rt"])
    ap.add_argument("--lpips_net", default="alex", choices=["alex", "vgg"])
    ap.add_argument("--wandb_project", default="spnn-churches-img2img")
    ap.add_argument("--grid_images", type=int, default=12)
    ap.add_argument("--no_wandb", dest="wandb", action="store_false", default=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.out_root) / args.tag
    pair = PairwiseMetrics(device=device, lpips_net=args.lpips_net)

    # paired intersection across arms, over iter_01 and every k
    paired = None
    for arm in args.arms:
        have = None
        for k in [1] + KS:
            keys = {p.stem for p in (root / arm / f"iter_{k:02d}").glob("*.png")}
            have = keys if have is None else (have & keys)
        paired = (have or set()) if paired is None else (paired & (have or set()))
    paired = sorted(paired or set())
    print(f"scoring {len(paired)} images present in every arm at every k\n")

    src = root / "source"
    have_src = src.exists()
    results = {}
    for arm in args.arms:
        results[arm] = {}
        for ref_name in (["x1"] + (["x0"] if have_src else [])):
            for k in KS:
                acc = {m: [] for m in ("mse", "psnr", "lpips", "ssim")}
                usable = []
                for key in paired:
                    a = load_png_safe(root / arm / f"iter_{k:02d}" / f"{key}.png")
                    b = (load_png_safe(root / arm / "iter_01" / f"{key}.png")
                         if ref_name == "x1" else load_png_safe(src / f"{key}.png"))
                    if a is None or b is None:
                        continue
                    usable.append(key)
                    for m, v in pair(pil_to_tensor01(a), pil_to_tensor01(b)).items():
                        acc[m].append(v)
                agg = {m: (sum(v) / len(v) if v else None) for m, v in acc.items()}
                agg["fid"] = compute_fid_paired(
                    root / arm / f"iter_{k:02d}",
                    (root / arm / "iter_01") if ref_name == "x1" else src,
                    usable, device=device)
                agg["n"] = len(usable)
                results[arm][f"{ref_name}@{k}"] = agg
                fid = "n/a" if agg["fid"] is None else f"{agg['fid']:.2f}"
                print(f"  {arm:<9} vs {ref_name} k={k:<3} n={agg['n']:<4} "
                      f"mse={agg['mse']:.4f} psnr={agg['psnr']:.2f} "
                      f"lpips={agg['lpips']:.3f} ssim={agg['ssim']:.3f} fid={fid}")

    (root / "metrics.json").write_text(json.dumps(results, indent=2))

    rows = ["arm,reference,iterations,n,mse,psnr,lpips,ssim,fid"]
    for arm, per in results.items():
        for key, m in per.items():
            ref, k = key.split("@")
            f = lambda x, p: "" if x is None else f"{x:.{p}f}"
            rows.append(f"{arm},{ref},{k},{m['n']},{f(m['mse'],5)},{f(m['psnr'],3)},"
                        f"{f(m['lpips'],4)},{f(m['ssim'],4)},{f(m['fid'],3)}")
    (root / "table.csv").write_text("\n".join(rows) + "\n")

    for ref in (["x1"] + (["x0"] if have_src else [])):
        print(f"\n{'='*96}\nvs {ref}{'   (the big table convention)' if ref=='x1' else '   (the original image)'}\n{'='*96}")
        hdr = f"{'arm':<10}" + "".join(
            f"{m:>8}{k:<4}" for m in ("MSE", "LPIPS", "SSIM", "PSNR") for k in KS) + \
            "".join(f"{'FID':>8}{k:<4}" for k in KS)
        print(hdr); print("-" * len(hdr))
        for arm in args.arms:
            line = f"{arm:<10}"
            for m, dp in (("mse", 4), ("lpips", 3), ("ssim", 3), ("psnr", 2)):
                for k in KS:
                    v = results[arm].get(f"{ref}@{k}", {}).get(m)
                    line += f"{'—' if v is None else format(v, f'.{dp}f'):>12}"
            for k in KS:
                v = results[arm].get(f"{ref}@{k}", {}).get("fid")
                line += f"{'—' if v is None else format(v, '.1f'):>12}"
            print(line)
    print(f"\nwrote {root/'metrics.json'}, {root/'table.csv'}")

    if args.wandb:
        log_to_wandb(args, root, paired, results)


def log_to_wandb(args, root, paired, results):
    """One comparison run: per-image grids stacking the arms as rows, plus the tables.

    The per-arm generation runs each log their own single-row grid while they work; this
    is the cross-arm view, which only exists once every arm has finished.
    """
    try:
        import wandb
        from .report import build_grid
    except Exception as e:
        print(f"wandb logging skipped: {e!r}")
        return
    try:
        run = wandb.init(project=args.wandb_project, name=f"{args.tag}-comparison",
                         group="comparison", job_type="evaluate",
                         tags=["comparison", args.tag], config=vars(args))
    except Exception as e:
        print(f"wandb init failed: {e!r}")
        return

    cols = ["source"] + [str(i) for i in range(1, 26)]
    for key in paired[:args.grid_images]:
        rows, labels = [], []
        for arm in args.arms:
            strip = [load_png_safe(root / "source" / f"{key}.png")]
            strip += [load_png_safe(root / arm / f"iter_{i:02d}" / f"{key}.png")
                      for i in range(1, 26)]
            if any(im is None for im in strip):
                continue
            rows.append(strip)
            labels.append(arm)
        if rows:
            run.log({f"{key}/arms": wandb.Image(
                build_grid(rows, labels, cols),
                caption=f"img {key} — iterations 1..25, one row per arm")})

    for arm, per in results.items():
        for k, m in per.items():
            ref, it = k.split("@")
            run.summary[f"{arm}/{ref}/k{it}"] = {
                kk: vv for kk, vv in m.items() if vv is not None}

    tbl = wandb.Table(columns=["arm", "reference", "iterations", "n",
                               "mse", "psnr", "lpips", "ssim", "fid"])
    for arm, per in results.items():
        for k, m in per.items():
            ref, it = k.split("@")
            tbl.add_data(arm, ref, int(it), m["n"], m["mse"], m["psnr"],
                         m["lpips"], m["ssim"], m["fid"])
    run.log({"metrics_table": tbl})
    run.finish()
    print("logged comparison grids + metrics table to wandb")


if __name__ == "__main__":
    main()
