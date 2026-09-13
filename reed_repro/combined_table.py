"""Render the full Table 1 comparison: paper vs. this reproduction, four rows per model.

    <Model>            <- REED-VAE Table 1, vanilla   (SD 2.1 base)
      + REED  (paper)  <- REED-VAE Table 1, + REED
      vanilla (ours)   <- this reproduction, each model's OWN VAE
      + SPNN  (ours)   <- this reproduction, SPNN-512 swapped in

Reads whatever per-model results exist (table1_<model>.csv), so it can be run while
some models are still generating and re-run when they finish.

  python -m reed_repro.combined_table --tag full179
"""

import argparse
import csv
import os

from .editors import EDITORS
from .run_repro import REED_TABLE1, SNAPSHOTS

NAME = {"ip2p": "IP2P [BHE23]", "magicbrush": "MagicBrush [ZMC*23]",
        "diffedit": "DiffEdit [CVSC22]", "pbe": "PbE [YGZ*23]",
        "sd_inpaint": "SD Inpainting [RBL*22]"}
# (key, header, decimals, width) — MSE needs 3dp: at the paper's 2dp our vanilla and
# SPNN collapse to the same value at k=15 and the comparison becomes unreadable.
MET = [("mse", "MSE ↓", 3, 5), ("lpips", "LPIPS ↓", 2, 5), ("ssim", "SSIM ↑", 2, 5),
       ("fid", "FID ↓", 1, 7), ("psnr", "PSNR ↑", 2, 6)]
LABEL_W = 26


def load(root, tag, models):
    out = {}
    for m in models:
        p = os.path.join(root, tag, f"table1_{m}.csv")
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            out[(r["model"], r["codec"], int(r["iterations"]))] = {
                k: float(r[k]) for k in ("mse", "psnr", "lpips", "ssim", "fid")}
            out.setdefault("_n", {})[r["model"]] = int(r["n"])
    return out


def render(D, models, ks):
    lines = []
    h1, h2 = " " * LABEL_W, "Method".ljust(LABEL_W)
    for _, lab, _, w in MET:
        h1 += lab.center(3 * (w + 2))
        h2 += "".join(str(k).rjust(w + 2) for k in ks)
    lines += [h1, h2, "-" * len(h2)]

    def row(label, get):
        s = label.ljust(LABEL_W)
        for key, _, dp, w in MET:
            for k in ks:
                v = get(key, k)
                s += ("—" if v is None else f"{v:.{dp}f}").rjust(w + 2)
        return s

    for m in models:
        have = (m, "vae", ks[0]) in D
        ref = REED_TABLE1.get(m)
        if ref:
            lines.append(row(NAME.get(m, m),
                             lambda kk, k, r=ref: r["vanilla"][kk][ks.index(k)]))
            lines.append(row("  + REED  (paper)",
                             lambda kk, k, r=ref: r["reed"][kk][ks.index(k)]))
        if have:
            lines.append(row("  vanilla (ours)", lambda kk, k, mm=m: D[(mm, "vae", k)][kk]))
            lines.append(row("  + SPNN  (ours)", lambda kk, k, mm=m: D[(mm, "spnn", k)][kk]))
        else:
            lines.append("  (this reproduction: not finished yet)".ljust(LABEL_W))
        lines.append("-" * len(h2))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full179")
    ap.add_argument("--out_root", default="reed_repro/results")
    ap.add_argument("--models", nargs="+", default=list(EDITORS))
    args = ap.parse_args()

    ks = [k for k in SNAPSHOTS if k != 1]
    D = load(args.out_root, args.tag, args.models)
    lines = render(D, args.models, ks)
    ns = D.get("_n", {})
    lines += [
        "",
        "Rows 1-2: transcribed from REED-VAE Table 1 (SD 2.1 base, VAE swapped in).",
        "Rows 3-4: this reproduction (SD 1.5 family; the baseline is each model's OWN VAE).",
        f"n per model: {ns}",
        "Compare WITHIN each pair (paper vs paper, ours vs ours) — the two setups differ",
        "in base model and in what the 'vanilla' baseline is, so cross-pair deltas mislead.",
    ]
    text = "\n".join(lines)
    print(text)
    dest = os.path.join(args.out_root, args.tag, "table1_combined.txt")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as f:
        f.write(text + "\n")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
