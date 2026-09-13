"""One sample through all five editors — do the edits actually happen, and look right?

Runs a few iterations of every model on the same ImagenHub samples and renders one
grid per sample: the conditioning (source, mask, PbE references) followed by each
model's first iterations. Also prints, per model, how much the output moved away from
the source and whether the change is concentrated inside the mask — a no-op edit or a
mask applied to the wrong region both show up numerically here, not just visually.

  python -m reed_repro.sanity_check --keys 102171_t1 100081_t1 --iters 3
"""

import argparse
import os

import numpy as np
import torch

from .data import load_samples, pil_to_tensor01
from .editors import EDITORS, TASK_TYPE
from .report import build_grid, mask_overlay
from .run_repro import seed_for


def change_stats(before, after, mask=None):
    a = pil_to_tensor01(before).numpy()
    b = pil_to_tensor01(after).numpy()
    d = np.abs(a - b).mean(axis=0)
    out = {"mean_abs_change": float(d.mean())}
    if mask is not None:
        m = np.array(mask) > 127
        if m.any() and not m.all():
            out["change_in_mask"] = float(d[m].mean())
            out["change_out_mask"] = float(d[~m].mean())
            out["in_over_out"] = out["change_in_mask"] / max(out["change_out_mask"], 1e-8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", nargs="+", default=["102171_t1", "100081_t1"])
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--models", nargs="+", default=list(EDITORS))
    ap.add_argument("--codec", default="vae", choices=["vae", "spnn"])
    ap.add_argument("--spnn_checkpoint", default=None)
    ap.add_argument("--latent_scale", default="prescaled")
    ap.add_argument("--out_dir", default="reed_repro/results/sanity")
    ap.add_argument("--wandb_project", default="reed-vae-repro")
    ap.add_argument("--no_wandb", dest="wandb", action="store_false", default=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    all_samples = {s.key: s for s in load_samples()}
    missing = [k for k in args.keys if k not in all_samples]
    if missing:
        raise SystemExit(f"unknown keys {missing}; e.g. {list(all_samples)[:5]}")
    samples = [all_samples[k] for k in args.keys]

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, name=f"sanity-{args.codec}",
                         job_type="sanity", config=vars(args))

    strips = {s.key: [] for s in samples}
    labels = {s.key: [] for s in samples}

    for model in args.models:
        print(f"\n{'='*66}\n{model}  ({TASK_TYPE[model]})\n{'='*66}", flush=True)
        editor = EDITORS[model](codec=args.codec, device=device, dtype=torch.float32,
                                spnn_checkpoint=args.spnn_checkpoint,
                                latent_scale=args.latent_scale)
        for s in samples:
            cur = s.source
            row = [s.source]
            for it in range(1, args.iters + 1):
                prev = cur
                cur = editor.edit(s, cur, it, seed_for(s.key, it))
                if cur.size != s.source.size:
                    cur = cur.resize(s.source.size)
                row.append(cur)
                st = change_stats(prev, cur, s.mask if model in ("pbe", "sd_inpaint") else None)
                extra = ""
                if "in_over_out" in st:
                    extra = (f"  in-mask={st['change_in_mask']:.4f} "
                             f"out={st['change_out_mask']:.4f} "
                             f"ratio={st['in_over_out']:.1f}")
                direction = "fwd" if it % 2 == 1 else "rev"
                print(f"  {s.key} iter{it} ({direction}) "
                      f"change_vs_prev={st['mean_abs_change']:.4f}{extra}", flush=True)
            vs_src = change_stats(s.source, cur)
            print(f"  {s.key} total change vs source = {vs_src['mean_abs_change']:.4f}"
                  f"{'   <-- WARNING: near-zero, edit may be a no-op' if vs_src['mean_abs_change'] < 0.005 else ''}",
                  flush=True)
            strips[s.key].append(row)
            labels[s.key].append(model)
        del editor
        if device == "cuda":
            torch.cuda.empty_cache()

    for s in samples:
        cond = [s.source, s.mask, mask_overlay(s.source, s.mask), s.ref_target, s.ref_source]
        rows = [cond + [None] * (args.iters + 1 - len(cond))] + strips[s.key]
        row_labels = ["inputs"] + labels[s.key]
        cols = ["source", "mask / it1", "overlay / it2", "x_r^1 / it3", "x_r^2"]
        cols = cols[:args.iters + 1] + [f"it{i}" for i in range(len(cols), args.iters + 1)]
        g = build_grid(rows, row_labels, cols, cell=170)
        path = os.path.join(args.out_dir, f"sanity_{s.key}_{args.codec}.png")
        g.save(path)
        print(f"\nwrote {path}")
        if run is not None:
            import wandb
            run.log({f"sanity/{s.key}": wandb.Image(g, caption=f"{s.key} — {args.codec}")})

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
