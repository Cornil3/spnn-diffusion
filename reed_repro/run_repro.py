"""
Driver for the REED-VAE Table 1 reproduction with SPNN-512 as the consistent VAE.

Two stages, deliberately separate so a protocol fix never costs a re-run of the
diffusion sweep:

  generate  one (model, codec) arm at a time; iterates each of the 179 ImagenHub
            samples 25 times and writes PNG snapshots at iterations 1, 5, 15, 25.
            Resumable - a sample whose snapshots already exist is skipped.
  evaluate  reads the PNGs back and computes MSE/PSNR/LPIPS/SSIM/FID of x^k against
            x^1, then renders the Table 1 layout.

The vanilla and SPNN arms are seeded identically per (sample, iteration), so the two
codecs see the same noise and the difference between them is the codec alone.

Examples
--------
  # pilot: 10 images, both codecs, all five models
  python -m reed_repro.run_repro generate --models all --codecs vae spnn \
      --limit 10 --spnn_checkpoint <ckpt> --latent_scale prescaled --tag pilot10
  python -m reed_repro.run_repro evaluate --tag pilot10

  # full run
  python -m reed_repro.run_repro generate --models all --codecs vae spnn \
      --spnn_checkpoint <ckpt> --latent_scale prescaled --tag full179
"""

import argparse
import json
import os
import time
import traceback
import zlib
from pathlib import Path

import torch

from .data import (RESOLUTION, load_png_safe, load_samples,
                   pil_to_tensor01, png_is_complete, save_png_atomic)
from .editors import EDITORS

SNAPSHOTS = (1, 5, 15, 25)
MAX_ITERS = 25

# Table 1 of REED-VAE, transcribed for side-by-side reference. Not directly
# comparable to our numbers: the paper's base is SD 2.1's VAE while every arm here
# is SD1.5-family, and REED's pipeline could not be reproduced (the repo at
# github.com/galmog/REED-VAE is a placeholder - README and a teaser image only).
REED_TABLE1 = {
    "ip2p":       {"vanilla": {"mse": [0.02, 0.11, 0.15], "lpips": [0.33, 0.69, 0.76], "ssim": [0.60, 0.23, 0.18], "fid": [105.75, 246.98, 271.72], "psnr": [17.78, 10.15, 8.59]},
                   "reed":    {"mse": [0.02, 0.06, 0.09], "lpips": [0.18, 0.45, 0.58], "ssim": [0.80, 0.53, 0.41], "fid": [62.84, 138.90, 187.97], "psnr": [19.81, 13.36, 11.36]}},
    "magicbrush": {"vanilla": {"mse": [0.02, 0.08, 0.14], "lpips": [0.31, 0.71, 0.80], "ssim": [0.65, 0.21, 0.13], "fid": [103.55, 266.99, 295.81], "psnr": [18.84, 11.35, 8.75]},
                   "reed":    {"mse": [0.01, 0.03, 0.05], "lpips": [0.19, 0.51, 0.69], "ssim": [0.81, 0.60, 0.45], "fid": [74.70, 174.69, 223.75], "psnr": [21.53, 16.66, 14.09]}},
    "diffedit":   {"vanilla": {"mse": [0.03, 0.06, 0.08], "lpips": [0.34, 0.62, 0.73], "ssim": [0.65, 0.36, 0.21], "fid": [160.73, 246.33, 301.91], "psnr": [15.99, 12.59, 11.21]},
                   "reed":    {"mse": [0.03, 0.07, 0.08], "lpips": [0.30, 0.55, 0.68], "ssim": [0.69, 0.48, 0.40], "fid": [160.28, 226.91, 246.52], "psnr": [16.24, 12.91, 11.44]}},
    "pbe":        {"vanilla": {"mse": [0.02, 0.04, 0.07], "lpips": [0.26, 0.60, 0.71], "ssim": [0.66, 0.33, 0.22], "fid": [83.49, 209.29, 253.57], "psnr": [18.55, 13.88, 11.74]},
                   "reed":    {"mse": [0.02, 0.03, 0.04], "lpips": [0.20, 0.44, 0.59], "ssim": [0.77, 0.61, 0.54], "fid": [74.24, 141.09, 178.29], "psnr": [19.62, 16.19, 14.43]}},
    "sd_inpaint": {"vanilla": {"mse": [0.01, 0.06, 0.11], "lpips": [0.29, 0.69, 0.78], "ssim": [0.67, 0.22, 0.14], "fid": [95.09, 255.78, 283.36], "psnr": [20.46, 12.31, 9.72]},
                   "reed":    {"mse": [0.01, 0.03, 0.05], "lpips": [0.17, 0.47, 0.65], "ssim": [0.80, 0.57, 0.41], "fid": [72.73, 166.06, 210.42], "psnr": [23.14, 16.66, 13.61]}},
}


def _out_suffix(models):
    """Per-model eval jobs must not clobber each other's result files.

    A whole-table run writes metrics.json / table1.{md,csv}; a run scoped to a subset
    writes metrics_<model>.json etc. That lets one eval job fire the moment each
    model's two arms finish, while a final all-models pass still produces the combined
    Table 1.
    """
    return "" if set(models) == set(EDITORS) else "_" + "-".join(sorted(models))


def arm_dir(root, tag, model, codec):
    return Path(root) / tag / model / codec


def seed_for(key, iteration):
    """Deterministic and codec-independent, so both arms share their noise."""
    return zlib.crc32(f"{key}|{iteration}".encode()) % (2 ** 31)


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def generate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.fp16 else torch.float32
    samples = load_samples(size=args.resolution, resize_mode=args.resize_mode,
                           limit=args.limit, from_disk=args.from_disk,
                           mask_open=args.mask_open, mask_close=args.mask_close)
    print(f"loaded {len(samples)} ImagenHub samples "
          f"({args.resolution}px, resize_mode={args.resize_mode})")

    for model in args.models:
        for codec in args.codecs:
            out = arm_dir(args.out_root, args.tag, model, codec)
            # Keeping every iteration costs ~6x the PNGs but makes the per-image
            # iteration grids (and any later analysis) possible without re-running
            # the sweep. Metrics still read only from the SNAPSHOTS dirs.
            keep = tuple(range(1, MAX_ITERS + 1)) if args.save_all_iters else SNAPSHOTS
            for i in keep:
                (out / f"iter_{i:02d}").mkdir(parents=True, exist_ok=True)

            pending = [s for s in samples
                       if not all(png_is_complete(out / f"iter_{i:02d}" / f"{s.key}.png")
                                  for i in keep)]
            if not pending:
                print(f"[{model}/{codec}] all {len(samples)} samples done, skipping")
                continue

            print(f"\n=== {model} / {codec} — {len(pending)}/{len(samples)} samples to run ===")
            t0 = time.time()
            editor = EDITORS[model](
                codec=codec, device=device, dtype=dtype,
                spnn_checkpoint=args.spnn_checkpoint, spnn_weights=args.spnn_weights,
                latent_scale=args.latent_scale, overrides=args.overrides.get(model))

            # Every run goes to wandb by default, grouped by editing task, so results
            # can be watched while the sweep runs rather than only at the end.
            run = None
            if args.wandb:
                try:
                    from .report import (build_grid, build_inputs_grid,
                                         inputs_caption, start_run)
                    run = start_run(args.wandb_project, args.wandb_entity, args.tag,
                                    model, codec,
                                    extra_config={"params": editor.params,
                                                  "vae_source": editor.vae_source,
                                                  "imagenhub_api": editor.used_imagenhub,
                                                  "n_samples": len(samples),
                                                  "resolution": args.resolution,
                                                  "max_iters": MAX_ITERS,
                                                  "latent_scale": args.latent_scale
                                                  if codec == "spnn" else None})
                except Exception as e:
                    # Never let telemetry cost us a 26-hour generation run.
                    print(f"  WARNING: wandb init failed, continuing without it: {e!r}")
                    run = None
            row_label = {"vae": "own VAE (baseline)", "spnn": "+ SPNN"}.get(codec, codec)
            col_labels = ["source"] + [str(i) for i in range(1, MAX_ITERS + 1)]

            failures = []
            for n, s in enumerate(pending, 1):
                try:
                    cur = s.source
                    strip = [s.source]
                    for it in range(1, MAX_ITERS + 1):
                        # ImagenHub's infer_one_image takes an int seed and calls
                        # torch.manual_seed itself, so we pass the seed rather than a
                        # Generator. Identical per (sample, iteration) across codecs,
                        # so both arms see the same noise.
                        cur = editor.edit(s, cur, it, seed_for(s.key, it))
                        if cur.size != (args.resolution, args.resolution):
                            cur = cur.resize((args.resolution, args.resolution))
                        strip.append(cur)
                        if it in keep:
                            save_png_atomic(cur, out / f"iter_{it:02d}" / f"{s.key}.png")
                except Exception as e:
                    failures.append({"key": s.key, "error": repr(e),
                                     "trace": traceback.format_exc()[-2000:]})
                    print(f"  !! {s.key} failed: {e!r}")
                else:
                    # Outside the try: a wandb hiccup must not mark a good sample
                    # failed, and the PNGs are already safely on disk by here.
                    if run is not None:
                        try:
                            import wandb
                            # Two panels per sample, keyed "<sample>/..." so wandb makes
                            # one section per image: the iteration strip, and the
                            # conditioning actually fed to this editor (mask, mask-on-
                            # source overlay, PbE references). Without the inputs panel
                            # the mask is invisible while the sweep runs, and a mask
                            # problem is exactly what you want to catch early.
                            grid = build_inputs_grid(s, model)
                            run.log({
                                f"{s.key}/iterations": wandb.Image(
                                    build_grid([strip], [row_label], col_labels),
                                    caption=f"{s.key} — {row_label}"),
                                f"{s.key}/inputs": wandb.Image(
                                    grid, caption=inputs_caption(s, model)),
                            })
                        except Exception as e:
                            print(f"  (wandb log failed for {s.key}: {e!r})")
                if run is not None:
                    el = time.time() - t0
                    try:
                        run.log({"progress/samples_done": n,
                                 "progress/samples_total": len(pending),
                                 "progress/frac": n / max(len(pending), 1),
                                 "progress/sec_per_sample": el / n,
                                 "progress/eta_min": (len(pending) - n) * el / n / 60,
                                 "progress/failures": len(failures)})
                    except Exception:
                        pass
                if n % 5 == 0 or n == len(pending):
                    el = time.time() - t0
                    print(f"  [{n}/{len(pending)}] {el/60:.1f} min elapsed, "
                          f"{el/n:.1f} s/sample, ETA {(len(pending)-n)*el/n/60:.1f} min")

            meta = {"model": model, "codec": codec, "n_samples": len(samples),
                    "params": editor.params, "resolution": args.resolution,
                    "vae_source": editor.vae_source,
                    "imagenhub_api": editor.used_imagenhub,
                    "resize_mode": args.resize_mode, "max_iters": MAX_ITERS,
                    "mask_open": args.mask_open, "mask_close": args.mask_close,
                    "from_disk": args.from_disk,
                    "snapshots": list(SNAPSHOTS), "saved_iters": list(keep),
                    "fp16": args.fp16,
                    "spnn_checkpoint": args.spnn_checkpoint if codec == "spnn" else None,
                    "spnn_weights": args.spnn_weights if codec == "spnn" else None,
                    "latent_scale": args.latent_scale if codec == "spnn" else None,
                    "failures": failures, "minutes": (time.time() - t0) / 60}
            (out / "meta.json").write_text(json.dumps(meta, indent=2))
            print(f"=== {model}/{codec} done in {meta['minutes']:.1f} min, "
                  f"{len(failures)} failures ===")
            if run is not None:
                try:
                    run.summary["n_failures"] = len(failures)
                    run.summary["minutes"] = meta["minutes"]
                    run.finish()
                except Exception as e:
                    print(f"  (wandb finish failed: {e!r})")

            del editor
            if device == "cuda":
                torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def evaluate(args):
    from PIL import Image
    from .metrics import PairwiseMetrics, compute_fid_paired

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pair = PairwiseMetrics(device=device, lpips_net=args.lpips_net)
    ks = [k for k in SNAPSHOTS if k != 1]
    results = {}

    for model in args.models:
        # Score only samples that every codec arm finished, at every k. Otherwise a
        # partially-complete arm is compared against a fuller one on a DIFFERENT set of
        # images, which silently confounds the codec comparison -- the arms must be
        # paired for the difference between them to mean anything.
        paired = None
        for codec in args.codecs:
            a = arm_dir(args.out_root, args.tag, model, codec)
            if not (a / "iter_01").exists():
                continue
            have = None
            for k in [1] + ks:
                kk = {p.stem for p in (a / f"iter_{k:02d}").glob("*.png")}
                have = kk if have is None else (have & kk)
            paired = (have or set()) if paired is None else (paired & (have or set()))
        paired = paired or set()

        for codec in args.codecs:
            out = arm_dir(args.out_root, args.tag, model, codec)
            ref_dir = out / "iter_01"
            if not ref_dir.exists():
                continue
            keys = sorted(paired)
            if not keys:
                continue
            arm_total = len({p.stem for p in ref_dir.glob("*.png")})
            if arm_total > len(keys):
                print(f"  note: {model}/{codec} has {arm_total} samples but only "
                      f"{len(keys)} are complete in every arm — scoring the paired "
                      f"intersection so the comparison stays like-for-like")
            print(f"\n=== evaluating {model}/{codec} ({len(keys)} images) ===")
            per_k = {}
            for k in ks:
                kdir = out / f"iter_{k:02d}"
                # Only score samples that produced BOTH snapshots.
                usable, acc = [], {"mse": [], "psnr": [], "lpips": [], "ssim": []}
                for key in keys:
                    ia = load_png_safe(kdir / f"{key}.png")
                    ib = load_png_safe(ref_dir / f"{key}.png")
                    if ia is None or ib is None:
                        continue          # missing or truncated -> not scoreable
                    usable.append(key)
                    for name, v in pair(pil_to_tensor01(ia), pil_to_tensor01(ib)).items():
                        acc[name].append(v)
                agg = {name: (sum(v) / len(v) if v else None) for name, v in acc.items()}
                agg["fid"] = compute_fid_paired(kdir, ref_dir, usable, device=device)
                agg["n"] = len(usable)
                per_k[k] = agg
                fid_s = "n/a" if agg["fid"] is None else f"{agg['fid']:.2f}"
                print(f"  k={k:>2} n={agg['n']:>3}  mse={agg['mse']:.4f} "
                      f"psnr={agg['psnr']:.2f} lpips={agg['lpips']:.3f} "
                      f"ssim={agg['ssim']:.3f} fid={fid_s}")
            results.setdefault(model, {})[codec] = per_k

    dest = Path(args.out_root) / args.tag
    dest.mkdir(parents=True, exist_ok=True)
    sfx = _out_suffix(args.models)
    mj, tm, tc = f"metrics{sfx}.json", f"table1{sfx}.md", f"table1{sfx}.csv"
    (dest / mj).write_text(json.dumps(results, indent=2))
    write_table(results, dest / tm, args.tag)
    write_csv(results, dest / tc)
    print(f"\nwrote {dest/mj}, {dest/tm}, {dest/tc}")
    return results


def write_csv(results, path):
    rows = ["model,codec,iterations,n,mse,psnr,lpips,ssim,fid"]
    for model, by_codec in results.items():
        for codec, per_k in by_codec.items():
            for k, m in sorted(per_k.items()):
                def f(x, p):
                    return "" if x is None else f"{x:.{p}f}"
                rows.append(f"{model},{codec},{k},{m['n']},{f(m['mse'],5)},"
                            f"{f(m['psnr'],3)},{f(m['lpips'],4)},{f(m['ssim'],4)},"
                            f"{f(m['fid'],3)}")
    Path(path).write_text("\n".join(rows) + "\n")


def write_table(results, path, tag):
    ks = [k for k in SNAPSHOTS if k != 1]
    # 3dp for MSE: the paper prints 2dp, but our values can land below 0.005 and
    # would render as a useless "0.00".
    metric_order = [("mse", 3, "MSE ↓"), ("lpips", 2, "LPIPS ↓"), ("ssim", 2, "SSIM ↑"),
                    ("fid", 2, "FID ↓"), ("psnr", 2, "PSNR ↑")]
    label = {"vae": "Vanilla (model own VAE)", "spnn": "+ SPNN (ours)"}

    L = [f"# REED-VAE Table 1 reproduction — SPNN-512 as the consistent VAE (`{tag}`)", "",
         "Metrics of x^k against **x¹** (the first edit's output), per REED-VAE Appendix A —",
         "not against the source image. MSE/PSNR/LPIPS on [0,1]-normalized samples; FID via",
         "pytorch-fid. Iterations 5/15/25 are all odd, so they share edit direction with x¹.", "",
         "The REED rows are transcribed from the published paper for orientation only. They are",
         "**not** directly comparable: REED's base is SD 2.1's VAE, every arm below is",
         "SD1.5-family (which is what SPNN-512 was distilled against), and REED's own code was",
         "never released, so their pipeline could not be re-run here.", ""]

    header = "| Method | " + " | ".join(
        f"{name.split()[0]} {k}" for _, _, name in metric_order for k in ks) + " |"
    L += [header, "|" + "---|" * (1 + len(metric_order) * len(ks))]

    for model in results:
        for codec in ("vae", "spnn"):
            per_k = results.get(model, {}).get(codec)
            if not per_k:
                continue
            cells = []
            for key, prec, _ in metric_order:
                for k in ks:
                    v = per_k.get(k, {}).get(key)
                    cells.append("—" if v is None else f"{v:.{prec}f}")
            L.append(f"| **{model}** / {label[codec]} | " + " | ".join(cells) + " |")
        ref = REED_TABLE1.get(model)
        if ref:
            for which, disp in (("vanilla", "_paper: Vanilla SD2.1_"), ("reed", "_paper: + REED_")):
                cells = [f"{v:.2f}" for key, _, _ in metric_order for v in ref[which][key]]
                L.append(f"| {model} / {disp} | " + " | ".join(cells) + " |")
    Path(path).write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# report — per-image iteration grids + the metrics table, into one wandb run
# ---------------------------------------------------------------------------

def report(args):
    from .report import log_report

    samples = load_samples(size=args.resolution, resize_mode=args.resize_mode,
                           limit=args.limit, from_disk=args.from_disk,
                           mask_open=args.mask_open, mask_close=args.mask_close)
    metrics_path = (Path(args.out_root) / args.tag /
                    f"metrics{_out_suffix(args.models)}.json")
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    if metrics is None:
        print(f"note: {metrics_path} not found — logging grids without the metrics table")
    else:
        # JSON turns the integer iteration keys into strings; put them back.
        metrics = {m: {c: {int(k): v for k, v in per_k.items()}
                       for c, per_k in by_c.items()} for m, by_c in metrics.items()}

    n = log_report(args.out_root, args.tag, args.models, args.codecs, samples,
                   project=args.wandb_project, entity=args.wandb_entity,
                   metrics=metrics, max_cols=args.grid_max_cols,
                   run_name=args.wandb_run_name)
    print(f"logged {n} grids to wandb project {args.wandb_project!r}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["generate", "evaluate", "report"])
    ap.add_argument("--tag", default="run", help="names the results subdirectory")
    ap.add_argument("--models", nargs="+", default=["all"],
                    help=f"any of {list(EDITORS)} or 'all'")
    ap.add_argument("--codecs", nargs="+", default=["vae", "spnn"],
                    choices=["vae", "spnn"])
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N of the 179 samples (pilot runs)")
    ap.add_argument("--out_root", default="reed_repro/results")
    ap.add_argument("--resolution", type=int, default=RESOLUTION)
    ap.add_argument("--resize_mode", default="resize", choices=["resize", "center_crop"])
    ap.add_argument("--from_disk", default=None,
                    help="path to a save_to_disk copy of the augmented `filtered` split "
                         "(one that already has reverse_instruction/processed_mask); "
                         "otherwise the hub copy is used and reverse prompts come from the CSV")
    ap.add_argument("--mask_open", type=int, default=5,
                    help="morphological opening radius used to de-speckle the raw mask_img "
                         "(0 disables)")
    ap.add_argument("--mask_close", type=int, default=9,
                    help="morphological closing radius used to fill mask pinholes (0 disables)")
    ap.add_argument("--fp16", action="store_true",
                    help="half precision; SPNN's Householder mixer accumulates in fp64 "
                         "regardless, but leave this off for the numbers of record")
    ap.add_argument("--spnn_checkpoint", default=None)
    ap.add_argument("--spnn_weights", default="ema", choices=["ema", "model"])
    ap.add_argument("--latent_scale", default="prescaled", choices=["prescaled", "raw"],
                    help="run verify_codec.py first; pass the regime it measures")
    ap.add_argument("--lpips_net", default="alex", choices=["alex", "vgg"])
    ap.add_argument("--save_all_iters", action="store_true", default=True,
                    help="write a PNG for every one of the 25 iterations (default). "
                         "Needed for the per-image iteration grids.")
    ap.add_argument("--snapshots_only", dest="save_all_iters", action="store_false",
                    help="write only iterations 1/5/15/25 — metrics still work, but "
                         "the grids will be sparse")
    ap.add_argument("--wandb", action="store_true", default=True,
                    help="log runs, iteration grids and metrics to wandb (default on)")
    ap.add_argument("--no_wandb", dest="wandb", action="store_false",
                    help="disable wandb logging")
    ap.add_argument("--wandb_project", default="reed-vae-repro")
    ap.add_argument("--wandb_entity",
                    default="yamitehrlich-technion-israel-institute-of-technology")
    ap.add_argument("--wandb_run_name", default=None)
    ap.add_argument("--grid_max_cols", type=int, default=None,
                    help="cap the columns per grid (default: every saved iteration)")
    ap.add_argument("--overrides", default="{}",
                    help='JSON of per-model pipeline kwargs, e.g. '
                         '\'{"ip2p": {"num_inference_steps": 50}}\'')
    args = ap.parse_args()

    if args.models == ["all"]:
        args.models = list(EDITORS)
    for m in args.models:
        if m not in EDITORS:
            ap.error(f"unknown model {m!r}; choose from {list(EDITORS)}")
    args.overrides = json.loads(args.overrides)

    if args.stage == "generate":
        if "spnn" in args.codecs and not args.spnn_checkpoint:
            ap.error("--spnn_checkpoint is required when generating the spnn arm")
        generate(args)
    elif args.stage == "evaluate":
        evaluate(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
