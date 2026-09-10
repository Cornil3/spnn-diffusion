"""
Iterative img2img on the LSUN Churches LDM: 2blk vs 2blk_rt vs the CompVis teacher.

Each cycle is encode -> add noise to `strength` -> DDIM denoise -> decode, repeated 25
times per image. The LDM is unconditional (`cond_stage_config: "__is_unconditional__"`),
so there is no prompt: `strength` is the only control over how far a cycle moves the
image, which is why it is the parameter that defines this experiment.

Arms:
  compvis   the KL-8 VAE both checkpoints were distilled from (the teacher/baseline)
  2blk      checkpoints_churches_gt_householder_2blk_2gpu/spnn_vae_best.pt
  2blk_rt   checkpoints_churches_gt_householder_2blk_rt_2gpu/spnn_vae_best.pt
            (identical except trained with lambda_roundtrip=1.0)
  2blk_dist checkpoints_churches_gt_householder_2blk_dist_2gpu/spnn_vae_best.pt
            (lambda_decoder_distill=1/gt=0, vs the baseline's distill=0/gt=1)

Configs below are the wandb record for each run, not the committed launcher scripts --
train_athena.slurm does NOT match what these _2gpu checkpoints were actually trained
with. All three share num_blocks=2, hidden=256, scale_bound=1.0, householder,
deep_convmlp=False, img_size=256, lr=1e-4, n_test=1000, lambda_lpips=1, align=1.

    2blk       distill=0  gt=1  roundtrip=0
    2blk_rt    distill=0  gt=1  roundtrip=1     <- clean single-variable vs 2blk
    2blk_dist  distill=1  gt=0  roundtrip=0     <- see caveat

CAVEAT on 2blk vs 2blk_dist: this is not a single-variable swap. Flipping
lambda_decoder_gt to 0 changes three things at once in train.py:
  1. the reconstruction target: ground-truth image -> the VAE's own decode
  2. the reconstruction norm:   l1_loss -> mse_loss  (gt uses L1, distill uses L2)
  3. the LPIPS target, via `lpips_target = images if lambda_decoder_gt > 0
     else vae_decoded`  (train.py ~line 298)
So it contrasts "everything targets ground truth" against "everything targets the
teacher" -- the right experiment for the distillation question, but the L1->L2 change
rides along and can move perceptual metrics on its own.

Arms share a tag, so they see identical source images and identical per-iteration
noise (seed_for is codec-independent) and are directly paired.

Diffusion parameters come from DDNM-main/configs/lsun_churches_2blk_2gpu.yml, i.e. the
churches defaults: quad betas 0.0015->0.0155 over 1000 timesteps, 100 sampling steps,
latent scale factor 0.24578 (baked in by scale_by_std during LDM training). DDNM's
"quad" schedule is diffusers' "scaled_linear" -- linspace over sqrt(beta), then squared.

Metrics are reported against BOTH x^1 (the big table's convention, per REED Appendix A)
and x^0 (the original image), since for pure noise-denoise cycling with no edit applied
the degradation from the original is also a natural reading.

  python -m reed_repro.churches_img2img --n_images 100 --strength 0.5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# DDNM-main is APPENDED, never prepended: it contains a `datasets/` package that would
# otherwise shadow HuggingFace `datasets` and break the LSUN loader. Appending leaves
# site-packages ahead of it while still exposing functions.codec / functions.compvis_unet.
_DDNM = os.path.join(_ROOT, "DDNM-main")
if _DDNM not in sys.path:
    sys.path.append(_DDNM)

from .data import load_png_safe, pil_to_tensor01, save_png_atomic, tensor01_to_pil
from .run_repro import seed_for

SNAPSHOTS = (1, 5, 15, 25)
MAX_ITERS = 25
# From lsun_churches_2blk_2gpu.yml — the churches defaults.
BETA_START, BETA_END, N_TRAIN_STEPS = 0.0015, 0.0155, 1000
SAMPLING_STEPS = 100                 # time_travel.T_sampling
SCALE_FACTOR = 0.24578019976615906   # codec.compvis_scale_factor
IMG_SIZE = 256

ARMS = {
    "compvis": None,   # the teacher VAE
    "2blk":    "checkpoints_churches_gt_householder_2blk_2gpu/spnn_vae_best.pt",
    "2blk_rt": "checkpoints_churches_gt_householder_2blk_rt_2gpu/spnn_vae_best.pt",
    "2blk_dist": "checkpoints_churches_gt_householder_2blk_dist_2gpu/spnn_vae_best.pt",
}


def build_scheduler():
    from diffusers import DDIMScheduler
    # DDNM's "quad" == diffusers' "scaled_linear": linspace over sqrt(beta), squared.
    return DDIMScheduler(num_train_timesteps=N_TRAIN_STEPS, beta_start=BETA_START,
                         beta_end=BETA_END, beta_schedule="scaled_linear",
                         clip_sample=False, set_alpha_to_one=False)


def load_ldm_state(ckpt_path):
    """Read the 2.7GB LDM checkpoint once; the UNet and VAE loaders both take it."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return ckpt["state_dict"] if "state_dict" in ckpt else ckpt


def load_arm(name, device, dtype, ldm_ckpt, ldm_cfg, root, ldm_sd=None):
    """Return a codec exposing encode(x[-1,1]) -> scaled latent and decode(z) -> x."""
    from functions.codec import CompVisVAECodec, SPNNCodec, _load_compvis_vae

    vae, ckpt_sf = _load_compvis_vae(ldm_ckpt, ldm_cfg, device, state_dict=ldm_sd)
    sf = SCALE_FACTOR if abs(ckpt_sf - 1.0) < 1e-9 else ckpt_sf
    if name == "compvis":
        return CompVisVAECodec(vae, sf), f"CompVis KL-8 teacher (sf={sf:.5f})"

    from models import SPNNAutoencoder
    path = os.path.join(root, ARMS[name])
    spnn = SPNNAutoencoder(mix_type="householder", hidden=256, r_hidden=256,
                           scale_bound=1.0, num_blocks=2, use_deep_convmlp=False)
    state = torch.load(path, map_location="cpu", weights_only=False)
    state = state.get("model_state_dict", state)
    missing, unexpected = spnn.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{name}: missing={len(missing)} unexpected={len(unexpected)}")
    spnn = spnn.to(device=device, dtype=dtype).eval().requires_grad_(False)
    return SPNNCodec(spnn, sf), f"SPNN {name} ({os.path.basename(path)})"


@torch.no_grad()
def img2img_once(unet, sched, codec, image_t, strength, steps, generator):
    """One cycle: encode -> noise to `strength` -> DDIM denoise -> decode."""
    sched.set_timesteps(steps)
    z = codec.encode(image_t)                      # codec applies the scale factor
    start = int(steps * (1 - strength))
    t_start = sched.timesteps[start]
    noise = torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=generator)
    z_t = sched.add_noise(z, noise, t_start)
    for t in sched.timesteps[start:]:
        eps = unet(z_t, torch.full((z_t.shape[0],), t, device=z_t.device,
                                   dtype=torch.long))
        z_t = sched.step(eps, t, z_t).prev_sample
    return codec.decode(z_t).clamp(-1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--n_images", type=int, default=100)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=SAMPLING_STEPS)
    ap.add_argument("--max_iters", type=int, default=MAX_ITERS)
    ap.add_argument("--tag", default="churches_img2img")
    ap.add_argument("--out_root", default="reed_repro/results")
    ap.add_argument("--root", default=_ROOT)
    ap.add_argument("--ldm_ckpt", default="DDNM-main/models/lsun_churches.ckpt")
    ap.add_argument("--ldm_cfg", default="DDNM-main/models/lsun_churches-ldm-kl-8.yaml")
    ap.add_argument("--lsun_dir", default=None)
    # main.py's default; training never overrode it, so the held-out test split is the
    # last 1000 HF images. Take the first n_images of THAT, not a differently-sized tail.
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--wandb_project", default="spnn-churches-img2img")
    ap.add_argument("--no_wandb", dest="wandb", action="store_false", default=True)
    args = ap.parse_args()

    max_iters = args.max_iters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    ldm_ckpt = os.path.join(args.root, args.ldm_ckpt)
    ldm_cfg = os.path.join(args.root, args.ldm_cfg)

    from dataset import LSUNChurchesDataset
    ds = LSUNChurchesDataset(img_size=IMG_SIZE, split="test",
                             n_test=args.n_test, data_dir=args.lsun_dir)
    n = min(args.n_images, len(ds))
    print(f"loaded {n} LSUN Churches test images at {IMG_SIZE}px", flush=True)

    from functions.compvis_unet import load_compvis_unet
    ldm_sd = load_ldm_state(ldm_ckpt)
    unet = load_compvis_unet(ldm_ckpt, device, state_dict=ldm_sd)
    sched = build_scheduler()
    print(f"churches LDM: {args.steps} steps, quad betas "
          f"{BETA_START}->{BETA_END}, strength={args.strength}, unconditional", flush=True)

    for arm in args.arms:
        out = Path(args.out_root) / args.tag / arm
        for i in range(1, max_iters + 1):
            (out / f"iter_{i:02d}").mkdir(parents=True, exist_ok=True)
        pending = [j for j in range(n)
                   if not all((out / f"iter_{i:02d}" / f"{j:04d}.png").exists()
                              for i in range(1, max_iters + 1))]
        if not pending:
            print(f"[{arm}] already complete", flush=True)
            continue

        codec, desc = load_arm(arm, device, dtype, ldm_ckpt, ldm_cfg, args.root,
                               ldm_sd=ldm_sd)
        print(f"\n=== {arm}: {desc} — {len(pending)}/{n} images ===", flush=True)

        run = None
        if args.wandb:
            try:
                import wandb
                run = wandb.init(project=args.wandb_project,
                                 name=f"{args.tag}-{arm}-s{args.strength}",
                                 group=f"strength_{args.strength}", job_type=arm,
                                 tags=[arm, args.tag], config=dict(vars(args), codec=desc))
            except Exception as e:
                print(f"  WARNING: wandb init failed: {e!r}", flush=True)

        t0, failures, consecutive = time.time(), [], 0
        for c, j in enumerate(pending, 1):
            try:
                x0 = ds[j].unsqueeze(0).to(device=device, dtype=dtype)  # [-1,1]
                src_pil = tensor01_to_pil((x0[0] + 1) / 2)
                # Persist x^0 once so metrics can be reported against the original as
                # well as against x^1; every arm sees the identical source image.
                src_dir = Path(args.out_root) / args.tag / "source"
                src_dir.mkdir(parents=True, exist_ok=True)
                if not (src_dir / f"{j:04d}.png").exists():
                    save_png_atomic(src_pil, src_dir / f"{j:04d}.png")
                cur, strip = x0, [src_pil]
                for it in range(1, max_iters + 1):
                    g = torch.Generator(device=device).manual_seed(seed_for(f"{j:04d}", it))
                    cur = img2img_once(unet, sched, codec, cur, args.strength, args.steps, g)
                    pil = tensor01_to_pil((cur[0] + 1) / 2)
                    strip.append(pil)
                    save_png_atomic(pil, out / f"iter_{it:02d}" / f"{j:04d}.png")
            except Exception as e:
                failures.append({"idx": j, "error": repr(e)})
                print(f"  !! image {j} failed: {e!r}", flush=True)
                consecutive += 1
                if isinstance(e, OSError) and getattr(e, "errno", None) == 28:
                    print("  ABORT: no space left on device", flush=True); break
                if consecutive >= 5:
                    print(f"  ABORT: {consecutive} consecutive failures", flush=True); break
            else:
                consecutive = 0
                if run is not None:
                    try:
                        from .report import build_grid
                        import wandb
                        cols = ["source"] + [str(i) for i in range(1, max_iters + 1)]
                        run.log({f"{j:04d}/iterations": wandb.Image(
                            build_grid([strip], [arm], cols), caption=f"img {j} — {arm}")})
                    except Exception as e:
                        print(f"  (wandb log failed for {j}: {e!r})", flush=True)
            if c % 5 == 0 or c == len(pending):
                el = time.time() - t0
                print(f"  [{c}/{len(pending)}] {el/60:.1f} min, {el/c:.1f} s/img, "
                      f"ETA {(len(pending)-c)*el/c/60:.1f} min", flush=True)

        (out / "meta.json").write_text(json.dumps(
            {"arm": arm, "codec": desc, "n": n, "strength": args.strength,
             "steps": args.steps, "scale_factor": SCALE_FACTOR, "max_iters": max_iters,
             "failures": failures, "minutes": (time.time() - t0) / 60}, indent=2))
        if run is not None:
            run.summary["n_failures"] = len(failures)
            run.finish()
        del codec
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
