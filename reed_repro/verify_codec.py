"""
Run this FIRST on athena, before spending any GPU-hours on the table.

Three things have to be true before the reproduction means anything:

  1. The checkpoint loads strictly into SPNNAutoencoder512 with the declared
     hyper-parameters (householder / hidden=128 / r_hidden=256 / scale_bound=1.0).
  2. We know which latent-scale regime the encoder is in. sd15_inpaint_latent.yml
     claims SPNN-512 is "prescaled into SD's latent space", but a wrong guess here
     hands the UNet latents that are off by 5.5x and quietly ruins every number, so
     we measure it against the real VAE instead of trusting the comment.
  3. SPNN's single-pass reconstruction is at least competitive with the VAE's. REED's
     whole claim is about *iterative* degradation; if SPNN were far behind at one
     pass the iterative comparison would be confounded from the start.

Also reports the 1/5/15/25-iteration encode-decode ladder for both codecs — the
paper's Table A2 experiment, and the cheapest possible preview of whether SPNN
behaves like REED does.

Usage:
  python -m reed_repro.verify_codec --checkpoint <path to ckpt_last.pt> [--weights ema|model]
"""

import argparse
import json
import os

import torch

from .codecs import SD15_SCALING_FACTOR, SPNNVAEShim, load_spnn512
from .data import load_samples, pil_to_tensor01


def psnr01(a, b):
    mse = torch.mean((a.clamp(0, 1) - b.clamp(0, 1)) ** 2).item()
    return float("inf") if mse == 0 else 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--weights", default="ema", choices=["ema", "model"])
    # Same env override the editors use, so a warm cache is reused. athena holds
    # `runwayml/stable-diffusion-v1-5` (hub-removed, byte-identical to the mirror) and
    # runs with HF_HUB_OFFLINE=1, where asking for the mirror would be a cache miss.
    ap.add_argument("--sd15", default=os.environ.get(
        "REED_SD15_ID", "stable-diffusion-v1-5/stable-diffusion-v1-5"))
    ap.add_argument("--n_images", type=int, default=8)
    ap.add_argument("--iters", type=int, nargs="+", default=[1, 5, 15, 25])
    ap.add_argument("--mix_type", default="householder")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--r_hidden", type=int, default=256)
    ap.add_argument("--scale_bound", type=float, default=1.0)
    ap.add_argument("--out", default="reed_repro/results/codec_verification.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from diffusers import AutoencoderKL

    print(f"[1/4] loading SD 1.5 VAE from {args.sd15}")
    vae = AutoencoderKL.from_pretrained(args.sd15, subfolder="vae").to(device).eval()
    vae.requires_grad_(False)
    sf = float(vae.config.scaling_factor)
    print(f"      scaling_factor = {sf}")

    print(f"[2/4] loading SPNN-512 ({args.weights}) from {args.checkpoint}")
    spnn = load_spnn512(args.checkpoint, weights=args.weights, device=device,
                        mix_type=args.mix_type, hidden=args.hidden,
                        r_hidden=args.r_hidden, scale_bound=args.scale_bound)
    print("      strict load OK")

    print(f"[3/4] loading {args.n_images} ImagenHub images")
    samples = load_samples(limit=args.n_images)
    x = torch.stack([pil_to_tensor01(s.source) for s in samples]).to(device)
    x = x * 2.0 - 1.0  # SPNN and the VAE both take [-1,1]

    # ---- latent scale regime -------------------------------------------------
    with torch.no_grad():
        z_spnn = spnn.encode(x)
        z_vae_raw = vae.encode(x).latent_dist.mode()
    z_vae_scaled = z_vae_raw * sf
    d_raw = (z_spnn - z_vae_raw).abs().mean().item()
    d_scaled = (z_spnn - z_vae_scaled).abs().mean().item()
    regime = "prescaled" if d_scaled < d_raw else "raw"

    print("\n--- latent scale ---")
    print(f"  SPNN latent      : std={z_spnn.std():.4f} absmax={z_spnn.abs().max():.4f}")
    print(f"  VAE raw latent   : std={z_vae_raw.std():.4f} absmax={z_vae_raw.abs().max():.4f}")
    print(f"  VAE scaled latent: std={z_vae_scaled.std():.4f} absmax={z_vae_scaled.abs().max():.4f}")
    print(f"  |SPNN - VAE_raw|    = {d_raw:.5f}")
    print(f"  |SPNN - VAE_scaled| = {d_scaled:.5f}")
    print(f"  => regime = {regime.upper()}  (pass --latent_scale {regime} to run_repro.py)")
    if abs(d_raw - d_scaled) / max(d_raw, d_scaled, 1e-9) < 0.15:
        print("  !! WARNING: the two are close; alignment may be weak. Inspect before trusting.")

    shim = SPNNVAEShim(spnn, vae.config, latent_scale=regime, scaling_factor=sf)

    # ---- iterative encode/decode ladder (paper Fig. 2 / Table A2) ------------
    print(f"\n[4/4] iterative encode-decode, {max(args.iters)} iterations")

    def ladder(enc_dec):
        cur, out = x.clone(), {}
        for i in range(1, max(args.iters) + 1):
            cur = enc_dec(cur)
            if i in args.iters:
                a, b = (cur + 1) / 2, (x + 1) / 2
                out[i] = {
                    "psnr": psnr01(a, b),
                    "mse": torch.mean((a.clamp(0, 1) - b.clamp(0, 1)) ** 2).item(),
                }
        return out

    with torch.no_grad():
        vae_ladder = ladder(lambda t: vae.decode(vae.encode(t).latent_dist.mode()).sample)
        spnn_ladder = ladder(lambda t: shim.decode(shim.encode(t).latent_dist.mode()).sample)

    print(f"\n  {'iter':>5} | {'VAE PSNR':>9} {'VAE MSE':>9} | {'SPNN PSNR':>10} {'SPNN MSE':>9} | {'dPSNR':>7}")
    for i in args.iters:
        v, s = vae_ladder[i], spnn_ladder[i]
        print(f"  {i:>5} | {v['psnr']:>9.2f} {v['mse']:>9.4f} | {s['psnr']:>10.2f} "
              f"{s['mse']:>9.4f} | {s['psnr'] - v['psnr']:>+7.2f}")

    report = {
        "checkpoint": args.checkpoint, "weights": args.weights,
        "scaling_factor": sf, "latent_scale_regime": regime,
        "l1_vs_vae_raw": d_raw, "l1_vs_vae_scaled": d_scaled,
        "n_images": len(samples),
        "vae_ladder": vae_ladder, "spnn_ladder": spnn_ladder,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
