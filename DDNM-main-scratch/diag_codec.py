"""
Diagnostic: measure each codec's raw encode->decode round-trip on real images,
with no diffusion. Isolates intrinsic codec fidelity (and whether SPNN-512 loaded
+ prescaled correctly) from the latent DDNM sampling loop.

Run inside the container from DDNM-main-scartch/:
    python diag_codec.py --config sd15_inpaint_latent.yml
"""

import os
import sys
import argparse
import yaml
import torch
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from functions.latent_codec import load_vae, load_spnn
from guided_diffusion.latent_diffusion import _PlainImageDataset, _to01


def dict2namespace(d):
    ns = argparse.Namespace()
    for k, v in d.items():
        setattr(ns, k, dict2namespace(v) if isinstance(v, dict) else v)
    return ns


def psnr(a, b):
    mse = torch.mean((a - b) ** 2)
    return (10 * torch.log10(1.0 / mse)).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="sd15_inpaint_latent.yml")
    args = p.parse_args()
    cfg = dict2namespace(yaml.safe_load(open(os.path.join("configs", args.config))))
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    vae = load_vae(dev)
    spnn = load_spnn(cfg, dev)

    size = int(cfg.data.image_size)
    tf = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])
    ds = _PlainImageDataset(cfg.data.dataset_root, tf)
    print(f"\n{len(ds)} images @ {size}x{size}\n")

    accv = accs = 0.0
    accv_x = accs_x = 0.0  # cross: decode VAE-latent through... n/a; keep simple
    for i in range(len(ds)):
        x, _ = ds[i]
        x = x.unsqueeze(0).to(dev) * 2 - 1  # [-1,1]
        zv = vae.encode(x); xv = vae.decode(zv)
        zs = spnn.encode(x); xs = spnn.decode(zs)
        pv, ps = psnr(_to01(x), _to01(xv)), psnr(_to01(x), _to01(xs))
        accv += pv; accs += ps
        print(f"img{i}: VAE rt {pv:5.2f} dB (z mean {zv.mean():+.3f} std {zv.std():.3f}) | "
              f"SPNN rt {ps:5.2f} dB (z mean {zs.mean():+.3f} std {zs.std():.3f})")

    n = len(ds)
    print(f"\nAVG encode->decode round-trip:  VAE {accv/n:.2f} dB  |  SPNN {accs/n:.2f} dB")
    print("Interpretation:")
    print("  * SPNN should be >> VAE here (that's the thesis). If they're ~equal/low,")
    print("    the SPNN checkpoint/EMA/prescaling is off.")
    print("  * Scaled-SD latents have std ~0.9-1.1. If SPNN z-std is far from VAE z-std,")
    print("    SPNN is NOT in SD's scaled space -> set codec.spnn_external_scale.")


if __name__ == "__main__":
    main()
