"""Is SPNN's flat ladder a genuine fixed point, or an artefact?

Checks, over 8 ImagenHub images:
  1. consecutive-iteration distance  ||x^k - x^{k+1}||  -> ~0 means true fixed point
  2. per-image PSNR spread           -> a constant/degenerate decoder would collapse this
  3. does one pass actually change the image (x^1 != x^0)?
"""
import os, sys, torch
sys.path.insert(0, os.getcwd())
from reed_repro.codecs import SPNNVAEShim, load_spnn512
from reed_repro.data import load_samples, pil_to_tensor01
from diffusers import AutoencoderKL

dev = "cuda"
CK = "imagenet_latent_ddnm/runs/spnn512_sd15_distill/ckpt_last.pt"
vae = AutoencoderKL.from_pretrained(os.environ.get("REED_SD15_ID"), subfolder="vae").to(dev).eval()
spnn = load_spnn512(CK, weights="ema", device=dev)
shim = SPNNVAEShim(spnn, vae.config, latent_scale="prescaled",
                   scaling_factor=float(vae.config.scaling_factor))

S = load_samples(limit=8)
x = (torch.stack([pil_to_tensor01(s.source) for s in S]).to(dev) * 2 - 1)

def psnr01(a, b):
    m = torch.mean(((a.clamp(-1,1)+1)/2 - (b.clamp(-1,1)+1)/2) ** 2, dim=(1,2,3))
    return 10 * torch.log10(1.0 / m.clamp_min(1e-12))

with torch.no_grad():
    cur = x.clone()
    prev = None
    print(f"{'k':>3} {'PSNR vs x0 (per-image)':>46} {'mean':>7} {'d(x_k,x_k-1)':>13}")
    for k in range(1, 26):
        prev = cur
        cur = shim.decode(shim.encode(cur).latent_dist.mode()).sample
        if k in (1, 2, 3, 5, 15, 25):
            p = psnr01(cur, x)
            step = torch.mean((cur - prev) ** 2).item()
            spread = f"[{p.min():.2f}..{p.max():.2f}]"
            print(f"{k:>3} {' '.join(f'{v:.2f}' for v in p.tolist()):>46} {p.mean():>7.2f} {step:>13.3e}")
    # did the first pass change anything at all?
    d0 = torch.mean((shim.decode(shim.encode(x).latent_dist.mode()).sample - x) ** 2).item()
    print(f"\nMSE(x^1, x^0) = {d0:.3e}   <- nonzero => the codec is not an identity passthrough")
    print(f"per-image PSNR spread at k=25: {psnr01(cur, x).std():.3f} dB "
          f"<- nonzero => not collapsing to a constant image")
