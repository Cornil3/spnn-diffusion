"""Is SD's 0.18215 scaling factor baked into SPNN-512's learned weights?

Rather than inferring it from an L1 comparison, fit the scalar alpha that best explains
    spnn.encode(x)  ~=  alpha * vae.encode(x).mode()      [the RAW, unscaled VAE latent]

  alpha ~= 0.18215 -> the factor IS baked in (encoder emits SD-scale latents)
  alpha ~= 1.0     -> it is NOT (encoder emits raw-VAE-scale latents)

Also reports per-channel alpha (a single global factor should be near-constant across
the 4 channels), the R^2 of the fit, and -- the thing that actually matters for
correctness -- whether the shim hands the UNet the same tensor the real VAE path would.
"""
import os, sys, torch
sys.path.insert(0, os.getcwd())
from diffusers import AutoencoderKL
from reed_repro.codecs import SPNNVAEShim, load_spnn512
from reed_repro.data import load_samples, pil_to_tensor01

dev = "cuda" if torch.cuda.is_available() else "cpu"
CK = "imagenet_latent_ddnm/runs/spnn512_sd15_distill/ckpt_last.pt"
vae = AutoencoderKL.from_pretrained(os.environ.get("REED_SD15_ID"), subfolder="vae").to(dev).eval()
sf = float(vae.config.scaling_factor)
spnn = load_spnn512(CK, weights="ema", device=dev)

S = load_samples(limit=16)
x = (torch.stack([pil_to_tensor01(s.source) for s in S]).to(dev) * 2 - 1)

with torch.no_grad():
    z_spnn = spnn.encode(x)
    z_raw = vae.encode(x).latent_dist.mode()

# least squares alpha = <z_spnn, z_raw> / <z_raw, z_raw>
a = (z_spnn * z_raw).sum() / (z_raw * z_raw).sum()
resid = z_spnn - a * z_raw
r2 = 1 - (resid.var() / z_spnn.var())
print(f"config vae_scaling / SD scaling_factor : {sf}")
print(f"fitted global alpha                    : {a.item():.6f}")
print(f"  alpha / 0.18215                      : {a.item()/sf:.4f}   (1.0 => exactly baked in)")
print(f"  R^2 of z_spnn ~ alpha * z_raw        : {r2.item():.4f}")
print(f"  1/alpha                              : {1/a.item():.3f}")

per_ch = [((z_spnn[:, c] * z_raw[:, c]).sum() / (z_raw[:, c] ** 2).sum()).item() for c in range(4)]
print(f"per-channel alpha                      : {[f'{v:.4f}' for v in per_ch]}")

print(f"\nstd  z_spnn={z_spnn.std():.4f}  z_raw={z_raw.std():.4f}  z_raw*sf={(z_raw*sf).std():.4f}")

# What the UNet actually receives, both paths. This is the correctness question.
shim = SPNNVAEShim(spnn, vae.config, latent_scale="prescaled", scaling_factor=sf)
unet_in_spnn = shim.encode(x).latent_dist.mode() * sf     # pipeline does the *sf itself
unet_in_vae = vae.encode(x).latent_dist.mode() * sf
print(f"\nUNet input via SPNN shim : std={unet_in_spnn.std():.4f} absmax={unet_in_spnn.abs().max():.3f}")
print(f"UNet input via real VAE  : std={unet_in_vae.std():.4f} absmax={unet_in_vae.abs().max():.3f}")
print(f"  mean |difference|      : {(unet_in_spnn-unet_in_vae).abs().mean():.5f}")
print(f"  ratio of stds          : {(unet_in_spnn.std()/unet_in_vae.std()).item():.4f}  (want ~1.0)")

# And the wrong choice, for contrast
wrong = SPNNVAEShim(spnn, vae.config, latent_scale="raw", scaling_factor=sf)
w = wrong.encode(x).latent_dist.mode() * sf
print(f"\nIf we had chosen 'raw'   : std={w.std():.4f} (off by {(w.std()/unet_in_vae.std()).item():.2f}x)")
