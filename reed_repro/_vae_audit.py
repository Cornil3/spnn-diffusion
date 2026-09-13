"""Do all five editing models share SD 1.5's VAE, or does each have its own?

SPNN-512 was distilled against SD 1.5's KL-VAE, so swapping it into a pipeline is only
sound if that pipeline's UNet expects latents in the same space. Two ways that can fail:

  1. a different `scaling_factor` -> latents off by a constant factor
  2. different VAE *weights* -> same nominal scale, but a different latent basis, which
     is the subtler and more damaging case (nothing would look obviously broken)

For each pipeline we report the scaling factor, the max weight deviation from SD 1.5's
VAE, and the latent statistics produced on identical images.
"""
import os, sys, torch
sys.path.insert(0, os.getcwd())
from diffusers import AutoencoderKL
from reed_repro.codecs import load_spnn512
from reed_repro.data import load_samples, pil_to_tensor01

dev = "cuda" if torch.cuda.is_available() else "cpu"
SD15 = os.environ.get("REED_SD15_ID", "runwayml/stable-diffusion-v1-5")
REPOS = [
    ("SD1.5 (reference)", SD15),
    ("IP2P",              os.environ.get("REED_IP2P_ID", "timbrooks/instruct-pix2pix")),
    ("MagicBrush",        os.environ.get("REED_MAGICBRUSH_ID", "vinesmsuic/magicbrush-jul7")),
    ("PbE",               os.environ.get("REED_PBE_ID", "Fantasy-Studio/Paint-by-Example")),
    ("SD Inpainting",     os.environ.get("REED_SD15_INPAINT_ID",
                                         "stable-diffusion-v1-5/stable-diffusion-inpainting")),
]

S = load_samples(limit=8)
x = (torch.stack([pil_to_tensor01(s.source) for s in S]).to(dev) * 2 - 1)

def psnr(a, b):
    m = torch.mean(((a.clamp(-1, 1) + 1) / 2 - (b + 1) / 2) ** 2).item()
    return 10 * torch.log10(torch.tensor(1.0 / max(m, 1e-12))).item()

ref_sd, rows = None, []
for name, repo in REPOS:
    try:
        vae = AutoencoderKL.from_pretrained(repo, subfolder="vae").to(dev).eval()
    except Exception as e:
        rows.append((name, repo, None, None, None, None, f"LOAD FAILED: {e!r}"))
        continue
    sf = float(vae.config.scaling_factor)
    sd = {k: v.float().cpu() for k, v in vae.state_dict().items()}
    if ref_sd is None:
        ref_sd, dev_max, note = sd, 0.0, "reference"
    else:
        if set(sd) != set(ref_sd):
            dev_max, note = float("nan"), "DIFFERENT KEYS"
        else:
            dev_max = max((sd[k] - ref_sd[k]).abs().max().item() for k in sd)
            note = "identical weights" if dev_max < 1e-6 else "WEIGHTS DIFFER"
    with torch.no_grad():
        z = vae.encode(x).latent_dist.mode()
        rec = vae.decode(z).sample
    rows.append((name, repo, sf, dev_max, (z * sf).std().item(), psnr(rec, x), note))
    del vae, sd
    if dev == "cuda":
        torch.cuda.empty_cache()

print(f"\n{'model':<20}{'scaling_f':>10}{'max|dW| vs SD1.5':>19}{'scaled z std':>14}{'recon PSNR':>12}  note")
print("-" * 100)
for name, repo, sf, dw, zs, ps, note in rows:
    sfs = "  n/a" if sf is None else f"{sf:.5f}"
    dws = "    n/a" if dw is None else f"{dw:.3e}"
    zss = "   n/a" if zs is None else f"{zs:.4f}"
    pss = "   n/a" if ps is None else f"{ps:.2f}"
    print(f"{name:<20}{sfs:>10}{dws:>19}{zss:>14}{pss:>12}  {note}")

sfs = [r[2] for r in rows if r[2] is not None]
print(f"\nall scaling factors identical : {len(set(sfs)) == 1}  -> {sorted(set(sfs))}")
same_w = all(r[3] is not None and r[3] < 1e-6 for r in rows[1:] if r[2] is not None)
print(f"all VAE weights identical     : {same_w}")

spnn = load_spnn512("imagenet_latent_ddnm/runs/spnn512_sd15_distill/ckpt_last.pt",
                    weights="ema", device=dev)
with torch.no_grad():
    zs_spnn = spnn.encode(x)
print(f"\nSPNN-512 native latent std    : {zs_spnn.std():.4f}")
print("(compare with the 'scaled z std' column: SPNN must match every pipeline it "
      "is swapped into)")
if len(set(sfs)) == 1 and same_w:
    print("\nVERDICT: one shared SD1.5 VAE across all five models -> a single SPNN "
          "codec is valid for every arm.")
else:
    print("\nVERDICT: the models do NOT share a VAE -> SPNN cannot be dropped into "
          "the mismatched ones without rescaling or redistillation.")
