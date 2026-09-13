#!/usr/bin/env python3
"""Does SPNN's decoder do worse on diffusion-generated content than on natural photos?
Iterated clamped encode/decode (the editing chain's operation) on two image sets."""
import sys, glob, numpy as np, torch
from pathlib import Path
from PIL import Image
sys.path.insert(0, "/home/ron.libman/churches-reboot")
from diffusers import AutoencoderKL, StableDiffusionPipeline
from spnn_model_opt import SPNNAutoencoder512Opt
import lpips as lpips_lib

DEV, SC, N, ROUNDS = "cuda", 0.18215, 16, 8
VID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
ck = torch.load("/rg/shocher_prj/ron.libman/laion1m/runs/spnn512/best.pt", map_location="cpu", weights_only=False)
spnn = SPNNAutoencoder512Opt(); spnn.load_state_dict(ck["model"])
sd = spnn.state_dict(); sd.update({k: v.to(sd[k].dtype) for k, v in ck["ema"].items()}); spnn.load_state_dict(sd)
spnn = spnn.to(DEV).float().eval().requires_grad_(False)
vae = AutoencoderKL.from_pretrained(VID, subfolder="vae", local_files_only=True).to(DEV).float().eval()
lp = lpips_lib.LPIPS(net="alex", verbose=False).to(DEV).eval()

def load(paths):
    xs = [torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((512,512), Image.LANCZOS),
                                      np.float32)/127.5-1).permute(2,0,1) for p in paths]
    return torch.stack(xs).to(DEV)

# set A: natural photos (ImagenHub sources)
nat = load(sorted(glob.glob("/rg/shocher_prj/ron.libman/reed_table1/data/*_src.png"))[:N])
# set B: SD-1.5 generated images, same count
pipe = StableDiffusionPipeline.from_pretrained(VID, torch_dtype=torch.float16, safety_checker=None,
                                               requires_safety_checker=False, local_files_only=True).to(DEV)
pipe.set_progress_bar_config(disable=True)
PROMPTS = ["a photo of a living room","an oil painting of a harbour","a street in tokyo at night",
           "a bowl of fruit on a table","a mountain landscape","a portrait of an old man",
           "a red sports car","a forest path in autumn"]
g = torch.Generator(device=DEV).manual_seed(0)
gen = []
for i in range(N):
    im = pipe(PROMPTS[i % len(PROMPTS)], num_inference_steps=30, guidance_scale=7.5,
              generator=g, output_type="pt").images
    gen.append(im[0].float()*2-1)
gen = torch.stack(gen).to(DEV)
del pipe; torch.cuda.empty_cache()

codecs = {"SD-VAE": (lambda x: vae.encode(x).latent_dist.mean, lambda z: vae.decode(z).sample),
          "SPNN":   (lambda x: spnn.encode(x)/SC,               lambda z: spnn.decode(z*SC))}
psnr = lambda a,b: 10*np.log10(4.0/(a-b).pow(2).mean().item())
print(f"iterated clamped encode/decode, {N} images per set, {ROUNDS} rounds\n")
print(f"{'set':10s} {'codec':8s} {'1x PSNR':>9s} {'1x LPIPS':>9s} {'8x PSNR':>9s} {'8x LPIPS':>9s}")
for name, x0 in (("natural", nat), ("SD-gen", gen)):
    for cn, (enc, dec) in codecs.items():
        x = x0.clone(); rec = []
        with torch.no_grad():
            for k in range(ROUNDS):
                x = dec(enc(x)).clamp(-1, 1)             # the editing chain clamps every round
                if k in (0, ROUNDS-1):
                    rec.append((psnr(x, x0), float(lp(x, x0).mean())))
        print(f"{name:10s} {cn:8s} {rec[0][0]:9.2f} {rec[0][1]:9.4f} {rec[1][0]:9.2f} {rec[1][1]:9.4f}")
