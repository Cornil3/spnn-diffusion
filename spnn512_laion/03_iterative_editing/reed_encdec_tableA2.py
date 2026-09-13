#!/usr/bin/env python3
"""REED-VAE Table A2 / Table 2 setting: iterative encode-decode with NO diffusion model, on the 179
ImagenHub images, metrics against the ORIGINAL image at 5/15/25 iterations.

Their numbers (Table A2, SD 2.1 family):   vanilla 26.0 / 19.30 / 14.83 dB,  + REED 30.5 / 24.10 / 20.93
We run the SD-1.5 VAE and our phase-2 SPNN. Both chain modes matter: `uint8` rounds to 8 bits between
iterations (what a real editing session does, and what knocks an idempotent codec off its fixed point),
`float` keeps the chain in float, where an exact projection is flat forever.
"""
import os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, "/home/ron.libman/churches-reboot")
import numpy as np, torch
from pathlib import Path
from PIL import Image
from diffusers import AutoencoderKL
from spnn_model_opt import SPNNAutoencoder512Opt
import lpips as lpips_lib
from skimage.metrics import structural_similarity as ssim
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance

D = Path("/rg/shocher_prj/ron.libman/reed_table1/data")
dev = "cuda"
SAVE_AT, N_ITERS = (5, 15, 25), 25
srcs = sorted(D.glob("*_src.png"))
X = torch.stack([torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), np.float32) / 255) for p in srcs]).permute(0, 3, 1, 2)
print(f"{len(X)} images")
vae = AutoencoderKL.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="vae").to(dev).eval()
ck = torch.load("/rg/shocher_prj/ron.libman/laion1m/runs/spnn512/best.pt", map_location="cpu", weights_only=False)
spnn = SPNNAutoencoder512Opt(); spnn.load_state_dict(ck["model"])
sd = spnn.state_dict(); sd.update({k: v.to(sd[k].dtype) for k, v in ck["ema"].items()}); spnn.load_state_dict(sd)
spnn = spnn.to(dev).eval()
print(f"SPNN step {ck['state']['step']} (epoch {ck['state']['step'] / 5426:.2f})")
rt = {"sd15-vae": lambda x: vae.decode(vae.encode(x).latent_dist.mode()).sample,
      "spnn": lambda x: spnn.decode(spnn.encode(x)).float()}
lp = lpips_lib.LPIPS(net="alex", verbose=False).to(dev).eval()
inc = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(dev).eval()
@torch.no_grad()
def feats(t, bs=16):
    return np.concatenate([inc(t[i:i+bs].to(dev))[0].flatten(1).cpu().numpy() for i in range(0, len(t), bs)])
F0 = feats(X)
print(f"\n  {'codec':9s} {'chain':6s} | " + " | ".join(f"{'@' + str(n):>26s}" for n in SAVE_AT))
print(f"  {'':16s} | " + " | ".join(f"{'MSE':>6s}{'LPIPS':>7s}{'SSIM':>6s}{'PSNR':>7s}" for _ in SAVE_AT))
for name, fn in rt.items():
    for mode in ("uint8", "float"):
        cur, out = X.clone(), {}
        for it in range(1, N_ITERS + 1):
            with torch.no_grad():
                for i in range(0, len(cur), 8):
                    cur[i:i+8] = fn(cur[i:i+8].to(dev) * 2 - 1).add(1).div(2).clamp(0, 1).cpu()
            if mode == "uint8":
                cur = (cur * 255).round() / 255
            if it in SAVE_AT:
                with torch.no_grad():
                    l = float(np.concatenate([lp(cur[i:i+16].to(dev) * 2 - 1, X[i:i+16].to(dev) * 2 - 1).flatten().cpu().numpy() for i in range(0, len(cur), 16)]).mean())
                mse = ((cur - X) ** 2).mean((1, 2, 3)).numpy()
                ss = np.mean([ssim(cur[i].permute(1, 2, 0).numpy(), X[i].permute(1, 2, 0).numpy(), channel_axis=2, data_range=1.0) for i in range(len(cur))])
                f = feats(cur)
                fid = calculate_frechet_distance(f.mean(0), np.cov(f, rowvar=False), F0.mean(0), np.cov(F0, rowvar=False))
                out[it] = (mse.mean(), l, ss, np.mean(10 * np.log10(1 / np.clip(mse, 1e-10, None))), fid)
        cells = " | ".join(f"{out[n][0]:6.4f}{out[n][1]:7.3f}{out[n][2]:6.2f}{out[n][3]:7.2f}" for n in SAVE_AT)
        print(f"  {name:9s} {mode:6s} | {cells}", flush=True)
        print(f"  {'':9s} {'FID':6s} | " + " | ".join(f"{out[n][4]:26.2f}" for n in SAVE_AT))
print("\n  paper Table A2 (SD 2.1 family, vs the original image):")
print("  vanilla SD2.1   | MSE .0031 LPIPS .19 SSIM .76 PSNR 26.00 | .013 .55 .49 19.30 | .034 .71 .26 14.83")
print("  + REED          | MSE .0011 LPIPS .075 SSIM .89 PSNR 30.50 | .0042 .18 .76 24.10 | .0086 .25 .68 20.93")
