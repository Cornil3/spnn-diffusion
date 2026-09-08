"""Does a FIXED seed across iterations explain our k=5 gap vs the paper?

ImagenHub's infer_one_image defaults to seed=42. If REED looped over it, every edit
reused that seed; EulerAncestral is stochastic, so identical noise each step makes the
edit chain far more self-consistent early on. We instead vary the seed per iteration.

Runs IP2P for 5 iterations under both regimes on the same images and reports the
paper's k=5 metrics (x^5 vs x^1).

  paper vanilla IP2P @k=5 : MSE 0.02  PSNR 17.78  LPIPS 0.33  SSIM 0.60
  ours  (varying seed)    : MSE 0.038 PSNR 15.67  LPIPS 0.40  SSIM 0.55
"""
import os, sys
sys.path.insert(0, os.getcwd())
import argparse
import numpy as np
import torch

from reed_repro.data import load_samples, pil_to_tensor01
from reed_repro.editors import EDITORS
from reed_repro.metrics import PairwiseMetrics
from reed_repro.run_repro import seed_for

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="ip2p")
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--iters", type=int, default=5)
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
S = load_samples(limit=a.n)
ed = EDITORS[a.model](codec="vae", device=dev, dtype=torch.float32)
pair = PairwiseMetrics(device=dev)

for regime in ("varying (ours)", "fixed 42 (ImagenHub default)"):
    acc = {k: [] for k in ("mse", "psnr", "lpips", "ssim")}
    for s in S:
        cur, x1 = s.source, None
        for it in range(1, a.iters + 1):
            seed = seed_for(s.key, it) if regime.startswith("varying") else 42
            cur = ed.edit(s, cur, it, seed)
            if cur.size != s.source.size:
                cur = cur.resize(s.source.size)
            if it == 1:
                x1 = cur
        m = pair(pil_to_tensor01(cur), pil_to_tensor01(x1))
        for k, v in m.items():
            acc[k].append(v)
    print(f"{regime:<32} n={len(S)}  "
          f"MSE={np.mean(acc['mse']):.4f}  PSNR={np.mean(acc['psnr']):.2f}  "
          f"LPIPS={np.mean(acc['lpips']):.3f}  SSIM={np.mean(acc['ssim']):.3f}", flush=True)

print(f"\npaper vanilla {a.model} @k=5      MSE=0.02    PSNR=17.78  LPIPS=0.33  SSIM=0.60")
print(f"our full run  {a.model} @k=5      MSE=0.038   PSNR=15.67  LPIPS=0.40  SSIM=0.55")
