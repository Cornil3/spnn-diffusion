#!/usr/bin/env python3
"""How far do clamping and 8-bit rounding push the codec off its own fixed point?

For x on the codec's manifold (x = D(E(x0))), compare:
  clamp only      x_c = clamp(x)
  round only      x_r = round(x * 127.5) / 127.5      (in-range part; what a file would store)
  clamp + round   both, i.e. what saving a PNG does
Report the latent displacement ||E(x') - E(x)|| / ||E(x)|| and the PSNR of one re-projection,
which is what compounds over iterations."""
import os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, "/home/ron.libman/churches-reboot")
import numpy as np, torch
from pathlib import Path
from PIL import Image
from spnn_model_opt import SPNNAutoencoder512Opt
D = Path("/rg/shocher_prj/ron.libman/reed_table1/data")
dev = "cuda"
X = torch.stack([torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), np.float32) / 127.5 - 1)
                 for p in sorted(D.glob("*_src.png"))[:24]]).permute(0, 3, 1, 2).to(dev)
ck = torch.load("/rg/shocher_prj/ron.libman/laion1m/runs/spnn512/best.pt", map_location="cpu", weights_only=False)
net = SPNNAutoencoder512Opt(); net.load_state_dict(ck["model"])
sd = net.state_dict(); sd.update({k: v.to(sd[k].dtype) for k, v in ck["ema"].items()}); net.load_state_dict(sd)
net = net.to(dev).eval()
rnd = lambda t: (t * 127.5).round() / 127.5
with torch.no_grad():
    x = net.decode(net.encode(X)).float()          # on the manifold
    z = net.encode(x)
    print(f"on-manifold decode: range [{x.min():.2f}, {x.max():.2f}], {100*(x.abs()>1).float().mean():.2f}% out of range\n")
    print(f"  {'perturbation':16s} | {'latent shift':>12s} | {'PSNR(x, x_perturbed)':>20s} | {'PSNR after 5 more round trips':>30s}")
    for name, f in (("none", lambda t: t), ("clamp", lambda t: t.clamp(-1, 1)),
                    ("round 8-bit", rnd), ("clamp + round", lambda t: rnd(t.clamp(-1, 1)))):
        xp = f(x)
        shift = float((net.encode(xp) - z).norm() / z.norm())
        p0 = float((10 * torch.log10(4.0 / ((xp.clamp(-1, 1) - x.clamp(-1, 1)) ** 2).mean((1, 2, 3)))).mean())
        cur = xp
        for _ in range(5):
            cur = f(net.decode(net.encode(cur)).float())
        p5 = float((10 * torch.log10(4.0 / ((cur.clamp(-1, 1) - x.clamp(-1, 1)) ** 2).mean((1, 2, 3)))).mean())
        print(f"  {name:16s} | {shift:12.2e} | {p0:20.2f} | {p5:30.2f}")
