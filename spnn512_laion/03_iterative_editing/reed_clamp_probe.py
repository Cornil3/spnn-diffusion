#!/usr/bin/env python3
"""Is the phase-2 codec still a fixed point? Iterate D(E(.)) with and without clamping to the
displayable range, and measure how far the decode leaves [-1,1]. An exact projection is flat."""
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
psnr = lambda a, b: float((10 * torch.log10(4.0 / ((a.clamp(-1, 1) - b) ** 2).mean((1, 2, 3)))).mean())
with torch.no_grad():
    z = net.encode(X); r = net.decode(z).float()
    print(f"one pass: PSNR {psnr(r, X):.2f} dB | decode range [{r.min():.2f}, {r.max():.2f}] | "
          f"pixels outside [-1,1]: {100 * (r.abs() > 1).float().mean():.1f}%")
    print(f"latent idempotency ||E(D(z))-z||/||z|| (fp32): {((net.encode(r) - z).norm() / z.norm()):.2e}")
    for mode in ("clamped", "unclamped"):
        cur = X.clone()
        out = []
        for it in range(1, 26):
            cur = net.decode(net.encode(cur)).float()
            if mode == "clamped":
                cur = cur.clamp(-1, 1)
            if it in (1, 2, 5, 15, 25):
                out.append(f"{it}: {psnr(cur, X):5.2f}")
        print(f"  chain {mode:10s} PSNR vs original at iters " + "  ".join(out))
