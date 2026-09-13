#!/usr/bin/env python3
"""Final REED Table 1: paper / +REED / ours SD-1.5 VAE / ours SPNN / ours SD-VAE-enc + SPNN-dec.
All ours-rows are 179 ImagenHub samples, 25 alternating edits, scored against each chain's own x^1."""
import numpy as np, torch
from pathlib import Path
from PIL import Image
import lpips as lpips_lib
from skimage.metrics import structural_similarity as ssim
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance

FULL = Path("/rg/shocher_prj/ron.libman/reed_table1/out/rollouts")
HYB  = Path("/rg/shocher_prj/ron.libman/reed_table1/out_vaeenc_full/rollouts")
ITERS = (5, 15, 25)
dev = "cuda" if torch.cuda.is_available() else "cpu"
lp = lpips_lib.LPIPS(net="alex", verbose=False).to(dev).eval()
inc = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(dev).eval()
rgb = lambda p: np.asarray(Image.open(p).convert("RGB"), np.float32) / 255

@torch.no_grad()
def feats(arrs, bs=16):
    out = []
    for i in range(0, len(arrs), bs):
        x = torch.from_numpy(np.stack(arrs[i:i+bs])).permute(0, 3, 1, 2).to(dev)
        out.append(inc(x)[0].flatten(1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 2048))

def arm(root, editor, codec):
    d, out = root / editor / codec, {}
    for it in ITERS:
        di = d / f"iter{it}"
        if not di.exists(): continue
        pairs = []
        for p in sorted(di.glob("*.png")):
            x1 = d / "all" / f"{p.stem}_step01.jpg"
            if x1.exists(): pairs.append((rgb(p), rgb(x1)))
        if not pairs: continue
        a = [q[0] for q in pairs]; b = [q[1] for q in pairs]
        mse = [float(((u - v) ** 2).mean()) for u, v in pairs]
        psnr = [10 * np.log10(1 / max(m, 1e-10)) for m in mse]
        ss = [ssim(u, v, channel_axis=2, data_range=1.0) for u, v in pairs]
        t = lambda x: torch.from_numpy(np.stack(x)).permute(0, 3, 1, 2).to(dev) * 2 - 1
        l = []
        with torch.no_grad():
            for i in range(0, len(a), 16):
                l += lp(t(a[i:i+16]), t(b[i:i+16])).flatten().cpu().tolist()
        fa, fb = feats(a), feats(b)
        fid = float(calculate_frechet_distance(fa.mean(0), np.cov(fa, rowvar=False),
                                               fb.mean(0), np.cov(fb, rowvar=False))) if len(a) >= 20 else float("nan")
        out[it] = (np.mean(mse), np.mean(l), np.mean(ss), fid, np.mean(psnr), len(a))
    return out

PAPER = {
 "IP2P":        {"base":[(.02,.33,.60,105.8,17.78),(.11,.69,.23,247.0,10.15),(.15,.76,.18,271.7,8.59)],
                 "reed":[(.02,.18,.80,62.8,19.81),(.06,.45,.53,138.9,13.36),(.09,.58,.41,188.0,11.36)]},
 "MagicBrush":  {"base":[(.02,.31,.65,103.5,18.84),(.08,.71,.21,267.0,11.35),(.14,.80,.13,295.8,8.75)],
                 "reed":[(.01,.19,.81,74.7,21.53),(.03,.51,.60,174.7,16.66),(.05,.69,.45,223.8,14.09)]},
 "DiffEdit":    {"base":[(.03,.34,.65,160.7,15.99),(.06,.62,.36,246.3,12.59),(.08,.73,.21,301.9,11.21)],
                 "reed":[(.03,.30,.69,160.3,16.24),(.07,.55,.48,226.9,12.91),(.08,.68,.40,246.5,11.44)]},
 "PbE":         {"base":[(.02,.26,.66,83.5,18.55),(.04,.60,.33,209.3,13.88),(.07,.71,.22,253.6,11.74)],
                 "reed":[(.02,.20,.77,74.2,19.62),(.03,.44,.61,141.1,16.19),(.04,.59,.54,178.3,14.43)]},
 "SDInpaint":   {"base":[(.01,.29,.67,95.1,20.46),(.06,.69,.22,255.8,12.31),(.11,.78,.14,283.4,9.72)],
                 "reed":[(.01,.17,.80,72.7,23.14),(.03,.47,.57,166.1,16.66),(.05,.65,.41,210.4,13.61)]},
}
NAMES = {"IP2P":"IP2P [BHE23]","MagicBrush":"MagicBrush [ZMC*23]","DiffEdit":"DiffEdit [CVSC22]",
         "PbE":"PbE [YGZ*23]","SDInpaint":"SD Inpainting [RBL*22]"}
hdr = f"{'Method':30s}" + "".join(f"{m:>22s}" for m in ("MSE ↓","LPIPS ↓","SSIM ↑","FID ↓","PSNR ↑"))
sub = f"{'':30s}" + "".join(f"{n:>7s}" for _ in range(5) for n in ("5","15","25"))
print("179 ImagenHub samples · 25 alternating edits · scored against x¹ (REED-VAE Appendix A)\n")
print(hdr); print(sub); print("-"*len(sub))
def line(label, vals, n=""):
    cells = ""
    for j in range(5):
        for i in range(3):
            v = vals[i][j] if vals[i] is not None else None
            cells += (f"{v:7.3f}" if j == 0 else f"{v:7.2f}") if v is not None else f"{'':>7s}"
    print(f"{label:30s}{cells}  {n}")
for e in NAMES:
    line(NAMES[e], PAPER[e]["base"]); line("  + REED  (paper)", PAPER[e]["reed"])
    for root, codec, lbl in ((FULL,"vanilla","  SD-1.5 VAE  (ours)"),
                             (FULL,"spnn","  + SPNN  (ours)"),
                             (HYB,"hybrid","  + SD-VAE enc/SPNN dec")):
        s = arm(root, e, codec)
        vals = [(s[i][0], s[i][1], s[i][2], s[i][3], s[i][4]) if i in s else None for i in ITERS]
        n = "n=" + "/".join(str(s[i][5]) if i in s else "0" for i in ITERS)
        line(lbl, vals, n)
    print("-"*len(sub))
