"""Why is DiffEdit producing nonsense? Look at the mask it generates.

DiffEdit derives its edit mask by contrasting noise predictions under the source and
target prompts. REED itself notes this "introduces ambiguity regarding the edit
location". Our caption pairs are near-identical whole-scene descriptions, so the
contrast may be almost nil -> the mask is noise -> it inpaints arbitrary regions.

Prints mask coverage and how well it agrees with ImagenHub's ground-truth mask.
"""
import os, sys
# Run as a plain script, Python puts THIS file's dir on sys.path, not the project
# root, so `reed_repro` would not import. Add cwd explicitly.
sys.path.insert(0, os.getcwd())

import numpy as np, torch
from PIL import Image
from reed_repro.data import load_samples
from reed_repro.editors import DiffEditEditor
from reed_repro.report import build_grid

dev = "cuda" if torch.cuda.is_available() else "cpu"
S = {s.key: s for s in load_samples(limit=6)}
ed = DiffEditEditor(codec="vae", device=dev, dtype=torch.float32)
print(f"DiffEdit base weight: {ed.weight}\n")

rows, labels = [], []
for key in ("100081_t1", "102171_t1"):
    s = S[key]
    src, tgt = s.source_caption, s.target_caption
    print(f"{key}")
    print(f"  C_s = {src!r}")
    print(f"  C_t = {tgt!r}")
    # word-level overlap between the two prompts
    a, b = set(src.lower().split()), set(tgt.lower().split())
    print(f"  prompt word overlap: {len(a & b)}/{len(a | b)} = {len(a&b)/max(len(a|b),1):.2f}")
    torch.manual_seed(42)
    m = ed.pipe.generate_mask(image=s.source.convert("RGB"),
                              source_prompt=src, target_prompt=tgt)
    m = np.asarray(m).squeeze()
    mb = m > 0.5
    print(f"  DiffEdit mask: shape={m.shape} coverage={mb.mean():.3f}")
    gt = np.array(s.mask.resize((mb.shape[-1], mb.shape[-2]), Image.NEAREST)) > 127
    inter = (mb & gt).sum(); union = (mb | gt).sum()
    print(f"  GT mask coverage={gt.mean():.3f}  IoU with DiffEdit mask={inter/max(union,1):.3f}")
    print(f"  -> {'mask is diffuse/uninformative' if mb.mean() > 0.5 else 'mask is localised'}\n")
    vis = Image.fromarray((mb * 255).astype(np.uint8)).resize((512, 512), Image.NEAREST)
    rows.append([s.source, s.mask, vis]); labels.append(key)

g = build_grid(rows, labels, ["source", "ImagenHub mask", "DiffEdit self-generated mask"], cell=190)
g.save("reed_repro/results/sanity/diffedit_mask_probe.png")
print("wrote reed_repro/results/sanity/diffedit_mask_probe.png")
