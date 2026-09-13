#!/usr/bin/env python3
"""Rebuild REED-VAE's 179-sample ImagenHub set from the cached ImagenHub parquet files.
Writes {img_id}_t{turn}_{src,tgt,mask}.png at 512x512 (the paper's resolution) and verifies every
row against imagenhub_edit_instructions.csv, which is the split the paper's Table 1 is computed on.

The masks ship as RGBA and the region to repaint lives in the ALPHA channel, not in RGB:
ImagenHub's `imagen_hub.utils.image_helper` does

    alpha_channel = np.array(image_rgba)[:, :, 3]
    image_bw      = (alpha_channel != 255).astype(np.uint8)

Reading luminance instead (what this script used to do, via `.convert("RGB")`) produces a mask
that agrees with the real one on only 47% of pixels and covers 37% of the frame instead of 16%,
which silently breaks PbE and SD Inpainting. Source and target keep their original BICUBIC
downsample so the IP2P / MagicBrush rollouts built against them stay comparable."""
import glob, io, sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
from PIL import Image

CSV = Path("/home/ron.libman/churches-reboot/imagenhub_edit_instructions (2).csv")
OUT = Path("/rg/shocher_prj/ron.libman/reed_table1/data"); OUT.mkdir(parents=True, exist_ok=True)
SIZE = 512
MASKS_ONLY = "--masks-only" in sys.argv
csv = pd.read_csv(CSV, index_col=0)
csv["key"] = list(zip(csv.img_id.astype(str), csv.turn_index.astype(int)))
rows = {}
for f in sorted(glob.glob("/home/ron.libman/.cache/huggingface/hub/datasets--ImagenHub--Mask_Guided_Image_Editing/snapshots/*/data/*.parquet")):
    for r in pq.read_table(f).to_pylist():
        rows[(str(r["img_id"]), int(r["turn_index"]))] = r
missing = [k for k in csv.key if k not in rows]
assert not missing, f"{len(missing)} CSV rows absent from the parquet files: {missing[:5]}"

def save(b, path):
    im = Image.open(io.BytesIO(b)).convert("RGB")
    if im.size != (SIZE, SIZE):
        im = im.resize((SIZE, SIZE), Image.BICUBIC)   # ImagenHub ships 512; resize only if not
    im.save(path)

def save_mask(b, path):
    """ImagenHub's convention: transparent (alpha != 255) marks the region to repaint.
    Binarise from alpha, LANCZOS to 512 as ImagenHub's benchmark/mask_guided_ie.py does,
    then re-binarise so the stored PNG is strictly 0/255 -- white = repaint here."""
    im = Image.open(io.BytesIO(b))
    if im.mode != "RGBA":
        raise ValueError(f"{path.name}: expected RGBA mask, got {im.mode}")
    bw = (np.asarray(im)[:, :, 3] != 255).astype(np.uint8) * 255
    m = Image.fromarray(bw, mode="L")
    if m.size != (SIZE, SIZE):
        m = m.resize((SIZE, SIZE), Image.LANCZOS)
    Image.fromarray(((np.asarray(m) > 127).astype(np.uint8) * 255), mode="L").save(path)

n_instr_mismatch, areas = 0, []
for _, c in csv.iterrows():
    r = rows[c.key]
    if str(r["instruction"]).strip() != str(c.instruction).strip():
        n_instr_mismatch += 1
    stem = f"{c.img_id}_t{c.turn_index}"
    if not MASKS_ONLY:
        save(r["source_img"]["bytes"], OUT / f"{stem}_src.png")
        save(r["target_img"]["bytes"], OUT / f"{stem}_tgt.png")
    save_mask(r["mask_img"]["bytes"], OUT / f"{stem}_mask.png")
    areas.append((np.asarray(Image.open(OUT / f"{stem}_mask.png")) > 127).mean())
print(f"wrote {(1 if MASKS_ONLY else 3) * len(csv)} PNGs for {len(csv)} samples -> {OUT}")
print(f"mask area: mean {np.mean(areas):.4f}  min {np.min(areas):.4f}  max {np.max(areas):.4f}")
print(f"instruction text differs from the CSV on {n_instr_mismatch}/{len(csv)} rows "
      f"(the CSV is the paper's revised split, so small differences are expected)")
