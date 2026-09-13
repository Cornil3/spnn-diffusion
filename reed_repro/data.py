"""
ImagenHub `filtered` split (179 samples) — the exact set REED-VAE's Table 1 is computed on.

`dataset_info.json` / `state.json` pin this down: ImagenHub/Mask_Guided_Image_Editing,
split `filtered`, `num_examples` = 179 — the paper's "179 images".

Two columns are NOT in the public dataset
-----------------------------------------
The live hub revision of `filtered` has 9 columns:

    img_id, turn_index, source_img, mask_img, target_img,
    instruction, source_global_caption, target_global_caption, target_local_caption

`dataset_info.json` lists 11, adding `reverse_instruction` and `processed_mask`. That
file (together with `state.json`, which is a `save_to_disk` state) describes a locally
*augmented* copy, not the hub's. `reverse_instruction` is REED's own manual addition —
the paper: "we manually add 'reverse prompts' to perform each given edit in the
opposite direction... We will make our full, revised dataset available with our code."
That release never happened.

So `reverse_instruction` comes from `imagenhub_edit_instructions.csv`, which was
verified to cover exactly the same 179 (img_id, turn_index) keys with a non-empty
reverse prompt on every row. Without it, IP2P and MagicBrush cannot alternate edit
direction and the whole protocol collapses. `processed_mask` is optional; we fall back
to `mask_img`.

Pass `from_disk=<path>` to use the augmented saved dataset directly if you have it.

Geometry
--------
Every image in the split is square (source 500x500 or 1024x1024; mask and target
always 1024x1024). Aspect ratios therefore always agree — checked across all 179 — so
resizing image and mask independently to 512x512 keeps them aligned, and `resize` vs
`center_crop` is a distinction without a difference on this data.
"""

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from PIL import Image

HF_DATASET = "ImagenHub/Mask_Guided_Image_Editing"
HF_SPLIT = "filtered"
RESOLUTION = 512


# ---------------------------------------------------------------------------
# Geometry: images and masks must go through *identical* spatial transforms or
# the mask stops lining up with the pixels it is supposed to select.
# ---------------------------------------------------------------------------

def _to_square(img: Image.Image, size: int, mode: str, resample) -> Image.Image:
    if mode == "resize":
        # Squash to size x size. Distorts aspect ratio but keeps the whole frame,
        # so a mask near an edge can never be cropped away.
        return img.resize((size, size), resample)
    if mode == "center_crop":
        w, h = img.size
        s = size / min(w, h)
        img = img.resize((max(size, int(round(w * s))), max(size, int(round(h * s)))), resample)
        w, h = img.size
        left, top = (w - size) // 2, (h - size) // 2
        return img.crop((left, top, left + size, top + size))
    raise ValueError(f"unknown resize mode: {mode}")


def prep_image(img: Image.Image, size: int = RESOLUTION, mode: str = "resize") -> Image.Image:
    # ImagenHub uses LANCZOS at 512 (benchmark/mask_guided_ie.py).
    return _to_square(img.convert("RGB"), size, mode, Image.LANCZOS)


def rgba_to_01_mask(image_rgba, reverse: bool = False):
    """ImagenHub's mask convention, verbatim from `imagen_hub.utils.image_helper`.

        alpha_channel = np.array(image_rgba)[:, :, 3]
        image_bw = (alpha_channel != 255).astype(np.uint8)

    The mask lives in the **alpha channel** — transparent pixels mark the region to
    edit — not in RGB luminance. Reading luminance instead (as an earlier version of
    this file did) gives a mask that agrees with the real one only 29% of the time and
    covers 70% of the frame instead of the true ~1-30%.
    """
    a = np.array(image_rgba)
    if a.ndim != 3 or a.shape[2] < 4:
        # Not RGBA (shouldn't happen for this split) — fall back to luminance.
        bw = (np.array(image_rgba.convert("L")) > 127).astype(np.uint8)
    else:
        bw = (a[:, :, 3] != 255).astype(np.uint8)
    if reverse:
        bw = 1 - bw
    return bw


def prep_mask(img: Image.Image, size: int = RESOLUTION, mode: str = "resize",
              open_r: int = 0, close_r: int = 0) -> Image.Image:
    """Binary mask at `size`, white (255) = repaint here — what diffusers expects.

    Follows ImagenHub's own pipeline: take the alpha-channel mask, then resize. Their
    `benchmark/mask_guided_ie.py` resizes source, target and mask to 512 with LANCZOS,
    so we do the same; the mask is binarised after resizing.

    Morphological cleanup defaults to OFF now. It existed to fight speckle in what
    turned out to be the wrong channel; the real alpha masks are clean.
    """
    from PIL import ImageFilter
    bw = rgba_to_01_mask(img)
    m = Image.fromarray((bw * 255).astype(np.uint8), mode="L")
    m = _to_square(m, size, mode, Image.LANCZOS)
    m = Image.fromarray(((np.array(m) > 127).astype(np.uint8) * 255), mode="L")
    if open_r and open_r >= 3:
        m = m.filter(ImageFilter.MinFilter(open_r)).filter(ImageFilter.MaxFilter(open_r))
    if close_r and close_r >= 3:
        m = m.filter(ImageFilter.MaxFilter(close_r)).filter(ImageFilter.MinFilter(close_r))
    return m


def largest_component_bbox(mask: Image.Image, pad: int = 8, grid: int = 128) -> tuple:
    """Bounding box of the mask's largest connected component.

    PbE needs a reference image "containing the original object". Using the bbox of the
    *whole* mask gives a box covering >90% of the frame on 168 of the 179 samples -
    the reference would be the entire image and PbE would degenerate. Restricting to
    the largest component brings the median box down to ~42% of the frame.

    Components are found on a `grid`x`grid` downsample (adequate for a bounding box)
    and the result is scaled back up, which keeps this fast in pure numpy.
    """
    from collections import deque
    W, H = mask.size
    a = np.array(mask.resize((grid, grid), Image.NEAREST)) > 127
    if not a.any():
        return (0, 0, W, H)
    seen = np.zeros_like(a, bool)
    best = None
    for y in range(grid):
        for x in range(grid):
            if a[y, x] and not seen[y, x]:
                q = deque([(y, x)]); seen[y, x] = True; px = [(y, x)]
                while q:
                    cy, cx = q.popleft()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < grid and 0 <= nx < grid and a[ny, nx] and not seen[ny, nx]:
                                seen[ny, nx] = True; q.append((ny, nx)); px.append((ny, nx))
                if best is None or len(px) > len(best):
                    best = px
    ys = [p[0] for p in best]; xs = [p[1] for p in best]
    sx, sy = W / grid, H / grid
    x0, x1 = int(min(xs) * sx), int((max(xs) + 1) * sx)
    y0, y1 = int(min(ys) * sy), int((max(ys) + 1) * sy)
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(W, x1 + pad), min(H, y1 + pad)
    return (x0, y0, x1, y1)


def mask_bbox(mask: Image.Image, pad: int = 0) -> tuple:
    """Tight bounding box around the mask's positive region, as PbE's reference crop.

    The paper: "we apply the mask m to x_s and generate a tight bounding box around
    the mask. We use this generated bounding box to create x_r^2, a new reference
    image containing the original object from x_s".
    """
    a = np.array(mask) > 127
    if not a.any():
        w, h = mask.size
        return (0, 0, w, h)
    ys, xs = np.where(a)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    if pad:
        w, h = mask.size
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
    return (x0, y0, x1, y1)


# ---------------------------------------------------------------------------

@dataclass
class Sample:
    """One ImagenHub row, fully preprocessed and ready for every editor."""
    index: int
    img_id: str
    turn_index: int
    source: Image.Image          # x_s, 512x512 RGB
    target: Image.Image          # x_t, 512x512 RGB (ground-truth edit)
    mask: Image.Image            # m, 512x512 L, white = edit here
    instruction: str             # I_s  (forward, text-guided)
    reverse_instruction: str     # I_s^-1 (backward, text-guided)
    source_caption: str          # C_s
    target_caption: str          # C_t
    target_local_caption: str    # C_t^local
    ref_target: Image.Image      # x_r^1: tight mask-bbox crop of x_t (the target object)
    ref_source: Image.Image      # x_r^2: tight mask-bbox crop of x_s (the original object)

    @property
    def key(self) -> str:
        return f"{self.img_id}_t{self.turn_index}"


DEFAULT_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "imagenhub_edit_instructions.csv")


def load_reverse_instructions(csv_path: str = DEFAULT_CSV) -> dict:
    """(img_id, turn_index) -> reverse_instruction, from REED's augmented CSV."""
    import csv as _csv
    out = {}
    with open(csv_path, newline="") as f:
        for row in _csv.DictReader(f):
            rev = (row.get("reverse_instruction") or "").strip()
            out[(str(row["img_id"]), int(row["turn_index"]))] = rev
    return out


def load_samples(size: int = RESOLUTION, resize_mode: str = "resize",
                 limit: Optional[int] = None, cache_dir: Optional[str] = None,
                 from_disk: Optional[str] = None, csv_path: str = DEFAULT_CSV,
                 mask_open: int = 0, mask_close: int = 0):
    """Load and preprocess the 179-sample `filtered` split, in stable order."""
    if from_disk:
        from datasets import load_from_disk
        ds = load_from_disk(from_disk)
    else:
        from datasets import load_dataset
        ds = load_dataset(HF_DATASET, split=HF_SPLIT, cache_dir=cache_dir)

    columns = set(ds.column_names)
    # The hub copy lacks reverse_instruction; the CSV supplies it.
    reverse = {} if "reverse_instruction" in columns else load_reverse_instructions(csv_path)

    # Stable, reproducible ordering independent of shard layout. Sort over a
    # scalar-only view: indexing the full dataset decodes each row's three 1024px
    # PNGs, so ordering all 179 rows that way costs a minute of pointless decoding
    # even when --limit means we only need ten of them.
    try:
        keycols = ds.select_columns(["img_id", "turn_index"])
    except Exception:
        keycols = ds
    order = sorted(range(len(ds)),
                   key=lambda i: (str(keycols[i]["img_id"]), int(keycols[i]["turn_index"])))
    if limit is not None:
        order = order[:limit]

    samples, missing_rev = [], []
    for n, i in enumerate(order):
        row = ds[i]
        key = (str(row["img_id"]), int(row["turn_index"]))
        if "reverse_instruction" in columns:
            rev = (row["reverse_instruction"] or "").strip()
        else:
            rev = reverse.get(key, "").strip()
        if not rev:
            missing_rev.append(key)

        source = prep_image(row["source_img"], size, resize_mode)
        target = prep_image(row["target_img"], size, resize_mode)
        # If the augmented copy supplies processed_mask it is already clean; only the
        # raw mask_img needs our morphological reconstruction.
        have_processed = ("processed_mask" in columns
                          and row["processed_mask"] is not None)
        raw_mask = row["processed_mask"] if have_processed else row["mask_img"]
        mask = prep_mask(raw_mask, size, resize_mode,
                         open_r=mask_open, close_r=mask_close)

        # PbE reference crop: largest component only, not the full mask bbox.
        box = largest_component_bbox(mask, pad=8)
        samples.append(Sample(
            index=n,
            img_id=str(row["img_id"]),
            turn_index=int(row["turn_index"]),
            source=source, target=target, mask=mask,
            instruction=(row["instruction"] or "").strip(),
            reverse_instruction=rev,
            source_caption=(row["source_global_caption"] or "").strip(),
            target_caption=(row["target_global_caption"] or "").strip(),
            target_local_caption=(row["target_local_caption"] or "").strip(),
            ref_target=target.crop(box),
            ref_source=source.crop(box),
        ))

    if missing_rev:
        # Silently proceeding would make IP2P/MagicBrush repeat the forward edit
        # every iteration, which is a different experiment than the paper's.
        raise RuntimeError(
            f"{len(missing_rev)} samples have no reverse_instruction "
            f"(e.g. {missing_rev[:3]}). Check {csv_path}.")
    return samples


def save_png_atomic(img: Image.Image, path) -> None:
    """Write a PNG via a temp file + rename.

    A job killed mid-write (preemption, scancel, wall-clock) otherwise leaves a
    truncated PNG at the final path. Nothing downstream can tell that apart from a
    finished one, so it silently corrupts grids and metrics. os.replace is atomic
    within a filesystem, so a reader only ever sees a complete file or none.
    """
    path = str(path)
    tmp = f"{path}.tmp{os.getpid()}"
    img.save(tmp, format="PNG")
    os.replace(tmp, path)


def png_is_complete(path) -> bool:
    """Cheap truncation check: a finished PNG ends with the IEND chunk.

    Resume decides what to skip by asking whether a file exists -- but a truncated PNG
    exists too, so a half-written sample would be marked done and never regenerated.
    Fully decoding 4475 files per arm to find out is far too slow; the IEND trailer is
    a couple of bytes to read and catches exactly the truncation case.
    """
    try:
        if os.path.getsize(path) < 16:
            return False
        with open(path, "rb") as f:
            f.seek(-12, os.SEEK_END)
            return b"IEND" in f.read()
    except OSError:
        return False


def load_png_safe(path):
    """Open a PNG, returning None if it is missing or unreadable/truncated."""
    try:
        img = Image.open(path)
        img.load()          # force decode now, so truncation surfaces here
        return img
    except Exception:
        return None


def pil_to_tensor01(img: Image.Image) -> torch.Tensor:
    """PIL RGB -> float tensor [3,H,W] in [0,1] (the range the paper evaluates in)."""
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).contiguous()


def tensor01_to_pil(t: torch.Tensor) -> Image.Image:
    a = (t.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(a)
