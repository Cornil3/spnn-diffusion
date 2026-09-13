"""
Pull every HF asset the sweep needs into the local cache.

Run this on athena's *login* node (compute nodes are often network-isolated, and the
existing athena slurm scripts already set HF_HUB_OFFLINE=1 for exactly that reason).

Note on SD 1.5: `runwayml/stable-diffusion-v1-5` was removed from the Hub, which is
why the older scripts in athena_scripts/ rely on a warm cache. The community mirror
`stable-diffusion-v1-5/stable-diffusion-v1-5` is live and is what this fetches.
"""

from huggingface_hub import snapshot_download

import os

MODELS = [
    os.environ.get("REED_SD15_ID", "stable-diffusion-v1-5/stable-diffusion-v1-5"),
    os.environ.get("REED_SD15_INPAINT_ID", "stable-diffusion-v1-5/stable-diffusion-inpainting"),
    os.environ.get("REED_IP2P_ID", "timbrooks/instruct-pix2pix"),
    os.environ.get("REED_MAGICBRUSH_ID", "vinesmsuic/magicbrush-jul7"),
    os.environ.get("REED_PBE_ID", "Fantasy-Studio/Paint-by-Example"),
]

if __name__ == "__main__":
    for m in MODELS:
        print(f"--> {m}", flush=True)
        try:
            snapshot_download(m, ignore_patterns=["*.ckpt", "*.onnx", "*.msgpack", "*.h5"])
        except Exception as e:
            print(f"    FAILED: {e!r}", flush=True)

    print("--> ImagenHub/Mask_Guided_Image_Editing (filtered split, 179 samples)")
    from datasets import load_dataset
    ds = load_dataset("ImagenHub/Mask_Guided_Image_Editing", split="filtered")
    print(f"    {len(ds)} rows, columns: {ds.column_names}")
    assert len(ds) == 179, f"expected the paper's 179 images, got {len(ds)}"

    print("--> LPIPS (alex) + pytorch-fid InceptionV3 weights")
    import lpips, torch
    lpips.LPIPS(net="alex")
    from pytorch_fid.inception import InceptionV3
    InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]])
    print("\nall assets cached.")
