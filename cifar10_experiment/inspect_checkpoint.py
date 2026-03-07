"""Inspect checkpoint keys and shapes to determine model architecture."""
import os, sys, torch

SLDM_ROOT = os.path.join(os.path.dirname(__file__), '..',
    'simple-latent-diffusion-model-master', 'simple-latent-diffusion-model')
models_dir = os.path.join(SLDM_ROOT, 'models')

for name in ['cifar_vae.pth', 'cifar_diffusion.pth']:
    path = os.path.join(models_dir, name)
    if not os.path.exists(path):
        print(f"NOT FOUND: {path}")
        continue
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    print(f"Type: {type(ckpt)}")
    if isinstance(ckpt, dict):
        print(f"Top-level keys: {list(ckpt.keys())}")
        for k, v in ckpt.items():
            if isinstance(v, (int, float, str, bool)):
                print(f"  {k}: {v}")
            elif isinstance(v, dict):
                print(f"  {k}: dict with {len(v)} keys")
                # Print all keys with shapes for state dicts
                for sk, sv in sorted(v.items()):
                    if hasattr(sv, 'shape'):
                        print(f"    {sk}: {list(sv.shape)}")
                    else:
                        print(f"    {sk}: {type(sv).__name__}")
