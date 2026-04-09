"""
Codec wrappers for bridging latent <-> pixel space in latent DDNM.
"""

import sys
import os
import torch
from diffusers import AutoencoderKL, VQModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
#from models import SPNNAutoencoder  # existing SD1.5-aligned one
from functions.spnn_model import SPNNAutoencoder256  # the 256 version used in cycle_test.py

class VAECodec:
    def __init__(self, vae):
        self.vae = vae
        self.sf = vae.config.scaling_factor
    def encode(self, x):
        return self.vae.encode(x).latent_dist.mode() * self.sf
    def decode(self, z):
        return self.vae.decode(z / self.sf).sample


class VQVAECodec:
    """Wraps diffusers VQModel. Uses pre-quantization continuous latents
    (same convention you trained SPNN256 against)."""
    def __init__(self, vqvae, scaling_factor=1.0):
        self.vqvae = vqvae
        self.sf = scaling_factor  # CompVis LDM-celebahq uses sf=1.0
    def encode(self, x):
        # Pre-quantization continuous latents — matches your cycle_test.py
        return self.vqvae.encode(x).latents * self.sf
    def decode(self, z):
        return self.vqvae.decode(z / self.sf).sample


class SPNNCodec:
    def __init__(self, spnn, scaling_factor=1.0):
        self.spnn = spnn
        self.sf = scaling_factor
    def encode(self, x):
        return self.spnn.encode(x) * self.sf
    def decode(self, z):
        return self.spnn.decode(z / self.sf)

def load_codec_celebahq(config, device):
    """
    Load VQ-VAE from CompVis/ldm-celebahq-256 and SPNNAutoencoder256.
    Both use scaling_factor = 1.0 (SPNN256 was trained on raw VQ-VAE latents).
    """
    print("Loading CompVis VQ-VAE (ldm-celebahq-256)...")
    vqvae = VQModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="vqvae")
    vqvae.eval().to(device)
    for p in vqvae.parameters():
        p.requires_grad = False
    vqvae_codec = VQVAECodec(vqvae, scaling_factor=1.0)

    spnn_codec = None
    if getattr(config.codec, "use_spnn", True):
        print(f"Loading SPNNAutoencoder256 from {config.codec.spnn_checkpoint}...")
        spnn = SPNNAutoencoder256(
            mix_type=getattr(config.codec, "spnn_mix_type", "cayley"),
            scale_bound=getattr(config.codec, "spnn_scale_bound", 2.0),
        )
        ck = torch.load(config.codec.spnn_checkpoint, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck.get("model", ck)) if isinstance(ck, dict) else ck
        clean = {}
        for k, v in sd.items():
            nk = k
            for p in ("spnn.", "module.", "model."):
                if nk.startswith(p):
                    nk = nk[len(p):]
            clean[nk] = v
        spnn.load_state_dict(clean, strict=True)
        spnn.eval().to(device)
        spnn_codec = SPNNCodec(spnn, scaling_factor=1.0)

    return vqvae_codec, spnn_codec
