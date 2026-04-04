"""
Codec wrappers for bridging latent <-> pixel space in latent DDNM.
"""

import sys
import os
import torch
from diffusers import AutoencoderKL

# Add parent dir so we can import SPNNAutoencoder from the main project
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from models import SPNNAutoencoder


class VAECodec:
    """Wraps SD VAE with scaling factor handling."""

    def __init__(self, vae):
        self.vae = vae
        self.sf = vae.config.scaling_factor

    def encode(self, x):
        """Pixel [-1,1] -> scaled latent."""
        return self.vae.encode(x).latent_dist.mode() * self.sf

    def decode(self, z):
        """Scaled latent -> pixel [-1,1]."""
        return self.vae.decode(z / self.sf).sample


class SPNNCodec:
    """Wraps SPNN autoencoder with VAE-compatible scaling factor."""

    def __init__(self, spnn, scaling_factor):
        self.spnn = spnn
        self.sf = scaling_factor

    def encode(self, x):
        """Pixel [-1,1] -> scaled latent."""
        return self.spnn.encode(x) * self.sf

    def decode(self, z):
        """Scaled latent -> pixel [-1,1]."""
        return self.spnn.decode(z / self.sf)


def load_codec(config, device):
    """Load VAE and optionally SPNN codec from config."""
    sd_id = "runwayml/stable-diffusion-v1-5"

    # Always load VAE (needed for scaling factor even with SPNN)
    print("Loading SD 1.5 VAE...")
    vae = AutoencoderKL.from_pretrained(sd_id, subfolder="vae")
    vae.eval().to(device)
    for p in vae.parameters():
        p.requires_grad = False

    vae_codec = VAECodec(vae)

    # Override scaling factor if CompVis LDM (different from SD 1.5's 0.18215)
    sf_override = getattr(config.codec, 'compvis_scale_factor', None)
    if sf_override is not None:
        print(f"Overriding scaling factor: {vae_codec.sf} -> {sf_override}")
        vae_codec.sf = sf_override

    codec_type = config.codec.type if hasattr(config, 'codec') else "vae"

    if codec_type == "spnn":
        print(f"Loading SPNN from {config.codec.spnn_checkpoint}...")
        spnn = SPNNAutoencoder(
            mix_type=config.codec.spnn_mix_type,
            hidden=config.codec.spnn_hidden,
            r_hidden=config.codec.spnn_hidden,
            scale_bound=config.codec.spnn_scale_bound,
        )
        state = torch.load(config.codec.spnn_checkpoint, map_location=device, weights_only=True)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        spnn.load_state_dict(state)
        spnn.eval().to(device)
        scaling_factor = sf_override if sf_override is not None else vae.config.scaling_factor
        spnn_codec = SPNNCodec(spnn, scaling_factor)
        return vae_codec, spnn_codec
    else:
        return vae_codec, None
