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
    """Wraps a diffusers AutoencoderKL with scaling factor handling."""

    def __init__(self, vae, scaling_factor=None):
        self.vae = vae
        self.sf = scaling_factor if scaling_factor is not None else vae.config.scaling_factor

    def encode(self, x):
        """Pixel [-1,1] -> scaled latent."""
        return self.vae.encode(x).latent_dist.mode() * self.sf

    def decode(self, z):
        """Scaled latent -> pixel [-1,1]."""
        return self.vae.decode(z / self.sf).sample


class CompVisVAECodec:
    """Wraps a CompVis ldm AutoencoderKL with scaling factor handling."""

    def __init__(self, vae, scaling_factor):
        self.vae = vae
        self.sf = scaling_factor

    def encode(self, x):
        """Pixel [-1,1] -> scaled latent."""
        posterior = self.vae.encode(x)
        return posterior.mode() * self.sf

    def decode(self, z):
        """Scaled latent -> pixel [-1,1]."""
        return self.vae.decode(z / self.sf)


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


def _load_compvis_vae(ckpt_path, model_config_path, device, state_dict=None):
    """Extract the first_stage_model (KL-VAE) from a CompVis LDM .ckpt file."""
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config

    print(f"Loading CompVis VAE from {ckpt_path}...")
    print(f"  Using model config: {model_config_path}")
    if state_dict is not None:
        sd = state_dict
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    # Extract scale_factor (baked in by scale_by_std during training)
    sf_tensor = sd.get("scale_factor", None)
    if sf_tensor is not None:
        scale_factor = float(sf_tensor.item() if torch.is_tensor(sf_tensor) else sf_tensor)
    else:
        scale_factor = 1.0
    print(f"  CompVis scale_factor from checkpoint: {scale_factor}")

    # Extract first_stage_model weights
    vae_sd = {}
    for k, v in sd.items():
        if k.startswith("first_stage_model."):
            vae_sd[k.replace("first_stage_model.", "")] = v
    print(f"  Extracted {len(vae_sd)} VAE weight tensors")

    # Read VAE config from the model YAML
    model_cfg = OmegaConf.load(model_config_path)
    vae_cfg = model_cfg.model.params.first_stage_config
    # Remove ckpt_path so instantiate_from_config doesn't try to load a
    # separate file — we load weights from the full LDM checkpoint above.
    if "ckpt_path" in vae_cfg.get("params", {}):
        del vae_cfg.params.ckpt_path
    vae = instantiate_from_config(vae_cfg)

    missing, unexpected = vae.load_state_dict(vae_sd, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing VAE keys (first 5: {missing[:5]})")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected VAE keys (first 5: {unexpected[:5]})")
    if not missing and not unexpected:
        print("  All VAE weights loaded successfully")

    vae.eval().to(device)
    for p in vae.parameters():
        p.requires_grad = False

    return vae, scale_factor


def load_codec(config, device, compvis_state_dict=None):
    """Load VAE and optionally SPNN codec from config."""
    sf_override = getattr(config.codec, 'compvis_scale_factor', None) if hasattr(config, 'codec') else None
    is_compvis = hasattr(config.model, 'type') and config.model.type == 'latent_compvis_ldm'

    if is_compvis:
        # Use the VAE from the CompVis checkpoint (matches the UNet)
        compvis_vae, ckpt_sf = _load_compvis_vae(
            config.model.compvis_ckpt_path, config.model.compvis_model_config, device,
            state_dict=compvis_state_dict,
        )
        scaling_factor = sf_override if sf_override is not None else ckpt_sf
        vae_codec = CompVisVAECodec(compvis_vae, scaling_factor)
    else:
        # SD 1.5 VAE for CelebA-HQ / other SD-based configs
        sd_id = "runwayml/stable-diffusion-v1-5"
        print("Loading SD 1.5 VAE...")
        vae = AutoencoderKL.from_pretrained(sd_id, subfolder="vae")
        vae.eval().to(device)
        for p in vae.parameters():
            p.requires_grad = False
        scaling_factor = sf_override if sf_override is not None else vae.config.scaling_factor
        vae_codec = VAECodec(vae, scaling_factor)

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
        spnn_codec = SPNNCodec(spnn, scaling_factor)
        return vae_codec, spnn_codec
    else:
        return vae_codec, None
