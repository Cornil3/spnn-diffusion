"""
Latent <-> pixel codecs for latent-space DDNM (Stable-Diffusion-1.5 latent space).

Both codecs map a 512x512 RGB image in [-1, 1] to/from a [4, 64, 64] latent that
lives in SD1.5's *scaled* latent space -- the space the SD1.5 UNet was trained to
denoise:

  * VAECodec   - the stock SD1.5 KL-VAE.        Lossy round-trip (~<20 dB).
  * SPNNCodec  - a near-invertible PINN auto-   Near-lossless round-trip (>22 dB).
                 encoder distilled into the
                 same scaled latent space.

Latent DDNM's data-consistency step decodes to pixels, pins the known region, and
re-encodes -- every diffusion step. A lossy codec makes that round-trip corrupt
the latent and the error compounds; a (near-)invertible codec does not. Comparing
the two codecs under identical sampling is the point of this experiment.
"""

import os
import sys
import torch

# SD1.5 weights are loaded by `diffusers`/`transformers` from the HF cache.
SD15_ID = "runwayml/stable-diffusion-v1-5"
SD15_SCALING = 0.18215  # SD1.5 vae.config.scaling_factor


class VAECodec:
    """Stock SD1.5 KL-VAE. Deterministic encode (posterior mode)."""

    def __init__(self, vae, scaling_factor=SD15_SCALING):
        self.vae = vae
        self.sf = scaling_factor

    @torch.no_grad()
    def encode(self, x):
        """pixel [-1,1] -> scaled latent."""
        return self.vae.encode(x).latent_dist.mode() * self.sf

    @torch.no_grad()
    def decode(self, z):
        """scaled latent -> pixel [-1,1]."""
        return self.vae.decode(z / self.sf).sample


class SPNNCodec:
    """Near-invertible SPNN-512 autoencoder.

    The distilled SPNN-512 already encodes/decodes directly in SD's *scaled*
    latent space, so there is no external scaling factor (unless `external_scale`
    is given, kept as an escape hatch in case a checkpoint lives in raw VAE space).
    """

    def __init__(self, spnn, external_scale=None):
        self.spnn = spnn
        self.sf = external_scale  # None => prescaled (identity)

    @torch.no_grad()
    def encode(self, x):
        z = self.spnn.encode(x)
        return z if self.sf is None else z * self.sf

    @torch.no_grad()
    def decode(self, z):
        return self.spnn.decode(z if self.sf is None else z / self.sf)


def _resolve_repo_root(config):
    """Absolute path to the simple-latent-diffusion-model repo root.

    `config.model.repo_root` is given relative to the DDNM folder (where main.py
    runs); resolve it against this file's location so it works from any cwd.
    """
    r = getattr(config.model, "repo_root", "..")
    if os.path.isabs(r):
        return r
    ddnm_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../DDNM-main-scartch
    return os.path.normpath(os.path.join(ddnm_dir, r))


def load_vae(device):
    """Load the stock SD1.5 KL-VAE codec."""
    from diffusers import AutoencoderKL

    print(f"Loading SD1.5 VAE ({SD15_ID})...")
    vae = AutoencoderKL.from_pretrained(SD15_ID, subfolder="vae")
    vae.eval().to(device)
    for p in vae.parameters():
        p.requires_grad_(False)
    return VAECodec(vae)


def load_spnn(config, device):
    """Instantiate SPNN-512 from `spnn_model.py` and load its distill checkpoint."""
    repo_root = _resolve_repo_root(config)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    spnn_class = getattr(config.codec, "spnn_class", "SPNNAutoencoder512")
    if spnn_class != "SPNNAutoencoder512":
        raise ValueError(
            f"latent_codec targets SD1.5/SPNN-512; got spnn_class={spnn_class!r}"
        )
    from spnn_model import SPNNAutoencoder512

    spnn = SPNNAutoencoder512(
        mix_type=config.codec.spnn_mix_type,
        hidden=config.codec.spnn_hidden,
        r_hidden=getattr(config.codec, "spnn_r_hidden", 256),
        scale_bound=config.codec.spnn_scale_bound,
    )

    ckpt_path = config.codec.spnn_checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(repo_root, ckpt_path)
    print(f"Loading SPNN-512 from {ckpt_path}...")

    # weights_only=False: distill checkpoints stash a non-tensor config dict too.
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    use_ema = getattr(config.codec, "spnn_use_ema", True)
    sd = state.get("ema") if use_ema else state.get("model")
    if sd is None:
        sd = state.get("model_state_dict", state)
    # Distill checkpoints prefix the autoencoder weights with "spnn."
    spnn_sd = {k[len("spnn."):]: v for k, v in sd.items() if k.startswith("spnn.")}
    missing, unexpected = spnn.load_state_dict(spnn_sd or sd, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing SPNN keys (first: {missing[:3]})")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected SPNN keys (first: {unexpected[:3]})")

    spnn.eval().to(device)
    for p in spnn.parameters():
        p.requires_grad_(False)

    external_scale = getattr(config.codec, "spnn_external_scale", None)
    return SPNNCodec(spnn, external_scale=external_scale)
