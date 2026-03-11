"""
Train a new LDM (UNet) using SPNN as the autoencoder instead of VAE.

Uses the original Simple LDM Trainer class for identical training config
(fp16, EMA, CosineAnnealingLR, gradient clipping, etc.).

A thin wrapper makes SPNN look like the VAE to LatentDiffusionModel.

If img2img results improve, it proves the distribution mismatch was the issue.
"""

import argparse
import os
import sys

import torch
import torch.nn as nn

# ── SPNN ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cifar10_experiment.train_cifar10 import (
    SPNNAutoencoderConfigurable, CIFAR10_STAGES, _load_checkpoint)

# ── Simple LDM ──
SLDM_ROOT = os.path.join(os.path.dirname(__file__), '..',
    'simple-latent-diffusion-model-master', 'simple-latent-diffusion-model')
sys.path.insert(0, SLDM_ROOT)
from diffusion_model.models.latent_diffusion_model import LatentDiffusionModel
from diffusion_model.network.unet_wrapper import UnetWrapper
from diffusion_model.network.unet import Unet
from diffusion_model.sampler.ddim import DDIM
from helper.cond_encoder import ClassEncoder
from helper.data_generator import DataGenerator
from helper.trainer import Trainer

CONFIG_PATH = os.path.join(SLDM_ROOT, 'configs', 'cifar10_config.yaml')


# ═══════════════════════════════════════════════════════════
# SPNN → VAE interface wrapper
# ═══════════════════════════════════════════════════════════

class _TensorAsDistribution:
    """Wraps a plain tensor to mimic DiagonalGaussianDistribution."""
    def __init__(self, z):
        self._z = z

    def sample(self):
        return self._z

    def mode(self):
        return self._z


class _FakeDecoder:
    """Provides .z_shape expected by LatentDiffusionModel.__init__."""
    def __init__(self, z_shape):
        self.z_shape = z_shape


class SPNNAsVAE(nn.Module):
    """
    Wraps SPNNAutoencoderConfigurable to match the VariationalAutoEncoder
    interface expected by LatentDiffusionModel:
      - .encode(x) returns object with .sample()
      - .decode(z)
      - .decoder.z_shape
      - .embed_dim
    """
    def __init__(self, spnn, embed_dim=3, spatial=16):
        super().__init__()
        self.spnn = spnn
        self.embed_dim = embed_dim
        # LatentDiffusionModel reads self.auto_encoder.decoder.z_shape
        self.decoder = _FakeDecoder([1, embed_dim, spatial, spatial])

    def encode(self, x):
        return _TensorAsDistribution(self.spnn.encode(x))

    def decode(self, z):
        return self.spnn.decode(z)

    def forward(self, x):
        return self.decode(self.encode(x).sample())


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main(args):
    # 1. Load frozen SPNN, wrap as VAE
    print(f"Loading frozen SPNN from {args.spnn_checkpoint}...")
    spnn = SPNNAutoencoderConfigurable(
        stages=CIFAR10_STAGES,
        mix_type=args.mix_type,
        hidden=args.hidden,
        scale_bound=args.scale_bound,
    )
    state = torch.load(args.spnn_checkpoint, map_location="cpu", weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    spnn.load_state_dict(state)
    spnn_vae = SPNNAsVAE(spnn)
    print("  SPNN loaded and wrapped as VAE interface")

    # 2. Build LatentDiffusionModel (same architecture as original)
    sampler = DDIM(CONFIG_PATH)
    cond_encoder = ClassEncoder(CONFIG_PATH)
    network = UnetWrapper(Unet, CONFIG_PATH, cond_encoder)
    ldm = LatentDiffusionModel(network, sampler, spnn_vae)
    print(f"  UNet params: {sum(p.numel() for p in network.parameters()):,}")

    # 3. Dataset (same as original training)
    data_generator = DataGenerator()
    data_loader = data_generator.cifar10(batch_size=args.batch_size)

    # 4. Train with original Trainer (fp16, EMA, cosine LR, grad clip)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    trainer = Trainer(ldm, ldm.loss)
    trainer.train(
        dl=data_loader,
        epochs=args.num_epochs,
        file_name=args.save_path,
        no_label=False,
    )
    print(f"\nTraining complete. Model saved to {args.save_path}.pth")


def parse_args():
    p = argparse.ArgumentParser(
        description="Train LDM (UNet) on SPNN latents using original Trainer")
    p.add_argument("--spnn_checkpoint", type=str, required=True,
                   help="Frozen SPNN model checkpoint")
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    # SPNN model args (must match checkpoint)
    p.add_argument("--mix_type", type=str, default="cayley")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--scale_bound", type=float, default=2.0)
    p.add_argument("--save_path", type=str,
                   default="cifar10_experiment/checkpoints_ldm_spnn/ldm_spnn")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
