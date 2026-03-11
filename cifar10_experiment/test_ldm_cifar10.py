"""
Quick sanity check: generate CIFAR-10 images from the trained LDM.

Verifies that the LDM + pretrained VAE produces reasonable images
before using it as the backbone for SPNN experiments.

Usage:
    python cifar10_experiment/test_ldm_cifar10.py
    python cifar10_experiment/test_ldm_cifar10.py --ldm_path path/to/cifar_ldm.pth
"""

import argparse
import os
import sys

import torch
from torchvision.utils import save_image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
SLDM_ROOT = os.path.join(os.path.dirname(__file__), '..',
    'simple-latent-diffusion-model-master', 'simple-latent-diffusion-model')
sys.path.insert(0, SLDM_ROOT)

from auto_encoder.models.variational_auto_encoder import VariationalAutoEncoder
from diffusion_model.models.latent_diffusion_model import LatentDiffusionModel
from diffusion_model.network.unet_wrapper import UnetWrapper
from diffusion_model.network.unet import Unet
from diffusion_model.sampler.ddim import DDIM
from helper.cond_encoder import ClassEncoder
from cifar10_experiment.train_cifar10 import _load_checkpoint

CONFIG_PATH = os.path.join(SLDM_ROOT, 'configs', 'cifar10_config.yaml')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


def main(args):
    # Load VAE
    vae = VariationalAutoEncoder(CONFIG_PATH)
    vae = _load_checkpoint(vae, os.path.join(SLDM_ROOT, 'models', 'cifar_vae.pth'))
    vae.eval().to(DEVICE)

    # Load trained LDM
    sampler = DDIM(CONFIG_PATH)
    cond_encoder = ClassEncoder(CONFIG_PATH)
    network = UnetWrapper(Unet, CONFIG_PATH, cond_encoder)
    ldm = LatentDiffusionModel(network, sampler, vae)
    ldm = _load_checkpoint(ldm, args.ldm_path)
    ldm.eval().to(DEVICE)
    print(f"LDM loaded from {args.ldm_path}")
    print(f"Latent shape: {ldm.image_shape}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Generate samples per class
    all_samples = []
    with torch.no_grad():
        for class_idx in range(10):
            y = torch.tensor([class_idx], device=DEVICE)
            # ldm.sample() decodes latents to pixels via VAE
            samples = ldm.sample(args.n_samples, y=y, gamma=args.gamma)
            samples = samples.clamp(-1, 1)
            all_samples.append(samples)
            print(f"  Class {class_idx} ({CIFAR10_CLASSES[class_idx]}): "
                  f"range [{samples.min():.2f}, {samples.max():.2f}]")

    # Save grid: 10 rows (classes) x n_samples columns
    grid = torch.cat(all_samples, dim=0)
    grid = (grid.cpu() + 1) / 2  # [-1,1] -> [0,1]
    grid_path = os.path.join(args.output_dir, "ldm_samples.png")
    save_image(grid, grid_path, nrow=args.n_samples, padding=2)
    print(f"\nSaved {10 * args.n_samples} samples to {grid_path}")
    print("Rows: " + " | ".join(CIFAR10_CLASSES))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Test trained LDM on CIFAR-10")
    p.add_argument("--ldm_path", type=str,
                   default=os.path.join(SLDM_ROOT, 'models', 'cifar_ldm.pth'),
                   help="Path to trained LDM checkpoint")
    p.add_argument("--n_samples", type=int, default=8,
                   help="Samples per class")
    p.add_argument("--gamma", type=float, default=3.0,
                   help="Classifier-free guidance scale")
    p.add_argument("--output_dir", type=str,
                   default="cifar10_experiment/results")
    main(p.parse_args())
