"""
Train an unconditional Latent Diffusion Model on CelebA-HQ (128x128).
Uses the pretrained VAE (frozen) and trains only the UNet in latent space.

Usage:
    python celebahq_experiment/train_ldm.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
SLDM_ROOT = os.path.join(os.path.dirname(__file__), '..',
    'simple-latent-diffusion-model-master', 'simple-latent-diffusion-model')
sys.path.insert(0, SLDM_ROOT)

from auto_encoder.models.variational_auto_encoder import VariationalAutoEncoder
from diffusion_model.models.latent_diffusion_model import LatentDiffusionModel
from diffusion_model.network.unet_wrapper import UnetWrapper
from diffusion_model.network.unet import Unet
from diffusion_model.sampler.ddim import DDIM
from helper.data_generator import DataGenerator
from helper.trainer import Trainer
from cifar10_experiment.train_cifar10 import _load_checkpoint

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'celebahq_config.yaml')
VAE_PATH = os.path.join(os.path.dirname(__file__), 'models', 'celebahq_vae.pth')
LDM_SAVE_PATH = os.path.join(os.path.dirname(__file__), 'models', 'celebahq_ldm')

if __name__ == '__main__':
    # 1. Load pretrained VAE (frozen)
    vae = VariationalAutoEncoder(CONFIG_PATH)
    vae = _load_checkpoint(vae, VAE_PATH)

    # 2. Build unconditional LatentDiffusionModel (no cond_encoder)
    sampler = DDIM(CONFIG_PATH)
    network = UnetWrapper(Unet, CONFIG_PATH)
    ldm = LatentDiffusionModel(network, sampler, vae)

    # 3. Train
    data_loader = DataGenerator().celebahq(batch_size=16, img_size=128)
    trainer = Trainer(ldm, ldm.loss)
    trainer.train(dl=data_loader, epochs=100, file_name=LDM_SAVE_PATH, no_label=True)
