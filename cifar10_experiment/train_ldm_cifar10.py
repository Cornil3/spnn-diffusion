"""
Train a class-conditional Latent Diffusion Model on CIFAR-10.

Uses the pretrained VAE (frozen) and trains only the UNet in latent space.
Leverages the simple-latent-diffusion-model repo's Trainer class.

Usage:
    python cifar10_experiment/train_ldm_cifar10.py
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
from helper.cond_encoder import ClassEncoder
from helper.data_generator import DataGenerator
from helper.trainer import Trainer
from cifar10_experiment.train_cifar10 import _load_checkpoint

CONFIG_PATH = os.path.join(SLDM_ROOT, 'configs', 'cifar10_config.yaml')
VAE_PATH = os.path.join(SLDM_ROOT, 'models', 'cifar_vae.pth')
LDM_SAVE_PATH = os.path.join(SLDM_ROOT, 'models', 'cifar_ldm')

if __name__ == '__main__':
    # 1. Load pretrained VAE (frozen)
    vae = VariationalAutoEncoder(CONFIG_PATH)
    vae = _load_checkpoint(vae, VAE_PATH)

    # 2. Build LatentDiffusionModel
    sampler = DDIM(CONFIG_PATH)
    cond_encoder = ClassEncoder(CONFIG_PATH)
    network = UnetWrapper(Unet, CONFIG_PATH, cond_encoder)
    ldm = LatentDiffusionModel(network, sampler, vae)

    # 3. Train
    data_generator = DataGenerator()
    data_loader = data_generator.cifar10(batch_size=128)

    trainer = Trainer(ldm, ldm.loss)
    trainer.train(dl=data_loader, epochs=100, file_name=LDM_SAVE_PATH, no_label=False)
