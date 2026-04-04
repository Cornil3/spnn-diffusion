"""
Train a VAE on CelebA-HQ (128x128) using simple-latent-diffusion-model's
VariationalAutoEncoder and Trainer.

Usage:
    python celebahq_experiment/train_vae.py
"""

import os
import sys

SLDM_ROOT = os.path.join(os.path.dirname(__file__), '..',
    'simple-latent-diffusion-model-master', 'simple-latent-diffusion-model')
sys.path.insert(0, SLDM_ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from auto_encoder.models.variational_auto_encoder import VariationalAutoEncoder
from helper.data_generator import DataGenerator
from helper.trainer import Trainer

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'celebahq_config.yaml')
VAE_SAVE_PATH = os.path.join(os.path.dirname(__file__), 'models', 'celebahq_vae')

if __name__ == '__main__':
    os.makedirs(os.path.dirname(VAE_SAVE_PATH), exist_ok=True)

    vae = VariationalAutoEncoder(CONFIG_PATH)
    data_loader = DataGenerator().celebahq(batch_size=64, img_size=128)

    trainer = Trainer(vae, vae.loss)
    trainer.train(dl=data_loader, epochs=200, file_name=VAE_SAVE_PATH, no_label=True)
