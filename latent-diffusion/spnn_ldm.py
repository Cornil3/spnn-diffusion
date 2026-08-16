"""
SPNN Autoencoder adapted for CompVis LDM training pipeline.

Subclasses AutoencoderKL to reuse LPIPSWithDiscriminator, PatchGAN,
adaptive weighting, logvar, and the two-optimizer Lightning training loop.

Usage:
    python main.py --base configs/spnn_churches.yaml -t --gpus 0,1,2,3,4,5,6,7
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from ldm.util import instantiate_from_config

# Add parent dir so we can import SPNNAutoencoder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models import SPNNAutoencoder


class DummyPosterior:
    """Dummy posterior that returns zero KL — SPNN has no variational component."""
    def __init__(self, z):
        self._z = z

    def kl(self):
        return torch.zeros(self._z.shape[0], device=self._z.device)

    def mode(self):
        return self._z

    def sample(self):
        return self._z


class SPNNAutoencoderLDM(pl.LightningModule):
    """
    SPNN autoencoder wrapped for the CompVis LDM training pipeline.

    Uses LPIPSWithDiscriminator for reconstruction + GAN loss,
    plus SPNN-specific losses (cycle, roundtrip, align).
    """

    def __init__(self,
                 spnn_config,
                 lossconfig,
                 frozen_vae_config,
                 lambda_cycle=0.3,
                 lambda_roundtrip=0.3,
                 lambda_align=0.1,
                 accumulate_grad_batches=2,
                 image_key="image",
                 monitor=None,
                 ckpt_path=None,
                 ):
        super().__init__()
        self.automatic_optimization = False
        self.accumulate_grad_batches = accumulate_grad_batches
        self.image_key = image_key

        # ── SPNN model ──
        self.spnn = SPNNAutoencoder(**spnn_config)

        # ── Loss (LPIPSWithDiscriminator — includes discriminator, logvar, adaptive weight) ──
        self.loss = instantiate_from_config(lossconfig)
        # Bypass adaptive weight: SPNN has no equivalent to VAE's decoder.conv_out
        # for meaningful gradient ratio computation. Use fixed disc_weight instead.
        # Original: d_weight = gradient_ratio * self.discriminator_weight
        # Ours: d_weight = 1.0 * self.discriminator_weight = disc_weight from config
        # So effective GAN multiplier = disc_weight * disc_factor * g_loss
        disc_w = self.loss.discriminator_weight
        self.loss.calculate_adaptive_weight = lambda nll_loss, g_loss, last_layer=None: torch.tensor(disc_w)

        # ── Frozen CompVis VAE for align/cycle losses ──
        self._load_frozen_vae(frozen_vae_config)

        # ── SPNN-specific loss weights ──
        self.lambda_cycle = lambda_cycle
        self.lambda_roundtrip = lambda_roundtrip
        self.lambda_align = lambda_align

        if monitor is not None:
            self.monitor = monitor

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def _load_frozen_vae(self, vae_config):
        """Load frozen CompVis VAE from checkpoint for align/cycle losses."""
        from omegaconf import OmegaConf

        ckpt_path = vae_config["ckpt_path"]
        model_config_path = vae_config["model_config"]

        print(f"Loading frozen CompVis VAE from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

        # Extract VAE weights
        vae_sd = {}
        for k, v in sd.items():
            if k.startswith("first_stage_model."):
                vae_sd[k.replace("first_stage_model.", "")] = v

        # Instantiate from YAML config
        model_cfg = OmegaConf.load(model_config_path)
        vae_cfg = model_cfg.model.params.first_stage_config
        if "ckpt_path" in vae_cfg.get("params", {}):
            del vae_cfg.params.ckpt_path
        frozen_vae = instantiate_from_config(vae_cfg)
        frozen_vae.load_state_dict(vae_sd, strict=False)
        frozen_vae.eval()
        for p in frozen_vae.parameters():
            p.requires_grad = False
        # Store as non-module attribute so DDP doesn't try to sync its (frozen) params
        self._frozen_vae = [frozen_vae]  # wrapped in list to hide from nn.Module
        print(f"  Loaded {len(vae_sd)} VAE weight tensors (frozen)")

    def init_from_ckpt(self, path):
        """Load SPNN weights from a checkpoint."""
        print(f"Loading SPNN weights from {path}...")
        state = torch.load(path, map_location="cpu")
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        missing, unexpected = self.spnn.load_state_dict(state, strict=False)
        if missing:
            print(f"  Warning: {len(missing)} missing keys")
        if unexpected:
            print(f"  Warning: {len(unexpected)} unexpected keys")
        if not missing and not unexpected:
            print(f"  All SPNN weights loaded successfully")

    def encode(self, x):
        """Encode image to latent (returns raw tensor, no posterior)."""
        return self.spnn.encode(x)

    def decode(self, z):
        """Decode latent to image."""
        return self.spnn.decode(z)

    def forward(self, input, sample_posterior=True):
        """Forward pass: encode -> decode, return (reconstruction, dummy_posterior)."""
        z = self.encode(input)
        dec = self.decode(z)
        posterior = DummyPosterior(z)
        return dec, posterior

    def get_input(self, batch, k):
        """Extract image from batch dict, permute HWC->CHW."""
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    @property
    def frozen_vae(self):
        """Access frozen VAE, ensuring it's on the correct device."""
        vae = self._frozen_vae[0]
        if next(vae.parameters()).device != self.device:
            vae.to(self.device)
        return vae

    @torch.no_grad()
    def _get_vae_latent(self, images):
        """Get frozen VAE latent for a batch of images."""
        posterior = self.frozen_vae.encode(images)
        return posterior.mode()

    def _spnn_losses(self, images, z_spnn, spnn_decoded):
        """Compute SPNN-specific losses: cycle, roundtrip, align."""
        loss = torch.tensor(0.0, device=images.device)
        log = {}

        # Get VAE latent once (shared by cycle and align)
        vae_latent = None
        if self.lambda_cycle > 0 or self.lambda_align > 0:
            vae_latent = self._get_vae_latent(images)

        # Cycle: encode(decode(z_vae)) ≈ z_vae
        if self.lambda_cycle > 0:
            spnn_from_vae = self.spnn.decode(vae_latent)
            re_encoded = self.spnn.encode(spnn_from_vae)
            cycle_loss = nn.functional.mse_loss(re_encoded, vae_latent)
            loss = loss + self.lambda_cycle * cycle_loss
            log["train/cycle_loss"] = cycle_loss.detach()

        # Roundtrip: decode(encode(x)) ≈ x
        if self.lambda_roundtrip > 0:
            roundtrip_recon = self.spnn.decode(z_spnn)
            roundtrip_loss = nn.functional.mse_loss(roundtrip_recon, images)
            loss = loss + self.lambda_roundtrip * roundtrip_loss
            log["train/roundtrip_loss"] = roundtrip_loss.detach()

        # Align: SPNN.encode(x) ≈ VAE.encode(x)
        if self.lambda_align > 0:
            align_loss = nn.functional.mse_loss(z_spnn, vae_latent)
            loss = loss + self.lambda_align * align_loss
            log["train/align_loss"] = align_loss.detach()

        return loss, log

    def training_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        opt_ae, opt_disc = self.optimizers()
        accum = self.accumulate_grad_batches
        should_step = (batch_idx + 1) % accum == 0

        # ── Step 1: Generator (SPNN) update ──
        aeloss, log_dict_ae = self.loss(
            inputs, reconstructions, posterior, 0,
            self.global_step, last_layer=self.get_last_layer(), split="train"
        )

        # SPNN-specific losses
        z_spnn = posterior.mode()
        spnn_loss, spnn_log = self._spnn_losses(inputs, z_spnn, reconstructions)

        total_loss = (aeloss + spnn_loss) / accum

        self.manual_backward(total_loss)
        if should_step:
            opt_ae.step()
            opt_ae.zero_grad()

        # ── Step 2: Discriminator update ──
        discloss, log_dict_disc = self.loss(
            inputs, reconstructions, posterior, 1,
            self.global_step, last_layer=self.get_last_layer(), split="train"
        )

        self.manual_backward(discloss / accum)
        if should_step:
            opt_disc.step()
            opt_disc.zero_grad()

        # ── Logging (unscaled losses) ──
        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("spnn_loss", spnn_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log_dict(spnn_log, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        aeloss, log_dict_ae = self.loss(
            inputs, reconstructions, posterior, 0, self.global_step,
            last_layer=self.get_last_layer(), split="val"
        )
        discloss, log_dict_disc = self.loss(
            inputs, reconstructions, posterior, 1, self.global_step,
            last_layer=self.get_last_layer(), split="val"
        )

        # sync_dist=True ensures val metrics are averaged across all GPUs.
        # This matters for val/rec_loss in particular since it's the monitor
        # metric used by ModelCheckpoint to decide "best" checkpoints.
        self.log("val/rec_loss", log_dict_ae["val/rec_loss"],
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_disc, on_step=False, on_epoch=True, sync_dist=True)

        # Penrose identity checks — averaged over the full val set via Lightning
        self._penrose_check(inputs)

        return self.log_dict

    @torch.no_grad()
    def _penrose_check(self, inputs):
        """
        Compute Penrose pseudo-inverse identities for the current val batch.
        Logged with on_epoch=True, so Lightning automatically averages across
        the full validation set.

        g  = spnn.encode  (forward)
        g' = spnn.decode  (pinv)
        """
        was_training = self.spnn.training
        self.spnn.eval()
        try:
            x = inputs
            # Use the frozen CompVis VAE latents as the reference z
            z = self._get_vae_latent(x)

            gx = self.spnn.encode(x)
            gpgx = self.spnn.decode(gx)
            ggpgx = self.spnn.encode(gpgx)

            gpz = self.spnn.decode(z)
            ggpz = self.spnn.encode(gpz)
            gpggpz = self.spnn.decode(ggpz)

            metrics = {
                "val/penrose/ggg_eq_g":    nn.functional.mse_loss(ggpgx, gx),
                "val/penrose/gpggp_eq_gp": nn.functional.mse_loss(gpggpz, gpz),
                "val/penrose/ggp_eq_id":   nn.functional.mse_loss(ggpz, z),
                "val/penrose/roundtrip":   nn.functional.mse_loss(gpgx, x),
            }

            # on_epoch=True (default) averages across all val batches
            self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)
        finally:
            if was_training:
                self.spnn.train()

    def configure_optimizers(self):
        lr = self.learning_rate
        # SPNN params + logvar
        opt_ae = torch.optim.Adam(
            list(self.spnn.parameters()) + [self.loss.logvar],
            lr=lr, betas=(0.5, 0.9)
        )
        # Discriminator params
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(),
            lr=lr, betas=(0.5, 0.9)
        )
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        """Last non-zero-init conv in SPNN decode path.
        Uses the s network's final ResBlock conv in blocks[1] (the last coupling block
        before PixelShuffle to output). This layer has standard init (non-zero) so the
        adaptive weight gradient ratio is meaningful.
        """
        return self.spnn.blocks[1].s.dec_blocks[2].block[3].weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, _ = self(x)
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log


# ──────────────────────── Dataset ────────────────────────


class LSUNChurchesHF(Dataset):
    """
    LSUN Churches from HuggingFace, returning dicts compatible with CompVis pipeline.
    Images are returned as numpy HWC float32 in [-1, 1].

    Uses a reproducible random 90/10 train/val split with seed 42.
    """

    def __init__(self, size=256, split="train", val_fraction=0.1, seed=42, flip_p=0.5):
        from datasets import load_dataset
        import random as _random

        print("Loading tglcourse/lsun_church_train from HuggingFace...")
        ds = load_dataset("tglcourse/lsun_church_train", split="train")

        n = len(ds)
        # Reproducible shuffle: same seed → same indices every run
        indices = list(range(n))
        _random.Random(seed).shuffle(indices)

        n_val = int(round(val_fraction * n))
        # Validation: first n_val from shuffled order
        val_indices = sorted(indices[:n_val])
        # Training: remaining indices
        train_indices = sorted(indices[n_val:])

        if split == "train":
            ds = ds.select(train_indices)
        elif split == "val":
            ds = ds.select(val_indices)

        print(f"  Split: {split} — {len(ds)} images "
              f"(total {n}, val_fraction={val_fraction}, seed={seed})")
        self.ds = ds
        self.size = size
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img = self.ds[idx]["image"]
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        img = img.convert("RGB")

        # Center crop to square, then resize (same as CompVis LSUNBase)
        w, h = img.size
        crop = min(w, h)
        left = (w - crop) // 2
        top = (h - crop) // 2
        img = img.crop((left, top, left + crop, top + crop))
        img = img.resize((self.size, self.size), Image.BICUBIC)
        img = self.flip(img)

        # To numpy HWC float32 in [-1, 1] (CompVis convention)
        img = np.array(img).astype(np.float32)
        img = (img / 127.5) - 1.0

        return {"image": img}


class LSUNChurchesHFTrain(LSUNChurchesHF):
    def __init__(self, **kwargs):
        super().__init__(split="train", **kwargs)


class LSUNChurchesHFValidation(LSUNChurchesHF):
    def __init__(self, flip_p=0.0, **kwargs):
        super().__init__(split="val", flip_p=flip_p, **kwargs)
