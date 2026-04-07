"""
Minimal training script for SPNNAutoencoderLDM using modern PyTorch Lightning.
Replaces CompVis main.py to avoid version compatibility issues.

Usage:
    python train_spnn.py --gpus 4 --batch_size 12
"""

import argparse
import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from spnn_ldm import SPNNAutoencoderLDM, LSUNChurchesHFTrain, LSUNChurchesHFValidation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--accumulate_grad_batches", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=-1)
    parser.add_argument("--base_lr", type=float, default=4.5e-6)
    parser.add_argument("--scale_lr", action="store_true", default=True)
    parser.add_argument("--no_scale_lr", action="store_true")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    # Model config
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--scale_bound", type=float, default=2.0)
    parser.add_argument("--disc_start", type=int, default=50001)
    parser.add_argument("--disc_weight", type=float, default=0.5)
    parser.add_argument("--lambda_cycle", type=float, default=0.3)
    parser.add_argument("--lambda_roundtrip", type=float, default=0.3)
    parser.add_argument("--lambda_align", type=float, default=0.1)

    # Frozen VAE
    parser.add_argument("--compvis_ckpt", type=str,
                        default="../DDNM-main/models/lsun_churches.ckpt")
    parser.add_argument("--compvis_model_config", type=str,
                        default="../DDNM-main/models/lsun_churches-ldm-kl-8.yaml")

    # wandb
    parser.add_argument("--wandb_project", type=str, default="spnn-vae")
    parser.add_argument("--wandb_entity", type=str,
                        default="yamitehrlich-technion-israel-institute-of-technology")
    parser.add_argument("--wandb_name", type=str, default="spnn_churches_gan")

    args = parser.parse_args()

    if args.no_scale_lr:
        args.scale_lr = False

    pl.seed_everything(args.seed)

    # ── Model ──
    model = SPNNAutoencoderLDM(
        spnn_config=dict(
            mix_type="cayley",
            hidden=args.hidden,
            r_hidden=args.hidden,
            scale_bound=args.scale_bound,
        ),
        lossconfig=dict(
            target="ldm.modules.losses.LPIPSWithDiscriminator",
            params=dict(
                disc_start=args.disc_start,
                kl_weight=0.0,
                disc_weight=args.disc_weight,
                perceptual_weight=1.0,
            ),
        ),
        frozen_vae_config=dict(
            ckpt_path=args.compvis_ckpt,
            model_config=args.compvis_model_config,
        ),
        lambda_cycle=args.lambda_cycle,
        lambda_roundtrip=args.lambda_roundtrip,
        lambda_align=args.lambda_align,
        monitor="val/rec_loss",
    )

    # ── Learning rate ──
    if args.scale_lr:
        lr = args.accumulate_grad_batches * args.gpus * args.batch_size * args.base_lr
        print(f"Setting learning rate to {lr:.2e} = "
              f"{args.accumulate_grad_batches} (accum) * {args.gpus} (gpus) * "
              f"{args.batch_size} (bs) * {args.base_lr:.2e} (base_lr)")
    else:
        lr = args.base_lr
        print(f"Setting learning rate to {lr:.2e} (no scaling)")
    model.learning_rate = lr

    # ── Data ──
    train_dataset = LSUNChurchesHFTrain(size=256)
    val_dataset = LSUNChurchesHFValidation(size=256)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    print(f"#### Data #####")
    print(f"train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"val:   {len(val_dataset)} images, {len(val_loader)} batches")

    # ── Logger ──
    logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        save_dir=args.logdir,
    )

    # ── Callbacks ──
    ckpt_dir = os.path.join(args.logdir, args.wandb_name, "checkpoints")
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="{epoch:06}",
            verbose=True,
            save_last=True,
            monitor="val/rec_loss",
            save_top_k=3,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # ── Trainer ──
    trainer = pl.Trainer(
        gpus=args.gpus,
        accelerator="ddp" if args.gpus > 1 else None,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
        benchmark=True,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    # ── Train ──
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
