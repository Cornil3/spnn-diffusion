"""
Distill an SPNN autoencoder (512x512) from the SD1.5 KL-VAE on ImageNet-512.

FIXED vs v1: steps_per_epoch / total_updates (the cosine T_max) are now
computed AFTER accelerator.prepare(), i.e. from the per-rank *sharded*
dataloader. v1 measured the unsharded loader (8x too long), which made the
cosine decay ~8x too slow — LR would have sat near max for the whole run.

Goal: SPNN becomes a drop-in replacement for the SD1.5 VAE so the *same*
SD1.5 UNet can run latent DDNM on SPNN latents. Because DDNM is ideally run
unconditioned, this script also exports the empty-prompt ("null") CLIP text
embedding once; at DDNM time you feed that embedding to the SD1.5 UNet on
every step, which makes the conditional model behave as an unconditional one
(the standard CFG "uncond" branch). No labels are used anywhere in training.

Latent convention: SPNN is trained to encode directly into the *scaled*
SD latent space, i.e. z = 0.18215 * vae.encode(x).mean, shape [4, 64, 64].
That way SPNN.encode / SPNN.decode plug straight into the UNet with no
scaling factor anywhere in your DDNM loop.

Launch (8x B200):
    accelerate launch --num_processes 8 --mixed_precision bf16 \
        train_spnn512_distill.py
or:
    torchrun --nproc_per_node 8 train_spnn512_distill.py

Requires: torch, torchvision, accelerate, diffusers, transformers, Pillow
"""

import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL

# Your SPNN module (the file you pasted). Adjust the import path if needed.
from spnn_model import ConvPINNBlock, PixelUnshuffleBlock

# ----------------------------------------------------------------------------
# Config (no argparse — edit here)
# ----------------------------------------------------------------------------
CONFIG = {
    # data
    "data_root": "/shared/cycle1_iit_shocher_prj/data/imagenet512",
    "num_test_from_train": 10_000,   # held-out test images taken from train/
    "seed": 42,

    # teacher
    "sd_model_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "vae_scaling": 0.18215,          # SD1.5 latent scaling factor
    "save_null_embedding": True,     # export empty-prompt CLIP embedding for
                                     # unconditional DDNM with the SD1.5 UNet

    # model
    "mix_type": "householder",
    "hidden": 128,
    "r_hidden": 256,
    "scale_bound": 1.0,

    # optimization
    "epochs": 100,
    "batch_per_gpu": 32,             # 8 GPUs -> global batch 128
    "grad_accum": 1,
    "lr": 1e-6,
    "min_lr": 1e-7,
    "warmup_steps": 2000,            # in optimizer-update steps
    "weight_decay": 1e-5,
    "betas": (0.9, 0.95),
    "grad_clip": 1.0,
    "ema_decay": 0.9999,

    # loss weights
    "w_latent": 1.0,                 # MSE(z_spnn, z_vae) in scaled space
    "w_rec": 1.0,                    # L1 of SPNN self round-trip (cycle)
    "w_cross": 0.5,                  # L1 of SPNN.decode(teacher latent) — this
                                     # is what DDNM exercises (latents come from
                                     # the diffusion process, not SPNN.encode)

    # runtime
    "num_workers": 12,
    "log_every": 200,
    "val_subset": 2048,              # images evaluated each epoch (of the 10k)
    "out_dir": "runs/spnn512_sd15_distill",
    "resume": "auto",   # "auto" = pick up out_dir/ckpt_last.pt if it exists
    "channels_last": True,

    "wandb_project": "spnn512-distill",
    "run_name": "4node_b1024_lr2e-4",      # name each run by its batch/lr — future you will thank you
}


# ----------------------------------------------------------------------------
# SPNN for 512x512 -> 4x64x64 (matches SD1.5 VAE latent shape, f=8)
# ----------------------------------------------------------------------------
class SPNNAutoencoder512(nn.Module):
    """
    Encoder path:
        [3, 512, 512]   PixelUnshuffle(4)   -> [48, 128, 128]
        [48, 128, 128]  ConvPINN(48 -> 16)  -> [16, 128, 128]   r: 16->32
        [16, 128, 128]  PixelUnshuffle(2)   -> [64,  64,  64]
        [64,  64,  64]  ConvPINN(64 ->  4)  -> [ 4,  64,  64]   latent

    Same 2-coupling-block design as the 256 model; only feat_size changes,
    which routes ConvMLP to the residual U-Net branch at both stages.
    """

    def __init__(self, mix_type="cayley", hidden=128, r_hidden=256,
                 scale_bound=2.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            PixelUnshuffleBlock(4),
            ConvPINNBlock(48, 16, hidden=hidden, r_hidden=r_hidden,
                          scale_bound=scale_bound, mix_type=mix_type,
                          feat_size=128),
            PixelUnshuffleBlock(2),
            ConvPINNBlock(64, 4, hidden=hidden, r_hidden=r_hidden,
                          scale_bound=scale_bound, mix_type=mix_type,
                          feat_size=64),
        ])

    def encode(self, x):
        for b in self.blocks:
            x = b(x)
        return x

    def decode(self, y):
        for b in reversed(self.blocks):
            y = b.pinv(y)
        return y

    def forward(self, x):
        return self.encode(x)


class DistillWrapper(nn.Module):
    """
    Single forward() that touches every trainable path, so DDP gradient
    hooks fire correctly. Never call spnn.encode/decode directly through
    the DDP-wrapped module during training — that silently skips grad sync.
    """

    def __init__(self, spnn: nn.Module):
        super().__init__()
        self.spnn = spnn

    def forward(self, x, z_teacher):
        z = self.spnn.encode(x)            # [B, 4, 64, 64], scaled space
        rec_self = self.spnn.decode(z)     # SPNN cycle (uses s, t, r, mix)
        rec_cross = self.spnn.decode(z_teacher)  # decode VAE-manifold latents
        return z, rec_self, rec_cross


# ----------------------------------------------------------------------------
# Unlabeled ImageNet-512 dataset
# ----------------------------------------------------------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG"}


def list_images(root: Path):
    files = [p for p in root.rglob("*") if p.suffix in IMG_EXTS]
    files.sort()  # deterministic across ranks
    return files


class UnlabeledImages(Dataset):
    def __init__(self, paths, train: bool):
        self.paths = paths
        ops = [transforms.CenterCrop(512)]  # no-op for true 512x512 files
        if train:
            ops.append(transforms.RandomHorizontalFlip())
        ops += [
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),  # [-1, 1] for SD VAE
        ]
        self.tf = transforms.Compose(ops)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img)


# ----------------------------------------------------------------------------
# EMA (fp32 shadow, updated only after real optimizer steps)
# ----------------------------------------------------------------------------
class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {
            k: v.detach().float().clone()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(
                    v.detach().float(), alpha=1.0 - self.decay
                )

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        for k in self.shadow:
            self.shadow[k].copy_(sd[k])


# ----------------------------------------------------------------------------
# LR schedule: linear warmup -> cosine to min_lr.
# T_max is counted in OPTIMIZER-UPDATE steps (after grad accumulation),
# measured on the SHARDED per-rank dataloader (i.e. after prepare()).
# ----------------------------------------------------------------------------
def lr_lambda_factory(warmup, total_updates, base_lr, min_lr):
    floor = min_lr / base_lr

    def f(step):
        if step < warmup:
            return step / max(1, warmup)
        t = (step - warmup) / max(1, total_updates - warmup)
        t = min(t, 1.0)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t))

    return f


def psnr_from_mse(mse):
    # images live in [-1, 1] -> dynamic range 2 -> PSNR = 10 log10(4 / mse)
    return 10.0 * math.log10(4.0 / max(mse, 1e-12))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    cfg = CONFIG
    set_seed(cfg["seed"])

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg["grad_accum"],
        mixed_precision="bf16",
    )
    
    device = accelerator.device
    is_main = accelerator.is_main_process
    out_dir = Path(cfg["out_dir"])

    if is_main:
        import wandb
        wandb.init(project=cfg["wandb_project"], name=cfg["run_name"], config=cfg)
        
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- data ----------------
    root = Path(cfg["data_root"])
    train_files = list_images(root / "train")
    assert len(train_files) > cfg["num_test_from_train"], "train/ looks empty"

    rng = random.Random(cfg["seed"])
    idx = list(range(len(train_files)))
    rng.shuffle(idx)
    test_idx = set(idx[: cfg["num_test_from_train"]])
    test_files = [train_files[i] for i in sorted(test_idx)]
    train_files = [train_files[i] for i in range(len(train_files))
                   if i not in test_idx]

    if is_main:
        print(f"train images: {len(train_files):,} | "
              f"held-out test: {len(test_files):,}")
        # persist the split so eval scripts use the exact same 10k
        with open(out_dir / "test_split.txt", "w") as f:
            f.writelines(str(p) + "\n" for p in test_files)

    train_ds = UnlabeledImages(train_files, train=True)
    test_ds = UnlabeledImages(test_files[: cfg["val_subset"]], train=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_per_gpu"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True, drop_last=True,
        persistent_workers=True, prefetch_factor=4,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["batch_per_gpu"], shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # ---------------- teacher (frozen, fp32 weights) ----------------
    # Keep the frozen teacher in fp32; low-precision *weights* on frozen
    # targets has bitten us before. bf16 autocast at call time is fine.
    vae = AutoencoderKL.from_pretrained(
        cfg["sd_model_id"], subfolder="vae", torch_dtype=torch.float32
    ).to(device).eval()
    vae.requires_grad_(False)
    sf = cfg["vae_scaling"]

    # one-time export of the null text embedding for unconditional DDNM
    if is_main and cfg["save_null_embedding"]:
        from transformers import CLIPTextModel, CLIPTokenizer
        tok = CLIPTokenizer.from_pretrained(cfg["sd_model_id"],
                                            subfolder="tokenizer")
        te = CLIPTextModel.from_pretrained(cfg["sd_model_id"],
                                           subfolder="text_encoder")
        ids = tok([""], padding="max_length", max_length=77,
                  return_tensors="pt")
        with torch.no_grad():
            null_emb = te(ids.input_ids)[0]  # [1, 77, 768]
        torch.save(null_emb, out_dir / "null_text_embedding.pt")
        del te, tok
        print("saved null_text_embedding.pt — feed this to the SD1.5 UNet "
              "every DDNM step to run it unconditionally")

    # ---------------- student ----------------
    spnn = SPNNAutoencoder512(
        mix_type=cfg["mix_type"], hidden=cfg["hidden"],
        r_hidden=cfg["r_hidden"], scale_bound=cfg["scale_bound"],
    )
    model = DistillWrapper(spnn)
    if cfg["channels_last"]:
        model = model.to(memory_format=torch.channels_last)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        print(f"SPNN-512 trainable params: {n_params / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], betas=cfg["betas"],
        weight_decay=cfg["weight_decay"],
    )

    model, optimizer, train_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )

    # ------------------------------------------------------------------
    # FIX: measure the dataloader AFTER prepare(). Accelerate shards the
    # loader across ranks, so len(train_loader) is now per-GPU steps
    # (≈ dataset / (batch_per_gpu * num_gpus)), which is what one epoch
    # actually runs. v1 measured the unsharded loader and inflated the
    # cosine T_max by num_gpus (8x) — same bug class as the old
    # CosineAnnealingLR T_max incident.
    # ------------------------------------------------------------------
    steps_per_epoch = len(train_loader)
    updates_per_epoch = math.ceil(steps_per_epoch / cfg["grad_accum"])
    total_updates = updates_per_epoch * cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda_factory(cfg["warmup_steps"], total_updates,
                          cfg["lr"], cfg["min_lr"]),
    )
    if is_main:
        print(f"{steps_per_epoch} steps/epoch/GPU (sharded), "
              f"{updates_per_epoch} updates/epoch, "
              f"{total_updates} total updates (cosine T_max)")
    # scheduler is stepped manually on sync boundaries — deliberately NOT
    # passed through accelerator.prepare(), otherwise accelerate steps it
    # once per process and the schedule compresses by num_gpus.

    ema = EMA(accelerator.unwrap_model(model), cfg["ema_decay"])

    start_epoch = 0
    global_update = 0

    resume_path = None
    if cfg["resume"] == "auto":
        candidate = out_dir / "ckpt_last.pt"
        if candidate.exists():
            resume_path = candidate
    elif cfg["resume"]:
        resume_path = Path(cfg["resume"])

    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location="cpu")
        accelerator.unwrap_model(model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        ema.load_state_dict(ckpt["ema"])
        start_epoch = ckpt["epoch"] + 1
        global_update = ckpt["global_update"]
        if is_main:
            print(f"auto-resumed from {resume_path} at epoch {start_epoch}")
    elif is_main:
        print("no checkpoint found — starting fresh")

    # ---------------- training ----------------
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        t0 = time.time()
        running = {"lat": 0.0, "rec": 0.0, "cross": 0.0, "n": 0}

        for it, x in enumerate(train_loader):
            if cfg["channels_last"]:
                x = x.to(memory_format=torch.channels_last)

            # teacher latent target, scaled space, deterministic (mean)
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z_t = vae.encode(x).latent_dist.mean
                z_t = (z_t.float() * sf)

            with accelerator.accumulate(model):
                z_s, rec_self, rec_cross = model(x, z_t)

                l_lat = F.mse_loss(z_s.float(), z_t)
                l_rec = F.l1_loss(rec_self.float(), x.float())
                l_cross = F.l1_loss(rec_cross.float(), x.float())
                loss = (cfg["w_latent"] * l_lat
                        + cfg["w_rec"] * l_rec
                        + cfg["w_cross"] * l_cross)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(),
                                                cfg["grad_clip"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # scheduler + EMA only on real optimizer updates,
            # EMA strictly AFTER optimizer.step()
            if accelerator.sync_gradients:
                scheduler.step()
                ema.update(accelerator.unwrap_model(model))
                global_update += 1

            running["lat"] += l_lat.item()
            running["rec"] += l_rec.item()
            running["cross"] += l_cross.item()
            running["n"] += 1

            if is_main and (it + 1) % cfg["log_every"] == 0:
                n = running["n"]
                ips = (n * cfg["batch_per_gpu"] * accelerator.num_processes
                       / (time.time() - t0))
                print(
                    f"ep {epoch} it {it + 1}/{steps_per_epoch} | "
                    f"lat {running['lat'] / n:.4f} "
                    f"rec {running['rec'] / n:.4f} "
                    f"cross {running['cross'] / n:.4f} | "
                    f"lr {scheduler.get_last_lr()[0]:.2e} | "
                    f"{ips:.0f} img/s (global)"
                )
                wandb.log({"train/lat": running["lat"]/n, "train/rec": running["rec"]/n,
               "train/cross": running["cross"]/n,
               "train/lr": scheduler.get_last_lr()[0],
               "train/img_per_s": ips}, step=global_update)

        # ---------------- eval on held-out test subset ----------------
        model.eval()
        sums = torch.zeros(4, device=device)  # lat_mse, self_mse, cross_mse, n
        with torch.no_grad():
            for x in test_loader:
                if cfg["channels_last"]:
                    x = x.to(memory_format=torch.channels_last)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z_t = vae.encode(x).latent_dist.mean
                z_t = z_t.float() * sf
                z_s, rec_self, rec_cross = model(x, z_t)
                b = x.shape[0]
                sums[0] += F.mse_loss(z_s.float(), z_t,
                                      reduction="sum") / z_t[0].numel()
                sums[1] += F.mse_loss(rec_self.float(), x.float(),
                                      reduction="sum") / x[0].numel()
                sums[2] += F.mse_loss(rec_cross.float(), x.float(),
                                      reduction="sum") / x[0].numel()
                sums[3] += b
        sums = accelerator.reduce(sums, reduction="sum")
        if is_main:
            n = sums[3].item()
            lat_mse = sums[0].item() / n
            self_mse = sums[1].item() / n
            cross_mse = sums[2].item() / n
            dt = (time.time() - t0) / 60
            print(
                f"[epoch {epoch} done in {dt:.1f} min] "
                f"latent MSE {lat_mse:.5f} | "
                f"self-cycle PSNR {psnr_from_mse(self_mse):.2f} dB | "
                f"cross-decode PSNR {psnr_from_mse(cross_mse):.2f} dB"
            )
            wandb.log({"eval/latent_mse": lat_mse,
           "eval/self_cycle_psnr": psnr_from_mse(self_mse),
           "eval/cross_decode_psnr": psnr_from_mse(cross_mse),
           "epoch": epoch}, step=global_update)

        # ---------------- checkpoint ----------------
        accelerator.wait_for_everyone()
        if is_main:
            ckpt = {
                "epoch": epoch,
                "global_update": global_update,
                "config": cfg,
                "model": accelerator.unwrap_model(model).state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            #torch.save(ckpt, out_dir / f"ckpt_ep{epoch:03d}.pt")
            torch.save(ckpt, out_dir / "ckpt_last.pt")
            print(f"saved checkpoint ep{epoch:03d}")

    if is_main:
        print("training complete")


if __name__ == "__main__":
    main()