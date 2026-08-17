import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers import AutoencoderKL
import wandb
import lpips
from accelerate import Accelerator
from models import SPNNAutoencoder
from dataset import CelebAHQDataset, LAIONAestheticDataset, LSUNChurchesDataset
from diagnostics import penrose_check, print_penrose_metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_sd_vae(device, verbose=True):
    """Load frozen SD-VAE to given device."""
    if verbose:
        print("Loading VAE from runwayml/stable-diffusion-v1-5...")
    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="vae"
    )
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae.to(device)


def load_compvis_vae(ckpt_path, model_config_path, device, verbose=True):
    """Load frozen CompVis KL-VAE from an LDM .ckpt file."""
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config

    if verbose:
        print(f"Loading CompVis VAE from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    # Extract first_stage_model weights
    vae_sd = {}
    for k, v in sd.items():
        if k.startswith("first_stage_model."):
            vae_sd[k.replace("first_stage_model.", "")] = v

    # Instantiate VAE from the model config YAML
    model_cfg = OmegaConf.load(model_config_path)
    vae_cfg = model_cfg.model.params.first_stage_config
    if "ckpt_path" in vae_cfg.get("params", {}):
        del vae_cfg.params.ckpt_path
    vae = instantiate_from_config(vae_cfg)

    missing, unexpected = vae.load_state_dict(vae_sd, strict=False)
    if verbose:
        if missing:
            print(f"  Warning: {len(missing)} missing VAE keys")
        if unexpected:
            print(f"  Warning: {len(unexpected)} unexpected VAE keys")
        if not missing and not unexpected:
            print(f"  All {len(vae_sd)} VAE weights loaded successfully")

    vae.eval().to(device)
    for p in vae.parameters():
        p.requires_grad = False
    return vae


class CompVisVAEWrapper:
    """Wraps CompVis VAE to match the diffusers VAE API used in get_vae_pairs."""
    def __init__(self, compvis_vae):
        self.vae = compvis_vae

    def encode(self, x):
        posterior = self.vae.encode(x)
        return _CompVisPosteriorWrapper(posterior)

    def decode(self, z):
        out = self.vae.decode(z)
        return _CompVisDecodeWrapper(out)

    def to(self, device):
        self.vae.to(device)
        return self

    def parameters(self):
        return self.vae.parameters()


class _CompVisPosteriorWrapper:
    """Makes CompVis posterior match diffusers .latent_dist API."""
    def __init__(self, posterior):
        self.latent_dist = posterior

    # Allow direct access: vae.encode(x).latent_dist.mode() / .sample()


class _CompVisDecodeWrapper:
    """Makes CompVis decode output match diffusers .sample API."""
    def __init__(self, tensor):
        self.sample = tensor


@torch.no_grad()
def get_vae_pairs(vae, images):
    """
    Get (latent, decoded_image) pairs from the frozen VAE.
    These are the training targets for our SPNN decoder.
    Works with both diffusers and CompVis (wrapped) VAEs.
    """
    posterior = vae.encode(images).latent_dist
    latent = posterior.sample()
    decoded = vae.decode(latent).sample
    return latent, decoded

def save_cycle_comparison(spnn, images, epoch, sample_dir, num_cycles=5):
    """Save grid: row 0 = original, rows 1..num_cycles = after each encode→decode cycle."""
    from torchvision.utils import save_image
    spnn.eval()
    n = min(4, images.size(0))
    x = images[:n]
    rows = [(x.cpu() + 1) / 2]
    for _ in range(num_cycles):
        x = spnn.decode(spnn.encode(x))
        rows.append((x.detach().cpu() + 1) / 2)
    grid = torch.cat(rows, dim=0)
    path = os.path.join(sample_dir, f"epoch{epoch:03d}_cycles.png")
    save_image(grid, path, nrow=n, padding=2)

def train(args):
    # ── Accelerator (handles DDP, mixed precision, device placement) ──
    accelerator = Accelerator(mixed_precision='bf16')
    device = accelerator.device
    is_main = accelerator.is_main_process

    if is_main:
        print(f"Device: {device}  |  Num processes: {accelerator.num_processes}")

    os.makedirs(args.output_dir, exist_ok=True)
    train_sample_dir = os.path.join(args.sample_dir, "train")
    if is_main:
        os.makedirs(train_sample_dir, exist_ok=True)

    if args.dataset == "laion":
        dataset = LAIONAestheticDataset(
            data_dir=args.laion_dir, img_size=args.img_size,
            split="train", n_test=args.n_test, max_images=args.max_images,
        )
    elif args.dataset == "lsun_churches":
        dataset = LSUNChurchesDataset(
            img_size=args.img_size, max_images=args.max_images,
            split="train", n_test=args.n_test,
            data_dir=getattr(args, 'lsun_dir', None),
        )
    else:
        dataset = CelebAHQDataset(
            img_size=args.img_size, max_images=args.max_images,
            split="train", n_test=args.n_test,
        )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    if is_main:
        print(f"Dataset: {len(dataset)} images, {len(loader)} batches/epoch")

    # ── Frozen models (not wrapped by accelerate — just move to device) ──
    if getattr(args, 'compvis_ckpt', None) is not None:
        compvis_vae = load_compvis_vae(
            args.compvis_ckpt, args.compvis_model_config, device, verbose=is_main)
        vae = CompVisVAEWrapper(compvis_vae)
    else:
        vae = load_sd_vae(device, verbose=is_main)
    spnn = SPNNAutoencoder(mix_type=args.mix_type, hidden=args.hidden,
                           r_hidden=args.hidden,
                           scale_bound=args.scale_bound,
                           num_blocks=args.num_blocks,
                           use_deep_convmlp=args.deep_convmlp).to(device)

    total_params = sum(p.numel() for p in spnn.parameters())
    if is_main:
        print(f"SPNN total params: {total_params:,}")

    # ── LPIPS perceptual loss (frozen, not wrapped) ──
    lpips_fn = None
    if args.lambda_lpips > 0:
        lpips_fn = lpips.LPIPS(net="vgg").to(device)
        lpips_fn.eval()
        for p in lpips_fn.parameters():
            p.requires_grad = False
        if is_main:
            print("LPIPS loss enabled (VGG backbone)")

    # ── Test dataset for Penrose checks (rank 0 only) ──
    penrose_test_dataset = None
    if is_main:
        if args.dataset == "laion":
            penrose_test_dataset = LAIONAestheticDataset(
                data_dir=args.laion_dir, img_size=args.img_size,
                split="test", n_test=args.n_test,
            )
        elif args.dataset == "lsun_churches":
            penrose_test_dataset = LSUNChurchesDataset(
                img_size=args.img_size, split="test", n_test=args.n_test,
                data_dir=getattr(args, 'lsun_dir', None),
            )
        else:
            penrose_test_dataset = CelebAHQDataset(
                img_size=args.img_size, split="test", n_test=args.n_test,
            )
        print(f"Penrose check: {len(penrose_test_dataset)} test images, "
              f"sampling {args.penrose_batch_size} random each time")

    # ── Optimizer: trains ALL of s, t, r, mix ──
    optimizer = torch.optim.AdamW(spnn.parameters(), lr=args.lr, weight_decay=1e-5)

    # ── Accelerate prepare (wraps model with DDP, splits dataloader) ──
    spnn, optimizer, loader = accelerator.prepare(spnn, optimizer, loader)

    # ── Scheduler (created AFTER prepare so len(loader) is per-GPU) ──
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs * len(loader), eta_min=1e-6
    )
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    best_loss = float('inf')
    best_ckpt_path = os.path.join(args.output_dir, "spnn_vae_best.pt")
    start_epoch = 1

    # ── Resume from checkpoint ──
    if args.resume is not None:
        if is_main:
            print(f"{'Finetuning' if args.finetune else 'Resuming'} from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        unwrapped_spnn = accelerator.unwrap_model(spnn)
        unwrapped_spnn.load_state_dict(ckpt["model_state_dict"])
        if args.finetune:
            # Finetune mode: only load model weights, fresh optimizer/scheduler/epoch
            if is_main:
                print(f"  Finetune mode: fresh optimizer (lr={args.lr}), starting from epoch 1")
        else:
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "loss" in ckpt:
                best_loss = ckpt["loss"]
            if args.resume_epoch is None:
                raise ValueError("--resume_epoch is required when using --resume")
            start_epoch = args.resume_epoch + 1
            # Advance scheduler to match resumed epoch
            steps_done = (start_epoch - 1) * len(loader)
            scheduler.last_epoch = steps_done
            if is_main:
                print(f"  Resumed at epoch {start_epoch}, best_loss={best_loss:.6f}")

    # ── WandB (rank 0 only) ──
    if is_main:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   name=args.wandb_run_name, config=vars(args))

    for epoch in range(start_epoch, args.num_epochs + 1):
        spnn.train()
        epoch_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.num_epochs}",
                    disable=not is_main)
        unwrapped = accelerator.unwrap_model(spnn)

        for batch_idx, images in enumerate(pbar):
            images = images.to(device)

            # ── Get VAE targets: latent -> decoded image ──
            vae_latent, vae_decoded = get_vae_pairs(vae, images)

            # ── Encode through DDP (arms gradient sync reducer) ──
            z_spnn = spnn(images)

            # ── Decoder distillation loss: feed VAE latent, match VAE output ──
            spnn_decoded = unwrapped.decode(vae_latent)
            decoder_distill_loss = mse_loss(spnn_decoded, vae_decoded)

            # ── Decoder GT loss: feed VAE latent, match original image (L1) ──
            decoder_gt_loss = torch.tensor(0.0, device=device)
            if args.lambda_decoder_gt > 0:
                decoder_gt_loss = l1_loss(spnn_decoded, images)

            # ── LPIPS perceptual loss ──
            lpips_loss = torch.tensor(0.0, device=device)
            if lpips_fn is not None:
                lpips_target = images if args.lambda_decoder_gt > 0 else vae_decoded
                lpips_loss = lpips_fn(spnn_decoded, lpips_target).mean()

            # ── Cycle loss (surjectivity): encode(decode(z)) ≈ z ──
            cycle_loss = torch.tensor(0.0, device=device)
            if args.lambda_cycle > 0:
                re_encoded = unwrapped.encode(spnn_decoded)
                cycle_loss = mse_loss(re_encoded, vae_latent)

            # ── Roundtrip loss (pseudo-inverse stability): decode(encode(x)) ≈ x ──
            roundtrip_loss = torch.tensor(0.0, device=device)
            if args.lambda_roundtrip > 0:
                spnn_recon = unwrapped.decode(z_spnn)
                roundtrip_loss = mse_loss(spnn_recon, images)

            # ── Latent alignment loss: SPNN.encode(x) ≈ VAE.encode(x) ──
            align_loss = torch.tensor(0.0, device=device)
            if args.lambda_align > 0:
                with torch.no_grad():
                    z_vae = vae.encode(images).latent_dist.mode()
                align_loss = mse_loss(z_spnn, z_vae)

            # ── Perturbation consistency: dec(enc(dec(z_vae)+δ)) ≈ dec(z_vae)+δ ──
            # Simulates what DDNM does per step (BP correction on decoded latents).
            # Enforces smooth local invertibility of enc∘dec off the training manifold.
            perturb_loss = torch.tensor(0.0, device=device)
            if getattr(args, 'lambda_perturb', 0.0) > 0:
                x_start = spnn_decoded.detach()
                delta = torch.randn_like(x_start) * args.perturb_std
                x_perturbed = x_start + delta
                z_reencoded = unwrapped.encode(x_perturbed)
                x_recovered = unwrapped.decode(z_reencoded)
                perturb_loss = mse_loss(x_recovered, x_perturbed)

            loss = (args.lambda_decoder_distill * decoder_distill_loss
                    + args.lambda_decoder_gt * decoder_gt_loss
                    + args.lambda_lpips * lpips_loss
                    + args.lambda_cycle * cycle_loss
                    + args.lambda_roundtrip * roundtrip_loss
                    + args.lambda_align * align_loss
                    + getattr(args, 'lambda_perturb', 0.0) * perturb_loss)

            optimizer.zero_grad()
            accelerator.backward(loss)
            if args.max_grad_norm > 0:
                grad_norm = accelerator.clip_grad_norm_(
                    spnn.parameters(), max_norm=args.max_grad_norm)
            else:
                grad_norm = accelerator.clip_grad_norm_(
                    spnn.parameters(), max_norm=float('inf'))

            # Skip optimizer step if loss or gradients are NaN
            if torch.isfinite(loss) and torch.isfinite(torch.tensor(grad_norm)):
                optimizer.step()
                scheduler.step()
            elif is_main:
                print(f"  [!] Skipping step {batch_idx} — NaN detected (loss={loss.item()}, grad_norm={grad_norm})")

            epoch_loss += loss.item()

            if is_main:
                log_dict = {
                    "train/loss": loss.item(),
                    "train/decoder_distill_loss": decoder_distill_loss.item(),
                    "train/decoder_gt_loss": decoder_gt_loss.item(),
                    "train/lpips_loss": lpips_loss.item(),
                    "train/cycle_loss": cycle_loss.item(),
                    "train/roundtrip_loss": roundtrip_loss.item(),
                    "train/align_loss": align_loss.item(),
                    "train/perturb_loss": perturb_loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                }
                wandb.log(log_dict)

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })

        avg_loss = epoch_loss / len(loader)
        if is_main:
            wandb.log({"train/epoch_avg_loss": avg_loss, "epoch": epoch})
            print(f"  Epoch {epoch} — avg decoder loss: {avg_loss:.6f}")

        # ── Save best model (rank 0 only, unwrapped state dict) ──
        if avg_loss < best_loss:
            best_loss = avg_loss
            if is_main:
                unwrapped_spnn = accelerator.unwrap_model(spnn)
                ckpt_dict = {
                    "epoch": epoch,
                    "model_state_dict": unwrapped_spnn.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                }
                torch.save(ckpt_dict, best_ckpt_path)
                print(f"  New best loss: {avg_loss:.6f} — saved: {best_ckpt_path}")

        # ── Penrose identity + cycle-grid checks (rank 0 only) ──
        penrose_every = getattr(args, 'penrose_every', args.save_every)
        if epoch % penrose_every == 0 and is_main:
            unwrapped_spnn = accelerator.unwrap_model(spnn)
            # Sample a fresh random batch each time
            penrose_loader = DataLoader(
                penrose_test_dataset, batch_size=args.penrose_batch_size,
                shuffle=True, num_workers=0)
            penrose_images = next(iter(penrose_loader)).to(device)
            with torch.no_grad():
                # Encode in chunks to avoid VAE OOM on large batches
                chunk_size = 16
                latent_chunks = []
                for i in range(0, penrose_images.size(0), chunk_size):
                    chunk = penrose_images[i:i+chunk_size]
                    latent_chunks.append(vae.encode(chunk).latent_dist.mode())
                penrose_latent = torch.cat(latent_chunks, dim=0)
            del penrose_loader
            torch.cuda.empty_cache()
            p_metrics = penrose_check(unwrapped_spnn, penrose_images, penrose_latent, device)
            print_penrose_metrics(p_metrics)

            # Threshold-based pass/fail markers for the 3 pseudo-inverse identities
            threshold = getattr(args, 'penrose_threshold', 1e-7)
            penrose_id_keys = ["penrose/ggg_eq_g", "penrose/gpggp_eq_gp", "penrose/ggp_eq_id"]
            n_fail = 0
            for key in penrose_id_keys:
                val = p_metrics[key]
                marker = "[OK]" if val <= threshold else "[WARN]"
                if val > threshold:
                    n_fail += 1
                print(f"  {marker} {key} = {val:.2e}  (target: <= {threshold:.0e})")
            p_metrics["penrose/n_identities_above_threshold"] = n_fail
            wandb.log({**p_metrics, "epoch": epoch})

            # Save cycle consistency grid (5 encode→decode cycles)
            with torch.no_grad():
                save_cycle_comparison(unwrapped_spnn, penrose_images, epoch,
                                     train_sample_dir, num_cycles=5)

            spnn.train()

        # ── Numbered checkpoint save (rank 0 only, less frequent than penrose check) ──
        if epoch % args.save_every == 0 and is_main:
            unwrapped_spnn = accelerator.unwrap_model(spnn)
            ckpt_path = os.path.join(args.output_dir, f"spnn_vae_epoch{epoch:03d}.pt")
            ckpt_dict = {
                "epoch": epoch,
                "model_state_dict": unwrapped_spnn.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }
            torch.save(ckpt_dict, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # ── Final save (rank 0 only) ──
    if is_main:
        unwrapped_spnn = accelerator.unwrap_model(spnn)
        final_path = os.path.join(args.output_dir, "spnn_vae_final.pt")
        final_dict = {
            "epoch": args.num_epochs,
            "model_state_dict": unwrapped_spnn.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": best_loss,
        }
        torch.save(final_dict, final_path)
        print(f"\nTraining complete. Final model: {final_path}")
        print(f"The encoder (spnn.encode / forward) now works automatically —")
        print(f"it uses the same s, t, mix that were trained through the decoder.")
        wandb.finish()
