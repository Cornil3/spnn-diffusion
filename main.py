import argparse
import os
import torch
import wandb
from train import train
from run_test_cycles import run_test
from diagnostics import latent_alignment_check, cross_decode_check


def parse_args():
    parser = argparse.ArgumentParser(description="SPNN VAE — train and/or test")

    # ── Mode flags ──
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--test", action="store_true", help="Run cycle consistency test")

    # ── Dataset ──
    parser.add_argument("--dataset", type=str, default="celebahq",
                        choices=["celebahq", "laion"],
                        help="Dataset to use for training")
    parser.add_argument("--laion_dir", type=str, default=None,
                        help="Path to LAION img2dataset output dir (required if --dataset=laion)")

    # ── Shared ──
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--n_test", type=int, default=1000,
                        help="Number of test images (rest used for training)")
    parser.add_argument("--wandb_project", type=str, default="spnn-vae")
    parser.add_argument("--wandb_entity", type=str, default="yamitehrlich-technion-israel-institute-of-technology")

    # ── Model ──
    parser.add_argument("--mix_type", type=str, default="cayley", choices=["cayley", "householder"])
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--scale_bound", type=float, default=2.0)

    parser.add_argument("--lambda_decoder", type=float, default=1.0,
                        help="Weight of decoder MSE loss")
    parser.add_argument("--lambda_lpips", type=float, default=0.5,
                        help="Weight of LPIPS perceptual loss (0 to disable)")
    parser.add_argument("--lambda_cycle", type=float, default=0.3,
                        help="Weight of cycle/surjectivity loss (0 to disable)")
    parser.add_argument("--lambda_roundtrip", type=float, default=0.3,
                        help="Weight of roundtrip/pseudo-inverse stability loss (0 to disable)")
    parser.add_argument("--lambda_gan", type=float, default=0.0,
                        help="Weight of PatchGAN adversarial loss (0 to disable)")
    parser.add_argument("--lambda_align", type=float, default=0.5,
                        help="Weight of latent alignment loss (0 to disable)")

    # ── Train args ──
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping (0 to disable)")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--penrose_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="checkpoints_lpips")
    parser.add_argument("--sample_dir", type=str, default="samples")

    # ── Test args ──
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path for testing (default: <output_dir>/spnn_vae_final.pt)")
    parser.add_argument("--num_cycles", type=int, default=10)
    parser.add_argument("--num_save_images", type=int, default=30)

    args = parser.parse_args()

    if args.checkpoint is None:
        args.checkpoint = os.path.join(args.output_dir, "spnn_vae_final.pt")

    # ── Dynamic wandb run name ──
    mode = "+".join(filter(None, ["train" if args.train else None, "test" if args.test else None]))
    args.wandb_run_name = (
        f"{mode}_ep{args.num_epochs}_bs{args.batch_size}_lr{args.lr:.0e}"
        f"_sb{args.scale_bound}_lpips{args.lambda_lpips}"
        f"_cyc{args.lambda_cycle}_rt{args.lambda_roundtrip}_gan{args.lambda_gan}_align{args.lambda_align}"
        f"_h{args.hidden}_{os.path.basename(args.output_dir)}"
    )

    return args


def run_latent_diagnostics(args):
    """Run latent alignment and cross-decode checks against the VAE."""
    from torch.utils.data import DataLoader
    from diffusers import AutoencoderKL
    from models import SPNNAutoencoder
    from dataset import CelebAHQDataset, LAIONAestheticDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*50}")
    print(f"Latent Space Diagnostics")
    print(f"{'='*50}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")

    # Load VAE
    print("Loading SD-VAE...")
    vae = AutoencoderKL.from_pretrained("timbrooks/instruct-pix2pix", subfolder="vae")
    vae.eval().to(device)
    for p in vae.parameters():
        p.requires_grad = False

    # Load SPNN
    print(f"Loading SPNN from {args.checkpoint}...")
    spnn = SPNNAutoencoder(mix_type=args.mix_type, hidden=args.hidden, r_hidden=args.hidden * 2, scale_bound=args.scale_bound)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    spnn.load_state_dict(state)
    spnn.eval().to(device)

    # Test dataset
    if args.dataset == "laion":
        test_dataset = LAIONAestheticDataset(
            data_dir=args.laion_dir, img_size=args.img_size,
            split="test", n_test=args.n_test,
        )
    else:
        test_dataset = CelebAHQDataset(
            img_size=args.img_size, split="test", n_test=args.n_test,
        )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"Test set: {len(test_dataset)} images")

    # Test 1: Latent alignment (all test samples)
    latent_alignment_check(spnn, vae, test_loader, device)

    # Test 2: Cross-decode (5 images)
    diag_dir = "latent_diagnostics"
    cross_decode_check(spnn, vae, test_loader, device, diag_dir, num_images=5)


if __name__ == "__main__":
    args = parse_args()

    if not args.train and not args.test and not os.path.exists(args.checkpoint):
        print("Specify --train and/or --test, or provide a valid --checkpoint for diagnostics")
        exit(1)

    if args.train:
        train(args)

    # After accelerate launch, only rank 0 should run test/diagnostics
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0:
        exit(0)

    if args.test:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   name=args.wandb_run_name, config=vars(args))
        run_test(args)
        wandb.finish()

    # Latent space diagnostics — runs whenever a checkpoint exists
    if os.path.exists(args.checkpoint):
        run_latent_diagnostics(args)
