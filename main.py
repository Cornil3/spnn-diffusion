import argparse
import os
import wandb
from train import train
from run_test_cycles import run_test


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

    parser.add_argument("--r_loss_weight", type=float, default=0.01,
                        help="Weight of direct r supervision loss")
    parser.add_argument("--lambda_lpips", type=float, default=0.1,
                        help="Weight of LPIPS perceptual loss (0 to disable)")

    # ── Train args ──
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--penrose_batch_size", type=int, default=64)
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
        args.checkpoint = os.path.join(args.output_dir, "spnn_vae_epoch015.pt")

    # ── Dynamic wandb run name ──
    mode = "+".join(filter(None, ["train" if args.train else None, "test" if args.test else None]))
    args.wandb_run_name = (
        f"{mode}_ep{args.num_epochs}_bs{args.batch_size}_lr{args.lr:.0e}"
        f"_sb{args.scale_bound}_rw{args.r_loss_weight}_lpips{args.lambda_lpips}"
        f"_h{args.hidden}_{os.path.basename(args.output_dir)}"
    )

    return args


if __name__ == "__main__":
    args = parse_args()

    if not args.train and not args.test:
        print("Specify --train and/or --test")
        exit(1)

    wandb.init(project=args.wandb_project, entity=args.wandb_entity,
               name=args.wandb_run_name, config=vars(args))

    if args.train:
        train(args)

    if args.test:
        run_test(args)

    wandb.finish()
