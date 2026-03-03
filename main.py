import argparse
from train import train
from run_test_cycles import run_test


def parse_args():
    parser = argparse.ArgumentParser(description="SPNN VAE — train and/or test")

    # ── Mode flags ──
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--test", action="store_true", help="Run cycle consistency test")

    # ── Shared ──
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--wandb_project", type=str, default="spnn-vae")
    parser.add_argument("--wandb_entity", type=str, default="yamitehrlich-technion-israel-institute-of-technology")

    # ── Train args ──
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--sample_dir", type=str, default="samples")

    # ── Test args ──
    parser.add_argument("--image", type=str, default="test2.jpg")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/spnn_final.pt")
    parser.add_argument("--num_cycles", type=int, default=10)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.train and not args.test:
        print("Specify --train and/or --test")
        exit(1)

    if args.train:
        train(args)

    if args.test:
        run_test(args)
