import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import math
from diffusers import AutoencoderKL
import wandb
from models import SPNNAutoencoder
from diagnostics import penrose_check, print_penrose_metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────── Helpers ────────────────────────

def load_image(path, img_size):
    img = Image.open(path).convert("RGB")
    t = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    return t(img).unsqueeze(0).to(DEVICE)


def to_display(tensor):
    return ((tensor.cpu() + 1) / 2).clamp(0, 1)


def calc_psnr(a, b):
    m = F.mse_loss(a, b).item()
    return 10 * math.log10(4.0 / m) if m > 0 else float("inf")


def calc_mse(a, b):
    return F.mse_loss(a, b).item()


# ──────────────────────── Load models ────────────────────────

def load_sd_vae():
    print("Loading SD-VAE from timbrooks/instruct-pix2pix...")
    vae = AutoencoderKL.from_pretrained(
        "timbrooks/instruct-pix2pix", subfolder="vae"
    )
    vae.eval().to(DEVICE)
    for p in vae.parameters():
        p.requires_grad = False
    return vae


def load_spnn(ckpt_path):
    print(f"Loading SPNN from {ckpt_path}...")
    spnn = SPNNAutoencoder(mix_type="cayley", hidden=128, scale_bound=2.0)
    state = torch.load(ckpt_path, map_location=DEVICE)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    spnn.load_state_dict(state)
    spnn.eval().to(DEVICE)
    return spnn


# ──────────────────────── Single cycle functions ────────────────────────

@torch.no_grad()
def vae_cycle(vae, x):
    """SD-VAE encode -> decode. Lossy each pass."""
    posterior = vae.encode(x).latent_dist
    z = posterior.mode()
    return vae.decode(z).sample


@torch.no_grad()
def spnn_cycle_exact(spnn, x):
    """
    SPNN encode -> decode WITH side-channel latents.
    encode() uses the same s, t, mix that were trained in the decoder.
    decode() receives the x1 side-channels from encode, so it reuses
    the exact s(x1) and t(x1) — mathematically exact inversion.
    """
    latent, side = spnn.encode(x, return_latents=True)
    return spnn.decode(latent, latents=side)


# ──────────────────────── Main ────────────────────────


def run_test(args):
    print(f"Image:      {args.image}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device:     {DEVICE}")
    print(f"Cycles:     {args.num_cycles}\n")

    original = load_image(args.image, args.img_size)
    vae = load_sd_vae()
    spnn = load_spnn(args.checkpoint)

    wandb.init(
        project=args.wandb_project, entity=args.wandb_entity, job_type="test",
        config=vars(args),
    )

    # ── Penrose pseudo-inverse checks ──
    with torch.no_grad():
        spnn_latent = spnn.encode(original)
    print("Penrose pseudo-inverse diagnostics:")
    p_metrics = penrose_check(spnn, original, spnn_latent, DEVICE)
    print_penrose_metrics(p_metrics)
    wandb.log({"test/" + k: v for k, v in p_metrics.items()})
    print()

    vae_x = original.clone()
    spnn_x = original.clone()

    vae_imgs = [original.clone()]
    spnn_imgs = [original.clone()]

    header = (f"{'Cycle':<7} "
              f"{'VAE MSE':<13} {'VAE PSNR':<13} "
              f"{'SPNN MSE':<14} {'SPNN PSNR':<13}")
    print(header)
    print("-" * len(header))

    for i in range(1, args.num_cycles + 1):
        vae_x = vae_cycle(vae, vae_x)
        spnn_x = spnn_cycle_exact(spnn, spnn_x)

        vae_imgs.append(vae_x.clone())
        spnn_imgs.append(spnn_x.clone())

        vae_mse = calc_mse(vae_x, original)
        vae_psnr = calc_psnr(vae_x, original)
        spnn_mse = calc_mse(spnn_x, original)
        spnn_psnr = calc_psnr(spnn_x, original)

        wandb.log({
            "test/cycle": i,
            "test/vae_mse": vae_mse,
            "test/vae_psnr": vae_psnr,
            "test/spnn_mse": spnn_mse,
            "test/spnn_psnr": spnn_psnr,
        })

        print(f"{i:<7} "
              f"{vae_mse:<13.6f} {vae_psnr:<13.2f} "
              f"{spnn_mse:<14.2e} {spnn_psnr:<13.2f}")

    # ── Save full grid ──
    # Row 1: VAE   (original + cycles 1..10)
    # Row 2: SPNN  (original + cycles 1..10)
    row_vae = torch.cat([to_display(img) for img in vae_imgs], dim=0)
    row_spnn = torch.cat([to_display(img) for img in spnn_imgs], dim=0)
    grid = torch.cat([row_vae, row_spnn], dim=0)

    save_image(grid, "comparison2.png", nrow=args.num_cycles + 1, padding=4, pad_value=1.0)
    print(f"\nSaved: comparison2.png")
    print(f"  Row 1: VAE   (original -> cycle 1..{args.num_cycles})")
    print(f"  Row 2: SPNN  (original -> cycle 1..{args.num_cycles})")

    # ── Save compact summary ──
    summary = torch.cat([
        to_display(original),
        to_display(vae_imgs[-1]),
        to_display(spnn_imgs[-1]),
    ], dim=0)
    save_image(summary, "comparison_summary2.png", nrow=3, padding=4, pad_value=1.0)
    print(f"Saved: comparison_summary2.png (original | VAE@{args.num_cycles} | SPNN@{args.num_cycles})")

    wandb.finish()