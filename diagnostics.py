import os
import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def penrose_check(spnn, images, latents, device):
    """
    Compute Penrose pseudo-inverse identities (diagnostic only, not a loss).
    Returns a dict of metrics ready for wandb logging.

    g  = spnn.encode (forward)
    g' = spnn.decode (pinv)

    Checks:
      1. g(g'(g(x))) ≈ g(x)       encode is stable after decode round-trip
      2. g'(g(g'(z))) ≈ g'(z)      decode is stable after encode round-trip (cycle consistency)
      3. g(g'(z)) ≈ z              right-inverse: latent recovery
      4. roundtrip                  decode(encode(x)) ≈ x  (r-network quality)

    Args:
        spnn: SPNNAutoencoder model
        images: [B, 3, H, W] pixel-space images
        latents: [B, 4, H/8, W/8] latent codes (from VAE encoder or SPNN encoder)
        device: torch device

    Returns:
        dict of metric_name -> float, keys prefixed with "penrose/" for wandb
    """
    spnn.eval()
    x = images.to(device)
    z = latents.to(device)

    # g = encode, g' = decode
    gx = spnn.encode(x)                            # g(x)
    gpgx = spnn.decode(gx)                          # g'(g(x))
    ggpgx = spnn.encode(gpgx)                       # g(g'(g(x)))

    gpz = spnn.decode(z)                             # g'(z)
    ggpz = spnn.encode(gpz)                          # g(g'(z))
    gpggpz = spnn.decode(ggpz)                       # g'(g(g'(z)))

    # Roundtrip via r network
    latent = spnn.encode(x)
    roundtrip = spnn.decode(latent)

    mse = F.mse_loss
    metrics = {
        "penrose/ggg_eq_g":         mse(ggpgx, gx).item(),
        "penrose/gpggp_eq_gp":      mse(gpggpz, gpz).item(),
        "penrose/ggp_eq_id":        mse(ggpz, z).item(),
        "penrose/roundtrip":        mse(roundtrip, x).item(),
    }
    return metrics


def print_penrose_metrics(metrics):
    """Pretty-print penrose metrics to console."""
    print(f"  Roundtrip error (r network):       {metrics['penrose/roundtrip']:.6f}")
    print(f"  Penrose checks:")
    print(f"    g(g'(g(x))) ≈ g(x)    | MSE: {metrics['penrose/ggg_eq_g']:.2e}")
    print(f"    g'(g(g'(z))) ≈ g'(z)  | MSE: {metrics['penrose/gpggp_eq_gp']:.2e}")
    print(f"    g(g'(z)) ≈ z           | MSE: {metrics['penrose/ggp_eq_id']:.2e}")


@torch.no_grad()
def latent_alignment_check(spnn, vae, dataloader, device):
    """
    Test 1: Compare SPNN.encode(x) vs VAE.encode(x) over the full test set.
    Checks whether SPNN's latent space aligns with the VAE's, which is
    required for compatibility with pre-trained diffusion models.

    Returns dict with MSE and cosine similarity metrics.
    """
    spnn.eval()

    total_mse = 0.0
    total_cos = 0.0
    n = 0
    num_el = None

    for images in tqdm(dataloader, desc="Latent alignment"):
        images = images.to(device)

        z_spnn = spnn.encode(images)
        z_vae = vae.encode(images).latent_dist.mode()

        bs = images.size(0)
        if num_el is None:
            num_el = z_spnn[0].numel()

        total_mse += F.mse_loss(z_spnn, z_vae, reduction="sum").item()
        cos = F.cosine_similarity(
            z_spnn.view(bs, -1), z_vae.view(bs, -1), dim=1
        ).sum().item()
        total_cos += cos
        n += bs

    metrics = {
        "latent_align/mse": total_mse / (n * num_el),
        "latent_align/cosine_sim": total_cos / n,
        "latent_align/num_samples": n,
    }

    print(f"\n{'='*50}")
    print(f"Latent Alignment Check ({n} samples)")
    print(f"{'='*50}")
    print(f"  MSE(z_spnn, z_vae):    {metrics['latent_align/mse']:.6f}")
    print(f"  Cosine similarity:     {metrics['latent_align/cosine_sim']:.6f}")
    print(f"  (cosine=1.0 means perfect alignment)")

    return metrics


@torch.no_grad()
def cross_decode_check(spnn, vae, dataloader, device, output_dir, num_images=5):
    """
    Test 2: Encode with SPNN, decode with VAE. If VAE can decode SPNN's
    latents into good images, then SPNN encodes into the VAE's latent space
    and diffusion will work.

    Saves a 4-column grid per image:
        Original | VAE(VAE(x)) | VAE(SPNN(x)) | SPNN(SPNN(x))

    Returns list of (vae_mse, cross_mse) per image.
    """
    from torchvision.utils import save_image

    spnn.eval()
    os.makedirs(output_dir, exist_ok=True)

    def to_disp(t):
        return ((t.cpu() + 1) / 2).clamp(0, 1)

    count = 0
    all_mse = []

    print(f"\n{'='*50}")
    print(f"Cross-Decode Check ({num_images} images)")
    print(f"{'='*50}")
    print(f"  Grid: Original | VAE(VAE(x)) | VAE(SPNN(x)) | SPNN(SPNN(x))")
    print()

    for images in dataloader:
        images = images.to(device)
        for i in range(images.size(0)):
            if count >= num_images:
                break

            x = images[i:i+1]

            # VAE encode + decode (baseline)
            z_vae = vae.encode(x).latent_dist.mode()
            vae_recon = vae.decode(z_vae).sample

            # SPNN encode (produces scaled latent)
            z_spnn = spnn.encode(x)

            # Cross-decode: VAE decodes SPNN's latent (unscale for VAE)
            cross_recon = vae.decode(z_spnn).sample

            # SPNN roundtrip
            spnn_recon = spnn.decode(z_spnn)

            # Per-image metrics
            vae_mse = F.mse_loss(vae_recon, x).item()
            cross_mse = F.mse_loss(cross_recon, x).item()
            spnn_mse = F.mse_loss(spnn_recon, x).item()
            z_mse = F.mse_loss(z_spnn, z_vae).item()
            all_mse.append((vae_mse, cross_mse))

            print(f"  Image {count}: VAE recon MSE={vae_mse:.6f}  "
                  f"Cross-decode MSE={cross_mse:.6f}  "
                  f"SPNN roundtrip MSE={spnn_mse:.6f}  "
                  f"Latent MSE={z_mse:.6f}")

            # Save grid
            grid = torch.cat([
                to_disp(x), to_disp(vae_recon),
                to_disp(cross_recon), to_disp(spnn_recon),
            ], dim=0)
            path = os.path.join(output_dir, f"cross_decode_{count:03d}.png")
            save_image(grid, path, nrow=4, padding=4, pad_value=1.0)
            count += 1

        if count >= num_images:
            break

    print(f"\n  Saved {count} grids to {output_dir}/")
    print(f"  If cross-decode (col 3) looks good, SPNN latents are diffusion-compatible.")
    return all_mse
