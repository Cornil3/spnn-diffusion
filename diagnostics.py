import torch
import torch.nn.functional as F


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
