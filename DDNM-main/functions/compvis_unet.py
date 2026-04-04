"""
Load CompVis LDM UNet directly from a .ckpt checkpoint.

Uses the CompVis/latent-diffusion repo's UNetModel. Requires:
    pip install git+https://github.com/CompVis/latent-diffusion.git

If that's not available, falls back to a manual reconstruction from the
openai diffusion modules.
"""

import torch


def load_compvis_unet(ckpt_path, device):
    """
    Load the UNet from a CompVis LDM checkpoint.

    The checkpoint contains the full LDM (VAE + UNet + etc).
    We extract only the UNet weights under 'model.diffusion_model.*'.

    Args:
        ckpt_path: path to the .ckpt file
        device: torch device

    Returns:
        unet: nn.Module with forward(x, t) -> noise prediction
    """
    print(f"Loading CompVis checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    # Extract UNet weights (strip prefix)
    unet_sd = {}
    for k, v in sd.items():
        if k.startswith("model.diffusion_model."):
            unet_sd[k.replace("model.diffusion_model.", "")] = v

    print(f"  Extracted {len(unet_sd)} UNet weight tensors")

    # Try loading via the ldm package (CompVis repo)
    try:
        from ldm.modules.diffusionmodules.openaimodel import UNetModel
        print("  Using ldm.modules.diffusionmodules.openaimodel.UNetModel")
    except ImportError:
        raise ImportError(
            "CompVis ldm package not found. Install with:\n"
            "  pip install git+https://github.com/CompVis/latent-diffusion.git"
        )

    # LSUN Churches KL-8 UNet config
    # From: configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml
    unet = UNetModel(
        image_size=32,
        in_channels=4,
        out_channels=4,
        model_channels=192,
        channel_mult=[1, 2, 2, 4, 4],
        attention_resolutions=[1, 2, 4, 8],
        num_res_blocks=2,
        num_heads=8,
        use_scale_shift_norm=True,
        resblock_updown=True,
    )

    # Load weights
    missing, unexpected = unet.load_state_dict(unet_sd, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys (first 5: {missing[:5]})")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys (first 5: {unexpected[:5]})")
    if not missing and not unexpected:
        print("  All weights loaded successfully")

    unet.eval().to(device)
    for p in unet.parameters():
        p.requires_grad = False

    print(f"  UNet params: {sum(p.numel() for p in unet.parameters()):,}")
    return unet
