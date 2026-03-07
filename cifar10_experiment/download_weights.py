"""Download pretrained Simple LDM weights from HuggingFace."""

import os
from huggingface_hub import hf_hub_download

REPO_ID = "JuyeopDang/simple-latent-diffusion-models"
SLDM_ROOT = os.path.join(os.path.dirname(__file__), '..',
    'simple-latent-diffusion-model-master', 'simple-latent-diffusion-model')

# Files at repo root: cifar_vae.pth, cifar_diffusion.pth
# Download into SLDM_ROOT/models/ for our scripts
MODELS = {
    "cifar_vae.pth": os.path.join(SLDM_ROOT, "models", "cifar_vae.pth"),
    "cifar_diffusion.pth": os.path.join(SLDM_ROOT, "models", "cifar_diffusion.pth"),
}


def main():
    models_dir = os.path.join(SLDM_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)

    for hf_filename, local_path in MODELS.items():
        if os.path.exists(local_path):
            print(f"Already exists: {local_path}")
            continue
        print(f"Downloading {hf_filename} -> {local_path}")
        downloaded = hf_hub_download(repo_id=REPO_ID, filename=hf_filename)
        # Symlink or copy to our target location
        os.symlink(downloaded, local_path)
    print("Done.")


if __name__ == "__main__":
    main()
