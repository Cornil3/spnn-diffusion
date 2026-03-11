import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from diffusers import StableDiffusionInstructPix2PixPipeline
from models import SPNNAutoencoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SPNNVAE(nn.Module):
    def __init__(self, spnn, original_vae):
        super().__init__()
        self.spnn = spnn
        self.config = original_vae.config

    @property
    def device(self): return next(self.spnn.parameters()).device

    @property
    def dtype(self): return next(self.spnn.parameters()).dtype

    def encode(self, x):
        z = self.spnn.encode(x)

        class DummyDist:
            def mode(self): return z

        class DummyOutput:
            def __init__(self, dist): self.latent_dist = dist

        return DummyOutput(DummyDist())

    def decode(self, z, return_dict=True):
        image = self.spnn.decode(z)

        if not return_dict:
            return (image,)

        class DummyOutput:
            def __init__(self, sample): self.sample = sample

        return DummyOutput(image)


def main():
    img_path = "test.jpg"
    spnn_ckpt = "checkpoints_spnn_fixed/spnn_vae_final.pt"
    output_path = "cycle_10_comparison.png"
    num_cycles = 10

    pipe_vae = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix", torch_dtype=torch.float32, safety_checker=None,
    ).to(DEVICE)

    pipe_spnn = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix", torch_dtype=torch.float32, safety_checker=None,
    ).to(DEVICE)
    spnn = SPNNAutoencoder(mix_type='cayley', hidden=192, scale_bound=2.0).to(DEVICE)
    ckpt = torch.load(spnn_ckpt, map_location=DEVICE)
    spnn.load_state_dict(ckpt.get("model_state_dict", ckpt))
    spnn.eval()
    pipe_spnn.vae = SPNNVAE(spnn, pipe_spnn.vae)

    init_image = Image.open(img_path).convert("RGB").resize((512, 512))

    curr_img_vae = init_image
    curr_img_spnn = init_image

    vae_results = [init_image]
    spnn_results = [init_image]

    for i in range(num_cycles):
        prompt = "make the image a bit brighter"

        seed = 42 + i

        curr_img_vae = pipe_vae(
            prompt=prompt,
            image=curr_img_vae,
            num_inference_steps=100,
            generator=torch.Generator(device=DEVICE).manual_seed(seed)
        ).images[0]
        vae_results.append(curr_img_vae)

        curr_img_spnn = pipe_spnn(
            prompt=prompt,
            image=curr_img_spnn,
            num_inference_steps=100,
            generator=torch.Generator(device=DEVICE).manual_seed(seed)
        ).images[0]
        spnn_results.append(curr_img_spnn)

    to_tensor = ToTensor()

    vae_tensors = [to_tensor(img) for img in vae_results]
    spnn_tensors = [to_tensor(img) for img in spnn_results]

    vae_row = torch.stack(vae_tensors)
    spnn_row = torch.stack(spnn_tensors)

    grid_tensors = torch.cat([vae_row, spnn_row], dim=0)

    save_image(grid_tensors, output_path, nrow=num_cycles + 1, padding=2, pad_value=1.0)


if __name__ == "__main__":
    main()