"""make variations of input image"""

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2img-samples"
    )

    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )

    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save indiviual samples. For speed measurements.",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=2,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--spnn_checkpoint",
        type=str,
        default=None,
        help="Optional: path to SPNN ckpt_last.pt to replace the SD 1.5 VAE.",
    )
    parser.add_argument(
        "--spnn_weights",
        type=str,
        default="ema",
        choices=["ema", "model"],
        help="Which state_dict key inside the SPNN ckpt to load.",
    )
    parser.add_argument(
        "--inpainting_mask",
        type=str,
        default=None,
        help="Optional: path to binary mask PNG (same size as init image). "
             "White (1) = inpaint (regenerate), black (0) = observed (keep). "
             "Enables per-step DDNM back-projection with A(x) = observed_mask * x, Ap = A.",
    )
    parser.add_argument(
        "--bp_schedule",
        type=str,
        default="every",
        choices=["every", "phased"],
        help="BP schedule. 'every' (default): BP every step with lambda=1.0. "
             "'phased': steps 0-2 lambda=1.0, steps 3-12 skip, then --bp_lambda every 10 steps.",
    )
    parser.add_argument(
        "--bp_lambda",
        type=float,
        default=0.5,
        help="Only used with --bp_schedule phased. Lambda for the every-10-steps late phase.",
    )
    parser.add_argument(
        "--mask_feather_sigma",
        type=float,
        default=0.0,
        help="Optional: Gaussian sigma (pixels) to blur the binary inpainting mask. "
             "0 = binary mask (hard boundary). >0 = soft observed_mask, removes the "
             "step-function discontinuity at the mask edge (helps invertible codecs "
             "like SPNN stay in-distribution). Typical values: 8-24 (~1-3 latent px).",
    )
    parser.add_argument(
        "--x0_smooth_sigma",
        type=float,
        default=0.0,
        help="Optional: Gaussian sigma (pixels) to smooth the entire x0_t_hat pixel "
             "image AFTER the BP correction and BEFORE re-encoding into latent. "
             "Removes high-freq content SPNN's invertibility would transmit into the "
             "latent OOD-territory. 0 = disabled. Typical values: 1-4.",
    )
    parser.add_argument(
        "--debug_dir",
        type=str,
        default=None,
        help="Optional: dir to save decoded x0_t and x0_t_hat at every DDIM step.",
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.spnn_checkpoint:
        sys.path.insert(0, "/home/yamitehrlich/work/spnn-diffusion")
        from imagenet_latent_ddnm.spnn_model import SPNNAutoencoder512
        spnn = SPNNAutoencoder512(mix_type="householder", hidden=128,
                                  r_hidden=256, scale_bound=1.0)
        state = torch.load(opt.spnn_checkpoint, map_location="cpu",
                           weights_only=False)
        sd = state[opt.spnn_weights]
        stripped = {k[len("spnn."):]: v for k, v in sd.items()
                    if k.startswith("spnn.")}
        missing, unexpected = spnn.load_state_dict(stripped, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"SPNN load mismatch: missing={len(missing)} "
                f"unexpected={len(unexpected)}; first_missing={list(missing)[:5]}"
            )
        spnn.eval().requires_grad_(False).to(device)

        class SPNNFirstStage(torch.nn.Module):
            """Minimal shim exposing (encode, decode) as CompVis LatentDiffusion
            expects. SPNN-512 emits SD's prescaled latents (already x0.18215),
            but LatentDiffusion multiplies by scale_factor after encode and
            divides before decode - so we invert those two ops here."""
            def __init__(self, spnn, sf):
                super().__init__()
                self.spnn = spnn
                self.sf = sf
            def encode(self, x):
                return self.spnn.encode(x) / self.sf
            def decode(self, z):
                return self.spnn.decode(z * self.sf)

        model.first_stage_model = SPNNFirstStage(spnn, model.scale_factor).to(device)
        print(f"Swapped VAE with SPNN from {opt.spnn_checkpoint} "
              f"(weights='{opt.spnn_weights}')")

    if opt.plms:
        raise NotImplementedError("PLMS sampler not (yet) supported")
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    assert os.path.isfile(opt.init_img)
    init_image = load_img(opt.init_img).to(device)
    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

    if opt.inpainting_mask:
        mask_img = Image.open(opt.inpainting_mask).convert("L").resize(
            (init_image.shape[-1], init_image.shape[-2]), resample=Image.NEAREST)
        mask_np = np.array(mask_img).astype(np.float32) / 255.0
        inpaint_mask = torch.from_numpy(mask_np)[None, None].to(device)   # (1,1,H,W)
        observed_mask = 1.0 - inpaint_mask                                # 1 = keep observed

        if opt.mask_feather_sigma > 0:
            from torchvision.transforms.functional import gaussian_blur
            sigma = float(opt.mask_feather_sigma)
            ksize = 2 * int(3 * sigma) + 1                                # odd kernel
            observed_mask = gaussian_blur(observed_mask, kernel_size=ksize, sigma=sigma)
            print(f"Feathered observed_mask with sigma={sigma}px (kernel={ksize})")

        # Inpainting degradation: A(x) = M * x, Ap = A (self-adjoint idempotent).
        def A_inpaint(x):  return observed_mask * x
        Ap_inpaint = A_inpaint
        y_inpaint  = A_inpaint(init_image)                                # fixed measurement

        sampler.bp_A  = A_inpaint
        sampler.bp_Ap = Ap_inpaint
        sampler.bp_y  = y_inpaint
        sampler.x0_smooth_sigma = opt.x0_smooth_sigma

        if opt.bp_schedule == "phased":
            def bp_lambda_fn(step_idx, lam=opt.bp_lambda):
                if step_idx <= 2:
                    return 1.0
                if step_idx <= 12:
                    return 0.0
                if (step_idx - 13) % 10 == 0:
                    return lam
                return 0.0
            sampler.bp_lambda_fn = bp_lambda_fn
            sched_desc = (f"phased — steps 0-2 lambda=1.0, steps 3-12 skip, "
                          f"then lambda={opt.bp_lambda} every 10 steps (13, 23, 33, ...)")
        else:
            # No lambda_fn -> defaults to lambda=1.0 every step in ddim.py
            sched_desc = "every step, lambda=1.0"
        print(f"BP inpainting enabled — mask: {opt.inpainting_mask}, "
              f"observed frac: {observed_mask.mean().item():.3f}")
        print(f"BP schedule: {sched_desc}")

    if opt.debug_dir:
        os.makedirs(opt.debug_dir, exist_ok=True)
        sampler.debug_dir = opt.debug_dir
        print(f"Per-step debug enabled — x0_t / x0_t_hat -> {opt.debug_dir}")

    sampler._step_idx = 0
    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

    assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(opt.strength * opt.ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)

                        # encode (scaled latent)
                        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                        # decode it
                        samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,)

                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        if not opt.skip_save:
                            for x_sample in x_samples:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                Image.fromarray(x_sample.astype(np.uint8)).save(
                                    os.path.join(sample_path, f"{base_count:05}.png"))
                                base_count += 1
                        all_samples.append(x_samples)

                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    grid_count += 1

                toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
