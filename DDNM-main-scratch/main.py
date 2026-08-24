import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb

# from runners.diffusion import Diffusion
from guided_diffusion.diffusion import Diffusion

torch.set_printoptions(sci_mode=False)

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Set different seeds for diverse results")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--deg", type=str, required=True, help="Degradation"
    )
    parser.add_argument(
        "--path_y",
        type=str,
        default="",
        help="Path of the test dataset (pixel DDNM). Latent DDNM uses config.data.dataset_root.",
    )
    parser.add_argument(
        "--sigma_y", type=float, default=0., help="sigma_y"
    )
    parser.add_argument(
        "--eta", type=float, default=0.85, help="Eta"
    )    
    parser.add_argument(
        "--simplified",
        action="store_true",
        help="Use simplified DDNM, without SVD",
    )    
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--deg_scale", type=float, default=0., help="deg_scale"
    )    
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument(
        "--ni",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument(
        '--subset_start', type=int, default=-1
    )
    parser.add_argument(
        '--subset_end', type=int, default=-1
    )
    parser.add_argument(
        "-n",
        "--noise_type",
        type=str,
        default="gaussian",
        help="gaussian | 3d_gaussian | poisson | speckle"
    )
    parser.add_argument(
        "--add_noise",
        action="store_true"
    )

    # ---- latent DDNM (SD1.5 + {VAE, SPNN-512}) ----
    parser.add_argument(
        "--codec_type", type=str, default=None, choices=["vae", "spnn"],
        help="Override config.codec.type. 'vae'=VAE only; 'spnn'=run VAE then SPNN-512 (compared).",
    )
    parser.add_argument(
        "--prompt", type=str, default="",
        help="Text prompt for the SD1.5 prior. Empty = unconditional (default).",
    )
    parser.add_argument(
        "--prompt_file", type=str, default="",
        help="Read the prompt from this file instead (avoids shell quoting). Overrides --prompt.",
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=-1.0,
        help="Classifier-free guidance scale. <0 = use config (1.0 = off). Needs a non-empty prompt.",
    )
    parser.add_argument(
        "--mask", type=str, default="",
        help="Path to an inpainting mask (.npy or image; 1=known). Overrides the generated mask.",
    )
    parser.add_argument(
        "--mask_kind", type=str, default="", choices=["", "box", "half", "random"],
        help="Generated-mask shape when --mask is unset (default from config.inpainting).",
    )
    parser.add_argument(
        "--mask_box_frac", type=float, default=-1.0,
        help="Central-hole side as a fraction of the image (box) or drop-prob (random). <0 = config.",
    )

    # ---- data-consistency lambda schedule + damping (like DDNM-main/diffusion.py) ----
    parser.add_argument(
        "--lambda_mode", type=str, default="const",
        choices=["const", "linear", "exp", "noise"],
        help="lambda_t schedule: const=lambda_val; linear/exp decay lambda_val->lambda_floor "
             "(exp tempo=--lambda_rate); noise tracks remaining noise (1-a_next).",
    )
    parser.add_argument(
        "--lambda_val", type=float, default=1.0,
        help="lambda_t peak / step-0 value (const uses exactly this).",
    )
    parser.add_argument(
        "--lambda_floor", type=float, default=0.0,
        help="lambda_t asymptotic floor the decay schedules approach.",
    )
    parser.add_argument(
        "--lambda_rate", type=float, default=5.0,
        help="Exp-decay tempo for --lambda_mode exp (higher = faster drop-off).",
    )
    parser.add_argument(
        "--damping_floor", type=float, default=1.0,
        help="Damp the et (direction) term: damp = 1 - a_next*(1-floor). "
             "1.0 = no damping; 0.0 = et fully off as a_next->1 (end of sampling).",
    )
    parser.add_argument(
        "--T_sampling", type=int, default=-1,
        help="Override config time_travel.T_sampling (reverse-diffusion steps). -1 = use config.",
    )
    parser.add_argument(
        "--travel_length", type=int, default=-1,
        help="RePaint time-travel jump length. -1 = use config (1 = off).",
    )
    parser.add_argument(
        "--travel_repeat", type=int, default=-1,
        help="RePaint resampling repeats per jump. -1 = use config (1 = off).",
    )
    parser.add_argument(
        "--no_final_known_pin", dest="final_known_pin", action="store_false",
        help="Disable the post-sampling pin that forces the unmasked region exactly to the "
             "measurement (inpainting only). On by default.",
    )
    parser.set_defaults(final_known_pin=True)



    args = parser.parse_args()

    # Resolve prompt from a file if given (sidesteps shell quoting of multi-word prompts).
    if args.prompt_file:
        with open(args.prompt_file, "r") as pf:
            args.prompt = pf.read().strip()

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    # CLI override for the latent codec choice.
    if args.codec_type is not None and hasattr(new_config, "codec"):
        new_config.codec.type = args.codec_type
    # CLI overrides for the sampling schedule (steps / RePaint time-travel).
    if hasattr(new_config, "time_travel"):
        if args.T_sampling > 0:    new_config.time_travel.T_sampling = args.T_sampling
        if args.travel_length > 0: new_config.time_travel.travel_length = args.travel_length
        if args.travel_repeat > 0: new_config.time_travel.travel_repeat = args.travel_repeat

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
    args.image_folder = os.path.join(
        args.exp, "image_samples", args.image_folder
    )
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    else:
        overwrite = False
        if args.ni:
            overwrite = True
        else:
            response = input(
                f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)"
            )
            if response.upper() == "Y":
                overwrite = True

        if overwrite:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        else:
            print("Output image folder exists. Program halted.")
            sys.exit(0)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    try:
        model_type = str(getattr(config.model, "type", ""))
        if model_type.startswith("latent"):
            # Latent-space DDNM (SD1.5 prior + VAE/SPNN codec).
            from guided_diffusion.latent_diffusion import LatentDiffusion
            runner = LatentDiffusion(args, config)
            runner.sample()
        else:
            # Official pixel-space DDNM.
            runner = Diffusion(args, config)
            runner.sample(args.simplified)
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
