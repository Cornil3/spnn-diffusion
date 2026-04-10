import os
import logging
import time
import glob

import numpy as np
from PIL import Image
import tqdm
import torch
import torch.utils.data as data

from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path, download
from functions.svd_ddnm import ddnm_diffusion, ddnm_plus_diffusion

import torchvision.utils as tvu

from guided_diffusion.models import Model
from guided_diffusion.script_util import create_model, create_classifier, classifier_defaults, args_to_dict
import random

from scipy.linalg import orth


def get_gaussian_noisy_img(img, noise_level):
    return img + torch.randn_like(img).cuda() * noise_level

def MeanUpsample(x, scale):
    n, c, h, w = x.shape
    out = torch.zeros(n, c, h, scale, w, scale).to(x.device) + x.view(n,c,h,1,w,1)
    out = out.view(n, c, scale*h, scale*w)
    return out

def color2gray(x):
    coef=1/3
    x = x[:,0,:,:] * coef + x[:,1,:,:]*coef +  x[:,2,:,:]*coef
    return x.repeat(1,3,1,1)

def gray2color(x):
    x = x[:,0,:,:]
    coef=1/3
    base = coef**2 + coef**2 + coef**2
    return torch.stack((x*coef/base, x*coef/base, x*coef/base), 1)    



def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self, simplified):
        # Latent SD 1.5 mode — bypass pixel-space model loading entirely
        if hasattr(self.config.model, 'type') and self.config.model.type == 'latent_sd15':
            print('Run Latent DDNM (SD 1.5 UNet + codec).',
                  f'{self.config.time_travel.T_sampling} sampling steps.',
                  f'travel_length = {self.config.time_travel.travel_length},',
                  f'travel_repeat = {self.config.time_travel.travel_repeat}.',
                  f'Task: {self.args.deg}.'
                  )
            self.latent_ddnm()
            return

        # CompVis unconditional LDM mode (e.g. LSUN Churches)
        if hasattr(self.config.model, 'type') and self.config.model.type == 'latent_compvis_ldm':
            print('Run Latent DDNM (CompVis unconditional UNet + codec).',
                  f'{self.config.time_travel.T_sampling} sampling steps.',
                  f'Task: {self.args.deg}.'
                  )
            self.latent_compvis_ldm()
            return

        cls_fn = None
        if self.config.model.type == 'simple':
            model = Model(self.config)

            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            elif self.config.data.dataset == 'CelebA_HQ':
                name = 'celeba_hq'
            else:
                raise ValueError
            if name != 'celeba_hq':
                ckpt = get_ckpt_path(f"ema_{name}", prefix=self.args.exp)
                print("Loading checkpoint {}".format(ckpt))
            elif name == 'celeba_hq':
                ckpt = os.path.join(self.args.exp, "logs/celeba/celeba_hq.ckpt")
                if not os.path.exists(ckpt):
                    download('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt',
                             ckpt)
            else:
                raise ValueError
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        elif self.config.model.type == 'openai':
            config_dict = vars(self.config.model)
            model = create_model(**config_dict)
            if self.config.model.use_fp16:
                model.convert_to_fp16()
            if self.config.model.class_cond:
                ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_diffusion.pt' % (
                self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_diffusion_uncond.pt' % (
                        self.config.data.image_size, self.config.data.image_size), ckpt)
            else:
                ckpt = os.path.join(self.args.exp, "logs/imagenet/256x256_diffusion_uncond.pt")
                if not os.path.exists(ckpt):
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt',
                        ckpt)

            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model.eval()
            model = torch.nn.DataParallel(model)

            if self.config.model.class_cond:
                ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_classifier.pt' % (
                self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    image_size = self.config.data.image_size
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_classifier.pt' % image_size,
                        ckpt)
                classifier = create_classifier(**args_to_dict(self.config.classifier, classifier_defaults().keys()))
                classifier.load_state_dict(torch.load(ckpt, map_location=self.device))
                classifier.to(self.device)
                if self.config.classifier.classifier_use_fp16:
                    classifier.convert_to_fp16()
                classifier.eval()
                classifier = torch.nn.DataParallel(classifier)

                import torch.nn.functional as F
                def cond_fn(x, t, y):
                    with torch.enable_grad():
                        x_in = x.detach().requires_grad_(True)
                        logits = classifier(x_in, t)
                        log_probs = F.log_softmax(logits, dim=-1)
                        selected = log_probs[range(len(logits)), y.view(-1)]
                        return torch.autograd.grad(selected.sum(), x_in)[0] * self.config.classifier.classifier_scale

                cls_fn = cond_fn

        if simplified:
            print('Run Simplified DDNM, without SVD.',
                  f'{self.config.time_travel.T_sampling} sampling steps.',
                  f'travel_length = {self.config.time_travel.travel_length},',
                  f'travel_repeat = {self.config.time_travel.travel_repeat}.',
                  f'Task: {self.args.deg}.'
                 )
            self.simplified_ddnm_plus(model, cls_fn)
        else:
            print('Run SVD-based DDNM.',
                  f'{self.config.time_travel.T_sampling} sampling steps.',
                  f'travel_length = {self.config.time_travel.travel_length},',
                  f'travel_repeat = {self.config.time_travel.travel_repeat}.',
                  f'Task: {self.args.deg}.'
                 )
            self.svd_based_ddnm_plus(model, cls_fn)
            
            
    def simplified_ddnm_plus(self, model, cls_fn, codec_name=None):
        args, config = self.args, self.config

        # Check if running in latent mode (codec set by latent_ddnm)
        codec = getattr(self, '_codec', None)
        latent_shape = getattr(self, '_latent_shape', None)
        is_latent = codec is not None

        dataset, test_dataset = get_dataset(args, config)

        device_count = torch.cuda.device_count()

        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)

        print(f'Dataset has size {len(test_dataset)}')

        def seed_worker(worker_id):
            worker_seed = args.seed % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(args.seed)
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )

        # get degradation operator (always in pixel space)
        print("args.deg:",args.deg)
        img_size = config.data.image_size
        if args.deg =='colorization':
            A = lambda z: color2gray(z)
            Ap = lambda z: gray2color(z)
        elif args.deg =='denoising':
            A = lambda z: z
            Ap = A
        elif args.deg =='sr_averagepooling':
            scale=round(args.deg_scale)
            A = torch.nn.AdaptiveAvgPool2d((img_size//scale,img_size//scale))
            Ap = lambda z: MeanUpsample(z,scale)
        elif args.deg =='inpainting':
            loaded = np.load("exp/inp_masks/mask.npy")
            mask = torch.from_numpy(loaded).to(self.device)
            if mask.shape[-1] != img_size:
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0).unsqueeze(0).float(),
                    size=(img_size, img_size), mode='nearest'
                ).squeeze(0).squeeze(0).to(mask.dtype)
            A = lambda z: z*mask
            Ap = A
        elif args.deg =='mask_color_sr':
            loaded = np.load("exp/inp_masks/mask.npy")
            mask = torch.from_numpy(loaded).to(self.device)
            if mask.shape[-1] != img_size:
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0).unsqueeze(0).float(),
                    size=(img_size, img_size), mode='nearest'
                ).squeeze(0).squeeze(0).to(mask.dtype)
            A1 = lambda z: z*mask
            A1p = A1

            A2 = lambda z: color2gray(z)
            A2p = lambda z: gray2color(z)

            scale=round(args.deg_scale)
            A3 = torch.nn.AdaptiveAvgPool2d((img_size//scale,img_size//scale))
            A3p = lambda z: MeanUpsample(z,scale)

            A = lambda z: A3(A2(A1(z)))
            Ap = lambda z: A1p(A2p(A3p(z)))
        elif args.deg =='diy':
            loaded = np.load("exp/inp_masks/mask.npy")
            mask = torch.from_numpy(loaded).to(self.device)
            if mask.shape[-1] != img_size:
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0).unsqueeze(0).float(),
                    size=(img_size, img_size), mode='nearest'
                ).squeeze(0).squeeze(0).to(mask.dtype)
            A1 = lambda z: z*mask
            A1p = A1

            A2 = lambda z: color2gray(z)
            A2p = lambda z: gray2color(z)

            scale=args.deg_scale
            A3 = torch.nn.AdaptiveAvgPool2d((img_size//scale,img_size//scale))
            A3p = lambda z: MeanUpsample(z,scale)

            A = lambda z: A3(A2(A1(z)))
            Ap = lambda z: A1p(A2p(A3p(z)))
        else:
            raise NotImplementedError("degradation type not supported")

        sigma_y = 2 * args.sigma_y #to account for scaling to [-1,1]
        
        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        pbar = tqdm.tqdm(val_loader)
        for x_orig, classes in pbar:
            x_orig = x_orig.to(self.device)
            x_orig = data_transform(self.config, x_orig)

            y = A(x_orig)

            if config.sampling.batch_size!=1:
                raise ValueError("please change the config file to set batch size as 1")

            Apy = Ap(y)

            # Save degraded and original images
            save_subfolder = f"Apy_{codec_name}" if codec_name else "Apy"
            os.makedirs(os.path.join(self.args.image_folder, save_subfolder), exist_ok=True)
            for i in range(len(Apy)):
                tvu.save_image(
                    inverse_data_transform(config, Apy[i]),
                    os.path.join(self.args.image_folder, f"{save_subfolder}/Apy_{idx_so_far + i}.png")
                )
                tvu.save_image(
                    inverse_data_transform(config, x_orig[i]),
                    os.path.join(self.args.image_folder, f"{save_subfolder}/orig_{idx_so_far + i}.png")
                )

            # init noise — latent shape for latent mode, pixel shape for pixel mode
            if is_latent:
                x = torch.randn(
                    y.shape[0], latent_shape[0], latent_shape[1], latent_shape[2],
                    device=self.device,
                )
            else:
                x = torch.randn(
                    y.shape[0], config.data.channels, img_size, img_size,
                    device=self.device,
                )

            with torch.no_grad():
                skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
                n = x.size(0)
                x0_preds = []
                xs = [x]

                times = get_schedule_jump(config.time_travel.T_sampling,
                                               config.time_travel.travel_length,
                                               config.time_travel.travel_repeat,
                                              )
                time_pairs = list(zip(times[:-1], times[1:]))


                # reverse diffusion sampling
                # Build BP schedule: more frequent at start (high noise), less at end
                # bp_schedule is a string like "1,1,1,2,2,5,5,10,10,20" (per phase)
                # or use bp_every for uniform spacing
                total_steps = config.time_travel.T_sampling
                bp_schedule_str = getattr(args, 'bp_schedule', '')
                if bp_schedule_str:
                    # Parse schedule: divide steps into equal phases,
                    # each phase has its own bp frequency
                    freqs = [int(x) for x in bp_schedule_str.split(',')]
                    n_phases = len(freqs)
                    steps_per_phase = total_steps // n_phases
                    bp_freq_per_step = []
                    for phase_idx, freq in enumerate(freqs):
                        count = steps_per_phase if phase_idx < n_phases - 1 else total_steps - phase_idx * steps_per_phase
                        bp_freq_per_step.extend([freq] * count)
                    print(f"BP schedule: {n_phases} phases, freqs={freqs}")
                else:
                    bp_every = getattr(args, 'bp_every', 1)
                    bp_freq_per_step = [bp_every] * total_steps
                # Accumulate debug frames for GIF (first N images only)
                max_debug_gifs = getattr(args, 'num_debug_gifs', 1)
                save_debug = is_latent and (idx_so_far - idx_init) < max_debug_gifs
                frames_x0_t = []
                frames_x0_t_hat = []

                step_count = 0
                for i, j in tqdm.tqdm(time_pairs):
                    i, j = i*skip, j*skip
                    if j<0: j=-1

                    if j < i: # normal sampling
                        step_count += 1
                        t = (torch.ones(n) * i).to(x.device)
                        next_t = (torch.ones(n) * j).to(x.device)
                        at = compute_alpha(self.betas, t.long())
                        at_next = compute_alpha(self.betas, next_t.long())
                        xt = xs[-1].to('cuda')

                        et = model(xt, t)

                        if et.size(1) == 6:
                            et = et[:, :3]

                        # Tweedie: estimate clean sample
                        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                        lambda_t = 1.
                        bp_stop_pct = getattr(args, 'bp_stop', 1.0)
                        bp_start_pct = getattr(args, 'bp_start', 0.0)
                        if is_latent:
                            freq = bp_freq_per_step[min(step_count - 1, len(bp_freq_per_step) - 1)]
                            bp_past_stop = step_count > total_steps * bp_stop_pct
                            bp_before_start = step_count < total_steps * bp_start_pct
                            do_bp = (step_count % freq == 0) and not bp_past_stop and not bp_before_start
                            if not do_bp:
                                x0_t_hat = x0_t
                                x0_t_hat_pixel = None
                            else:
                                x0_t_pixel = codec.decode(x0_t)
                                x0_t_hat_pixel = x0_t_pixel - lambda_t*Ap(A(x0_t_pixel) - y)
                                #x0_t_hat_pixel = x0_t_pixel
                                x0_t_hat = codec.encode(x0_t_hat_pixel)
                        else:
                            x0_t_hat = x0_t - lambda_t*Ap(A(x0_t) - y)

                        # DDIM step with configurable eta (0=deterministic, 1=full stochastic)
                        eta = getattr(args, 'eta', 0.0)
                        sigma_t = eta * ((1 - at_next) / (1 - at) * (1 - at / at_next)).clamp(min=0).sqrt()
                        pred_dir = (1 - at_next - sigma_t ** 2).clamp(min=0).sqrt()
                        if j == -1:
                            xt_next = x0_t_hat
                        else:
                            xt_next = at_next.sqrt() * x0_t_hat + pred_dir * et + sigma_t * torch.randn_like(x0_t)

                        # Collect debug frames for GIF
                        if save_debug:
                            with torch.no_grad():
                                # x0_t frame (raw Tweedie)
                                dbg_x0 = codec.decode(x0_t).clamp(-1, 1)
                                dbg_x0 = inverse_data_transform(config, dbg_x0)
                                frame_x0 = (dbg_x0[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                                frames_x0_t.append(frame_x0)

                                # x0_t_hat frame (after BP)
                                if is_latent and x0_t_hat_pixel is not None:
                                    # Reuse already-decoded pixel from BP
                                    dbg_hat = x0_t_hat_pixel.clamp(-1, 1)
                                else:
                                    dbg_hat = codec.decode(x0_t_hat).clamp(-1, 1)
                                dbg_hat = inverse_data_transform(config, dbg_hat)
                                frame_hat = (dbg_hat[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                                frames_x0_t_hat.append(frame_hat)

                        x0_preds.append(x0_t_hat.to('cpu'))
                        xs.append(xt_next.to('cpu'))
                    else: # time-travel back
                        next_t = (torch.ones(n) * j).to(x.device)
                        at_next = compute_alpha(self.betas, next_t.long())
                        x0_t = x0_preds[-1].to('cuda')

                        xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

                        xs.append(xt_next.to('cpu'))

                x = xs[-1]

            # Save debug GIFs
            if save_debug and frames_x0_t:
                gif_dir = os.path.join(self.args.image_folder, f"debug_{codec_name}" if codec_name else "debug")
                os.makedirs(gif_dir, exist_ok=True)

                def _save_gif(frames, path, duration=500):
                    pil_frames = [Image.fromarray(f).resize((256, 256), Image.LANCZOS) for f in frames]
                    pil_frames[0].save(path, save_all=True, append_images=pil_frames[1:],
                                       duration=duration, loop=0)

                gif_x0_path = os.path.join(gif_dir, f"x0_t_{idx_so_far}.gif")
                gif_hat_path = os.path.join(gif_dir, f"x0_t_hat_{idx_so_far}.gif")
                _save_gif(frames_x0_t, gif_x0_path)
                _save_gif(frames_x0_t_hat, gif_hat_path)
                print(f"Saved debug GIFs: {gif_x0_path}, {gif_hat_path}")

                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({
                            f"{codec_name}/x0_t_gif_{idx_so_far}": wandb.Video(gif_x0_path, fps=10, format="gif"),
                            f"{codec_name}/x0_t_hat_gif_{idx_so_far}": wandb.Video(gif_hat_path, fps=10, format="gif"),
                        })
                except ImportError:
                    pass

            # Final output: decode to pixel space if latent mode
            if is_latent:
                with torch.no_grad():
                    x = codec.decode(x.to(self.device)).clamp(-1, 1)
            x = [inverse_data_transform(config, x)]

            save_folder = codec_name if codec_name else ""
            if save_folder:
                os.makedirs(os.path.join(self.args.image_folder, save_folder), exist_ok=True)

            for jj in range(x[0].size(0)):
                save_path = os.path.join(self.args.image_folder, save_folder, f"{idx_so_far + jj}.png") if save_folder \
                    else os.path.join(self.args.image_folder, f"{idx_so_far + jj}_{0}.png")
                tvu.save_image(x[0][jj], save_path)
                orig = inverse_data_transform(config, x_orig[jj])
                mse = torch.mean((x[0][jj].to(self.device) - orig) ** 2)
                psnr = 10 * torch.log10(1 / mse)
                avg_psnr += psnr

            idx_so_far += y.shape[0]

            pbar.set_description("PSNR: %.2f" % (avg_psnr / (idx_so_far - idx_init)))

        avg_psnr = avg_psnr / (idx_so_far - idx_init)
        print("Total Average PSNR: %.2f" % avg_psnr)
        print("Number of samples: %d" % (idx_so_far - idx_init))
        
        

    def svd_based_ddnm_plus(self, model, cls_fn):
        args, config = self.args, self.config

        dataset, test_dataset = get_dataset(args, config)

        device_count = torch.cuda.device_count()

        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)

        print(f'Dataset has size {len(test_dataset)}')

        def seed_worker(worker_id):
            worker_seed = args.seed % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(args.seed)
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )

        # get degradation matrix
        deg = args.deg
        A_funcs = None
        if deg == 'cs_walshhadamard':
            compress_by = round(1/args.deg_scale)
            from functions.svd_operators import WalshHadamardCS
            A_funcs = WalshHadamardCS(config.data.channels, self.config.data.image_size, compress_by,
                                      torch.randperm(self.config.data.image_size ** 2, device=self.device), self.device)
        elif deg == 'cs_blockbased':
            cs_ratio = args.deg_scale
            from functions.svd_operators import CS
            A_funcs = CS(config.data.channels, self.config.data.image_size, cs_ratio, self.device)
        elif deg == 'inpainting':
            from functions.svd_operators import Inpainting
            loaded = np.load("exp/inp_masks/mask.npy")
            mask = torch.from_numpy(loaded).to(self.device).reshape(-1)
            missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
            missing_g = missing_r + 1
            missing_b = missing_g + 1
            missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
            A_funcs = Inpainting(config.data.channels, config.data.image_size, missing, self.device)
        elif deg == 'denoising':
            from functions.svd_operators import Denoising
            A_funcs = Denoising(config.data.channels, self.config.data.image_size, self.device)
        elif deg == 'colorization':
            from functions.svd_operators import Colorization
            A_funcs = Colorization(config.data.image_size, self.device)
        elif deg == 'sr_averagepooling':
            blur_by = int(args.deg_scale)
            from functions.svd_operators import SuperResolution
            A_funcs = SuperResolution(config.data.channels, config.data.image_size, blur_by, self.device)
        elif deg == 'sr_bicubic':
            factor = int(args.deg_scale)
            from functions.svd_operators import SRConv
            def bicubic_kernel(x, a=-0.5):
                if abs(x) <= 1:
                    return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
                elif 1 < abs(x) and abs(x) < 2:
                    return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
                else:
                    return 0
            k = np.zeros((factor * 4))
            for i in range(factor * 4):
                x = (1 / factor) * (i - np.floor(factor * 4 / 2) + 0.5)
                k[i] = bicubic_kernel(x)
            k = k / np.sum(k)
            kernel = torch.from_numpy(k).float().to(self.device)
            A_funcs = SRConv(kernel / kernel.sum(), \
                             config.data.channels, self.config.data.image_size, self.device, stride=factor)
        elif deg == 'deblur_uni':
            from functions.svd_operators import Deblurring
            A_funcs = Deblurring(torch.Tensor([1 / 9] * 9).to(self.device), config.data.channels,
                                 self.config.data.image_size, self.device)
        elif deg == 'deblur_gauss':
            from functions.svd_operators import Deblurring
            sigma = 10
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(self.device)
            A_funcs = Deblurring(kernel / kernel.sum(), config.data.channels, self.config.data.image_size, self.device)
        elif deg == 'deblur_aniso':
            from functions.svd_operators import Deblurring2D
            sigma = 20
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(
                self.device)
            sigma = 1
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(
                self.device)
            A_funcs = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), config.data.channels,
                                   self.config.data.image_size, self.device)
        else:
            raise ValueError("degradation type not supported")
        sigma_y = 2 * args.sigma_y #to account for scaling to [-1,1]
        
        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        pbar = tqdm.tqdm(val_loader)
        for x_orig, classes in pbar:
            x_orig = x_orig.to(self.device)
            x_orig = data_transform(self.config, x_orig)

            y = A_funcs.A(x_orig)
            
            b, hwc = y.size()
            if 'color' in deg:
                hw = hwc / 1
                h = w = int(hw ** 0.5)
                y = y.reshape((b, 1, h, w))
            elif 'inp' in deg or 'cs' in deg:
                pass
            else:
                hw = hwc / 3
                h = w = int(hw ** 0.5)
                y = y.reshape((b, 3, h, w))
                
            if self.args.add_noise: # for denoising test
                y = get_gaussian_noisy_img(y, sigma_y) 
            
            y = y.reshape((b, hwc))

            Apy = A_funcs.A_pinv(y).view(y.shape[0], config.data.channels, self.config.data.image_size,
                                                self.config.data.image_size)

            if deg[:6] == 'deblur':
                Apy = y.view(y.shape[0], config.data.channels, self.config.data.image_size,
                                    self.config.data.image_size)
            elif deg == 'colorization':
                Apy = y.view(y.shape[0], 1, self.config.data.image_size, self.config.data.image_size).repeat(1,3,1,1)
            elif deg == 'inpainting':
                Apy += A_funcs.A_pinv(A_funcs.A(torch.ones_like(Apy))).reshape(*Apy.shape) - 1

            os.makedirs(os.path.join(self.args.image_folder, "Apy"), exist_ok=True)
            for i in range(len(Apy)):
                tvu.save_image(
                    inverse_data_transform(config, Apy[i]),
                    os.path.join(self.args.image_folder, f"Apy/Apy_{idx_so_far + i}.png")
                )
                tvu.save_image(
                    inverse_data_transform(config, x_orig[i]),
                    os.path.join(self.args.image_folder, f"Apy/orig_{idx_so_far + i}.png")
                )

            #Start DDIM
            x = torch.randn(
                y.shape[0],
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )

            with torch.no_grad():
                if sigma_y==0.: # noise-free case, turn to ddnm
                    x, _ = ddnm_diffusion(x, model, self.betas, self.args.eta, A_funcs, y, cls_fn=cls_fn, classes=classes, config=config)
                else: # noisy case, turn to ddnm+
                    x, _ = ddnm_plus_diffusion(x, model, self.betas, self.args.eta, A_funcs, y, sigma_y, cls_fn=cls_fn, classes=classes, config=config)

            x = [inverse_data_transform(config, xi) for xi in x]


            for j in range(x[0].size(0)):
                tvu.save_image(
                    x[0][j], os.path.join(self.args.image_folder, f"{idx_so_far + j}_{0}.png")
                )
                orig = inverse_data_transform(config, x_orig[j])
                mse = torch.mean((x[0][j].to(self.device) - orig) ** 2)
                psnr = 10 * torch.log10(1 / mse)
                avg_psnr += psnr

            idx_so_far += y.shape[0]

            pbar.set_description("PSNR: %.2f" % (avg_psnr / (idx_so_far - idx_init)))

        avg_psnr = avg_psnr / (idx_so_far - idx_init)
        print("Total Average PSNR: %.2f" % avg_psnr)
        print("Number of samples: %d" % (idx_so_far - idx_init))

    def latent_ddnm(self):
        """Load SD 1.5 UNet + codecs, then reuse simplified_ddnm_plus for sampling."""
        args, config = self.args, self.config

        print("Importing diffusers/transformers...")
        from diffusers import UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer
        from functions.codec import load_codec

        sd_id = "runwayml/stable-diffusion-v1-5"

        # Load SD 1.5 UNet
        print("Loading SD 1.5 UNet...")
        unet = UNet2DConditionModel.from_pretrained(sd_id, subfolder="unet")
        unet.eval().to(self.device)
        for p in unet.parameters():
            p.requires_grad = False

        # Text embeddings for CFG
        print("Loading text encoder...")
        tokenizer = CLIPTokenizer.from_pretrained(sd_id, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(sd_id, subfolder="text_encoder")
        text_encoder.eval().to(self.device)

        # Unconditional embedding (empty string "")
        uncond_tokens = tokenizer("", padding="max_length", max_length=77,
                                  return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            uncond_emb = text_encoder(uncond_tokens).last_hidden_state  # [1, 77, 768]

        # Conditional embedding (text prompt if provided)
        prompt = getattr(args, 'prompt', '') or ''
        if prompt:
            print(f"Using prompt: '{prompt}'")
        cond_tokens = tokenizer(prompt if prompt else "", padding="max_length", max_length=77,
                                return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            cond_emb = text_encoder(cond_tokens).last_hidden_state
        del text_encoder, tokenizer

        # Load codec(s)
        vae_codec, spnn_codec = load_codec(config, self.device)
        guidance_scale = config.model.guidance_scale if hasattr(config.model, 'guidance_scale') else 1.0

        # Wrap UNet to match pixel-space model(xt, t) interface
        def make_latent_model(guidance_scale, uncond_emb, cond_emb):
            def model_fn(zt, t):
                uc = uncond_emb.expand(zt.size(0), -1, -1)
                if guidance_scale > 1.0:
                    cc = cond_emb.expand(zt.size(0), -1, -1)
                    z_in = torch.cat([zt, zt], dim=0)
                    t_in = t.repeat(2) if t.dim() > 0 else t.unsqueeze(0).repeat(2 * zt.size(0))
                    e_in = torch.cat([uc, cc], dim=0)
                    noise_all = unet(z_in, t_in, encoder_hidden_states=e_in).sample
                    noise_uncond, noise_cond = noise_all.chunk(2)
                    return noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                else:
                    return unet(zt, t, encoder_hidden_states=uc).sample
            return model_fn

        model_fn = make_latent_model(guidance_scale, uncond_emb, cond_emb)

        # Run for each codec
        codecs_to_run = {"VAE": vae_codec}
        if spnn_codec is not None:
            codecs_to_run["SPNN"] = spnn_codec

        for codec_name, codec in codecs_to_run.items():
            print(f"\n--- Running with {codec_name} codec ---")
            # Re-seed global RNG so each codec run sees the same initial noise
            # and the same DDIM stochastic draws (fair decoder comparison).
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            # Store codec on self so simplified_ddnm_plus can use it
            self._codec = codec
            self._latent_shape = (4, config.data.image_size // 8, config.data.image_size // 8)
            self.simplified_ddnm_plus(model_fn, cls_fn=None, codec_name=codec_name)

        self._codec = None
        self._latent_shape = None

    def latent_compvis_ldm(self):
        """Load CompVis unconditional UNet directly from .ckpt + codecs for DDNM."""
        args, config = self.args, self.config

        from functions.codec import load_codec
        from functions.compvis_unet import load_compvis_unet

        # Load CompVis checkpoint once, pass state dict to UNet and codec loaders
        ckpt_path = config.model.compvis_ckpt_path
        print(f"Loading CompVis checkpoint from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        compvis_sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

        unet = load_compvis_unet(ckpt_path, self.device, state_dict=compvis_sd)
        vae_codec, spnn_codec = load_codec(config, self.device, compvis_state_dict=compvis_sd)
        del compvis_sd

        # Simple wrapper — unconditional, no CFG, no text embeddings
        def model_fn(zt, t):
            # CompVis UNet uses sinusoidal timestep embedding (accepts float)
            if not torch.is_tensor(t):
                t = torch.tensor([t], device=zt.device)
            if t.dim() == 0:
                t = t.unsqueeze(0).expand(zt.size(0))
            return unet(zt, t)

        # Run for each codec
        codecs_to_run = {"VAE": vae_codec}
        if spnn_codec is not None:
            codecs_to_run["SPNN"] = spnn_codec

        for codec_name, codec in codecs_to_run.items():
            print(f"\n--- Running with {codec_name} codec ---")
            # Re-seed global RNG so each codec run sees the same initial noise
            # and the same DDIM stochastic draws (fair decoder comparison).
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            self._codec = codec
            self._latent_shape = (4, config.data.image_size // 8,
                                  config.data.image_size // 8)
            self.simplified_ddnm_plus(model_fn, cls_fn=None,
                                      codec_name=codec_name)

        self._codec = None
        self._latent_shape = None


# Code form RePaint
def get_schedule_jump(T_sampling, travel_length, travel_repeat):
    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, T_sampling)
    return ts

def _check_times(times, t_0, T_sampling):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)
        
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a
