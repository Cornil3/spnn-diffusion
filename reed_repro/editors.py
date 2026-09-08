"""
The five editing models, built on ImagenHub's own `infermodels` wherever they exist.

Using ImagenHub's wrappers rather than re-deriving each pipeline's call convention is
what caught the mask bug: their `rgba_to_01_mask` reads the **alpha channel**, while a
hand-rolled `.convert("L")` reads luminance and produces a mask that matches the real
one only 29% of the time. Their wrappers also pin the sampler settings each model was
published with (IP2P: EulerAncestral, 100 steps, gs 7.5, image_gs 1.5).

  ip2p        -> imagen_hub.infermodels.InstructPix2Pix
  magicbrush  -> imagen_hub.infermodels.MagicBrush
  diffedit    -> imagen_hub.infermodels.DiffEdit
  sd_inpaint  -> imagen_hub.infermodels.SDInpaint
  pbe         -> ours; ImagenHub has no Paint-by-Example wrapper (their subject-driven
                 benchmark uses PhotoSwap / DreamEdit / BLIP-Diffusion on a different
                 dataset), so REED must have improvised this row too.

Iteration protocol (1-based; odd = forward, even = reverse), per REED §5.2-5.4. 5/15/25
are all odd, so they share edit direction with x^1 and are comparable to it.

  IP2P / MagicBrush  instruction            <-> reverse_instruction
  DiffEdit           C_s -> C_t             <-> C_t -> C_s
  PbE                ref = x_r^1 (target)   <-> x_r^2 (original)
  SD Inpainting      C_t^local into m, identical every iteration
"""

import os

import torch

from .codecs import build_spnn_vae

# Overridable so a warm cache can be reused (athena holds the hub-removed runwayml SD1.5).
SD15 = os.environ.get("REED_SD15_ID", "stable-diffusion-v1-5/stable-diffusion-v1-5")
SD15_INPAINT = os.environ.get("REED_SD15_INPAINT_ID",
                              "stable-diffusion-v1-5/stable-diffusion-inpainting")
IP2P = os.environ.get("REED_IP2P_ID", "timbrooks/instruct-pix2pix")
MAGICBRUSH = os.environ.get("REED_MAGICBRUSH_ID", "vinesmsuic/magicbrush-jul7")
PBE = os.environ.get("REED_PBE_ID", "Fantasy-Studio/Paint-by-Example")

# ImagenHub's DiffEdit defaults to stabilityai/stable-diffusion-2-1. We override to
# SD 1.5 because SPNN-512 was distilled against SD 1.5's VAE — dropping it into an
# SD2.1 pipeline would put it in the wrong latent space and invalidate that arm.
# Set REED_DIFFEDIT_ID=stabilityai/stable-diffusion-2-1 to follow ImagenHub exactly
# (only meaningful for the baseline arm).
DIFFEDIT_WEIGHT = os.environ.get("REED_DIFFEDIT_ID", SD15)

TASK_TYPE = {
    "ip2p": "text-guided",
    "magicbrush": "text-guided",
    "diffedit": "text-guided",
    "pbe": "subject-guided",
    "sd_inpaint": "mask-guided",
}


def imagenhub_class(name):
    """Return ImagenHub's infermodel class, or None if the package can't be imported.

    ImagenHub 0.4.0 pulls `openai`, `omegaconf`, `google-genai`, and imports
    `CLIPFeatureExtractor`, which transformers removed in v5. Where it imports we use
    it directly; where it doesn't we fall back to `_fallback_*` below, which reproduce
    their wrappers line-for-line (same weights, scheduler, steps and guidance) so the
    two paths are behaviourally identical. Which path was taken is recorded in the run
    config as `imagenhub_api`.
    """
    if os.environ.get("REED_USE_IMAGENHUB", "1") == "0":
        # Escape hatch: importing imagen_hub.infermodels pulls in dozens of model
        # modules and was observed to hang on athena when the network was reachable
        # but slow. Forcing the fallback keeps a 9-hour job from stalling at startup.
        print("  note: REED_USE_IMAGENHUB=0 — using the vendored wrappers")
        return None
    try:
        import importlib
        return getattr(importlib.import_module("imagen_hub.infermodels"), name)
    except Exception as e:  # ImportError, ModuleNotFoundError, or their import chain
        print(f"  note: imagen_hub.{name} unavailable ({type(e).__name__}); "
              f"using the vendored equivalent of their wrapper")
        return None


class _Wrapped:
    """Minimal stand-in exposing the `.pipe` / `.infer_one_image` surface we use."""
    def __init__(self, pipe):
        self.pipe = pipe


def _fallback_ip2p(weight, device, dtype):
    """Verbatim from imagen_hub/infermodels/instructpix2pix.py (InstructPix2Pix)."""
    from diffusers import (EulerAncestralDiscreteScheduler,
                           StableDiffusionInstructPix2PixPipeline)
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        weight, torch_dtype=dtype, safety_checker=None)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    return _Wrapped(pipe)


def _fallback_diffedit(weight, device, dtype):
    """Verbatim from imagen_hub/infermodels/diffedit.py (DiffEdit)."""
    from diffusers import (DDIMInverseScheduler, DDIMScheduler,
                           StableDiffusionDiffEditPipeline)
    pipe = StableDiffusionDiffEditPipeline.from_pretrained(
        weight, torch_dtype=dtype, safety_checker=None)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    return _Wrapped(pipe)


def _fallback_sd_inpaint(weight, device, dtype):
    """Verbatim from imagen_hub/infermodels/sd.py (SDInpaint)."""
    from diffusers import StableDiffusionInpaintPipeline
    return _Wrapped(StableDiffusionInpaintPipeline.from_pretrained(
        weight, torch_dtype=dtype, safety_checker=None))



class Editor:
    """Adapter: an ImagenHub infermodel + our iteration protocol + the codec swap."""

    name = "base"
    weight = None

    def __init__(self, codec="vae", device="cuda", dtype=torch.float32,
                 spnn_checkpoint=None, spnn_weights="ema",
                 latent_scale="prescaled", overrides=None):
        self.codec, self.device, self.dtype = codec, device, dtype
        self.params = dict(overrides or {})
        self.used_imagenhub = None      # set by _build
        self.model = self._build()
        self.pipe = self.model.pipe
        # ImagenHub loads every pipeline in fp16. SPNN is a precision-sensitive
        # projection (its encode/decode fixed point sits at ~1e-8 in fp32), so we cast
        # to `dtype` — fp32 by default — and apply it to BOTH arms so precision can
        # never be what separates them.
        self.pipe.to(device=device, dtype=dtype)
        self.model.device = device
        self.vae_source = self.weight

        if codec == "spnn":
            if not spnn_checkpoint:
                raise ValueError("codec='spnn' requires --spnn_checkpoint")
            self.pipe.vae = build_spnn_vae(
                self.pipe, spnn_checkpoint, weights=spnn_weights,
                latent_scale=latent_scale, device=device, dtype=dtype)
            self.vae_source = f"SPNN-512 ({os.path.basename(spnn_checkpoint)})"
        elif codec != "vae":
            raise ValueError(f"unknown codec: {codec}")

    def _build(self):
        raise NotImplementedError

    def edit(self, sample, image, iteration, seed):
        raise NotImplementedError

    @staticmethod
    def _forward(iteration):
        return iteration % 2 == 1


class IP2PEditor(Editor):
    name = "ip2p"
    weight = IP2P

    ih_name = "InstructPix2Pix"

    def _build(self):
        cls = imagenhub_class(self.ih_name)
        self.used_imagenhub = cls is not None
        if cls is not None:
            return cls(device=self.device, weight=self.weight)
        return _fallback_ip2p(self.weight, self.device, self.dtype)

    def edit(self, sample, image, iteration, seed):
        instr = sample.instruction if self._forward(iteration) else sample.reverse_instruction
        if self.used_imagenhub:
            return self.model.infer_one_image(src_image=image, instruct_prompt=instr,
                                              seed=seed)
        torch.manual_seed(seed)
        # ImagenHub's configs, from timothybrooks/instruct-pix2pix edit_cli.py
        return self.pipe(instr, image=image.convert("RGB"), num_inference_steps=100,
                         image_guidance_scale=1.5, guidance_scale=7.5).images[0]


class MagicBrushEditor(IP2PEditor):
    name = "magicbrush"
    weight = MAGICBRUSH

    ih_name = "MagicBrush"


class DiffEditEditor(Editor):
    name = "diffedit"
    weight = DIFFEDIT_WEIGHT

    def _build(self):
        cls = imagenhub_class("DiffEdit")
        self.used_imagenhub = cls is not None
        if cls is not None:
            return cls(device=self.device, weight=self.weight)
        return _fallback_diffedit(self.weight, self.device, self.dtype)

    def edit(self, sample, image, iteration, seed):
        if self._forward(iteration):
            src, tgt = sample.source_caption, sample.target_caption
        else:
            src, tgt = sample.target_caption, sample.source_caption
        if self.used_imagenhub:
            return self.model.infer_one_image(src_image=image, src_prompt=src,
                                              target_prompt=tgt, seed=seed)
        # Match imagen_hub/infermodels/diffedit.py exactly: one generator from
        # torch.manual_seed(seed), threaded through all three calls.
        generator = torch.manual_seed(seed)
        img = image.convert("RGB")
        mask = self.pipe.generate_mask(image=img, source_prompt=src, target_prompt=tgt,
                                       generator=generator)
        inv = self.pipe.invert(prompt=src, image=img, generator=generator).latents
        return self.pipe(prompt=tgt, mask_image=mask, image_latents=inv,
                         generator=generator, negative_prompt=src).images[0]


class SDInpaintEditor(Editor):
    name = "sd_inpaint"
    weight = SD15_INPAINT

    def _build(self):
        cls = imagenhub_class("SDInpaint")
        self.used_imagenhub = cls is not None
        if cls is not None:
            return cls(device=self.device, weight=self.weight)
        return _fallback_sd_inpaint(self.weight, self.device, self.dtype)

    def edit(self, sample, image, iteration, seed):
        # §5.3: the same object is inpainted into the same mask at every iteration.
        # `sample.mask` is already ImagenHub's alpha-derived 0/255 L-mode mask, which
        # is what SDInpaint.infer_one_image expects once it is no longer RGBA.
        if self.used_imagenhub:
            return self.model.infer_one_image(
                src_image=image, local_mask_prompt=sample.target_local_caption,
                mask_image=sample.mask, seed=seed)
        torch.manual_seed(seed)
        return self.pipe(prompt=sample.target_local_caption, image=image.convert("RGB"),
                         mask_image=sample.mask).images[0]


class PbEEditor(Editor):
    """Paint-by-Example — no ImagenHub wrapper exists, so this one stays ours."""

    name = "pbe"
    weight = PBE
    defaults = dict(num_inference_steps=50, guidance_scale=5.0)

    def _build(self):
        from diffusers import DiffusionPipeline

        class _Wrap:
            pass

        self.used_imagenhub = False   # ImagenHub has no PbE wrapper
        w = _Wrap()
        w.pipe = DiffusionPipeline.from_pretrained(
            self.weight, torch_dtype=self.dtype, safety_checker=None,
            requires_safety_checker=False)
        w.pipe.set_progress_bar_config(disable=True)
        w.pipe.safety_checker = None
        return w

    def edit(self, sample, image, iteration, seed):
        # §5.4: alternate the reference between the target object and the original
        # object recovered from x_s through the mask's bounding box.
        ref = sample.ref_target if self._forward(iteration) else sample.ref_source
        p = dict(self.defaults)
        p.update(self.params)
        return self.pipe(
            image=image, mask_image=sample.mask, example_image=ref,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            num_inference_steps=p["num_inference_steps"],
            guidance_scale=p["guidance_scale"],
        ).images[0]


EDITORS = {
    "ip2p": IP2PEditor,
    "magicbrush": MagicBrushEditor,
    "diffedit": DiffEditEditor,
    "pbe": PbEEditor,
    "sd_inpaint": SDInpaintEditor,
}
