"""
DiffQRCoder (Liao et al., WACV 2025, arXiv 2409.06355; code in DiffQRCoder-main/) as a plain sampling loop for
diffusers 0.24 / torch 2.0, with the image codec as a parameter, so the SD 1.5 VAE and the SPNN
(spnn_model_opt.SPNNAutoencoder512Opt) can be swapped under the same SD 1.5 UNet + QR Code Monster v2 ControlNet.
The loss code is the repo's, imported unmodified (ScanningRobustLoss, PerceptualLoss, ScanningRobustPerceptualGuidance).

Stage 2 applies Scanning Robust Perceptual Guidance (SRPG) in one of two modes:

  mode="grad"      the official one (pipeline_diffqrcoder.py, _run_stage2): F_SRP(D(z0|t)) is back-propagated through
                   the decoder AND the UNet/ControlNet to z_t;  eps_hat = eps + sqrt(1-a_t) * grad_{z_t} F.
  mode="sandwich"  the latent version: every step is sandwiched between a decode and an encode, as in latent DDNM:
                       z0|t --D--> x --(x - c_t * grad_x F(x))--> x' --E--> z0'
                   followed by the DDIM step from (z0', eps'), eps' = (z_t - sqrt(a_t) z0') / sqrt(1-a_t): the same kind of
                   (z0, eps) pair the official update produces. Nothing is back-propagated through the UNet or the codec.
                   Guided pixels are clamped to [-1, 1] (guided_step); unguided pixels are passed through untouched.
                   With an idempotent codec (E(D(z)) == z, the SPNN) a zero pixel step leaves the chain untouched; the
                   SD-VAE's E(D(z)) != z injects its round-trip error at every step.

Sandwich step size. With a unit-variance Gaussian prior on the scaled latent, Tweedie gives d z0|t / d z_t = sqrt(a_t) I,
so the official update moves z0|t by -(1-a_t) J_D^T g (g = grad_x F, J_D the decoder Jacobian), i.e. the image by about
-(1-a_t) J_D J_D^T g. The sandwich replaces J_D J_D^T by kappa * I, where kappa = |J_D^T g|^2 / |g|^2 is measured on the
SD-VAE decoder (estimate_kappa), and lets E do the projection:  c_t = pixel_step * kappa * (1 - a_t).

SR-MPGD (Sec. 3.3, post-processing) has the same two modes: "grad" = SGD on z through D (official); "sandwich" = projected
gradient descent in pixel space, x <- D(E(x - lr * kappa * grad_x L)). For the SPNN, D(E(.)) is an exact idempotent projection.
"""
import sys
import types
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent / "DiffQRCoder-main" / "diffqrcoder"
if "diffqrcoder" not in sys.modules:      # the package __init__ imports the diffusers>=0.27 pipeline; the losses do not need it
    _pkg = types.ModuleType("diffqrcoder")
    _pkg.__path__ = [str(_REPO)]
    sys.modules["diffqrcoder"] = _pkg
from diffqrcoder.image_processor import convert_to_gray, crop_padding, image_binarize  # noqa: E402
from diffqrcoder.losses import PerceptualLoss  # noqa: E402
from diffqrcoder.losses.scanning_robust_loss import CenterPixelExtractor, RegionMeanFilter  # noqa: E402
from diffqrcoder.srpg import GRADIENT_SCALE, ScanningRobustPerceptualGuidance  # noqa: E402


# ----------------------------------------------------------------------------------------------------------- codecs
class Codec:
    """encode: [-1, 1] image -> SD-convention latent (z * 0.18215); decode: the way back. Differentiable (the caller
    decides about no_grad), so the same object serves the grad and the sandwich modes."""

    def __init__(self, name, encode, decode):
        self.name, self.encode, self.decode = name, encode, decode


def sd_vae_codec(vae, scaling=0.18215):
    enc = lambda x: vae.encode(x.to(vae.dtype)).latent_dist.mean.float() * scaling
    dec = lambda z: torch.nan_to_num(vae.decode((z / scaling).to(vae.dtype)).sample.float())
    return Codec("SD-VAE", enc, dec)


def spnn_codec(spnn):
    return Codec("SPNN", lambda x: spnn.encode(x.float()).float(), lambda z: torch.nan_to_num(spnn.decode(z.float()).float()))


def load_spnn(ckpt, device, use_ema=True):
    """SPNNAutoencoder512Opt with the class-default config (= the trained one); EMA weights as validation used them."""
    from spnn_model_opt import SPNNAutoencoder512Opt
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = SPNNAutoencoder512Opt()
    net.load_state_dict(ck["model"])
    if use_ema:                                   # the EMA shadow holds only the floating-point tensors
        sd = net.state_dict()
        sd.update({k: v.to(sd[k].dtype) for k, v in ck["ema"].items()})
        net.load_state_dict(sd)
    return net.to(device).eval().requires_grad_(False), ck


# --------------------------------------------------------------------------------------------------------- QR codes
def make_qr(message="Thanks reviewer!", version=3, ec="M", mask=4, module_size=14, size=512):
    """The repo's thanks_reviewer.png (version 3, level M, mask 4) re-rendered so it fills `size` exactly.
    Returns ([1,3,size,size] in [0,1], 1 = white; padding in px; the module matrix, 1 = dark)."""
    import qrcode
    q = qrcode.QRCode(version=version, error_correction=getattr(qrcode.constants, f"ERROR_CORRECT_{ec}"),
                      border=0, mask_pattern=mask)
    q.add_data(message)
    q.make(fit=False)
    m = np.array(q.get_matrix(), dtype=np.float32)
    side = m.shape[0] * module_size
    pad = (size - side) // 2
    assert pad >= 0 and 2 * pad + side == size, f"size - {m.shape[0]} * module_size must be even and >= 0"
    img = np.ones((size, size), np.float32)
    img[pad:pad + side, pad:pad + side] = 1 - np.kron(m, np.ones((module_size, module_size), np.float32))
    return torch.from_numpy(img)[None, None].expand(1, 3, size, size).contiguous(), pad, m


def gray_margin(qr01, pad, value=128 / 255):
    """The ControlNet condition the QR Monster v2 card recommends: the code with a gray (#808080) margin, which lets the
    image blend outside the code. The SRPG target stays the plain binary code."""
    c = qr01.clone()
    c[..., :pad, :] = value; c[..., -pad:, :] = value; c[..., :, :pad] = value; c[..., :, -pad:] = value
    return c


def to_u8(x):
    """[-1, 1] tensor [B,3,H,W] -> list of HxWx3 uint8 arrays."""
    x = ((x.detach().float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    return list(x)


WECHAT_MODELS = Path("/rg/shocher_prj/ron.libman/laion1m/runs/spnn512/diffqr/wechat_models")
_wechat = None


def _wechat_detector():
    """OpenCV-contrib's WeChat QR detector (CNN detection + super-resolution), the decoder behind qr-verify, which the
    paper scored with. Needs an opencv-contrib cv2 on the path and the four model files in WECHAT_MODELS."""
    global _wechat
    if _wechat is None:
        _wechat = False
        if hasattr(cv2, "wechat_qrcode_WeChatQRCode") and WECHAT_MODELS.is_dir():
            m = [str(WECHAT_MODELS / f) for f in ("detect.prototxt", "detect.caffemodel", "sr.prototxt", "sr.caffemodel")]
            _wechat = cv2.wechat_qrcode_WeChatQRCode(*m)
    return _wechat or None


def scan(img_u8):
    """Decoded text per scanner (None = not decoded). wechat = qr-verify's decoder (the paper's scanner); zxing-cpp is
    the ZXing library the paper cites; OpenCV's two classic detectors are stricter extra votes. A scanner whose library
    is missing reports None."""
    out = {"wechat": None, "zxing": None}
    det = _wechat_detector()
    if det is not None:
        txt = det.detectAndDecode(np.ascontiguousarray(img_u8[..., ::-1]))[0]
        out["wechat"] = txt[0] if len(txt) else None
    try:
        import zxingcpp
        r = zxingcpp.read_barcodes(img_u8, formats=zxingcpp.BarcodeFormat.QRCode)
        out["zxing"] = r[0].text if r else None
    except ImportError:
        pass
    bgr = np.ascontiguousarray(img_u8[..., ::-1])
    for name, det in (("opencv", cv2.QRCodeDetector()), ("opencv_aruco", cv2.QRCodeDetectorAruco())):
        try:
            txt = det.detectAndDecode(bgr)[0]
            out[name] = txt or None
        except cv2.error:
            out[name] = None
    return out


# ------------------------------------------------------------------------------------------------------------- SRPG
class PerceptualLossWithGrad(PerceptualLoss):
    """The repo's PerceptualLoss (same VGG16 slices and normalisation) with torch.stack in place of torch.tensor([...]),
    which copies the per-layer values out of the graph: in the repo the perceptual term has no gradient."""

    def forward(self, x, y):
        return torch.stack([F.mse_loss(fx, fy) for fx, fy in zip(self.extractor(x), self.extractor(y))]).mean()


class SRPG:
    """The repo's ScanningRobustPerceptualGuidance bound to one target code.
    perceptual="official": the repo's PerceptualLoss (no gradient, so the guidance is lam_sr * SRL whatever lam_perc is)
    perceptual="grad":     the same VGG loss with its gradient, so lam_perc acts as in the paper."""

    def __init__(self, qr01, module_size, padding, lam_sr=500, lam_perc=2, perceptual="official", device="cuda"):
        self.core = ScanningRobustPerceptualGuidance(module_size, lam_sr, lam_perc).to(device)
        if perceptual == "grad":
            self.core.perceptual_loss_fn = PerceptualLossWithGrad().to(device)
        elif perceptual != "official":
            raise ValueError(perceptual)
        self.pad = padding
        self.qr_bin = crop_padding(image_binarize(qr01.to(device).float()), padding)        # [1,1,n*ms,n*ms], 1 = white
        self.center_mean = RegionMeanFilter(module_size).to(device)
        self.target = CenterPixelExtractor(module_size).to(device)(self.qr_bin)            # [1,1,n,n]

    def prep(self, x):
        """VaeImageProcessor.denormalize ([-1,1] -> [0,1], clamped), then the repo's crop_padding."""
        return crop_padding((x / 2 + 0.5).clamp(0, 1), self.pad)

    def score(self, x, ref01):
        """lam_sr * L_SR + lam_perc * L_perc on a [-1,1] decode (= the repo's compute_loss / GRADIENT_SCALE), summed over
        the batch: the repo's losses average over it, so this gives every image its own batch-of-one gradient."""
        return x.shape[0] * self.core.compute_loss(self.prep(x), self.qr_bin, crop_padding(ref01, self.pad)) / GRADIENT_SCALE

    def grad_x(self, x, ref01):
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            return torch.autograd.grad(self.score(x, ref01), x)[0]

    @torch.no_grad()
    def error_rate(self, x):
        """Paper Eq. 19, per image: fraction of modules whose binarised central-submodule mean (Eq. 4) differs from the
        target module's centre pixel."""
        g = convert_to_gray(self.prep(x.float()))
        return ((self.center_mean(g) >= 0.5).float() != self.target).float().flatten(1).mean(1)


def guided_step(x, g, step):
    """x - step * g, clamped to the image range [-1, 1] only where the guidance moved a pixel (g != 0): the modules the
    SRL still flags would otherwise be pushed far outside what either encoder has seen, and untouched pixels stay
    bit-identical, so E(x') == E(D(z0)) == z0 when nothing is guided and the codec is idempotent."""
    return torch.where(g != 0, (x - step * g).clamp(-1, 1), x)


def estimate_kappa(codec, z0, srpg, ref01):
    """Rayleigh quotient |J_D^T g|^2 / |g|^2 of the decoder Jacobian along the SRPG pixel gradient g at D(z0), per image:
    the gain the official latent-space step applies along g, which the sandwich step has to supply explicitly."""
    with torch.enable_grad():
        z = z0.detach().requires_grad_(True)
        x = codec.decode(z)
        g = srpg.grad_x(x.detach(), ref01)
        jtg = torch.autograd.grad(x, z, grad_outputs=g)[0]
    return (jtg.flatten(1).pow(2).sum(1) / g.flatten(1).pow(2).sum(1).clamp_min(1e-30)).cpu()


# ---------------------------------------------------------------------------------------------------------- sampler
class DiffQRCoder:
    """Two-stage DiffQRCoder with DDIM (eta = 0), classifier-free guidance, and ControlNet on both CFG halves
    (guess_mode=False), as the repo's pipeline."""

    def __init__(self, unet, controlnet, scheduler, tokenizer, text_encoder, num_inference_steps=40,
                 guidance_scale=7.5, controlnet_conditioning_scale=1.35):
        self.unet, self.cn, self.tok, self.te = unet, controlnet, tokenizer, text_encoder
        self.dev = unet.device
        self.gs, self.ccs = guidance_scale, controlnet_conditioning_scale
        scheduler.set_timesteps(num_inference_steps)
        self.timesteps = scheduler.timesteps.tolist()
        self.gap = scheduler.config.num_train_timesteps // num_inference_steps
        self.ac = scheduler.alphas_cumprod.float()
        self.final_ac = float(scheduler.final_alpha_cumprod)

    def alpha(self, t):
        return float(self.ac[t]) if t >= 0 else self.final_ac

    @torch.no_grad()
    def embed(self, prompts, negative_prompt=""):
        """[negative x B; prompt x B] text embeddings, the pipeline's CFG batch order."""
        def enc(texts):
            ids = self.tok(texts, padding="max_length", max_length=self.tok.model_max_length, truncation=True,
                           return_tensors="pt").input_ids.to(self.dev)
            return self.te(ids)[0].to(self.unet.dtype)
        return torch.cat([enc([negative_prompt] * len(prompts)), enc(list(prompts))])

    def eps(self, z, t, emb, qr01):
        """CFG noise prediction with the ControlNet residuals. Differentiable in z."""
        b = z.shape[0]
        zin = torch.cat([z, z]).to(self.unet.dtype)
        tt = torch.full((2 * b,), t, device=self.dev, dtype=torch.long)
        cond = qr01.to(self.dev, self.cn.dtype).expand(2 * b, -1, -1, -1)
        down, mid = self.cn(zin, tt, encoder_hidden_states=emb, controlnet_cond=cond,
                            conditioning_scale=self.ccs, return_dict=False)
        e = self.unet(zin, tt, encoder_hidden_states=emb, down_block_additional_residuals=down,
                      mid_block_additional_residual=mid, return_dict=False)[0].float()
        eu, ec = e.chunk(2)
        return eu + self.gs * (ec - eu)

    def noised(self, z0, start, noise):
        """z0 taken to timesteps[start] (img2img): sqrt(a_t) z0 + sqrt(1 - a_t) noise."""
        a = self.alpha(self.timesteps[start])
        return a ** 0.5 * z0 + (1 - a) ** 0.5 * noise

    @torch.no_grad()
    def stage1(self, emb, qr01, z, start=0):
        """ControlNet-only generation (Algorithm 1, lines 2-6). Returns the final latent; the codec only decodes it.
        start > 0 skips the first `start` steps (img2img: z = noised(E(image), start, noise))."""
        for t in self.timesteps[start:]:
            a, ap = self.alpha(t), self.alpha(t - self.gap)
            e = self.eps(z, t, emb, qr01)
            z0 = (z - (1 - a) ** 0.5 * e) / a ** 0.5
            z = ap ** 0.5 * z0 + (1 - ap) ** 0.5 * e
        return z

    def stage2(self, emb, qr01, z, codec, srpg, ref01, mode="sandwich", kappa=1.0, pixel_step=1.0, diag=False, start=0):
        """SRPG-guided generation (Algorithm 1, lines 10-20, without the tau switch, as the repo); start as in stage1.
        Returns (latent, trace):
        trace["err"] is the module error rate of D(z0|t) per step (the paper's Fig. 9a), [B, steps]. With diag=True
        (sandwich only; one extra decode + encode per step) also: err_step = error of the guided image x',
        err_proj = error of D(E(x')) (what survives the projection), drift = |E(D(z0|t)) - z0|t| / |z0|t|."""
        trace = defaultdict(list)
        for t in self.timesteps:
            a, ap = self.alpha(t), self.alpha(t - self.gap)
            sa, sb = a ** 0.5, (1 - a) ** 0.5
            if mode == "grad":
                with torch.enable_grad():
                    zg = z.detach().requires_grad_(True)
                    e = self.eps(zg, t, emb, qr01)
                    xg = codec.decode((zg - sb * e) / sa)
                    score = torch.autograd.grad(srpg.score(xg, ref01), zg)[0]
                x = xg.detach()
                del xg
                e = e.detach() + sb * score
                z0 = (z - sb * e) / sa
            elif mode == "sandwich":
                with torch.no_grad():
                    e = self.eps(z, t, emb, qr01)
                    z0 = (z - sb * e) / sa
                    x = codec.decode(z0)
                g = srpg.grad_x(x, ref01)
                with torch.no_grad():
                    x_step = guided_step(x, g, pixel_step * kappa * (1 - a))
                    z0_new = codec.encode(x_step)
                    if diag:
                        trace["err_step"].append(srpg.error_rate(x_step).cpu())
                        trace["err_proj"].append(srpg.error_rate(codec.decode(z0_new)).cpu())
                        trace["drift"].append(((codec.encode(x) - z0).flatten(1).norm(dim=1) / z0.flatten(1).norm(dim=1)).cpu())
                    z0 = z0_new
                    e = (z - sa * z0) / sb
            else:
                raise ValueError(mode)
            trace["err"].append(srpg.error_rate(x).cpu())
            z = (ap ** 0.5 * z0 + (1 - ap) ** 0.5 * e).detach()
        return z, {k: torch.stack(v, 1) for k, v in trace.items()}

    def srmpgd(self, z, codec, srpg, ref01, mode="sandwich", iters=20, lr=0.1, kappa=1.0):
        """SR-MPGD post-processing (Sec. 3.3). grad: the repo's torch.optim.SGD(lr) on compute_loss(D(z)) (which carries
        the x GRADIENT_SCALE). sandwich: x <- D(E(x - lr * GRADIENT_SCALE * kappa * grad_x)), PGD with the codec as the
        projection. Returns (latent, error rate of the decode before each iteration, [B, iters])."""
        z, errs = z.detach(), []
        for _ in range(iters):
            if mode == "grad":
                with torch.enable_grad():
                    zg = z.requires_grad_(True)
                    xg = codec.decode(zg)
                    gz = torch.autograd.grad(GRADIENT_SCALE * srpg.score(xg, ref01), zg)[0]
                x = xg.detach()
                del xg
                z = (z.detach() - lr * gz).detach()
            elif mode == "sandwich":
                with torch.no_grad():
                    x = codec.decode(z)
                g = srpg.grad_x(x, ref01)
                with torch.no_grad():
                    z = codec.encode(guided_step(x, g, lr * GRADIENT_SCALE * kappa))
            else:
                raise ValueError(mode)
            errs.append(srpg.error_rate(x).cpu())
        return z, (torch.stack(errs, 1) if errs else None)


# ------------------------------------------------------------------------------------------------------- CLIP score
class ClipScore:
    """Cosine similarity of CLIP image and text embeddings (the paper's CLIP-score column is on this 0-1 scale).
    ViT-B/32 because it is the CLIP in the local cache; the paper does not say which CLIP it used."""

    def __init__(self, device, name="openai/clip-vit-base-patch32"):
        from transformers import CLIPModel, CLIPProcessor
        self.m = CLIPModel.from_pretrained(name).to(device).eval()
        self.p = CLIPProcessor.from_pretrained(name)
        self.dev = device

    @torch.no_grad()
    def __call__(self, imgs_u8, texts):
        inp = self.p(text=list(texts), images=list(imgs_u8), return_tensors="pt", padding=True, truncation=True).to(self.dev)
        i = F.normalize(self.m.get_image_features(pixel_values=inp["pixel_values"]), dim=-1)
        t = F.normalize(self.m.get_text_features(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"]), dim=-1)
        return (i * t).sum(-1).cpu()
