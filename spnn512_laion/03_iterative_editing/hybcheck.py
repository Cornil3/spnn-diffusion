"""Is the hybrid arm really 'SD-VAE encoder, SPNN decoder' and nothing else?"""
import sys, torch, numpy as np
from pathlib import Path
from PIL import Image
sys.path.insert(0, "/home/ron.libman/churches-reboot")
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import DecoderOutput
from spnn_model_opt import SPNNAutoencoder512Opt

DEV, SC = "cuda", 0.18215
VID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
ck = torch.load("/rg/shocher_prj/ron.libman/laion1m/runs/spnn512/best.pt", map_location="cpu", weights_only=False)
spnn = SPNNAutoencoder512Opt(); spnn.load_state_dict(ck["model"])
sd = spnn.state_dict(); sd.update({k: v.to(sd[k].dtype) for k, v in ck["ema"].items()}); spnn.load_state_dict(sd)
spnn = spnn.to(DEV).float().eval().requires_grad_(False)
print("spnn step", ck["state"]["step"])

vae = AutoencoderKL.from_pretrained(VID, subfolder="vae", local_files_only=True).to(DEV, torch.float16).eval()

class Hybrid(AutoencoderKL):
    def attach(self, s, sc=SC): self.spnn, self.latent_scale = s, sc; return self
    def _decode(self, z, return_dict=True):
        x = self.spnn.decode(z.float() * self.latent_scale).to(z.dtype)
        return (x,) if not return_dict else DecoderOutput(sample=x)
    def decode(self, z, return_dict=True, generator=None):
        return DecoderOutput(sample=self._decode(z).sample)
hyb = Hybrid.from_pretrained(VID, subfolder="vae", local_files_only=True).to(DEV, torch.float16).eval().attach(spnn)

im = Image.open("/rg/shocher_prj/ron.libman/reed_table1/data/108871_t2_src.png").convert("RGB")
x16 = (torch.from_numpy(np.asarray(im, np.float32)/127.5-1).permute(2,0,1)[None].to(DEV)).half()
rel = lambda a, b: (a.float()-b.float()).norm().item() / b.float().norm().item()
with torch.no_grad():
    zv = vae.encode(x16).latent_dist.mean
    zh = hyb.encode(x16).latent_dist.mean
    zs = spnn.encode(x16.float()) / SC
    print(f"\nENCODER   ||E_hyb - E_vae|| / ||E_vae||  = {rel(zh, zv):.3e}   <- 0 means the hybrid encoder IS the SD VAE")
    print(f"          ||E_spnn - E_vae|| / ||E_vae||  = {rel(zs, zv):.3e}   <- how far SPNN's encoder actually is")
    print(f"          latent rms: SD-VAE {zv.float().square().mean().sqrt():.4f}   SPNN {zs.square().mean().sqrt():.4f}")
    sv = vae.encode(x16).latent_dist.std
    print(f"          SD-VAE posterior std: mean {sv.float().mean():.4f}  vs |mean latent| {zv.float().abs().mean():.4f}"
          f"   (SPNN shim uses std ~3e-7, i.e. deterministic)")
    dh = hyb.decode(zv).sample
    ds = spnn.decode(zv.float()*SC)
    dv = vae.decode(zv).sample
    print(f"\nDECODER   ||D_hyb - D_spnn|| / ||D_spnn|| = {rel(dh, ds):.3e}   <- 0 means the hybrid decoder IS SPNN")
    print(f"          ||D_hyb - D_vae||  / ||D_vae||  = {rel(dh, dv):.3e}")
    # round-trip fidelity of each full codec, one pass
    psnr = lambda a,b: 10*np.log10(4.0/ (a.float()-b.float()).pow(2).mean().item())
    rt_v = vae.decode(vae.encode(x16).latent_dist.mean).sample
    rt_s = spnn.decode(spnn.encode(x16.float()))
    rt_h = hyb.decode(hyb.encode(x16).latent_dist.mean).sample
    print(f"\n1x ROUND TRIP PSNR   SD-VAE {psnr(rt_v,x16):.2f}   SPNN {psnr(rt_s,x16):.2f}   hybrid {psnr(rt_h,x16):.2f}")
    # idempotency of the latent round trip
    for nm, enc, dec in (("SD-VAE", lambda t: vae.encode(t).latent_dist.mean, lambda t: vae.decode(t).sample),
                         ("SPNN",   lambda t: spnn.encode(t.float())/SC,      lambda t: spnn.decode(t.float()*SC)),
                         ("hybrid", lambda t: hyb.encode(t).latent_dist.mean, lambda t: hyb.decode(t).sample)):
        z = enc(x16); z2 = enc(dec(z).to(x16.dtype))
        print(f"latent idempotency ||E(D(z))-z||/||z||  {nm:7s} {rel(z2, z):.3e}")
