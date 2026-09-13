#!/usr/bin/env python3
"""Re-render specific eval images from the restoration benchmark as full-res strips.

Reuses eval_restore_fid's loaders, codecs and operators, and reproduces the benchmark's own
per-batch seed (1_000_003*(task_index+1) + batch), so the pictures are literally the ones scored.
Usage:  ddnm_examples.py --out <eval dir> --task mask --T 200 --ids 1365,3008 --dest <dir>
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np, torch

import eval_restore_fid as E
import latent_ddnm as ld
from latent_ddnm_idem import latent_ddnm_idem
from spnn_model_opt import SPNNAutoencoder512Opt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--task", required=True, choices=E.TASKS)
    p.add_argument("--T", type=int, required=True)
    p.add_argument("--ids", required=True)
    p.add_argument("--eta", type=float, default=0.85)
    p.add_argument("--lam-hole", type=float, default=0.0)
    p.add_argument("--bs", type=int, default=4)
    a = p.parse_args()

    dev = torch.device("cuda")
    out, dest = Path(a.out), Path(a.dest)
    want = sorted({int(x) for x in a.ids.split(",") if x.strip()})
    man = json.load(open(out / "manifest.json"))
    ev = sorted(man["eval"], key=lambda e: e["id"])
    batches = [ev[k:k + a.bs] for k in range(0, len(ev), a.bs)]
    need = sorted({i // a.bs for i in want})
    print(f"task={a.task} T={a.T} ids={want} -> batches {need}", flush=True)

    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer
    unet = UNet2DConditionModel.from_pretrained(E.SD_ID, subfolder="unet", torch_dtype=torch.float16).to(dev).eval()
    betas = DDIMScheduler.from_pretrained(E.SD_ID, subfolder="scheduler").betas.to(dev)
    tok = CLIPTokenizer.from_pretrained(E.SD_ID, subfolder="tokenizer")
    te = CLIPTextModel.from_pretrained(E.SD_ID, subfolder="text_encoder").to(dev)
    with torch.no_grad():
        cond = te(tok([""], padding="max_length", max_length=tok.model_max_length,
                      return_tensors="pt").input_ids.to(dev))[0].half()
    del te
    eps = lambda z, t: unet(z.half(), t, encoder_hidden_states=cond.expand(z.shape[0], -1, -1)).sample.float()

    vae = AutoencoderKL.from_pretrained(E.SD_ID, subfolder="vae", torch_dtype=torch.float32).to(dev).eval()
    snap = torch.load(out / "spnn_eval.pt", map_location="cpu", weights_only=False)
    spnn = SPNNAutoencoder512Opt(); spnn.load_state_dict(snap["state_dict"]); spnn.to(dev).eval()
    codecs = {"sdvae": ld.SDVAECodec(vae, 0.18215), "spnn": ld.SPNNCodec(spnn)}
    print(f"spnn step={snap['step']} ema={snap['ema']}", flush=True)

    scorer = E.Scorer(dev)
    ddnm_mask = ld.load_mask(E.DDNM_MASK, E.SIZE, dev)
    ti = E.TASKS.index(a.task)
    imgs = E.read_items([it for b in need for it in batches[b]], 6)

    for b in need:
        ids = [it["id"] for it in batches[b]]
        x01 = E.to_batch([imgs[i] for i in ids], dev)
        x = x01 * 2 - 1
        if a.task == "color":
            A, Ap = ld.color_ops(); m = None
        else:
            m = ddnm_mask.expand(len(ids), -1, -1, -1) if a.task == "mask" else \
                torch.cat([E.scatter_mask(E.SIZE, 0.66, 9, seed=i) for i in ids]).to(dev)
            A, Ap = ld.inpainting_ops(m)
        y = A(x)
        seed = 1_000_003 * (ti + 1) + b
        res = {}
        for name, codec in codecs.items():
            g = torch.Generator(device=dev).manual_seed(seed)
            t0 = time.time()
            r = latent_ddnm_idem(eps, betas, codec, A, Ap, y, [4, 64, 64], T_sampling=a.T, eta=a.eta,
                                 generator=g, mask=m, lam_hole=a.lam_hole if m is not None else None)
            xo = E.quant(r["x"].to(dev))
            res[name] = (xo, scorer.score(xo, x01, a.task, m))
            print(f"  b{b} {name}: {time.time()-t0:.1f}s", flush=True)
        shown = E.quant(Ap(y)) if a.task == "color" else E.quant(y)
        (xv, dv), (xs, ds) = res["sdvae"], res["spnn"]
        for k, i in enumerate(ids):
            if i not in want:
                continue
            labels = ["original", "degraded input",
                      f"SD-VAE   LPIPS {dv['lpips'][k]:.3f}   PSNR {dv['psnr'][k]:.1f}",
                      f"SPNN   LPIPS {ds['lpips'][k]:.3f}   PSNR {ds['psnr'][k]:.1f}"]
            E.save_strip(dest / f"{a.task}_T{a.T}_{i:05d}.jpg", [x01[k], shown[k], xv[k], xs[k]], labels)
            print(f"  wrote {a.task}_T{a.T}_{i:05d}.jpg  "
                  f"sdvae {dv['lpips'][k]:.3f} / spnn {ds['lpips'][k]:.3f}", flush=True)


if __name__ == "__main__":
    main()
