#!/usr/bin/env python3
"""
Restoration benchmark for the SPNN-512 codec vs the SD-VAE: latent DDNM (latent_ddnm_idem, the notebook's sampler) on
held-out test images, scored against the originals.

  python3 eval_restore_fid.py prepare  --out OUT     # CPU, once: choose the images -> OUT/manifest.json
  python3 eval_restore_fid.py snapshot --out OUT     # CPU, once: freeze the SPNN checkpoint (EMA) -> OUT/spnn_eval.pt
  torchrun --standalone --nproc_per_node=4 eval_restore_fid.py run --out OUT   # 1 process per GPU, resumable
  python3 eval_restore_fid.py report   --out OUT     # FID/KID + averages -> OUT/report.{json,md}, sample grids

Images come from the training run's own split (train_spnn512.build_keep_and_splits, seed 42): nothing here was trained on.
  eval set  N_EVAL test images, LAION-test / COCO-test in the training mix (86% / 14% of the train images)
  FID, KID  against the originals of those same test images (restoration-paper convention)
Tasks use the notebook's settings: DDNM noise rule, eta 0.85, empty prompt, hole damping 0 for inpainting.
Every batch has a fixed seed, so both codecs see the same noise. SD-VAE results live under a tag that does not depend on
the SPNN checkpoint, so a later checkpoint only reruns the SPNN (~16% of the cost).
During the run, OUT/highlights/<task>/ collects full-res strips (original | input | SD-VAE | SPNN): `periodic` every
--save-every batches, `spnn_best_lpips` the most faithful SPNN results, `spnn_beats_sdvae` where the SPNN beats the SD-VAE
by the widest LPIPS margin (top --top-k per GPU each; kept across restarts).
Metrics are computed on uint8-quantised images (what a saved PNG holds). FID features: the standard FID Inception
(pytorch-fid weights) on an antialiased bicubic 299 resize (clean-fid style).
"""
import argparse
import io
import json
import os
import random
import sys
import tarfile
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "simple-latent-diffusion-model-master" / "simple-latent-diffusion-model"))
sys.path.insert(0, str(ROOT))
PYLIBS = "/rg/shocher_prj/ron.libman/pylibs_eval"          # scipy, lpips, pytorch-fid (pip --target; /home is near quota)
sys.path.insert(0, PYLIBS)
DATA = Path("/rg/shocher_prj/ron.libman/laion1m")
SD_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DDNM_MASK = ROOT / "DDNM-main" / "exp" / "inp_masks" / "mask.npy"
TASKS = ("mask", "scatter66", "color")                     # fixed order: the task index is part of every batch seed
SIZE = 512


# ------------------------------------------------------------------------------------------------------------ images
def load_crop(data, size=SIZE):
    """The notebook's load_img: short side -> size (bicubic, up or down), center crop. Returns uint8 HWC."""
    im = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = im.size
    s = size / min(w, h)
    im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    w, h = im.size
    l, t = (w - size) // 2, (h - size) // 2
    return np.asarray(im.crop((l, t, l + size, t + size)), dtype=np.uint8)


def _keep_numpy(x):
    return x


class TarItems(torch.utils.data.IterableDataset):
    """Streams (id, uint8 HWC crop) for the wanted members of each tar; DataLoader workers split the tar list."""

    def __init__(self, groups):                            # groups: [(tar_path, {member_name: id})]
        self.groups = groups

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        w, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        for tar, want in self.groups[w::nw]:
            with tarfile.open(tar) as t:
                for m in t:
                    i = want.get(m.name)
                    if i is not None:
                        yield i, load_crop(t.extractfile(m).read())


def read_items(items, workers):
    """[{"id", "tar", "member"}] -> {id: uint8 HWC numpy array}."""
    groups = {}
    for it in items:
        groups.setdefault(it["tar"], {})[it["member"]] = it["id"]
    dl = torch.utils.data.DataLoader(TarItems(sorted(groups.items())), batch_size=None, collate_fn=_keep_numpy,
                                     num_workers=min(workers, len(groups)))
    return dict(dl)


def to_batch(arrs, dev):
    """list of uint8 HWC -> float [B,3,H,W] in [0,1] on dev."""
    return torch.from_numpy(np.stack(arrs)).to(dev).permute(0, 3, 1, 2).float() / 255.0


def scatter_mask(size, missing_frac, n=9, seed=0, candidates=10):
    """The notebook's scattered squares: n equal squares at seeded best-candidate positions (overlaps allowed), side
    chosen so they cover `missing_frac` of the image. [1,1,size,size] float, 1 = known pixel, 0 = hole."""
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(1, 2, generator=g)
    for _ in range(n - 1):
        c = torch.rand(candidates, 2, generator=g)
        u = torch.cat([u, c[torch.cdist(c, u).min(1).values.argmax()][None]])

    def build(side):
        m = np.ones((size, size), np.float32)
        for a, b in (u * (size - side)).long().tolist():
            m[a:a + side, b:b + side] = 0
        return m
    side = min(range(8, size + 1), key=lambda s: abs(1 - float(build(s).mean()) - missing_frac))
    return torch.from_numpy(build(side))[None, None]


# ----------------------------------------------------------------------------------------------------------- metrics
def quant(x):
    """[-1,1] -> uint8-quantised [0,1]."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round() / 255.0


def psnr(x, y, w=None):
    """Per image, data range 1. w: optional [B,1,H,W] pixel weights (region PSNR)."""
    d = (x - y) ** 2
    if w is None:
        mse = d.mean((1, 2, 3))
    else:
        mse = (d * w).sum((1, 2, 3)) / (w.sum((1, 2, 3)) * x.shape[1]).clamp_min(1)
    return 10 * torch.log10(1.0 / mse.clamp_min(1e-10))


_GAUSS = {}


def ssim(x, y):
    """Wang et al. 2004: 11x11 Gaussian window (sigma 1.5), K1 0.01, K2 0.03, data range 1, valid window, mean over RGB."""
    key = str(x.device)
    if key not in _GAUSS:
        g = torch.exp(-((torch.arange(11, dtype=torch.float32) - 5) ** 2) / (2 * 1.5 ** 2))
        g = g / g.sum()
        _GAUSS[key] = (g[:, None] * g[None, :]).expand(3, 1, 11, 11).contiguous().to(x.device)
    f = lambda t: F.conv2d(t, _GAUSS[key], groups=3)
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
    return s.mean((1, 2, 3))


def colorfulness(x):
    """Hasler & Suesstrunk 2003 on [0,255]: ~0 for gray, higher = more colourful."""
    r, g, b = (x * 255).unbind(1)
    rg, yb = r - g, 0.5 * (r + g) - b
    return torch.sqrt(rg.std((1, 2)) ** 2 + yb.std((1, 2)) ** 2) + 0.3 * torch.sqrt(rg.mean((1, 2)) ** 2 + yb.mean((1, 2)) ** 2)


class Scorer:
    def __init__(self, dev):
        import lpips
        from pytorch_fid.inception import InceptionV3
        self.lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
        self.inc = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]], resize_input=False, normalize_input=True).to(dev).eval()

    @torch.no_grad()
    def feats(self, x):                                    # [0,1] -> [B,2048]
        x = F.interpolate(x, size=(299, 299), mode="bicubic", antialias=True, align_corners=False).clamp(0, 1)
        return self.inc(x)[0].flatten(1)

    @torch.no_grad()
    def score(self, xo, x0, task, m):
        """xo: quantised output, x0: original, both [0,1]; m: [B,1,H,W] known-pixel mask for inpainting."""
        d = {"psnr": psnr(xo, x0), "ssim": ssim(xo, x0), "lpips": self.lp(xo * 2 - 1, x0 * 2 - 1).flatten(),
             "colorful": colorfulness(xo)}
        if m is not None:
            d["hole_psnr"] = psnr(xo, x0, 1 - m)                     # the filled region only
            d["cons_psnr"] = psnr(xo, x0, m)                         # observed pixels kept?
        elif task == "color":
            d["cons_psnr"] = psnr(xo.mean(1, keepdim=True), x0.mean(1, keepdim=True))   # gray kept?
        return d


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **{k: (v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)) for k, v in arrays.items()})
    os.replace(tmp, path)


def save_tile(x01, path, size=256):
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray((x01.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy())
    im.resize((size, size), Image.LANCZOS).save(path, quality=92)


def save_strip(path, tiles, labels):
    """One row of full-res tiles ([3,H,W] in [0,1]) with a caption above each."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ims = [Image.fromarray((t.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()) for t in tiles]
    w, h = ims[0].size
    canvas = Image.new("RGB", (len(ims) * (w + 4) + 4, h + 30), "white")
    dr = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
    for k, (im, lb) in enumerate(zip(ims, labels)):
        canvas.paste(im, (4 + k * (w + 4), 26))
        dr.text((8 + k * (w + 4), 4), lb, fill="black", font=font)
    tmp = path.with_name(path.stem + ".tmp.jpg")
    canvas.save(tmp, quality=92)
    os.replace(tmp, path)


class Highlights:
    """The top-k images of one criterion (higher = better) as strips on disk. Scores live in the file names, so a
    restarted job picks up where it left off instead of adding a second top-k."""

    def __init__(self, d, rank, k):
        self.d, self.rank, self.k = Path(d), rank, k
        self.items = []
        for f in self.d.glob(f"r{rank}_*.jpg"):
            try:
                self.items.append((float(f.stem.split("_")[1]), f))
            except (IndexError, ValueError):
                pass

    def offer(self, score, name, make):
        if len(self.items) >= self.k and score <= min(self.items)[0]:
            return
        p = self.d / f"r{self.rank}_{score:+.4f}_{name}.jpg"
        make(p)
        self.items.append((score, p))
        if len(self.items) > self.k:
            worst = min(self.items)
            self.items.remove(worst)
            worst[1].unlink(missing_ok=True)


# ----------------------------------------------------------------------------------------------------------- prepare
def cmd_prepare(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mf = out / "manifest.json"
    if mf.exists() and not a.force:
        m = json.load(open(mf))
        print(f"[prepare] keeping existing {mf}: {len(m['eval'])} eval images (--force rebuilds)")
        return
    from train_spnn512 import build_keep_and_splits
    keep, splits, _ = build_keep_and_splits(str(DATA), 512, 4, 42, 8, 4)      # the training run's exact split
    sid = lambda p: f"{Path(p).parent.name}/{Path(p).stem}"
    by_shard = {}
    for k in keep:
        s, key = k.rsplit("/", 1)
        by_shard.setdefault(s, []).append(key)
    n_train = sum(len(by_shard.get(sid(p), [])) for p in splits["train_laion"])
    assert n_train == 595824, f"LAION train count {n_train:,} != training log 595,824: split differs from the training run"
    laion = lambda tars: [(str(p), k + ".jpg") for p in tars for k in sorted(by_shard.get(sid(p), []))]

    def coco(tars):
        items = []
        for p in tars:
            with tarfile.open(p) as t:
                items += [(str(p), m.name) for m in t if m.name.endswith(".jpg")]
        return items

    rng = random.Random(a.seed)
    pool_l, pool_c = laion(splits["test_laion"]), coco(splits["test_coco"])
    n_l = round(a.n_eval * a.laion_frac)
    n_c = a.n_eval - n_l
    assert n_l <= len(pool_l) and n_c <= len(pool_c), f"asked {n_l}+{n_c}, have {len(pool_l)}+{len(pool_c)}"
    ev = [("laion", x) for x in rng.sample(pool_l, n_l)] + [("coco", x) for x in rng.sample(pool_c, n_c)]
    rng.shuffle(ev)                                      # mixed order: every rank and every sample grid sees both
    ev_tars = {t for _, (t, _) in ev}
    seen_tars = {str(p) for k in ("train_laion", "train_coco", "val_laion", "val_coco") for p in splits[k]}
    assert not (ev_tars & seen_tars), "an eval image comes from a train/val shard"
    man = {"settings": {"seed": a.seed, "n_eval": a.n_eval, "laion_frac": a.laion_frac,
                        "split": "train_spnn512.build_keep_and_splits(seed=42, val_shards=4, test_shards=4)",
                        "available_test": {"laion": len(pool_l), "coco": len(pool_c)}},
           "eval": [{"id": i, "src": s, "tar": t, "member": m} for i, (s, (t, m)) in enumerate(ev)]}
    json.dump(man, open(mf, "w"))
    cnt = {k: sum(1 for r in ev if r[0] == k) for k in ("laion", "coco")}
    print(f"[prepare] eval: {cnt} of available test {man['settings']['available_test']} from {len(ev_tars)} test shards")
    print(f"[prepare] wrote {mf}")


def cmd_snapshot(a):
    out = Path(a.out)
    p = out / "spnn_eval.pt"
    if p.exists() and not a.force:
        print(f"[snapshot] keeping existing {p} (step {torch.load(p, map_location='cpu', weights_only=False)['step']}; --force refreshes)")
        return
    from spnn_model_opt import SPNNAutoencoder512Opt
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    spnn = SPNNAutoencoder512Opt()
    spnn.load_state_dict(ck["model"])
    sd = spnn.state_dict()
    if not a.no_ema:
        sd.update({k: v.to(sd[k].dtype) for k, v in ck["ema"].items()})
    out.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    torch.save({"state_dict": sd, "step": ck["state"]["step"], "ema": not a.no_ema, "src": str(a.ckpt)}, tmp)
    os.replace(tmp, p)
    print(f"[snapshot] {a.ckpt} step {ck['state']['step']} ({'ema' if not a.no_ema else 'online'} weights) -> {p}")


# --------------------------------------------------------------------------------------------------------------- run
def cmd_run(a):
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    dev = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    torch.cuda.set_device(dev)
    log = lambda s: print(f"[rank {rank}/{world} {time.strftime('%H:%M:%S')}] {s}", flush=True)
    out, res = Path(a.out), Path(a.out) / "res"
    man = json.load(open(out / "manifest.json"))
    tasks = [t for t in a.tasks.split(",") if t]
    assert all(t in TASKS for t in tasks), f"tasks must be in {TASKS}"
    codec_names = [c for c in a.codecs.split(",") if c]
    ev = sorted(man["eval"], key=lambda e: e["id"])
    batches = [ev[k:k + a.bs] for k in range(0, len(ev), a.bs)]          # global batches: independent of world size
    mine = [b for b in range(len(batches)) if b % world == rank]
    scorer = Scorer(dev)

    # 1. models (the notebook's setup)
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer
    import latent_ddnm as ld
    from latent_ddnm_idem import latent_ddnm_idem
    from spnn_model_opt import SPNNAutoencoder512Opt
    unet = UNet2DConditionModel.from_pretrained(SD_ID, subfolder="unet", torch_dtype=torch.float16).to(dev).eval()
    betas = DDIMScheduler.from_pretrained(SD_ID, subfolder="scheduler").betas.to(dev)
    tok = CLIPTokenizer.from_pretrained(SD_ID, subfolder="tokenizer")
    te = CLIPTextModel.from_pretrained(SD_ID, subfolder="text_encoder").to(dev)
    with torch.no_grad():
        cond = te(tok([""], padding="max_length", max_length=tok.model_max_length, return_tensors="pt").input_ids.to(dev))[0].half()
    del te
    eps = lambda z, t: unet(z.half(), t, encoder_hidden_states=cond.expand(z.shape[0], -1, -1)).sample.float()
    codecs = {}
    if "sdvae" in codec_names:
        vae = AutoencoderKL.from_pretrained(SD_ID, subfolder="vae", torch_dtype=torch.bfloat16 if a.vae_dtype == "bf16" else torch.float32).to(dev).eval()
        codecs["sdvae"] = (f"sdvae_{a.vae_dtype}", ld.SDVAECodec(vae, 0.18215))
    if "spnn" in codec_names:
        snap = torch.load(out / "spnn_eval.pt", map_location="cpu", weights_only=False)
        spnn = SPNNAutoencoder512Opt()
        spnn.load_state_dict(snap["state_dict"])
        spnn.to(dev).eval()
        codecs["spnn"] = (f"spnn_s{snap['step']}{'' if snap['ema'] else '_online'}", ld.SPNNCodec(spnn))
    for name, (tag, _) in codecs.items():
        if rank == 0:
            (res / tag).mkdir(parents=True, exist_ok=True)
            json.dump({"codec": name, "tag": tag, "vae_dtype": a.vae_dtype, "spnn": str(out / "spnn_eval.pt") if name == "spnn" else None},
                      open(res / tag / "info.json", "w"))
    ddnm_mask = ld.load_mask(DDNM_MASK, SIZE, dev)
    steps = lambda task: a.T_color if task == "color" else a.T
    tdir = lambda task: f"{task}_T{steps(task)}_eta{a.eta}" + ("" if task == "color" else f"_lh{a.lam_hole}") + f"_bs{a.bs}"

    # 2. this rank's eval images
    imgs = read_items([it for b in mine for it in batches[b]], a.workers)
    log(f"{len(imgs)} eval images loaded; {len(mine)} batches of {a.bs}; codecs {list(codecs)}; "
        f"tasks {[f'{t} (T={steps(t)})' for t in tasks]}")
    X = lambda b: to_batch([imgs[it["id"]] for it in batches[b]], dev)
    samples = out / "samples"

    # 3. originals (the FID/KID reference) and plain reconstruction D(E(x)) = each codec's floor
    for b in mine:
        ids = np.array([it["id"] for it in batches[b]])
        x01 = None
        p = res / "orig" / f"b{b:05d}.npz"
        if not p.exists():
            x01 = X(b)
            save_npz(p, ids=ids, feats=scorer.feats(x01).half(), colorful=colorfulness(x01))
        for name, (tag, codec) in codecs.items():
            p = res / tag / "recon" / f"b{b:05d}.npz"
            if p.exists():
                continue
            x01 = X(b) if x01 is None else x01
            torch.cuda.synchronize(); t0 = time.time()
            with torch.no_grad():
                xr = quant(codec.decode(codec.encode(x01 * 2 - 1)))
            torch.cuda.synchronize(); sec = (time.time() - t0) / len(ids)
            d = scorer.score(xr, x01, "recon", None)
            save_npz(p, ids=ids, feats=scorer.feats(xr).half(), sec=np.full(len(ids), sec), **d)
        if b < a.sample_batches:
            x01 = X(b) if x01 is None else x01
            for k, i in enumerate(ids):
                save_tile(x01[k], samples / f"{i:05d}_orig.jpg")
    log("originals + reconstruction floors done")

    # 4. restoration tasks
    for task in tasks:
        ti = TASKS.index(task)
        t_task, n_run = time.time(), 0
        sec_sum = {n: 0.0 for n in codecs}
        hl = {c: Highlights(out / "highlights" / tdir(task) / c, rank, a.top_k) for c in ("spnn_best_lpips", "spnn_beats_sdvae")}
        for nb, b in enumerate(mine):
            todo = [(n, tag, c) for n, (tag, c) in codecs.items() if not (res / tag / tdir(task) / f"b{b:05d}.npz").exists()]
            if todo:
                ids = [it["id"] for it in batches[b]]
                x01 = X(b)
                x = x01 * 2 - 1
                if task == "color":
                    A, Ap = ld.color_ops()
                    m = None
                else:
                    m = ddnm_mask.expand(len(ids), -1, -1, -1) if task == "mask" else \
                        torch.cat([scatter_mask(SIZE, 0.66, 9, seed=i) for i in ids]).to(dev)   # a new layout per image
                    A, Ap = ld.inpainting_ops(m)
                y = A(x)
                seed = 1_000_003 * (ti + 1) + b                              # same noise for every codec
                outs = {}
                for name, tag, codec in todo:
                    g = torch.Generator(device=dev).manual_seed(seed)
                    torch.cuda.synchronize(); t0 = time.time()
                    r = latent_ddnm_idem(eps, betas, codec, A, Ap, y, [4, 64, 64], T_sampling=steps(task), eta=a.eta, generator=g,
                                         mask=m, lam_hole=a.lam_hole if m is not None else None)
                    torch.cuda.synchronize(); sec = (time.time() - t0) / len(ids)
                    xo = quant(r["x"].to(dev))
                    d = scorer.score(xo, x01, task, m)
                    save_npz(res / tag / tdir(task) / f"b{b:05d}.npz", ids=np.array(ids), feats=scorer.feats(xo).half(),
                             sec=np.full(len(ids), sec), **d)
                    sec_sum[name] += sec * len(ids)
                    outs[name] = (xo, d)
                    if b < a.sample_batches:
                        for k, i in enumerate(ids):
                            save_tile(xo[k], samples / tdir(task) / f"{i:05d}_{name}.jpg")
                shown = quant(Ap(y)) if task == "color" else quant(y)
                if b < a.sample_batches:
                    for k, i in enumerate(ids):
                        save_tile(shown[k], samples / tdir(task) / f"{i:05d}_input.jpg")
                if "sdvae" in outs and "spnn" in outs:                   # highlights: periodic + the best SPNN results
                    (xv, dv), (xs, ds) = outs["sdvae"], outs["spnn"]

                    def strip(k):
                        labels = ["original", "input", f"SD-VAE  LPIPS {dv['lpips'][k]:.3f}  PSNR {dv['psnr'][k]:.1f}",
                                  f"SPNN  LPIPS {ds['lpips'][k]:.3f}  PSNR {ds['psnr'][k]:.1f}"]
                        return lambda p: save_strip(p, [x01[k], shown[k], xv[k], xs[k]], labels)
                    if b % a.save_every == 0:
                        strip(0)(out / "highlights" / tdir(task) / "periodic" / f"b{b:05d}_{ids[0]:05d}.jpg")
                    for k, i in enumerate(ids):
                        hl["spnn_best_lpips"].offer(-float(ds["lpips"][k]), f"{i:05d}", strip(k))
                        hl["spnn_beats_sdvae"].offer(float(dv["lpips"][k] - ds["lpips"][k]), f"{i:05d}", strip(k))
                n_run += len(ids)
            if (nb + 1) % 10 == 0 or nb + 1 == len(mine):
                el = time.time() - t_task
                left = (len(mine) - nb - 1) * el / (nb + 1) if n_run else 0
                per = " ".join(f"{n} {sec_sum[n] / max(n_run, 1):.1f}s/img" for n in codecs)
                log(f"{task}: batch {nb + 1}/{len(mine)} | {per} | task ETA {left / 3600:.2f} h")
    log("all done")


# ------------------------------------------------------------------------------------------------------------ report
def _load(d):
    files = sorted(Path(d).glob("b*.npz"))
    if not files:
        return None
    rows = [dict(np.load(f)) for f in files]
    return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}


def _stats(f):
    f = f.astype(np.float64)
    return f.mean(0), np.cov(f, rowvar=False)


def _kid(f1, f2, n_sub=100, sub=1000, seed=0, dev="cpu"):
    """Unbiased KID (polynomial kernel, degree 3), mean +- std over subsets; as in clean-fid / torch-fidelity."""
    g = torch.Generator().manual_seed(seed)
    a, b = torch.from_numpy(f1).double().to(dev), torch.from_numpy(f2).double().to(dev)
    m, d = min(sub, len(a), len(b)), a.shape[1]
    vals = []
    for _ in range(n_sub):
        x = a[torch.randperm(len(a), generator=g)[:m].to(dev)]
        y = b[torch.randperm(len(b), generator=g)[:m].to(dev)]
        kxx, kyy, kxy = (x @ x.T / d + 1) ** 3, (y @ y.T / d + 1) ** 3, (x @ y.T / d + 1) ** 3
        t = (kxx.sum() - kxx.diagonal().sum() + kyy.sum() - kyy.diagonal().sum()) / (m - 1) - 2 * kxy.sum() / m
        vals.append((t / m).item())
    return float(np.mean(vals)), float(np.std(vals))


def cmd_report(a):
    from pytorch_fid.fid_score import calculate_frechet_distance
    out, res = Path(a.out), Path(a.out) / "res"
    man = json.load(open(out / "manifest.json"))
    src = {e["id"]: e["src"] for e in man["eval"]}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    orig = _load(res / "orig")
    assert orig is not None, "no originals yet: run first"
    o_idx = {int(i): k for k, i in enumerate(orig["ids"])}
    fid = lambda f1, f2: float(calculate_frechet_distance(*_stats(f1), *_stats(f2)))
    rows = []
    for tag_dir in sorted(p for p in res.iterdir() if p.is_dir() and p.name != "orig"):
        for task_dir in sorted(p for p in tag_dir.iterdir() if p.is_dir()):
            r = _load(task_dir)
            if r is None:
                continue
            k = np.array([o_idx[int(i)] for i in r["ids"]])
            fo, fr = orig["feats"][k], r["feats"]
            row = {"codec": tag_dir.name, "task": task_dir.name, "n": len(r["ids"]), "complete": len(r["ids"]) == len(man["eval"]),
                   "fid_orig": fid(fr, fo)}
            row["kid_orig_x1e3"], row["kid_orig_std_x1e3"] = [v * 1e3 for v in _kid(fr, fo, dev=dev)]
            for met in ("psnr", "hole_psnr", "ssim", "lpips", "cons_psnr"):
                if met in r:
                    row[met] = float(np.mean(r[met]))
            row["colorful_ratio"] = float(np.mean(r["colorful"]) / max(np.mean(orig["colorful"][k]), 1e-6))
            row["sec_per_img"] = float(np.mean(r["sec"]))
            for s in ("laion", "coco"):
                sel = np.array([src[int(i)] == s for i in r["ids"]])
                if sel.sum() >= 2:
                    row[f"{s}_n"] = int(sel.sum())
                    for met in ("psnr", "hole_psnr", "ssim", "lpips"):
                        if met in r:
                            row[f"{s}_{met}"] = float(np.mean(r[met][sel]))
                    if sel.sum() >= 500:
                        row[f"{s}_fid_orig"] = fid(fr[sel], fo[sel])
            rows.append(row)
            print(f"[report] {tag_dir.name}/{task_dir.name}: n={row['n']} FID(orig) {row['fid_orig']:.2f}", flush=True)
    json.dump({"settings": man["settings"], "rows": rows}, open(out / "report.json", "w"), indent=1)
    # markdown
    cols = [("n", "{:d}"), ("fid_orig", "{:.2f}"), ("kid_orig_x1e3", "{:.2f}"), ("psnr", "{:.2f}"),
            ("hole_psnr", "{:.2f}"), ("ssim", "{:.4f}"), ("lpips", "{:.4f}"), ("cons_psnr", "{:.1f}"),
            ("colorful_ratio", "{:.2f}"), ("sec_per_img", "{:.2f}")]
    md = ["# Restoration benchmark", "", f"eval images: {len(man['eval'])} ({man['settings']['laion_frac']:.0%} LAION test, rest COCO test); "
          "FID / KID against the originals of the same test images", "",
          "| codec | task | " + " | ".join(c for c, _ in cols) + " |", "|" + "---|" * (len(cols) + 2)]
    for r in rows:
        cells = [(f.format(r[c]) if r.get(c) is not None else "") for c, f in cols]
        md.append(f"| {r['codec']} | {r['task']}{'' if r.get('complete', True) else ' (partial)'} | " + " | ".join(cells) + " |")
    md += ["", "Per source (paired metrics; FID when >= 500 images):", "",
           "| codec | task | laion psnr | laion lpips | laion fid | coco psnr | coco lpips | coco fid |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "laion_n" in r or "coco_n" in r:
            g = lambda k, f="{:.2f}": f.format(r[k]) if r.get(k) is not None else ""
            md.append(f"| {r['codec']} | {r['task']} | {g('laion_psnr')} | {g('laion_lpips', '{:.4f}')} | {g('laion_fid_orig')} | "
                      f"{g('coco_psnr')} | {g('coco_lpips', '{:.4f}')} | {g('coco_fid_orig')} |")
    open(out / "report.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    # sample grids: original | input | one column per codec
    smp = out / "samples"
    if smp.exists():
        for td in sorted(p for p in smp.iterdir() if p.is_dir()):
            ids = sorted({f.name.split("_")[0] for f in td.glob("*_input.jpg")})
            names = sorted({f.stem.split("_", 1)[1] for f in td.glob("*.jpg")} - {"input"})
            colnames = ["orig", "input"] + names
            tile = 256
            grid = Image.new("RGB", (len(colnames) * (tile + 4) + 4, len(ids) * (tile + 4) + 24), "white")
            from PIL import ImageDraw
            dr = ImageDraw.Draw(grid)
            for c, n in enumerate(colnames):
                dr.text((4 + c * (tile + 4) + 4, 4), n, fill="black")
                for rr, i in enumerate(ids):
                    f = (smp / f"{i}_orig.jpg") if n == "orig" else (td / f"{i}_{n}.jpg")
                    if f.exists():
                        grid.paste(Image.open(f).resize((tile, tile)), (4 + c * (tile + 4), 24 + rr * (tile + 4)))
            grid.save(out / f"grid_{td.name}.png")
    print(f"[report] wrote {out / 'report.md'}, {out / 'report.json'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("prepare")
    pp.add_argument("--out", required=True)
    pp.add_argument("--n-eval", type=int, default=5000)
    pp.add_argument("--laion-frac", type=float, default=0.858,
                    help="share of LAION-test images (rest COCO-test); default = the training mix, 595,824 / (595,824 + 98,509)")
    pp.add_argument("--seed", type=int, default=0)
    pp.add_argument("--force", action="store_true")
    ps = sub.add_parser("snapshot")
    ps.add_argument("--out", required=True)
    ps.add_argument("--ckpt", default=str(DATA / "runs" / "spnn512" / "best.pt"))
    ps.add_argument("--no-ema", action="store_true")
    ps.add_argument("--force", action="store_true")
    pr = sub.add_parser("run")
    pr.add_argument("--out", required=True)
    pr.add_argument("--tasks", default="mask,scatter66,color")
    pr.add_argument("--codecs", default="sdvae,spnn")
    pr.add_argument("--T", type=int, default=100, help="DDNM steps for inpainting (mask, scatter66)")
    pr.add_argument("--T-color", type=int, default=20, help="DDNM steps for colorization")
    pr.add_argument("--eta", type=float, default=0.85)
    pr.add_argument("--lam-hole", type=float, default=0.0, help="hole damping for inpainting (notebook LAM_HOLE)")
    pr.add_argument("--vae-dtype", default="fp32", choices=["fp32", "bf16"])
    pr.add_argument("--bs", type=int, default=4)
    pr.add_argument("--workers", type=int, default=6)
    pr.add_argument("--sample-batches", type=int, default=4, help="save 256px tiles of the first N batches for grids")
    pr.add_argument("--save-every", type=int, default=25, help="a full-res highlight strip every N batches (~100 images)")
    pr.add_argument("--top-k", type=int, default=6, help="best-SPNN strips kept per GPU and criterion")
    pq_ = sub.add_parser("report")
    pq_.add_argument("--out", required=True)
    a = p.parse_args()
    {"prepare": cmd_prepare, "snapshot": cmd_snapshot, "run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
