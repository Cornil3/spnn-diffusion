#!/usr/bin/env python3
"""
Train SPNNAutoencoder512Opt on the LAION(Re-LAION-safe) + COCO webdataset.

Distils the SD1.5 VAE into an SPNN whose decode is the EXACT pseudo-inverse of
its encode, so P = decode.encode is idempotent by construction.

Key properties this script preserves / relies on:
  * benchmark COCO shards (MagicBrush, 4,778 imgs) are NEVER placed in train
  * samples are deduplicated by sha256 (8,139 dupes in the raw set)
  * images whose STORED short side < crop are dropped (3,088 placeholders/1x1)
  * never upscales: LAION crops at 512, COCO buckets at 384 (only 7% of COCO
    reaches 512, and upscaling would defeat the whole no-upscale policy)

Checkpoints: only `latest.pt` and `best.pt`, written atomically.
Resume: `--resume auto` picks up latest.pt (model/opt/sched/ema/step/RNG).
Perceptual: `--w-perc W --perc-type lpips` adds LPIPS (VGG) - the loss the SD1.5 VAE itself was trained
with - to the reconstruction branches in --perc-branches; pixel MSE+L1 alone gives blurry decodes.
`--rewarmup-steps N` re-ramps the LR once after such a loss change (the start step is stored in the
checkpoint, so later resumes do not re-warm). `--best-metric lpips` makes best.pt track val cycle LPIPS
(PSNR rewards blur). A finished run writes OUT/DONE, which ends the sbatch resubmit chain.
"""
import argparse, glob, io, json, math, os, random, signal, sys, tarfile, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spnn_model_opt import SPNNAutoencoder512Opt

DATA = "/rg/shocher_prj/ron.libman/laion1m"
SD15 = "sd-legacy/stable-diffusion-v1-5"


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def build_keep_and_splits(root, min_side, val_shards, seed=0, max_repeat=8,
                          test_shards=None):
    """From img2dataset's parquet sidecars: pick ONE sample per sha256 and drop
    anything whose stored short side is below `min_side`. Returns
    keep = {"<dirname>/<shard>/<key>"} plus the shard->split assignment.

    Reading sidecars avoids walking ~100 GB of tars (seconds vs ~30 min)."""
    import pyarrow.parquet as pq
    import collections
    # Pass 1: count sha256 frequency. Images repeated many times are dead-host
    # placeholders served with HTTP 200 (one appears 1,928x, several are 1x1).
    # Deduping keeps one copy; we want them GONE, so count first and drop all.
    freq = collections.Counter()
    laion_dirs0 = sorted(d for d in glob.glob(f"{root}/shards*")
                         if os.path.isdir(d) and not d.endswith("shards_coco"))
    for d in laion_dirs0:
        for f in sorted(glob.glob(f"{d}/*.parquet")):
            try:
                t = pq.read_table(f, columns=["sha256", "status"])
            except Exception:
                continue
            for s, st in zip(t["sha256"].to_pylist(), t["status"].to_pylist()):
                if st == "success" and s:
                    freq[s] += 1
    banned = {s for s, c in freq.items() if c >= max_repeat}

    keep, seen = set(), set()
    laion_dirs = sorted(d for d in glob.glob(f"{root}/shards*")
                        if os.path.isdir(d) and not d.endswith("shards_coco"))
    for d in laion_dirs:
        dn = os.path.basename(d)
        for f in sorted(glob.glob(f"{d}/*.parquet")):
            sn = os.path.basename(f)[:-8]
            try:
                t = pq.read_table(f, columns=["key", "sha256", "status", "width", "height"])
            except Exception:
                continue
            k = t["key"].to_pylist(); sh = t["sha256"].to_pylist()
            st = t["status"].to_pylist()
            w = t["width"].to_pylist(); h = t["height"].to_pylist()
            for i, s in enumerate(st):
                if s != "success" or not sh[i] or not w[i] or not h[i]:
                    continue
                if min(w[i], h[i]) < min_side:
                    continue
                if sh[i] in banned:          # placeholder / junk image
                    continue
                if sh[i] in seen:            # exact duplicate
                    continue
                seen.add(sh[i])
                keep.add(f"{dn}/{sn}/{k[i]}")

    stats_ = {"banned_hashes": len(banned),
              "banned_samples": sum(freq[s] for s in banned),
              "unique_kept": len(keep)}
    laion_tars = sorted(t for d in laion_dirs for t in glob.glob(f"{d}/*.tar"))
    coco_clean = sorted(glob.glob(f"{root}/shards_coco/clean-*.tar"))
    coco_bench = sorted(glob.glob(f"{root}/shards_coco/benchmark-*.tar"))

    if test_shards is None:
        test_shards = val_shards
    rng = random.Random(seed)
    rng.shuffle(laion_tars); rng.shuffle(coco_clean)

    # Every dataset gets a real 3-way split. Test shards are held out entirely:
    # nothing in train/val touches them, so they stay a clean final measurement.
    cv = max(1, round(val_shards * len(coco_clean) / max(len(laion_tars), 1)))
    ct = max(1, round(test_shards * len(coco_clean) / max(len(laion_tars), 1)))
    splits = {
        "test_laion":  laion_tars[:test_shards],
        "val_laion":   laion_tars[test_shards:test_shards + val_shards],
        "train_laion": laion_tars[test_shards + val_shards:],
        "test_coco":   coco_clean[:ct],
        "val_coco":    coco_clean[ct:ct + cv],
        "train_coco":  coco_clean[ct + cv:],
        "test_bench":  coco_bench,          # MagicBrush: quarantined, test only
    }
    # hard guarantees, not conventions
    for s in ("train_laion", "train_coco", "val_laion", "val_coco"):
        assert not any("benchmark-" in os.path.basename(p) for p in splits[s]), \
            f"benchmark shard leaked into {s}"
    for a, b in (("train_laion", "val_laion"), ("train_laion", "test_laion"),
                 ("val_laion", "test_laion"), ("train_coco", "val_coco"),
                 ("train_coco", "test_coco"), ("val_coco", "test_coco")):
        assert not (set(splits[a]) & set(splits[b])), f"{a} overlaps {b}"
    for s in ("train_laion", "val_laion", "test_laion", "train_coco", "val_coco", "test_coco"):
        assert splits[s], f"split {s} is EMPTY - lower --val-shards/--test-shards"
    return keep, splits, stats_


class TarImageDataset(IterableDataset):
    """Minimal webdataset-style reader (avoids a `webdataset` dependency).
    Streams .jpg/.json pairs from tars, shards across DDP ranks and workers."""

    def __init__(self, tars, keep, crop, rank=0, world=1, seed=0,
                 shuffle_buf=1000, use_keep=True, min_std=2.0):
        self.tars, self.keep, self.crop = tars, keep, crop
        self.rank, self.world, self.seed = rank, world, seed
        self.shuffle_buf, self.use_keep = shuffle_buf, use_keep
        self.min_std = min_std
        self.epoch = 0

    def set_epoch(self, e):
        self.epoch = e

    def _my_tars(self):
        wi = get_worker_info()
        nw, wid = (wi.num_workers, wi.id) if wi else (1, 0)
        tars = list(self.tars)
        random.Random(self.seed + self.epoch).shuffle(tars)
        tars = tars[self.rank::self.world]     # split by rank first
        return tars[wid::nw]                   # then by worker

    def _samples(self):
        import cv2
        for tp in self._my_tars():
            dn = os.path.basename(os.path.dirname(tp))
            sn = os.path.basename(tp)[:-4]
            try:
                tf = tarfile.open(tp)
            except Exception:
                continue
            cur, buf = None, {}
            try:
                for m in tf:
                    if not m.isfile():
                        continue
                    key, _, ext = m.name.partition(".")
                    if cur is not None and key != cur:
                        buf = {}
                    cur = key
                    if ext in ("jpg", "jpeg", "png"):
                        buf["img"] = tf.extractfile(m).read()
                    elif ext == "json":
                        buf["meta"] = tf.extractfile(m).read()
                    if "img" in buf and "meta" in buf:
                        if self.use_keep and f"{dn}/{sn}/{key}" not in self.keep:
                            buf = {}; continue
                        a = cv2.imdecode(np.frombuffer(buf["img"], np.uint8), cv2.IMREAD_COLOR)
                        buf = {}
                        if a is None:
                            continue
                        h, w = a.shape[:2]
                        if min(h, w) < self.crop:      # never upscale
                            continue
                        yield a
            except Exception:
                pass
            finally:
                tf.close()

    def __iter__(self):
        import cv2
        rng = random.Random(self.seed + self.epoch * 7919 + (get_worker_info().id if get_worker_info() else 0))
        pool = []
        for a in self._samples():
            pool.append(a)
            if len(pool) >= self.shuffle_buf:
                i = rng.randrange(len(pool))
                pool[i], pool[-1] = pool[-1], pool[i]
                t = self._to_tensor(pool.pop(), rng)
                if t is not None:
                    yield t
        rng.shuffle(pool)
        for a in pool:
            t = self._to_tensor(a, rng)
            if t is not None:
                yield t

    def _to_tensor(self, a, rng):
        """uint8 BGR HWC -> float32 RGB CHW in [-1, 1], EXACTLY.
        255 -> 255/127.5-1 = +1.0 ; 0 -> -1.0. No clamp needed or wanted here:
        the arithmetic cannot leave the range, and a clamp would only hide a bug."""
        import cv2
        c = self.crop
        for _ in range(4):                        # retry a few crops before giving up
            h, w = a.shape[:2]
            y0 = rng.randrange(0, h - c + 1)      # RANDOM crop, not center: a codec
            x0 = rng.randrange(0, w - c + 1)      # has no composition to preserve,
            cr = a[y0:y0 + c, x0:x0 + c]          # and the 576 cap exists for this
            if float(cr.std()) >= self.min_std:   # reject blank/solid CROPS, not
                break                             # just blank whole images
        else:
            return None
        if rng.random() < 0.5:
            cr = cr[:, ::-1]
        cr = cv2.cvtColor(cr, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(np.ascontiguousarray(cr.transpose(2, 0, 1)))




# ----------------------------------------------------------------------------
# perceptual loss (torchvision VGG16; `lpips` package is not installed)
# ----------------------------------------------------------------------------
class VGGPerceptual(nn.Module):
    LAYERS = (3, 8, 15, 22)

    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        v = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:23].eval()
        for p in v.parameters():
            p.requires_grad_(False)
        self.v = v
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, a, b):
        # NO clamp on the prediction. clamp() has zero gradient outside the range,
        # so clamping here would leave an over-shooting decoder with no signal to
        # pull it back - the perceptual term would silently stop doing its job on
        # exactly the pixels that need it most. VGG tolerates mild out-of-range.
        a = ((a + 1) / 2 - self.mean) / self.std
        b = ((b.detach() + 1) / 2 - self.mean) / self.std
        loss, x, y = 0.0, a, b
        for i, layer in enumerate(self.v):
            x, y = layer(x), layer(y)
            if i in self.LAYERS:
                loss = loss + F.l1_loss(x, y)
        return loss


class LPIPSPerceptual(nn.Module):
    """LPIPS (Zhang et al. 2018). net="vgg" is the loss the SD1.5 VAE was trained with (taming's
    LPIPSWithDiscriminator, perceptual weight 1.0); net="alex" is the usual evaluation metric.
    The package is pip-installed with --target on /rg (/home is near its quota)."""

    def __init__(self, net="vgg"):
        super().__init__()
        p = "/rg/shocher_prj/ron.libman/pylibs_eval"
        if p not in sys.path:
            sys.path.insert(0, p)
        import lpips
        self.m = lpips.LPIPS(net=net, verbose=False).eval().requires_grad_(False)

    def forward(self, a, b):
        # fp32: LPIPS compares unit-normalised features, bf16 would swamp small differences.
        # No clamp on the prediction, for the reason given in VGGPerceptual.
        with torch.autocast("cuda", enabled=False):
            return self.m(a.float(), b.detach().float()).mean()


# ----------------------------------------------------------------------------
# ema / checkpoint
# ----------------------------------------------------------------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)

    def copy_to(self, model):
        sd = model.state_dict()
        bak = {k: sd[k].detach().clone() for k in self.shadow}
        model.load_state_dict({**sd, **{k: v.to(sd[k].dtype) for k, v in self.shadow.items()}}, strict=False)
        return bak

    @staticmethod
    def restore(model, bak):
        sd = model.state_dict()
        model.load_state_dict({**sd, **bak}, strict=False)


class SPNNTrainWrapper(nn.Module):
    """Every tensor op goes through forward() so DDP's autograd hooks fire.
    Calling net.encode()/net.decode() directly on a DDP-wrapped module silently
    skips gradient synchronisation across ranks - the model trains, but each
    rank diverges. This is why the churches trainer had a wrapper too."""

    def __init__(self, net, need_enc, need_enc_traj, need_traj, need_self):
        super().__init__()
        self.net = net
        self.need_enc, self.need_enc_traj = need_enc, need_enc_traj
        self.need_traj, self.need_self = need_traj, need_self

    def forward(self, x, z_clean, z_traj, x_traj):
        dec_c = self.net.decode(z_clean)
        dec_t = self.net.decode(z_traj) if (self.need_traj and z_traj is not None) else None
        enc_c = self.net.encode(x) if (self.need_enc or self.need_self) else None
        enc_t = self.net.encode(x_traj) if (self.need_enc_traj and x_traj is not None) else None
        dec_s = self.net.decode(enc_c) if self.need_self else None
        return dec_c, dec_t, enc_c, enc_t, dec_s


def save_ckpt(path, **payload):
    """Atomic: write to .tmp then rename, so a kill mid-write cannot corrupt
    the only good checkpoint."""
    tmp = f"{path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DATA)
    p.add_argument("--out", default="/rg/shocher_prj/ron.libman/laion1m/runs/spnn512")
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--coco-crop", type=int, default=384,
                   help="only 7%% of COCO reaches 512; bucket it lower instead of upscaling")
    p.add_argument("--coco-every", type=int, default=6,
                   help="every Nth optimiser step uses a COCO batch (~17%% of data)")
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--accum", type=int, default=8, help="effective batch = bs*accum")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--val-shards", type=int, default=4)
    p.add_argument("--test-shards", type=int, default=4,
                   help="held out entirely; never touched by train or val")
    # lr: linear-scaled from the churches recipe (1e-4 @ effective batch 64)
    p.add_argument("--base-lr", type=float, default=1e-4)
    p.add_argument("--base-bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=None, help="override the scaled lr")
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--min-lr-frac", type=float, default=0.05)
    p.add_argument("--lr-schedule", default="wsd", choices=["wsd", "cosine", "constant"],
                   help="wsd = warmup, hold at peak, then decay over the last "
                        "--decay-frac. Lets you stop at any epoch and anneal from there.")
    p.add_argument("--decay-frac", type=float, default=0.2,
                   help="fraction of max_steps spent decaying at the end")
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--epochs", type=float, default=None,
                   help="passes over ALL train images (laion+coco). Derives --max-steps "
                        "and --coco-every from the real counts so both datasets finish a "
                        "pass together. Overrides --max-steps / --coco-every.")
    p.add_argument("--max-steps", type=int, default=60000)
    p.add_argument("--max-hours", type=float, default=11.5, help="stop before the QOS wall")
    # loss weights
    p.add_argument("--w-clean", type=float, default=1.0)
    p.add_argument("--w-traj", type=float, default=1.0)
    p.add_argument("--w-enc", type=float, default=1.0)
    p.add_argument("--w-enc-traj", type=float, default=1.0)
    p.add_argument("--w-self", type=float, default=0.25,
                   help="decode(encode(x)) vs x; keep BELOW w_enc or the latent drifts off SD1.5")
    p.add_argument("--w-perc", type=float, default=0.0, help="stage in after recon converges")
    p.add_argument("--perc-type", default="vgg", choices=["vgg", "lpips"],
                   help="vgg = raw VGG16 feature L1; lpips = LPIPS(VGG), the SD1.5 VAE's own loss")
    p.add_argument("--perc-branches", default="clean",
                   help="comma list of reconstruction branches with the perceptual term: clean,traj,self "
                        "(each scaled by its own branch weight)")
    p.add_argument("--rewarmup-steps", type=int, default=0,
                   help="one-off linear LR re-warmup after a loss change, from the first step run with it")
    p.add_argument("--best-metric", default="psnr", choices=["psnr", "lpips"],
                   help="what best.pt tracks on val (cycle): psnr rewards blur, lpips does not")
    p.add_argument("--p-qsample", type=float, default=0.5,
                   help="fraction of the batch given forward-diffused latents (no LDM bank needed)")
    p.add_argument("--decoder-only", action="store_true", help="train only the r networks")
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every-min", type=float, default=20.0)
    p.add_argument("--resume", default="auto", help="auto | none | /path/to.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compile", action="store_true")
    # efficiency
    p.add_argument("--channels-last", action="store_true", default=True)
    p.add_argument("--vae-dtype", default="bf16", choices=["bf16", "fp16", "fp32"],
                   help="frozen reference VAE runs in low precision; it is only a target generator")
    p.add_argument("--max-repeat", type=int, default=8,
                   help="drop EVERY copy of an image whose sha256 occurs >= this many times")
    p.add_argument("--min-crop-std", type=float, default=2.0)
    # latent scaling
    p.add_argument("--latent-loss-norm", default="perchannel",
                   choices=["perchannel", "none"],
                   help="balance the latent loss per channel; the OUTPUT convention stays z*0.18215")
    p.add_argument("--latent-stats-batches", type=int, default=64)
    # wandb
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="spnn512")
    p.add_argument("--wandb-run", default=None)
    p.add_argument("--wandb-images", type=int, default=8)
    args = p.parse_args()

    # ---- DDP (launch with: torchrun --nproc_per_node=N train_spnn512.py ...) ----
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    ddp = world > 1
    if ddp:
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    is_main = rank == 0

    if is_main:
        os.makedirs(args.out, exist_ok=True)
    dev = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    eff_bs = args.bs * args.accum * world          # DDP multiplies the effective batch
    lr = args.lr if args.lr is not None else args.base_lr * eff_bs / args.base_bs

    def log(msg):
        if is_main:
            print(msg, flush=True)

    log(f"[cfg] dev={dev} world={world} crop={args.crop} bs={args.bs} accum={args.accum} "
        f"eff_bs={eff_bs} lr={lr:.2e} (scaled from {args.base_lr:.0e}@{args.base_bs})")

    # ---- data ----
    log("[data] building keep-list from parquet sidecars ...")
    keep, splits, dstats = build_keep_and_splits(args.data, args.crop, args.val_shards,
                                                 args.seed, args.max_repeat, args.test_shards)
    log(f"[data] keep={len(keep):,} unique samples (deduped by sha256, short side >= {args.crop})")
    log(f"[data] dropped {dstats['banned_hashes']:,} placeholder hashes "
        f"= {dstats['banned_samples']:,} samples (seen >= {args.max_repeat}x)")
    for k, v in splits.items():
        log(f"[data]   {k:12s} {len(v):4d} shards")
    log("[data] benchmark quarantine: asserted (MagicBrush shards are test-only)")

    # ---- epoch bookkeeping ----
    # "1 epoch" = one pass over every train image, laion AND coco. To make that
    # true, coco's share of optimiser steps must equal its share of the images -
    # otherwise the two datasets drift apart (the old 5:1 step ratio against a
    # 6:1 size ratio is why epochs came out fractional).
    train_pref = {f"{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)[:-4]}"
                  for p in splits["train_laion"]}
    laion_n = sum(1 for k in keep if k.rsplit("/", 1)[0] in train_pref)
    try:
        cc = json.loads(open(f"{args.data}/shards_coco/counts.json").read())["clean"]
    except Exception:
        cc = 118509
    n_clean_shards = len(splits["train_coco"]) + len(splits["val_coco"]) + len(splits["test_coco"])
    coco_n = int(round(cc * len(splits["train_coco"]) / max(n_clean_shards, 1)))
    total_n = laion_n + coco_n
    steps_per_epoch = max(1, int(round(total_n / eff_bs)))
    if args.epochs is not None:
        args.coco_every = max(2, int(round(total_n / max(coco_n, 1))))
        args.max_steps = int(round(args.epochs * steps_per_epoch))
    log(f"[epochs] train images: laion {laion_n:,} + coco {coco_n:,} = {total_n:,}")
    log(f"[epochs] steps/epoch {steps_per_epoch:,} @ eff_bs {eff_bs} | "
        f"coco_every {args.coco_every} (coco = {100/args.coco_every:.1f}% of steps, "
        f"{100*coco_n/total_n:.1f}% of images)")
    log(f"[epochs] max_steps {args.max_steps:,} = {args.max_steps/steps_per_epoch:.2f} epochs")

    def loader(tars, crop, bs, use_keep=True, shuffle_buf=1000):
        ds = TarImageDataset(tars, keep, crop, rank=rank, world=world, seed=args.seed,
                             shuffle_buf=shuffle_buf, use_keep=use_keep,
                             min_std=args.min_crop_std)
        return ds, DataLoader(ds, batch_size=bs, num_workers=args.workers,
                              pin_memory=True, drop_last=True,
                              persistent_workers=args.workers > 0,
                              prefetch_factor=4 if args.workers else None)

    ds_l, dl_l = loader(splits["train_laion"], args.crop, args.bs)
    ds_c, dl_c = loader(splits["train_coco"], args.coco_crop, args.bs, use_keep=False)
    ds_vl, dl_vl = loader(splits["val_laion"], args.crop, args.bs, shuffle_buf=1)
    ds_vc, dl_vc = loader(splits["val_coco"], args.coco_crop, args.bs,
                          use_keep=False, shuffle_buf=1)
    log(f"[data] val on BOTH: laion@{args.crop} and coco@{args.coco_crop}; "
        f"test held out ({len(splits['test_laion'])} laion + {len(splits['test_coco'])} coco "
        f"+ {len(splits['test_bench'])} benchmark shards)")

    # ---- reference VAE (frozen) ----
    from diffusers import AutoencoderKL, DDPMScheduler
    vdt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.vae_dtype]
    vae = AutoencoderKL.from_pretrained(SD15, subfolder="vae", torch_dtype=vdt)
    vae = vae.to(dev).eval().requires_grad_(False)
    # SD1.5 CONVENTION: the UNet consumes z * scaling_factor (0.18215). Our encoder
    # must emit exactly that or it is not a drop-in for IP2P. Note the churches
    # trainer used per-channel (z-mu)/sigma instead - good for loss balance, but
    # NOT interchangeable with SD's convention, so we keep them separate:
    #   output convention -> z * sf          (fixed, non-negotiable)
    #   loss weighting    -> per-channel     (see latent_w below)
    sf = float(getattr(vae.config, "scaling_factor", 0.18215))
    sched = DDPMScheduler.from_pretrained(SD15, subfolder="scheduler")
    ab = sched.alphas_cumprod.to(dev).float()
    log(f"[vae] SD1.5 AutoencoderKL | dtype={args.vae_dtype} | scaling_factor={sf}")

    # ---- model ----
    net = SPNNAutoencoder512Opt().to(dev)
    if args.channels_last:
        net = net.to(memory_format=torch.channels_last)
    if args.decoder_only:
        for n_, prm in net.named_parameters():
            prm.requires_grad_(".r." in n_)
    core = net                                  # unwrapped, for state_dict/EMA
    log(f"[model] SPNNAutoencoder512Opt {sum(q.numel() for q in net.parameters())/1e6:.1f}M params | "
        f"trainable {sum(q.numel() for q in net.parameters() if q.requires_grad)/1e6:.1f}M"
        f"{' (decoder/r only)' if args.decoder_only else ''}")

    need_enc = args.w_enc > 0
    need_enc_traj = args.w_enc_traj > 0
    need_traj = args.w_traj > 0
    need_self = args.w_self > 0
    wrapper = SPNNTrainWrapper(net, need_enc, need_enc_traj, need_traj, need_self)
    if args.compile:
        wrapper = torch.compile(wrapper)
    if ddp:
        wrapper = torch.nn.parallel.DistributedDataParallel(
            wrapper, device_ids=[local_rank], output_device=local_rank,
            # only needed when a loss weight is 0 and its branch is skipped;
            # otherwise it costs an extra autograd traversal every iteration
            find_unused_parameters=not all([need_enc, need_enc_traj, need_traj, need_self]),
            gradient_as_bucket_view=True)
    train_params = [q for q in core.parameters() if q.requires_grad]

    perc = None
    if args.w_perc > 0:
        perc = (LPIPSPerceptual("vgg") if args.perc_type == "lpips" else VGGPerceptual()).to(dev)
    perc_br = {b for b in args.perc_branches.split(",") if b}
    try:
        val_lp = LPIPSPerceptual("alex").to(dev)       # val metric; same net as eval_restore_fid.py
    except Exception as e:
        val_lp = None
        log(f"[val] LPIPS unavailable ({type(e).__name__}: {e}); best.pt falls back to PSNR")
    log(f"[loss] perceptual: {'off' if perc is None else f'{args.perc_type} x{args.w_perc} on {sorted(perc_br)}'} "
        f"| best.pt tracks val cycle {args.best_metric}")
    try:
        opt = torch.optim.AdamW(train_params, lr=lr, weight_decay=args.wd,
                                betas=(0.9, 0.99), fused=True)
    except (TypeError, RuntimeError):
        opt = torch.optim.AdamW(train_params, lr=lr, weight_decay=args.wd, betas=(0.9, 0.99))
    ema = EMA(core, args.ema)

    decay_steps = max(1, int(round(args.decay_frac * args.max_steps)))
    decay_start = max(args.warmup, args.max_steps - decay_steps)

    def lr_sched(step):
        if step < args.warmup:
            return lr * (step + 1) / args.warmup
        if args.lr_schedule == "constant":
            return lr
        if args.lr_schedule == "wsd":
            if step < decay_start:              # stable: stop anywhere, no penalty
                return lr
            t = min(1.0, (step - decay_start) / decay_steps)
        else:                                   # plain cosine over the whole run
            t = min(1.0, (step - args.warmup) / max(1, args.max_steps - args.warmup))
        return lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * t)))

    def lr_at(step):
        # one-off linear re-warmup after a loss change; its start step lives in the checkpoint
        rf = state.get("rewarm_from")
        f = 1.0
        if args.rewarmup_steps > 0 and rf is not None and step < rf + args.rewarmup_steps:
            f = (step - rf + 1) / args.rewarmup_steps
        return lr_sched(step) * f

    # ---- resume ----
    # no "best_metric" here: a checkpoint from before the key existed must read as "psnr" so the reset below fires
    state = {"step": 0, "epoch": 0, "best": float("inf")}
    ck = None
    if args.resume == "auto":
        cand = os.path.join(args.out, "latest.pt")
        ck = cand if os.path.exists(cand) else None
    elif args.resume not in ("none", ""):
        ck = args.resume
    if ck:
        d = torch.load(ck, map_location=dev, weights_only=False)
        core.load_state_dict(d["model"])
        opt.load_state_dict(d["opt"])
        ema.shadow = {k: v.to(dev) for k, v in d["ema"].items()}
        state.update(d["state"])
        try:
            torch.set_rng_state(d["rng_cpu"]); torch.cuda.set_rng_state_all(d["rng_cuda"])
            random.setstate(d["rng_py"]); np.random.set_state(d["rng_np"])
        except Exception:
            log("[resume] RNG restore skipped")
        log(f"[resume] {ck} @ step {state['step']} epoch {state['epoch']} best {state['best']:.4f}")
    else:
        log("[resume] fresh start")
    if state.get("best_metric", "psnr") != args.best_metric:
        log(f"[resume] best.pt metric {state.get('best_metric', 'psnr')} -> {args.best_metric}: best score reset")
        state["best"], state["best_metric"] = float("inf"), args.best_metric
    if args.rewarmup_steps > 0 and "rewarm_from" not in state:
        state["rewarm_from"] = state["step"]
        log(f"[lr] one-off re-warmup over {args.rewarmup_steps} steps from step {state['step']}")
    # continue the data order where the last job stopped (otherwise every resume replays epoch 0's order)
    ds_l.set_epoch(state.get("epoch", 0)); ds_c.set_epoch(state.get("coco_epoch", 0))

    # ---- graceful stop on preemption ----
    stop = {"flag": False}
    def _sig(signum, frame):
        stop["flag"] = True
        log(f"[signal] {signum} received - will checkpoint and exit")
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    logf = open(os.path.join(args.out, "log.jsonl"), "a") if is_main else None

    @torch.no_grad()
    def encode_ref(x):
        return (vae.encode(x.to(vdt)).latent_dist.mode().float() * sf)

    @torch.no_grad()
    def decode_ref(z):
        return vae.decode((z / sf).to(vdt)).sample.float()

    # ---- per-channel latent std, for LOSS BALANCE only ----
    # SD1.5's 4 latent channels have unequal variance; an unweighted MSE in
    # z*0.18215 space lets the widest channel dominate the encoder gradient.
    # We rescale the RESIDUAL, never the model's output.
    latent_w = torch.ones(1, 4, 1, 1, device=dev)
    if args.latent_loss_norm == "perchannel":
        cache = os.path.join(args.out, "latent_stats.json")
        if os.path.exists(cache):
            d = json.loads(open(cache).read())
            latent_w = 1.0 / torch.tensor(d["sigma"], device=dev).view(1, -1, 1, 1)
            log(f"[latent] loaded per-channel sigma {['%.3f' % s for s in d['sigma']]}")
        else:
            acc_, n_ = [], 0
            for x in dl_l:            # TRAIN loader: val must stay untouched
                acc_.append(encode_ref(x.to(dev)).float())
                n_ += 1
                if n_ >= args.latent_stats_batches:
                    break
            if acc_:
                z_ = torch.cat(acc_, 0)
                sigma = z_.std(dim=(0, 2, 3)).clamp_min(1e-3)
                mu = z_.mean(dim=(0, 2, 3))
                # every rank saw a DIFFERENT shard slice, so each would derive its
                # own sigma and scale the encoder loss differently - i.e. optimise
                # a slightly different objective. rank 0's value wins.
                if ddp:
                    torch.distributed.broadcast(sigma, src=0)
                    torch.distributed.broadcast(mu, src=0)
                latent_w = (1.0 / sigma).view(1, -1, 1, 1)
                if is_main:
                    json.dump({"sigma": sigma.tolist(), "mu": mu.tolist(),
                               "scaling_factor": sf}, open(cache, "w"))
                log(f"[latent] per-channel sigma {['%.3f' % s for s in sigma.tolist()]} "
                    f"mu {['%.3f' % m for m in mu.tolist()]}")

    def lat_l(a, b):
        d = (a - b) * latent_w
        return d.pow(2).mean() + d.abs().mean()

    # ---- wandb ----
    wb = None
    if args.wandb and is_main:
        try:
            import wandb as _wb
            wb = _wb
            wb.init(project=args.wandb_project, name=args.wandb_run,
                    config={**vars(args), "world_size": world, "eff_bs": eff_bs,
                            "lr_scaled": lr,
                            "params_M": sum(q.numel() for q in core.parameters()) / 1e6,
                            "trainable_M": sum(q.numel() for q in train_params) / 1e6,
                            "keep_samples": len(keep),
                            "banned_hashes": dstats["banned_hashes"],
                            "banned_samples": dstats["banned_samples"],
                            "train_laion_shards": len(splits["train_laion"]),
                            "train_coco_shards": len(splits["train_coco"]),
                            "vae_scaling_factor": sf, "gpu": torch.cuda.get_device_name(0)},
                    resume="allow", id=args.wandb_run)
            wb.watch(core, log="gradients", log_freq=1000)
        except Exception as e:
            log(f"[wandb] disabled ({type(e).__name__}: {e}); pip install wandb")
            wb = None

    def losses(x):
        """Returns (scalar loss, dict of detached parts, dec_c)."""
        with torch.no_grad():
            z_clean = encode_ref(x)
            n = int(x.shape[0] * args.p_qsample)
            if n > 0 and (need_traj or need_enc_traj):
                k = torch.randint(0, ab.shape[0], (n,), device=dev)
                a_ = ab[k].view(-1, 1, 1, 1)
                z_traj = a_.sqrt() * z_clean[:n] + (1 - a_).sqrt() * torch.randn_like(z_clean[:n])
                x_traj = decode_ref(z_traj)
            else:
                z_traj = x_traj = None

        dec_c, dec_t, enc_c, enc_t, dec_s = wrapper(x, z_clean, z_traj, x_traj)

        parts = {}
        l = F.mse_loss(dec_c, x) + F.l1_loss(dec_c, x)
        out = args.w_clean * l; parts["clean"] = l.detach()
        if dec_t is not None:
            l = F.mse_loss(dec_t, x_traj) + F.l1_loss(dec_t, x_traj)
            out = out + args.w_traj * l; parts["traj"] = l.detach()
        if enc_c is not None and need_enc:
            l = lat_l(enc_c, z_clean)
            out = out + args.w_enc * l; parts["enc"] = l.detach()
        if enc_t is not None:
            l = lat_l(enc_t, z_traj)
            out = out + args.w_enc_traj * l; parts["enc_traj"] = l.detach()
        if dec_s is not None:
            l = F.mse_loss(dec_s, x) + F.l1_loss(dec_s, x)
            out = out + args.w_self * l; parts["self"] = l.detach()
        if perc is not None:
            for br, dec, tgt, w in (("clean", dec_c, x, args.w_clean), ("traj", dec_t, x_traj, args.w_traj),
                                    ("self", dec_s, x, args.w_self)):
                if br in perc_br and dec is not None:
                    l = perc(dec, tgt)
                    out = out + w * args.w_perc * l; parts[f"perc_{br}"] = l.detach()
        return out, parts, dec_c

    @torch.no_grad()
    def validate(dl, tag):
        """Reports teacher-forced PSNR, the TRUE cycle PSNR, and the SD1.5 VAE
        reference. The churches log only ever showed the first."""
        bak = ema.copy_to(core)
        wrapper.eval()
        tf_ps = cy_ps = ref_ps = idem = lp_cy = lp_rf = 0.0
        oor = 0.0
        n = 0
        sample = None
        for i, x in enumerate(dl):
            if i >= 25:
                break
            x = x.to(dev, non_blocking=True)
            if args.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = encode_ref(x)
                d_tf = core.decode(z).float()
                e = core.encode(x)
                d_cy = core.decode(e).float()
                d_rf = decode_ref(z).float()
                e2 = core.encode(d_cy)          # idempotency check on real data
            psnr = lambda a, b: (10 * torch.log10(4.0 / F.mse_loss(a.clamp(-1, 1), b).clamp_min(1e-10))).item()
            tf_ps += psnr(d_tf, x); cy_ps += psnr(d_cy, x); ref_ps += psnr(d_rf, x)
            idem += ((e2 - e).norm() / e.norm().clamp_min(1e-8)).item()
            if val_lp is not None:                  # the perceptual number that PSNR hides
                lp_cy += val_lp(d_cy.clamp(-1, 1), x).item()
                lp_rf += val_lp(d_rf.clamp(-1, 1), x).item()
            oor = max(oor, float(d_cy.abs().max()))   # raw range BEFORE any clamp
            if sample is None:
                sample = (x[:args.wandb_images].float().cpu(),
                          d_cy[:args.wandb_images].float().cpu())
            n += 1
        wrapper.train()
        EMA.restore(core, bak)
        n = max(n, 1)
        lp = (lp_cy / n, lp_rf / n) if val_lp is not None else (None, None)
        return tf_ps / n, cy_ps / n, ref_ps / n, idem / n, oor, sample, lp

    # ---- train ----
    t0 = time.time(); last_save = time.time()
    steps_at_t0 = state["step"]          # img/s must measure THIS job, not all time
    nonfinite = [0]
    it_l, it_c = iter(dl_l), iter(dl_c)
    wrapper.train()
    log(f"[train] start at step {state['step']} | max_steps {args.max_steps} "
        f"| wall limit {args.max_hours}h")

    while state["step"] < args.max_steps:
        use_coco = args.coco_every > 0 and (state["step"] % args.coco_every == 0)
        opt.zero_grad(set_to_none=True)
        agg = {}
        for mb in range(args.accum):
            try:
                x = next(it_c if use_coco else it_l)
            except StopIteration:
                try:
                    if use_coco:
                        state["coco_epoch"] = state.get("coco_epoch", 0) + 1
                        ds_c.set_epoch(ds_c.epoch + 1); it_c = iter(dl_c); x = next(it_c)
                    else:
                        state["epoch"] += 1
                        ds_l.set_epoch(ds_l.epoch + 1); it_l = iter(dl_l); x = next(it_l)
                except StopIteration:
                    # a loader that is empty even after reset would otherwise kill
                    # the job hours in; fall back to the other stream instead
                    other = it_l if use_coco else it_c
                    try:
                        x = next(other)
                    except StopIteration:
                        raise RuntimeError(
                            "both train loaders are empty - check --crop vs the "
                            "keep-list and the shard split sizes")
            x = x.to(dev, non_blocking=True)
            if args.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, parts, _ = losses(x)
            if not torch.isfinite(loss):
                # skip ONLY this micro-batch. zeroing here would throw away every
                # micro-batch already accumulated in this optimiser step.
                nonfinite[0] += 1
                continue
            # DDP all-reduces on every backward() by default; with accum=8 that is
            # 8x the communication for one optimiser step. Sync only on the last.
            if ddp and mb < args.accum - 1:
                with wrapper.no_sync():
                    (loss / args.accum).backward()
            else:
                (loss / args.accum).backward()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v.item() / args.accum
            agg["total"] = agg.get("total", 0.0) + loss.item() / args.accum

        cur_lr = lr_at(state["step"])
        for g in opt.param_groups:
            g["lr"] = cur_lr
        gn = torch.nn.utils.clip_grad_norm_(train_params, args.clip)
        opt.step()
        ema.update(core)
        state["step"] += 1

        if state["step"] % args.log_every == 0:
            el = time.time() - t0
            ips = (state["step"] - steps_at_t0) * eff_bs / max(el, 1e-9)
            mem = torch.cuda.max_memory_allocated() / 1e9 if dev.startswith("cuda") else 0
            if dev.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            rec = {"step": state["step"], "epoch": round(state["step"] / steps_per_epoch, 3),
                   "lr": cur_lr, "grad_norm": float(gn),
                   "img_s": round(ips, 2), "elapsed_h": round(el / 3600, 3),
                   "peak_gb": round(mem, 2), "batch": "coco" if use_coco else "laion",
                   **{k: round(v, 5) for k, v in agg.items()}}
            log("[train] " + " ".join(f"{k} {v}" for k, v in rec.items()))
            if logf:
                logf.write(json.dumps(rec) + "\n"); logf.flush()
            if wb:
                wb.log({f"train/{k}": v for k, v in agg.items()} |
                       {"train/lr": cur_lr, "train/grad_norm": float(gn),
                        "train/epoch": state["step"] / steps_per_epoch,
                        "train/nonfinite": nonfinite[0],
                        "perf/img_per_s": ips, "perf/peak_gb": mem,
                        "perf/elapsed_h": el / 3600,
                        "perf/hours_per_epoch": (steps_per_epoch * eff_bs) / max(ips, 1e-9) / 3600},
                       step=state["step"])

        if state["step"] % args.val_every == 0:
            res, lps, wbd, rec = {}, {}, {}, {"step": state["step"], "nonfinite": nonfinite[0]}
            for tag, dl_ in (("laion", dl_vl), ("coco", dl_vc)):
                tf_ps, cy_ps, ref_ps, idem, oor, sample, (lp_cy, lp_rf) = validate(dl_, tag)
                gap = tf_ps - cy_ps
                res[tag], lps[tag] = cy_ps, lp_cy
                lp_txt = f" | LPIPS cycle {lp_cy:.4f} (sd15 ref {lp_rf:.4f})" if lp_cy is not None else ""
                log(f"[val:{tag}] step {state['step']} | teacher-forced {tf_ps:.2f} dB | "
                    f"CYCLE {cy_ps:.2f} dB | gap {gap:.2f} | sd15-vae ref {ref_ps:.2f} dB "
                    f"| idem {idem:.2e} | raw|max| {oor:.3f}{lp_txt}")
                rec.update({f"{tag}_psnr_tf": tf_ps, f"{tag}_psnr_cycle": cy_ps,
                            f"{tag}_psnr_gap": gap, f"{tag}_psnr_ref": ref_ps,
                            f"{tag}_idem_rel": idem, f"{tag}_raw_absmax": oor,
                            f"{tag}_lpips_cycle": lp_cy, f"{tag}_lpips_ref": lp_rf})
                wbd.update({f"val_{tag}/psnr_teacher_forced": tf_ps,
                            f"val_{tag}/psnr_cycle": cy_ps,
                            f"val_{tag}/psnr_gap": gap,
                            f"val_{tag}/psnr_sd15_ref": ref_ps,
                            f"val_{tag}/psnr_vs_ref": cy_ps - ref_ps,
                            f"val_{tag}/idempotency_rel": idem,
                            f"val_{tag}/raw_absmax": oor})
                if lp_cy is not None:
                    wbd.update({f"val_{tag}/lpips_cycle": lp_cy, f"val_{tag}/lpips_sd15_ref": lp_rf})
                if wb and sample is not None:
                    x_, r_ = sample
                    # explicit uint8: wandb's float handling is version-dependent,
                    # and clamping HERE is correct - this is display, not a loss
                    to_u8 = lambda t: (((t.clamp(-1, 1) + 1) / 2 * 255)
                                       .round().to(torch.uint8)
                                       .permute(0, 2, 3, 1).numpy())
                    wbd[f"val_{tag}/recon"] = [wb.Image(im) for im in to_u8(r_)]
                    wbd[f"val_{tag}/input"] = [wb.Image(im) for im in to_u8(x_)]
            if logf:
                logf.write(json.dumps(rec) + "\n"); logf.flush()
            if wb:
                wb.log(wbd, step=state["step"])
            # weight the selection metric by dataset share (~83% laion / 17% coco)
            cy_ps = 0.83 * res["laion"] + 0.17 * res["coco"]
            if args.best_metric == "lpips" and None not in lps.values():
                cy_lp = 0.83 * lps["laion"] + 0.17 * lps["coco"]
                score, desc = cy_lp, f"cycle LPIPS {cy_lp:.4f}, {cy_ps:.2f} dB"
            else:                                   # the CYCLE psnr (teacher-forced hides encoder error)
                cy_lp = None
                score, desc = -cy_ps, f"cycle {cy_ps:.2f} dB"
            if score < state["best"] and is_main:
                state["best"] = score
                save_ckpt(os.path.join(args.out, "best.pt"),
                          model=core.state_dict(),
                          ema=ema.shadow, state=dict(state), args=vars(args),
                          psnr_cycle=cy_ps, lpips_cycle=cy_lp, per_dataset=dict(res),
                          per_dataset_lpips=dict(lps))
                log(f"[ckpt] new best ({desc}) -> best.pt")

        # rank 0 decides, then broadcasts: a per-rank time check can desync DDP
        flags = torch.tensor(
            [1.0 if (time.time() - last_save) / 60 >= args.save_every_min else 0.0,
             1.0 if (time.time() - t0) / 3600 >= args.max_hours else 0.0,
             1.0 if stop["flag"] else 0.0], device=dev)
        if ddp:
            torch.distributed.broadcast(flags, src=0)
        due, over, halt = (bool(f) for f in flags.tolist())
        if (due or over or halt or state["step"] >= args.max_steps) and is_main:
            save_ckpt(os.path.join(args.out, "latest.pt"),
                      model=core.state_dict(),
                      opt=opt.state_dict(), ema=ema.shadow, state=dict(state),
                      args=vars(args), rng_cpu=torch.get_rng_state(),
                      rng_cuda=torch.cuda.get_rng_state_all(),
                      rng_py=random.getstate(), rng_np=np.random.get_state())
            last_save = time.time()
            log(f"[ckpt] latest.pt @ step {state['step']}")
        if over or halt:
            log(f"[stop] {'wall limit ' + str(args.max_hours) + 'h' if over else 'signal'} "
                f"- resume with --resume auto")
            break

    if logf:
        logf.close()
    if wb:
        wb.finish()
    if is_main and state["step"] >= args.max_steps:
        with open(os.path.join(args.out, "DONE"), "w") as f:  # ends the sbatch resubmit chain
            f.write(f"step {state['step']}\n")
    log(f"[done] step {state['step']} | {(time.time()-t0)/3600:.2f}h | "
        f"best val cycle {state.get('best_metric', 'psnr')} score {state['best']:.4f}")
    if ddp:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
