# Reproducing REED-VAE Table 1 with SPNN-512 as the consistent VAE

Goal: run REED-VAE's iterative-editing benchmark (Table 1) with our ImageNet-trained
SPNN-512 codec substituted for the VAE, to see whether SPNN buys the same resistance to
iterative encode/decode degradation that REED's re-encode-decode training does.

## What we can and cannot reproduce

**REED itself cannot be re-run.** `github.com/galmog/REED-VAE` exists but holds only a
README and `docs/teaser.png` (last pushed 2025-04-17). No code, no weights, and nothing
on HF under that name. So the published REED numbers are transcribed into our table as a
clearly-marked reference block, never as a measured arm.

**The dataset is `ImagenHub/Mask_Guided_Image_Editing`, split `filtered` — 179 rows,
exactly the paper's "179 images".** But two columns the protocol needs are *not* on the
hub, and this took some digging:

| | on the hub | in `dataset_info.json` | where we get it |
|---|---|---|---|
| `reverse_instruction` | ✗ | ✓ | **`imagenhub_edit_instructions.csv`** |
| `processed_mask` | ✗ | ✓ | reconstructed from `mask_img` (below) |

`dataset_info.json` lists 11 features; the live `filtered` split returns 9. That file,
together with `state.json` (a `save_to_disk` state, `_split: "filtered"`), describes a
locally *augmented* copy rather than the hub's. `reverse_instruction` is REED's own
manual addition — the paper says "we manually add 'reverse prompts'... We will make our
full, revised dataset available with our code", and that release never happened.

So **the CSV is not redundant, it is load-bearing**: it is the only source of the reverse
prompts, and without them IP2P and MagicBrush would repeat the forward edit every
iteration, which is a different experiment. Verified: the CSV covers exactly the same 179
`(img_id, turn_index)` keys as the hub split, with a non-empty reverse prompt on every
row. `load_samples()` raises rather than proceeding if any reverse prompt is missing.

If you still have the augmented `save_to_disk` directory that `state.json` came from,
pass `--from_disk <path>` and it will be used directly, extra columns and all.

**Base model differs from the paper, deliberately.** REED fine-tuned SD **2.1**'s VAE
decoder and used that as its base. Our SPNN-512 was distilled against **SD 1.5**'s
KL-VAE, and all five editing models (IP2P, MagicBrush, DiffEdit, PbE, SD Inpainting) are
SD1.5-family anyway. So every arm here is SD1.5-family and the vanilla baseline is SD
1.5's VAE. Consequence: our absolute numbers will not match Table 1. The vanilla → +SPNN
*trend* is the comparable quantity.

## The SPNN checkpoint

`imagenet_latent_ddnm/runs/spnn512_sd15_distill/ckpt_last.pt` on athena (1.0 GB), epoch
27, 139 020 updates. Its stored config confirms the settings `codecs.py` defaults to:

    mix_type: householder    hidden: 128    r_hidden: 256    scale_bound: 1.0
    sd_model_id: stable-diffusion-v1-5/stable-diffusion-v1-5
    vae_scaling: 0.18215     ema_decay: 0.9999
    w_latent: 1.0  w_rec: 1.0  w_cross: 0.5
    data_root: .../imagenet512

Both `model` and `ema` hold 506 tensors under a `spnn.` prefix. We load **`ema`**
(`ema_decay: 0.9999`), matching `spnn_use_ema: true` in `sd15_inpaint_latent.yml`;
`--spnn_weights model` switches.

**Latent scale: settled, measured, not assumed.** Two separate questions here, and
conflating them causes confusion:

1. *Which regime?* Binary, and decisive. Fitting alpha in `spnn.encode(x) ~= alpha *
   vae_raw` gives **alpha = 0.1764** — nowhere near the 1.0 a raw-regime encoder would
   give. So SPNN emits latents that already carry SD's 0.18215, and the shim divides it
   back out so the pipeline's own multiply restores it. Choosing `raw` instead would
   have handed the UNet latents **5.5x too small** and silently ruined every number.
   This is the risk `verify_codec.py` exists to close, and it is closed.

2. *How closely does SPNN match the VAE latent?* alpha/0.18215 = 0.968 and R^2 = 0.965,
   i.e. SPNN's latents run ~3% smaller with ~3.5% of their variance unexplained by any
   scalar multiple of the VAE latent. That is **distillation error**, a property of the
   checkpoint (epoch 27 of a planned 100) — not a misconfiguration. Do **not** "fix" it
   by rescaling: a 1.03x fudge would mean evaluating a codec that is not SPNN. Carry it
   as a caveat instead (a mildly weaker latent may slightly attenuate edit strength).

The empirical check is stronger than either number: badly misaligned latents would make
SD's UNet produce incoherent edits, and it does not — SPNN holds a coherent scene across
all 25 iterations and reconstructs at 32.5 dB vs the VAE's 30.4 dB.

## Protocol, as the paper defines it

From Appendix A ("Metric calculations") and sections 5.2–5.4:

- Each sample is edited **25 times in sequence**, each edit taking the previous output as
  input. Snapshots at iterations **1, 5, 15, 25**.
- Metrics compare x^k (k ∈ {5,15,25}) against **x¹, the first edit's output** — *not*
  against the source image. This isolates iterative-autoencoding damage from the editing
  model's own imperfection. 5, 15 and 25 are all odd, so they share edit direction with
  x¹ and the comparison is between "aligned edit operations".
- MSE, PSNR and LPIPS on samples normalized to **[0,1]**. SSIM and FID by their standard
  definitions; FID via `pytorch-fid`.
- With only 179 images FID is strongly biased upward — which is why the paper's own FID
  values sit at 60–300. Read only the gap at a fixed k.

Per-model iteration schemes (1-based; odd = forward, even = reverse):

| Model | Type | Odd iteration | Even iteration |
|---|---|---|---|
| IP2P, MagicBrush | text-guided | `instruction` | `reverse_instruction` |
| DiffEdit | text-guided | `source_global_caption` → `target_global_caption` | swapped |
| PbE | subject-guided | reference = mask-bbox crop of `target_img` | crop of `source_img` |
| SD Inpainting | mask-guided | `target_local_caption` + mask | *same every iteration* |

## Deviations we had to make, and why

1. **Masks come from the ALPHA channel — this was a serious bug, now fixed.**
   ImagenHub stores the mask in `mask_img`'s alpha channel. Their own helper is three
   lines:

   ```python
   def rgba_to_01_mask(image_rgba, reverse=False, ...):
       alpha_channel = np.array(image_rgba)[:, :, 3]
       image_bw = (alpha_channel != 255).astype(np.uint8)   # transparent => edit here
   ```

   An earlier version of this harness called `.convert("L")`, which reads RGB
   luminance and discards alpha entirely. Measured against where source and target
   actually differ (the ground-truth edit region), over all 179 samples:

   | mask source | aligned with true edit region | median ratio in/out | coverage |
   |---|---|---|---|
   | luminance (the bug) | 1.7% | 0.08 | 0.36 |
   | luminance, inverted | 97.2% | 11.9 | 0.63 |
   | **ImagenHub alpha (correct)** | **100.0%** | **36.3** | **0.111** |

   Note the trap in row 2: inverting the wrong channel reaches 97% alignment and looks
   like a fix. It is not — it only partially compensates. Reading their code is what
   gave the real answer.

   Consequences while the bug was live: SD Inpainting repainted the background while
   preserving the object, and PbE pasted its reference into the background. Any
   `pbe`/`sd_inpaint` result predating this is void. IP2P, MagicBrush and DiffEdit take
   no mask and were unaffected.

   The morphological opening/closing and largest-connected-component machinery that
   used to live here existed only to fight speckle in the wrong channel. The real alpha
   masks are clean single blobs, so cleanup now defaults to off
   (`--mask_open 0 --mask_close 0`).

2. **PbE reference images — and where PbE's data comes from.** ImagenHub has **no**
   Paint-by-Example wrapper: the only `paint`/`pbe` matches in their repo are unrelated
   vendored assets, and their subject-driven benchmark runs PhotoSwap / DreamEdit /
   BLIPDiffusion_Edit instead.

   REED did **not** take PbE's data from ImagenHub's subject-driven subset either. That
   subset has **154** eval images, whereas the paper reports 179 for every row, and §5.4
   states PbE's inputs as `{x_s, C_t^local, m}` — `target_local_caption` and `mask_img`
   are Mask_Guided_Image_Editing fields. So PbE was run on the same 179-image mask-guided
   split as the rest of the table, with the reference images constructed by hand.

   We do the same: both references are the mask's bounding box (largest connected
   component, 8px pad), cropped from `target_img` for `x_r^1` and from `source_img` for
   `x_r^2` — the latter is exactly what the paper describes building. With the correct
   alpha masks these are genuine object crops: median 17% of frame, only 2/179 above
   90%, none under 32px a side.

3. **512×512 is an assumption — the paper never states its evaluation resolution.**
   The only "512 × 512" in REED-VAE is Appendix A under *Training*: "During training, we
   preprocess the image resolution to 512 × 512" — that is the VAE fine-tune on
   LAION-5B. The subsections covering the experiments ("Experiments and comparisons",
   "Metric calculations") specify only the x¹ reference and the [0,1] normalisation;
   neither mentions resizing or cropping, and §5's notation keeps images generic
   (`x_s ∈ R^{H×W×3}`).

   We use 512 because it is the only resolution in-distribution for every component:
   REED's VAE was fine-tuned at 512, every editing model is SD-family (native 512), and
   ImagenHub — which REED builds its evaluation on — standardises to 512 in
   `imagen_hub.utils.image_helper.load_512` ("Load, resize, and center-crop an image to
   a 512x512 resolution"). Something must have been resized regardless: ImagenHub ships
   these at 500² and 1024², never 512².

   Residual risk: REED claims SD **2.1** as its base, which exists as a 512 (`-base`)
   and a 768 (`-v`) variant. If they evaluated at 768 their absolute numbers sit at a
   different resolution than ours. Unresolvable from the paper — one more reason only
   the vanilla → +SPNN *trend* is comparable, on top of the SD2.1-vs-SD1.5 VAE gap.

4. **Resize mode.** Images are squashed to 512×512. This is a non-issue on this data:
   every image in the split is square (source 500² or 1024²; mask and target always
   1024²) and aspect ratios agree on all 179 samples, so image and mask stay aligned.
   `--resize_mode center_crop` exists but has no advantage here. Note ImagenHub's
   `load_512` centre-crops to the square *then* resizes; on this all-square data
   that is the identical operation, so our pipeline matches their convention.
5. **Sampler settings** come from ImagenHub's wrappers rather than from us, since the
   paper reports none: IP2P/MagicBrush use `EulerAncestralDiscreteScheduler`, 100
   steps, guidance 7.5, image_guidance 1.5 (their comment cites
   `timothybrooks/instruct-pix2pix/edit_cli.py`); SDInpaint and DiffEdit take the
   diffusers defaults. PbE is ours: 50 steps, guidance 5.0, overridable via
   `--overrides`.

6. **LPIPS backbone** is AlexNet (the metric's usual default); the paper does not say.

## Order of operations

```bash
# 0. once, on the athena LOGIN node (compute nodes are network-isolated)
python reed_repro/prefetch_assets.py
pip install -r reed_repro/requirements.txt

# 1. FIRST — before spending any GPU-hours on the table
sbatch reed_repro/slurm/verify_codec.slurm
```

Step 1 is not optional. It checks three things:

- the checkpoint loads strictly into `SPNNAutoencoder512` (householder, hidden=128,
  r_hidden=256, scale_bound=1.0), from the **`ema`** state dict with the `spnn.` prefix
  stripped — matching `spnn_use_ema: true` in `sd15_inpaint_latent.yml`;
- **which latent-scale regime the encoder is in.** `sd15_inpaint_latent.yml` claims
  SPNN-512 is "prescaled into SD's latent space", but every diffusers pipeline multiplies
  the encoder output by `scaling_factor` itself. If that comment is right, the shim must
  divide it back out; if we guess wrong the UNet gets latents 5.5× off and every number in
  the table is quietly garbage. So we *measure* it against the real VAE. Pass the reported
  regime to `--latent_scale`;
- that SPNN's single-pass reconstruction is competitive with the VAE's — if it were not,
  the iterative comparison would be confounded from the start.

It also prints the 1/5/15/25 encode-decode ladder for both codecs (the paper's Table A2
experiment), which is the cheapest possible preview of the whole result.

```bash
# 2. pilot: 10 of the 179 samples, both codecs, all five models
bash reed_repro/slurm/submit.sh pilot10 10
sbatch --export=ALL,TAG=pilot10 reed_repro/slurm/evaluate.slurm

# 3. full run, after the pilot's table looks sane
bash reed_repro/slurm/submit.sh full179
sbatch --export=ALL,TAG=full179 reed_repro/slurm/evaluate.slurm
```

`submit.sh` fans out one job per (model, codec) arm — 10 jobs. Generation is **resumable**:
a re-submitted job skips any sample whose four snapshots already exist, so preemption
costs only the in-flight sample.

## The seed regime — resolved: fixed 42 for the four ImagenHub-wrapped models

Both regimes were run to completion at n=179 and are kept side by side:

    --seed_mode varying  ->  results/full179/         wandb: full179-<model>-<codec>
    --seed_mode fixed    ->  results/full179_seed42/  wandb: full179_seed42-<model>-<codec>-j<jobid>

**Which one reproduces the paper depends on the model, and the split is explained by
whether ImagenHub wraps that model at all.** Every ImagenHub wrapper
(`InstructPix2Pix`, `MagicBrush`, `DiffEdit`, `SDInpaint`) declares
`infer_one_image(..., seed: int = 42)` and calls `torch.manual_seed(seed)`, so looping
over it reuses one seed on every iteration.

| model | ImagenHub wrapper | matches the paper's vanilla under |
|---|---|---|
| IP2P, MagicBrush | yes, `seed=42` | **fixed 42** |
| SD Inpainting | yes, `seed=42` | **fixed 42** |
| DiffEdit | yes, `seed=42` | both track closely |
| **PbE** | **none exists** | **varying seed** |

Fixed-seed vanilla against REED's vanilla:

| | k=5 | k=15 | k=25 |
|---|---|---|---|
| IP2P paper | 0.020 | 0.110 | 0.150 |
| IP2P ours (fixed) | 0.024 | 0.099 | 0.142 |
| IP2P ours (varying) | 0.038 | 0.112 | 0.147 |
| SD-Inpaint paper | 0.010 | 0.060 | 0.110 |
| SD-Inpaint ours (fixed) | 0.010 | 0.060 | 0.112 |

MagicBrush's SSIM under fixed seed matches the paper cell-for-cell (0.65/0.21/0.13).

PbE runs the other way: its *varying*-seed vanilla is near-exact against the paper
(MSE 0.019/0.044/0.070 vs 0.020/0.040/0.070; LPIPS 0.26/0.60/0.71 identical) while
fixed-42 drifts (0.011/0.066/0.111). That is consistent rather than contradictory: PbE
is the one model ImagenHub does **not** wrap, so there was no `seed=42` default for
either REED or us to inherit — both had to improvise it.

Why the two EulerAncestral models are the sensitive ones: that scheduler is stochastic,
so seed reuse makes the early edit chain far more self-consistent, which shows up at
k=5 and washes out by k=15/25. PbE and SD Inpainting use far less noise-sensitive
sampling.

**None of this changes the finding.** SPNN beats each model's own VAE on all five
metrics at k=15 and k=25, for all five models, under *both* regimes. Only the baseline's
agreement with the paper depends on the seed protocol.

Caveat: this is inference from behaviour. REED documents no seed, scheduler, step count
or guidance scale anywhere.

## Using ImagenHub's own code

Everything that can come from ImagenHub does, because re-deriving their conventions is
what produced the mask bug above. `editors.py` calls
`imagen_hub.infermodels.{InstructPix2Pix, MagicBrush, DiffEdit, SDInpaint}` through
their uniform `infer_one_image` API when the package imports, and otherwise falls back
to a line-for-line copy of those same wrappers (same weights, scheduler, steps and
guidance). Which path ran is recorded per run as `imagenhub_api`.

Why a fallback is needed: ImagenHub 0.4.0 imports `CLIPFeatureExtractor`, which
transformers removed in v5, and pulls `openai` / `omegaconf` / `google-genai`. It is
installed with `--no-deps` so it cannot drag a different torch or diffusers into the
venv the rest of the project depends on.

Preprocessing follows their `benchmark/mask_guided_ie.py` exactly: source, target and
mask all resized to 512x512 with **LANCZOS**.

**Default base weights in ImagenHub**, which matter for interpreting REED:

| model | ImagenHub default | family |
|---|---|---|
| InstructPix2Pix | `timbrooks/instruct-pix2pix` | SD 1.5 |
| MagicBrush | `vinesmsuic/magicbrush-jul7` | SD 1.5 |
| DiffEdit | `stabilityai/stable-diffusion-2-1` | **SD 2.1** |
| SDInpaint | `runwayml/stable-diffusion-inpainting` | SD 1.5 |
| Paint-by-Example | *no wrapper exists* | SD 1.4/1.5 |

DiffEdit is the **only** one of REED's five that is SD 2.1. We pin it to SD 1.5
(`REED_DIFFEDIT_ID` overrides) because SPNN-512 was distilled against SD 1.5's VAE and
an SD2.1 pipeline would place it in the wrong latent space.

This also casts doubt on REED's "All experiments conducted are based on the released
v2.1 of Stable Diffusion": four of their five models are SD 1.x checkpoints that cannot
run on an SD 2.1 UNet. The sentence most likely describes the VAE they fine-tuned,
which they then swapped into SD1.x pipelines — meaning REED's own "vanilla" baseline is
probably an SD2.1 VAE inserted into SD1.5 models, whereas ours is each model's own VAE.

Precision: ImagenHub loads every pipeline in fp16. We cast both arms to fp32, because
SPNN is a projection whose encode/decode fixed point sits at ~1e-8 in fp32 and fp16
would blunt the property under test. Applied to both arms, so precision can never be
what separates them.

## Reporting to wandb

`report` logs everything into one run in the **`reed-vae-repro`** project:

- **A section per ImagenHub sample.** Grids are logged under the key
  `<img_id>_t<turn>/<model>`, and wandb splits workspace sections on the first `/` —
  so each of the 179 samples gets its own collapsible section holding one panel per
  editing model.
- **Each panel is a VAE-vs-SPNN iteration grid**: row 1 the vanilla SD1.5 VAE, row 2
  `+ SPNN`, columns running `source, 1, 2, … 25`. Stacking the codecs makes the
  divergence readable vertically at any fixed iteration, which is the comparison the
  whole experiment is about.
- **`table1`**, a wandb Table of the full metric sweep, plus flat summary keys
  `metrics/<model>/<codec>/<metric>@<k>` so arms can be charted against each other.

Because the grids need every iteration (not just 5/15/25), `generate` writes a PNG for
all 25 by default — about 45k files, ~18 GB for the full run. `--snapshots_only`
reverts to the four metric snapshots if disk is ever tight; metrics are unaffected
either way, the grids just get sparse.

```bash
sbatch --export=ALL,TAG=pilot10,LIMIT=10 reed_repro/slurm/report.slurm
```

That runs `evaluate` then `report`. If a compute node ever cannot reach wandb, pass
`WANDB_MODE=offline` and `wandb sync` the run directory from the login node.

## Layout

```
reed_repro/
  data.py             ImagenHub filtered split -> 512px Samples (image, mask, captions, PbE refs)
  codecs.py           SPNN-512 as a drop-in diffusers AutoencoderKL; latent-scale handling
  editors.py          the five editing models + their per-model iteration schemes
  metrics.py          MSE/PSNR/LPIPS/SSIM on [0,1]; FID via pytorch-fid
  verify_codec.py     RUN FIRST: strict load, latent-scale regime, encode/decode ladder
  run_repro.py        generate | evaluate; writes metrics.json, table1.csv, table1.md
  prefetch_assets.py  warm the HF cache from the login node
  slurm/              verify_codec, generate (per-arm), evaluate, submit.sh fan-out
results/<tag>/<model>/<codec>/iter_{01,05,15,25}/<img_id>_t<turn>.png
results/<tag>/{metrics.json,table1.csv,table1.md}
```

Both codecs are seeded identically per (sample, iteration) via `seed_for()`, so the two
arms see the same noise and the measured difference is attributable to the codec alone.
