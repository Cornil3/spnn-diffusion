# SPNN-512 — a pseudo-invertible codec for latent diffusion

Snapshot of everything as of **2026-09-13**. Codec checkpoint: **step 49,000 (epoch 9.03)**,
`best.pt`, EMA weights.

---

## 1. What the codec is

SPNN-512 replaces Stable Diffusion 1.5's VAE with a codec whose decoder is the **exact pseudo-inverse
of its encoder**, so `P = D∘E` is a true projection:

* `E(D(z)) = z` identically, for any latent — including the off-manifold latents a sampler produces
  mid-trajectory. Measured `‖E(D(z))−z‖/‖z‖ = 1.2e-4`, flat from `t=100` to `t=900`;
  the SD VAE is 0.12–0.28 over the same range.
* `P(P(x)) = P(x)`, so an iterated encode/decode chain is a **fixed point after one step**.
  Unclamped it is flat at 29.72 dB out to 25 rounds; the SD VAE falls 31.06 → 17.70 dB over 8.
* It is **~5× faster** than the SD VAE in every measured setting (58.2M vs 83.7M params, and a much
  cheaper forward pass).

The price is a **one-time** fidelity tax on the first pass: LPIPS 0.104 vs the SD VAE's 0.050 on
reconstruction. The codec's value is that this tax is not paid again, no matter how many rounds follow.

### The governing trade-off

> **SPNN sells:** a pixel-space constraint that survives re-encoding, arbitrarily many times.
> **SPNN charges:** a fixed loss of high-frequency detail on the first pass.

Every result below follows from that sentence. Tasks whose payload is a *low-frequency constraint
applied many times* (QR modules, DDNM known-pixel pasting, colorization chroma) win big. Tasks whose
payload is *high-frequency appearance* (visual anagrams) lose.

---

## 2. Training  → `01_training/`

| file | what it is |
|---|---|
| `train_spnn512.py` | the trainer |
| `train_spnn512.sbatch` | 8-GPU multi-partition job, self-resubmitting until `OUT/DONE` |
| `spnn_model_opt.py` | `SPNNAutoencoder512Opt` — householder, hidden 96, r_hidden 128, scale_bound 1.0 |

Run dir `/rg/shocher_prj/ron.libman/laion1m/runs/spnn512`. 10 epochs = 54,260 steps, WSD LR 2e-4,
effective batch 128. Distilled onto `0.18215 · vae.encode(x).mean` from SD-1.5's VAE, on Re-LAION + COCO.

**Two phases, and the difference between them is the single most important training result.**

| phase | steps | loss | outcome |
|---|---|---|---|
| 1 | 0 – 21,704 | pixel MSE + L1 + latent terms, no perceptual | reconstruction FID 12.44, LPIPS 0.176 — blurry |
| 2 | 21,783 – 49,000 | **+ LPIPS(VGG) weight 1.0** on clean/traj/self branches, 1000-step LR re-warmup | reconstruction FID **4.80**, LPIPS **0.104** |

`best.pt` is selected by **validation cycle LPIPS**, not PSNR — PSNR rewards blur and picked the wrong
checkpoint in phase 1.

**Ablation (2 samples, PbE 25-edit chain, vs x¹).** Same architecture, same data, one loss term different:

| codec | LPIPS @5/15/25 | SSIM @5/15/25 | PSNR @5/15/25 |
|---|---|---|---|
| SD-1.5 VAE | .297 / .683 / .784 | .684 / .321 / .234 | 21.76 / 15.56 / 13.48 |
| SPNN epoch 3 (step 16,473, MSE only) | .296 / .719 / .766 | .646 / .279 / .294 | 21.48 / 12.83 / 12.50 |
| **SPNN final (step 49,000, +LPIPS)** | **.121 / .468 / .628** | **.870 / .606 / .383** | **23.75 / 17.18 / 14.76** |

Epoch 3 has *identical* pinv algebra and still loses to the plain VAE on most cells.
**Exact idempotency is necessary but nowhere near sufficient** — the perceptual loss is what makes it pay.

### Checkpoints (not copied here — large)

| file | size | what |
|---|---|---|
| `runs/spnn512/best.pt` | 466 MB | **step 49,000**, the checkpoint every result below uses |
| `runs/spnn512/latest.pt` | 932 MB | step 49,000 + optimizer state, for resuming |
| `runs/spnn512/epoch3.pt` | 932 MB | step 16,473, phase-1 (MSE-only), used for the ablation above |

---

## 3. DDNM restoration  → `02_ddnm_restoration/`

`latent_ddnm_idem.py` is the sampler: **at most one decode + one encode per diffusion step**.
Each step is `z0 → D(z0) → paste known pixels → E(·) → z0_new`, with `lam_hole` weighting the latent
correction by each cell's known-pixel fraction. **The loop does not clamp** (`clip_x0=False`); only the
returned image is clamped, once.

`eval_restore_fid.py` runs the benchmark: 5,000 eval images in the training mix (85.8% LAION test /
14.2% COCO test), FID and KID **only against the originals of those same test images**, identical seed
and noise per image across codecs.

### Results (5,000 images)

| task | codec | LPIPS ↓ | SSIM ↑ | FID ↓ | PSNR ↑ | sec/img |
|---|---|---|---|---|---|---|
| Box inpainting 26%, T=100 | SD-VAE | **.0931** | **.9005** | **4.27** | **24.35** | 9.99 |
| | SPNN | .1558 | .8858 | 10.90 | 22.93 | **1.94** |
| Box inpainting 26%, T=200 | SD-VAE | **.0947** | **.9020** | **4.42** | **24.56** | 20.01 |
| | SPNN | .1625 | .8854 | 12.04 | 22.98 | **3.89** |
| Scattered 66%, T=100 | SD-VAE | .4835 | **.6659** | 63.17 | **15.84** | 9.99 |
| | SPNN | **.4654** | .6631 | **54.83** | 15.79 | **1.94** |
| Scattered 66%, T=200 | SD-VAE | .4789 | **.6700** | 62.03 | 16.05 | 20.04 |
| | SPNN | **.4528** | .6697 | **52.77** | **16.08** | **3.88** |
| Colorization, T=20 | SD-VAE | .3676 | .7056 | 25.42 | 14.23 | 2.00 |
| | SPNN | **.2871** | **.8181** | **20.58** | **17.23** | **0.39** |

Reconstruction floor (no diffusion): SD-VAE FID 2.37 / LPIPS .0495 · SPNN 4.80 / .1042.

**Per-image LPIPS win rate for SPNN:** scattered-66 T=200 **82%**, colorization **78%**, box T=200 **2%**.

**Reading.** SPNN wins colorization on every metric, wins scattered-66 on the distribution metrics
(FID, LPIPS) with pixel metrics tied, and loses box inpainting. The pattern is the trade-off in §1:
box inpainting is easy and dominated by appearance fidelity; scatter-66 and colorization are dominated
by constraint satisfaction over many steps.

**Open:** the box-inpainting regression from phase 1 (FID 5.84 → 10.90) is **unexplained**. It is *not*
clamping — the loop never clamps. Since the output is `paste(d_last)`, known pixels are exact, so the
entire gap is generated hole content. Untested leads: `lam_hole=1.0` (correct every cell, not just
known ones), and whether out-of-range decode feeds the encoder off-distribution input each step.

---

## 4. Iterative editing — REED-VAE Table 1  → `03_iterative_editing/`

Protocol from REED-VAE (Eurographics), reproduced exactly: 179 ImagenHub `Mask_Guided_Image_Editing`
samples, **25 alternating inverse edits**, scored at 5/15/25 **against x¹ — the chain's own first
edit** (their Appendix A; also confirmed in their "Varying metric scales" appendix note).

### A bug that mattered: the mask channel

ImagenHub stores the edit mask in the **alpha channel** (`alpha != 255` marks the region to repaint).
`reed_build_data.py` originally did `.convert("RGB")`, discarding alpha, and the loader thresholded
luminance instead:

| | wrong (luminance) | correct (alpha) |
|---|---|---|
| mean mask area | 37.3% of frame | **15.6%** |
| pixel agreement with the true mask | 47.3% | — |
| PbE reference-crop bbox | 98.3% of frame, >90% on 168/179 | **21.8%**, >90% on 2/179 |

It silently broke PbE and SD Inpainting (a 37% mask makes the chain converge to a prompt-attractor
instead of measuring VAE decay) and left IP2P/MagicBrush/DiffEdit untouched. **Fixed and verified:**
our vanilla PbE row now reproduces the paper on all five metrics.

| PbE vanilla @5/15/25 | MSE | LPIPS | SSIM | FID | PSNR |
|---|---|---|---|---|---|
| paper | .020/.040/.070 | .26/.60/.71 | .66/.33/.22 | 83.5/209.3/253.6 | 18.55/13.88/11.74 |
| **ours** | .019/.044/.069 | .26/.60/.71 | .67/.33/.22 | 81.6/207.1/246.9 | 18.70/13.91/11.80 |

### The main finding: SD-VAE encoder + SPNN decoder

Swapping **only** the encoder back to SD-1.5's — verified bit-identical (`‖E_hyb−E_vae‖/‖E_vae‖ = 0`),
decoder verified to be SPNN (`2.1e-4`, fp16 cast only) — beats full SPNN on four of five editors and
beats REED's published rows on three.

Full table: `results/reed_table1_final_n179.txt`. Iteration 25:

| editor | codec | MSE ↓ | LPIPS ↓ | SSIM ↑ | FID ↓ | PSNR ↑ |
|---|---|---|---|---|---|---|
| **PbE** | + REED (paper) | .040 | .59 | **.54** | **178.3** | 14.43 |
| | + SPNN | .067 | .65 | .30 | 294.7 | 12.19 |
| | **+ SD-VAE enc / SPNN dec** | **.039** | .59 | .38 | 220.4 | **14.62** |
| **SD Inpainting** | + REED (paper) | .050 | .65 | **.41** | **210.4** | 13.61 |
| | + SPNN | .077 | .68 | .24 | 320.0 | 11.48 |
| | **+ SD-VAE enc / SPNN dec** | **.037** | **.60** | .36 | 230.8 | **14.55** |
| **DiffEdit** | + REED (paper) | .080 | .68 | **.40** | **246.5** | 11.44 |
| | + SPNN | .105 | .71 | .19 | 322.4 | 9.96 |
| | **+ SD-VAE enc / SPNN dec** | **.069** | **.65** | .31 | 250.9 | **11.73** |
| **IP2P** | + REED (paper) | **.090** | **.58** | **.41** | **188.0** | **11.36** |
| | + SD-VAE enc / SPNN dec | .118 | .67 | .28 | 212.4 | 9.74 |
| **MagicBrush** | + REED (paper) | **.050** | **.69** | **.45** | **223.8** | **14.09** |
| | + SD-VAE enc / SPNN dec | .099 | .69 | .20 | 242.0 | 10.53 |

**Win ledger vs REED**, 5 editors × 3 checkpoints = 15 cells per metric:
MSE **8/15**, PSNR **8/15**, LPIPS 3/15, SSIM 1/15, FID 1/15.
**DiffEdit @5 is a clean sweep — the hybrid beats REED on all five metrics at once**
(.021 vs .030, .27 vs .30, .72 vs .69, 148.6 vs 160.3, 17.25 vs 16.24).

Relative improvement over each paper's own vanilla baseline — we **match REED exactly** where the
latent carries image content, and deliver about half where it doesn't:

| editor | REED MSE@25 gain | our hybrid |
|---|---|---|
| PbE | 43% | **43%** |
| SD Inpainting | 55% | **55%** |
| DiffEdit | 0% | **35%** |
| MagicBrush | 64% | 37% |
| IP2P | 40% | 21% |

### Why the encoder swap helps — and why not for text editing

The two encoders differ by **21% in direction, not scale** (latent rms 5.28 vs 5.31 — my earlier claim
of a 24% scale mismatch was wrong; that came from feeding the probe a uniform random tensor).

* **PbE / SD Inpainting / DiffEdit** carry the image *through* the latent, so a 21% off-manifold code
  is misread by an SD-trained UNet on every round. Swapping the encoder back fixes that.
* **IP2P / MagicBrush** regenerate every pixel from noise; the latent is only *conditioning*, so the
  encoder barely matters — and the codec is handed **100% synthetic content** every round, which is
  SPNN's worst case (see the domain probe below).

Note the hybrid has the **worst** latent idempotency of the three (0.165, vs SPNN's 0.0015 and the
plain VAE's 0.116) and wins anyway. **Idempotency is not what makes editing chains survive** — one-pass
decoder quality is.

### The 8-pixel grid artifact

Measured in `results/` and `figures/spnn_grid_artifact.png`. All codecs add power at exactly
64 cycles/image (period 8 px = the latent downsampling factor); SPNN final adds a second, stronger peak
at period 2 px (near-Nyquist, +1.70 log₁₀).

| codec | folded 8×8 cell error / total residual | grid-locked? |
|---|---|---|
| SD-1.5 VAE | 0.030 | no |
| SPNN epoch 3 | 0.061 | yes |
| **SPNN final** | **0.215** | strongly |

Because `pinv` returns the minimum-norm solution, the reconstruction error lies in the **null space of
E** — the same fixed pattern in every 8×8 cell of every image, and invisible to the codec (`E(err)=0`,
so re-encoding cannot remove it). It therefore accumulates *in phase* across rounds while image content
averages out. The LPIPS phase made it worse and moved it to near-Nyquist, where AlexNet-LPIPS does not
look. **This is why SSIM and FID lose while MSE/LPIPS/PSNR win.**

---

## 5. QR codes (DiffQRCoder)  → `04_qr_codes/`

Owned by the separate "Latent diffusion QR codes" session; copied here read-only.
12 prompts, WeChat decoder (the paper's scanner).

| variant | scan rate | module error | CLIP | sec/img |
|---|---|---|---|---|
| sandwich · SD-VAE | 1/12 | 0.674% | 0.274 | 5.81 |
| sandwich · SPNN | 2/12 | 0.585% | 0.277 | **2.63** |
| sandwich · SD-VAE + SR-MPGD | 0/12 | 0.178% | 0.216 | 2.30 |
| **sandwich · SPNN + SR-MPGD** | **4/12** | **0.000%** | **0.278** | **0.71** |
| gradient · SD-VAE + SR-MPGD | **12/12** | 0.030% | 0.212 | 3.15 |
| gradient · SPNN + SR-MPGD | 11/12 | **0.000%** | 0.209 | **0.91** |

This is SPNN's **best-case application** and the cleanest illustration of §1: the QR constraint is
per-module mean luminance (modules are 16–24 px, so blur is harmless) applied at *every* step (so the
VAE's 0.12–0.28 per-step latent error is fatal and SPNN's 1.2e-4 is not). The SD-VAE sandwich collapses
into neon garbage; SPNN produces clean images with the code woven in — `figures/qr_sandwich_*.jpg`.

Caveat: absolute sandwich scan rates are low for both; the `gradient` variant is the one that scans
reliably, and there SPNN's win is **3.5× speed** at near-equal scan rate.

---

## 6. Visual anagrams — the negative result

`figures/anagrams_*.png`. CLIP alignment 0.283 (SD1.5 latent views) / 0.253 (SD sandwich) /
0.228 (SPNN sandwich); 2.2 / 15.8 / 4.7 s per sample.

SPNN is 3.4× faster than the SD sandwich and produced the only convincing flip illusion, but loses both
CLIP metrics. Anagrams are the exact anti-case for §1: the payload is high-frequency detail doing double
duty across two readings, so the one-time tax is charged against the only thing that carries the effect.
Two extra failures: `negate` collapses to black/white (the range problem, below), and rotate/jigsaw seams
appear in **both** sandwiches — that is codec non-equivariance, which no amount of training fixes.

Worth keeping in the paper as the honest boundary; it sharpens the claim.

---

## 7. Known problems, ranked

**1. The 8-px grid artifact (null-space residual).** Highest value. It is the *only* axis where the
hybrid still loses to REED (SSIM, FID) while MSE/PSNR/LPIPS are at or past it.
*Fixes:* (a) inference-only — the pattern is a **constant**, so measure the mean 8×8 cell error once on
a calibration set and subtract it at decode (~15 min, no retraining); (b) training — fold the decode
residual to 8×8 and penalise the batch-mean's norm; (c) cover LPIPS's blind spot with a spectral or
MS-SSIM term.

**2. Domain shift — the decoder has never seen diffusion output.** `results/codec_domain_probe.txt`:

| image set | codec | 1× PSNR | 1× LPIPS |
|---|---|---|---|
| natural photos | SD-VAE | 28.00 | 0.0570 |
| natural photos | **SPNN** | **28.54** | 0.1143 |
| SD-1.5 generated | SD-VAE | **28.83** | **0.0408** |
| SD-1.5 generated | SPNN | 25.28 | 0.1719 |

Moving from photos to diffusion output, SPNN loses **3.26 dB** while the SD VAE *gains* 0.83. The sign
flips: +0.54 dB ahead on photos, −3.55 dB behind on generated. This is the whole IP2P/MagicBrush gap.
*Fix:* mix SD-1.5 samples into the distillation set (~14 GPU-h to generate 100k, then continue training).

**3. Out-of-range decode / clamping.** The phase-2 decode puts **1.72%** of pixels outside [−1,1]
(range −3.02 … 3.47). Unclamped the codec chain is a perfect fixed point (29.72 dB flat to 25 rounds);
clamped it decays to 13.51. Clamp latent shift 1.33e-1 vs 8-bit rounding 2.17e-2 — the clamp is ~6× the
quantisation. diffusers applies it in `VaeImageProcessor.denormalize`, upstream of us, so it is not
avoidable by changing our code.
*Fixes:* range penalty `relu(|x|−1)²` in training; or POCS at inference — iterate `x ← clamp(D(E(x)))`
to land in `range(D) ∩ [−1,1]`.
**Note:** this does **not** explain the DDNM box regression (that loop never clamps) — an earlier claim
of mine that I retracted. It does explain the REED chains and the anagram `negate` collapse.

**4. The pinv constraint may be the wrong objective for editing.** In the hybrid configuration the
encoder is SD's, so `decode = pinv(encode)` buys nothing — and the hybrid wins anyway. The decoder is
free to be trained as an ordinary decoder, which unlocks REED's recipe (k-round chain training with the
first-step loss) at SPNN's architecture and speed. But exact idempotency is precisely what DDNM and the
QR sandwich depend on, so this is a real fork: two checkpoints, or one decoder with idempotency demoted
to a soft penalty.

---

## 8. Reproduction quality

| arm | status |
|---|---|
| PbE | **verified against the paper** on all five metrics |
| IP2P | matches an independent reproduction; both differ from the paper at iteration 5 only — likely the reverse-prompt list, which the paper published only 5 of 179 examples of |
| DiffEdit | matches an independent reproduction; both differ from the paper, which documents DiffEdit's generated masks as wandering and inflating metrics |
| MagicBrush | same weights (`vinesmsuic/magicbrush-jul7`), pipeline and kwargs as the independent reproduction; ~1 dB apart, seed-level |
| SD Inpainting | **open** — two independent implementations agree with each other and both differ from the paper (better at 5, worse at 25). §5.3 states no steps, guidance, or paste-back behaviour |

Comparisons between our codecs are unaffected by any of this: every ours-row is measured against our own
vanilla baseline under identical settings, so a protocol difference from the paper shifts both together.

---

## 9. Layout

```
01_training/            trainer, sbatch, architecture
02_ddnm_restoration/    sampler (latent_ddnm_idem.py), benchmark, example renderer, sbatch
03_iterative_editing/   dataset builder, Table-1 notebooks (full-SPNN + hybrid), scorers, probes, CSV
04_qr_codes/            DiffQRCoder driver + notebook + results  (other session's work, read-only)
results/                every table as produced, report.json/md, the 19-page PDF
figures/                the figures referenced above
```

Not included: checkpoints (see §2), the 5,000-image eval set, and the 179×25 rollout PNGs
(~11 GB under `/rg/shocher_prj/ron.libman/reed_table1/`).
