# One-off diagnostics (`_*.py`)

Each of these settled a specific question that the results depend on. They are kept
because the claims in `REPRODUCTION.md` are only as good as the measurements behind
them, and every one of these overturned or sharpened an assumption.

| script | question it settled | answer |
|---|---|---|
| `_scale_check.py` | Is SD's 0.18215 baked into SPNN's encoder, or does it emit raw-VAE-scale latents? | **Baked in.** Least-squares α = 0.1764 (raw would give ≈1.0), but α/0.18215 = 0.968 with R² = 0.965 — approximately, not exactly. Choosing `raw` would have fed the UNet latents 5.5× too small. |
| `_vae_audit.py` | Do the five editing models share a VAE, or does each have its own? | **All five are bit-identical to SD 1.5's** (`max|dW| = 0.000e+00`, scaling_factor 0.18215). One SPNN codec is therefore valid for every arm, and "each model's own VAE" is a well-defined baseline. |
| `_idem_check.py` | Is SPNN's flat 32.50 dB encode/decode ladder a genuine fixed point or an artefact? | **Genuine.** `MSE(x¹,x⁰) = 2.3e-03` (so not a passthrough), then consecutive iterations differ by ~1e-8 from k=2 onward; per-image PSNR spread 1.71 dB (so not collapsing to a constant). |
| `_de_mask_probe.py` | Why does DiffEdit produce structurally wrong edits? | Its **self-generated mask is misplaced**, not diffuse: coverage 15–17% but IoU only **0.044 / 0.094** against ground truth. Caused by 70–82% word overlap between the caption pair it contrasts. Not a codec or base-model problem. |
| `_seed_probe.py` | Why is our IP2P/MagicBrush vanilla worse than the paper's at k=5 but matching at k=15/25? | Tests whether a **fixed seed across iterations** (ImagenHub's `infer_one_image` default is `seed=42`) explains it. EulerAncestral is stochastic, so seed reuse makes the early chain far more self-consistent. |

Run them from the project root (they insert `os.getcwd()` on `sys.path`), on a GPU node:

```bash
python reed_repro/_vae_audit.py
python reed_repro/_scale_check.py
```

`reed_repro/sanity_check.py` is the non-throwaway counterpart — a supported entry point
that runs one sample through all five editors and reports whether edits actually happen
and whether they land inside the mask.
