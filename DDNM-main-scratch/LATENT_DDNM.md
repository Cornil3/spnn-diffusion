# Latent-space DDNM (SD1.5 + SPNN-512)

A clean-room conversion of the official pixel-space DDNM (ICLR 2023) into a
**latent-space** restorer, built directly on top of the official code in this
folder. The pixel-space DDNM (`main.py` with the original configs) still works
unchanged; the latent path is additive.

## Idea

Pixel DDNM enforces measurement consistency on the predicted clean image each
reverse step:

```
x0      = (x_t - sqrt(1-a_t) eps) / sqrt(a_t)
x0_hat  = x0 - A^+(A x0 - y)        # range space <- measurement y, null space <- prior
x_{t-1} = sqrt(a_{t-1}) x0_hat + DDIM noise
```

We keep the diffusion in **SD1.5's latent space** (UNet predicts `eps` on
`z`), but the operator `A` and measurement `y` are in **pixel space**, so we
bridge with a codec `(E, D)` every step:

```
z0      = (z_t - sqrt(1-a_t) eps) / sqrt(a_t)
x0      = D(z0)                     # decode
x0_hat  = x0 - A^+(A x0 - y)        # exact, in pixel space
z0_hat  = E(x0_hat)                 # re-encode
z_{t-1} = sqrt(a_{t-1}) z0_hat + DDIM noise
```

The `decode -> pin -> encode` round-trip is exact data consistency but relies on
`E(D(.)) ~= identity`. The lossy KL-VAE corrupts `z0_hat` every step and the
error compounds; the near-invertible **SPNN-512** codec does not. Running both
codecs under identical sampling exposes that gap (the experiment).

## Files added

| file | role |
|------|------|
| `functions/latent_codec.py`      | SD-VAE + SPNN-512 codec wrappers and loaders |
| `functions/latent_ddnm.py`       | latent DDNM sampling loop, mask + schedule helpers |
| `guided_diffusion/latent_diffusion.py` | runner: SD1.5 UNet/scheduler/text-encoder + codecs + data + PSNR |
| `configs/sd15_inpaint_latent.yml`| config (codec choice, mask, steps) |
| `run_latent_ddnm.sbatch`         | container launch script for the cluster |

`main.py` dispatches to the latent runner when `config.model.type` starts with
`latent`; otherwise it runs the original pixel-space `Diffusion`.

## Run

Inside the pytorch container (diffusers + transformers; SD1.5 in `$HF_HOME`):

```bash
cd DDNM-main-scartch
python main.py --config sd15_inpaint_latent.yml --deg inpainting --ni -i myrun
# or on SLURM:
sbatch run_latent_ddnm.sbatch myrun
```

Outputs: `exp/image_samples/myrun/{VAE,SPNN}/<i>_{orig,masked,recon}.png` and a
final per-codec average PSNR.

## Notes / choices

- The exact `alphas_cumprod` is built analytically from SD1.5's `scaled_linear`
  schedule (beta 0.00085 -> 0.012), so the UNet's `eps` stays calibrated without
  needing the scheduler config (which isn't in the local HF cache). SD1.5 weights
  load offline from `$HF_HOME` (`HF_HUB_OFFLINE=1`).
- SPNN-512 is treated as *prescaled* (it encodes/decodes directly in SD's scaled
  latent space). If a checkpoint instead lives in raw VAE space, set
  `codec.spnn_external_scale: 0.18215`.
- `eta` (DDIM stochasticity) defaults to the official DDNM value 0.85; pass
  `--eta 0` for deterministic sampling.
- Data consistency uses constant `lambda = 1` (the noise-free DDNM projection).
  No `lambda`/back-projection schedules -- this is a clean baseline.
