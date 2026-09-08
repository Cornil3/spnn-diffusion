"""
Metrics exactly as REED-VAE defines them (Appendix A, "Metric calculations").

Two details from the paper drive everything here:

  * "MSE, PSNR, and LPIPS are computed with each image sample normalized to the [0,1]
    range prior to evaluation." So all four pairwise metrics run on [0,1] tensors.
  * "we compute metrics between the given target image (one of iterations 5, 15, or
    25) and x^1 - *not* to the source image." The reference is the first edit's
    output, which isolates iterative-autoencoding damage from the editing model's own
    imperfection. `run_repro.py` is what enforces this; metrics here just take a pair.

FID uses pytorch-fid, which is the implementation the paper cites [Sei20]. With only
179 images FID is heavily biased upward - which is why the paper's own FID values sit
in the 60-300 range rather than the single digits. Ours will be too; only the
vanilla-vs-SPNN gap at a fixed iteration count is meaningful.
"""

import math
import os

import torch


def mse01(a, b):
    return torch.mean((a.clamp(0, 1) - b.clamp(0, 1)) ** 2).item()


def psnr01(a, b):
    m = mse01(a, b)
    return float("inf") if m == 0 else 10.0 * math.log10(1.0 / m)


class PairwiseMetrics:
    """LPIPS + SSIM models held once and reused across the whole sweep."""

    def __init__(self, device="cuda", lpips_net="alex"):
        self.device = device
        import lpips
        self.lpips = lpips.LPIPS(net=lpips_net).to(device).eval()
        for p in self.lpips.parameters():
            p.requires_grad_(False)
        try:
            from torchmetrics.functional.image import structural_similarity_index_measure
            self._ssim = structural_similarity_index_measure
        except Exception:
            self._ssim = None

    def _ssim_value(self, a, b):
        if self._ssim is not None:
            return float(self._ssim(a, b, data_range=1.0).item())
        from skimage.metrics import structural_similarity
        x = a[0].permute(1, 2, 0).cpu().numpy()
        y = b[0].permute(1, 2, 0).cpu().numpy()
        return float(structural_similarity(x, y, channel_axis=2, data_range=1.0))

    @torch.no_grad()
    def __call__(self, a, b):
        """a, b: [3,H,W] or [1,3,H,W] float tensors in [0,1]. a=x^k, b=x^1."""
        if a.dim() == 3:
            a = a.unsqueeze(0)
        if b.dim() == 3:
            b = b.unsqueeze(0)
        a = a.to(self.device).clamp(0, 1)
        b = b.to(self.device).clamp(0, 1)
        # lpips(normalize=True) takes [0,1] and rescales to [-1,1] internally.
        d = self.lpips(a, b, normalize=True).item()
        return {
            "mse": mse01(a, b),
            "psnr": psnr01(a, b),
            "lpips": d,
            "ssim": self._ssim_value(a, b),
        }


def compute_fid_paired(dir_a, dir_b, keys, device="cuda", batch_size=32, dims=2048):
    """FID over an explicit key list, so it scores the same samples as the other metrics.

    pytorch-fid only takes directories, and it would otherwise consume every PNG present
    -- leaving FID unpaired while MSE/PSNR/LPIPS/SSIM are paired. We stage symlinks to
    just the paired keys in temp dirs and point it at those.
    """
    import shutil
    import tempfile
    if not keys:
        return None
    tmp = tempfile.mkdtemp(prefix="reed_fid_")
    try:
        staged = []
        for label, src in (("a", dir_a), ("b", dir_b)):
            d = os.path.join(tmp, label)
            os.makedirs(d, exist_ok=True)
            for k in keys:
                s_ = os.path.join(str(src), f"{k}.png")
                if os.path.exists(s_):
                    os.symlink(os.path.abspath(s_), os.path.join(d, f"{k}.png"))
            staged.append(d)
        return compute_fid(staged[0], staged[1], device=device,
                           batch_size=batch_size, dims=dims)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def compute_fid(path_a, path_b, device="cuda", batch_size=32, dims=2048):
    """FID between two directories of PNGs, via pytorch-fid (the paper's [Sei20]).

    Returns None rather than raising if pytorch-fid is absent or a directory is empty,
    so a missing optional dependency cannot cost us the four pairwise metrics too.
    """
    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths
    except ImportError:
        print("  [fid] pytorch-fid not installed — skipping FID "
              "(pip install -r reed_repro/requirements.txt)")
        return None

    n_a = len([f for f in os.listdir(path_a) if f.endswith(".png")])
    n_b = len([f for f in os.listdir(path_b) if f.endswith(".png")])
    if n_a == 0 or n_b == 0:
        return None
    if min(n_a, n_b) < 50:
        # The covariance of a 2048-d feature is not estimable from a handful of
        # samples. At the full 179 it is already biased; below ~50 it is noise.
        print(f"  [fid] only {min(n_a, n_b)} images — FID is not meaningful at this "
              f"sample size; reported for smoke-testing only")
    return float(calculate_fid_given_paths(
        [str(path_a), str(path_b)], batch_size=min(batch_size, n_a, n_b),
        device=device, dims=dims, num_workers=0))
