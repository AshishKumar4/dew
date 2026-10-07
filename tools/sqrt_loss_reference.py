#!/usr/bin/env python3
"""Diffusion-LM's training loss for an x_0 model on its sqrt schedule, for
tests/fixtures/sqrt_loss/loss.npz.

The reference is `GaussianDiffusion.training_losses` of Diffusion-LM's
improved_diffusion/gaussian_diffusion.py, with the nn.py and losses.py
beside it, at XiangLi1999/Diffusion-LM@759889d, fetched and run as published
in PyTorch: its `emb` training mode, the model predicting x_0 (`START_X`),
fixed variances and the plain `MSE` loss, on the 2000-step `sqrt` table its
own `get_named_beta_schedule` builds. That is Dew's `Sqrt` preset.

The published code reads its float64 table as float32 (`_extract_into_tensor`
calls `.float()`), so it runs once as published, and once with that read
widened to float64 for the float64 truth, its one change. The network is a
stand-in that ignores time, x_0 = a * x_t + c, and an offset on its output
carries the gradient. Both runs read the same float32 inputs. What lands:
the images, a, c, the step indices and the noise, the per-example loss and
the gradient of the batch mean in the network's output, in float32 and in
float64.

    python tools/sqrt_loss_reference.py --out tests/fixtures/sqrt_loss/loss.npz
"""

import argparse
import importlib
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO, COMMIT = "XiangLi1999/Diffusion-LM", "759889d58ef38e2eed41a8c34db8032e072826f4"
STEPS = 2000
SHAPE = (6, 3, 4, 4)


def published():
    """Diffusion-LM's gaussian_diffusion module, from its own files at the pinned commit."""
    package = Path(tempfile.mkdtemp()) / "improved_diffusion"
    package.mkdir()
    (package / "__init__.py").write_text("")
    for name in ("gaussian_diffusion.py", "nn.py", "losses.py"):
        url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/improved-diffusion/improved_diffusion/{name}"
        with urllib.request.urlopen(url, timeout=60) as response:
            (package / name).write_bytes(response.read())
    sys.path.insert(0, str(package.parent))
    return importlib.import_module("improved_diffusion.gaussian_diffusion")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "tests" / "fixtures" / "sqrt_loss" / "loss.npz")
    out = parser.parse_args().out
    gd = published()
    rng = np.random.default_rng(0)
    # Both runs read these float32 values, the float64 one widened.
    arrays = {name: value.astype(np.float32) for name, value in (
        ("images", rng.uniform(-1, 1, SHAPE)), ("a", rng.normal(0, 0.5, SHAPE[1:])),
        ("c", rng.normal(0, 0.3, SHAPE[1:])), ("noise", rng.standard_normal(SHAPE)))}
    # The last step is left out: Diffusion-LM clips its beta at 0.999 there
    # where Dew's schedule reaches (0, 1).
    indices = rng.integers(0, STEPS - 1, SHAPE[0])
    diffusion = gd.GaussianDiffusion(betas=gd.get_named_beta_schedule("sqrt", STEPS),
                                     model_mean_type=gd.ModelMeanType.START_X,
                                     model_var_type=gd.ModelVarType.FIXED_LARGE, loss_type=gd.LossType.MSE,
                                     training_mode="emb")
    landed = {**arrays, "indices": indices}
    published_read = gd._extract_into_tensor

    def widened(arr, timesteps, broadcast_shape):
        """`_extract_into_tensor` as published but for its `.float()`."""
        res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps]
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    for dtype, suffix in ((torch.float32, ""), (torch.float64, "_f64")):
        gd._extract_into_tensor = widened if dtype == torch.float64 else published_read
        a, c = (torch.tensor(arrays[name], dtype=dtype) for name in ("a", "c"))
        offset = torch.zeros(SHAPE, dtype=dtype, requires_grad=True)

        def model(x_t, t, a=a, c=c, offset=offset):
            return a * x_t + c + offset

        images, noise = (torch.tensor(arrays[name], dtype=dtype) for name in ("images", "noise"))
        loss = diffusion.training_losses(model, images, torch.tensor(indices), noise=noise)
        landed[f"loss{suffix}"] = loss["loss"].detach().numpy()
        loss["loss"].mean().backward()
        landed[f"grad{suffix}"] = offset.grad.numpy()
    gd._extract_into_tensor = published_read
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **landed)
    print(f"{out}: loss {landed['loss_f64']}")


if __name__ == "__main__":
    main()
