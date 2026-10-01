"""REPA-E's autoencoder regularizer and latent batch norm by its official
code, for tests/fixtures/repae.

- The KL: End2End-Diffusion/REPA-E's `DiagonalGaussianDistribution`
  (models/autoencoder.py, read at a pinned commit, the class extracted and
  run as published), summed over each latent and averaged over the batch
  as `ReconstructionLoss_Single_Stage` does, beside the L1 reconstruction,
  at configs/l1_lpips_kl_gan.yaml's weights.
- The latent normalization: `models/sit.py`'s `BatchNorm2d(eps=1e-4,
  momentum=0.1, affine=False)`, started from `init_bn`'s statistics, in
  training mode and then in evaluation mode.

    PYTHONPATH=src python tools/repae_reference.py
"""

from __future__ import annotations

import ast
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

AUTOENCODER = ("https://raw.githubusercontent.com/End2End-Diffusion/REPA-E/"
               "2ad4e9f69234c109497fb41d3e5e555de7b4b0de/models/autoencoder.py")
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "repae"
KL_WEIGHT, SHIFT, SCALE = 1e-6, 0.1, 0.8


def main() -> None:
    text = urllib.request.urlopen(AUTOENCODER).read().decode()
    scope = {"torch": torch, "np": np}
    for node in ast.parse(text).body:
        if isinstance(node, ast.ClassDef) and node.name == "DiagonalGaussianDistribution":
            exec(ast.get_source_segment(text, node), scope)
    generator = torch.Generator().manual_seed(0)
    moments = torch.randn(3, 8, 4, 4, generator=generator, dtype=torch.float64) * 2
    images = torch.rand(3, 3, 16, 16, generator=generator, dtype=torch.float64) * 2 - 1
    reconstruction = images + 0.1 * torch.randn(3, 3, 16, 16, generator=generator, dtype=torch.float64)
    kl = scope["DiagonalGaussianDistribution"](moments).kl()
    regularizer = F.l1_loss(images, reconstruction) + KL_WEIGHT * torch.sum(kl) / kl.shape[0]

    latents = torch.randn(3, 4, 4, 4, generator=generator, dtype=torch.float64) * 1.7 + 0.4
    norm = torch.nn.BatchNorm2d(4, eps=1e-4, momentum=0.1, affine=False).double()
    norm.running_mean = torch.full((4,), SHIFT, dtype=torch.float64)
    norm.running_var = torch.full((4,), 1 / SCALE, dtype=torch.float64).pow(2)
    trained = norm.train()(latents)
    evaluated = norm.eval()(latents)
    channels_last = (0, 2, 3, 1)
    FIXTURE.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURE / "regularizer.npz",
             moments=moments.permute(*channels_last).numpy(), images=images.permute(*channels_last).numpy(),
             reconstruction=reconstruction.permute(*channels_last).numpy(), regularizer=regularizer.numpy(),
             kl=(torch.sum(kl) / kl.shape[0]).numpy(), latents=latents.permute(*channels_last).numpy(),
             trained=trained.permute(*channels_last).numpy(), evaluated=evaluated.permute(*channels_last).numpy(),
             running_mean=norm.running_mean.numpy(), running_var=norm.running_var.numpy())
    print(f"{FIXTURE}: the regularizer and one training and one evaluation batch norm")


if __name__ == "__main__":
    main()
