#!/usr/bin/env python3
"""Write P2's and min-SNR's loss weights, as their authors compute them, to
tests/fixtures/weighting/weights.npz.

Each repo's `guided_diffusion` package (gaussian_diffusion.py with the nn.py
and losses.py it imports) is fetched at a pinned commit and imported as it
is: jychoi118/P2-weighting (Choi et al. 2022) and
TiankaiHang/Min-SNR-Diffusion-Training (Hang et al. 2023). Its own
`GaussianDiffusion.training_losses` then runs at every timestep of a beta
table on a zero clean sample with zero noise and a stand-in model that
predicts one everywhere: every target is zero there, so the squared error
is exactly one and `terms["mse"]` is the weight the authors apply at that
step.

The cases: P2 on an epsilon model at (k, gamma) = (1, 1), (1, 0.5) and
(2, 1), and min-SNR-5 on epsilon (`min_snr_5`), v (`vmin_snr_5`) and x_0
(`min_snr_5` with START_X) models, over improved-diffusion's linear table
(tests/fixtures/schedules/betas.npz) and over a cosine table whose last
beta is one, whose zero terminal SNR min-SNR's code weights as one.

Each case runs as written, in the fp32 the authors' `_extract_into_tensor`
casts to, and again with that one helper keeping float64, which gives the
exact weights both fp32 runs are measured from (tests/reference_error.py).

Run with Torch 2.14.0 CPU and NumPy:
  python tools/weighting_reference.py
"""

import argparse
import importlib
import math
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import torch

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
REPOS = {
    "p2": ("jychoi118/P2-weighting", "3da0947ac350072e457c211401218175bc94e137"),
    "min_snr": ("TiankaiHang/Min-SNR-Diffusion-Training", "5189997d6bbde5a1301bcf31a54d2b3589f7309c"),
}
CASES = {
    "p2": (("EPSILON", {"p2_k": 1, "p2_gamma": 1}), ("EPSILON", {"p2_k": 1, "p2_gamma": 0.5}),
           ("EPSILON", {"p2_k": 2, "p2_gamma": 1})),
    "min_snr": (("EPSILON", {"mse_loss_weight_type": "min_snr_5"}),
                ("VELOCITY", {"mse_loss_weight_type": "vmin_snr_5"}),
                ("START_X", {"mse_loss_weight_type": "min_snr_5"})),
}


def zero_terminal_cosine(steps: int = 1000) -> np.ndarray:
    """Improved-diffusion's cosine betas with the last one at 1: a table
    that reaches zero SNR, which min-SNR's code weights specially."""
    def alpha_bar(t):
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    betas = [min(1 - alpha_bar((i + 1) / steps) / alpha_bar(i / steps), 0.999) for i in range(steps)]
    betas[-1] = 1.0
    return np.array(betas, np.float64)


def guided_diffusion(name: str, directory: Path):
    """The repo's own gaussian_diffusion module, imported from its files."""
    repo, commit = REPOS[name]
    package = directory / f"guided_diffusion_{name}"
    package.mkdir()
    (package / "__init__.py").write_text("")
    for file in ("gaussian_diffusion.py", "nn.py", "losses.py"):
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/guided_diffusion/{file}"
        with urllib.request.urlopen(url, timeout=60) as response:
            (package / file).write_bytes(response.read())
    sys.path.insert(0, str(directory))
    return importlib.import_module(f"guided_diffusion_{name}.gaussian_diffusion")


def weights(module, betas: np.ndarray, mean: str, options: dict, exact: bool) -> np.ndarray:
    """The authors' weight at every step of `betas`, read off `terms["mse"]`."""
    original = module._extract_into_tensor
    if exact:
        def keep_float64(array, timesteps, shape):
            result = torch.from_numpy(array)[timesteps]
            while result.dim() < len(shape):
                result = result[..., None]
            return result.expand(shape)
        module._extract_into_tensor = keep_float64
    try:
        diffusion = module.GaussianDiffusion(
            betas=betas, model_mean_type=module.ModelMeanType[mean],
            model_var_type=module.ModelVarType.FIXED_SMALL, loss_type=module.LossType.MSE, **options)
        dtype = torch.float64 if exact else torch.float32
        steps = torch.arange(len(betas))
        zero = torch.zeros((len(betas), 1, 1, 1), dtype=dtype)
        terms = diffusion.training_losses(lambda x, t, **_: torch.ones_like(x), zero, steps, noise=zero)
        return terms["mse"].double().numpy()
    finally:
        module._extract_into_tensor = original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=FIXTURES / "weighting" / "weights.npz")
    out = parser.parse_args().out
    tables = {"linear": np.load(FIXTURES / "schedules" / "betas.npz")["improved_diffusion_linear_1000"],
              "zero_terminal_cosine": zero_terminal_cosine()}
    saved = {f"betas_{name}": table for name, table in tables.items()}
    with tempfile.TemporaryDirectory() as directory:
        for name, cases in CASES.items():
            module = guided_diffusion(name, Path(directory))
            for mean, options in cases:
                label = "_".join([name, mean.lower(), *(f"{key}{value}" for key, value in options.items())])
                for table, betas in tables.items():
                    key = f"{label}_{table}".replace(".", "p")
                    saved[key] = weights(module, betas, mean, options, exact=False).astype(np.float32)
                    saved[f"{key}_f64"] = weights(module, betas, mean, options, exact=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **saved)
    print(out, len(saved), "arrays")


if __name__ == "__main__":
    main()
