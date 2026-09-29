"""REPA's and iREPA's projection losses by their official code, for tests/fixtures/repa.

- REPA: sihyun-yu/REPA's `build_mlp` (`models/sit.py`) projects the hidden
  tokens, and `SILoss.__call__`'s projection loss (`loss.py`) scores them
  against the encoder's features: the loop is run as published, extracted
  from the file at a pinned commit.
- iREPA: End2End-Diffusion/iREPA's `ProjectionLayer` ("conv", kernel 3) and
  `spatial_zscore` (`ldm/models/sit.py`, `ldm/utils.py`), the same way.
- The resize both apply to the encoder's input: torch's bicubic
  `F.interpolate`, from 16 to 14 pixels, as REPA's 256 to 224.

Each case lands its inputs, the projector's weights in Flax's layout, and
the loss in float64 and float32.

    PYTHONPATH=src python tools/repa_reference.py
"""

from __future__ import annotations

import ast
import json
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPA = "https://raw.githubusercontent.com/sihyun-yu/REPA/67f714503e3892f993844aab088ffc5791c92613/"
IREPA = "https://raw.githubusercontent.com/End2End-Diffusion/iREPA/99ad4ac234efe8de52ce157120f72856e836d09f/ldm/"
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "repa"
BATCH, SIDE, WIDTH, FEATURES, HIDDEN = 2, 4, 12, 10, 16
GAMMA = 0.6


def extracted(url: str, names: tuple[str, ...], scope: dict) -> dict:
    """The named top-level definitions of the file at `url`, run as published."""
    text = urllib.request.urlopen(url).read().decode()
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign):
            defined = {target.id for target in node.targets if isinstance(target, ast.Name)}
        else:
            defined = {node.name} if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else set()
        if defined & set(names):
            exec(ast.get_source_segment(text, node), scope)
    return scope


def projection_loss(zs, zs_tilde) -> torch.Tensor:
    """`SILoss.__call__`'s projection-loss loop, verbatim, over one encoder."""
    text = urllib.request.urlopen(REPA + "loss.py").read().decode()
    start = text.index("        proj_loss = 0.")
    end = text.index("        return denoising_loss, proj_loss")
    body = "\n".join(line[8:] for line in text[start:end].splitlines())
    scope = {"torch": torch, "zs": zs, "zs_tilde": zs_tilde,
             "mean_flat": lambda x: torch.mean(x, dim=list(range(1, len(x.size()))))}
    exec(body, scope)
    return scope["proj_loss"]


def dense(layer: torch.nn.Linear) -> dict:
    return {"kernel": layer.weight.detach().double().numpy().T, "bias": layer.bias.detach().double().numpy()}


def main() -> None:
    FIXTURE.mkdir(parents=True, exist_ok=True)
    repa = extracted(REPA + "models/sit.py", ("build_mlp",), {"nn": torch.nn})
    irepa = extracted(IREPA + "models/sit.py", ("build_mlp", "ProjectionLayer", "ALL_PROJECTION_LAYER_TYPES"),
                      {"nn": torch.nn, "math": __import__("math")})
    zscore = extracted(IREPA + "utils.py", ("spatial_zscore",), {"torch": torch})["spatial_zscore"]
    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(BATCH, SIDE * SIDE, WIDTH, generator=generator)
    features = torch.randn(BATCH, SIDE * SIDE, FEATURES, generator=generator) + 0.7

    torch.manual_seed(1)
    mlp = repa["build_mlp"](WIDTH, HIDDEN, FEATURES)
    torch.manual_seed(2)
    conv = irepa["ProjectionLayer"]("conv", hidden_size=WIDTH, z_dim=FEATURES, proj_kwargs_kernel_size=3)
    arrays = {"hidden": hidden.numpy(), "features": features.numpy(),
              "settings": np.asarray(json.dumps({"width": HIDDEN, "gamma": GAMMA}))}
    for name, layer in zip(("mlp.Dense_0", "mlp.Dense_1", "mlp.Dense_2"), (mlp[0], mlp[2], mlp[4]), strict=True):
        for leaf, value in dense(layer).items():
            arrays[f"{name}.{leaf}"] = value
    weight = conv.projection_layer.weight.detach().double().numpy()
    arrays["conv.Conv_0.kernel"] = weight.transpose(2, 3, 1, 0)
    arrays["conv.Conv_0.bias"] = conv.projection_layer.bias.detach().double().numpy()

    for dtype, suffix in ((torch.float64, ""), (torch.float32, "32")):
        with torch.no_grad():
            h, z = hidden.to(dtype), features.to(dtype)
            arrays[f"repa{suffix}"] = projection_loss([z], [mlp.to(dtype)(h)]).double().numpy()
            arrays[f"irepa{suffix}"] = projection_loss(
                [zscore(z, alpha=GAMMA)], [conv.to(dtype)(h)]).double().numpy()
            image = torch.randn(BATCH, 3, 16, 16, generator=torch.Generator().manual_seed(3))
            arrays["image"] = image.numpy().transpose(0, 2, 3, 1)
            arrays[f"resized{suffix}"] = F.interpolate(image.to(dtype), 14, mode="bicubic").double().numpy().transpose(0, 2, 3, 1)
    np.savez(FIXTURE / "losses.npz", **arrays)
    print(f"{FIXTURE}: REPA and iREPA projection losses, one bicubic resize")


if __name__ == "__main__":
    main()
