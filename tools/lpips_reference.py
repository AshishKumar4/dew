#!/usr/bin/env python3
"""Write tests/fixtures/lpips: REPA-E's perceptual loss, LPIPS on VGG16, run
as published.

REPA-E's loss/lpips.py (`LPIPS`, `ScalingLayer`, `NetLinLayer`, `vgg16`,
`normalize_tensor`, `spatial_average`) is fetched at a pinned commit, the
module copied from taming-transformers that REPA-E's
`ReconstructionLoss_Single_Stage` takes its perceptual loss from
(`PerceptualLoss("lpips")`, the mean of `LPIPS(input, target)`). Its
network is built as published, in evaluation mode; only where the weights
come from is replaced: `models.vgg16(weights=IMAGENET1K_V1)` and
`load_pretrained` read the weights given here instead of downloading.

- drawn.npz: weights drawn by `drawn_weights` (VGG16's convolutions He
  normal, the linear heads non-negative), whose per-tensor sums the
  fixture keeps so a test can check it drew the same; two 64-pixel image
  pairs in [-1, 1]; each pair's distance and the gradient of their mean
  with respect to the first image, in float32 and in float64.
- published.npz: the published weights, torchvision's vgg16-397923af.pth
  and v0.1's vgg.pth (MD5 d507d734..., which REPA-E's `get_ckpt_path`
  checks), downloaded to this tool's cache: four 64-pixel pairs' distances
  in float32 and float64.

    python tools/lpips_reference.py
"""

from __future__ import annotations

import ast
import hashlib
import urllib.request
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
from torchvision import models

REPAE = "https://raw.githubusercontent.com/End2End-Diffusion/REPA-E/2ad4e9f69234c109497fb41d3e5e555de7b4b0de/"
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "lpips"
CACHE = Path.home() / ".cache" / "dew" / "lpips"
PUBLISHED = {"vgg16-397923af.pth": ("https://download.pytorch.org/models/vgg16-397923af.pth",
                                    "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"),
             "vgg.pth": ("https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/"
                         "082bb24f84c091ea94de2867d34c4544f68e0963/lpips/weights/v0.1/vgg.pth",
                         "a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868")}
LPIPS_MD5 = "d507d7349b931f0638a25a48a722f98a"
CONVOLUTIONS = (0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28)
CHANNELS = (64, 128, 256, 512, 512)


def drawn_weights(seed: int = 0) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """VGG16's `features` and the five linear heads, drawn in torchvision's
    order from `np.random.default_rng(seed)`: each kernel He normal over its
    fan-in, each bias normal at 0.01, each head's weights the absolute value
    of a normal at 0.1, in float32."""
    rng = np.random.default_rng(seed)
    vgg, inputs = {}, 3
    for index, width in zip(CONVOLUTIONS, (64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512),
                            strict=True):
        vgg[f"features.{index}.weight"] = (rng.standard_normal((width, inputs, 3, 3), dtype=np.float32)
                                           * np.float32(np.sqrt(2 / (9 * inputs))))
        vgg[f"features.{index}.bias"] = rng.standard_normal(width, dtype=np.float32) * np.float32(0.01)
        inputs = width
    linear = {f"lin{stage}.model.1.weight": np.abs(rng.standard_normal((1, width, 1, 1), dtype=np.float32))
              * np.float32(0.1) for stage, width in enumerate(CHANNELS)}
    return vgg, linear


def lpips_class(vgg: dict[str, torch.Tensor], linear: dict[str, torch.Tensor]) -> type:
    """REPA-E's `LPIPS` with its module scope, reading these weights."""
    text = urllib.request.urlopen(REPAE + "loss/lpips.py").read().decode()
    names = {"LPIPS", "ScalingLayer", "NetLinLayer", "vgg16", "normalize_tensor", "spatial_average"}

    def vgg16(weights=None):
        network = models.vgg16(weights=None)
        network.features.load_state_dict({key.removeprefix("features."): value for key, value in vgg.items()})
        return network

    scope = {"torch": torch, "nn": torch.nn, "namedtuple": namedtuple,
             "models": type("models", (), {"vgg16": staticmethod(vgg16),
                                           "VGG16_Weights": models.VGG16_Weights}),
             "get_ckpt_path": None}
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(target, "id", "") in ("_LPIPS_MEAN", "_LPIPS_STD")
                                                  for target in node.targets):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "lpips.py", "exec"), scope)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "lpips.py", "exec"), scope)
    published = scope["LPIPS"]

    class Loaded(published):
        def load_pretrained(self):
            self.load_state_dict(linear, strict=False)

    return Loaded


def distances(vgg, linear, images, references, dtype, *, gradient: bool) -> dict[str, np.ndarray]:
    network = lpips_class({key: torch.as_tensor(value) for key, value in vgg.items()},
                          {key: torch.as_tensor(value) for key, value in linear.items()})().eval().to(dtype)
    first = torch.as_tensor(images, dtype=dtype).permute(0, 3, 1, 2).requires_grad_(gradient)
    second = torch.as_tensor(references, dtype=dtype).permute(0, 3, 1, 2)
    value = network(first, second)
    result = {"distance": value.detach().reshape(-1).numpy()}
    if gradient:
        value.mean().backward()
        result["gradient"] = first.grad.permute(0, 2, 3, 1).numpy()
    return result


def published_file(name: str) -> Path:
    url, digest = PUBLISHED[name]
    path = CACHE / name
    if not path.is_file():
        CACHE.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path
    return path


def images(rng, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Smooth images in [-1, 1], and each moved by noise and a shift."""
    base = rng.uniform(-1, 1, (count, 8, 8, 3))
    noise = 0.2 * rng.standard_normal((count, 64, 64, 3))
    smooth = np.clip(np.kron(base, np.ones((1, 8, 8, 1))) + noise, -1, 1)
    moved = np.clip(smooth + 0.3 * rng.standard_normal(smooth.shape) + 0.1, -1, 1)
    return smooth, moved


def main() -> None:
    # One thread, so the convolutions' backward sums in one order and the
    # fixture regenerates byte for byte.
    torch.set_num_threads(1)
    FIXTURE.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    vgg, linear = drawn_weights()
    first, second = images(rng, 2)
    arrays = {"images": first, "references": second,
              **{f"sum/{key}": np.float64(value.astype(np.float64).sum())
                 for key, value in {**vgg, **linear}.items()}}
    for dtype, tail in ((torch.float64, "_f64"), (torch.float32, "")):
        for key, value in distances(vgg, linear, first, second, dtype, gradient=True).items():
            arrays[f"{key}{tail}"] = value
    np.savez(FIXTURE / "drawn.npz", **arrays)

    assert hashlib.md5(published_file("vgg.pth").read_bytes()).hexdigest() == LPIPS_MD5
    state = torch.load(published_file("vgg16-397923af.pth"), map_location="cpu", weights_only=True)
    vgg = {key: value.numpy() for key, value in state.items() if key.startswith("features.")}
    linear = {key: value.numpy() for key, value in
              torch.load(published_file("vgg.pth"), map_location="cpu", weights_only=True).items()}
    first, second = images(rng, 4)
    published = {"images": first, "references": second}
    for dtype, tail in ((torch.float64, "_f64"), (torch.float32, "")):
        published[f"distance{tail}"] = distances(vgg, linear, first, second, dtype,
                                                 gradient=False)["distance"]
    np.savez(FIXTURE / "published.npz", **published)
    print(f"{FIXTURE}: drawn {arrays['distance_f64']}, published {published['distance_f64']}")


if __name__ == "__main__":
    main()
