"""REPA's and iREPA's projection losses by their official code, for tests/fixtures/repa.

- REPA: sihyun-yu/REPA's `build_mlp` (`models/sit.py`) projects the hidden
  tokens, and `SILoss.__call__`'s projection loss (`loss.py`) scores them
  against the encoder's features: the loop is run as published, extracted
  from the file at a pinned commit.
- iREPA: End2End-Diffusion/iREPA's `ProjectionLayer` ("conv", kernel 3) and
  `spatial_zscore` (`ldm/models/sit.py`, `ldm/utils.py`), the same way.

Each case lands its inputs, the projector's weights in Flax's layout, and
the loss in float64 and float32.

Two more fixtures:

- composed.npz: REPA's whole training loss as train.py takes it,
  `loss_mean + proj_loss_mean * proj_coeff` over `SILoss.__call__` (linear
  path, v prediction, uniform times) with `build_mlp`'s projector, the
  encoder's input from `preprocess_raw_image` (the "dinov1" branch: /255
  and ImageNet's normalization at the data's size), all run as published,
  and its gradient in every weight. The model is a stand-in, a patch
  network whose block's tokens are aligned, and so is the encoder, a patch
  projection of the preprocessed pixels. `SILoss`'s time and noise are the
  ones `DiffusionObjective.loss` draws from `jax.random.key(3)`: the split's
  third key for the times and its fourth for the noise. Beside the
  gradient, REPA's fp32 gradient's distance from its float64 one over
  ORDERS orders of the stand-in's hidden units (tests/residual_orders.py),
  for tests/reference_error.py's K-order rule; order 1 in float64 is
  checked to land on the float64 gradient.
- preprocessed.npz: 256-pixel images through `preprocess_raw_image`'s
  "dinov2" branch (/255, ImageNet's normalization, bicubic to 224) and
  iREPA's `DINOv2Encoder.preprocess`, then iREPA's `spatial_zscore` of
  them read as features, as published.

    PYTHONPATH=src python tools/repa_reference.py
"""

from __future__ import annotations

import ast
import copy
import json
import sys
import types
import urllib.request
from pathlib import Path

import jax
import numpy as np
import torch

REPA = "https://raw.githubusercontent.com/sihyun-yu/REPA/67f714503e3892f993844aab088ffc5791c92613/"
IREPA = "https://raw.githubusercontent.com/End2End-Diffusion/iREPA/99ad4ac234efe8de52ce157120f72856e836d09f/ldm/"
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "repa"
BATCH, SIDE, WIDTH, FEATURES, HIDDEN = 2, 4, 12, 10, 16
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from reference_error import ORDERS, distance  # noqa: E402
from residual_orders import orders  # noqa: E402

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
    np.savez(FIXTURE / "losses.npz", **arrays)
    print(f"{FIXTURE}: REPA and iREPA projection losses")


COMPOSED = {"batch": 2, "side": 8, "patch": 2, "width": 8, "projector": 16, "features": 6,
            "proj_coeff": 0.5, "key": 3}
# timm.data's ImageNet statistics, which REPA's train.py and iREPA's encoders import.
TIMM = {"IMAGENET_DEFAULT_MEAN": (0.485, 0.456, 0.406), "IMAGENET_DEFAULT_STD": (0.229, 0.224, 0.225)}


def patches(images: torch.Tensor, patch: int) -> torch.Tensor:
    """`[B, H, W, C]` as `[B, N, p * p * C]` tokens in raster order."""
    b, h, w, c = images.shape
    grid = images.reshape(b, h // patch, patch, w // patch, patch, c).permute(0, 1, 3, 2, 4, 5)
    return grid.reshape(b, (h // patch) * (w // patch), patch * patch * c)


def image(tokens: torch.Tensor, patch: int, side: int) -> torch.Tensor:
    b, _, _ = tokens.shape
    grid = tokens.reshape(b, side // patch, side // patch, patch, patch, -1).permute(0, 1, 3, 2, 4, 5)
    return grid.reshape(b, side, side, -1)


class Network(torch.nn.Module):
    """The stand-in model: tokens of the input's patches plus the time, one
    block whose output REPA aligns, and a head back to patches. It returns
    REPA's `(output, zs_tilde)`, the projector applied to the block's tokens,
    in `[B, C, H, W]` as SiT's are."""

    def __init__(self, weights: dict, projector: torch.nn.Module, patch: int, side: int):
        super().__init__()
        self.weights = torch.nn.ParameterDict({name: torch.nn.Parameter(value)
                                               for name, value in weights.items()})
        self.projector, self.patch, self.side = projector, patch, side

    def forward(self, x, t):
        tokens = patches(x.permute(0, 2, 3, 1), self.patch)
        hidden = tokens @ self.weights["embed"] + t.reshape(-1, 1, 1) * self.weights["time"]
        block = torch.tanh(hidden @ self.weights["block"])
        output = image(block @ self.weights["head"], self.patch, self.side).permute(0, 3, 1, 2)
        return output, [self.projector(block)]


def replayed(times: torch.Tensor, noise: torch.Tensor) -> types.SimpleNamespace:
    """torch whose `rand` is `times` and `randn_like` `noise`, SILoss's two draws."""
    return types.SimpleNamespace(**{**vars(torch), "rand": lambda *_, **__: times,
                                    "randn_like": lambda like: noise})


def composed() -> None:
    settings = COMPOSED
    b, side, patch, width = settings["batch"], settings["side"], settings["patch"], settings["width"]
    channels = 3
    generator = np.random.default_rng(11)
    pixels = generator.integers(0, 256, (b, side, side, channels), dtype=np.uint8)
    flat = patch * patch * channels
    weights = {"embed": generator.standard_normal((flat, width)) * 0.4,
               "time": generator.standard_normal(width),
               "block": generator.standard_normal((width, width)) * 0.5,
               "head": generator.standard_normal((width, flat)) * 0.4}
    encoder = generator.standard_normal((flat, settings["features"])) * 0.3
    repa = extracted(REPA + "models/sit.py", ("build_mlp",), {"nn": torch.nn})
    torch.manual_seed(3)
    projector = repa["build_mlp"](width, settings["projector"], settings["features"])
    from torchvision.transforms import Normalize
    preprocess = extracted(REPA + "train.py", ("preprocess_raw_image",),
                           {"torch": torch, "Normalize": Normalize, **TIMM})
    loss_scope = extracted(REPA + "loss.py", ("SILoss", "mean_flat"), {"torch": torch, "np": np})
    # DiffusionObjective.loss's draws: split(key, 5), times from the third key, noise the fourth.
    keys = jax.random.split(jax.random.key(settings["key"]), 5)
    times = np.array(jax.random.uniform(keys[2], (b,)))
    noise = np.array(jax.random.normal(keys[3], (b, side, side, channels)))
    text = urllib.request.urlopen(REPA + "train.py").read().decode()
    composition = next(line.strip() for line in text.splitlines()
                       if "loss = loss_mean + proj_loss_mean" in line)
    arrays = {"pixels": pixels, "encoder": encoder, "times": times, "noise": noise,
              "settings": np.asarray(json.dumps(settings)), "composition": np.asarray(composition)}
    arrays.update({f"weights/{name}": value for name, value in weights.items()})
    for name, layer in zip(("Dense_0", "Dense_1", "Dense_2"), (projector[0], projector[2], projector[4]),
                           strict=True):
        for leaf, value in dense(layer).items():
            arrays[f"projector/{name}/{leaf}"] = value

    def run(dtype, order: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """The composed loss and its gradient in `dtype`, the network's hidden
        units in `order` (embed's columns, time, block's rows and columns,
        head's rows, the projector's inputs), the gradient back in the
        fixture's order."""
        moved = {"embed": weights["embed"][:, order], "time": weights["time"][order],
                 "block": weights["block"][np.ix_(order, order)], "head": weights["head"][order]}
        layers = copy.deepcopy(projector).to(dtype)
        with torch.no_grad():
            layers[0].weight.copy_(layers[0].weight[:, torch.from_numpy(order)])
        # C order: a fancy index leaves embed's copy strided, which takes another BLAS path.
        network = Network({name: torch.as_tensor(np.ascontiguousarray(value), dtype=dtype)
                           for name, value in moved.items()}, layers, patch, side)
        raw = torch.as_tensor(pixels, dtype=dtype).permute(0, 3, 1, 2)
        prepared = preprocess["preprocess_raw_image"](raw, "dinov1").permute(0, 2, 3, 1)
        zs = [patches(prepared, patch) @ torch.as_tensor(encoder, dtype=dtype)]
        loss_scope["torch"] = replayed(torch.as_tensor(times, dtype=dtype).reshape(-1, 1, 1, 1),
                                       torch.as_tensor(noise, dtype=dtype).permute(0, 3, 1, 2))
        images = raw / 127.5 - 1
        denoising, projection = loss_scope["SILoss"]()(network, images, zs=zs)
        scope = {"loss": denoising, "proj_loss": projection, "args": types.SimpleNamespace(**settings)}
        exec(f"loss_mean = loss.mean()\nproj_loss_mean = proj_loss.mean()\n{composition}", scope)
        total = scope["loss"]
        total.backward()
        back = np.argsort(order)
        found = {name: parameter.grad.double().numpy() for name, parameter in network.weights.items()}
        grads = {"grad/embed": found["embed"][:, back], "grad/time": found["time"][back],
                 "grad/block": found["block"][np.ix_(back, back)], "grad/head": found["head"][back]}
        for name, layer in zip(("Dense_0", "Dense_1", "Dense_2"), (layers[0], layers[2], layers[4]),
                               strict=True):
            kernel = layer.weight.grad.double().numpy().T
            grads[f"grad/projector/{name}/kernel"] = kernel[back] if name == "Dense_0" else kernel
            grads[f"grad/projector/{name}/bias"] = layer.bias.grad.double().numpy()
        return total.detach().double().numpy(), grads

    drawn = orders(width, ORDERS, settings["key"])
    for dtype, suffix in ((torch.float64, "_f64"), (torch.float32, "")):
        arrays[f"loss{suffix}"], grads = run(dtype, drawn[0])
        arrays.update({f"{name}{suffix}": value for name, value in grads.items()})
    # The orders are a symmetry of REPA's computation too: order 1 in float64
    # lands on the fixture's float64 gradient within float64 rounding.
    _, moved = run(torch.float64, drawn[1])
    for name, value in moved.items():
        truth = arrays[f"{name}_f64"]
        assert np.abs(value - truth).max() <= 1e-12 * np.abs(truth).max(), name
    distances = {name: [distance(arrays[name], arrays[f"{name}_f64"])] for name in grads}
    for order in drawn[1:]:
        for name, value in run(torch.float32, order)[1].items():
            distances[name].append(distance(value, arrays[f"{name}_f64"]))
    arrays["orders"] = drawn.astype(np.uint8)
    arrays.update({f"orders/{name}": np.asarray(values) for name, values in distances.items()})
    np.savez(FIXTURE / "composed.npz", **arrays)
    print(f"{FIXTURE}: REPA's composed loss and its gradient")


def preprocessed() -> None:
    generator = np.random.default_rng(12)
    pixels = generator.integers(0, 256, (2, 256, 256, 3), dtype=np.uint8)
    from torchvision.transforms import Normalize
    repa = extracted(REPA + "train.py", ("preprocess_raw_image",),
                     {"torch": torch, "Normalize": Normalize, **TIMM})
    irepa = extracted(IREPA + "vision_encoder.py", ("DINOv2Encoder",),
                      {"torch": torch, "Normalize": Normalize, "VisionEncoder": object, "Dict": dict,
                       "Optional": object, **TIMM})
    zscore = extracted(IREPA + "utils.py", ("spatial_zscore",), {"torch": torch})["spatial_zscore"]
    arrays = {"pixels": pixels, "gamma": np.asarray(GAMMA)}
    for dtype, suffix in ((torch.float64, "_f64"), (torch.float32, "")):
        raw = torch.as_tensor(pixels, dtype=dtype).permute(0, 3, 1, 2)
        dinov2 = repa["preprocess_raw_image"](raw, "dinov2")
        encoder = types.SimpleNamespace(resolution=256)
        same = irepa["DINOv2Encoder"].preprocess(encoder, raw)
        assert torch.equal(same, dinov2), "REPA's and iREPA's DINOv2 preprocessing differ"
        features = dinov2.permute(0, 2, 3, 1).reshape(2, -1, 3)
        arrays[f"dinov2{suffix}"] = features.double().numpy()
        arrays[f"zscore{suffix}"] = zscore(features, alpha=GAMMA).double().numpy()
    np.savez(FIXTURE / "preprocessed.npz", **arrays)
    print(f"{FIXTURE}: REPA's and iREPA's DINOv2 preprocessing at 256 pixels")


if __name__ == "__main__":
    main()
    composed()
    preprocessed()
