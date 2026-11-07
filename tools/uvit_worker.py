"""Bao et al.'s own U-ViT (baofff/U-ViT at COMMIT, libs/uvit_t2i.py and libs/timm.py, checked against
their SHA-256) run on drawn weights and inputs, in float32 and float64: the reference arm of
tools/venv_reference.py's `uvit-<case>` sites (its protocol tools/reference_worker.py), run in an
environment holding torch, on the CPU.

    python tools/uvit_worker.py uvit-CASE    `IN OUT` lines on stdin, each `done`

IN holds each draw's U-ViT configuration, `<draw>.config.<argument>` each a 0-d array of `UViT`'s
constructor argument, its `<draw>.param.<name>` under U-ViT's state-dict names, its `image`
`[B, C, H, W]`, `context` `[B, T, D]`, `times` `[B]` in [0, 1] and `probe`, a cotangent of the output's
shape. OUT gets `<draw>.reference.<leaf>` and `<draw>.truth.<leaf>` in float32 and float64:
the `output` of U-ViT called as its continuous-time SDE calls it (on the time times 999), and the
`gradients` of `sum(output * probe)` with respect to the image, the text states and every parameter in
sorted name order, concatenated. The float64 run is the model's float64 twin (`torch_precision.float64_twin`)
under `torch_precision.widened`, which widens the sinusoids' float32 `arange` and `.float()`. Attention runs
U-ViT's "math" mode: its "flash" mode casts q, k and v to float32.
"""

from __future__ import annotations

import importlib
import sys

import numpy as np
import torch
from reference_source import package, published_files
from reference_worker import serve

REPO, COMMIT = "baofff/U-ViT", "ce551708dc9cde9818d2af7d84dfadfeb7bd9034"
FILES = {
    "libs/timm.py": "f1e414e5060038baba231ba30959bb1274839ebb1ba7649d3564288731a6babd",
    "libs/uvit_t2i.py": "2537800bd7251ea68f268b9125f5fbbfc64c1680bbc659767f37c737da468894",
}
def uvit():
    """U-ViT's libs at `COMMIT`, as the package `uvit_libs`, in its math attention."""
    package("uvit_libs", published_files(REPO, COMMIT, FILES) / "libs")
    t2i = importlib.import_module("uvit_libs.uvit_t2i")
    t2i.ATTENTION_MODE = "math"
    return t2i


def walked(model, inputs: dict[str, np.ndarray], dtype) -> dict[str, np.ndarray]:
    """One call at `dtype`: its output, and the gradients of `sum(output * probe)` with respect to the image,
    the text states and every parameter by sorted name, concatenated."""
    image, context = (torch.from_numpy(inputs[name]).to(dtype).requires_grad_()
                      for name in ("image", "context"))
    named = sorted(model.named_parameters())
    output = model(image, torch.from_numpy(inputs["times"]).to(dtype) * 999, context)
    probe = torch.from_numpy(inputs["probe"]).to(dtype)
    grads = torch.autograd.grad((output * probe).sum(), [image, context] + [p for _, p in named])
    return {"output": output.detach().numpy(),
            "gradients": np.concatenate([np.ravel(gradient.numpy()) for gradient in grads])}


def run(cases: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every draw's output and gradients under its configuration and weights, in float32 and float64."""
    from torch_precision import float64_twin, widened

    t2i = uvit()
    out = {}
    for name in sorted({key.split(".", 1)[0] for key in cases}):
        inputs = {key.removeprefix(f"{name}."): value for key, value in cases.items()
                  if key.startswith(f"{name}.")}
        config = {key.removeprefix("config."): value.item() for key, value in inputs.items()
                  if key.startswith("config.")}
        model = t2i.UViT(**config).eval()
        weights = {key.removeprefix("param."): torch.from_numpy(value) for key, value in inputs.items()
                   if key.startswith("param.")}
        model.load_state_dict(weights, strict=True)
        for leaf, value in walked(model, inputs, torch.float32).items():
            out[f"{name}.reference.{leaf}"] = value
        wide = float64_twin(model, lambda config=config: t2i.UViT(**config))
        with widened():
            for leaf, value in walked(wide, inputs, torch.float64).items():
                out[f"{name}.truth.{leaf}"] = value
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2 or not sys.argv[1].startswith("uvit-"):
        raise SystemExit(__doc__)
    torch.set_num_threads(1)
    serve(run)
