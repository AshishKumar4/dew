"""Diffusers' own `WanTransformer3DModel` (Diffusers 0.34.0, the cohort tests/fixtures/wan_transformer.tar.xz
was saved by, tools/diffusers_wan_reference.py) run on drawn latents, prompt states and cotangents: the
reference arm of tools/venv_reference.py's `wan-<case>` and `wan-bf16-<case>` sites (its protocol
tools/reference_worker.py), run in an environment of that Diffusers and torch, on the CPU.

    python tools/wan_worker.py wan-CASE         `IN OUT` lines on stdin, each `done`
    python tools/wan_worker.py wan-bf16-CASE

IN holds each draw's `latents` `[B, C, T, H, W]`, `context` `[B, L, D]`, `times` `[B]` and `probe`, a
cotangent of the flow. OUT gets `<draw>.reference.<leaf>` and `<draw>.truth.<leaf>`: for `wan-CASE`, the
case's `output` and the `gradients` of `sum(output * probe)` with respect to the latents, the prompt states
and every parameter by sorted name, concatenated (`diffusers_wan_reference.walk`), in float32 and, on its
float64 twin under `torch_precision.widened`, in float64; for `wan-bf16-CASE`, the flow alone, the model read
in bfloat16 (its time embedder kept in float32, as Diffusers keeps it) and in float64.
"""

import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import torch
from reference_worker import serve

ARCHIVE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "wan_transformer.tar.xz"


def flows(model, inputs: dict[str, np.ndarray], dtype, *, gradients: bool) -> dict[str, np.ndarray]:
    """The flow at `dtype`, and with `gradients` those `diffusers_wan_reference.walk` gives, concatenated."""
    from diffusers_wan_reference import walk

    tensors = {name: torch.from_numpy(np.asarray(inputs[name])) for name in ("latents", "context", "times")}
    if not gradients:
        # A bfloat16 call's times stay float32, its time embedder's.
        times = tensors["times"].to(torch.float32 if dtype == torch.bfloat16 else dtype)
        with torch.no_grad():
            output = model(tensors["latents"].to(dtype), times, tensors["context"].to(dtype),
                           return_dict=False)[0]
        return {"output": output.to(torch.float64).numpy()}
    walked = walk(model, tensors, torch.from_numpy(np.asarray(inputs["probe"])), dtype)
    parameters = sorted(key for key in walked if key.startswith("grad_param."))
    leaves = [walked["grad_latents"], walked["grad_context"], *(walked[key] for key in parameters)]
    return {"output": walked["output"], "gradients": np.concatenate([np.ravel(leaf) for leaf in leaves])}


def run(site: str, directory: Path, cases: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every draw's flow (and gradients), at the site's narrow precision and in float64."""
    from diffusers import WanTransformer3DModel

    from tools.torch_precision import float64_twin, widened

    half = site.startswith("wan-bf16-")
    case = site.removeprefix("wan-bf16-" if half else "wan-")
    checkpoint = directory / case / "transformer"
    narrow = WanTransformer3DModel.from_pretrained(checkpoint,
                                                   torch_dtype=torch.bfloat16 if half else torch.float32)
    wide = float64_twin(WanTransformer3DModel.from_pretrained(checkpoint, torch_dtype=torch.float32).eval())
    out = {}
    for name in sorted({key.split(".", 1)[0] for key in cases}):
        inputs = {key.removeprefix(f"{name}."): value for key, value in cases.items()
                  if key.startswith(f"{name}.")}
        narrowed = flows(narrow.eval(), inputs, torch.bfloat16 if half else torch.float32, gradients=not half)
        for leaf, value in narrowed.items():
            out[f"{name}.reference.{leaf}"] = value.astype(np.float32)
        with widened():
            for leaf, value in flows(wide, inputs, torch.float64, gradients=not half).items():
                out[f"{name}.truth.{leaf}"] = value
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2 or not sys.argv[1].startswith("wan-"):
        raise SystemExit(__doc__)
    sys.path.insert(0, str(ARCHIVE.parents[2]))
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as scratch, tarfile.open(ARCHIVE) as archive:
        archive.extractall(scratch, filter="data")
        serve(lambda cases: run(sys.argv[1], Path(scratch), cases))
