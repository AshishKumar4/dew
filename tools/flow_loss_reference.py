#!/usr/bin/env python3
"""Write tests/fixtures/flow/loss.npz: Diffusers' flow-matching training
loss and its gradient on the draws Dew's DiffusionObjective makes.

The loss is the Flux DreamBooth script's (examples/dreambooth/
train_dreambooth_flux.py at Diffusers v0.34.0): its statements from the
noising to the reduction, `noisy_model_input = (1.0 - sigmas) * ...`,
`weighting = compute_loss_weighting_for_sd3(...)`, `target = noise -
model_input` and the two-line mean, read out of the published file's
`main` and executed as written, with the installed Diffusers 0.34.0's
`compute_loss_weighting_for_sd3` at its "logit_normal" scheme. Between the
noising and the weighting a stand-in velocity network, the same function as
the test's, takes the place of the transformer call, at the script's
timestep (sigma times 1000); the script's latent packing and the model call
itself are the transformer's and not the loss's.

The draws are DiffusionObjective.loss's from `jax.random.key(KEY)`: its
split's third key draws the training times, as `presets.Flow(shift=...)`'s
schedule draws them, and the fourth the noise; sigma is the schedule's
noise rate at those times, so the script runs on Dew's own sigmas. The
static shift that maps the drawn time to sigma is held on its own, against
`FlowMatchEulerDiscreteScheduler(shift=3)`'s published grid, which lands
here too. Each case runs in float32 and in float64, `.float()` reading the
run's precision.

    PYTHONPATH=src python tools/flow_loss_reference.py
"""

from __future__ import annotations

import ast
import json
import types
import urllib.request
from pathlib import Path

import jax
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_loss_weighting_for_sd3

from dew.diffusion import presets

SCRIPT = ("https://raw.githubusercontent.com/huggingface/diffusers/50dea89dc6036e71a00bc3d57ac062a80206d9eb/"
          "examples/dreambooth/train_dreambooth_flux.py")
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "flow" / "loss.npz"
SHAPE, KEY, SHIFTS = (6, 4, 4, 3), 9, (1.0, 3.0)
STATEMENTS = ("noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise",
              "weighting = compute_loss_weighting_for_sd3(", "target = noise - model_input",
              "loss = torch.mean(", "loss = loss.mean()")


def published() -> tuple[str, str]:
    """The script's statements before the model call and after it."""
    text = urllib.request.urlopen(SCRIPT).read().decode()
    main = next(node for node in ast.parse(text).body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    compound = (ast.For, ast.If, ast.With, ast.While, ast.Try)
    found = {}
    for node in ast.walk(main):
        if isinstance(node, ast.stmt) and not isinstance(node, compound):
            source = ast.get_source_segment(text, node) or ""
            for start in STATEMENTS:
                if source.startswith(start) and start not in found:
                    found[start] = source
    missing = [start for start in STATEMENTS if start not in found]
    if missing:
        raise SystemExit(f"the script no longer holds {missing}")
    return found[STATEMENTS[0]], "\n".join(found[start] for start in STATEMENTS[1:])


def velocity(weights, x, timestep):
    """The stand-in velocity network the test runs too: four terms, each
    with a weight per entry, at the model's timestep."""
    time = timestep.reshape(-1, 1, 1, 1) / 1000
    return (torch.tanh(x) * weights[0] + torch.sin(2 * time) * x * weights[1]
            + torch.cos(x) * time * weights[2] + time * weights[3])


class Precision:
    """`.float()` reads as the run's own precision."""

    def __init__(self, dtype):
        self.dtype = dtype

    def __enter__(self):
        self.kept = torch.Tensor.float
        torch.Tensor.float = lambda tensor, *args, **kwargs: tensor.to(self.dtype)
        return self

    def __exit__(self, *_):
        torch.Tensor.float = self.kept


def main() -> None:
    noising, loss_lines = published()
    generator = np.random.default_rng(13)
    pixels = generator.integers(0, 256, SHAPE, dtype=np.uint8)
    x = (pixels.astype(np.float32) - np.float32(127.5)) / np.float32(127.5)
    weights = (generator.standard_normal((4, *SHAPE[1:])) * 0.4).astype(np.float32)
    keys = jax.random.split(jax.random.key(KEY), 5)
    noise = np.array(jax.random.normal(keys[3], SHAPE))
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
    arrays = {"pixels": pixels, "weights": weights, "key": np.asarray(KEY), "shifts": np.asarray(SHIFTS),
              "noising": np.asarray(noising), "loss_lines": np.asarray(loss_lines),
              # The scheduler's grid before its static shift, and after it.
              "grid": np.linspace(1, 1000, 1000, dtype=np.float32)[::-1] / 1000,
              "grid_shifted_3": scheduler.sigmas.numpy()}
    for index, shift in enumerate(SHIFTS):
        schedule = presets.Flow(shift=shift)().schedule
        t = schedule.sample_t(keys[2], SHAPE[0])
        sigmas = np.array(schedule.rates(t)[1])
        arrays[f"sigmas_{index}"] = sigmas
        for dtype, suffix in ((torch.float32, ""), (torch.float64, "_f64")):
            parameters = torch.tensor(weights, dtype=dtype, requires_grad=True)
            scope = {"torch": torch, "compute_loss_weighting_for_sd3": compute_loss_weighting_for_sd3,
                     "args": types.SimpleNamespace(weighting_scheme="logit_normal"),
                     "model_input": torch.tensor(x, dtype=dtype), "noise": torch.tensor(noise, dtype=dtype),
                     "sigmas": torch.tensor(sigmas, dtype=dtype).reshape(-1, 1, 1, 1)}
            with Precision(dtype):
                exec(noising, scope)
                scope["model_pred"] = velocity(parameters, scope["noisy_model_input"],
                                               scope["sigmas"].flatten() * 1000)
                exec(loss_lines, scope)
                scope["loss"].backward()
            arrays[f"loss_{index}{suffix}"] = scope["loss"].detach().double().numpy()
            arrays[f"grad_{index}{suffix}"] = parameters.grad.double().numpy()
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURE, **arrays)
    losses = {f"shift {shift}": float(arrays[f"loss_{index}_f64"]) for index, shift in enumerate(SHIFTS)}
    print(f"{FIXTURE}: {json.dumps(losses)}")


if __name__ == "__main__":
    main()
