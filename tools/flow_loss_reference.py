#!/usr/bin/env python3
"""Write tests/fixtures/flow/loss.npz: the inputs on which the flow-matching
test runs Diffusers' training loss against Dew's DiffusionObjective, and
`published_loss`, the loss it runs.

The loss is the Flux DreamBooth script's (examples/dreambooth/
train_dreambooth_flux.py at Diffusers v0.34.0): its statements from the
noising to the reduction, `noisy_model_input = (1.0 - sigmas) * ...`,
`weighting = compute_loss_weighting_for_sd3(...)`, `target = noise -
model_input` and the two-line mean, read out of the published file's
`main` and recorded here, executed as written with the installed Diffusers
0.34.0's `compute_loss_weighting_for_sd3` at its "logit_normal" scheme.
Between the noising and the weighting a stand-in velocity network, the
same function as the test's, takes the place of the transformer call, at
the script's timestep (sigma times 1000); the script's latent packing and
the model call itself are the transformer's and not the loss's. The test
draws the times and noise per seed and hands them to both sides.

The static shift that maps the drawn time to sigma is held on its own,
against `FlowMatchEulerDiscreteScheduler(shift=3)`'s published grid, which
lands here too.

    PYTHONPATH=src python tools/flow_loss_reference.py
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_loss_weighting_for_sd3
from reference_source import fetched, selected, simple_statements
from torch_precision import float_as, replaced

SCRIPT = ("https://raw.githubusercontent.com/huggingface/diffusers/50dea89dc6036e71a00bc3d57ac062a80206d9eb/"
          "examples/dreambooth/train_dreambooth_flux.py")
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "flow" / "loss.npz"
SHAPE, SHIFTS = (6, 4, 4, 3), (1.0, 3.0)
STATEMENTS = ("noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise",
              "weighting = compute_loss_weighting_for_sd3(", "target = noise - model_input",
              "loss = torch.mean(", "loss = loss.mean()")


def published() -> tuple[str, str]:
    """The script's statements before the model call and after it."""
    text = fetched(SCRIPT)
    [main] = selected(text, {"main"}, filename=SCRIPT)
    before, *after = simple_statements(text, main, STATEMENTS)
    return before, "\n".join(after)


def velocity(weights, x, timestep):
    """The stand-in velocity network the test runs too: four terms, each
    with a weight per entry, at the model's timestep."""
    time = timestep.reshape(-1, 1, 1, 1) / 1000
    return (torch.tanh(x) * weights[0] + torch.sin(2 * time) * x * weights[1]
            + torch.cos(x) * time * weights[2] + time * weights[3])


def published_loss(statements: tuple[str, str], weights, pixels, noise, sigmas, dtype) -> dict:
    """The script's loss statements, as published, at `dtype` on the stand-in velocity network: the loss
    and its gradient in the network's weights, from the pixels, noise and sigmas it is handed."""
    noising, loss_lines = statements
    x = (np.asarray(pixels).astype(np.float32) - np.float32(127.5)) / np.float32(127.5)
    parameters = torch.tensor(np.asarray(weights), dtype=dtype, requires_grad=True)
    scope = {"torch": torch, "compute_loss_weighting_for_sd3": compute_loss_weighting_for_sd3,
             "args": types.SimpleNamespace(weighting_scheme="logit_normal"),
             "model_input": torch.tensor(x, dtype=dtype),
             "noise": torch.tensor(np.asarray(noise), dtype=dtype),
             "sigmas": torch.tensor(np.asarray(sigmas), dtype=dtype).reshape(-1, 1, 1, 1)}
    with replaced(float_as(dtype)):
        exec(noising, scope)
        scope["model_pred"] = velocity(parameters, scope["noisy_model_input"],
                                       scope["sigmas"].flatten() * 1000)
        exec(loss_lines, scope)
        scope["loss"].backward()
    return {"loss": scope["loss"].detach().numpy(), "gradient": parameters.grad.numpy()}


def main() -> None:
    noising, loss_lines = published()
    generator = np.random.default_rng(13)
    pixels = generator.integers(0, 256, SHAPE, dtype=np.uint8)
    weights = (generator.standard_normal((4, *SHAPE[1:])) * 0.4).astype(np.float32)
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURE, pixels=pixels, weights=weights, shifts=np.asarray(SHIFTS), noising=np.asarray(noising),
             loss_lines=np.asarray(loss_lines),
             # The scheduler's grid before its static shift, and after it.
             grid=np.linspace(1, 1000, 1000, dtype=np.float32)[::-1] / 1000,
             grid_shifted_3=scheduler.sigmas.numpy())
    print(FIXTURE)

if __name__ == "__main__":
    main()
