"""One reference VJP of a torch model, as the autoencoder generators record it.

The diffusers VAE generators (`diffusers_dc_ae_reference.py`,
`diffusers_wan_vae_reference.py`, `diffusers_qwen_image_vae_reference.py`)
take each of their encode and decode the same way: probes drawn for the
outputs, and the gradient of the probed sum in the inputs and in every named
parameter, so a test checks Dew's input and parameter gradients against
torch's. Each walks its model twice (`twice`), in float32 and in float64 over
the same draws (`Draws`), so a test holds Dew to the float32 reference's own
distance from float64 (tests/reference_error.py).
"""

from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np
import torch
from diffusers_wan_reference import float64


class Draws:
    """Named random draws from one generator, replayed by name: a walk that draws its inputs
    and probes through it draws them anew the first time and reads them back after, so a
    model walked in float32 and then in float64 (`twice`) sees the same values, each at the
    walk's dtype."""

    def __init__(self, generator: torch.Generator):
        self.generator, self.held = generator, {}

    def __call__(self, name: str, shape, dtype: torch.dtype, *, uniform: bool = False) -> torch.Tensor:
        if name not in self.held:
            draw = torch.rand if uniform else torch.randn
            self.held[name] = draw(tuple(shape), generator=self.generator)
        return self.held[name].to(dtype)


def twice(walk: Callable[[torch.nn.Module, torch.dtype], dict[str, np.ndarray]],
          model: torch.nn.Module) -> dict[str, np.ndarray]:
    """`walk(model, dtype)` in float32 and, a float64 copy of `model` built and run under
    `float64`, in float64 under `fp64.` keys."""
    narrow = walk(model, torch.float32)
    with float64():
        wide = walk(copy.deepcopy(model).double(), torch.float64)
    return {**narrow, **{f"fp64.{key}": value for key, value in wide.items()}}


def recorded_vjp(side: str, outputs: dict[str, torch.Tensor], inputs: dict[str, torch.Tensor],
                 named: list[tuple[str, torch.Tensor]], draw: Draws) -> dict[str, np.ndarray]:
    """`side`'s (encode or decode) VJP: a standard normal probe drawn (`draw`) for each of
    `outputs`, in order, and the gradient of the sum of each output times its probe in each
    of `inputs` and each of the model's `named` parameters, zeros where the sum does not
    reach one.

    Keys: each probe under its output's name in `outputs`, an input's
    gradient as `{side}.grad_{input}`, a parameter's as `{side}.grad.{name}`.
    """
    probes = {name: draw(name, output.shape, output.dtype) for name, output in outputs.items()}
    loss = sum((output * probes[name]).sum() for name, output in outputs.items())
    every = [*inputs.values(), *(value for _, value in named)]
    found = torch.autograd.grad(loss, every, allow_unused=True)
    gradients = [torch.zeros_like(value) if gradient is None else gradient
                 for gradient, value in zip(found, every, strict=True)]
    return {**{name: probe.numpy() for name, probe in probes.items()},
            **{f"{side}.grad_{name}": gradient.numpy() for name, gradient in zip(inputs, gradients)},
            **{f"{side}.grad.{name}": gradient.numpy()
               for (name, _), gradient in zip(named, gradients[len(inputs):], strict=True)}}
