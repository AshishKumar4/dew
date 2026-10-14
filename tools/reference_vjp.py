"""One reference VJP of a torch model, as the autoencoder generators record it.

The diffusers VAE generators (`diffusers_dc_ae_reference.py`,
`diffusers_wan_vae_reference.py`, `diffusers_qwen_image_vae_reference.py`)
take each of their encode and decode the same way: probes drawn for the
outputs, and the gradient of the probed sum in the inputs and in every named
parameter, so a test checks Dew's input and parameter gradients against
torch's.
"""

from __future__ import annotations

import numpy as np
import torch


def recorded_vjp(side: str, outputs: dict[str, torch.Tensor], inputs: dict[str, torch.Tensor],
                 named: list[tuple[str, torch.Tensor]], generator: torch.Generator) -> dict[str, np.ndarray]:
    """`side`'s (encode or decode) VJP: a standard normal probe drawn from
    `generator` for each of `outputs`, in order, and the gradient of the sum
    of each output times its probe in each of `inputs` and each of the
    model's `named` parameters, zeros where the sum does not reach one.

    Keys: each probe under its output's name in `outputs`, an input's
    gradient as `{side}.grad_{input}`, a parameter's as `{side}.grad.{name}`.
    """
    probes = {name: torch.randn(output.shape, generator=generator) for name, output in outputs.items()}
    loss = sum((output * probes[name]).sum() for name, output in outputs.items())
    every = [*inputs.values(), *(value for _, value in named)]
    found = torch.autograd.grad(loss, every, allow_unused=True)
    gradients = [torch.zeros_like(value) if gradient is None else gradient
                 for gradient, value in zip(found, every, strict=True)]
    return {**{name: probe.numpy() for name, probe in probes.items()},
            **{f"{side}.grad_{name}": gradient.numpy() for name, gradient in zip(inputs, gradients)},
            **{f"{side}.grad.{name}": gradient.numpy()
               for (name, _), gradient in zip(named, gradients[len(inputs):], strict=True)}}
