"""Rounded timestep frequencies, replayed draws and tiny text encoders for Diffusers fixtures."""

from __future__ import annotations

import contextlib
import json
import math
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np
import torch


def frequency_exponent(half: int, shift: float, max_period: float) -> torch.Tensor:
    """`get_timestep_embedding`'s float32 exponent, its own arithmetic."""
    exponent = -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32)
    return exponent / (half - shift)


def torch_exp_disagreements(half: int) -> int:
    """How many of a `half`-entry table's frequencies torch's float32 `exp`
    rounds differently from the float64 one; the attribution a record carries."""
    exponent = frequency_exponent(half, 0, 10000)
    return int((torch.exp(exponent) != torch.exp(exponent.double()).float()).sum())


def rounded_timestep_embedding(timesteps, embedding_dim, flip_sin_to_cos=False,
                               downscale_freq_shift=1, scale=1, max_period=10000):
    """The source's `get_timestep_embedding` with its exponential taken in
    float64 and rounded once to float32.

    Torch's float32 `exp` is off by one unit in the last place at some of the
    entries, and one ulp of a frequency is one ulp of the 3500-radian angle a
    distilled guidance embeds; the rounded table is the one Dew builds on the
    host, so a walk over it holds the rest of the source to the suite's bound
    with nothing else altered.
    """
    half = embedding_dim // 2
    table = torch.exp(frequency_exponent(half, downscale_freq_shift, max_period).double()).float()
    angle = scale * (timesteps[:, None].float() * table[None, :])
    embedded = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)
    if flip_sin_to_cos:
        embedded = torch.cat([embedded[:, half:], embedded[:, :half]], dim=-1)
    return embedded


@contextlib.contextmanager
def rounded_frequency_table():
    from diffusers.models import embeddings

    original = embeddings.get_timestep_embedding
    embeddings.get_timestep_embedding = rounded_timestep_embedding
    try:
        yield
    finally:
        embeddings.get_timestep_embedding = original


def step_noise(shape: tuple[int, ...], count: int,
               dtype: type[np.float32] | type[np.float64]) -> list[np.ndarray]:
    """Draw float32 normals under the walk's folded key, then store in `dtype`."""
    import jax
    import jax.numpy as jnp

    key = jax.random.PRNGKey(0)
    return [np.asarray(jax.random.normal(jax.random.fold_in(key, index), shape, jnp.float32),
                       dtype) for index in range(count)]


@contextlib.contextmanager
def fed_noise(module: ModuleType, noises: list[np.ndarray]) -> Iterator[list[int]]:
    """Replay draws in the requested dtype and count successful draws.

    An omitted dtype keeps the input array's dtype. Modules without
    `randn_tensor` take no draws; the original function is restored on exit.
    """
    taken: list[int] = []

    def draw(shape, generator=None, device=None, dtype=None, layout=None):
        noise = torch.tensor(noises[len(taken)], dtype=dtype)
        assert tuple(noise.shape) == tuple(shape), (noise.shape, shape)
        taken.append(1)
        return noise

    if getattr(module, "randn_tensor", None) is None:
        yield taken
        return
    with patch.object(module, "randn_tensor", draw):
        yield taken


def tiny_qwen3_encoder(source: Path, tok, *, hidden_size: int, layers: int, seed: int):
    """The published Qwen3 text encoder's config under `source`, narrowed to `hidden_size` wide and
    `layers` deep with two query heads over one key head, every field it does not name the
    release's own, its weights drawn at `seed + 5` and perturbed at `seed + 6`."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = json.loads((source / "text_encoder" / "config.json").read_text())
    config.pop("torch_dtype", None)
    config.pop("dtype", None)
    config.update(hidden_size=hidden_size, intermediate_size=32, num_hidden_layers=layers,
                  layer_types=["full_attention"] * layers, max_window_layers=layers, num_attention_heads=2,
                  num_key_value_heads=1, head_dim=8, vocab_size=len(tok), max_position_embeddings=1024,
                  bos_token_id=None, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    torch.manual_seed(seed + 5)
    model = Qwen3ForCausalLM(Qwen3Config(**config)).eval()
    generator = torch.Generator().manual_seed(seed + 6)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=generator) * 0.05)
    return model
