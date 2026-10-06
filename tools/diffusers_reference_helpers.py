"""Rounded timestep frequencies and replayed draws for Diffusers fixtures."""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator
from types import ModuleType
from unittest.mock import patch

import numpy as np
import torch


def frequency_exponent(half: int, shift: float, max_period: float) -> torch.Tensor:
    """`get_timestep_embedding`'s float32 exponent, its own arithmetic."""
    exponent = -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32)
    return exponent / (half - shift)


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
