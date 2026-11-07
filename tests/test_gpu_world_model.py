"""Dreamer 4's tokenizer trained as tutorial 12 trains it, on a GPU: the scale at which its reconstruction is
judged takes minutes there and hours on a CPU."""

import itertools

import jax
import numpy as np
import optax
import pytest

from dew.data.dataset import Loader
from dew.objectives.world_model import CausalTokenizer, TokenizerObjective
from dew.training import Trainer

pytestmark = pytest.mark.gpu(reason="tutorial 12's tokenizer at its own scale, 8,000 updates of 32 clips")


def bouncing(rows: int, frames: int, size: int, seed: int) -> np.ndarray:
    """Videos `[rows, frames, size, size, 1]` in [-1, 1] of two balls bouncing off the walls, as tutorial 12
    draws them."""
    rng = np.random.default_rng(seed)
    grid = np.stack(np.meshgrid(np.arange(size), np.arange(size), indexing="ij"), -1) + 0.5
    videos = np.full((rows, frames, size, size, 1), -1.0, np.float32)
    for video in videos:
        position, velocity = rng.uniform(3, size - 3, (2, 2)), rng.uniform(-1.5, 1.5, (2, 2))
        for frame in video:
            for ball in range(2):
                frame[np.sum(np.square(grid - position[ball]), -1) <= 2.5 ** 2, 0] = 1.0
            position = position + velocity
            velocity = np.where((position < 2.5) | (position > size - 2.5), -velocity, velocity)
            position = np.clip(position, 2.5, size - 2.5)
    return videos


def test_tutorial_12s_tokenizer_reconstructs_the_balls_it_was_trained_on():
    """Tutorial 12's tokenizer, trained as the tutorial trains it (2,048 clips of 12 frames, 32 clips an
    update, Adam 3e-4) for 8,000 updates, reconstructs 8 held-out clips at under a quarter of the mean
    frame's squared error. Its tokens know their places by Dreamer 4's rotary positions (`positions`, row
    4669): by learned vectors alone, as before, it reconstructed the mean frame (an H100 at fdd1da52c9:
    squared error 0.1406 against the mean frame's 0.1417, where rotary positions reach 0.0067, a twentieth
    of it, so a quarter holds a five-fold margin)."""
    videos, held = bouncing(2048, 12, 32, 0), bouncing(8, 12, 32, 1)
    clips = Loader(train=lambda partition: ({"video": videos[start:start + 32]}
                                            for start in itertools.cycle(range(0, 2048, 32))),
                   val=None, records=2048, batch=32)
    tokenizer = CausalTokenizer((32, 32, 1), width=128, depth=3, latents=16, bottleneck=8)
    state = Trainer(TokenizerObjective(tokenizer), optax.adam(3e-4), key=jax.random.key(9)).fit(
        clips, steps=8000, log_every=4000)
    frames = held.reshape(-1, 32, 32, 1)
    rebuilt = np.asarray(jax.jit(tokenizer.apply)(state.variables, frames))
    mean = videos.reshape(-1, 32, 32, 1).mean(0)
    assert np.mean(np.square(rebuilt - frames)) < np.mean(np.square(mean - frames)) / 4
