"""Block diffusion over a token canvas, the DiffusionGemma sampler side.

The released design pairs a causal encoder over the prompt, filling a KV
cache, with a bidirectional decoder over a fixed-length canvas that also
attends to that cache (TF diffusion_gemma modeling_diffusion_gemma.py:281,
:383, :1326-1440). There is no mask token and no timestep embedding: the
canvas starts as uniform random ids, a temperature annealed by the remaining
step count sharpens the logits
(generation_diffusion_gemma.py:276-316, cur_step counts down), and positions
whose categorical entropy fits an independence bound are accepted while the
rest are renoised (:343-469). The previous step's tempered logits condition
the next one through the self-conditioning MLP in dew.nn.diffusion_gemma.

What lives here needs no model: the canvas distribution, the temperature
schedule, the acceptance rule and the renoise. The denoiser itself, encoder
cache plus canvas plus self-conditioning, needs a backbone forward that takes
a cached prefix beside a bidirectional canvas, which CausalTransformer has no
entry point for today; `sample_canvas` takes it as a callable so the loop is
testable now and complete when that entry point lands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class BlockProcess:
    """A canvas of `canvas_length` ids over `vocab_size` tokens."""

    canvas_length: int
    vocab_size: int
    t_max: float = 0.8
    t_min: float = 0.4
    entropy_bound: float = 0.1
    max_steps: int = 48

    def __post_init__(self):
        if self.canvas_length < 1:
            raise ValueError(
                f"canvas_length is {self.canvas_length}, a canvas holds ids")
        if self.vocab_size < 1:
            raise ValueError(
                f"vocab_size is {self.vocab_size}, ids come from somewhere")
        if not 0 < self.t_min <= self.t_max:
            raise ValueError(
                f"temperatures t_min/t_max read ({self.t_min}, {self.t_max}), "
                "the schedule cools from t_max to t_min")
        if self.entropy_bound <= 0:
            raise ValueError(
                f"entropy_bound is {self.entropy_bound}, the reference refuses "
                "anything but a positive float")
        if self.max_steps < 1:
            raise ValueError(f"max_steps is {self.max_steps}, steps denoise")

    def temperature(self, cur_step: int) -> float:
        """The temperature with cur_step steps remaining, t_max down to t_min."""
        return self.t_min + (self.t_max - self.t_min) * (cur_step / self.max_steps)

    def noise(self, key: jax.Array, shape: Sequence[int]) -> jax.Array:
        """A uniform random canvas, the sampler's starting point."""
        return jax.random.randint(key, shape, 0, self.vocab_size, jnp.int32)

    def temper(self, logits: jax.typing.ArrayLike, cur_step: int) -> jax.Array:
        """Raw decoder logits divided by the step's temperature."""
        return jnp.asarray(logits, jnp.float32) / self.temperature(cur_step)

    def accept(self, current: jax.typing.ArrayLike, denoised: jax.typing.ArrayLike,
               logits: jax.typing.ArrayLike) -> tuple[jax.Array, jax.Array]:
        """The accepted canvas and its mask under the entropy bound.

        Positions sort by ascending entropy; the first k stay where their
        cumulative entropy minus the running maximum fits the bound, which caps
        the joint mutual information of what is accepted at once. `logits` are
        the tempered ones the denoiser canvas was drawn from.
        """
        log_probs = jax.nn.log_softmax(jnp.asarray(logits, jnp.float32), axis=-1)
        entropy = -(jnp.exp(log_probs) * log_probs).sum(axis=-1)
        order = jnp.argsort(entropy, axis=-1)
        ranked = jnp.take_along_axis(entropy, order, axis=-1)
        cumulative = jnp.cumsum(ranked, axis=-1)
        running_max = jnp.maximum.accumulate(ranked, axis=-1)
        chosen = cumulative - running_max <= self.entropy_bound
        mask = jnp.take_along_axis(chosen, jnp.argsort(order, axis=-1), axis=-1)
        return jnp.where(mask, denoised, current), mask

    def renoise(self, key: jax.Array, accepted: jax.typing.ArrayLike,
                mask: jax.typing.ArrayLike) -> jax.Array:
        """Accepted positions kept, the rest drawn uniform again."""
        fresh = self.noise(key, jnp.shape(accepted))
        return jnp.where(jnp.asarray(mask, bool), accepted, fresh)


def sample_canvas(key: jax.Array, denoise, shape: Sequence[int], *,
                  steps: int) -> jax.Array:
    """A canvas denoised by `denoise(canvas, prev_logits) -> raw logits`.

    The first step conditions on nothing; every later step conditions on the
    previous step's tempered logits. `steps` counts down like the reference
    loop. What returns is the argmax of the last step's tempered logits; the
    reference appends that canvas to the sequence.
    """
    process = denoise.process
    key, canvas_key = jax.random.split(key)
    canvas = process.noise(canvas_key, shape)
    prev: Optional[jax.Array] = None
    for remaining in range(steps, 0, -1):
        key, step_key, draw_key, renoise_key = jax.random.split(key, 4)
        raw = denoise(canvas, prev)
        tempered = process.temper(raw, remaining)
        drawn = jax.random.categorical(draw_key, tempered, axis=-1).astype(jnp.int32)
        accepted, mask = process.accept(canvas, drawn, tempered)
        canvas = process.renoise(renoise_key, accepted, mask)
        prev = tempered
    final = process.temper(denoise(canvas, prev), 1)
    return jnp.argmax(final, axis=-1).astype(jnp.int32)
