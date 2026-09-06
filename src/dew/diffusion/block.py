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

What lives here besides the schedule is the denoiser wiring: `prefill_cache`
runs the causal encoder over the prompt into a KV cache, and `denoise_logits`
runs the bidirectional canvas against that cache with self-conditioning
folded in through the decoder's input-embeddings hook. `sample_canvas` takes
a denoiser callable so the loop stays testable on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence

import jax
import jax.numpy as jnp

from dew.nn.diffusion_gemma import soft_embeddings


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


def prefill_cache(encoder, variables, prompt) -> dict:
    """The encoder's KV cache after reading the prompt, the prefix the canvas
    attends to.

    `encoder` is the causal model sharing the denoiser's weights; `variables`
    its `{'params': ...}` tree. The prompt goes through once in decode mode
    with a mutable cache, exactly like the sampler's prefill, and what comes
    back is the cache collection alone.
    """
    prompt = jnp.asarray(prompt, jnp.int32)
    cache = encoder.apply(variables, prompt.shape[0],
                          method=type(encoder).init_cache,
                          mutable=["cache"])[1]["cache"]
    return encoder.apply({**variables, "cache": cache}, prompt, decode=True,
                         mutable=["cache"])[1]["cache"]


def denoise_logits(decoder, sc, variables, sc_variables, cache, canvas,
                   prev_logits=None):
    """One canvas step's raw logits: self-conditioning folded in, prefix
    cached, bidirectional over the canvas.

    `decoder` is the same weights as the encoder with `causal=False`;
    `variables` carries its params without the cache. The previous step's
    tempered logits become soft embeddings through the embedding table, or
    zeros on the first step, and the self-conditioning module folds them into
    the scaled canvas embeddings the input-embeddings hook carries. The canvas
    positions continue past the prefix as decoder_position_ids.
    """
    canvas = jnp.asarray(canvas, jnp.int32)
    table = variables["params"]["embed_tokens"]["embedding"]
    width = table.shape[-1]
    scaled = (table[canvas].astype(jnp.float32)
              * jnp.asarray(math.sqrt(width), jnp.float32))
    if prev_logits is None:
        soft = jnp.zeros_like(scaled)
    else:
        soft = soft_embeddings(prev_logits, table,
                               math.sqrt(width)).astype(scaled.dtype)
    conditioned = sc.apply(sc_variables, scaled, soft.astype(scaled.dtype))
    length = canvas.shape[1]
    positions = jnp.broadcast_to(jnp.arange(length), canvas.shape)
    full = {**variables, "cache": cache}
    out, _ = decoder.apply(full, canvas, decode=True,
                           input_embeddings=conditioned,
                           embedding_positions=positions, mutable=["cache"])
    return out


def sample_canvas(key: jax.Array, denoise, shape: Sequence[int], *,
                  steps: int) -> jax.Array:
    """A canvas denoised by `denoise(canvas, prev_logits) -> raw logits`.

    The first step conditions on nothing; every later step conditions on the
    previous step's tempered logits. `steps` counts down like the reference
    loop. The argmax of the last step's tempered logits is the denoised canvas.
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
