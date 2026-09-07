"""Entropy-bounded block generation for a shared-weight diffusion language model.

DiffusionGemma Technical Report (2608.00146), Algorithm 1: initialize a uniform
canvas, refine it with tempered self-conditioning, stop stable/confident rows,
and commit the final argmax canvas through the causal encoder. The compiled
loops keep fixed bounds and mask finished rows, including their step counts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Sequence
from functools import partial

from flax import struct
import jax
import jax.numpy as jnp

from dew.nn.diffusion_gemma import DiffusionGemma
from dew.nn.inputs import ModelInputs
from dew.objectives.base import Variables


@struct.dataclass
class CanvasGeneration:
    """Prompt plus padded response, with no autoregressive likelihood claim.

    ``lengths`` counts response tokens including the first EOS, not prompt
    tokens. ``decoder_steps`` counts useful refinements per row across canvases.
    ``terminated`` distinguishes EOS from the requested token limit.
    """

    tokens: jax.Array
    lengths: jax.Array
    terminated: jax.Array
    decoder_steps: jax.Array


@struct.dataclass
class CanvasState:
    """The refinement state; previous logits reset between canvases."""

    canvas: jax.Array
    logits: jax.Array
    argmax: jax.Array
    stable_steps: jax.Array
    finished: jax.Array
    decoder_steps: jax.Array


def _entropy(logits: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    return -(jnp.exp(log_probs) * log_probs).sum(axis=-1)


@dataclass(frozen=True)
class BlockProcess:
    """Uniform-vocabulary diffusion with the published entropy-bound sampler.

    The temperature uses steps remaining, N through 1, rather than a second
    independently configured time grid. Stability counts previous identical
    argmax canvases; threshold one needs two matching predictions.
    """

    canvas_length: int
    vocab_size: int
    t_max: float = 0.8
    t_min: float = 0.4
    entropy_bound: float = 0.1
    max_steps: int = 48
    stability_threshold: int = 1
    confidence_threshold: float = 0.005

    def __post_init__(self):
        for name in ("canvas_length", "vocab_size", "max_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (isinstance(self.stability_threshold, bool)
                or not isinstance(self.stability_threshold, int) or self.stability_threshold < 0):
            raise ValueError("stability_threshold must be a nonnegative integer")
        if not (math.isfinite(self.t_min) and math.isfinite(self.t_max)
                and 0 < self.t_min <= self.t_max):
            raise ValueError("temperatures t_min/t_max must be finite with 0 < t_min <= t_max")
        for name in ("entropy_bound", "confidence_threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")

    def temperature(self, cur_step: jax.typing.ArrayLike) -> jax.Array:
        return self.t_min + (self.t_max - self.t_min) * jnp.asarray(cur_step) / self.max_steps

    def noise(self, key: jax.Array, shape: Sequence[int]) -> jax.Array:
        return jax.random.randint(key, shape, 0, self.vocab_size, jnp.int32)

    def temper(self, logits: jax.typing.ArrayLike, cur_step: jax.typing.ArrayLike) -> jax.Array:
        return jnp.asarray(logits, jnp.float32) / self.temperature(cur_step)

    def accept(self, current: jax.typing.ArrayLike, denoised: jax.typing.ArrayLike,
               logits: jax.typing.ArrayLike) -> tuple[jax.Array, jax.Array]:
        """Accept the lowest-entropy prefix whose cumulative excess fits the bound."""
        entropy = _entropy(jnp.asarray(logits))
        order = jnp.argsort(entropy, axis=-1, stable=True)
        ranked = jnp.take_along_axis(entropy, order, axis=-1)
        chosen = jnp.cumsum(ranked, axis=-1) - ranked <= self.entropy_bound
        inverse = jnp.argsort(order, axis=-1)
        mask = jnp.take_along_axis(chosen, inverse, axis=-1)
        return jnp.where(mask, denoised, current), mask

    def renoise(self, key: jax.Array, accepted: jax.typing.ArrayLike,
                mask: jax.typing.ArrayLike) -> jax.Array:
        return jnp.where(mask, accepted, self.noise(key, jnp.shape(accepted)))

    @partial(jax.jit, static_argnames=("self", "model", "batch"))
    def refine(self, model: DiffusionGemma, variables: Variables, cache: Variables,
               key: jax.Array, batch: int, finished: jax.Array) -> CanvasState:
        """Refine one canvas without updating its prefix cache."""
        shape = (batch, self.canvas_length)
        initial = CanvasState(
            canvas=self.noise(jax.random.fold_in(key, 0), shape),
            logits=jnp.zeros((*shape, self.vocab_size), jnp.float32),
            argmax=jnp.full(shape, -1, jnp.int32),
            stable_steps=jnp.zeros((batch,), jnp.int32),
            finished=finished,
            decoder_steps=jnp.zeros((batch,), jnp.int32))

        def step(index, state: CanvasState) -> CanvasState:
            raw = model.apply(
                {**variables, "cache": cache}, state.canvas,
                self_conditioning_logits=state.logits,
                self_conditioning_mask=jnp.full((batch,), index != 0))
            if not isinstance(raw, jax.Array):
                raise TypeError("a canvas forward must return a logits array")
            logits = self.temper(raw, self.max_steps - index)
            draw_key, noise_key = jax.random.split(jax.random.fold_in(key, index + 1))
            drawn = jax.random.categorical(draw_key, logits, axis=-1).astype(jnp.int32)
            accepted, mask = self.accept(state.canvas, drawn, logits)
            canvas = self.renoise(noise_key, accepted, mask)
            argmax = jnp.argmax(logits, axis=-1).astype(jnp.int32)
            stable_steps = jnp.where(jnp.all(argmax == state.argmax, axis=-1),
                                     state.stable_steps + 1, 0)
            confident = _entropy(logits).mean(axis=-1) < self.confidence_threshold
            stop = (stable_steps >= self.stability_threshold) & confident
            active = ~state.finished
            return CanvasState(
                canvas=jnp.where(active[:, None], canvas, state.canvas),
                logits=jnp.where(active[:, None, None], logits, state.logits),
                argmax=jnp.where(active[:, None], argmax, state.argmax),
                stable_steps=jnp.where(active, stable_steps, state.stable_steps),
                finished=state.finished | stop,
                decoder_steps=state.decoder_steps + active.astype(jnp.int32))

        return jax.lax.fori_loop(0, self.max_steps, step, initial)

    def generate(self, model: DiffusionGemma, variables: Variables, inputs: ModelInputs,
                 max_new_tokens: int, *, key: jax.Array, eos_token_ids: tuple[int, ...] = (),
                 pad_token_id: int = 0) -> CanvasGeneration:
        """Run prefill, refinement and clean-token commits as one device computation.

        The last canvas is fully refined even when only part is requested;
        output is cropped to the token limit. Finished rows are padded after
        their first EOS. No host-side decisions depend on generated tokens.
        """
        inputs.validate()
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        if self.vocab_size != model.vocab_size or self.canvas_length != model.canvas_length:
            raise ValueError("BlockProcess geometry must match the loaded model")
        if not 0 <= pad_token_id < self.vocab_size:
            raise ValueError("pad_token_id is outside the vocabulary")
        if any(not 0 <= token < self.vocab_size for token in eos_token_ids):
            raise ValueError("eos_token_ids contain an id outside the vocabulary")
        batch, prompt_length = inputs.tokens.shape
        if prompt_length == 0:
            raise ValueError("a block-diffusion prompt must contain at least one token")
        blocks = (max_new_tokens + self.canvas_length - 1) // self.canvas_length
        if prompt_length + blocks * self.canvas_length > model.max_seq_len:
            raise ValueError("prompt plus rounded-up canvases exceeds max_seq_len")
        return _generate(model, variables, inputs, max_new_tokens, key, self,
                         eos_token_ids, pad_token_id)


@jax.jit(static_argnames=("model", "max_new_tokens", "process", "eos_token_ids", "pad_token_id"))
def _generate(model: DiffusionGemma, variables: Variables, inputs: ModelInputs,
              max_new_tokens: int, key: jax.Array, process: BlockProcess,
              eos_token_ids: tuple[int, ...], pad_token_id: int) -> CanvasGeneration:
    batch, prompt_length = inputs.tokens.shape
    length = process.canvas_length
    blocks = (max_new_tokens + length - 1) // length
    output = jnp.full((batch, prompt_length + blocks * length), pad_token_id, jnp.int32)
    output = output.at[:, :prompt_length].set(inputs.tokens)
    initial = CanvasGeneration(
        tokens=output, lengths=jnp.zeros((batch,), jnp.int32),
        terminated=jnp.zeros((batch,), bool), decoder_steps=jnp.zeros((batch,), jnp.int32))
    if max_new_tokens == 0:
        return initial
    cache = model.apply(variables, batch, method=model.init_cache, mutable=["cache"])[1]["cache"]
    cache = model.apply(
        {**variables, "cache": cache}, inputs,
        method=lambda module, batch: module.encode(batch.tokens, **batch.kwargs()),
        mutable=["cache"])[1]["cache"]

    def canvas_step(index, carry):
        cache, result = carry
        canvas_key = jax.random.fold_in(key, index)
        state = process.refine(model, variables, cache, canvas_key, batch, result.terminated)
        available = jnp.minimum(length, max_new_tokens - index * length)
        is_eos = jnp.isin(state.argmax, jnp.asarray(eos_token_ids, jnp.int32))
        valid = jnp.arange(length)[None, :] < available
        is_eos = is_eos & valid
        first_eos = jnp.min(jnp.where(is_eos, jnp.arange(length)[None, :], length), axis=-1)
        emitted = jnp.where(result.terminated, 0, jnp.minimum(first_eos + 1, available))
        keep = jnp.arange(length)[None, :] < emitted[:, None]
        clean = jnp.where(keep, state.argmax, pad_token_id)
        tokens = jax.lax.dynamic_update_slice(result.tokens, clean, (0, prompt_length + index * length))
        result = CanvasGeneration(
            tokens=tokens, lengths=result.lengths + emitted,
            terminated=result.terminated | jnp.any(is_eos, axis=-1),
            decoder_steps=result.decoder_steps + state.decoder_steps)
        # Only preceding canvases enter the cache. Both branch choices are
        # scalar loop counters shared by every device, never rank-local EOS.
        cache = jax.lax.cond(
            index + 1 < blocks,
            lambda old: model.apply({**variables, "cache": old}, clean, method=model.encode,
                                    mutable=["cache"])[1]["cache"],
            lambda old: old, cache)
        return cache, result

    _, result = jax.lax.fori_loop(0, blocks, canvas_step, (cache, initial))
    return result.replace(tokens=result.tokens[:, :prompt_length + max_new_tokens])
