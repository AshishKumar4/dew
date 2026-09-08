"""Entropy-bounded block generation for a shared-weight diffusion language model.

DiffusionGemma Technical Report (2608.00146), Algorithm 1: initialize a uniform
canvas, refine it with tempered self-conditioning, stop stable/confident rows,
and commit the final argmax canvas through the causal encoder. The compiled
loops keep fixed bounds and mask finished rows, including their step counts.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, replace
from collections.abc import Callable, Sequence
from functools import partial
from typing import Generic, overload

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils

from dew.artifacts import agree_process_phase
from dew.nn.diffusion_gemma import DiffusionGemma
from dew.nn.inputs import (ArrayT, ModelInputs, RowPlan, continuation_keys, generation_signature,
                           local_rows, mesh_of, prompt_major, request_key)
from dew.objectives.base import Variables


@struct.dataclass
class CanvasGeneration(Generic[ArrayT]):
    """Prompt plus padded response, with no autoregressive likelihood claim.

    ``lengths`` counts response tokens including the first EOS, not prompt
    tokens. ``decoder_steps`` counts useful refinements per row across canvases.
    ``terminated`` distinguishes EOS from the requested token limit. A request
    for ``n`` continuations per prompt gives every array ``[B * n, ...]``
    rows, each prompt's continuations together and in prompt order. Arrays
    keep the placement the task ran with; ``host()`` reads this process's
    ``rows`` real rows back, and ``text`` decodes them through the bound
    processor.
    """

    tokens: ArrayT
    lengths: ArrayT
    terminated: ArrayT
    decoder_steps: ArrayT
    rows: int | None = struct.field(pytree_node=False, default=None)
    prompt_width: int | None = struct.field(pytree_node=False, default=None)
    decoder: Callable[[jax.typing.ArrayLike, jax.typing.ArrayLike, int], tuple[str, ...]] | None = struct.field(
        pytree_node=False, default=None)

    def host(self) -> CanvasGeneration[np.ndarray]:
        """This process's real rows as host arrays."""
        return jax.tree.map(lambda leaf: local_rows(leaf)[:self.rows], self)

    @functools.cached_property
    def text(self) -> tuple[str, ...]:
        """Each real row's response, decoded on first access."""
        if self.decoder is None:
            raise ValueError("this generation carries no processor to decode with")
        if self.prompt_width is None:
            raise ValueError("this generation has no prompt width")
        rows = self.host()
        return self.decoder(rows.tokens, rows.lengths, self.prompt_width)


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

    @overload
    def generate(self, model: DiffusionGemma, variables: Variables,
                 inputs: ModelInputs | jax.typing.ArrayLike | Sequence[Sequence[int]],
                 max_new_tokens: int, *, key: jax.Array, seed: None = None, n: int = 1,
                 eos_token_ids: tuple[int, ...] = (), pad_token_id: int = 0) -> CanvasGeneration: ...

    @overload
    def generate(self, model: DiffusionGemma, variables: Variables,
                 inputs: ModelInputs | jax.typing.ArrayLike | Sequence[Sequence[int]],
                 max_new_tokens: int, *, key: None = None, seed: int, n: int = 1,
                 eos_token_ids: tuple[int, ...] = (), pad_token_id: int = 0) -> CanvasGeneration: ...

    def generate(self, model: DiffusionGemma, variables: Variables,
                 inputs: ModelInputs | jax.typing.ArrayLike | Sequence[Sequence[int]],
                 max_new_tokens: int, *, key: jax.Array | None = None, seed: int | None = None,
                 n: int = 1, eos_token_ids: tuple[int, ...] = (), pad_token_id: int = 0) -> CanvasGeneration:
        """Run prefill, refinement and clean-token commits as one device computation.

        The last canvas is fully refined even when only part is requested;
        output is cropped to the token limit. Finished rows are padded after
        their first EOS. No host-side decisions depend on generated tokens.
        Weights keep their placement; on a mesh, rows split over its batch
        axes and the result keeps that sharding. The canvas sampler draws
        one batch-wide key per refinement, so a row's draw depends on the
        rows placed with it.

        ``n`` continuations of each prompt share its prefill and refine
        independently from it. They leave as ``n`` consecutive rows per
        prompt, in prompt order. Continuation zero refines with the
        request's own key, so it is what a single continuation draws.
        """
        prepared = None
        error = None
        request = None
        try:
            request = request_key(key, seed)
            canonical = ModelInputs.from_value(inputs)
            prepared = jax.tree.map(local_rows, canonical)
            _validated(model, self, prepared, max_new_tokens, eos_token_ids, pad_token_id, n)
        except BaseException as failure:
            error = failure
        agree_process_phase(error, phase="canvas generation setup")
        assert prepared is not None and request is not None
        if jax.process_count() > 1:
            controls = (max_new_tokens, n, self, eos_token_ids, pad_token_id, model)
            multihost_utils.assert_equal(
                generation_signature(prepared, controls),
                "canvas input schemas, model geometry and generation policy must agree")
        plan = RowPlan.over(mesh_of(variables), prepared.tokens.shape[0])
        placed = plan.place(plan.pad(prepared))
        result = _compiled(plan.sharding)(
            model, variables, placed, request,
            CanvasPlan(self, tuple(eos_token_ids), pad_token_id, max_new_tokens), n)
        return replace(result, rows=plan.rows * n, prompt_width=prepared.tokens.shape[1])


@dataclass(frozen=True)
class CanvasPlan:
    """Static controls of one canvas request; every field enters the compiled step."""

    process: BlockProcess
    eos_token_ids: tuple[int, ...]
    pad_token_id: int
    max_new_tokens: int

    @property
    def blocks(self) -> int:
        length = self.process.canvas_length
        return (self.max_new_tokens + length - 1) // length


@struct.dataclass
class CanvasDecodeState:
    """Committed output and prefix cache between complete canvas refinements.

    index counts refined canvases even after individual rows finish. It is
    the random-key fold and commit position, independent of emitted lengths.
    """

    cache: Variables
    result: CanvasGeneration
    index: jax.Array


def _validated(model: DiffusionGemma, process: BlockProcess, inputs: ModelInputs, max_new_tokens: int,
               eos_token_ids: tuple[int, ...], pad_token_id: int, n: int) -> None:
    """Host checks before the compiled loop."""
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer")
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer number of continuations")
    if process.vocab_size != model.vocab_size or process.canvas_length != model.canvas_length:
        raise ValueError("BlockProcess geometry must match the loaded model")
    if not 0 <= pad_token_id < process.vocab_size:
        raise ValueError("pad_token_id is outside the vocabulary")
    if any(not 0 <= token < process.vocab_size for token in eos_token_ids):
        raise ValueError("eos_token_ids contain an id outside the vocabulary")
    _, prompt_length = inputs.tokens.shape
    if prompt_length == 0:
        raise ValueError("a block-diffusion prompt must contain at least one token")
    blocks = (max_new_tokens + process.canvas_length - 1) // process.canvas_length
    if prompt_length + blocks * process.canvas_length > model.max_seq_len:
        raise ValueError("prompt plus rounded-up canvases exceeds max_seq_len")


def _begin(model: DiffusionGemma, variables: Variables, inputs: ModelInputs,
           plan: CanvasPlan) -> CanvasDecodeState:
    """The deterministic prefill state: the encoded prompt and an empty result."""
    batch, prompt_length = inputs.tokens.shape
    output = jnp.full((batch, prompt_length + plan.blocks * plan.process.canvas_length),
                      plan.pad_token_id, jnp.int32)
    output = output.at[:, :prompt_length].set(inputs.tokens)
    result = CanvasGeneration(
        tokens=output, lengths=jnp.zeros((batch,), jnp.int32),
        terminated=jnp.zeros((batch,), bool), decoder_steps=jnp.zeros((batch,), jnp.int32))
    cache = {}
    if plan.blocks > 0:
        cache = model.apply(variables, batch, method=model.init_cache, mutable=["cache"])[1]["cache"]
        cache = model.apply(
            {**variables, "cache": cache}, inputs,
            method=lambda module, batch: module.encode(batch.tokens, **batch.kwargs()),
            mutable=["cache"])[1]["cache"]
    return CanvasDecodeState(cache, result, jnp.asarray(0, jnp.int32))


def _advance(model: DiffusionGemma, variables: Variables, state: CanvasDecodeState,
             key: jax.Array, plan: CanvasPlan) -> CanvasDecodeState:
    cache, result, index = state.cache, state.result, state.index
    process, length = plan.process, plan.process.canvas_length
    blocks = plan.blocks
    batch, width = result.tokens.shape
    prompt_length = width - blocks * length
    canvas_key = jax.random.fold_in(key, index)
    refined = process.refine(model, variables, cache, canvas_key, batch, result.terminated)
    available = jnp.minimum(length, plan.max_new_tokens - index * length)
    is_eos = jnp.isin(refined.argmax, jnp.asarray(plan.eos_token_ids, jnp.int32))
    valid = jnp.arange(length)[None, :] < available
    is_eos = is_eos & valid
    first_eos = jnp.min(jnp.where(is_eos, jnp.arange(length)[None, :], length), axis=-1)
    emitted = jnp.where(result.terminated, 0, jnp.minimum(first_eos + 1, available))
    keep = jnp.arange(length)[None, :] < emitted[:, None]
    clean = jnp.where(keep, refined.argmax, plan.pad_token_id)
    tokens = jax.lax.dynamic_update_slice(result.tokens, clean, (0, prompt_length + index * length))
    result = CanvasGeneration(
        tokens=tokens, lengths=result.lengths + emitted,
        terminated=result.terminated | jnp.any(is_eos, axis=-1),
        decoder_steps=result.decoder_steps + refined.decoder_steps)
    # This branch depends only on the request's canvas counter, never on a
    # rank-local sampled token or termination outcome.
    cache = jax.lax.cond(
        index + 1 < blocks,
        lambda old: model.apply({**variables, "cache": old}, clean, method=model.encode,
                                mutable=["cache"])[1]["cache"],
        lambda old: old, cache)
    return CanvasDecodeState(cache, result, index + 1)


def _materialize(state: CanvasDecodeState, prompt_length: int, max_new_tokens: int) -> CanvasGeneration:
    return replace(state.result, tokens=state.result.tokens[:, :prompt_length + max_new_tokens])


def _generate(model: DiffusionGemma, variables: Variables, inputs: ModelInputs,
              key: jax.Array, plan: CanvasPlan, n: int) -> CanvasGeneration:
    initial = _begin(model, variables, inputs, plan)
    prompt_length = inputs.tokens.shape[1]

    def continued(canvas_key: jax.Array) -> CanvasGeneration:
        if plan.blocks == 0:
            return _materialize(initial, prompt_length, plan.max_new_tokens)

        def step(_, state):
            return _advance(model, variables, state, canvas_key, plan)

        return _materialize(jax.lax.fori_loop(0, plan.blocks, step, initial),
                            prompt_length, plan.max_new_tokens)

    if n == 1:
        return continued(key)
    # A mapped loop over the continuations, not a wider batch: each one
    # refines the shared prefill state over the original prompt rows, so a
    # row's draws depend on the rows placed with it exactly as they do for
    # one continuation.
    return prompt_major(jax.lax.map(continued, continuation_keys(key, n)))


@functools.lru_cache(maxsize=None)
def _compiled(rows: jax.sharding.NamedSharding | None):
    return jax.jit(_generate, static_argnames=("model", "plan", "n"),
                   in_shardings=(None, rows, None), out_shardings=rows)
