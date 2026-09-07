"""Entropy-bounded block generation for a shared-weight diffusion language model.

DiffusionGemma Technical Report (2608.00146), Algorithm 1: initialize a uniform
canvas, refine it with tempered self-conditioning, stop stable/confident rows,
and commit the final argmax canvas through the causal encoder. The compiled
loops keep fixed bounds and mask finished rows, including their step counts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from collections.abc import Sequence
from functools import partial
from typing import ClassVar

from flax import linen as nn, struct
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
import numpy as np

from dew.artifacts import agree_process_phase
from dew.nn.diffusion_gemma import DiffusionGemma
from dew.nn.inputs import ModelInputs, generation_signature
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

    def generate(self, model: DiffusionGemma, variables: Variables,
                 inputs: ModelInputs | jax.typing.ArrayLike | Sequence[Sequence[int]],
                 max_new_tokens: int, *, key: jax.Array, eos_token_ids: tuple[int, ...] = (),
                 pad_token_id: int = 0) -> CanvasGeneration:
        """Run prefill, refinement and clean-token commits as one device computation.

        The last canvas is fully refined even when only part is requested;
        output is cropped to the token limit. Finished rows are padded after
        their first EOS. No host-side decisions depend on generated tokens.
        """
        prepared = None
        error = None
        try:
            prepared = ModelInputs.from_value(inputs)
            if jax.random.key_data(key).ndim != 1:
                raise ValueError("key must be one PRNG key, not a batch of keys")
            _validated(model, self, prepared, max_new_tokens, eos_token_ids, pad_token_id)
        except BaseException as failure:
            error = failure
        agree_process_phase(error, phase="canvas generation setup")
        assert prepared is not None
        if jax.process_count() > 1:
            controls = (max_new_tokens, self, eos_token_ids, pad_token_id, model)
            multihost_utils.assert_equal(
                generation_signature(prepared, controls),
                "canvas input schemas, model geometry and generation policy must agree")
        return _generate(model, variables, prepared, key,
                         CanvasPlan(self, tuple(eos_token_ids), pad_token_id, max_new_tokens))


@dataclass(frozen=True)
class CanvasGeometry:
    """What the deterministic prefill state depends on besides the inputs."""

    canvas_length: int
    max_new_tokens: int
    pad_token_id: int

    @property
    def blocks(self) -> int:
        return (self.max_new_tokens + self.canvas_length - 1) // self.canvas_length


@dataclass(frozen=True)
class CanvasPlan:
    """Static controls of one canvas request; every field enters the compiled step."""

    process: BlockProcess
    eos_token_ids: tuple[int, ...]
    pad_token_id: int
    max_new_tokens: int

    @property
    def controls(self) -> CanvasPlan:
        return self

    @property
    def geometry(self) -> CanvasGeometry:
        return CanvasGeometry(self.process.canvas_length, self.max_new_tokens, self.pad_token_id)

    @property
    def steps(self) -> int:
        return self.geometry.blocks


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
               eos_token_ids: tuple[int, ...], pad_token_id: int) -> None:
    """Host checks shared by batch generation and the engine."""
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer")
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
           geometry: CanvasGeometry) -> CanvasDecodeState:
    batch, prompt_length = inputs.tokens.shape
    output = jnp.full((batch, prompt_length + geometry.blocks * geometry.canvas_length),
                      geometry.pad_token_id, jnp.int32)
    output = output.at[:, :prompt_length].set(inputs.tokens)
    result = CanvasGeneration(
        tokens=output, lengths=jnp.zeros((batch,), jnp.int32),
        terminated=jnp.zeros((batch,), bool), decoder_steps=jnp.zeros((batch,), jnp.int32))
    cache = {}
    if geometry.blocks > 0:
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
    blocks = plan.steps
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


@jax.jit(static_argnames=("model", "plan"))
def _generate(model: DiffusionGemma, variables: Variables, inputs: ModelInputs,
              key: jax.Array, plan: CanvasPlan) -> CanvasGeneration:
    initial = _begin(model, variables, inputs, plan.geometry)
    if plan.steps == 0:
        return _materialize(initial, inputs.tokens.shape[1], plan.max_new_tokens)

    def step(_, state):
        return _advance(model, variables, state, key, plan)

    final = jax.lax.fori_loop(0, plan.steps, step, initial)
    return _materialize(final, inputs.tokens.shape[1], plan.max_new_tokens)


@dataclass(frozen=True)
class SpanEvent:
    """Tokens one row committed from one refined canvas."""

    row: int
    start: int
    tokens: tuple[int, ...]
    terminated: bool
    decoder_steps: int


class CanvasProgress:
    """Host view of committed canvases; the result stays in the device state."""

    def __init__(self, inputs: ModelInputs, plan: CanvasPlan) -> None:
        self.plan = plan
        self.prompt = np.asarray(inputs.tokens)
        self.batch, self.prompt_length = self.prompt.shape
        self.lengths = np.zeros(self.batch, np.int32)
        self.decoder_steps = np.zeros(self.batch, np.int32)
        self.terminated = np.zeros(self.batch, bool)
        self.remaining = plan.steps
        self.spans: list[SpanEvent] = []

    @property
    def bytes(self) -> int:
        return 0

    @property
    def done(self) -> bool:
        return self.remaining == 0 or bool(self.terminated.all())

    def record(self, outputs: CanvasGeneration) -> None:
        """Consume [steps, ...] committed results, one per refined canvas."""
        tokens, lengths = np.asarray(outputs.tokens), np.asarray(outputs.lengths)
        terminated, steps = np.asarray(outputs.terminated), np.asarray(outputs.decoder_steps)
        for step in range(tokens.shape[0]):
            for row in range(self.batch):
                start, stop = int(self.lengths[row]), int(lengths[step, row])
                if stop == start and not (terminated[step, row] and not self.terminated[row]):
                    continue
                self.spans.append(SpanEvent(
                    row, start, tuple(int(token) for token in
                                      tokens[step, row, self.prompt_length + start:self.prompt_length + stop]),
                    bool(terminated[step, row]), int(steps[step, row] - self.decoder_steps[row])))
            self.lengths, self.terminated, self.decoder_steps = lengths[step], terminated[step], steps[step]
            self.remaining -= 1

    @property
    def count(self) -> int:
        return len(self.spans)

    def events(self, start: int) -> list[SpanEvent]:
        return self.spans[start:]

    def result(self, state: CanvasDecodeState | None) -> CanvasGeneration:
        if state is None:
            if self.plan.steps:
                raise ValueError("a canvas result is read from its device state")
            return CanvasGeneration(jnp.asarray(self.prompt), jnp.asarray(self.lengths),
                                    jnp.asarray(self.terminated), jnp.asarray(self.decoder_steps))
        return _materialize(state, self.prompt_length, self.plan.max_new_tokens)


def _canvas_model(model: nn.Module) -> DiffusionGemma:
    if not isinstance(model, DiffusionGemma):
        raise TypeError("canvas generation requires a DiffusionGemma model")
    return model


@dataclass(frozen=True)
class CanvasFamily:
    """Block-diffusion generation over the shared prefill and canvas kernels.

    A refinement draws noise for its whole batch, so one request's rows form
    an indivisible state; an engine never merges rows of different requests.
    ``process`` is the policy used when a request passes no generation value.
    """

    process: BlockProcess
    eos_token_ids: tuple[int, ...] = ()
    pad_token_id: int = 0
    shared_rows: ClassVar[bool] = False

    def prepare(self, model: nn.Module, inputs: ModelInputs, max_new_tokens: int,
                generation: object | None) -> tuple[ModelInputs, CanvasPlan]:
        process = self.process if generation is None else generation
        if not isinstance(process, BlockProcess):
            raise TypeError("DiffusionGemma generation takes a BlockProcess value")
        _validated(_canvas_model(model), process, inputs, max_new_tokens, self.eos_token_ids, self.pad_token_id)
        return inputs, CanvasPlan(process, self.eos_token_ids, self.pad_token_id, max_new_tokens)

    def begin(self, model: nn.Module, variables: Variables, inputs: ModelInputs,
              geometry: CanvasGeometry) -> CanvasDecodeState:
        return _begin(_canvas_model(model), variables, inputs, geometry)

    def advance(self, model: nn.Module, variables: Variables, state: CanvasDecodeState,
                keys: jax.Array, controls: CanvasPlan, steps: int
                ) -> tuple[CanvasDecodeState, CanvasGeneration]:
        canvas_model = _canvas_model(model)

        def step(carry, _):
            following = _advance(canvas_model, variables, carry, keys, controls)
            return following, following.result

        return jax.lax.scan(step, state, None, length=steps)

    def keys(self, key: jax.Array, rows: int) -> jax.Array:
        return key

    def progress(self, inputs: ModelInputs, plan: CanvasPlan) -> CanvasProgress:
        return CanvasProgress(inputs, plan)

    def generate(self, model: nn.Module, variables: Variables,
                 inputs: ModelInputs | jax.typing.ArrayLike | Sequence[Sequence[int]],
                 max_new_tokens: int, *, key: jax.Array,
                 generation: object | None = None) -> CanvasGeneration:
        process = self.process if generation is None else generation
        if not isinstance(process, BlockProcess):
            raise TypeError("DiffusionGemma generation takes a BlockProcess value")
        return process.generate(_canvas_model(model), variables, inputs, max_new_tokens, key=key,
                                eos_token_ids=self.eos_token_ids, pad_token_id=self.pad_token_id)
