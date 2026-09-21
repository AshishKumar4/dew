"""Reusable inference tasks: a model, its weights and its host processing.

A task binds what a generation needs beyond the request itself: the native
model, a captured variables mapping and, when the source ships one, the host
processor that turns text and media into `ModelInputs`. Controls stay the
typed values training already uses (`Sampling`, `BlockProcess`); results
stay the typed records the kernels produce. `bind` gives the same task over
other weights, which is how a training loop draws from a policy snapshot
without holding the trainer's mutable mapping. Array buffers are shared; do
not mutate, donate or delete those buffers while the task is using them.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol, overload

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.core import freeze
from jax.typing import ArrayLike

from dew.artifacts import agree_process_phase
from dew.diffusion.block import BlockProcess, CanvasGeneration
from dew.diffusion.discrete import MDLM_STEPS, DiscreteProcess, Unmask
from dew.nn.diffusion_gemma import DiffusionGemma
from dew.nn.inputs import ModelInputs, mesh_of, request_key
from dew.objectives.base import Variables
from dew.sampling.decoding import LogitsTransform, Stopping
from dew.sampling.strategies import Strategy
from dew.sampling.text import Criteria, Generation, Sampling, Transforms, generate
from dew.telemetry.profile import active_profile

Rows = ModelInputs | ArrayLike | Sequence[Sequence[int]]
Request = str | Sequence[str] | Rows

SHAPE_BUCKETS = tuple(1 << exponent for exponent in range(5, 21))
"""The shapes a text request is rounded up to: powers of two from 32.

Every distinct prompt width, batch, budget and continuation count is its
own several-second XLA compile, and served requests are rarely the same
length twice. So a prompt pads left to the smallest bucket of 64 or more,
where the attention mask hides the filler as it already hides the padding
a ragged batch needs; a budget rounds up to the smallest bucket of 32 or
more, and the trips past the request come off the result; the cache for
the call is the two together, rounded up again, in place of the model's
whole `max_seq_len`. Rows are the caller's and are not bucketed. A request
whose buckets would need more capacity than the model's `max_seq_len`
keeps its own shapes, so the ceiling refuses what it refuses today.
"""


class Processor(Protocol):
    """Host preprocessing and decoding, as a loaded source's processor does."""

    def __call__(self, text: str | Sequence[str], *, images: object | None = None) -> ModelInputs: ...

    def decode(self, tokens: ArrayLike) -> list[str]: ...


def _prepared(processor: Processor | None, request: Request, *, images: object | None) -> ModelInputs:
    """Classify raw text without changing numeric token identities or row order."""
    if isinstance(request, str):
        text = [request]
    elif isinstance(request, Sequence) and request and all(isinstance(entry, str) for entry in request):
        text = [entry for entry in request if isinstance(entry, str)]
    else:
        text = None
    if text is not None:
        if processor is None:
            raise ValueError("text requests need a processor; pass ModelInputs or token rows")
        return processor(text, images=images)
    if images is not None:
        raise ValueError("images travel with text through the processor; prepared rows carry them in ModelInputs")
    if isinstance(request, ModelInputs):
        return ModelInputs.from_value(request)
    if isinstance(request, Sequence):
        for row in request:
            if isinstance(row, str):
                raise ValueError("a request cannot mix text prompts and numeric token rows")
            if isinstance(row, np.ndarray) and not np.issubdtype(row.dtype, np.integer):
                raise ValueError("token rows must contain integers")
            if isinstance(row, Sequence) and any(
                    isinstance(token, (bool, np.bool_)) or not isinstance(token, (int, np.integer))
                    for token in row):
                raise ValueError("token rows must contain integers, not coerced token IDs")
        return ModelInputs.from_value(np.asarray(request))
    # A resident array stays where it is; a global one cannot be fetched.
    return ModelInputs.from_value(request)


def _task_inputs(processor: Processor | None, request: Request, *, images: object | None,
                 collective: bool, max_new_tokens: int | None, default_tokens: int | None,
                 max_length: int | None, key: jax.Array | None, seed: int | None) -> tuple[ModelInputs, int, jax.Array]:
    inputs = None
    budget = None
    random_key = None
    error = None
    try:
        random_key = request_key(key, seed)
        inputs = _prepared(processor, request, images=images)
        budget = _budget(max_new_tokens, default_tokens, max_length, inputs.tokens.shape[1])
    except Exception as failure:
        error = failure
    if collective:
        agree_process_phase(error, phase="inference task input preparation")
    elif error is not None:
        raise error
    assert inputs is not None and budget is not None and random_key is not None
    return inputs, budget, random_key


def _decoded(processor: Processor | None, tokens: ArrayLike, lengths: ArrayLike, width: int) -> tuple[str, ...]:
    if processor is None:
        return ()
    rows, counts = np.asarray(tokens), np.asarray(lengths)
    return tuple(processor.decode(rows[row:row + 1, width:width + int(counts[row])])[0]
                 for row in range(rows.shape[0]))


def _budget(requested: int | None, default: int | None, max_length: int | None, prompt_width: int) -> int:
    if requested is not None:
        return requested
    if default is not None:
        return default
    if max_length is not None:
        if type(max_length) is not int or max_length < prompt_width:
            raise ValueError("source max_length must be an integer at least as large as the prompt width")
        return max_length - prompt_width
    raise ValueError("max_new_tokens is required; the source declares no default budget")


def _bucket(value: int, smallest: int) -> int:
    """The smallest shape bucket that holds `value`, never below `smallest`."""
    for bucket in SHAPE_BUCKETS:
        if bucket >= value and bucket >= smallest:
            return bucket
    return value


def _ceiling(model: nn.Module) -> int | None:
    """The largest cache the model admits, or None where it declares none.

    The model is a boundary the task reads a shape across: `nn.Module`
    declares no context length, a decoder declares `max_seq_len` and a
    wrapper answers for the decoder it holds. `dew.sampling.text._validated`
    reads the same field the same way to refuse a request too large for it.
    """
    declared = getattr(model, "max_seq_len", None)
    return declared if type(declared) is int else None


def _bucketed(inputs: ModelInputs, budget: int, ceiling: int | None
              ) -> tuple[ModelInputs, int, int | None]:
    """The request at bucket shapes: padded inputs, scan trips and cache capacity.

    A capacity of None leaves the request its own shapes and the model its
    own cache. That is what a request too large for the ceiling gets, so it
    is refused where it is refused today, and what one carrying media or
    logical positions gets: validity is the only sequence field a filler
    slot has a value for, since such a slot holds no coordinate and no
    media feature.
    """
    if ceiling is None or budget < 1:
        return inputs, budget, None
    if set(inputs.token_fields) - {"attention_mask"} or inputs.conditioning:
        return inputs, budget, None
    width = _bucket(inputs.tokens.shape[1], 64)
    trips = _bucket(budget, 32)
    capacity = _bucket(width + trips, 64)
    if capacity > ceiling:
        return inputs, budget, None
    return _padded(inputs, width), trips, capacity


def _padded(inputs: ModelInputs, width: int) -> ModelInputs:
    """`inputs` left-padded to `width` slots, with the filler marked invalid."""
    extra = width - inputs.tokens.shape[1]
    if extra < 1:
        return inputs
    valid = inputs.token_fields.get("attention_mask")
    if valid is None:
        valid = jnp.ones(inputs.tokens.shape, bool)
    return replace(inputs, tokens=jnp.pad(inputs.tokens, ((0, 0), (extra, 0))),
                   token_fields={"attention_mask": jnp.pad(valid, ((0, 0), (extra, 0)))})


@functools.cache
def _sized(model: nn.Module, capacity: int | None) -> nn.Module:
    """`model` with a decode cache of `capacity` slots, one clone per capacity.

    A model's `max_seq_len` is the only channel its layers read a cache size
    from (`dew.nn.attention.open_kv_cache`), so a per-request capacity is a
    model per capacity. The clones are kept because the model is a static
    argument of the compiled generation: one object per capacity is one
    compile per capacity rather than one per call.
    """
    if capacity is None or not any(field.name == "max_seq_len"
                                   for field in dataclasses.fields(model)):
        return model
    return model.clone(max_seq_len=capacity)


def _requested(generated: Generation, budget: int, padding: int) -> Generation:
    """`generated` cut back to the shapes the caller asked for.

    A bucket pads the prompt on the left and scans past the budget, so the
    filler comes off the front of the rows and the extra trips off the back.
    A row the scan stopped in one of those trips did not stop inside the
    budget: it reaches the caller at the budget's length, unterminated,
    which is what the unbucketed scan reports for it.
    """
    trips = generated.behavior_log_probs.shape[1]
    if not padding and trips == budget:
        return generated
    lengths = generated.lengths
    return replace(generated,
                   tokens=generated.tokens[:, padding:generated.tokens.shape[1] - trips + budget],
                   lengths=jnp.minimum(lengths, budget),
                   terminated=generated.terminated & (lengths <= budget),
                   behavior_log_probs=generated.behavior_log_probs[:, :budget],
                   raw_log_probs=generated.raw_log_probs[:, :budget])


@dataclass(frozen=True)
class TextGeneration:
    """Next-token generation bound to a decoder, its weights and its processor.

    A call runs the shared cached prefill and decode; the result is the
    `Generation` a training rollout consumes, with the actual and raw-policy
    likelihood of every drawn action. `sampling` is the policy a call uses
    when it passes none, `max_new_tokens` the budget and `n` the number of
    continuations per prompt; a loaded source fills all three from its
    generation config. `n` continuations of a prompt leave as `n` consecutive
    rows, in prompt order. Weights keep their placement: on a mesh, rows
    split over its batch axes and results keep that sharding.

    `logits` is the whole transform chain, `stopping` the criteria that run
    beside the policy's EOS one, and `strategy` the device loop. `logits=None`
    means the chain `sampling` compiles to. A call replaces each of them
    whole, so a caller that wants to add to a bound chain writes
    `logits=task.logits + (mine,)`, and an explicit `sampling=` on a call
    replaces a bound chain with its own, because the policy it overrides is
    what that chain was built from.

    A call runs at `SHAPE_BUCKETS` shapes over a cache the bucket sizes,
    and hands back the shapes the request asked for, so two requests of
    nearby lengths share one compiled executable and neither pays for the
    model's whole context.
    """

    model: nn.Module
    variables: Variables
    processor: Processor | None = None
    sampling: Sampling = Sampling()
    max_new_tokens: int | None = None
    max_length: int | None = None
    n: int = 1
    logits: tuple[LogitsTransform, ...] | None = None
    stopping: tuple[Stopping, ...] = ()
    strategy: Strategy | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "variables", freeze(dict(self.variables)))

    def bind(self, variables: Variables) -> TextGeneration:
        """The same task over other weights, such as a training policy snapshot."""
        return replace(self, variables=variables)

    @overload
    def __call__(self, request: Request, max_new_tokens: int | None = None, *, key: jax.Array,
                 n: int | None = None, sampling: Sampling | None = None,
                 images: object | None = None, logits: Transforms | None = None,
                 stopping: Criteria | None = None,
                 strategy: Strategy | None = None) -> Generation: ...

    @overload
    def __call__(self, request: Request, max_new_tokens: int | None = None, *, seed: int,
                 n: int | None = None, sampling: Sampling | None = None,
                 images: object | None = None, logits: Transforms | None = None,
                 stopping: Criteria | None = None,
                 strategy: Strategy | None = None) -> Generation: ...

    def __call__(self, request: Request, max_new_tokens: int | None = None, *,
                 key: jax.Array | None = None, seed: int | None = None, n: int | None = None,
                 sampling: Sampling | None = None, images: object | None = None,
                 logits: Transforms | None = None, stopping: Criteria | None = None,
                 strategy: Strategy | None = None) -> Generation:
        annotation = None
        if active_profile() is not None:
            annotation = jax.profiler.TraceAnnotation("inference.text")
            annotation.__enter__()
        try:
            inputs, budget, random_key = _task_inputs(self.processor, request, images=images,
                                          collective=mesh_of(self.variables) is not None,
                                          max_new_tokens=max_new_tokens, default_tokens=self.max_new_tokens,
                                          max_length=self.max_length, key=key, seed=seed)
            shaped, trips, capacity = _bucketed(inputs, budget, _ceiling(self.model))
            chain = self.logits if sampling is None else None
            generated = generate(_sized(self.model, capacity), self.variables, shaped, trips, key=random_key,
                              sampling=self.sampling if sampling is None else sampling,
                              n=self.n if n is None else n,
                              logits=chain if logits is None else logits,
                              stopping=self.stopping if stopping is None else stopping,
                              strategy=self.strategy if strategy is None else strategy)
            decoder = None if self.processor is None else functools.partial(_decoded, self.processor)
            padding = shaped.tokens.shape[1] - inputs.tokens.shape[1]
            return replace(_requested(generated, budget, padding), decoder=decoder)
        finally:
            if annotation is not None:
                annotation.__exit__(None, None, None)

    def decode(self, generation: Generation) -> tuple[str, ...]:
        """Each row's valid continuation as text; empty without a processor."""
        annotation = None
        if active_profile() is not None:
            annotation = jax.profiler.TraceAnnotation("inference.text.decode")
            annotation.__enter__()
        try:
            rows = generation.host()
            return _decoded(self.processor, rows.tokens, rows.lengths, generation.prompt_width)
        finally:
            if annotation is not None:
                annotation.__exit__(None, None, None)



@dataclass(frozen=True)
class BlockGeneration:
    """Block-diffusion generation bound to a DiffusionGemma model and its weights.

    A call runs prefill, refinement and clean-token commits as one device
    computation; the `CanvasGeneration` result carries no autoregressive
    likelihoods. `process` is the published sampler configuration used when
    a call passes none, `max_new_tokens` the budget a call omits and `n` the
    number of continuations per prompt, which leave as `n` consecutive rows
    in prompt order.
    """

    model: DiffusionGemma
    variables: Variables
    process: BlockProcess
    processor: Processor | None = None
    eos_token_ids: tuple[int, ...] = ()
    pad_token_id: int = 0
    max_new_tokens: int | None = None
    max_length: int | None = None
    n: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "variables", freeze(dict(self.variables)))

    def bind(self, variables: Variables) -> BlockGeneration:
        """The same task over other weights."""
        return replace(self, variables=variables)

    @overload
    def __call__(self, request: Request, max_new_tokens: int | None = None, *, key: jax.Array,
                 n: int | None = None, process: BlockProcess | None = None,
                 images: object | None = None) -> CanvasGeneration: ...

    @overload
    def __call__(self, request: Request, max_new_tokens: int | None = None, *, seed: int,
                 n: int | None = None, process: BlockProcess | None = None,
                 images: object | None = None) -> CanvasGeneration: ...

    def __call__(self, request: Request, max_new_tokens: int | None = None, *,
                 key: jax.Array | None = None, seed: int | None = None, n: int | None = None,
                 process: BlockProcess | None = None, images: object | None = None) -> CanvasGeneration:
        annotation = None
        if active_profile() is not None:
            annotation = jax.profiler.TraceAnnotation("inference.block")
            annotation.__enter__()
        try:
            inputs, budget, random_key = _task_inputs(self.processor, request, images=images, collective=True,
                                          max_new_tokens=max_new_tokens, default_tokens=self.max_new_tokens,
                                          max_length=self.max_length, key=key, seed=seed)
            generated = (self.process if process is None else process).generate(
                self.model, self.variables, inputs, budget, key=random_key,
                n=self.n if n is None else n,
                eos_token_ids=self.eos_token_ids, pad_token_id=self.pad_token_id)
            decoder = None if self.processor is None else functools.partial(_decoded, self.processor)
            return replace(generated, decoder=decoder)
        finally:
            if annotation is not None:
                annotation.__exit__(None, None, None)

    def decode(self, generation: CanvasGeneration) -> tuple[str, ...]:
        """Each row's valid continuation as text; empty without a processor."""
        annotation = None
        if active_profile() is not None:
            annotation = jax.profiler.TraceAnnotation("inference.block.decode")
            annotation.__enter__()
        try:
            if generation.prompt_width is None:
                raise ValueError("this generation has no prompt width")
            rows = generation.host()
            return _decoded(self.processor, rows.tokens, rows.lengths, generation.prompt_width)
        finally:
            if annotation is not None:
                annotation.__exit__(None, None, None)



@dataclass(frozen=True)
class MaskedGeneration:
    """Native MDLM sampling of the whole requested response, with a fixed prompt.

    This is not LLaDA's or Dream's source-specific remasking recipe. EOS trims
    the finished response, not the bidirectional denoising trajectory. Results
    carry refinement counts, never autoregressive action likelihoods.
    """

    model: nn.Module
    variables: Variables
    process: DiscreteProcess
    processor: Processor | None = None
    sampler: Unmask = Unmask()
    steps: int = MDLM_STEPS
    eos_token_ids: tuple[int, ...] = ()
    pad_token_id: int = 0
    max_new_tokens: int | None = None
    max_length: int | None = None
    n: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "variables", freeze(dict(self.variables)))

    def bind(self, variables: Variables) -> MaskedGeneration:
        """The same native MDLM task over another weight snapshot."""
        return replace(self, variables=variables)

    @overload
    def __call__(self, request: Request, max_new_tokens: int | None = None, *, key: jax.Array,
                 n: int | None = None, steps: int | None = None,
                 images: object | None = None) -> CanvasGeneration: ...

    @overload
    def __call__(self, request: Request, max_new_tokens: int | None = None, *, seed: int,
                 n: int | None = None, steps: int | None = None,
                 images: object | None = None) -> CanvasGeneration: ...

    def __call__(self, request: Request, max_new_tokens: int | None = None, *,
                 key: jax.Array | None = None, seed: int | None = None, n: int | None = None,
                 steps: int | None = None, images: object | None = None) -> CanvasGeneration:
        annotation = None
        if active_profile() is not None:
            annotation = jax.profiler.TraceAnnotation("inference.masked")
            annotation.__enter__()
        try:
            inputs, budget, random_key = _task_inputs(self.processor, request, images=images, collective=True,
                max_new_tokens=max_new_tokens, default_tokens=self.max_new_tokens,
                max_length=self.max_length, key=key, seed=seed)
            generated = self.process.generate(self.model, self.variables, inputs, budget, key=random_key,
                sampler=self.sampler, steps=self.steps if steps is None else steps,
                n=self.n if n is None else n,
                eos_token_ids=self.eos_token_ids, pad_token_id=self.pad_token_id)
            decoder = None if self.processor is None else functools.partial(_decoded, self.processor)
            return replace(generated, decoder=decoder)
        finally:
            if annotation is not None:
                annotation.__exit__(None, None, None)

    def decode(self, generation: CanvasGeneration) -> tuple[str, ...]:
        """Each row's valid response as text; empty without a processor."""
        annotation = None
        if active_profile() is not None:
            annotation = jax.profiler.TraceAnnotation("inference.masked.decode")
            annotation.__enter__()
        try:
            if generation.prompt_width is None:
                raise ValueError("this generation has no prompt width")
            rows = generation.host()
            return _decoded(self.processor, rows.tokens, rows.lengths, generation.prompt_width)
        finally:
            if annotation is not None:
                annotation.__exit__(None, None, None)


