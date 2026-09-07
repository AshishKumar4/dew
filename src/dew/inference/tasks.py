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

import functools
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol, overload

import jax
import numpy as np
from flax import linen as nn
from flax.core import freeze
from jax.typing import ArrayLike

from dew.diffusion.block import BlockProcess, CanvasGeneration
from dew.artifacts import agree_process_phase
from dew.nn.diffusion_gemma import DiffusionGemma
from dew.nn.inputs import ModelInputs, mesh_of, request_key
from dew.objectives.base import Variables
from dew.sampling.text import Generation, Sampling, generate

Rows = ModelInputs | ArrayLike | Sequence[Sequence[int]]
Request = str | Sequence[str] | Rows


class Processor(Protocol):
    """Host preprocessing and decoding, as a loaded source's processor does."""

    def __call__(self, text: str | Sequence[str], *, images: object | None = None) -> ModelInputs: ...

    def decode(self, tokens: ArrayLike) -> list[str]: ...


def _prepared(processor: Processor | None, request: Request, *, images: object | None) -> ModelInputs:
    """Classify raw text without changing numeric token identities or row order."""
    if isinstance(request, str):
        text = [request]
    elif isinstance(request, Sequence) and request and all(isinstance(item, str) for item in request):
        text = [item for item in request if isinstance(item, str)]
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
            if isinstance(row, Sequence):
                if any(isinstance(token, (bool, np.bool_)) or not isinstance(token, (int, np.integer))
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


@dataclass(frozen=True)
class TextGeneration:
    """Next-token generation bound to a decoder, its weights and its processor.

    A call runs the shared cached prefill and decode; the result is the
    `Generation` a training rollout consumes, with the actual and raw-policy
    likelihood of every drawn action. `sampling` is the policy a call uses
    when it passes none, and `max_new_tokens` the budget; a loaded source
    fills both from its generation config. Weights keep their placement: on
    a mesh, rows split over its batch axes and results keep that sharding.
    """

    model: nn.Module
    variables: Variables
    processor: Processor | None = None
    sampling: Sampling = Sampling()
    max_new_tokens: int | None = None
    max_length: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "variables", freeze(dict(self.variables)))

    def bind(self, variables: Variables) -> TextGeneration:
        """The same task over other weights, such as a training policy snapshot."""
        return replace(self, variables=variables)

    @overload
    def __call__(self, request: Request, max_new_tokens: int | None = None, *,
                 key: jax.Array, sampling: Sampling | None = None, images: object | None = None) -> Generation: ...

    @overload
    def __call__(self, request: Request, max_new_tokens: int | None = None, *,
                 seed: int, sampling: Sampling | None = None, images: object | None = None) -> Generation: ...

    def __call__(self, request: Request, max_new_tokens: int | None = None, *,
                 key: jax.Array | None = None, seed: int | None = None,
                 sampling: Sampling | None = None, images: object | None = None) -> Generation:
        inputs, budget, random_key = _task_inputs(self.processor, request, images=images,
                                      collective=mesh_of(self.variables) is not None,
                                      max_new_tokens=max_new_tokens, default_tokens=self.max_new_tokens,
                                      max_length=self.max_length, key=key, seed=seed)
        result = generate(self.model, self.variables, inputs, budget,
                          key=random_key, sampling=self.sampling if sampling is None else sampling)
        decoder = None if self.processor is None else functools.partial(_decoded, self.processor)
        return replace(result, decoder=decoder)

    def decode(self, generation: Generation) -> tuple[str, ...]:
        """Each row's valid continuation as text; empty without a processor."""
        rows = generation.host()
        return _decoded(self.processor, rows.tokens, rows.lengths, generation.prompt_width)


@dataclass(frozen=True)
class BlockGeneration:
    """Block-diffusion generation bound to a DiffusionGemma model and its weights.

    A call runs prefill, refinement and clean-token commits as one device
    computation; the `CanvasGeneration` result carries no autoregressive
    likelihoods. `process` is the published sampler configuration used when
    a call passes none, and `max_new_tokens` the budget a call omits.
    """

    model: DiffusionGemma
    variables: Variables
    process: BlockProcess
    processor: Processor | None = None
    eos_token_ids: tuple[int, ...] = ()
    pad_token_id: int = 0
    max_new_tokens: int | None = None
    max_length: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "variables", freeze(dict(self.variables)))

    def bind(self, variables: Variables) -> BlockGeneration:
        """The same task over other weights."""
        return replace(self, variables=variables)

    @overload
    def __call__(self, request: Request, max_new_tokens: int | None = None, *,
                 key: jax.Array, process: BlockProcess | None = None, images: object | None = None) -> CanvasGeneration: ...

    @overload
    def __call__(self, request: Request, max_new_tokens: int | None = None, *,
                 seed: int, process: BlockProcess | None = None, images: object | None = None) -> CanvasGeneration: ...

    def __call__(self, request: Request, max_new_tokens: int | None = None, *,
                 key: jax.Array | None = None, seed: int | None = None,
                 process: BlockProcess | None = None, images: object | None = None) -> CanvasGeneration:
        inputs, budget, random_key = _task_inputs(self.processor, request, images=images, collective=True,
                                      max_new_tokens=max_new_tokens, default_tokens=self.max_new_tokens,
                                      max_length=self.max_length, key=key, seed=seed)
        result = (self.process if process is None else process).generate(
            self.model, self.variables, inputs, budget,
            key=random_key, eos_token_ids=self.eos_token_ids, pad_token_id=self.pad_token_id)
        decoder = None if self.processor is None else functools.partial(_decoded, self.processor)
        return replace(result, decoder=decoder)

    def decode(self, generation: CanvasGeneration) -> tuple[str, ...]:
        """Each row's valid continuation as text; empty without a processor."""
        if generation.prompt_width is None:
            raise ValueError("this generation has no prompt width")
        rows = generation.host()
        return _decoded(self.processor, rows.tokens, rows.lengths, generation.prompt_width)
