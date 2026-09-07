"""Reusable inference tasks: a model, its weights and its host processing.

A task binds what a generation needs beyond the request itself: the native
model, one immutable variables tree and, when the source ships one, the host
processor that turns text and media into `ModelInputs`. Controls stay the
typed values training already uses (`Sampling`, `BlockProcess`); results
stay the typed records the kernels produce. `bind` gives the same task over
other weights, which is how a training loop draws from a policy snapshot
without holding the trainer's mutable state.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

import jax
import numpy as np
from flax import linen as nn
from jax.typing import ArrayLike

from dew.diffusion.block import BlockProcess, CanvasGeneration
from dew.nn.diffusion_gemma import DiffusionGemma
from dew.nn.inputs import ModelInputs
from dew.objectives.base import Variables
from dew.sampling.text import Generation, Sampling, generate

Rows = ModelInputs | ArrayLike | Sequence[Sequence[int]]
Request = str | Sequence[str] | Rows


class Processor(Protocol):
    """Host preprocessing and decoding, as a loaded source's processor does."""

    def __call__(self, text: str | Sequence[str], *, images: object | None = None) -> ModelInputs: ...

    def decode(self, tokens: ArrayLike) -> list[str]: ...


def _text(request: Request) -> list[str] | None:
    """The request as prompt strings, or None when it holds prepared rows."""
    if isinstance(request, str):
        return [request]
    if isinstance(request, ModelInputs) or not isinstance(request, Sequence):
        return None
    strings = [item for item in request if isinstance(item, str)]
    return strings if strings and len(strings) == len(request) else None


def _rows(request: Request) -> Rows:
    if isinstance(request, ModelInputs) or not isinstance(request, Sequence):
        return request
    return [[int(token) for token in row] for row in request if not isinstance(row, str)]


def _prepared(processor: Processor | None, request: Request, *, images: object | None) -> ModelInputs:
    """Numeric inputs from text through the processor, or from prepared rows."""
    text = _text(request)
    if text is not None:
        if processor is None:
            raise ValueError("text requests need a processor; pass ModelInputs or token rows")
        return processor(text, images=images)
    if images is not None:
        raise ValueError("images travel with text through the processor; prepared rows carry them in ModelInputs")
    return ModelInputs.from_value(_rows(request))


def _decoded(processor: Processor | None, tokens: ArrayLike, lengths: ArrayLike, width: int) -> tuple[str, ...]:
    if processor is None:
        return ()
    rows, counts = np.asarray(tokens), np.asarray(lengths)
    return tuple(processor.decode(rows[row:row + 1, width:width + int(counts[row])])[0]
                 for row in range(rows.shape[0]))


@dataclass(frozen=True)
class TextGeneration:
    """Next-token generation bound to a decoder, its weights and its processor.

    A call runs the shared cached prefill and decode; the result is the
    `Generation` a training rollout consumes, with the actual and raw-policy
    likelihood of every drawn action. `sampling` is the policy a call uses
    when it passes none; a loaded source fills it from its generation config.
    """

    model: nn.Module
    variables: Variables
    processor: Processor | None = None
    sampling: Sampling = Sampling()

    def bind(self, variables: Variables) -> TextGeneration:
        """The same task over other weights, such as a training policy snapshot."""
        return replace(self, variables=variables)

    def __call__(self, request: Request, max_new_tokens: int, *, key: jax.Array,
                 sampling: Sampling | None = None, images: object | None = None) -> Generation:
        inputs = _prepared(self.processor, request, images=images)
        return generate(self.model, self.variables, inputs, max_new_tokens, key=key,
                        sampling=self.sampling if sampling is None else sampling)

    def decode(self, generation: Generation) -> tuple[str, ...]:
        """Each row's valid continuation as text; empty without a processor."""
        width = generation.tokens.shape[1] - generation.behavior_log_probs.shape[1]
        return _decoded(self.processor, generation.tokens, generation.lengths, width)


@dataclass(frozen=True)
class BlockGeneration:
    """Block-diffusion generation bound to a DiffusionGemma model and its weights.

    A call runs prefill, refinement and clean-token commits as one device
    computation; the `CanvasGeneration` result carries no autoregressive
    likelihoods. `process` is the published sampler configuration used when
    a call passes none.
    """

    model: DiffusionGemma
    variables: Variables
    process: BlockProcess
    processor: Processor | None = None
    eos_token_ids: tuple[int, ...] = ()
    pad_token_id: int = 0

    def bind(self, variables: Variables) -> BlockGeneration:
        """The same task over other weights."""
        return replace(self, variables=variables)

    def __call__(self, request: Request, max_new_tokens: int, *, key: jax.Array,
                 process: BlockProcess | None = None) -> CanvasGeneration:
        inputs = _prepared(self.processor, request, images=None)
        return (self.process if process is None else process).generate(
            self.model, self.variables, inputs, max_new_tokens, key=key,
            eos_token_ids=self.eos_token_ids, pad_token_id=self.pad_token_id)

    def decode(self, generation: CanvasGeneration, prompt_width: int) -> tuple[str, ...]:
        """Each row's valid continuation as text; empty without a processor."""
        return _decoded(self.processor, generation.tokens, generation.lengths, prompt_width)
