"""Block-diffusion response rows of rendered conversations, and prompt record encodings."""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Mapping, Sequence
from typing import Literal

import grain.python as grain
import numpy as np

from dew._records import integer, record
from dew.batch import Batch
from dew.data.chat import ROLES_KEY, Role
from dew.data.facts import shaped, unshaped
from dew.data.pipeline import Rejected, cut
from dew.data.prompts import (
    FIELDS as PROMPT_FIELDS,
    INFO_KEY,
    LENGTH_KEY,
    PROMPT_KEY,
    PROMPT_LEN,
    PROMPT_LONG,
    PROMPT_PAD,
    SOURCE_KEY,
    TRUTH_KEY,
    PromptRecord,
    _prompt_ids,
    _utf8,
)
from dew.data.text import ByteTokenizer, HFTokenizer, tokenizer_for


@dataclasses.dataclass(frozen=True)
class ResponseCanvases:
    """One block-diffusion row per assistant turn of a rendered conversation.

    The row's `text` is everything before the turn, `pad_id`-padded to
    `prompt_length`, then the turn's completion over `count` canvases of
    `canvas_size`: `eos_id` fills the rest of its last canvas, `pad_id` the
    canvases after it, and `canvas_mask` marks the filled canvases, as the
    reference's `CanvasChunker` lays a response out, and `prompt_mask` the
    context's own ids, so no id the vocabulary also pads with reads as padding (gemma bf0b49901,
    `diffusion/hackable_diffusion_adapter/data/data.py`). A longer prompt or
    completion keeps its first ids, as the reference's `Pad(truncate=True)`
    and `CanvasChunker` cut them (`gm/data/_functional.py:83`). Either cut
    marks the row truncated.
    """

    prompt_length: int
    canvas_size: int
    count: int
    eos_id: int
    pad_id: int = 0

    def __post_init__(self) -> None:
        if min(self.prompt_length, self.canvas_size, self.count) < 1:
            raise ValueError("response canvases need a positive prompt_length, canvas_size and count")

    def declared(self, facts: Mapping[str, object]) -> Mapping[str, object]:
        """A row for each assistant turn: its context and completion canvases as `text`, the
        canvases filled as `canvas_mask` and the context's own ids as `prompt_mask`."""
        width = self.prompt_length + self.count * self.canvas_size
        rows = shaped(shaped(unshaped(facts, ROLES_KEY), "text", [width]), "canvas_mask",
                      [self.count * self.canvas_size])
        return shaped(rows, "prompt_mask", [self.prompt_length])

    def flat_map(self, element: Batch) -> Sequence[Batch]:
        ids = np.asarray(element["text"], np.int32)
        assistant = (np.asarray(element[ROLES_KEY]) == Role.ASSISTANT).astype(np.int8)
        edges = np.flatnonzero(np.diff(assistant, prepend=0, append=0))
        capacity = self.count * self.canvas_size
        kept = {key: value for key, value in element.items() if key not in ("text", ROLES_KEY)}
        rows = []
        for start, end in zip(edges[::2], edges[1::2], strict=True):
            response = ids[start:min(end, start + capacity)]
            filled = -(-len(response) // self.canvas_size) * self.canvas_size
            text = np.full(self.prompt_length + capacity, self.pad_id, np.int32)
            text[:min(start, self.prompt_length)] = ids[:min(start, self.prompt_length)]
            text[self.prompt_length:self.prompt_length + filled] = self.eos_id
            text[self.prompt_length:self.prompt_length + len(response)] = response
            # Each cut's record, composed with earlier cuts': the context's tokens past the prompt's
            # length, the response's past its canvases.
            removed = {"context": max(0, int(start) - self.prompt_length),
                       "response": max(0, int(end - start) - capacity)}
            rows.append(cut({**kept, "text": text, "canvas_mask": np.arange(capacity) < filled,
                             "prompt_mask": np.arange(self.prompt_length) < min(start, self.prompt_length)},
                            removed))
        return rows



def prompt_facts(facts: Mapping[str, object], tokenizer: str, max_prompt_len: int,
                 reward_bytes: int | None) -> Mapping[str, object]:
    """The facts of rendered prompts: `max_prompt_len` ids and their length, and the reward fields
    `reward_bytes` wide where known, in place of the prompt's columns."""
    rows = shaped(shaped(unshaped(facts, *PROMPT_FIELDS), PROMPT_KEY, [max_prompt_len]), LENGTH_KEY, [])
    for name in (SOURCE_KEY, TRUTH_KEY, INFO_KEY):
        rows = unshaped(rows, name) if reward_bytes is None else shaped(rows, name, [reward_bytes])
    return {**rows, "tokenizer": tokenizer}


@dataclasses.dataclass(frozen=True)
class RenderPrompt(grain.MapTransform):
    tokenizer: str
    max_prompt_len: int = PROMPT_LEN
    pad_id: int = PROMPT_PAD
    reward_bytes: int = 1024
    long: Literal["drop", "truncate"] = PROMPT_LONG
    thinking: bool | None = None
    rejects = True

    def __post_init__(self) -> None:
        if self.max_prompt_len < 1 or self.reward_bytes < 1:
            raise ValueError("prompt max_prompt_len and reward_bytes must be positive")

    @functools.cached_property
    def vocabulary(self) -> ByteTokenizer | HFTokenizer:
        """The tokenizer and its template, resolved once as `Tokenize.vocabulary`."""
        return tokenizer_for(self.tokenizer)

    def dependencies(self) -> Mapping[str, object]:
        return {"tokenizer": self.vocabulary.dependencies()}

    def declared(self, facts: Mapping[str, object]) -> Mapping[str, object]:
        return prompt_facts(facts, self.tokenizer, self.max_prompt_len, self.reward_bytes)

    def map(self, element: Batch) -> Batch | Rejected:
        element = record(element, "prompt row")
        if "prompt" not in element:
            raise ValueError("RenderPrompt needs a prompt column")
        ids = _prompt_ids(self.vocabulary, element["prompt"], element.get("tools"), "prompt", self.thinking)
        removed = max(0, len(ids) - self.max_prompt_len)
        if removed:
            if self.long == "drop":
                return Rejected("over_length")
            # The prompt keeps its end; its cut's record counts the start it removed.
            ids = ids[-self.max_prompt_len :]
        prompt = np.full(self.max_prompt_len, self.pad_id, np.int32)
        prompt[-len(ids) :] = ids
        context = PromptRecord.from_row(element).columns()
        for name, value in context.items():
            if len(value.encode("utf-8")) > self.reward_bytes:
                raise ValueError(f"prompt {name} exceeds reward_bytes={self.reward_bytes}")
        return cut({
            **{key: value for key, value in element.items() if key not in PROMPT_FIELDS},
            PROMPT_KEY: prompt,
            LENGTH_KEY: np.int32(len(ids)),
            **{name: _utf8(value, self.reward_bytes) for name, value in context.items()},
        }, {"prompt": removed})


@dataclasses.dataclass(frozen=True)
class ConfigurePrompt:
    tokenizer: str
    max_prompt_len: int
    pad_id: int
    long: Literal["drop", "truncate"]
    thinking: bool | None

    def declared(self, facts: Mapping[str, object]) -> Mapping[str, object]:
        """What the step it binds makes: its reward width the facts state, else measured."""
        width = facts.get("reward_bytes")
        stated = width if isinstance(width, int) else None
        return prompt_facts(facts, self.tokenizer, self.max_prompt_len, stated)

    def __call__(self, metadata: Mapping[str, object]) -> RenderPrompt:
        return RenderPrompt(
            self.tokenizer,
            self.max_prompt_len,
            self.pad_id,
            integer(metadata["reward_bytes"], "reward_bytes"),
            self.long,
            self.thinking,
        )
