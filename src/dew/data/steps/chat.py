"""Block-diffusion response rows of rendered conversations, and prompt record encodings."""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Mapping, Sequence
from typing import Literal

import grain.python as grain
import numpy as np

from dew.data.chat import ROLES_KEY, Role
from dew.data.dataset import Batch
from dew.data.pipeline import Rejected
from dew.data.prompts import (
    FIELDS as PROMPT_FIELDS,
    INFO_KEY,
    LENGTH_KEY,
    PROMPT_KEY,
    SOURCE_KEY,
    TRUTH_KEY,
    PromptRecord,
    _ground_truth,
    _prompt_ids,
    _utf8,
)
from dew.data.text import ByteTokenizer, HFTokenizer, tokenizer_for
from dew.records import integer, json_value, record


@dataclasses.dataclass(frozen=True)
class ResponseCanvases:
    """One block-diffusion row per assistant turn of a rendered conversation.

    The row's `text` is everything before the turn, `pad_id`-padded to
    `prompt_length`, then the turn's completion over `count` canvases of
    `canvas_size`: `eos_id` fills the rest of its last canvas, `pad_id` the
    canvases after it, and `canvas_mask` marks the filled canvases, as the
    reference's `CanvasChunker` lays a response out (gemma bf0b49901,
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
            cut = start > self.prompt_length or end - start > capacity
            rows.append({**kept, "text": text, "canvas_mask": np.arange(capacity) < filled,
                         **({"_dew_truncated": True} if cut else {})})
        return rows


@dataclasses.dataclass(frozen=True)
class RenderPrompt(grain.MapTransform):
    tokenizer: str
    max_prompt_len: int = 256
    pad_id: int = 0
    reward_bytes: int = 1024
    long: Literal["drop", "truncate"] = "drop"
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

    def map(self, element: Batch) -> Batch | Rejected:
        element = record(element, "prompt row")
        if "prompt" not in element:
            raise ValueError("RenderPrompt needs a prompt column")
        ids = _prompt_ids(self.vocabulary, element["prompt"], element.get("tools"), "prompt", self.thinking)
        if len(ids) > self.max_prompt_len:
            if self.long == "drop":
                return Rejected("over_length")
            ids = ids[-self.max_prompt_len :]
        prompt = np.full(self.max_prompt_len, self.pad_id, np.int32)
        prompt[-len(ids) :] = ids
        row = PromptRecord.of(json_value(element.get("data_source"), "prompt data_source"),
                              _ground_truth(element, "prompt"),
                              json_value(element.get("extra_info"), "prompt extra_info"))
        context = {SOURCE_KEY: row.data_source, TRUTH_KEY: row.ground_truth, INFO_KEY: row.extra_info}
        for name, value in context.items():
            if len(value.encode("utf-8")) > self.reward_bytes:
                raise ValueError(f"prompt {name} exceeds reward_bytes={self.reward_bytes}")
        return {
            **{key: value for key, value in element.items() if key not in PROMPT_FIELDS},
            PROMPT_KEY: prompt,
            LENGTH_KEY: np.int32(len(ids)),
            **{name: _utf8(value, self.reward_bytes) for name, value in context.items()},
        }


@dataclasses.dataclass(frozen=True)
class ConfigurePrompt:
    tokenizer: str
    max_prompt_len: int
    pad_id: int
    long: Literal["drop", "truncate"]
    thinking: bool | None

    def __call__(self, metadata: Mapping[str, object]) -> RenderPrompt:
        return RenderPrompt(
            self.tokenizer,
            self.max_prompt_len,
            self.pad_id,
            integer(metadata["reward_bytes"], "reward_bytes"),
            self.long,
            self.thinking,
        )
