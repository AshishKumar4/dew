"""Conversation, prompt and preference record encodings."""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Mapping, Sequence
from typing import Literal

import grain.python as grain
import numpy as np

from dew.data.chat import ROLES_KEY, Conversation, Role, render_conversation
from dew.data.dataset import Batch
from dew.data.pipeline import Rejected
from dew.data.preferences import FIELDS as PREFERENCE_FIELDS, IDS_KEY, MASK_KEY, _ids, _mask
from dew.data.prompts import (
    FIELDS as PROMPT_FIELDS,
    INFO_KEY,
    LENGTH_KEY,
    PROMPT_KEY,
    SOURCE_KEY,
    TRUTH_KEY,
    _ground_truth,
    _prompt_ids,
    _text,
    _utf8,
)
from dew.data.text import ByteTokenizer, HFTokenizer, tokenizer_for
from dew.records import integer, record


@dataclasses.dataclass(frozen=True)
class RenderChat(grain.MapTransform):
    tokenizer: str
    column: str = "messages"

    @functools.cached_property
    def vocabulary(self) -> HFTokenizer:
        """The tokenizer and its template, resolved once as `Tokenize.vocabulary`."""
        return HFTokenizer(self.tokenizer)

    def dependencies(self) -> Mapping[str, object]:
        return {"tokenizer": self.vocabulary.dependencies()}

    def map(self, element: Batch) -> Batch:
        messages = element.get(self.column, element.get("prompt"))
        conversation = Conversation.parse(messages, element.get("tools"), self.tokenizer)
        ids, roles = render_conversation(self.vocabulary.tokenizer, conversation, self.tokenizer)
        kept = {key: value for key, value in element.items() if key not in (self.column, "prompt", "tools")}
        return {**kept, "text": ids, "text_roles": roles}


@dataclasses.dataclass(frozen=True)
class ResponseCanvases:
    """One block-diffusion row per assistant turn of a rendered conversation.

    The row's `text` is everything before the turn, `pad_id`-padded to
    `prompt_length`, then the turn's completion over `count` canvases of
    `canvas_size`: `eos_id` fills the rest of its last canvas, `pad_id` the
    canvases after it, and `canvas_mask` marks the filled canvases, as the
    reference's `CanvasChunker` lays a response out (gemma bf0b49901,
    `diffusion/hackable_diffusion_adapter/data/data.py`). A longer prompt
    keeps its last ids, which hold the turn's header; a longer completion
    keeps its first, as the reference cuts it. Either cut marks the row
    truncated.
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
            text[:min(start, self.prompt_length)] = ids[max(0, start - self.prompt_length):start]
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
        context = {
            SOURCE_KEY: _text(element.get("data_source")),
            TRUTH_KEY: _ground_truth(element, "prompt"),
            INFO_KEY: _text(element.get("extra_info")),
        }
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


@dataclasses.dataclass(frozen=True)
class PreferencePair(grain.MapTransform):
    seq_len: int = 256
    pad_id: int = 0

    def __post_init__(self) -> None:
        if self.seq_len < 1:
            raise ValueError("preference seq_len must be positive")

    def map(self, element: Batch) -> Batch:
        element = record(element, "preference row")
        if "chosen" not in element or "rejected" not in element:
            raise ValueError("a preference pair needs both chosen and rejected ids")
        ids = np.full((2, self.seq_len), self.pad_id, np.int32)
        mask = np.zeros((2, self.seq_len), np.int32)
        for row, name in enumerate(("chosen", "rejected")):
            tokens = _ids(element.get(name), name, "preference")
            if len(tokens) > self.seq_len:
                raise ValueError(f"preference {name} has {len(tokens)} tokens for seq_len={self.seq_len}")
            completion = _mask(
                element.get(f"{name}_mask", [1] * len(tokens)), f"{name}_mask", len(tokens), "preference"
            )
            ids[row, : len(tokens)], mask[row, : len(tokens)] = tokens, completion
        kept = {key: value for key, value in element.items() if key not in PREFERENCE_FIELDS}
        return {**kept, IDS_KEY: ids, MASK_KEY: mask}
