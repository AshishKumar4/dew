"""Conversation, prompt and preference record encodings."""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Mapping
from typing import Literal

import grain.python as grain
import numpy as np

from dew.data.chat import Conversation, render_conversation
from dew.data.dataset import Batch
from dew.data.pipeline import Rejected
from dew.data.preferences import IDS_KEY, MASK_KEY, _ids, _mask
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
from dew.data.text import PinnedTokenizer, pin_tokenizer
from dew.records import integer, record


@dataclasses.dataclass(frozen=True)
class RenderChat(grain.MapTransform):
    tokenizer: str
    column: str = "messages"

    @functools.cached_property
    def pinned(self) -> PinnedTokenizer:
        """The tokenizer and template resolved once, as `Tokenize.pinned`."""
        return pin_tokenizer(self.tokenizer)

    def dependencies(self) -> Mapping[str, object]:
        return {"tokenizer": self.pinned.snapshot}

    def map(self, element: Batch) -> Batch:
        messages = element.get(self.column, element.get("prompt"))
        conversation = Conversation.parse(messages, element.get("tools"), self.tokenizer)
        ids, roles = render_conversation(self.pinned.transformers(), conversation, self.tokenizer)
        return {"text": ids, "text_roles": roles}


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
    def pinned(self) -> PinnedTokenizer:
        """The tokenizer and template resolved once, as `Tokenize.pinned`."""
        return pin_tokenizer(self.tokenizer)

    def dependencies(self) -> Mapping[str, object]:
        return {"tokenizer": self.pinned.snapshot}

    def map(self, element: Batch) -> Batch | Rejected:
        element = record(element, "prompt row")
        unknown = element.keys() - set(PROMPT_FIELDS)
        if unknown:
            raise ValueError(f"prompt unknown fields {sorted(unknown)}")
        if "prompt" not in element:
            raise ValueError("RenderPrompt needs a prompt column")
        ids = _prompt_ids(self.pinned, element["prompt"], element.get("tools"), "prompt", self.thinking)
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
        unknown = element.keys() - {"chosen", "rejected", "chosen_mask", "rejected_mask"}
        if unknown:
            raise ValueError(f"preference unknown fields {sorted(unknown)}")
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
        return {IDS_KEY: ids, MASK_KEY: mask}
