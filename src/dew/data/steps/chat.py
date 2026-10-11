"""Conversation, prompt and preference record encodings."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Literal

import grain.python as grain
import numpy as np

from dew.data.chat import Conversation, render_conversation
from dew.data.dataset import Batch
from dew.data.pipeline import Rejected
from dew.data.preferences import IDS_KEY, MASK_KEY, _ids, _mask
from dew.data.prompts import (
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
from dew.data.text import load_tokenizer, tokenizer_snapshot


@dataclasses.dataclass(frozen=True)
class RenderChat(grain.MapTransform):
    tokenizer: str
    column: str = "messages"

    def dependencies(self) -> Mapping[str, object]:
        return tokenizer_snapshot(self.tokenizer)

    def map(self, element: Batch) -> Batch:
        messages = element.get(self.column, element.get("prompt"))
        conversation = Conversation.parse(messages, element.get("tools"), self.tokenizer)
        ids, roles = render_conversation(load_tokenizer(self.tokenizer), conversation, self.tokenizer)
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

    def dependencies(self) -> Mapping[str, object]:
        return tokenizer_snapshot(self.tokenizer)

    def map(self, element: Batch) -> Batch | Rejected:
        if "prompt" not in element:
            raise ValueError("RenderPrompt needs a prompt column")
        ids = _prompt_ids(self.tokenizer, element["prompt"], element.get("tools"), "prompt", self.thinking)
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
class PreferencePair(grain.MapTransform):
    seq_len: int = 256
    pad_id: int = 0

    def map(self, element: Batch) -> Batch:
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
