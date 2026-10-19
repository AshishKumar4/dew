"""Prompts for online RL: fixed-width rows with their reward context.

A `Prompts` source reads a parquet file or in-memory JSON rows in the verl
layout. Each row becomes a left-padded `prompt` of `max_prompt_len` ids with
its `prompt_length`, and the reward columns `data_source`, `ground_truth`
and `extra_info` as fixed-width UTF-8 byte arrays, so every leaf survives the
device transfer.

A row's prompt is messages rendered with the tokenizer's chat template and
generation prompt, a string encoded on its own, or token ids used as they
are. Chat messages use the SFT parser and carry optional top-level `tools`
schemas into the template. Text parts join in order; nontext parts require a
processor and are refused before truncation. Prompts longer than the window
keep their tail, and the three reward columns default to the empty string
when the row lacks them.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from dew.nn.inputs import token_ids
from dew.records import JSON, json_value

from .chat import Conversation, render_prompt
from .text import ByteTokenizer, HFTokenizer

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from dew.data.steps.chat import RenderPrompt

PROMPT_KEY = "prompt"
"""Batch key the prompts pipeline left-pads `[B, max_prompt_len]` ids under."""

LENGTH_KEY = "prompt_length"
"""Batch key holding each row's real token count before padding."""

SOURCE_KEY = "data_source"
"""Batch key holding the reward's rule name as UTF-8 bytes."""

TRUTH_KEY = "ground_truth"
"""Batch key holding the reference answer as UTF-8 bytes."""

INFO_KEY = "extra_info"
"""Batch key holding the reward's extra context as UTF-8 bytes."""

@dataclass(frozen=True)
class PromptRecord:
    """What a reward is told about the prompt it scores, read from the prompt's row.

    `data_source` names the row's dataset or rule, `ground_truth` is its
    reference answer as the source normalized it, and `extra_info` is any
    other context the row carries, each the empty string where the row
    lacks it. A reward reads the fields it needs.
    """

    data_source: str = ""
    ground_truth: str = ""
    extra_info: str = ""

    @classmethod
    def of(cls, data_source: JSON, ground_truth: JSON, extra_info: JSON) -> PromptRecord:
        """The record of a row's three values as a prompt batch stores them: a string is itself,
        a missing one is empty, and any other value travels as its JSON (`_text`)."""
        return cls(_text(data_source), _text(ground_truth), _text(extra_info))


FIELDS = ("prompt", "data_source", "ground_truth", "extra_info", "tools", "reward_model")
"""The fields prompt encoding reads, leaving the others unchanged.

verl's RL parquet (`rl_dataset.py`, `reward_loop/reward_manager/naive.py`
L42-L46 at 12ebe0c) nests the answer as `reward_model.ground_truth` and
keeps `tools_kwargs` inside `extra_info`."""



def _text(value: JSON) -> str:
    """One reward column as a string: missing is empty, a string is itself, and any other value
    travels as its JSON (a row's value narrowed by `dew.records.json_value` first)."""
    match value:
        case None:
            return ""
        case str():
            return value
        case _:
            return json.dumps(value, sort_keys=True)


def _ground_truth(row: Mapping[str, object], where: str) -> str:
    """The reference answer: top-level, or verl's `reward_model.ground_truth`."""
    nested = row.get("reward_model")
    if nested is None:
        return _text(json_value(row.get("ground_truth"), f"{where} ground_truth"))
    if not isinstance(nested, Mapping):
        raise ValueError(f"{where}: reward_model is an object holding ground_truth, got {nested!r}")
    if row.get("ground_truth") is not None:
        raise ValueError(f"{where}: ground_truth is both top-level and in reward_model")
    return _text(json_value(nested.get("ground_truth"), f"{where} reward_model.ground_truth"))


def _prompt_ids(tokenizer: ByteTokenizer | HFTokenizer, prompt: object, tools: object, origin: str,
                thinking: bool | None) -> list[int]:
    """Encodes a prompt, retaining structured messages until text rendering.

    A row's prompt is a string, token ids (a list, a tuple or a
    one-dimensional array, as `dew.data.text.Tokenize` makes them), or the
    messages of a conversation. A list of anything else is read as messages, which is where
    a part that is not one is named. Tools belong to messages only.
    """
    if tools is not None and not isinstance(tools, (str, list)):
        raise ValueError(
            f"{origin}: tools are a list of schemas or the JSON text of one, "
            f"got {tools!r}")
    if isinstance(prompt, str):
        if tools is not None:
            raise ValueError(f"{origin}: tools require chat messages, not a plain string")
        if not prompt.strip():
            raise ValueError(f"{origin}: the prompt is blank")
        ids = tokenizer.encode(prompt, add_special_tokens=False)
    elif isinstance(prompt, np.ndarray) or (isinstance(prompt, (list, tuple)) and all(
            isinstance(token, (int, np.integer)) for token in prompt)):
        if tools is not None:
            raise ValueError(f"{origin}: tools require chat messages, not pretokenized ids")
        ids = token_ids(prompt, where=f"{origin}: a pretokenized prompt")
    elif isinstance(prompt, list):
        conversation = Conversation.parse(prompt, tools, origin)
        if not isinstance(tokenizer, HFTokenizer):
            raise ValueError(
                f"{origin}: chat templates need a Hugging Face tokenizer, not the byte vocabulary"
            )
        ids = render_prompt(tokenizer.tokenizer, conversation, origin, thinking)
    else:
        raise ValueError(
            f"{origin}: a prompt is messages, a string or token ids, "
            f"got {prompt!r}")
    if not ids:
        raise ValueError(f"{origin}: the prompt renders to no tokens")
    return ids


def _utf8(text: str, width: int) -> np.ndarray:
    """`text` as int32 UTF-8 bytes, zero-padded to `width`.

    `width` is the longest reward string the source scanned, so every row
    fits without truncation.
    """
    raw = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int32)
    out = np.zeros(width, np.int32)
    out[:len(raw)] = raw
    return out


def encode(tokenizer: str | PreTrainedTokenizerBase, *, max_prompt_len: int = 256, pad_id: int = 0,
           long: Literal["drop", "truncate"] = "drop", thinking: bool | None = None) -> RenderPrompt:
    """A record step that renders a prompt row (verl's columns: "prompt",
    "data_source", "ground_truth", ...) to `max_prompt_len` left-padded ids,
    dropping (counted) or cutting a longer one."""
    from dew.data.steps.chat import RenderPrompt
    from dew.data.text import tokenizer_name

    return RenderPrompt(tokenizer_name(tokenizer), max_prompt_len, pad_id, long=long, thinking=thinking)
