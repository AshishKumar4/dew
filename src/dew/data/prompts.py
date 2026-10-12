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

import numpy as np

from .chat import Conversation, render_prompt
from .text import ByteTokenizer, HFTokenizer

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

FIELDS = ("prompt", "data_source", "ground_truth", "extra_info", "tools", "reward_model", "ability",
          "agent_name")
"""The row fields the source reads; anything else raises.

verl's RL parquet (`rl_dataset.py`, `reward_loop/reward_manager/naive.py`
L42-L46 at 12ebe0c) nests the answer as `reward_model.ground_truth` and
keeps `tools_kwargs` inside `extra_info`. `ability` and `agent_name` (which
of verl's agent loops runs the row, a choice a Dew rollout source makes) are
read and dropped."""



def _text(value: object) -> str:
    """One reward column as a string.

    Missing is empty and a string is itself. Anything else travels as JSON,
    and a value that is not JSON is refused.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        raise ValueError(
            f"a reward column holds text or a JSON value, got {value!r}") from None


def _ground_truth(row: Mapping[str, object], where: str) -> str:
    """The reference answer: top-level, or verl's `reward_model.ground_truth`."""
    nested = row.get("reward_model")
    if nested is None:
        return _text(row.get("ground_truth"))
    if not isinstance(nested, Mapping):
        raise ValueError(f"{where}: reward_model is an object holding ground_truth, got {nested!r}")
    if row.get("ground_truth") is not None:
        raise ValueError(f"{where}: ground_truth is both top-level and in reward_model")
    return _text(nested.get("ground_truth"))


def _prompt_ids(tokenizer: ByteTokenizer | HFTokenizer, prompt: object, tools: object, origin: str,
                thinking: bool | None) -> list[int]:
    """Encodes a prompt, retaining structured messages until text rendering.

    A row's prompt is a string, a list of token ids, or the messages of a
    conversation. A list of anything else is read as messages, which is where
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
    elif isinstance(prompt, list):
        ids = [token for token in prompt if isinstance(token, int)]
        if len(ids) != len(prompt):
            conversation = Conversation.parse(prompt, tools, origin)
            if not isinstance(tokenizer, HFTokenizer):
                raise ValueError(
                    f"{origin}: chat templates need a Hugging Face tokenizer, not the byte vocabulary"
                )
            ids = render_prompt(tokenizer.tokenizer, conversation, origin, thinking)
        elif tools is not None:
            raise ValueError(f"{origin}: tools require chat messages, not pretokenized ids")
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


