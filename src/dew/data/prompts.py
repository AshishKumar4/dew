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

import dataclasses
import json
from collections.abc import Mapping, Sequence

import numpy as np

from dew.registry import datasets

from .chat import Conversation, render_prompt
from .dataset import Batch, Dataset, DatasetSpec, Tokenize
from .rows import json_records, parquet_names, parquet_rows, row_dataset
from .text import load_tokenizer

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


@dataclasses.dataclass(frozen=True)
class _Row:
    """One row's raw prompt, tool schemas and three reward strings."""

    prompt: object
    tools: object
    data_source: str
    ground_truth: str
    extra_info: str


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


def _prompt_ids(tokenizer: str, prompt: object, tools: object, origin: str,
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
        ids = load_tokenizer(tokenizer).encode(prompt, add_special_tokens=False)
    elif isinstance(prompt, list):
        ids = [token for token in prompt if isinstance(token, int)]
        if len(ids) != len(prompt):
            conversation = Conversation.parse(prompt, tools, origin)
            ids = render_prompt(load_tokenizer(tokenizer), conversation, origin, thinking)
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


class PromptSource:
    """Reads normalized prompt rows by index, encoding each on read.

    A row comes back as the prompt left-padded to the window and the reward
    columns as UTF-8 bytes of the scanned width. Workers unpickle the rows
    and the tokenizer path and render their own copies. The repr names the
    origin file, or the record count for in-memory rows, which is what a
    saved position compares against (`describe`).
    """

    def __init__(self, rows: Sequence[object], origin: str,
                 tokenizer: str, max_prompt_len: int, pad_id: int,
                 thinking: bool | None = None):
        if max_prompt_len < 1:
            raise ValueError(
                f"max_prompt_len is {max_prompt_len}: prompts need at least one token")
        normalized: list[_Row] = []
        for index, row in enumerate(rows):
            where = f"{origin} row {index}"
            if not isinstance(row, Mapping):
                raise ValueError(f"{where}: a row is an object, got {row!r}")
            unknown = sorted(key for key in row if key not in FIELDS)
            if unknown:
                raise ValueError(
                    f"{where}: unknown fields {unknown}; the fields are {list(FIELDS)}")
            if row.get("prompt") is None:
                raise ValueError(
                    f"{where}: a row without a prompt has nothing to sample from")
            normalized.append(_Row(
                prompt=row["prompt"],
                tools=row.get("tools"),
                data_source=_text(row.get("data_source")),
                ground_truth=_ground_truth(row, where),
                extra_info=_text(row.get("extra_info"))))
        if not normalized:
            raise ValueError(f"{origin} holds no prompts")
        self._rows = normalized
        self._origin = origin
        self._tokenizer = tokenizer
        self._thinking = thinking
        self._window = max_prompt_len
        self._pad_id = pad_id
        self._info_width = max(
            1,
            max(len(text.encode("utf-8")) for row in normalized
                for text in (row.data_source, row.ground_truth, row.extra_info)))

    @classmethod
    def from_parquet(cls, path: str, tokenizer: str, max_prompt_len: int,
                     pad_id: int, thinking: bool | None = None) -> PromptSource:
        """The parquet file's rows.

        `prompt` is required; the reward columns ride along when present.
        """
        names = parquet_names(path, "prompt")
        if "prompt" not in names:
            raise ValueError(
                f"{path}: the prompt column is required, the file has {names}")
        return cls(parquet_rows(path, FIELDS, names), path, tokenizer, max_prompt_len, pad_id, thinking)

    @classmethod
    def from_records(cls, records: tuple[str, ...], tokenizer: str,
                     max_prompt_len: int, pad_id: int,
                     thinking: bool | None = None) -> PromptSource:
        """In-memory rows as JSON, for tests and small sweeps."""
        rows = json_records(records)
        return cls(rows, f"{len(rows)} in-memory records", tokenizer,
                   max_prompt_len, pad_id, thinking)

    def __repr__(self) -> str:
        # A saved position compares this; the switch renders other ids, so it counts when set.
        thinking = "" if self._thinking is None else f", thinking={self._thinking!r}"
        return (f"{self.__class__.__name__}(origin={self._origin!r}, "
                f"tokenizer={self._tokenizer!r}{thinking})")

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> Batch:
        row = self._rows[index]
        ids = _prompt_ids(self._tokenizer, row.prompt, row.tools,
                          f"{self._origin} row {index}", self._thinking)[-self._window:]
        prompt = np.full(self._window, self._pad_id, np.int32)
        prompt[self._window - len(ids):] = ids
        return {
            PROMPT_KEY: prompt,
            LENGTH_KEY: np.int32(len(ids)),
            SOURCE_KEY: _utf8(row.data_source, self._info_width),
            TRUTH_KEY: _utf8(row.ground_truth, self._info_width),
            INFO_KEY: _utf8(row.extra_info, self._info_width),
        }


@datasets("prompts")
@dataclasses.dataclass(frozen=True)
class Prompts(DatasetSpec):
    """Reads prompts with their reward context, in fixed-width batches.

    `path` is a parquet file in the verl layout and `records` is JSON rows
    for tests and small sweeps; exactly one of the two is set. Each batch
    holds `prompt` left-padded to `max_prompt_len` with `pad_id`,
    `prompt_length`, and the reward columns as UTF-8 bytes.

    An optional `tools` column holds schemas as a list or JSON string for
    chat prompts. Schemas are rendered into prompt tokens and never copied
    into the device batch. `val_path` is a second parquet file scored as one
    pass; None trains without validation. `thinking` sets a reasoning
    template's `enable_thinking` (Qwen3's switch); None keeps its default.
    """

    tokenizer: str
    path: str | None = None
    records: tuple[str, ...] = ()
    val_path: str | None = None
    max_prompt_len: int = 128
    pad_id: int = 0
    val_batches: int | None = 4
    thinking: bool | None = None

    def load(self, *, batch: int, tokenize: Tokenize | None = None) -> Dataset:
        self.uncaptioned(tokenize)
        return row_dataset(
            self, batch=batch, path=self.path, records=self.records,
            val_path=self.val_path, val_batches=self.val_batches,
            from_parquet=lambda path: PromptSource.from_parquet(
                path, self.tokenizer, self.max_prompt_len, self.pad_id, self.thinking),
            from_records=lambda records: PromptSource.from_records(
                tuple(records), self.tokenizer, self.max_prompt_len, self.pad_id, self.thinking))
