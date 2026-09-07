"""Prompts for online RL: fixed-width rows with their reward context.

A `Prompts` source reads a parquet file or in-memory JSON rows in the verl
layout and encodes each into a left-padded `prompt` of `max_prompt_len` ids
plus `prompt_length`, and the reward columns `data_source`, `ground_truth`
and `extra_info` as fixed-width UTF-8 byte arrays, so every leaf survives
the device transfer. A row's prompt is messages rendered with the tokenizer's
chat template and generation prompt, a string encoded on its own, or token
ids used as they are. Chat messages use the SFT parser and carry optional
top-level `tools` schemas into the template. Text parts join in order;
nontext parts require a processor and are refused before truncation.
Prompts longer than the window keep their tail; the three reward columns
default to the empty string when the row lacks them.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence

import numpy as np

from dew.registry import datasets

from .chat import Conversation, load_tokenizer, render_prompt
from .dataset import (Batch, Dataset, DatasetSpec, Loading, local_batch,
                      train_stream, validation_pass)
from .tokens import bounded

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

FIELDS = ("prompt", "data_source", "ground_truth", "extra_info", "tools")
"""The row fields the source reads; anything else raises."""


@dataclasses.dataclass(frozen=True)
class _Row:
    """One row's raw prompt, tool schemas and three reward strings."""

    prompt: object
    tools: object
    data_source: str
    ground_truth: str
    extra_info: str


def _text(value: object) -> str:
    """One reward column as a string: missing is empty, anything else that is
    not a string travels as JSON."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def _list_ids(tokenizer: str, prompt: list[object], tools: object, origin: str) -> list[int]:
    """Read token ids directly, or render messages through the SFT parser."""
    ids: list[int] = []
    for token in prompt:
        if not isinstance(token, int):
            conversation = Conversation.parse(prompt, tools, origin)
            return render_prompt(load_tokenizer(tokenizer), conversation, origin)
        ids.append(token)
    if tools is not None:
        raise ValueError(f"{origin}: tools require chat messages, not pretokenized ids")
    return ids


def _prompt_ids(tokenizer: str, prompt: object, tools: object, origin: str) -> list[int]:
    """Encode a prompt, retaining structured messages until text rendering."""
    if isinstance(prompt, str):
        if tools is not None:
            raise ValueError(f"{origin}: tools require chat messages, not a plain string")
        if not prompt.strip():
            raise ValueError(f"{origin}: the prompt is blank")
        ids = load_tokenizer(tokenizer).encode(prompt, add_special_tokens=False)
    elif isinstance(prompt, list):
        ids = _list_ids(tokenizer, prompt, tools, origin)
    else:
        raise ValueError(
            f"{origin}: a prompt is messages, a string or token ids, "
            f"got {prompt!r}")
    if not ids:
        raise ValueError(f"{origin}: the prompt renders to no tokens")
    return ids


def _utf8(text: str, width: int) -> np.ndarray:
    """The string as int32 UTF-8 bytes in a zero-padded row of the scanned
    width, which fits every row by construction."""
    raw = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int32)
    out = np.zeros(width, np.int32)
    out[:len(raw)] = raw
    return out


class PromptSource:
    """Random access over normalized prompt rows, encoded on read.

    Rows come back encoded: the prompt left-padded to the window, the reward
    columns as UTF-8 bytes of the scanned width, so workers unpickle rows and
    the tokenizer path and render their own copies. The repr names the origin
    file for grain's checkpoint matching, or the record count for in-memory
    rows.
    """

    def __init__(self, rows: Sequence[Mapping[str, object]], origin: str,
                 tokenizer: str, max_prompt_len: int, pad_id: int):
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
                ground_truth=_text(row.get("ground_truth")),
                extra_info=_text(row.get("extra_info"))))
        if not normalized:
            raise ValueError(f"{origin} holds no prompts")
        self._rows = normalized
        self._origin = origin
        self._tokenizer = tokenizer
        self._window = max_prompt_len
        self._pad_id = pad_id
        self._info_width = max(
            1,
            max(len(text.encode("utf-8")) for row in normalized
                for text in (row.data_source, row.ground_truth, row.extra_info)))

    @classmethod
    def from_parquet(cls, path: str, tokenizer: str, max_prompt_len: int,
                     pad_id: int) -> PromptSource:
        """The file's rows: `prompt` is required, the reward columns ride
        along when present."""
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise ImportError(
                "reading prompt parquet needs pyarrow: pip install pyarrow") from exc
        names = [field.name for field in parquet.read_schema(path)]
        if "prompt" not in names:
            raise ValueError(
                f"{path}: the prompt column is required, the file has {names}")
        table = parquet.read_table(path, columns=[name for name in FIELDS if name in names])
        return cls(table.to_pylist(), path, tokenizer, max_prompt_len, pad_id)

    @classmethod
    def from_records(cls, records: tuple[str, ...], tokenizer: str,
                     max_prompt_len: int, pad_id: int) -> PromptSource:
        """In-memory rows as JSON, for tests and small sweeps."""
        rows = []
        for index, record in enumerate(records):
            try:
                rows.append(json.loads(record))
            except json.JSONDecodeError as exc:
                raise ValueError(f"record {index} is not JSON: {exc}") from exc
        return cls(rows, f"{len(rows)} in-memory records", tokenizer,
                   max_prompt_len, pad_id)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(origin={self._origin!r}, "
                f"tokenizer={self._tokenizer!r})")

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> Batch:
        row = self._rows[index]
        ids = _prompt_ids(self._tokenizer, row.prompt, row.tools,
                          f"{self._origin} row {index}")[-self._window:]
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
    """Prompts with their reward context, in fixed-width batches.

    `path` is a parquet file in the verl layout; `records` is JSON rows for
    tests and small sweeps; exactly one of the two is set. Each batch holds
    `prompt` left-padded to `max_prompt_len` with `pad_id`, `prompt_length`,
    and the reward columns as UTF-8 bytes. An optional `tools` column holds
    schemas as a list or JSON string for chat prompts; schemas are rendered
    into prompt tokens and never copied into the device batch. `val_path` is
    a second parquet file scored as one pass; None trains without validation.
    """

    tokenizer: str
    path: str | None = None
    records: tuple[str, ...] = ()
    val_path: str | None = None
    max_prompt_len: int = 128
    pad_id: int = 0
    val_batches: int | None = 4
    seed: int = 0
    loading: Loading = Loading()

    def load(self, *, batch: int) -> Dataset:
        if (self.path is None) == (not self.records):
            raise ValueError(
                "Prompts reads one source: --data.path names a parquet file, "
                "or records holds JSON rows")
        if self.path is not None:
            source = PromptSource.from_parquet(
                self.path, self.tokenizer, self.max_prompt_len, self.pad_id)
        else:
            source = PromptSource.from_records(
                self.records, self.tokenizer, self.max_prompt_len, self.pad_id)
        per_process = local_batch(batch)
        val = None
        if self.val_path is not None:
            val_source = PromptSource.from_parquet(
                self.val_path, self.tokenizer, self.max_prompt_len, self.pad_id)
            val = bounded(validation_pass(
                val_source, [], batch=per_process, seed=self.seed,
                loading=self.loading), self.val_batches)
        return Dataset(
            train=train_stream(source, [], batch=per_process, seed=self.seed,
                               loading=self.loading),
            val=val,
            records=len(source),
            batch=batch,
        )
