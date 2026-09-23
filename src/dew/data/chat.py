"""Supervised fine-tuning data: conversations with a role on every token.

A `ChatMessages` source reads conversations from a parquet file, a JSONL
file or a Hub dataset id, and renders each with the tokenizer's chat
template. Every token gets the role of the message that wrote it, so
`LMObjective` with `loss_role=Role.ASSISTANT` trains on assistant tokens
only.

Packing is the token pipeline's plan over the whole corpus
(`PackedWindows`) with `text_roles` as one more per-token field. A window
carries `text`, `text_roles`, `text_segment_ids`, `text_positions` and the
identical `text_roles_segment_ids`, `text_roles_positions`, all aligned. The
training stream's position is a global window count that resumes on any
process count.

Conversations are structured. A `Message` carries what the Hugging Face
chat-template contract reads: a role; content that is a string, a list of
typed parts, or nothing; an assistant's `tool_calls`; a tool response's
`tool_call_id` and `name`; and any further keys the template wants
(`reasoning_content`, `thinking`) untouched. A `Conversation` adds the
`tools` schemas the template renders into its system block. The stored
messages retain typed content. Text-tokenizer rendering joins all-text parts
into strings and refuses other part types, which require a processor.
Tool-call fields and message metadata reach the template unchanged.
"""

from __future__ import annotations

import dataclasses
import functools
import json
from collections.abc import Iterator, Mapping, Sequence
from enum import Enum
from pathlib import Path, PurePath
from typing import TYPE_CHECKING

import grain.python as pygrain
import numpy as np
from jinja2 import TemplateError

from dew.registry import datasets

from .dataset import (
    Batch,
    Dataset,
    DatasetSpec,
    Tokenize,
    describe,
    local_batch,
    train_stream,
    validation_pass,
)
from .rows import parquet_names, parquet_rows
from .sources.hf import HFOptions, HubOptions
from .text import load_tokenizer
from .tokens import PackedWindows, bounded

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

ROLES_KEY = "text_roles"
"""Batch key the chat pipeline packs `[B, seq_len + 1]` int8 roles under."""


class Role(int, Enum):
    """Who wrote a token of a rendered conversation.

    The values are the `text_roles` column: 0 pads, the rest name the message
    whose span holds the token. An assistant turn's opening header belongs to
    no span; it reads as padding, so only the completion counts. `DEVELOPER`
    is the Harmony instruction role; templates without it render the message
    to nothing, which the renderer refuses.
    """

    PAD = 0
    SYSTEM = 1
    USER = 2
    ASSISTANT = 3
    TOOL = 4
    DEVELOPER = 5


def _role(raw: object, where: str) -> Role:
    if not isinstance(raw, str):
        raise ValueError(f"{where}: a message names its role as a string, got {raw!r}")
    try:
        role = Role[raw.upper()]
    except KeyError:
        role = Role.PAD
    if role is Role.PAD:
        names = ", ".join(member.name.lower() for member in Role if member is not Role.PAD)
        raise ValueError(f"{where}: unknown role {raw!r}; the roles are {names}")
    return role


def _text(raw: object, name: str, where: str) -> str:
    if not isinstance(raw, str):
        raise ValueError(f"{where}: {name} is a string, got {raw!r}")
    return raw


def _mapping(raw: object, name: str, where: str) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where}: {name} is an object, got {raw!r}")
    return raw


def _records(raw: object, name: str, where: str) -> list[Mapping[str, object]]:
    """`raw` as a list of objects, parsing it first when it is JSON text.

    Parquet carries a structured column as JSON text when its schema varies
    across rows. Every array this file reads is an array of objects, messages,
    content parts, tool calls and tool schemas alike, so the entries are
    narrowed here and the readers below take a record rather than an unknown.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{where}: {name} is not JSON: {exc}") from None
    if not isinstance(raw, list):
        raise ValueError(f"{where}: {name} is a list, got {raw!r}")
    return [_mapping(entry, f"an entry of {name}", where) for entry in raw]


@dataclasses.dataclass(frozen=True)
class ContentPart:
    """One typed block of a message's content.

    `type` names the block; `fields` is the rest of it verbatim, `text` for
    a text block, a url or a path for media. Text rendering accepts text
    parts only; media stays available in the stored conversation.
    """

    type: str
    fields: Mapping[str, object]

    @classmethod
    def parse(cls, part: Mapping[str, object], where: str) -> ContentPart:
        kind = _text(part.get("type"), "a content part's type", where)
        fields = {key: value for key, value in part.items() if key != "type"}
        if kind == "text":
            _text(fields.get("text"), "a text part's text", where)
        return cls(kind, fields)

    def as_template(self) -> Mapping[str, object]:
        return {"type": self.type, **self.fields}


@dataclasses.dataclass(frozen=True)
class ToolCall:
    """One function call an assistant message asks for.

    `arguments` is the parsed object. A source may hold it as a JSON string,
    since parquet cannot carry a struct whose fields differ per call, while
    the Hugging Face contract hands templates a mapping. The string parses
    here, and a string that is not a JSON object fails. `id` links the call
    to the `tool_call_id` of its response where the template uses ids.
    `type` is the contract's `"function"`. A source may write the call flat
    (`{name, arguments}`) or nested under `function`; both read the same.
    """

    name: str
    arguments: Mapping[str, object]
    id: str | None = None
    type: str = "function"

    @classmethod
    def parse(cls, call: Mapping[str, object], where: str) -> ToolCall:
        outer = {key: value for key, value in call.items() if value is not None}
        function = outer.pop("function", None)
        if function is not None:
            body = dict(_mapping(function, "a tool call's function", where))
            body = {key: value for key, value in body.items() if value is not None}
        else:
            body = {key: outer.pop(key) for key in ("name", "arguments") if key in outer}
        kind = outer.pop("type", "function")
        call_id = outer.pop("id", None)
        if outer:
            raise ValueError(
                f"{where}: a tool call carries id, type and function; "
                f"unknown fields {sorted(outer)}")
        name = _text(body.pop("name", None), "a tool call's name", where)
        arguments = body.pop("arguments", None)
        if body:
            raise ValueError(
                f"{where}: a tool call's function carries name and arguments; "
                f"unknown fields {sorted(body)}")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{where}: tool call {name!r} has arguments that are not JSON: "
                    f"{exc}") from None
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, Mapping):
            raise ValueError(
                f"{where}: tool call {name!r} has arguments that are not an "
                f"object, got {arguments!r}")
        return cls(
            name, dict(arguments),
            None if call_id is None else _text(call_id, "a tool call's id", where),
            _text(kind, "a tool call's type", where))

    def as_template(self) -> Mapping[str, object]:
        call: dict[str, object] = {
            "type": self.type,
            "function": {"name": self.name, "arguments": dict(self.arguments)},
        }
        if self.id is not None:
            call["id"] = self.id
        return call


RESERVED = ("role", "content", "tool_calls", "tool_call_id", "name")
"""The message keys the boundary types; every other key rides in `extra`."""


@dataclasses.dataclass(frozen=True)
class Message:
    """One turn, in the shape the chat template reads.

    `content` is the text, the typed parts, or None when the turn is only its
    tool calls. None reaches the template as None, the wire form tool loops
    send. `tool_calls` belong to assistant turns and `tool_call_id` to tool
    responses; `name` is the function a response answers, or a participant's
    name on other roles. `extra` holds every further key verbatim, so
    `reasoning_content` or `thinking` reach the template that reads them. A
    value of None on an optional key reads as absent, which is how parquet
    spells a field a row does not have.
    """

    role: Role
    content: str | tuple[ContentPart, ...] | None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None
    extra: Mapping[str, object] = dataclasses.field(default_factory=dict)

    @classmethod
    def parse(cls, message: Mapping[str, object], where: str) -> Message:
        role = _role(message.get("role"), where)
        content = message.get("content")
        if isinstance(content, str) or content is None:
            parts = content
        else:
            parts = tuple(ContentPart.parse(part, where)
                          for part in _records(content, "content", where))
        calls = message.get("tool_calls")
        if calls is not None and role is not Role.ASSISTANT:
            raise ValueError(
                f"{where}: only an assistant message calls tools, "
                f"this one is {role.name.lower()}")
        tool_calls = () if calls is None else tuple(
            ToolCall.parse(call, where) for call in _records(calls, "tool_calls", where))
        call_id = message.get("tool_call_id")
        if call_id is not None and role is not Role.TOOL:
            raise ValueError(
                f"{where}: only a tool response carries tool_call_id, "
                f"this one is {role.name.lower()}")
        name = message.get("name")
        extra = {key: value for key, value in message.items()
                 if key not in RESERVED and value is not None}
        return cls(
            role, parts, tool_calls,
            None if call_id is None else _text(call_id, "tool_call_id", where),
            None if name is None else _text(name, "name", where),
            extra)

    def as_template(self) -> Mapping[str, object]:
        """The structured HF message, retaining content parts and metadata.

        `Conversation.text_rows` adapts these fields for text tokenizers.
        """
        if isinstance(self.content, tuple):
            content: object = [part.as_template() for part in self.content]
        else:
            content = self.content
        row: dict[str, object] = {"role": self.role.name.lower(), "content": content}
        if self.tool_calls:
            row["tool_calls"] = [call.as_template() for call in self.tool_calls]
        if self.tool_call_id is not None:
            row["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            row["name"] = self.name
        row.update(self.extra)
        return row


@dataclasses.dataclass(frozen=True)
class Conversation:
    """The messages of one row and the tool schemas its template renders.

    `tools` are the JSON schemas `apply_chat_template` takes; a source may
    hold them as a JSON string. With none, the template renders its plain
    system block.
    """

    messages: tuple[Message, ...]
    tools: tuple[Mapping[str, object], ...] = ()

    @classmethod
    def parse(cls, messages: object, tools: object = None,
              where: str = "conversation") -> Conversation:
        """One row's messages and tool schemas, as a parquet column holds them.

        Each is a list, or the JSON text a column of varying schema carries.
        """
        if not isinstance(messages, (str, list)):
            raise ValueError(
                f"{where}: messages are a list of turns or the JSON text of one, "
                f"got {messages!r}")
        if tools is not None and not isinstance(tools, (str, list)):
            raise ValueError(
                f"{where}: tools are a list of schemas or the JSON text of one, "
                f"got {tools!r}")
        parsed = tuple(Message.parse(message, f"{where} message {index}")
                       for index, message in enumerate(_records(messages, "messages", where)))
        if not parsed:
            raise ValueError(f"{where} holds an empty conversation, which has no tokens to train on")
        schemas = () if tools is None else tuple(_records(tools, "tools", where))
        return cls(parsed, schemas)

    def rows(self) -> list[Mapping[str, object]]:
        return [message.as_template() for message in self.messages]

    def text_rows(self, where: str) -> list[Mapping[str, object]]:
        """The text-tokenizer input, with all-text parts joined in order.

        The stored messages and `rows` retain their structured content.
        Media needs a processor to expand its payload into model inputs;
        a text tokenizer cannot do that, even if its template emits a marker.
        """
        rows = []
        for index, (message, row) in enumerate(zip(self.messages, self.rows(), strict=True)):
            if not isinstance(message.content, tuple):
                rows.append(row)
                continue
            texts: list[str] = []
            for part in message.content:
                at = f"{where} message {index}"
                if part.type != "text":
                    raise ValueError(
                        f"{at}: content part {part.type!r} requires a processor; "
                        "text tokenizers accept only text parts")
                texts.append(_text(part.fields.get("text"), "a text part's text", at))
            rows.append({**row, "content": "".join(texts)})
        return rows


def _token_ids(tokenizer: PreTrainedTokenizerBase, rows, tools: Sequence[Mapping[str, object]],
               generation_prompt: bool, where: str, thinking: bool | None = None) -> list[int]:
    """Renders `rows`, the messages' template dicts, to token ids.

    `thinking` sets a reasoning template's `enable_thinking` variable
    (Qwen3's switch); None leaves the template's default, and templates
    without the variable ignore it.

    A template that refuses the messages, or reads a key they lack, fails
    here with `where` and the template's own message.

    `rows` carries no annotation. Transformers declares the conversation as
    string-valued dicts while its contract reads lists under `tool_calls` and
    `content`, so the honest type has no spelling the checker accepts.
    """
    try:
        render = functools.partial(
            tokenizer.apply_chat_template, rows, tools=[dict(tool) for tool in tools] or None,
            tokenize=True, return_dict=False, add_generation_prompt=generation_prompt)
        rendered = render() if thinking is None else render(enable_thinking=thinking)
    except (TemplateError, TypeError) as exc:
        raise ValueError(f"{where}: the chat template refused the conversation: {exc}") from exc
    if not isinstance(rendered, list):
        raise ValueError(
            f"tokenizer {where!r} answered the chat template with "
            f"{type(rendered).__name__}, not token ids")
    ids: list[int] = []
    for token in rendered:
        if not isinstance(token, int):
            raise ValueError(
                f"tokenizer {where!r} answered the chat template with a "
                f"non-id {token!r}")
        ids.append(token)
    return ids


def render_prompt(tokenizer: PreTrainedTokenizerBase, conversation: Conversation,
                  where: str, thinking: bool | None = None) -> list[int]:
    """Tokenizes a text conversation with the next assistant header.

    SFT and prompt sampling use the same message conversion and tool schemas.
    All-text parts concatenate in order; nontext parts require a processor.
    `thinking` is `_token_ids`'s.
    """
    return _token_ids(tokenizer, conversation.text_rows(where), conversation.tools,
                      generation_prompt=True, where=where, thinking=thinking)


def _agreement(rendered: Sequence[int], full: Sequence[int]) -> int:
    """How many leading ids `rendered` shares with `full`."""
    count = 0
    for mine, theirs in zip(rendered, full, strict=False):
        if mine != theirs:
            break
        count += 1
    return count


def render_conversation(tokenizer: PreTrainedTokenizerBase, conversation: Conversation,
                        where: str) -> tuple[np.ndarray, np.ndarray]:
    """Token ids and per-token roles for one conversation.

    Message k's span comes from prefix rendering. An assistant turn is exact:
    `messages[:k]` with the generation prompt against `messages[:k+1]`
    without it, and both must be prefixes of the whole render. The completion
    the loss counts is then the completion the model will see. A template
    that renders the turn differently once later messages follow it, or whose
    generation prompt is not the turn's header, fails here with the message
    index rather than mis-masking it.

    Every other turn spans from where the previous turn's render ended to
    where its own render stops agreeing with the whole. That is what a
    template needs when it re-segments a run of tool responses into one
    block. A turn that renders to no tokens, or rewrites an earlier turn's
    tokens, fails. What a template emits before any message, a tools block
    for one, has no message of its own and counts as the first message's
    span.

    `where` names the tokenizer and the row for those refusals. The arrays
    are int32 ids and int8 roles.
    """
    rows = conversation.text_rows(where)
    tools = conversation.tools
    full = _token_ids(tokenizer, rows, tools, generation_prompt=False, where=where)
    roles = np.zeros(len(full), np.int8)
    agreed = 0
    for position, message in enumerate(conversation.messages):
        following = _token_ids(tokenizer, rows[:position + 1], tools, generation_prompt=False, where=where)
        if message.role is Role.ASSISTANT:
            start, end = _completion_span(tokenizer, rows, tools, full, following, agreed,
                                          position, where)
        else:
            start = agreed
            end = _turn_end(full, following, agreed, message.role, position, where)
        roles[start:end] = message.role.value
        agreed = end
    return np.asarray(full, np.int32), roles


def _completion_span(tokenizer: PreTrainedTokenizerBase, rows, tools: Sequence[Mapping[str, object]],
                     full: Sequence[int], following: Sequence[int], agreed: int,
                     position: int, where: str) -> tuple[int, int]:
    """Where assistant message `position`'s completion begins and ends in `full`.

    The generation prompt over the messages before it is that turn's opening
    header, so the completion is what follows it up to the turn's own render.
    Both renders have to be prefixes of the whole conversation, or the span
    the loss counts is not the span the model will see.
    """
    if position == 0:
        raise ValueError(
            f"{where}: the first message is an assistant turn, so its "
            "opening header cannot be separated from its completion "
            "through the template; start the conversation with a system "
            "or user message")
    prefix = _token_ids(tokenizer, rows[:position], tools, generation_prompt=True, where=where)
    if full[:len(prefix)] != prefix or len(prefix) < agreed:
        raise ValueError(
            f"tokenizer {where!r} does not render incrementally at message "
            f"{position} (assistant): the generation prompt tokenizes to "
            f"{prefix[-8:]}, the conversation to "
            f"{full[max(0, len(prefix) - 8):len(prefix)]}")
    if full[:len(following)] != following:
        raise ValueError(
            f"tokenizer {where!r} renders message {position} (assistant) "
            "differently once later messages follow it, so its completion "
            "cannot be masked from the whole conversation")
    return len(prefix), len(following)


def _turn_end(full: Sequence[int], following: Sequence[int], agreed: int, role: Role,
              position: int, where: str) -> int:
    """Where message `position`'s span ends in `full`, for a turn that is not
    an assistant's.

    The span runs from where the previous turn ended to where this turn's own
    render stops agreeing with the whole conversation. A turn that rewrites
    an earlier turn's tokens, or renders to none of its own, fails here.
    """
    end = _agreement(following, full)
    if end < agreed:
        raise ValueError(
            f"tokenizer {where!r} rewrites message {position - 1}'s tokens when "
            f"message {position} ({role.name.lower()}) follows it")
    if end == agreed:
        raise ValueError(
            f"tokenizer {where!r} renders message {position} ({role.name.lower()}) "
            "to no tokens; the template does not read this role or content")
    return end


def _conversation_column(names: Sequence[str], column: str, where: str) -> str:
    """Which column of a source holds the conversations.

    The one named, or `prompt` where the rows carry that instead, which is
    what the verl layout and every file `dew.data.prompts` reads call it.
    """
    for name in (column, "prompt"):
        if name in names:
            return name
    raise ValueError(f"{where}: no {column!r} or 'prompt' column of conversations, "
                     f"only {list(names)}")


def _parquet_conversations(path: str, column: str) -> tuple[list, list]:
    """One parquet file, with only the two columns this reads taken off it."""
    names = parquet_names(path, "chat")
    held = _conversation_column(names, column, path)
    rows = parquet_rows(path, [held, "tools"], names)
    return ([row[held] for row in rows], [row.get("tools") for row in rows])


def _jsonl_conversations(path: str, column: str) -> tuple[list, list]:
    """One JSON object per line, the form a chat corpus is published in.

    Lines are read one at a time and only the two keys this needs are kept,
    so a corpus larger than memory costs its conversations and nothing else.
    """
    conversations: list = []
    tools: list = []
    with open(path, encoding="utf-8") as lines:
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            where = f"{path} line {number}"
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"{where}: a row is a JSON object of columns, "
                                 f"not a {type(row).__name__}")
            conversations.append(row[_conversation_column(list(row), column, where)])
            tools.append(row.get("tools"))
    return conversations, tools


def _hub_conversations(path: str, split: str, options: HFOptions, column: str) -> tuple[list, list]:
    """One Hub split, through `datasets.load_dataset` on the library's terms.

    `options` is the value the `hf` provider forwards, so the cache, the
    config name, a revision and a token are the library's own arguments. The
    load names one split and asks for a table, so a directory of splits or a
    streamed split is refused here rather than indexed into. The import is
    the load's own, which has already raised the missing-extra message.
    """
    loaded = options.load(path, split, streaming=False)
    from datasets import Dataset as ArrowDataset
    if not isinstance(loaded, ArrowDataset):
        raise TypeError(
            f"{path!r} split {split!r} loaded as {type(loaded).__name__}; conversations are "
            f"read off one Arrow-backed split, so name one split of the dataset")
    names = list(loaded.column_names)
    held = _conversation_column(names, column, f"{path} split {split!r}")
    return (list(loaded[held]),
            list(loaded["tools"]) if "tools" in names else [None] * loaded.num_rows)


class ConversationSource:
    """Reads the conversations at `path` by index, with their tool schemas
    beside them where the rows carry any.

    Three things hold conversations and one reader takes all three: a parquet
    file, a `.jsonl` file, and a Hub dataset id resolved through `HFOptions`.
    The suffix decides which. `chat.jsonl` is lines, `chat.parquet` is a
    table, an existing file without either suffix is a table too, and
    anything else is a repo id at `split`.

    One record is one conversation, a list of messages in the verl layout,
    and its tool schemas, a list or a JSON string. The rows are read once and
    come back as plain dicts, which pickle across to grain workers. Other
    columns are not read.
    """

    def __init__(self, path: str, *, column: str = "messages", split: str = "train",
                 options: HFOptions | None = None):
        self.path = path
        self.column = column
        self.split = split
        self.options = options if options is not None else HFOptions()
        suffix = PurePath(path).suffix
        if suffix == ".jsonl":
            conversations, tools = _jsonl_conversations(path, column)
        elif suffix == ".parquet" or Path(path).is_file():
            conversations, tools = _parquet_conversations(path, column)
        else:
            conversations, tools = _hub_conversations(path, split, self.options, column)
        if not conversations:
            raise ValueError(f"{path} holds no conversations")
        self._conversations = conversations
        self._tools = tools

    def __repr__(self) -> str:
        # The description a saved position compares against (`describe`). A
        # repo id names which rows only with its split and load arguments.
        return (f"{self.__class__.__name__}(path={self.path!r}, column={self.column!r}, "
                f"split={self.split!r}, options={self.options!r})")

    def __len__(self) -> int:
        return len(self._conversations)

    def __iter__(self) -> Iterator[Batch]:
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: int) -> Batch:
        return {"messages": self._conversations[index], "tools": self._tools[index]}


def _rendered(tokenizer: PreTrainedTokenizerBase, record: Batch,
              where: str) -> tuple[np.ndarray, np.ndarray]:
    """One source row's token ids and per-token roles.

    The row is parsed and rendered in one place, so the length pass and the
    render map read a conversation the same way.
    """
    return render_conversation(
        tokenizer, Conversation.parse(record["messages"], record["tools"], where), where)


class RenderConversation:
    """Turns source rows into rendered ids and per-token roles, for
    `map_with_index`.

    Holds only the tokenizer path, so grain workers unpickle the name and
    load their own copy. Failures name the tokenizer and the row.
    """

    def __init__(self, tokenizer: str):
        self.tokenizer = tokenizer

    def __call__(self, index: int, record: Batch) -> Batch:
        where = f"{self.tokenizer} row {index}"
        ids, roles = _rendered(load_tokenizer(self.tokenizer), record, where)
        return {"text": ids, ROLES_KEY: roles}


def _lengths(source: ConversationSource, tokenizer: str) -> list[int]:
    """Every row's rendered length, validating the file up front.

    The chunk table needs the lengths before the lazy render map runs, so
    this pass renders each row once and keeps only its count. A bad row
    fails the run here with its index.
    """
    load = load_tokenizer(tokenizer)
    return [len(_rendered(load, record, f"{source.path} row {index}")[0])
            for index, record in enumerate(source)]


@datasets("chat_messages")
@dataclasses.dataclass(frozen=True)
class ChatMessages(DatasetSpec):
    """Renders conversations with the tokenizer's chat template and packs them.

    `path` names the conversations: a parquet file, a `.jsonl` file, or a Hub
    dataset id read at `split` through `options`, which is the same value the
    `hf` provider forwards to `datasets.load_dataset`. Whichever it is, the
    rows carry lists of messages under `column` (or `prompt`, which is what
    the verl layout calls it) and their tool schemas under `tools` where they
    have any. `tokenizer` is the hub name or local path whose chat template
    renders them.

    Each conversation, in chunks when it outgrows the window, is one element
    the packing plan adds to the first window with room. Every window carries
    `text_roles` beside the ids, so the loss can count one role's targets.
    The plan runs over the whole corpus in row order, ahead of the shard, as
    `PackedTokens` plans its documents, so `records` is the windows of a pass
    exactly and a saved position is a global window count.

    `val_path` is a second source of the same three kinds, read at
    `val_split` and scored as one pass; None trains without validation.
    """

    tokenizer: str
    path: str | None = None
    val_path: str | None = None
    column: str = "messages"
    """The column of conversations; `prompt` is read where a row has that."""
    split: str = "train"
    """Which split `path` is read at, when it names a Hub dataset."""
    val_split: str | None = None
    """Which split `val_path` is read at; None reads `split`."""
    options: HubOptions = HFOptions()
    """What `datasets.load_dataset` takes beside the id and the split."""
    seq_len: int = 256
    val_batches: int | None = 4
    packing_bins: int = 8

    def load(self, *, batch: int, tokenize: Tokenize | None = None) -> Dataset:
        self.uncaptioned(tokenize)
        if self.path is None:
            raise ValueError(
                "ChatMessages reads a parquet file, a .jsonl file or a hub dataset id: "
                "--data.path names it")
        rows, window = local_batch(batch), self.seq_len + 1

        def packed(path: str, split: str) -> PackedWindows:
            source = ConversationSource(path, column=self.column, split=split,
                                        options=self.options)
            rendered = pygrain.MapDataset.source(source).map_with_index(
                RenderConversation(self.tokenizer))
            return PackedWindows(rendered, _lengths(source, self.tokenizer), window,
                                 self.packing_bins,
                                 f"{describe(source)} rendered by {self.tokenizer!r}")

        train = packed(self.path, self.split)
        validation = None
        if self.val_path is not None:
            scored = packed(self.val_path, self.val_split or self.split)
            validation = bounded(validation_pass(scored, [], batch=rows, seed=self.seed,
                                                 loading=self.loading),
                                 self.val_batches)
        return Dataset(
            train=train_stream(train, [], batch=rows, seed=self.seed, loading=self.loading),
            val=validation,
            records=len(train),
            batch=batch,
        )
