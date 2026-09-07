"""Supervised fine-tuning data: conversations with a role on every token.

A `ChatMessages` source reads a parquet file of conversations and renders
each with the tokenizer's chat template. Every token gets the role of the
message that wrote it, so `LMObjective` with `loss_role=Role.ASSISTANT`
trains on assistant tokens only. Packing reuses the token pipeline's
first-fit bins with `text_roles` as one more per-token feature. Grain emits
segment ids and positions per packed feature, so a bin carries `text`,
`text_roles`, `text_segment_ids`, `text_positions` and the identical
`text_roles_segment_ids`, `text_roles_positions`, all aligned.

Conversations are structured. A `Message` carries what the Hugging Face
chat-template contract reads: a role, content that is a string, a list of
typed parts, or nothing, an assistant's `tool_calls`, a tool response's
`tool_call_id` and `name`, and any further keys the template wants
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
import threading
from collections.abc import Iterator, Mapping, Sequence
from enum import Enum
from typing import TYPE_CHECKING

import grain.python as pygrain
import jax
import numpy as np
from jinja2 import TemplateError

from dew.registry import datasets

from .dataset import Batch, Dataset, DatasetSpec, Loading, local_batch
from .tokens import DocumentChunks, bounded, chunk_counts

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


def _sequence(raw: object, name: str, where: str) -> list[object]:
    """A list, or a JSON string holding one: parquet carries structured
    columns as JSON text when their schema varies across rows."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{where}: {name} is not JSON: {exc}") from None
    if not isinstance(raw, list):
        raise ValueError(f"{where}: {name} is a list, got {raw!r}")
    return raw


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
    def parse(cls, raw: object, where: str) -> ContentPart:
        part = _mapping(raw, "a content part", where)
        kind = _text(part.get("type"), "a content part's type", where)
        fields = {key: value for key, value in part.items() if key != "type"}
        if kind == "text":
            _text(fields.get("text"), "a text part's text", where)
        return cls(kind, fields)

    def as_template(self) -> dict[str, object]:
        return {"type": self.type, **self.fields}


@dataclasses.dataclass(frozen=True)
class ToolCall:
    """One function call an assistant message asks for.

    `arguments` is the parsed object. A source may hold it as a JSON string,
    since parquet cannot carry a struct whose fields differ per call, and
    the Hugging Face contract hands templates a mapping; the string parses
    here and a string that is not a JSON object fails. `id` links the call
    to the `tool_call_id` of its response where the template uses ids.
    `type` is the contract's `"function"`. A source may write the call flat
    (`{name, arguments}`) or nested under `function`; both read the same.
    """

    name: str
    arguments: Mapping[str, object]
    id: str | None = None
    type: str = "function"

    @classmethod
    def parse(cls, raw: object, where: str) -> ToolCall:
        call = _mapping(raw, "a tool call", where)
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

    def as_template(self) -> dict[str, object]:
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

    `content` is the text, the typed parts, or None when the turn is only
    its tool calls; None reaches the template as None, the wire form tool
    loops send. `tool_calls` belong to assistant turns and `tool_call_id` to
    tool responses; `name` is the function a response answers, or a
    participant's name on other roles. `extra` holds every further key
    verbatim, so `reasoning_content` or `thinking` reach the template that
    reads them. A value of None on an optional key reads as absent, which is
    how parquet spells a field a row does not have.
    """

    role: Role
    content: str | tuple[ContentPart, ...] | None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None
    extra: Mapping[str, object] = dataclasses.field(default_factory=dict)

    @classmethod
    def parse(cls, raw: object, where: str) -> Message:
        message = _mapping(raw, "a message", where)
        role = _role(message.get("role"), where)
        content = message.get("content")
        if isinstance(content, str) or content is None:
            parts = content
        else:
            parts = tuple(ContentPart.parse(part, where)
                          for part in _sequence(content, "content", where))
        calls = message.get("tool_calls")
        if calls is not None and role is not Role.ASSISTANT:
            raise ValueError(
                f"{where}: only an assistant message calls tools, "
                f"this one is {role.name.lower()}")
        tool_calls = () if calls is None else tuple(
            ToolCall.parse(call, where) for call in _sequence(calls, "tool_calls", where))
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

    def as_template(self) -> dict[str, object]:
        """The structured HF message, retaining content parts and metadata.
        Conversation.text_rows adapts these fields for text tokenizers."""
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
        parsed = tuple(Message.parse(message, f"{where} message {index}")
                       for index, message in enumerate(_sequence(messages, "messages", where)))
        if not parsed:
            raise ValueError(f"{where} holds an empty conversation, which has no tokens to train on")
        schemas = () if tools is None else tuple(
            _mapping(tool, "a tool schema", where) for tool in _sequence(tools, "tools", where))
        return cls(parsed, schemas)

    def rows(self) -> list[dict[str, object]]:
        return [message.as_template() for message in self.messages]

    def text_rows(self, source: str) -> list[dict[str, object]]:
        """The text-tokenizer input, with all-text parts joined in order.

        The stored messages and `rows` retain their structured content.
        Media needs a processor to expand its payload into model inputs;
        a text tokenizer cannot do that, even if its template emits a marker.
        """
        rows = self.rows()
        for index, (message, row) in enumerate(zip(self.messages, rows, strict=True)):
            if isinstance(message.content, tuple):
                texts: list[str] = []
                for part in message.content:
                    where = f"{source} message {index}"
                    if part.type != "text":
                        raise ValueError(
                            f"{where}: content part {part.type!r} requires a processor; "
                            "text tokenizers accept only text parts")
                    texts.append(_text(part.fields.get("text"), "a text part's text", where))
                row["content"] = "".join(texts)
        return rows


def _token_ids(tokenizer: PreTrainedTokenizerBase, rows, tools: Sequence[Mapping[str, object]],
               generation_prompt: bool, source: str) -> list[int]:
    """`rows`, the messages' template dicts, through the chat template to
    token ids. A template that refuses the messages, or reads a key they
    lack, fails here with the source and the template's own message.

    `rows` carries no annotation: transformers declares the conversation
    as string-valued dicts while its contract reads lists under
    `tool_calls` and `content`, so the honest type has no spelling the
    checker accepts."""
    try:
        rendered = tokenizer.apply_chat_template(
            rows, tools=[dict(tool) for tool in tools] or None, tokenize=True,
            return_dict=False, add_generation_prompt=generation_prompt)
    except (TemplateError, TypeError) as exc:
        raise ValueError(f"{source}: the chat template refused the conversation: {exc}") from exc
    if not isinstance(rendered, list):
        raise ValueError(
            f"tokenizer {source!r} answered the chat template with "
            f"{type(rendered).__name__}, not token ids")
    ids: list[int] = []
    for token in rendered:
        if not isinstance(token, int):
            raise ValueError(
                f"tokenizer {source!r} answered the chat template with a "
                f"non-id {token!r}")
        ids.append(token)
    return ids


def render_prompt(tokenizer: PreTrainedTokenizerBase, conversation: Conversation,
                  source: str) -> list[int]:
    """Tokenize a text conversation with the next assistant header.

    SFT and prompt sampling use the same message conversion and tool schemas.
    All-text parts concatenate in order; nontext parts require a processor.
    """
    return _token_ids(tokenizer, conversation.text_rows(source), conversation.tools,
                      True, source)


def _agreement(rendered: Sequence[int], full: Sequence[int]) -> int:
    """How many leading ids `rendered` shares with `full`."""
    count = 0
    for mine, theirs in zip(rendered, full):
        if mine != theirs:
            break
        count += 1
    return count


def render_conversation(tokenizer: PreTrainedTokenizerBase, conversation: Conversation,
                        source: str) -> tuple[np.ndarray, np.ndarray]:
    """Token ids and per-token roles for one conversation.

    Message k's span comes from prefix rendering. An assistant turn is
    exact: `messages[:k]` with the generation prompt against `messages[:k+1]`
    without it, and both must be prefixes of the whole render, so the
    completion the loss counts is the completion the model will see. A
    template that renders the turn differently once later messages follow
    it, or whose generation prompt is not the turn's header, fails here
    with the message index rather than mis-masking it. Every other turn
    spans from where the previous turn's render ended to where its own
    render stops agreeing with the whole, which is what a template needs
    when it re-segments a run of tool responses into one block; a turn that
    renders to no tokens, or rewrites an earlier turn's tokens, fails. What
    a template emits before any message, a tools block for one, has no
    message of its own and counts as the first message's span.

    `source` names the tokenizer and the row for those refusals. The arrays
    are int32 ids and int8 roles.
    """
    rows = conversation.text_rows(source)
    tools = conversation.tools
    full = _token_ids(tokenizer, rows, tools, False, source)
    roles = np.zeros(len(full), np.int8)
    agreed = 0
    for position, message in enumerate(conversation.messages):
        role = message.role
        following = _token_ids(tokenizer, rows[:position + 1], tools, False, source)
        if role is Role.ASSISTANT:
            if position == 0:
                raise ValueError(
                    f"{source}: the first message is an assistant turn, so its "
                    "opening header cannot be separated from its completion "
                    "through the template; start the conversation with a system "
                    "or user message")
            prefix = _token_ids(tokenizer, rows[:position], tools, True, source)
            if full[:len(prefix)] != prefix or len(prefix) < agreed:
                raise ValueError(
                    f"tokenizer {source!r} does not render incrementally at message "
                    f"{position} (assistant): the generation prompt tokenizes to "
                    f"{prefix[-8:]}, the conversation to "
                    f"{full[max(0, len(prefix) - 8):len(prefix)]}")
            if full[:len(following)] != following:
                raise ValueError(
                    f"tokenizer {source!r} renders message {position} (assistant) "
                    "differently once later messages follow it, so its completion "
                    "cannot be masked from the whole conversation")
            roles[len(prefix):len(following)] = role.value
            agreed = len(following)
            continue
        end = _agreement(following, full)
        if end < agreed:
            raise ValueError(
                f"tokenizer {source!r} rewrites message {position - 1}'s tokens when "
                f"message {position} ({role.name.lower()}) follows it")
        if end == agreed:
            raise ValueError(
                f"tokenizer {source!r} renders message {position} ({role.name.lower()}) "
                "to no tokens; the template does not read this role or content")
        roles[agreed:end] = role.value
        agreed = end
    return np.asarray(full, np.int32), roles


_tokenizer_lock = threading.Lock()


@functools.lru_cache(maxsize=None)
def load_tokenizer(path: str) -> PreTrainedTokenizerBase:
    """The chat template's tokenizer, loaded once per process.

    The render map runs inside grain workers, which unpickle only the path,
    so each worker loads its own copy on its first record. The lock
    serializes that first import: transformers swaps in its lazy module
    object while it initializes, and two threads importing it together can
    catch it half built. The prompt source shares this cache.
    """
    with _tokenizer_lock:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(path)


class ConversationSource:
    """Random access over the `prompt` column of a parquet file, with the
    `tools` column beside it when the file has one.

    One record is one conversation, a list of messages in the verl layout,
    and its tool schemas, a list or a JSON string. The table is read once;
    rows come back as plain dicts, which pickle across to grain workers.
    Other columns are not read.
    """

    def __init__(self, path: str):
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise ImportError(
                "reading chat parquet needs pyarrow: pip install pyarrow") from exc
        names = [field.name for field in parquet.read_schema(path)]
        if "prompt" not in names:
            raise ValueError(f"{path}: the prompt column is required, the file has {names}")
        columns = ["prompt", "tools"] if "tools" in names else ["prompt"]
        table = parquet.read_table(path, columns=columns)
        if table.num_rows == 0:
            raise ValueError(f"{path} holds no conversations")
        self._conversations = table.column("prompt").to_pylist()
        self._tools = (table.column("tools").to_pylist() if "tools" in names
                       else [None] * table.num_rows)
        self.path = path

    def __repr__(self) -> str:
        # Grain writes repr(source) into a data loader iterator's checkpoint,
        # so a resumed run needs the file here, not an address in this process.
        return f"{self.__class__.__name__}(path={self.path!r})"

    def __len__(self) -> int:
        return len(self._conversations)

    def __iter__(self) -> Iterator[Batch]:
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: int) -> Batch:
        return {"messages": self._conversations[index], "tools": self._tools[index]}


class RenderConversation:
    """Source rows into rendered ids and per-token roles, for `map_with_index`.

    Holds only the tokenizer path, so grain workers unpickle the name and
    load their own copy. Failures name the tokenizer and the row.
    """

    def __init__(self, tokenizer: str):
        self.tokenizer = tokenizer

    def __call__(self, index: int, record: Batch) -> Batch:
        where = f"{self.tokenizer} row {index}"
        ids, roles = render_conversation(
            load_tokenizer(self.tokenizer),
            Conversation.parse(record["messages"], record["tools"], where), where)
        return {"text": ids, ROLES_KEY: roles}


def _lengths(source: ConversationSource, tokenizer: str) -> list[int]:
    """Every row's rendered length, validating the file up front.

    The chunk table needs the lengths before the lazy render map runs, so
    this pass renders each row once and keeps only its count. A bad row
    fails the run here with its index.
    """
    load = load_tokenizer(tokenizer)
    lengths = []
    for index, record in enumerate(source):
        where = f"{source.path} row {index}"
        conversation = Conversation.parse(record["messages"], record["tools"], where)
        lengths.append(len(render_conversation(load, conversation, where)[0]))
    return lengths


@datasets("chat_messages")
@dataclasses.dataclass(frozen=True)
class ChatMessages(DatasetSpec):
    """Conversations rendered with the tokenizer's chat template, packed.

    `path` is a parquet file whose `prompt` column holds lists of messages
    and whose `tools` column, when present, holds each row's tool schemas;
    `tokenizer` is the hub name or local path whose chat template renders
    them. Each conversation (in chunks, when it outgrows the window) is one
    element the packer adds to the first bin with room, and every emitted
    window carries `text_roles` beside the ids, so the loss can count one
    role's targets. `val_path` is a second parquet file scored as one pass;
    None trains without validation.
    """

    tokenizer: str
    path: str | None = None
    val_path: str | None = None
    seq_len: int = 256
    val_batches: int | None = 4
    seed: int = 0
    loading: Loading = Loading()
    packing_bins: int = 8

    def load(self, *, batch: int) -> Dataset:
        from grain.experimental import FirstFitPackIterDataset

        if self.path is None:
            raise ValueError("ChatMessages reads a parquet file: --data.path names it")
        per_process = local_batch(batch)
        window = self.seq_len + 1
        train_source = ConversationSource(self.path)
        train_lengths = _lengths(train_source, self.tokenizer)

        def stream(source: ConversationSource, lengths: list[int], shuffle: bool,
                   epochs: int | None):
            rendered = pygrain.MapDataset.source(source).map_with_index(
                RenderConversation(self.tokenizer))
            chunks: pygrain.MapDataset[Batch] = DocumentChunks(rendered, lengths, window)
            chunks = chunks[jax.process_index()::jax.process_count()]  # a slice is a dataset
            if shuffle:
                chunks = chunks.shuffle(self.seed)
            reads = chunks.repeat(epochs).to_iter_dataset()
            if self.loading.workers:
                # The workers render records, and the packer stays behind them
                # in this process. Grain runs a whole pipeline per worker, so
                # packing inside them would fill bins from one worker's slice
                # of the records and make the windows depend on worker_count.
                reads = reads.mp_prefetch(pygrain.MultiprocessingOptions(
                    num_workers=self.loading.workers,
                    per_worker_buffer_size=self.loading.worker_buffer))
            packed = FirstFitPackIterDataset(
                reads,
                length_struct={"text": window, ROLES_KEY: window},
                num_packing_bins=self.packing_bins,
                seed=self.seed,
                # Bins come out in packing order for val, so a validation pass
                # is the same batches every time.
                shuffle_bins=shuffle,
                padding_struct={"text": 0, ROLES_KEY: 0},
            )
            return iter(packed.batch(per_process, drop_remainder=True))

        val = None
        if self.val_path is not None:
            val_source = ConversationSource(self.val_path)
            val_lengths = _lengths(val_source, self.tokenizer)
            val = bounded(lambda: stream(val_source, val_lengths, False, 1), self.val_batches)
        return Dataset(
            train=lambda: stream(train_source, train_lengths, True, None),
            val=val,
            records=int(chunk_counts(train_lengths, window).sum()),
            batch=batch,
        )
