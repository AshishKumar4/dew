"""Supervised fine-tuning data: conversations with a role on every token.

The `Chat` recipe reads conversations from a parquet file, a JSONL file or
an `hf://` Hub dataset, and renders each with the tokenizer's chat
template. Every token gets the role of the message that wrote it, so
`LMObjective` with `loss_role=Role.ASSISTANT` trains on assistant tokens
only.

The rendered conversations are packed by the token pipeline's plan over
the whole corpus (`PackedWindows`), with `text_roles` as one more per-token
field. Each batch holds `text` and `text_roles`, aligned, and the windows'
documents as `text_coordinates`. The training stream's position is a global
window count, so a run can resume on any number of processes.

Block diffusion reads a prompt and the canvases after it instead, so its
`ChatCanvasData` recipe lays each assistant turn out as a row of its own,
unpacked: the conversation before the turn, then the turn on whole canvases.

A `Message` holds the fields the Hugging Face chat-template contract reads:
a role; content that is a string, a list of typed parts, or nothing; an
assistant's `tool_calls`; a tool response's `tool_call_id` and `name`; and
any further keys the template wants, such as `reasoning_content` or
`thinking`, unchanged. A `Conversation` adds the `tools` schemas that the
template renders into its system block. The stored messages keep their typed
content. When a text tokenizer renders them, each message's text parts are
joined into one string, and any other part type is refused because it needs
a processor. Tool-call fields and message metadata reach the template
unchanged.
"""

from __future__ import annotations

import dataclasses
import functools
import json
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Literal

import grain.python as grain
import numpy as np
from jinja2 import TemplateError

from dew.data.text import HFTokenizer, tokenizer_name

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from dew.data.dataset import Batch
    from dew.data.steps.text import LengthPolicy

ROLES_KEY = "text_roles"
"""Batch key the chat pipeline packs `[B, seq_len + 1]` int8 roles under."""


class Role(int, Enum):
    """Who wrote a token of a rendered conversation.

    The values are what the `text_roles` column stores. 0 marks padding, and
    every other value is the role of the message whose span contains the
    token. An assistant turn's opening header is in no span, so it reads as
    padding and only the completion counts as assistant tokens. `DEVELOPER`
    is the Harmony instruction role. A template that does not know it renders
    the message to no tokens, and the renderer refuses that.
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

    `type` names the kind of block, and `fields` holds the block's other keys
    as they were: `text` for a text block, or a url or a path for media.
    Rendering with a text tokenizer accepts only text parts, but media parts
    are kept in the stored conversation.
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

    @property
    def needs_processor(self) -> bool:
        """Whether this part needs processing beyond a text tokenizer."""
        return self.type != "text"


@dataclasses.dataclass(frozen=True)
class ToolCall:
    """One function call that an assistant message requests.

    `arguments` is the parsed object. A source may store it as a JSON string,
    because parquet cannot hold a struct whose fields differ from call to
    call, while the Hugging Face contract gives templates a mapping. So a
    string is parsed here, and one that is not a JSON object raises
    `ValueError`. `id` matches the call to its response's `tool_call_id`, for
    templates that use ids. `type` is `"function"`, as the contract defines
    it. A source may write the call flat (`{name, arguments}`) or nested
    under `function`; both parse to the same call.
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
    """One turn of a conversation, in the shape the chat template reads.

    `content` is the text, the typed parts, or None when the turn consists
    only of tool calls. None is passed to the template as None, which is how
    tool loops send such a turn. `tool_calls` belong to assistant turns and
    `tool_call_id` to tool responses, and `parse` refuses either on any other
    role. `name` is the function a tool response answers, or a participant's
    name on other roles. `extra` holds every other key unchanged, so keys
    such as `reasoning_content` or `thinking` reach the templates that read
    them. An optional key whose value is None counts as absent, because that
    is how parquet stores a field a row does not have.
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
        """Return the message as the Hugging Face template dict, with its
        content parts and metadata.

        `Conversation.text_rows` converts these dicts for text tokenizers.
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
        """Parse one row's messages and tool schemas as a parquet column
        stores them.

        Each is a list or, when the column's schema varies across rows, the
        JSON text of one. A malformed row or an empty conversation raises
        `ValueError`.
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
        """Return the messages as text-tokenizer input, with each message's
        text parts joined in order.

        The stored messages and `rows` keep their structured content. A part
        that is not text raises `ValueError`, because media needs a processor
        to turn its payload into model inputs. A text tokenizer cannot do
        that, even if its template emits a marker for the media.
        """
        rows = []
        for index, (message, row) in enumerate(zip(self.messages, self.rows(), strict=True)):
            if not isinstance(message.content, tuple):
                rows.append(row)
                continue
            texts: list[str] = []
            for part in message.content:
                at = f"{where} message {index}"
                if part.needs_processor:
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
    for index, message in enumerate(conversation.messages):
        following = _token_ids(tokenizer, rows[:index + 1], tools, generation_prompt=False, where=where)
        if message.role is Role.ASSISTANT:
            start, end = _completion_span(tokenizer, rows, tools, full, following, agreed,
                                          index, where)
        else:
            start = agreed
            end = _turn_end(full, following, agreed, message.role, index, where)
        roles[start:end] = message.role.value
        agreed = end
    return np.asarray(full, np.int32), roles


def _completion_span(tokenizer: PreTrainedTokenizerBase, rows, tools: Sequence[Mapping[str, object]],
                     full: Sequence[int], following: Sequence[int], agreed: int,
                     index: int, where: str) -> tuple[int, int]:
    """Where assistant message `index`'s completion begins and ends in `full`.

    The generation prompt over the messages before it is that turn's opening
    header, so the completion is what follows it up to the turn's own render.
    Both renders have to be prefixes of the whole conversation, or the span
    the loss counts is not the span the model will see.
    """
    if index == 0:
        raise ValueError(
            f"{where}: the first message is an assistant turn, so its "
            "opening header cannot be separated from its completion "
            "through the template; start the conversation with a system "
            "or user message")
    prefix = _token_ids(tokenizer, rows[:index], tools, generation_prompt=True, where=where)
    if full[:len(prefix)] != prefix or len(prefix) < agreed:
        raise ValueError(
            f"tokenizer {where!r} does not render incrementally at message "
            f"{index} (assistant): the generation prompt tokenizes to "
            f"{prefix[-8:]}, the conversation to "
            f"{full[max(0, len(prefix) - 8):len(prefix)]}")
    if full[:len(following)] != following:
        raise ValueError(
            f"tokenizer {where!r} renders message {index} (assistant) "
            "differently once later messages follow it, so its completion "
            "cannot be masked from the whole conversation")
    return len(prefix), len(following)


def _turn_end(full: Sequence[int], following: Sequence[int], agreed: int, role: Role,
              index: int, where: str) -> int:
    """Where message `index`'s span ends in `full`, for a turn that is not
    an assistant's.

    The span runs from where the previous turn ended to where this turn's own
    render stops agreeing with the whole conversation. A turn that rewrites
    an earlier turn's tokens, or renders to none of its own, fails here.
    """
    end = _agreement(following, full)
    if end < agreed:
        raise ValueError(
            f"tokenizer {where!r} rewrites message {index - 1}'s tokens when "
            f"message {index} ({role.name.lower()}) follows it")
    if end == agreed:
        raise ValueError(
            f"tokenizer {where!r} renders message {index} ({role.name.lower()}) "
            "to no tokens; the template does not read this role or content")
    return end


@dataclasses.dataclass(frozen=True)
class RenderChat(grain.MapTransform):
    """The conversation in `column` rendered through the chat template of the
    tokenizer `tokenizer` names: token ids under "text", each token's speaker
    under "text_roles"."""

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


def template(tokenizer: str | PreTrainedTokenizerBase, *, messages: str = "messages") -> RenderChat:
    """A record step that renders the conversation in `messages` through the
    tokenizer's chat template to token ids under "text" and each token's
    speaker under "text_roles", which an assistant-only loss reads.
    `tokenizer` is a Hub repo or directory, or a transformers tokenizer."""
    return RenderChat(tokenizer_name(tokenizer), messages)


def fit_length(length: int, *, long: Literal["truncate", "drop"] = "truncate") -> LengthPolicy:
    """A record step that keeps a rendered conversation within `length`
    tokens, cutting its end (or dropping it, counted) and its roles alike."""
    from dew.data.steps.text import LengthPolicy

    return LengthPolicy(length, long=long, aligned=(ROLES_KEY,))
