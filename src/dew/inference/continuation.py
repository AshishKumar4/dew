"""A policy's stop ids carried through SGLang, which also ends a draw on its model's own EOS ids.

SGLang ends a draw on `stop_token_ids` and on its model's EOS ids unless the
request ignores every token stop, and no field turns its own off alone
(sglang/srt/managers/schedule_batch.py:1707-1729 at v0.5.20). So a choice
SGLang ended on an id the policy's stops leave out (the choice's
`matched_stop`) goes on as the policy's draw would: its prompt and the ids
it drew become the prompt of a continuation on the completions route, drawn
under the same policy on the rest of its budget, until it ends on one of the
policy's stops or the budget. That is sequential sampling, so the joined
draw is the policy's; under a seed it is the uninterrupted draw itself, as
SGLang keys a seeded draw by the seed and each token's place in the sequence
(`multinomial_with_seed`, sglang/srt/layers/sampler.py), which the
continuation keeps.

A continuation carries what shapes the policy (`OwnStops.continuation`): its
sampling controls and seed, its row's LoRA adapter, the minimum output still
owed, and a frequency penalty, whose counts of the ids drawn before it go in
as logit bias, which SGLang adds to the same logits. A continued choice's
text is SGLang's own decoding of the joined ids (`/v1/detokenize`), trimmed
of its policy stop as SGLang trims one. What a continuation cannot carry is
refused where a draw needs one, by name (`_uncarried`): a presence or
repetition penalty, which counts the ids before it once whichever segment
draws them again; stop strings, a stop regex and constrained decoding, whose
matching restarts; a chat's media, which the completions route takes none
of; and a stream, whose text deltas restart their decoding at the cut. The
non-streaming call draws all but those.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Awaitable, Callable, Generator, Mapping, Sequence
from types import TracebackType
from typing import TYPE_CHECKING, Literal

from dew.records import JSON

if TYPE_CHECKING:
    import httpx2
    from openai import AsyncStream, Stream
    from openai.types import Completion as OpenAIResponse
    from openai.types.chat import ChatCompletion, ChatCompletionChunk

type Request = Mapping[str, object]
"""The fields of one SDK request, as its `create` takes them."""

type Call = tuple[Literal["completions", "detokenize"], Request]
"""One request a continuation sends: a completions request, or a detokenization of ids."""

type Plan[Result] = Generator[Call, JSON, Result]
"""A computation that yields the requests it sends, is sent each one's response JSON, and returns
`Result`: `driven` and `driven_async` answer it."""

_CARRIED = ("model", "temperature", "top_p", "frequency_penalty", "logit_bias", "seed", "user",
            "extra_headers", "extra_query", "timeout")
"""The request fields a continuation keeps as they are: the policy's controls the completions and
chat routes read alike, its seed, and the SDK's transport options. `extra_body` carries the
engine's own controls, and the rest are written per continuation (`OwnStops.continuation`)."""

_RESTARTED = ("stop", "stop_regex", "regex", "json_schema", "ebnf", "structural_tag", "response_format")
"""The request fields whose matching a continuation would start again with the ids before it as
prompt: stop strings and patterns, and constrained decoding."""


def _listed(value: JSON, name: str) -> list[JSON]:
    """`value`, read at the engine's boundary as the list its response holds at `name`."""
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _ids(value: JSON, name: str) -> list[int]:
    """`value`, read at the engine's boundary as the token ids its response holds at `name`."""
    listed = _listed(value, name)
    if any(type(token) is not int or token < 0 for token in listed):
        raise ValueError(f"{name} must hold nonnegative token ids")
    return [token for token in listed if isinstance(token, int)]


def _record(value: JSON, name: str) -> dict[str, JSON]:
    """`value`, read at the engine's boundary as the object its response holds at `name`."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _fields(value: object, name: str) -> Mapping[str, object]:
    """`value`, read at the SDK request's boundary as the mapping of fields `name` holds."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping of fields")
    return {str(key): entry for key, entry in value.items()}


def _given(request: Request, name: str, *neutral: JSON) -> bool:
    """Whether the request sets `name`, at its top level or in `extra_body`, to other than None and
    the `neutral` values that leave the draw as it is."""
    value = request.get(name)
    value = _fields(request.get("extra_body"), "extra_body").get(name) if value is None else value
    return value is not None and value not in neutral


def _switch(request: Request, name: str, default: bool) -> bool:
    """What the request sets the switch `name` to, at its top level or in `extra_body`, `default`
    where it is unset."""
    value = request.get(name)
    value = _fields(request.get("extra_body"), "extra_body").get(name) if value is None else value
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def _uncarried(request: Request) -> str | None:
    """The control of `request` a continuation cannot carry exactly, named, or None where it carries
    them all (the module's account)."""
    restarted = [name for name in _RESTARTED if _given(request, name, "", [], {})]
    if restarted:
        return f"{', '.join(restarted)}, whose matching would start again at the cut"
    for name, neutral in (("presence_penalty", 0), ("repetition_penalty", 1)):
        if _given(request, name, neutral):
            return f"{name}, which counts each id drawn before the cut once whichever segment draws it again"
    if not _switch(request, "spaces_between_special_tokens", True):
        return "spaces_between_special_tokens=False, which SGLang's /v1/detokenize does not take"
    messages = request.get("messages")
    for message in messages if isinstance(messages, list) else ():
        content = _fields(message, "message").get("content")
        if isinstance(content, list) and any(_fields(part, "content part").get("type") != "text"
                                             for part in content):
            return "a message's media, which the completions route takes none of"
    return None


@dataclasses.dataclass(frozen=True)
class OwnStops:
    """The policy's stop ids `stops` as one SGLang request draws them on `budget` tokens a choice."""

    stops: frozenset[int]
    budget: int

    def ended_outside(self, reason: JSON, matched: JSON) -> bool:
        """Whether SGLang ended a choice on an id of its own the policy's stops leave out."""
        return reason == "stop" and type(matched) is int and matched not in self.stops

    def continuation(self, request: Request, index: int, prompt: Sequence[int],
                     drawn: Sequence[int]) -> Request:
        """The completions request that draws on choice `index` of `request` from `prompt` and the ids
        it `drawn`, under the same policy on the rest of the budget: its own row's LoRA adapter, the
        minimum output still owed, its logprob count, and a frequency penalty's counts of `drawn` as
        logit bias. Raises `ValueError` for a control it cannot carry (`_uncarried`)."""
        if (uncarried := _uncarried(request)) is not None:
            raise ValueError(
                f"SGLang ended choice {index} on its model's EOS id {drawn[-1]}, which the policy does not "
                f"stop on, and Dew goes on from the ids it drew, which cannot carry {uncarried}. Name "
                f"{drawn[-1]} among the policy's EOS ids, drop that control, or draw on vLLM")
        extra = {name: value for name, value in _fields(request.get("extra_body"), "extra_body").items()
                 if name != "return_prompt_token_ids"}
        extra["return_token_ids"] = True
        prompts = request.get("prompt")
        batched = isinstance(prompts, list) and bool(prompts) and not isinstance(prompts[0], int)
        rows = len(prompts) if batched and isinstance(prompts, list) else 1
        if isinstance(adapters := extra.get("lora_path"), list):
            # SGLang repeats a batch's rows once per sample, adapters alike (io_struct.py `_expand_inputs`).
            extra["lora_path"] = adapters[index % rows]
        if isinstance(least := extra.get("min_tokens"), int):
            extra["min_tokens"] = max(least - len(drawn), 0)
        body: dict[str, object] = {name: request[name] for name in _CARRIED if name in request}
        frequency = request.get("frequency_penalty")
        if isinstance(frequency, int | float) and frequency:
            given = _fields(request.get("logit_bias"), "logit_bias")
            bias = {str(token): value for token, value in given.items()}
            for token, count in Counter(drawn).items():
                bias[str(token)] = _number(bias.get(str(token), 0)) - frequency * count
            body["logit_bias"] = bias
        if "messages" in request:
            logprobs = (request.get("top_logprobs") or 0) if request.get("logprobs") else None
        else:
            logprobs = request.get("logprobs")
        if logprobs is not None:
            body["logprobs"] = logprobs
        # An empty suffix is SGLang's own pass-through of an already formatted prompt, whose
        # completion template would otherwise rewrite it (code_completion_parser.py).
        return {**body, "prompt": [[*prompt, *drawn]], "max_tokens": self.budget - len(drawn), "n": 1,
                "suffix": "", "extra_body": extra}

    def decoding(self, request: Request, choice: Mapping[str, JSON]) -> Request:
        """The `/v1/detokenize` request that decodes a continued `choice`'s ids as SGLang decodes a
        draw's: special tokens skipped as `request` asks, and a policy stop it ended on trimmed
        unless `request` keeps it (`no_stop_trim`, detokenizer_manager.py `trim_matched_stop`)."""
        ids = _ids(choice.get("token_ids"), "token_ids")
        trimmed = (choice.get("finish_reason") == "stop" and type(choice.get("matched_stop")) is int
                   and not _switch(request, "no_stop_trim", False))
        return {"model": request.get("model"), "tokens": ids[:-1] if trimmed else ids,
                "skip_special_tokens": _switch(request, "skip_special_tokens", True)}

    def completed(self, payload: JSON, request: Request) -> Plan[JSON]:
        """`payload`, a completions response to `request`, with each choice SGLang ended on an id
        of its own outside the policy's stops continued to the policy's stop or the budget: its ids
        and log-probabilities joined, its text decoded again from the joined ids, and the usage
        summed over every request."""
        response = _record(payload, "completion response")
        choices: list[JSON] = []
        usage = response.get("usage")
        for choice in _listed(response.get("choices"), "completion choices"):
            entry = _record(choice, "completion choice")
            index = entry.get("index")
            continued = False
            while self.ended_outside(entry.get("finish_reason"), entry.get("matched_stop")):
                drawn = _ids(entry.get("token_ids"), "token_ids")
                if len(drawn) >= self.budget:
                    entry = {**entry, "finish_reason": "length", "matched_stop": None}
                    break
                prompt = _ids(entry.get("prompt_token_ids"), "prompt_token_ids")
                follow = self.continuation(request, index if type(index) is int else 0, prompt, drawn)
                answer = _record((yield "completions", follow), "completion response")
                (then,) = _listed(answer.get("choices"), "continued choices")
                entry = _joined(entry, _record(then, "completion choice"))
                usage = _summed(usage, answer.get("usage"))
                continued = True
            if continued:
                decoded = _record((yield "detokenize", self.decoding(request, entry)), "detokenized text")
                entry = {**entry, "text": decoded.get("text")}
            choices.append(entry)
        return {**response, "choices": choices, "usage": usage}

    def chat_completed(self, response: ChatCompletion, request: Request) -> Plan[ChatCompletion]:
        """`response` with each choice SGLang ended on an id of its own outside the policy's stops
        continued on the completions route (`completed`): its message's content decoded from the
        joined ids, its log-probabilities and token ids extended and the usage's counts summed. A
        message the server parsed tool calls or reasoning out of holds no text a continuation could
        extend, and is refused there."""
        from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob

        choices = []
        usage: JSON = None if response.usage is None else {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens}
        for choice in response.choices:
            extra = choice.model_extra or {}
            matched = extra.get("matched_stop")
            if not self.ended_outside(choice.finish_reason, matched):
                choices.append(choice)
                continue
            ids = _ids(extra.get("response_token_ids"), "response_token_ids")
            if len(ids) >= self.budget:
                choices.append(choice.model_copy(update={"finish_reason": "length", "matched_stop": None}))
                continue
            message = choice.message
            if message.tool_calls or (message.model_extra or {}).get("reasoning_content"):
                raise ValueError(
                    f"SGLang ended chat choice {choice.index} on its model's EOS id {matched}, which the "
                    "policy does not stop on, after it parsed tool calls or reasoning out of the message, "
                    "which a continuation's text cannot extend; draw it on the completions route")
            drawn: dict[str, JSON] = {
                "index": choice.index, "text": message.content or "", "finish_reason": "stop",
                "matched_stop": matched, "token_ids": list(ids),
                "prompt_token_ids": extra.get("prompt_token_ids")}
            if choice.logprobs is not None:
                drawn["logprobs"] = {"tokens": [], "token_logprobs": [], "top_logprobs": []}
            payload = _record((yield from self.completed({"choices": [drawn], "usage": usage}, request)),
                              "continued response")
            usage = payload.get("usage")
            (done,) = _listed(payload.get("choices"), "continued choices")
            done = _record(done, "continued choice")
            update: dict[str, object] = {"finish_reason": done.get("finish_reason"),
                                         "matched_stop": done.get("matched_stop"),
                                         "response_token_ids": done.get("token_ids"),
                                         "message": message.model_copy(update={"content": done.get("text")})}
            if choice.logprobs is not None:
                added = _record(done.get("logprobs"), "continued logprobs")
                tokens = _listed(added.get("tokens"), "tokens")
                tops = _listed(added.get("top_logprobs") or [], "top_logprobs") or [None] * len(tokens)
                content = [ChatCompletionTokenLogprob.model_validate({
                    "token": token, "logprob": value, "bytes": _bytes(token),
                    "top_logprobs": [{"token": name, "logprob": score, "bytes": _bytes(name)}
                                     for name, score in _record(top or {}, "top logprobs").items()]})
                    for token, value, top in zip(
                        tokens, _listed(added.get("token_logprobs"), "token_logprobs"), tops, strict=True)]
                update["logprobs"] = choice.logprobs.model_copy(
                    update={"content": [*(choice.logprobs.content or []), *content]})
            choices.append(choice.model_copy(update=update))
        changed: dict[str, object] = {"choices": choices}
        if response.usage is not None and usage is not None:
            # The counts alone: the usage's nested details stay the SDK's own models.
            changed["usage"] = response.usage.model_copy(update=_record(usage, "usage"))
        return response.model_copy(update=changed)


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("a logit bias must be a number")
    return value


def _bytes(token: JSON) -> list[int]:
    """A log-probability's token as the bytes SGLang's chat route reports beside it."""
    if not isinstance(token, str):
        raise ValueError("a logprob token must be text")
    return list(token.encode("utf-8"))


def _joined(first: dict[str, JSON], then: dict[str, JSON]) -> dict[str, JSON]:
    """A completions choice and its continuation, as the engine's boundary reads them, as one
    choice: ids and log-probabilities end to end, and the continuation's finish. The text is
    decoded again from the joined ids (`OwnStops.decoding`), so a text offset past the cut is
    SGLang's own unknown, -1, which v0.5.20 reports for every token (openai/utils.py)."""
    joined: dict[str, JSON] = {
        **first,
        "token_ids": [*_ids(first.get("token_ids"), "token_ids"), *_ids(then.get("token_ids"), "token_ids")],
        "finish_reason": then.get("finish_reason"), "matched_stop": then.get("matched_stop")}
    before = first.get("logprobs")
    if isinstance(before, dict):
        added = _record(then.get("logprobs"), "continued logprobs")
        logprobs: dict[str, JSON] = {}
        for name in ("tokens", "token_logprobs", "top_logprobs"):
            logprobs[name] = None if before.get(name) is None else [*_listed(before.get(name), name),
                                                                     *_listed(added.get(name) or [], name)]
        if before.get("text_offset") is not None:
            logprobs["text_offset"] = [*_listed(before.get("text_offset"), "text_offset"),
                                       *([-1] * len(_listed(added.get("tokens"), "tokens")))]
        joined["logprobs"] = logprobs
    return joined


def _summed(first: JSON, then: JSON) -> JSON:
    """Two requests' usage, as the engine's boundary reads it, as one: each count summed where both
    report it."""
    if not isinstance(first, dict) or not isinstance(then, dict):
        return first
    names = ("prompt_tokens", "completion_tokens", "total_tokens")
    counts = {name: (first[name], then[name]) for name in names if name in first and name in then}
    return {**first, **{name: left + right for name, (left, right) in counts.items()
                        if isinstance(left, int) and isinstance(right, int)}}


def driven[Result](plan: Plan[Result], call: Callable[[Call], JSON]) -> Result:
    """`plan`'s result, each request it yields answered by `call`."""
    try:
        request = next(plan)
        while True:
            request = plan.send(call(request))
    except StopIteration as done:
        return done.value


async def driven_async[Result](plan: Plan[Result], call: Callable[[Call], Awaitable[JSON]]) -> Result:
    """`driven` with an awaited `call`."""
    try:
        request = next(plan)
        while True:
            request = plan.send(await call(request))
    except StopIteration as done:
        return done.value


def _stream_refusal(chunk: OpenAIResponse | ChatCompletionChunk, own: OwnStops) -> str | None:
    """Why a streamed `chunk` cannot go on as the policy's draw: SGLang ended a choice of it on an id
    of its own outside the policy's stops, and a stream's text deltas would restart their decoding
    at a continuation's cut. None where every choice in it is the policy's."""
    for choice in chunk.choices:
        matched = (choice.model_extra or {}).get("matched_stop")
        if own.ended_outside(choice.finish_reason, matched):
            return (f"SGLang ended streamed choice {choice.index} on its model's EOS id {matched}, which the "
                    "policy does not stop on, and a stream's text cannot go on past it exactly; ask for the "
                    "whole completion or chat, which goes on from the ids it drew")
    return None


class Checked[ChunkT: (OpenAIResponse, ChatCompletionChunk)]:
    """An SGLang `stream`'s chunks as they come, refused at a chunk SGLang ended a choice of on an id
    of its own outside the policy's stops (`_stream_refusal`). It owns the stream: closing it, or
    leaving its `with` block, closes the stream, before the first read too."""

    def __init__(self, stream: Stream[ChunkT], own: OwnStops):
        self._stream, self._own = stream, own

    def __iter__(self) -> Checked[ChunkT]:
        return self

    def __next__(self) -> ChunkT:
        chunk = next(self._stream)
        if (refusal := _stream_refusal(chunk, self._own)) is not None:
            self.close()
            raise ValueError(refusal)
        return chunk

    @property
    def response(self) -> httpx2.Response:
        """The stream's HTTP response, as the SDK's stream names it."""
        return self._stream.response

    def close(self) -> None:
        self._stream.close()

    def __enter__(self) -> Checked[ChunkT]:
        return self

    def __exit__(self, kind: type[BaseException] | None, raised: BaseException | None,
                 traceback: TracebackType | None) -> None:
        self.close()


class AsyncChecked[ChunkT: (OpenAIResponse, ChatCompletionChunk)]:
    """`Checked` over an async stream."""

    def __init__(self, stream: AsyncStream[ChunkT], own: OwnStops):
        self._stream, self._own = stream, own

    def __aiter__(self) -> AsyncChecked[ChunkT]:
        return self

    async def __anext__(self) -> ChunkT:
        chunk = await self._stream.__anext__()
        if (refusal := _stream_refusal(chunk, self._own)) is not None:
            await self.close()
            raise ValueError(refusal)
        return chunk

    @property
    def response(self) -> httpx2.Response:
        """The stream's HTTP response, as the SDK's stream names it."""
        return self._stream.response

    async def close(self) -> None:
        await self._stream.close()

    async def __aenter__(self) -> AsyncChecked[ChunkT]:
        return self

    async def __aexit__(self, kind: type[BaseException] | None, raised: BaseException | None,
                        traceback: TracebackType | None) -> None:
        await self.close()
