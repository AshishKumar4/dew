"""A policy's stop ids carried through SGLang, which also ends a draw on its model's own EOS ids.

SGLang ends a draw on `stop_token_ids` and on its model's EOS ids unless the
request ignores every token stop, and no field turns its own off alone
(sglang/srt/managers/schedule_batch.py:1707-1729 at v0.5.20). So a choice
SGLang ended on an id the policy's stops leave out (the choice's
`matched_stop`) goes on as the policy's draw would: its prompt and the ids
it drew become the prompt of a continuation on the completions route, drawn
under the same policy on the rest of its budget, until it ends on one of the
policy's stops or the budget. That is sequential sampling, so the joined
draw is the policy's.

A continuation reads the ids before it as prompt, which SGLang's repetition,
frequency and presence penalties do not count (penaltylib counts output ids
alone), so a policy with one is refused where its draw needs continuing. A
chat stream reports no token ids to continue from, and a chat message the
server parsed tool calls or reasoning out of holds no text a continuation
could extend, so those are refused there too; the completions route draws
them.
"""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator, Awaitable, Callable, Generator, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from dew.records import JSON

if TYPE_CHECKING:
    from openai import AsyncStream, Stream
    from openai.types import Completion as OpenAIResponse
    from openai.types.chat import ChatCompletion, ChatCompletionChunk

type Request = Mapping[str, object]
"""The fields of one SDK request, as its `create` takes them."""

type Plan[Result] = Generator[Request, JSON, Result]
"""A computation that yields completions requests, is sent each one's response JSON, and returns
`Result`: `driven` and `driven_async` answer it."""

_CARRIED = ("model", "temperature", "top_p", "frequency_penalty", "presence_penalty", "logit_bias", "user",
            "extra_headers", "extra_query", "timeout")
"""The request fields a continuation keeps, which shape the policy or reach the engine alike on
the completions and chat routes; `extra_body` carries the engine's own controls."""


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


@dataclasses.dataclass(frozen=True)
class OwnStops:
    """The policy's stop ids `stops` as one SGLang request draws them on `budget` tokens a choice,
    its continuations seeded from `seed` where the request is seeded. `penalized` says whether the
    policy sets a repetition, frequency or presence penalty."""

    stops: frozenset[int]
    budget: int
    seed: int | None
    penalized: bool

    def ended_outside(self, reason: JSON, matched: JSON) -> bool:
        """Whether SGLang ended a choice on an id of its own the policy's stops leave out."""
        return reason == "stop" and type(matched) is int and matched not in self.stops

    def continuation(self, request: Request, index: int, prompt: Sequence[int], drawn: Sequence[int], *,
                     stream: bool = False) -> Request:
        """The completions request that continues choice `index` of `request` from `prompt` and the
        ids it `drawn`, under the same policy on the rest of the budget."""
        if self.penalized:
            raise ValueError(
                f"SGLang ended choice {index} on its model's EOS id {drawn[-1]}, which the policy does not "
                "stop on; Dew continues such a draw from the ids it drew, which SGLang's repetition, "
                f"frequency and presence penalties would not count. Name {drawn[-1]} among the policy's "
                "EOS ids, drop the penalties, or draw on vLLM")
        extra = {name: value for name, value in _record(request.get("extra_body") or {}, "extra_body").items()
                 if name != "return_prompt_token_ids"}
        chat = "messages" in request
        logprobs = (request.get("top_logprobs", 0) if request.get("logprobs") else None) if chat else (
            request.get("logprobs"))
        continued: dict[str, object] = {
            **{name: request[name] for name in _CARRIED if name in request},
            "prompt": [[*prompt, *drawn]], "max_tokens": self.budget - len(drawn), "n": 1, "stream": stream,
            "extra_body": {**extra, "return_token_ids": True},
            **({} if logprobs is None else {"logprobs": logprobs})}
        if self.seed is not None:
            # A continuation draws on a stream of its own: one of the request's would repeat its draws.
            keyed = np.random.SeedSequence((self.seed, index, len(drawn)))
            continued["seed"] = int(keyed.generate_state(1)[0])
        return continued

    def completed(self, payload: JSON, request: Request) -> Plan[JSON]:
        """`payload`, a completions response to `request`, with each choice SGLang ended on an id
        of its own outside the policy's stops continued, joined to its continuations, and the usage
        summed over every request."""
        response = _record(payload, "completion response")
        choices: list[JSON] = []
        usage = response.get("usage")
        for choice in _listed(response.get("choices"), "completion choices"):
            entry = _record(choice, "completion choice")
            index = entry.get("index")
            while self.ended_outside(entry.get("finish_reason"), entry.get("matched_stop")):
                drawn = _ids(entry.get("token_ids"), "token_ids")
                if len(drawn) >= self.budget:
                    entry = {**entry, "finish_reason": "length", "matched_stop": None}
                    break
                prompt = _ids(entry.get("prompt_token_ids"), "prompt_token_ids")
                answer = _record((yield self.continuation(request, index if type(index) is int else 0, prompt,
                                                          drawn)), "completion response")
                (then,) = _listed(answer.get("choices"), "continued choices")
                entry = _joined(entry, _record(then, "completion choice"))
                usage = _summed(usage, answer.get("usage"))
            choices.append(entry)
        return {**response, "choices": choices, "usage": usage}

    def chat_completed(self, response: ChatCompletion, request: Request) -> Plan[ChatCompletion]:
        """`response` with each choice SGLang ended on an id of its own outside the policy's stops
        continued on the completions route (`completed`), its message's content, log-probabilities
        and token ids extended and the usage summed."""
        from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob

        choices = []
        usage: JSON = None if response.usage is None else response.usage.model_dump()
        for choice in response.choices:
            extra = choice.model_extra or {}
            matched = extra.get("matched_stop")
            if not self.ended_outside(choice.finish_reason, matched):
                choices.append(choice)
                continue
            message = choice.message
            if message.tool_calls or (message.model_extra or {}).get("reasoning_content"):
                raise ValueError(
                    f"SGLang ended chat choice {choice.index} on its model's EOS id {matched}, which the "
                    "policy does not stop on, after it parsed tool calls or reasoning out of the message, "
                    "which a continuation's text cannot extend; draw it on the completions route")
            drawn: JSON = {"index": choice.index, "text": message.content or "", "finish_reason": "stop",
                           "matched_stop": matched, "token_ids": extra.get("response_token_ids"),
                           "prompt_token_ids": extra.get("prompt_token_ids")}
            if choice.logprobs is not None:
                drawn["logprobs"] = {"tokens": [], "token_logprobs": [], "top_logprobs": [],
                                     "text_offset": None}
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
                tops = added.get("top_logprobs")
                content = [ChatCompletionTokenLogprob.model_validate({
                    "token": token, "logprob": value,
                    "top_logprobs": [{"token": name, "logprob": score}
                                     for name, score in _record(top or {}, "top logprobs").items()]})
                    for token, value, top in zip(
                        tokens, _listed(added.get("token_logprobs"), "token_logprobs"),
                        [None] * len(tokens) if tops is None else _listed(tops, "top_logprobs"), strict=True)]
                update["logprobs"] = choice.logprobs.model_copy(
                    update={"content": [*(choice.logprobs.content or []), *content]})
            choices.append(choice.model_copy(update=update))
        changed: dict[str, object] = {"choices": choices}
        if response.usage is not None and usage is not None:
            changed["usage"] = response.usage.model_copy(update=_record(usage, "usage"))
        return response.model_copy(update=changed)


def _joined(first: dict[str, JSON], then: dict[str, JSON]) -> dict[str, JSON]:
    """A completions choice and its continuation, as the engine's boundary reads them, as one
    choice: text, ids and log-probabilities end to end, and the continuation's finish."""
    text = first.get("text")
    later = then.get("text")
    if not isinstance(text, str) or not isinstance(later, str):
        raise ValueError("each completion choice needs string text")
    merged: dict[str, JSON] = {
        **first, "text": text + later,
        "token_ids": [*_ids(first.get("token_ids"), "token_ids"), *_ids(then.get("token_ids"), "token_ids")],
        "finish_reason": then.get("finish_reason"), "matched_stop": then.get("matched_stop")}
    before, after = first.get("logprobs"), then.get("logprobs")
    if isinstance(before, dict):
        added = _record(after, "continued logprobs")
        logprobs: dict[str, JSON] = {}
        for name in ("tokens", "token_logprobs", "top_logprobs"):
            logprobs[name] = None if before.get(name) is None else [*_listed(before.get(name), name),
                                                                     *_listed(added.get(name) or [], name)]
        logprobs["text_offset"] = None if before.get("text_offset") is None else [
            *_listed(before.get("text_offset"), "text_offset"),
            *(offset + len(text) for offset in _ids(added.get("text_offset") or [], "text_offset"))]
        merged["logprobs"] = logprobs
    return merged


def _summed(first: JSON, then: JSON) -> JSON:
    """Two requests' usage, as the engine's boundary reads it, as one: each count summed where both
    report it."""
    if not isinstance(first, dict) or not isinstance(then, dict):
        return first
    names = ("prompt_tokens", "completion_tokens", "total_tokens")
    counts = {name: (first[name], then[name]) for name in names if name in first and name in then}
    return {**first, **{name: left + right for name, (left, right) in counts.items()
                        if isinstance(left, int) and isinstance(right, int)}}


def driven[Result](plan: Plan[Result], call: Callable[[Request], JSON]) -> Result:
    """`plan`'s result, each request it yields answered by `call`."""
    try:
        request = next(plan)
        while True:
            request = plan.send(call(request))
    except StopIteration as done:
        return done.value


async def driven_async[Result](plan: Plan[Result], call: Callable[[Request], Awaitable[JSON]]) -> Result:
    """`driven` with an awaited `call`."""
    try:
        request = next(plan)
        while True:
            request = plan.send(await call(request))
    except StopIteration as done:
        return done.value


@dataclasses.dataclass
class _Streamed:
    """Each streamed completion choice's prompt and drawn ids, by the index the caller reads it
    under."""

    own: OwnStops
    request: Request
    prompts: dict[int, list[int]] = dataclasses.field(default_factory=dict)
    drawn: dict[int, list[int]] = dataclasses.field(default_factory=dict)

    def fed(self, chunk: OpenAIResponse,
            index: int | None) -> tuple[OpenAIResponse, list[tuple[int, Request]]]:
        """A chunk of the stream (`index` None), or of choice `index`'s continuation, with each choice
        under the caller's index, and the continuation's request for each choice SGLang ended on an
        id of its own outside the policy's stops, which that chunk then carries no finish for."""
        continued: list[tuple[int, Request]] = []
        choices = []
        for choice in chunk.choices:
            extra = choice.model_extra or {}
            at = choice.index if index is None else index
            if index is None:
                self.prompts.setdefault(at, []).extend(_ids(extra.get("prompt_token_ids") or [],
                                                            "prompt_token_ids"))
            self.drawn.setdefault(at, []).extend(_ids(extra.get("token_ids") or [], "token_ids"))
            update: dict[str, object] = {"index": at}
            if self.own.ended_outside(choice.finish_reason, extra.get("matched_stop")):
                if len(self.drawn[at]) >= self.own.budget:
                    update.update(finish_reason="length", matched_stop=None)
                else:
                    update.update(finish_reason=None, matched_stop=None)
                    continued.append((at, self.own.continuation(self.request, at, self.prompts[at],
                                                                  self.drawn[at], stream=True)))
            choices.append(choice.model_copy(update=update))
        return chunk.model_copy(update={"choices": choices}), continued


def streamed(stream: Stream[OpenAIResponse], own: OwnStops, request: Request,
             opened: Callable[[Request], Stream[OpenAIResponse]]) -> Iterator[OpenAIResponse]:
    """A completions `stream`'s chunks with each choice SGLang ended on an id of its own outside the
    policy's stops continued by a completions stream of its own (`opened`), whose chunks follow
    under the choice's index. Every stream it opens is closed when it ends or the reader stops."""
    state = _Streamed(own, request)

    def run(chunks: Stream[OpenAIResponse], index: int | None) -> Iterator[OpenAIResponse]:
        with chunks:
            for chunk in chunks:
                out, continued = state.fed(chunk, index)
                yield out
                for at, follow in continued:
                    yield from run(opened(follow), at)

    return run(stream, None)


async def streamed_async(stream: AsyncStream[OpenAIResponse], own: OwnStops, request: Request,
                         opened: Callable[[Request], Awaitable[AsyncStream[OpenAIResponse]]]
                         ) -> AsyncIterator[OpenAIResponse]:
    """`streamed` over an async stream."""
    state = _Streamed(own, request)

    async def run(chunks: AsyncStream[OpenAIResponse], index: int | None) -> AsyncIterator[OpenAIResponse]:
        async with chunks:
            async for chunk in chunks:
                out, continued = state.fed(chunk, index)
                yield out
                for at, follow in continued:
                    async for later in run(await opened(follow), at):
                        yield later

    async for chunk in run(stream, None):
        yield chunk


def _chat_refusal(chunk: ChatCompletionChunk, own: OwnStops) -> str | None:
    """Why a chat stream's `chunk` cannot go on as the policy's draw: SGLang ended a choice on an id
    of its own outside the policy's stops, and a chat stream reports no token ids to continue it
    from. None where every choice in it is the policy's."""
    for choice in chunk.choices:
        matched = (choice.model_extra or {}).get("matched_stop")
        if own.ended_outside(choice.finish_reason, matched):
            return (f"SGLang ended chat stream choice {choice.index} on its model's EOS id {matched}, which "
                    "the policy does not stop on, and a chat stream reports no token ids to continue it "
                    "from; stream on the completions route (OpenAICompletion.stream) or ask for the whole "
                    "chat")
    return None


def chat_streamed(stream: Stream[ChatCompletionChunk], own: OwnStops) -> Iterator[ChatCompletionChunk]:
    """A chat `stream`'s chunks, refused at a chunk SGLang ended a choice of on an id of its own
    outside the policy's stops (`_chat_refusal`), the stream closed when it ends or the reader
    stops."""
    with stream:
        for chunk in stream:
            if (refusal := _chat_refusal(chunk, own)) is not None:
                raise ValueError(refusal)
            yield chunk


async def chat_streamed_async(stream: AsyncStream[ChatCompletionChunk], own: OwnStops
                              ) -> AsyncIterator[ChatCompletionChunk]:
    """`chat_streamed` over an async stream."""
    async with stream:
        async for chunk in stream:
            if (refusal := _chat_refusal(chunk, own)) is not None:
                raise ValueError(refusal)
            yield chunk
