"""A policy's stop ids carried through SGLang, which also ends a draw on its model's own EOS ids.

SGLang ends a draw on `stop_token_ids` and on its model's EOS ids unless the
request ignores every token stop, and no field turns its own off alone
(sglang/srt/managers/schedule_batch.py:1707-1729 at v0.5.20). So a completion
SGLang ended on an id the policy's stops leave out (the choice's
`matched_stop`) goes on as the policy's draw would: its prompt and the ids it
drew become the prompt of a continuation, drawn under the same policy on the
rest of its budget, until it ends on one of the policy's stops or the budget.
That is sequential sampling, so the joined draw is the policy's; under a seed
it is the uninterrupted draw itself, as SGLang keys a seeded draw by the seed
and each token's place in the sequence (`multinomial_with_seed`,
sglang/srt/layers/sampler.py), and gives every parallel sample of a request
that seed, which the continuation keeps.

A continuation carries what shapes the policy (`OwnStops.continuation`): its
sampling controls and seed, its row's adapter and cache keys, and a frequency
penalty, whose counts of the ids drawn
before it go in as logit bias, which SGLang adds to the same logits. A
continued choice's text is decoded again from the joined ids as SGLang's
detokenizer decodes a draw, after its prompt's last ids (`OwnStops.decoding`).
What a continuation cannot carry is refused where a draw needs one, by name
(`_uncarried`): a presence or repetition penalty, which counts the ids before
it once whichever segment draws them again; stop strings, a stop regex and
constrained decoding, whose matching restarts; and special tokens decoded
without spaces between them, which `/v1/detokenize` does not take. An owed
minimum output is refused before a draw (`OwnStops.refusal`): SGLang masks its
own EOS ids too while one is owed, which the policy draws as ordinary ids.

This carries the completions route (`OpenAICompletion.__call__` and
`acall`). A chat, and a stream on either route, that SGLang ended on its own
EOS outside the policy's stops is refused there by name (`chat_refusal`,
`Checked`), where it would otherwise end early.
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
"""One request a continuation sends: a completions request, or a detokenization of ids with the
SDK's transport options (`TRANSPORT`) beside its body."""

type Plan[Result] = Generator[Call, JSON, Result]
"""A computation that yields the requests it sends, is sent each one's response JSON, and returns
`Result`: `driven` and `driven_async` answer it."""

TRANSPORT = ("extra_headers", "extra_query", "timeout")
"""The SDK's per-request transport options, which every request of a draw carries."""

_CARRIED = ("model", "temperature", "top_p", "frequency_penalty", "seed", "user", *TRANSPORT)
"""The named request fields a continuation keeps as they are: the policy's controls, its seed,
and the transport. `extra_body` carries the engine's own controls; the rest are written per
continuation (`OwnStops.continuation`)."""

_WRITTEN = ("prompt", "max_tokens", "n", "stream", "suffix", "logprobs", "logit_bias", "echo", "rid",
            "stream_options")
"""The fields a continuation writes itself, which an `extra_body` entry would override in the SDK's
request body (it merges `extra_body` over the named fields), so none is left there."""

_ROWS = ("lora_path", "extra_key", "cache_salt")
"""The engine controls a request may give per prompt row, which a continuation of one row takes
its own of."""

_RESTARTED = ("stop", "stop_regex", "regex", "json_schema", "ebnf", "structural_tag", "response_format")
"""The request fields whose matching a continuation would start again with the ids before it as
prompt: stop strings and patterns, and constrained decoding."""

_CONTEXT = 5
"""How many prompt ids SGLang's detokenizer decodes a draw after, so a draw's first token reads as
it does inside the sequence (`INIT_INCREMENTAL_DETOKENIZATION_OFFSET`, schedule_batch.py:156)."""


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


def _effective(request: Request) -> Mapping[str, object]:
    """The request's fields as the SDK sends them: `extra_body` merged over the named ones."""
    return {**{name: value for name, value in request.items() if name != "extra_body"},
            **_fields(request.get("extra_body"), "extra_body")}


def _switch(fields: Mapping[str, object], name: str, *, default: bool) -> bool:
    """What `fields` set the switch `name` to, `default` where they leave it unset."""
    value = fields.get(name)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def _uncarried(request: Request) -> str | None:
    """The control of `request` a continuation cannot carry exactly, named, or None where it carries
    them all (the module's account)."""
    fields = _effective(request)
    restarted = [name for name in _RESTARTED if fields.get(name) not in (None, "", [], {})]
    if restarted:
        return f"{', '.join(restarted)}, whose matching would start again at the cut"
    for name, neutral in (("presence_penalty", 0), ("repetition_penalty", 1)):
        if fields.get(name) not in (None, neutral):
            return f"{name}, which counts each id drawn before the cut once whichever segment draws it again"
    return _undecodable(fields)


def _undecodable(fields: Mapping[str, object]) -> str | None:
    """The control of a request's `fields` its joined draw's text cannot be decoded again under,
    named, or None where `/v1/detokenize` decodes it as the draw's own was."""
    if not _switch(fields, "spaces_between_special_tokens", default=True):
        return "spaces_between_special_tokens=False, which SGLang's /v1/detokenize does not take"
    return None


@dataclasses.dataclass(frozen=True)
class OwnStops:
    """The policy's stop ids `stops` as one SGLang request draws them on `budget` tokens a choice."""

    stops: frozenset[int]
    budget: int

    def refusal(self, request: Request) -> str | None:
        """Why `request` cannot draw the policy on SGLang whatever it draws, or None: an owed
        minimum output, while which SGLang masks its model's own EOS ids (penaltylib/
        min_new_tokens.py), ids a policy draws as any other where its stops leave them out, a
        policy with no stops of its own among them."""
        least = _effective(request).get("min_tokens")
        if least in (None, 0):
            return None
        return (f"min_tokens={least!r} masks SGLang's model EOS ids while it is owed, and the policy "
                "draws those ids as any other; drop min_tokens or draw on vLLM")

    def ended_outside(self, reason: JSON, matched: JSON) -> bool:
        """Whether SGLang ended a choice on an id of its own the policy's stops leave out."""
        return reason == "stop" and type(matched) is int and matched not in self.stops

    def continuation(self, request: Request, index: int, prompt: Sequence[int],
                     drawn: Sequence[int]) -> Request:
        """The completions request that draws on choice `index` of `request` from `prompt` and the ids
        it `drawn`, under the same policy on the rest of the budget: its own row's per-row controls,
        its logprob count, and a frequency penalty's counts of `drawn` as logit bias. Raises
        `ValueError` for a control it cannot carry (`_uncarried`)."""
        if (uncarried := _uncarried(request)) is not None:
            raise ValueError(
                f"SGLang ended choice {index} on its model's EOS id {drawn[-1]}, which the policy does not "
                f"stop on, and Dew goes on from the ids it drew, which cannot carry {uncarried}. Name "
                f"{drawn[-1]} among the policy's EOS ids, drop that control, or draw on vLLM")
        fields = _effective(request)
        extra = {name: value for name, value in _fields(request.get("extra_body"), "extra_body").items()
                 if name not in (*_WRITTEN, "return_prompt_token_ids")}
        extra["return_token_ids"] = True
        prompts = fields.get("prompt")
        batched = isinstance(prompts, list) and bool(prompts) and not isinstance(prompts[0], int)
        rows = len(prompts) if batched and isinstance(prompts, list) else 1
        samples = fields.get("n")
        # SGLang answers a batch prompt by prompt, each prompt's samples together
        # (tokenizer_manager.py `_handle_batch_request`).
        row = index // (samples if isinstance(samples, int) and samples > 0 else 1)
        for name in _ROWS:
            if isinstance(given := extra.get(name), list) and len(given) == rows:
                extra[name] = given[row]
        body: dict[str, object] = {name: request[name] for name in _CARRIED if name in request}
        bias = {str(token): value for token, value in _fields(fields.get("logit_bias"), "logit_bias").items()}
        frequency = fields.get("frequency_penalty")
        if isinstance(frequency, int | float) and frequency:
            for token, count in Counter(drawn).items():
                bias[str(token)] = _number(bias.get(str(token), 0)) - frequency * count
        if bias:
            body["logit_bias"] = bias
        if fields.get("logprobs") is not None:
            body["logprobs"] = fields["logprobs"]
        # An empty suffix is SGLang's own pass-through of an already formatted prompt, whose
        # completion template would otherwise rewrite it (code_completion_parser.py).
        return {**body, "prompt": [[*prompt, *drawn]], "max_tokens": self.budget - len(drawn), "n": 1,
                "suffix": "", "extra_body": extra}

    def decoding(self, request: Request, prompt: Sequence[int], choice: Mapping[str, JSON]) -> Request:
        """The `/v1/detokenize` request that decodes a continued `choice` as SGLang's detokenizer
        decodes a draw (detokenizer_manager.py `_decode_batch_token_id_output`): the prompt's last
        `_CONTEXT` ids, and those with the choice's ids after them, the draw's text being what the
        second adds to the first; special tokens skipped as `request` asks, and a policy stop it
        ended on trimmed unless the request keeps it (`trim_matched_stop`). It carries the
        request's transport options."""
        fields = _effective(request)
        ids = _ids(choice.get("token_ids"), "token_ids")
        trimmed = (choice.get("finish_reason") == "stop" and type(choice.get("matched_stop")) is int
                   and not _switch(fields, "no_stop_trim", default=False))
        context = list(prompt[-_CONTEXT:])
        return {**{name: request[name] for name in TRANSPORT if name in request},
                "model": request.get("model"),
                "tokens": [context, [*context, *(ids[:-1] if trimmed else ids)]],
                "skip_special_tokens": _switch(fields, "skip_special_tokens", default=True)}

    def echoed(self, request: Request, index: int) -> Call | str | None:
        """What a continued choice's text starts with where `request` echoes its prompt, as SGLang
        writes it (serving_completions.py `_prepare_echo_prompts`): the request's own prompt row,
        not the model prompt a completion template may have made of it, its text as it is or its
        token ids decoded with special tokens skipped; None without echo."""
        fields = _effective(request)
        if not _switch(fields, "echo", default=False):
            return None
        prompts, samples = fields.get("prompt"), fields.get("n")
        row = index // (samples if isinstance(samples, int) and samples > 0 else 1)
        if isinstance(prompts, str):
            return prompts
        if not isinstance(prompts, list) or not prompts:
            raise ValueError("an echoed request needs its prompt")
        chosen = prompts if isinstance(prompts[0], int) else prompts[row]
        if isinstance(chosen, str):
            return chosen
        if not isinstance(chosen, list) or any(type(token) is not int for token in chosen):
            raise ValueError("a prompt row is text or token ids")
        return "detokenize", {**{name: request[name] for name in TRANSPORT if name in request},
                              "model": request.get("model"), "tokens": list(chosen),
                              "skip_special_tokens": True}

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
            prompt: list[int] = []
            redrawn = False
            while self.ended_outside(entry.get("finish_reason"), entry.get("matched_stop")):
                prompt = prompt if redrawn else _ids(entry.get("prompt_token_ids"), "prompt_token_ids")
                redrawn = True
                drawn = _ids(entry.get("token_ids"), "token_ids")
                if len(drawn) >= self.budget:
                    entry = {**entry, "finish_reason": "length", "matched_stop": None}
                    break
                follow = self.continuation(request, index if type(index) is int else 0, prompt, drawn)
                answer = _record((yield "completions", follow), "completion response")
                (then,) = _listed(answer.get("choices"), "continued choices")
                entry = _joined(entry, _record(then, "completion choice"))
                usage = _summed(usage, answer.get("usage"))
            if redrawn:
                if (undecodable := _undecodable(_effective(request))) is not None:
                    raise ValueError(f"SGLang ended choice {index} on its model's EOS id, which the policy "
                                     f"does not stop on; its text cannot be decoded again with {undecodable}")
                decoded = _record((yield "detokenize", self.decoding(request, prompt, entry)), "detokenized")
                context, whole = _listed(decoded.get("text"), "detokenized text")
                if not (isinstance(context, str) and isinstance(whole, str)):
                    raise ValueError("SGLang's detokenization must answer text")
                # SGLang's own offset, not a prefix match: a prompt's broken UTF-8 tail can read
                # otherwise once the draw completes it.
                text = whole[len(context):]
                echo = self.echoed(request, index if type(index) is int else 0)
                if isinstance(echo, tuple):
                    echo = _record((yield echo), "detokenized echo").get("text")
                    if not isinstance(echo, str):
                        raise ValueError("SGLang's detokenization must answer text")
                entry = {**entry, "text": text if echo is None else echo + text}
            choices.append(entry)
        return {**response, "choices": choices, "usage": usage}


def chat_refusal(response: ChatCompletion | ChatCompletionChunk | OpenAIResponse,
                 own: OwnStops) -> str | None:
    """Why a chat response, or a streamed chunk of either route, cannot stand as the policy's draw:
    SGLang ended a choice of it on an id of its own outside the policy's stops, which only the
    completions route carries on. None where every choice in it is the policy's."""
    for choice in response.choices:
        matched = (choice.model_extra or {}).get("matched_stop")
        if own.ended_outside(choice.finish_reason, matched):
            return (f"SGLang ended choice {choice.index} on its model's EOS id {matched}, which the policy "
                    "does not stop on, and only a whole completion goes on past it; draw it through "
                    "OpenAICompletion.__call__ (or acall), or name that id among the policy's EOS ids")
    return None


def _number(value: object) -> float:
    """A logit bias, read at the SDK request's boundary as the number it is."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("a logit bias must be a number")
    return value


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


class Checked[ChunkT: (OpenAIResponse, ChatCompletionChunk)]:
    """An SGLang `stream`'s chunks as they come, refused at a chunk SGLang ended a choice of on an id
    of its own outside the policy's stops (`chat_refusal`). It owns the stream: closing it, or
    leaving its `with` block, closes the stream, before the first read too."""

    def __init__(self, stream: Stream[ChunkT], own: OwnStops):
        self._stream, self._own = stream, own

    @property
    def response(self) -> httpx2.Response:
        """The stream's HTTP response, as the SDK's stream names it."""
        return self._stream.response

    def __iter__(self) -> Checked[ChunkT]:
        return self

    def __next__(self) -> ChunkT:
        chunk = next(self._stream)
        if (refusal := chat_refusal(chunk, self._own)) is not None:
            self.close()
            raise ValueError(refusal)
        return chunk

    def close(self) -> None:
        self._stream.close()

    def __enter__(self) -> Checked[ChunkT]:
        return self

    def __exit__(self, kind: type[BaseException] | None, raised: BaseException | None,
                 traceback: TracebackType | None) -> None:
        self.close()


class AsyncChecked[ChunkT: (OpenAIResponse, ChatCompletionChunk)]:
    """`Checked` over an async stream; `aclose` closes it, as the SDK's async stream's does."""

    def __init__(self, stream: AsyncStream[ChunkT], own: OwnStops):
        self._stream, self._own = stream, own

    @property
    def response(self) -> httpx2.Response:
        """The stream's HTTP response, as the SDK's stream names it."""
        return self._stream.response

    def __aiter__(self) -> AsyncChecked[ChunkT]:
        return self

    async def __anext__(self) -> ChunkT:
        chunk = await self._stream.__anext__()
        if (refusal := chat_refusal(chunk, self._own)) is not None:
            await self.aclose()
            raise ValueError(refusal)
        return chunk

    async def aclose(self) -> None:
        await self._stream.close()

    async def close(self) -> None:
        await self.aclose()

    async def __aenter__(self) -> AsyncChecked[ChunkT]:
        return self

    async def __aexit__(self, kind: type[BaseException] | None, raised: BaseException | None,
                        traceback: TracebackType | None) -> None:
        await self.aclose()
