"""Adapt the official Ollama and OpenAI Python clients to a bound model.

The convenience call returns associated text/usage records. Chat and stream
methods return the SDK's native responses, retaining tools, media, structured
outputs, thinking, token data and provider extensions. Provider options are
not squeezed into Dew's native Sampling value. No remote likelihood is
relabeled as a native raw-policy or behavior-policy likelihood.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Literal, Protocol

from dew.records import JSON
from dew.sampling.text import Sampling

if TYPE_CHECKING:
    from ollama import (
        AsyncClient as AsyncOllamaClient,
        ChatResponse as OllamaChat,
        Client as OllamaClient,
        GenerateResponse as OllamaResponse,
        Message as OllamaMessage,
        Options as OllamaOptions,
    )
    from openai import AsyncOpenAI, AsyncStream, OpenAI, Stream
    from openai.types import Completion as OpenAIResponse
    from openai.types.chat import ChatCompletion, ChatCompletionChunk

# One SDK request field as the adapters forward it: JSON the way the wire
# carries it, the bytes of an image, an ollama Options value, or the Sampling
# policy whose backend options the adapter derives. The SDK owns the schema,
# so a field is checked where it is read and passed on where it is not.
type RequestField = JSON | bytes | Sequence[bytes] | Sampling | OllamaOptions
# One chat message: the JSON object a request carries, or the SDK's own value.
type ChatMessage = Mapping[str, object] | OllamaMessage


class _HTTPResponse(Protocol):
    def json(self) -> JSON: ...


class _RawResponse[T](Protocol):
    """Describes what the SDK's public `with_raw_response` returns.

    It holds the parsed model and its HTTP response. `T` is the parsed model
    the caller relies on; a caller that narrows the body itself takes `object`.
    """

    @property
    def http_response(self) -> _HTTPResponse: ...

    def parse(self) -> T: ...


@dataclass(frozen=True)
class Usage:
    """Holds the reported aggregate usage. None means the backend did not report it."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class Completion:
    """Holds the choices in prompt-major order, with the SDK responses retained.

    Per-choice counts stay None when only aggregate usage was reported.
    finish_reasons retain the backend's values, including an absent reason.
    `tokens` and `log_probs` hold each choice's sampled ids and their
    log-probabilities where the backend reported them: a vLLM completion
    asked for `logprobs` with `return_tokens_as_token_ids` in extra_body
    reports both. The log-probabilities are whatever distribution the engine
    was configured to report; this record does not relabel them.
    """

    texts: tuple[str, ...]
    finish_reasons: tuple[str | None, ...]
    token_counts: tuple[int | None, ...]
    usage: Usage | None
    responses: tuple[OllamaResponse | OpenAIResponse, ...]
    tokens: tuple[tuple[int, ...] | None, ...]
    log_probs: tuple[tuple[float, ...] | None, ...]


def _invoke[T](call: Callable[..., T], fields: Mapping[str, object]) -> T:
    """Call `call` with `fields` as keyword arguments.

    The SDK owns the request schema, so nothing here validates the names.
    """
    return call(**fields)


async def _ainvoke[T](call: Callable[..., Awaitable[T]], fields: Mapping[str, object]) -> T:
    return await call(**fields)


def _object(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object")
    return {key: entry for key, entry in value.items() if isinstance(key, str)}


def _count(value: object, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer or absent")
    return value


def _reason(value: object) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError("finish reason must be a string or absent")
    return value


def _prompts(prompts: str | Sequence[str], budget: int, seed: int | None) -> list[str]:
    """Return `prompts` as a list of rows, refusing a bad budget, seed or row.

    A caller with nothing to tokenize passes "" and discards the rows, using
    this for the budget and seed checks alone.
    """
    if type(budget) is not int or budget < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer")
    if seed is not None and type(seed) is not int:
        raise ValueError("seed must be an integer or None")
    rows = [prompts] if isinstance(prompts, str) else list(prompts)
    if not rows or any(not isinstance(row, str) for row in rows):
        raise ValueError("prompts must be a nonempty sequence of strings")
    return rows


def _bound(options: Mapping[str, object], fixed: Mapping[str, object]) -> Mapping[str, object]:
    overlap = options.keys() & fixed.keys()
    if options.get("extra_body") is not None:
        extensions = _object(options["extra_body"], "extra_body")
        overlap |= extensions.keys() & (fixed.keys() | {"n"})
    if overlap:
        raise ValueError(f"fields {sorted(overlap)} are supplied by the bound task")
    return {**options, **fixed}


@lru_cache(maxsize=2)
def _ollama_request_names(kind: Literal["generate", "chat"]) -> frozenset[str]:
    """Return the request fields the SDK's public method accepts, from its signature."""
    from ollama import Client
    return frozenset(inspect.signature(getattr(Client, kind)).parameters) - {"self"}


def _ollama_budget(options: object, budget: int, seed: int | None,
                   sampling: object = None) -> Mapping[str, object]:
    from ollama import Options

    supplied = options.model_dump(exclude_none=True) if isinstance(options, Options) else options
    fields = {} if supplied is None else _object(supplied, "options")
    request_only = (_ollama_request_names("generate") | _ollama_request_names("chat")) - Options.model_fields.keys()
    misplaced = fields.keys() & request_only
    if misplaced:
        raise ValueError(f"{sorted(misplaced)} are request fields, not Ollama options")
    if sampling is not None:
        if not isinstance(sampling, Sampling):
            raise TypeError("sampling must be a Sampling value")
        if sampling.eos_id is not None or sampling.pad_id != 0:
            raise ValueError("Ollama text completion cannot implement native EOS-token or padding IDs")
        native: dict[str, object] = {"temperature": sampling.temperature,
                                     "top_k": 0 if sampling.top_k is None else sampling.top_k,
                                     "top_p": sampling.top_p, "min_p": sampling.min_p,
                                     "repeat_penalty": 1.0, "presence_penalty": 0.0,
                                     "frequency_penalty": 0.0, "typical_p": 1.0, "tfs_z": 1.0, "mirostat": 0}
        conflicts = [name for name in fields.keys() & native.keys() if fields[name] != native[name]]
        if conflicts:
            raise ValueError(f"options conflict with the requested Sampling policy: {sorted(conflicts)}")
        fields = {**fields, **native}
    fixed: dict[str, object] = {"num_predict": budget}
    if seed is not None:
        fixed["seed"] = seed
    return _bound(fields, fixed)


def _ollama_body(model: str, kind: Literal["generate", "chat"],
                 fields: Mapping[str, object]) -> Mapping[str, object]:
    """Build the keyword arguments for the SDK method.

    The SDK owns image, message and tool serialization.
    """
    body = _bound(fields, {"model": model})
    unknown = body.keys() - _ollama_request_names(kind)
    if unknown:
        raise ValueError(f"fields {sorted(unknown)} are not accepted by this Ollama SDK's {kind}")
    return body


def _ollama_result(responses: Sequence[OllamaResponse]) -> Completion:
    # The SDK exposes no raw-response hook, so these are its parsed values: it
    # rejects non-integral counts and non-text responses; negatives fail here.
    texts: list[str] = []
    counts: list[int | None] = []
    reasons: list[str | None] = []
    for response in responses:
        if not isinstance(response.response, str):
            raise ValueError("Ollama text completion is missing its response text")
        texts.append(response.response)
        counts.append(_count(response.eval_count, "eval_count"))
        reasons.append(_reason(response.done_reason))
    unreported = (None,) * len(texts)
    return Completion(tuple(texts), tuple(reasons), tuple(counts), None, tuple(responses),
                      unreported, unreported)


@dataclass(frozen=True)
class OllamaCompletion:
    """Bind a model to an injected ollama.Client or ollama.AsyncClient.

    __call__/acall accept batches of text and a finite token budget. Additional
    request fields use the SDK schema, including images, format, context,
    raw/template/system, think and logprobs; options accepts the SDK Options
    value or a mapping of backend options. chat/achat support tools and tool
    results. stream/astream yield native GenerateResponse objects.
    """

    model: str
    client: OllamaClient | AsyncOllamaClient

    def _sync(self) -> OllamaClient:
        from ollama import Client
        if not isinstance(self.client, Client):
            raise TypeError("use the async methods with ollama.AsyncClient")
        return self.client

    def _async(self) -> AsyncOllamaClient:
        from ollama import AsyncClient
        if not isinstance(self.client, AsyncClient):
            raise TypeError("async methods require ollama.AsyncClient")
        return self.client

    def _request(self, kind: Literal["generate", "chat"], supplied: Mapping[str, object],
                 fixed: Mapping[str, object], seed: int | None,
                 max_new_tokens: int) -> Mapping[str, object]:
        fields = dict(supplied)
        fields["options"] = _ollama_budget(fields.get("options"), max_new_tokens, seed,
                                           fields.pop("sampling", None))
        return _ollama_body(self.model, kind, _bound(fields, fixed))

    def __call__(self, prompts: str | Sequence[str], max_new_tokens: int, *,
                 seed: int | None = None, **parameters: RequestField) -> Completion:
        rows = _prompts(prompts, max_new_tokens, seed)
        client = self._sync()
        responses = []
        for index, prompt in enumerate(rows):
            body = self._request("generate", parameters, {"prompt": prompt, "stream": False},
                                 None if seed is None else seed + index, max_new_tokens)
            responses.append(_invoke(client.generate, body))
        return _ollama_result(responses)

    def stream(self, prompt: str, max_new_tokens: int, *, seed: int | None = None,
               **parameters: RequestField) -> Iterator[OllamaResponse]:
        _prompts(prompt, max_new_tokens, seed)
        body = self._request("generate", parameters, {"prompt": prompt, "stream": True}, seed, max_new_tokens)
        return _invoke(self._sync().generate, body)

    def chat(self, messages: Sequence[ChatMessage], max_new_tokens: int, *, seed: int | None = None,
             stream: bool = False, **parameters: RequestField) -> OllamaChat | Iterator[OllamaChat]:
        _prompts("", max_new_tokens, seed)
        body = self._request("chat", parameters, {"messages": messages, "stream": stream}, seed, max_new_tokens)
        return _invoke(self._sync().chat, body)

    async def acall(self, prompts: str | Sequence[str], max_new_tokens: int, *,
                    seed: int | None = None, **parameters: RequestField) -> Completion:
        rows = _prompts(prompts, max_new_tokens, seed)
        client = self._async()
        responses = []
        for index, prompt in enumerate(rows):
            body = self._request("generate", parameters, {"prompt": prompt, "stream": False},
                                 None if seed is None else seed + index, max_new_tokens)
            responses.append(await _ainvoke(client.generate, body))
        return _ollama_result(responses)

    async def astream(self, prompt: str, max_new_tokens: int, *, seed: int | None = None,
                      **parameters: RequestField) -> AsyncIterator[OllamaResponse]:
        _prompts(prompt, max_new_tokens, seed)
        body = self._request("generate", parameters, {"prompt": prompt, "stream": True}, seed, max_new_tokens)
        return await _ainvoke(self._async().generate, body)

    async def achat(self, messages: Sequence[ChatMessage], max_new_tokens: int, *, seed: int | None = None,
                    stream: bool = False, **parameters: RequestField) -> OllamaChat | AsyncIterator[OllamaChat]:
        _prompts("", max_new_tokens, seed)
        body = self._request("chat", parameters, {"messages": messages, "stream": stream}, seed, max_new_tokens)
        return await _ainvoke(self._async().chat, body)


def _choice_tokens(entry: Mapping[str, object]) -> tuple[tuple[int, ...] | None, tuple[float, ...] | None]:
    """Read one choice's reported sampled ids and log-probabilities, if any.

    Ids come from the choice's `token_ids` list when the engine returns one
    (SGLang's `return_token_ids`), else from vLLM's `token_id:<n>` rendering
    of the logprob tokens; text tokens are not reverse-mapped through a
    vocabulary.
    """
    listed = entry.get("token_ids")
    if listed is not None and (not isinstance(listed, list) or any(
            type(token) is not int or token < 0 for token in listed)):
        raise ValueError("choice token_ids must be a list of nonnegative ids")
    ids = None if listed is None else tuple(listed)
    logprobs = entry.get("logprobs")
    if logprobs is None:
        return ids, None
    record = _object(logprobs, "choice logprobs")
    rendered, values = record.get("tokens"), record.get("token_logprobs")
    if not isinstance(rendered, list) or not isinstance(values, list) or len(rendered) != len(values):
        raise ValueError("choice logprobs need aligned tokens and token_logprobs lists")
    probabilities: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("each reported token log-probability must be a number")
        probabilities.append(float(value))
    if ids is not None:
        if len(ids) != len(probabilities):
            raise ValueError("choice token_ids must be aligned with its token_logprobs")
        return ids, tuple(probabilities)
    rendered_ids: list[int] = []
    for token in rendered:
        if not isinstance(token, str) or not token.startswith("token_id:"):
            return None, tuple(probabilities)
        rendered_ids.append(int(token.removeprefix("token_id:")))
    return tuple(rendered_ids), tuple(probabilities)


def _openai_result(raw: JSON, response: object, expected: int) -> Completion:
    from openai.types import Completion as SDKCompletion
    if not isinstance(response, SDKCompletion):
        raise ValueError("a non-streaming completion request returned an incompatible response")
    fields = _object(raw, "completion response")
    choices = fields.get("choices")
    if not isinstance(choices, list) or len(choices) != expected:
        raise ValueError(f"expected {expected} completion choices")
    ordered: dict[int, tuple[str, str | None, tuple[int, ...] | None, tuple[float, ...] | None]] = {}
    for choice in choices:
        entry = _object(choice, "completion choice")
        index, text = entry.get("index"), entry.get("text")
        if type(index) is not int or index < 0 or index >= expected or index in ordered:
            raise ValueError("choice indices must be a permutation of the expected prompt-choice indices")
        if not isinstance(text, str):
            raise ValueError("each completion choice needs string text")
        ordered[index] = (text, _reason(entry.get("finish_reason")), *_choice_tokens(entry))
    usage = None
    if fields.get("usage") is not None:
        supplied = _object(fields["usage"], "usage")
        usage = Usage(*(_count(supplied.get(name), name) for name in
                        ("prompt_tokens", "completion_tokens", "total_tokens")))
    per_choice = (usage.completion_tokens,) if expected == 1 and usage is not None else (None,) * expected
    choice = [ordered[index] for index in range(expected)]
    return Completion(tuple(entry[0] for entry in choice), tuple(entry[1] for entry in choice),
                      per_choice, usage, (response,), tuple(entry[2] for entry in choice),
                      tuple(entry[3] for entry in choice))


type TokenRows = Sequence[Sequence[int]]
"""Prompts as token ids, one row per prompt, which the completions API accepts in place of text."""


def _token_rows(prompts: object) -> list[list[int]] | None:
    """Read `prompts` as token-id rows, or None when they are text."""
    if isinstance(prompts, str) or not isinstance(prompts, Sequence) or not prompts:
        return None
    if all(isinstance(row, str) for row in prompts):
        return None
    rows: list[list[int]] = []
    for row in prompts:
        if isinstance(row, str) or not isinstance(row, Sequence) or not row:
            raise ValueError("token prompts must be nonempty rows of token ids")
        if any(type(token) is not int or token < 0 for token in row):
            raise ValueError("token prompts must hold nonnegative integer ids")
        rows.append(list(row))
    return rows


def _openai_fields(model: str, prompts: str | Sequence[str] | TokenRows, budget: int, seed: int | None,
                   parameters: Mapping[str, object], *,
                   stream: bool = False) -> tuple[Mapping[str, object], int]:
    tokens = _token_rows(prompts)
    if tokens is None:
        # No token rows: one string, or rows that are all strings (an empty
        # list reaches `_prompts` empty and is refused there).
        texts = _prompts(prompts if isinstance(prompts, str) else [row for row in prompts if isinstance(row, str)],
                         budget, seed)
        count = len(texts)
        prompt: object = prompts if isinstance(prompts, str) else texts
    else:
        _prompts("", budget, seed)
        count, prompt = len(tokens), tokens
    n = parameters.get("n", 1)
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    fixed: dict[str, object] = {"model": model, "prompt": prompt, "max_tokens": budget, "stream": stream}
    if seed is not None:
        fixed["seed"] = seed
    return _bound(parameters, fixed), count * n


@dataclass(frozen=True)
class OpenAICompletion:
    """Bind an OpenAI client, including one configured for a vLLM or SGLang base_url.

    Native SDK request options pass through to completion/chat resources.
    Engine-only controls such as top_k/min_p/stop_token_ids, which vLLM and
    SGLang both accept, belong explicitly in extra_body. SDK responses and
    streaming chunks retain backend logprobs, token IDs/extensions, tool calls
    and structured output fields unchanged. Completion prompts are text or
    token-id rows; a row of ids reaches the engine as ids, with no
    detokenize/retokenize round trip.
    """

    model: str
    client: OpenAI | AsyncOpenAI
    provider: Literal["openai", "vllm", "sglang"] = "openai"

    def __post_init__(self) -> None:
        if self.provider not in ("openai", "vllm", "sglang"):
            raise ValueError("provider must be openai, vllm or sglang")

    def _parameters(self, supplied: Mapping[str, object]) -> Mapping[str, object]:
        fields = dict(supplied)
        sampling = fields.pop("sampling", None)
        if sampling is None:
            return fields
        if not isinstance(sampling, Sampling):
            raise TypeError("sampling must be a Sampling value")
        if sampling.pad_id != 0:
            raise ValueError("remote text completion does not implement padded token rows")
        if self.provider == "openai" and (sampling.top_k is not None or sampling.min_p != 0 or sampling.eos_id is not None):
            raise ValueError("top-k, min-p and EOS-token controls require provider='vllm' or 'sglang'")
        native: dict[str, object] = {"temperature": sampling.temperature, "top_p": sampling.top_p,
                                     "frequency_penalty": 0.0, "presence_penalty": 0.0}
        controls: dict[str, object] = {"top_k": -1 if sampling.top_k is None else sampling.top_k,
                                      "min_p": sampling.min_p, "repetition_penalty": 1.0}
        if sampling.eos_id is not None:
            controls["stop_token_ids"] = list(sampling.eos_id) if isinstance(sampling.eos_id, tuple) else [sampling.eos_id]
        extra = {} if fields.get("extra_body") is None else _object(fields["extra_body"], "extra_body")
        if self.provider == "openai" and controls.keys() & extra.keys():
            raise ValueError("top-k, min-p, repetition and EOS-token controls in extra_body require provider='vllm' or 'sglang'")
        # The SDK writes extra_body over the named parameters, so the policy
        # is checked against both namespaces of the final request body.
        policy = {**native, **controls}
        conflicts = sorted(name for name in policy
                           if (name in fields and fields[name] != policy[name])
                           or (name in extra and extra[name] != policy[name]))
        if conflicts:
            raise ValueError(f"parameters conflict with Sampling: {conflicts}")
        fields.update(native)
        if self.provider != "openai":
            fields["extra_body"] = {**extra, **controls}
        return fields

    def _sync(self) -> OpenAI:
        from openai import OpenAI
        if not isinstance(self.client, OpenAI):
            raise TypeError("use async methods with AsyncOpenAI")
        return self.client

    def _async(self) -> AsyncOpenAI:
        from openai import AsyncOpenAI
        if not isinstance(self.client, AsyncOpenAI):
            raise TypeError("async methods require AsyncOpenAI")
        return self.client

    def __call__(self, prompts: str | Sequence[str] | TokenRows, max_new_tokens: int, *,
                 seed: int | None = None, **parameters: RequestField) -> Completion:
        fields, expected = _openai_fields(self.model, prompts, max_new_tokens, seed, self._parameters(parameters))
        create: Callable[..., _RawResponse[object]] = self._sync().completions.with_raw_response.create
        raw = _invoke(create, fields)
        return _openai_result(raw.http_response.json(), raw.parse(), expected)

    def stream(self, prompts: str | Sequence[str] | TokenRows, max_new_tokens: int, *, seed: int | None = None,
               **parameters: RequestField) -> Stream[OpenAIResponse]:
        fields, _ = _openai_fields(self.model, prompts, max_new_tokens, seed,
                                   self._parameters(parameters), stream=True)
        create: Callable[..., Stream[OpenAIResponse]] = self._sync().completions.create
        return _invoke(create, fields)

    def _chat_body(self, messages: Sequence[ChatMessage], max_new_tokens: int,
                   seed: int | None, stream: bool,
                   parameters: Mapping[str, RequestField]) -> Mapping[str, object]:
        """Build the chat request body for `messages`.

        `_prompts` runs for its checks on the budget and the seed; a chat
        request carries messages where a completion carries prompt strings.
        """
        _prompts("", max_new_tokens, seed)
        fields = _bound(self._parameters(parameters), {"model": self.model, "messages": messages, "max_completion_tokens": max_new_tokens, "stream": stream})
        return fields if seed is None else _bound(fields, {"seed": seed})

    def chat(self, messages: Sequence[ChatMessage], max_new_tokens: int, *, seed: int | None = None,
             stream: bool = False, **parameters: RequestField) -> ChatCompletion | Stream[ChatCompletionChunk]:
        fields = self._chat_body(messages, max_new_tokens, seed, stream, parameters)
        create: Callable[..., ChatCompletion | Stream[ChatCompletionChunk]] = self._sync().chat.completions.create
        return _invoke(create, fields)

    async def acall(self, prompts: str | Sequence[str] | TokenRows, max_new_tokens: int, *,
                    seed: int | None = None, **parameters: RequestField) -> Completion:
        fields, expected = _openai_fields(self.model, prompts, max_new_tokens, seed, self._parameters(parameters))
        raw = await _ainvoke(self._async().completions.with_raw_response.create, fields)
        return _openai_result(raw.http_response.json(), raw.parse(), expected)

    async def astream(self, prompts: str | Sequence[str] | TokenRows, max_new_tokens: int, *, seed: int | None = None,
                      **parameters: RequestField) -> AsyncStream[OpenAIResponse]:
        fields, _ = _openai_fields(self.model, prompts, max_new_tokens, seed,
                                   self._parameters(parameters), stream=True)
        return await _ainvoke(self._async().completions.create, fields)

    async def achat(self, messages: Sequence[ChatMessage], max_new_tokens: int, *, seed: int | None = None,
                    stream: bool = False, **parameters: RequestField) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
        fields = self._chat_body(messages, max_new_tokens, seed, stream, parameters)
        return await _ainvoke(self._async().chat.completions.create, fields)
