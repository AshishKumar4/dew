"""Bound-model adapters over the official Ollama and OpenAI Python clients.

The convenience call returns associated text/usage records. Chat and stream
methods return the SDK's native responses, retaining tools, media, structured
outputs, thinking, token data and provider extensions. Provider options are
not squeezed into Dew's native Sampling value. No remote likelihood is
relabeled as a native raw-policy or behavior-policy likelihood.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

from dew.sampling.text import Sampling

if TYPE_CHECKING:
    from ollama import AsyncClient as AsyncOllamaClient, Client as OllamaClient
    from ollama import ChatResponse as OllamaChat, GenerateResponse as OllamaResponse
    from openai import AsyncOpenAI, AsyncStream, OpenAI, Stream
    from openai.types import Completion as OpenAIResponse
    from openai.types.chat import ChatCompletion, ChatCompletionChunk
    from openai._legacy_response import LegacyAPIResponse


@dataclass(frozen=True)
class Usage:
    """Reported aggregate usage; None means the backend did not report it."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class Completion:
    """Choices in prompt-major order, with the original SDK responses retained.

    Per-choice counts stay None when only aggregate usage was reported.
    finish_reasons retain the backend's values, including an absent reason.
    """

    texts: tuple[str, ...]
    finish_reasons: tuple[str | None, ...]
    token_counts: tuple[int | None, ...]
    usage: Usage | None
    responses: tuple[OllamaResponse | OpenAIResponse, ...]


def _invoke[T](call: Callable[..., T], fields: Mapping[str, object]) -> T:
    """Forward SDK-owned options without maintaining a second parameter schema."""
    return call(**fields)


async def _ainvoke[T](call: Callable[..., Awaitable[T]], fields: Mapping[str, object]) -> T:
    return await call(**fields)


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object")
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _count(value: object, name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer or absent")
    return value


def _reason(value: object) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError("finish reason must be a string or absent")
    return value


def _prompts(prompts: str | Sequence[str], budget: int, seed: int | None) -> list[str]:
    if type(budget) is not int or budget < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer")
    if seed is not None and type(seed) is not int:
        raise ValueError("seed must be an integer or None")
    rows = [prompts] if isinstance(prompts, str) else list(prompts)
    if not rows or any(not isinstance(row, str) for row in rows):
        raise ValueError("prompts must be a nonempty sequence of strings")
    return rows


def _bound(options: Mapping[str, object], fixed: Mapping[str, object]) -> dict[str, object]:
    overlap = options.keys() & fixed.keys()
    if options.get("extra_body") is not None:
        extensions = _object(options["extra_body"], "extra_body")
        overlap |= extensions.keys() & (fixed.keys() | {"n"})
    if overlap:
        raise ValueError(f"fields {sorted(overlap)} are supplied by the bound task")
    return {**options, **fixed}


def _ollama_budget(options: object, budget: int, seed: int | None, sampling: object = None) -> dict[str, object]:
    from ollama import Options
    from ollama._types import GenerateRequest, ChatRequest

    supplied = options.model_dump(exclude_none=True) if isinstance(options, Options) else options
    fields = {} if supplied is None else _object(supplied, "options")
    request_only = (GenerateRequest.model_fields.keys() | ChatRequest.model_fields.keys()) - Options.model_fields.keys()
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


def _ollama_body(model: str, path: str, fields: dict[str, object]) -> dict[str, object]:
    # SDK request models own image serialization, tool/schema fields and options.
    from ollama._types import ChatRequest, GenerateRequest
    from ollama._client import _copy_images, _copy_messages, _copy_tools

    request_type = ChatRequest if path == "/api/chat" else GenerateRequest
    body = _bound(fields, {"model": model})
    for name, convert in (("images", _copy_images), ("messages", _copy_messages), ("tools", _copy_tools)):
        if name in body:
            values = body[name]
            if values is not None and not isinstance(values, Sequence):
                raise TypeError(f"{name} must be a sequence")
            body[name] = list(convert(values))
    unknown = body.keys() - request_type.model_fields.keys()
    if unknown:
        raise ValueError(f"fields {sorted(unknown)} are not supported by this Ollama SDK")
    return request_type.model_validate(body).model_dump(exclude_none=True)


@lru_cache(maxsize=1)
def _ollama_responses() -> tuple[type[OllamaResponse], type[OllamaChat]]:
    from ollama import ChatResponse, GenerateResponse
    from pydantic import model_validator

    def checked(value: object) -> object:
        fields = _object(value, "Ollama response")
        for name in ("eval_count", "prompt_eval_count"):
            _count(fields.get(name), name)
        _reason(fields.get("done_reason"))
        if "response" in fields and not isinstance(fields["response"], str):
            raise ValueError("Ollama response text must be a string")
        return value

    class Generate(GenerateResponse):
        @model_validator(mode="before")
        @classmethod
        def validate_wire(cls, value: object) -> object:
            return checked(value)

    class Chat(ChatResponse):
        @model_validator(mode="before")
        @classmethod
        def validate_wire(cls, value: object) -> object:
            return checked(value)

    return Generate, Chat


def _ollama_result(responses: Sequence[OllamaResponse]) -> Completion:
    texts: list[str] = []
    counts: list[int | None] = []
    reasons: list[str | None] = []
    for response in responses:
        if not isinstance(response.response, str):
            raise ValueError("Ollama text completion is missing its response text")
        texts.append(response.response)
        counts.append(_count(response.eval_count, "eval_count"))
        reasons.append(_reason(response.done_reason))
    return Completion(tuple(texts), tuple(reasons), tuple(counts), None, tuple(responses))


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

    def __call__(self, prompts: str | Sequence[str], max_new_tokens: int, *,
                 seed: int | None = None, **parameters: object) -> Completion:
        rows = _prompts(prompts, max_new_tokens, seed)
        response_type, _ = _ollama_responses()
        responses = []
        for index, prompt in enumerate(rows):
            fields = dict(parameters)
            fields["options"] = _ollama_budget(fields.get("options"), max_new_tokens, None if seed is None else seed + index, fields.pop("sampling", None))
            body = _ollama_body(self.model, "/api/generate", _bound(fields, {"prompt": prompt, "stream": False}))
            # _request is the SDK's typed response hook. Using a checked SDK
            # subclass rejects bool/float counts before Pydantic can coerce them;
            # HTTP, errors and streaming framing remain the SDK's implementation.
            responses.append(self._sync()._request(response_type, "POST", "/api/generate", json=body))
        return _ollama_result(responses)

    def stream(self, prompt: str, max_new_tokens: int, *, seed: int | None = None,
               **parameters: object) -> Iterator[OllamaResponse]:
        _prompts(prompt, max_new_tokens, seed)
        fields = dict(parameters)
        fields["options"] = _ollama_budget(fields.get("options"), max_new_tokens, seed, fields.pop("sampling", None))
        body = _ollama_body(self.model, "/api/generate", _bound(fields, {"prompt": prompt, "stream": True}))
        response_type, _ = _ollama_responses()
        return self._sync()._request(response_type, "POST", "/api/generate", json=body, stream=True)

    def chat(self, messages: Sequence[object], max_new_tokens: int, *, seed: int | None = None,
             stream: bool = False, **parameters: object) -> OllamaChat | Iterator[OllamaChat]:
        _prompts("", max_new_tokens, seed)
        fields = dict(parameters)
        fields["options"] = _ollama_budget(fields.get("options"), max_new_tokens, seed, fields.pop("sampling", None))
        body = _ollama_body(self.model, "/api/chat", _bound(fields, {"messages": messages, "stream": stream}))
        _, response_type = _ollama_responses()
        return self._sync()._request(response_type, "POST", "/api/chat", json=body, stream=stream)

    async def acall(self, prompts: str | Sequence[str], max_new_tokens: int, *,
                    seed: int | None = None, **parameters: object) -> Completion:
        rows = _prompts(prompts, max_new_tokens, seed)
        response_type, _ = _ollama_responses()
        responses = []
        for index, prompt in enumerate(rows):
            fields = dict(parameters)
            fields["options"] = _ollama_budget(fields.get("options"), max_new_tokens, None if seed is None else seed + index, fields.pop("sampling", None))
            body = _ollama_body(self.model, "/api/generate", _bound(fields, {"prompt": prompt, "stream": False}))
            responses.append(await self._async()._request(response_type, "POST", "/api/generate", json=body))
        return _ollama_result(responses)

    async def astream(self, prompt: str, max_new_tokens: int, *, seed: int | None = None,
                      **parameters: object) -> AsyncIterator[OllamaResponse]:
        _prompts(prompt, max_new_tokens, seed)
        fields = dict(parameters)
        fields["options"] = _ollama_budget(fields.get("options"), max_new_tokens, seed, fields.pop("sampling", None))
        body = _ollama_body(self.model, "/api/generate", _bound(fields, {"prompt": prompt, "stream": True}))
        response_type, _ = _ollama_responses()
        return await self._async()._request(response_type, "POST", "/api/generate", json=body, stream=True)

    async def achat(self, messages: Sequence[object], max_new_tokens: int, *, seed: int | None = None,
                    stream: bool = False, **parameters: object) -> OllamaChat | AsyncIterator[OllamaChat]:
        _prompts("", max_new_tokens, seed)
        fields = dict(parameters)
        fields["options"] = _ollama_budget(fields.get("options"), max_new_tokens, seed, fields.pop("sampling", None))
        body = _ollama_body(self.model, "/api/chat", _bound(fields, {"messages": messages, "stream": stream}))
        _, response_type = _ollama_responses()
        return await self._async()._request(response_type, "POST", "/api/chat", json=body, stream=stream)


def _openai_result(raw: object, response: object, expected: int) -> Completion:
    from openai.types import Completion as SDKCompletion
    if not isinstance(response, SDKCompletion):
        raise ValueError("a non-streaming completion request returned an incompatible response")
    fields = _object(raw, "completion response")
    choices = fields.get("choices")
    if not isinstance(choices, list) or len(choices) != expected:
        raise ValueError(f"expected {expected} completion choices")
    ordered: dict[int, tuple[str, str | None]] = {}
    for choice in choices:
        item = _object(choice, "completion choice")
        index, text = item.get("index"), item.get("text")
        if type(index) is not int or index < 0 or index >= expected or index in ordered:
            raise ValueError("choice indices must be a permutation of the expected prompt-choice indices")
        if not isinstance(text, str):
            raise ValueError("each completion choice needs string text")
        ordered[index] = text, _reason(item.get("finish_reason"))
    usage = None
    if fields.get("usage") is not None:
        supplied = _object(fields["usage"], "usage")
        usage = Usage(*(_count(supplied.get(name), name) for name in
                        ("prompt_tokens", "completion_tokens", "total_tokens")))
    per_choice = (usage.completion_tokens,) if expected == 1 and usage is not None else (None,) * expected
    return Completion(tuple(ordered[index][0] for index in range(expected)),
                      tuple(ordered[index][1] for index in range(expected)), per_choice, usage, (response,))


def _openai_fields(model: str, prompts: str | Sequence[str], budget: int, seed: int | None,
                   parameters: Mapping[str, object]) -> tuple[dict[str, object], int]:
    rows = _prompts(prompts, budget, seed)
    n = parameters.get("n", 1)
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer")
    fixed: dict[str, object] = {"model": model, "prompt": prompts if isinstance(prompts, str) else rows,
                                "max_tokens": budget, "stream": False}
    if seed is not None:
        fixed["seed"] = seed
    return _bound(parameters, fixed), len(rows) * n


@dataclass(frozen=True)
class OpenAICompletion:
    """Bind an OpenAI client, including one configured for vLLM's base_url.

    Native SDK request options pass through to completion/chat resources.
    vLLM-only controls such as top_k/min_p/stop_token_ids belong explicitly in
    extra_body. SDK responses and streaming chunks retain backend logprobs,
    token IDs/extensions, tool calls and structured output fields unchanged.
    """

    model: str
    client: OpenAI | AsyncOpenAI
    provider: Literal["openai", "vllm"] = "openai"

    def __post_init__(self) -> None:
        if self.provider not in ("openai", "vllm"):
            raise ValueError("provider must be openai or vllm")

    def _parameters(self, supplied: Mapping[str, object]) -> dict[str, object]:
        fields = dict(supplied)
        sampling = fields.pop("sampling", None)
        if sampling is None:
            return fields
        if not isinstance(sampling, Sampling):
            raise TypeError("sampling must be a Sampling value")
        if sampling.pad_id != 0:
            raise ValueError("remote text completion does not implement padded token rows")
        if self.provider == "openai" and (sampling.top_k is not None or sampling.min_p != 0 or sampling.eos_id is not None):
            raise ValueError("top-k, min-p and EOS-token controls require provider='vllm'")
        native = {"temperature": sampling.temperature, "top_p": sampling.top_p,
                  "frequency_penalty": 0.0, "presence_penalty": 0.0}
        conflicts = [name for name in native.keys() & fields.keys() if fields[name] != native[name]]
        if conflicts:
            raise ValueError(f"parameters conflict with Sampling: {sorted(conflicts)}")
        fields.update(native)
        if self.provider == "vllm":
            extra = {} if fields.get("extra_body") is None else _object(fields["extra_body"], "extra_body")
            controls: dict[str, object] = {"top_k": -1 if sampling.top_k is None else sampling.top_k,
                                          "min_p": sampling.min_p, "repetition_penalty": 1.0}
            if sampling.eos_id is not None:
                controls["stop_token_ids"] = list(sampling.eos_id) if isinstance(sampling.eos_id, tuple) else [sampling.eos_id]
            clashes = [name for name in controls.keys() & extra.keys() if extra[name] != controls[name]]
            if clashes:
                raise ValueError(f"extra_body conflicts with Sampling: {sorted(clashes)}")
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

    def __call__(self, prompts: str | Sequence[str], max_new_tokens: int, *,
                 seed: int | None = None, **parameters: object) -> Completion:
        fields, expected = _openai_fields(self.model, prompts, max_new_tokens, seed, self._parameters(parameters))
        create: Callable[..., LegacyAPIResponse[OpenAIResponse]] = self._sync().completions.with_raw_response.create
        raw = _invoke(create, fields)
        return _openai_result(raw.http_response.json(), raw.parse(), expected)

    def stream(self, prompts: str | Sequence[str], max_new_tokens: int, *, seed: int | None = None,
               **parameters: object) -> Stream[OpenAIResponse]:
        fields, _ = _openai_fields(self.model, prompts, max_new_tokens, seed, self._parameters(parameters))
        fields["stream"] = True
        create: Callable[..., Stream[OpenAIResponse]] = self._sync().completions.create
        return _invoke(create, fields)

    def chat(self, messages: Sequence[object], max_new_tokens: int, *, seed: int | None = None,
             stream: bool = False, **parameters: object) -> ChatCompletion | Stream[ChatCompletionChunk]:
        _prompts("", max_new_tokens, seed)
        fields = _bound(self._parameters(parameters), {"model": self.model, "messages": messages, "max_completion_tokens": max_new_tokens, "stream": stream})
        if seed is not None:
            fields = _bound(fields, {"seed": seed})
        create: Callable[..., ChatCompletion | Stream[ChatCompletionChunk]] = self._sync().chat.completions.create
        return _invoke(create, fields)

    async def acall(self, prompts: str | Sequence[str], max_new_tokens: int, *,
                    seed: int | None = None, **parameters: object) -> Completion:
        fields, expected = _openai_fields(self.model, prompts, max_new_tokens, seed, self._parameters(parameters))
        raw = await _ainvoke(self._async().completions.with_raw_response.create, fields)
        return _openai_result(raw.http_response.json(), raw.parse(), expected)

    async def astream(self, prompts: str | Sequence[str], max_new_tokens: int, *, seed: int | None = None,
                      **parameters: object) -> AsyncStream[OpenAIResponse]:
        fields, _ = _openai_fields(self.model, prompts, max_new_tokens, seed, self._parameters(parameters))
        fields["stream"] = True
        return await _ainvoke(self._async().completions.create, fields)

    async def achat(self, messages: Sequence[object], max_new_tokens: int, *, seed: int | None = None,
                    stream: bool = False, **parameters: object) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
        _prompts("", max_new_tokens, seed)
        fields = _bound(self._parameters(parameters), {"model": self.model, "messages": messages, "max_completion_tokens": max_new_tokens, "stream": stream})
        if seed is not None:
            fields = _bound(fields, {"seed": seed})
        return await _ainvoke(self._async().chat.completions.create, fields)
