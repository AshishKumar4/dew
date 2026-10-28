"""The seam between an agent harness and the policy: rllm-model-gateway, recording every model call.

A harness reaches the model only through a recording gateway,
`rllm-model-gateway`, at a base URL that names the session in its path,
`/sessions/{session}/v1`. The gateway forwards each call to a vLLM or SGLang
replica, asks it for token ids and log-probabilities, and records them under
that session. The session is part of the URL because a CLI harness may drop
a header. The gateway is a separate service in its own environment; Dew
reads a session's traces over HTTP and imports none of it. It must persist
traces before it answers (`sync_traces: true`), so a session's traces are
complete once its harness has exited.

`calls` turns gateway traces into `Call` records, in submission order. Each
holds the engine's prompt ids, sampled ids and behavior log-probabilities
as recorded, and the version the gateway stamped when the request arrived
(the publication sets that stamp; see `Publication`). The gateway extracts
vLLM's ids and SGLang's prompt ids. SGLang's sampled ids
(`choices[0].response_token_ids`) are read from the raw response the
gateway keeps, and each call's `Transcript` from the chat messages the
harness sent and the message the engine answered, which the gateway records
beside the ids. `engine_failure` and `refused_prompt` read what the engine
itself said of a session, which every harness's verdict weighs the same way
(`dew.interop.harness`).
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dew._records import record as section
from dew.inference.clients import decode_routed_experts
from dew.objectives.rl.records import messages
from dew.objectives.rl.sessions import Call, Transcript

if TYPE_CHECKING:
    import httpx

SESSION = re.compile(r"[A-Za-z0-9._:-]+")
"""What a session id may hold: it goes in a URL's path."""


class Gateway:
    """Connects a session source to an rllm-model-gateway.

    It builds each session's base URL, waits for the gateway to be ready,
    stamps the weight version and reads a session's traces. `url` is the
    gateway's root as Dew reaches it. `sandbox_url` is the address the
    sandboxes reach it at, and a harness's base URL is built from it.

    rllm-model-gateway (3b40c37) has no authentication. One port serves the
    model proxy and also `GET /sessions`, every session's traces, `POST
    /traces/query`, session deletes, `POST /admin/workers` and `POST
    /admin/weight_version`. A sandbox runs the policy's own commands, so a
    policy that reaches that port can read its group members' transcripts,
    reset the version stamp the staleness bound reads, or register a worker
    that returns fabricated ids and likelihoods. To prevent that:

    - Put `url` on an interface only Dew reaches, by binding the gateway to
      loopback or a private trainer network.
    - Make `sandbox_url` a reverse proxy in front of the gateway, on its own
      address, that forwards only `POST /sessions/<session>/v1/chat/completions`
      (nginx `location ~ ^/sessions/[^/]+/v1/chat/completions$`) and answers
      403 to everything else.
    - Let the task's agent reach that address and nothing else. Under
      Harbor, that is `[agent] network_mode = "allowlist"` and
      `allowed_hosts` naming the proxy, or `--allow-agent-host`; this needs
      a provider that supports allowlists (see Harbor's
      `tasks/network-policy` page; Docker needs nftables `fib` support). Set
      `[verifier] network_mode = "no-network"` as well. Harbor's allowlist
      filters by host and not by path, which is why the proxy is needed.

    `sandbox_url` defaults to `url`. Leave it unset only for trusted
    harnesses and tests.
    """

    def __init__(self, url: str, *, sandbox_url: str | None = None, client: httpx.Client | None = None):
        import httpx

        self.url = url.rstrip("/")
        self.sandbox_url = (sandbox_url or url).rstrip("/")
        self._client = client or httpx.Client(timeout=30.0)

    def session(self, session: str) -> str:
        """Return the OpenAI base URL that makes the gateway record a harness's calls under `session`.

        Raises ValueError unless `session` is URL-path safe: letters, digits
        and `._:-`.
        """
        if not SESSION.fullmatch(session):
            raise ValueError(f"session ids are URL-path safe: letters, digits and ._:-; got {session!r}")
        # The session goes in the path because a CLI harness may drop a header (memo section 5.7).
        return f"{self.sandbox_url}/sessions/{session}/v1"

    def traces(self, session: str) -> Sequence[object]:
        response = self._client.get(f"{self.url}/sessions/{session}/traces")
        response.raise_for_status()
        traces = response.json()
        if not isinstance(traces, list):
            raise ValueError(f"the gateway's traces are a JSON list, got {type(traces).__name__}")
        return traces

    def ready(self, timeout: float, *, poll: float = 2.0) -> None:
        """Wait until the gateway routes to a healthy worker that answers, or raise after `timeout` seconds.

        Ready means `/health/workers` and `GET /v1/models`, proxied through the gateway to an engine,
        answer 200. Only their status is read; the bodies are closed unread. The remaining deadline
        bounds each network operation, and a status received after it is never accepted. The check
        runs every `poll` seconds, and expiry raises RuntimeError.

        rllm-model-gateway marks a worker dead after three failed health checks, which happens to an
        engine that is still loading when the gateway starts. Until a later check revives the worker,
        every proxied call gets a plain-text 500 that leaves no trace. A session submitted in that
        window would fail because of the gateway, not the policy.
        """
        import httpx

        url = f"{self.url}/health/workers"
        deadline, seen = time.monotonic() + timeout, "no answer"
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                with self._client.stream("GET", url, timeout=remaining) as response:
                    status = response.status_code
                    seen = f"GET /health/workers: {status}"
                if status == 200:
                    # The gateway calls a worker healthy until its first probe fails, so also see the
                    # engine answer through it; a request without a session leaves no trace.
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    with self._client.stream("GET", f"{self.url}/v1/models", timeout=remaining) as listing:
                        status = listing.status_code
                        seen = f"GET /v1/models: {status}"
                    if status == 200 and time.monotonic() < deadline:
                        return
            except httpx.HTTPError as error:
                seen = f"{type(error).__name__}: {error}"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(poll, remaining))
        raise RuntimeError(
            f"the gateway reported no healthy worker within {timeout:g} s at {url} (last: {seen}); "
            "check that the engines are up and registered, or raise ready_timeout")

    def stamp(self, version: int) -> None:
        """Make the gateway label the calls it records from now on with `version` (a `Publication` stamp)."""
        response = self._client.post(f"{self.url}/admin/weight_version", json={"weight_version": version})
        answer = response.json()
        if (
            response.status_code != 200
            or not isinstance(answer, dict)
            or answer.get("weight_version") != version
        ):
            raise RuntimeError(
                f"the gateway refused version {version}: {response.status_code} {response.text}"
            )

    def forget(self, session: str) -> None:
        self._client.delete(f"{self.url}/sessions/{session}").raise_for_status()


def number(name: str, value: object) -> float:
    """`value`, a JSON number named `name`, as a float; ValueError for anything else, a bool included."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} is not a number: {value!r}")
    return float(value)


def _ids(name: str, values: object) -> tuple[int, ...]:
    if not isinstance(values, list):
        raise ValueError(f"the trace's {name} are not a list of token ids")
    ids = tuple(value for value in values if type(value) is int)
    if len(ids) != len(values):
        raise ValueError(f"the trace's {name} are not a list of token ids")
    return ids


@dataclass(frozen=True)
class Recorded:
    """One session as the gateway recorded it.

    `calls` are its model calls in submission order. `errors` are the
    messages of the error replies the engine sent instead of a completion.
    """

    calls: tuple[Call, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True)
class _Trace:
    """One gateway trace, read: when its request arrived, and either the call or the engine's error."""

    arrival: float
    call: Call | None
    error: str


def _trace(trace: object, unstamped: int) -> _Trace:
    """Read one gateway trace, refusing any field that is not the JSON type the gateway writes."""
    if not isinstance(trace, dict):
        raise ValueError(f"a gateway trace is a JSON object, got {type(trace).__name__}")
    raw = trace.get("raw_response") or {}
    if not isinstance(raw, dict):
        raise ValueError(f"a trace's raw response is a JSON object, got {type(raw).__name__}")
    arrival = (
        number("timestamp", trace.get("timestamp")) - number("latency_ms", trace.get("latency_ms")) / 1000
    )
    # vLLM answers {"error": {"message": ...}}; SGLang {"object": "error", "message": ...}.
    error = raw.get("error") or (raw.get("message") if raw.get("object") == "error" else None)
    if error:
        return _Trace(arrival, None, str(error.get("message", error) if isinstance(error, dict) else error))
    sampled = trace.get("completion_token_ids") or []
    if not sampled:
        # SGLang lists the sampled ids as choices[0].response_token_ids, which the gateway does not extract.
        choices = raw.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else None
        sampled = (choice.get("response_token_ids") if isinstance(choice, dict) else None) or []
    prompt = trace.get("prompt_token_ids") or []
    if not prompt:
        raise ValueError(
            "a trace carries no prompt ids: the engine was not asked for them or cannot list them"
        )
    version = trace.get("weight_version")
    if version is not None and type(version) is not int:
        raise ValueError(f"a trace's weight version is an integer, got {version!r}")
    choices = raw.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    routed = choice.get("routed_experts")
    likelihoods = trace.get("logprobs") or []
    if not isinstance(likelihoods, list):
        raise ValueError("a trace's log-probabilities are a list")
    call = Call(_ids("prompt ids", prompt), _ids("sampled ids", sampled),
                tuple(number("a log-probability", value) for value in likelihoods),
                str(trace.get("finish_reason")), unstamped if version is None else version,
                routed_experts=None if routed is None else decode_routed_experts(routed),
                transcript=_transcript(trace))
    return _Trace(arrival, call, "")


def _transcript(trace: Mapping[str, object]) -> Transcript | None:
    """The chat messages the harness sent and the message the engine answered, as the gateway kept
    them, or None for a trace that kept no messages."""
    asked, answered = trace.get("messages"), trace.get("response_message")
    listed = () if asked is None else messages(asked, "messages")
    # rllm-model-gateway writes an empty object for a response it holds no message of.
    reply = None if answered is None else section(answered, "response_message") or None
    if not listed and reply is None:
        return None
    return Transcript(listed, reply)


def calls(traces: Sequence[object], *, unstamped: int) -> Recorded:
    """The gateway's traces of one session as `Call`s in submission order, and its engine errors.

    A trace records its arrival indirectly: `timestamp` is when the answer
    was stored and `latency_ms` how long the engine took, so the calls are
    ordered by their difference. A trace without a version stamp, from a
    gateway no publication has stamped, takes `unstamped`, the version the
    session was submitted under: no later push can have served it anything
    older. The gateway also records an engine's error reply (a prompt past
    the context length, an engine fault): such a trace carries `error` in its
    raw response and is an event of the session, returned as its message, not
    a call. A successful reply without ids, with one likelihood too few or
    too many, or with a field of the wrong JSON type raises `ValueError`;
    training on it would mean guessing or re-tokenizing text.
    """
    read = sorted((_trace(trace, unstamped) for trace in traces), key=lambda trace: trace.arrival)
    return Recorded(tuple(trace.call for trace in read if trace.call is not None),
                    tuple(trace.error for trace in read if trace.call is None))


# How vLLM 0.30.0 ("This model's maximum context length is ...") and SGLang 0.5.20 ("The input (N
# tokens) is longer than the model's context length (M tokens).") word a prompt past the context length;
# vLLM's input processor, which token-id prompts reach, says
# "... is longer than the maximum model length of M".
_OVERFLOW = re.compile(r"maximum context length|longer than the model's context length"
                       r"|longer than the maximum model length", re.IGNORECASE)


def engine_failure(recorded: Recorded) -> str | None:
    """Why the engine failed the session, or None: an error reply other than a prompt past its
    context length, or an aborted call. Either makes the session infra, whatever the harness did
    after it, since the policy did not cause it."""
    broken = [error for error in recorded.errors if not _OVERFLOW.search(error)]
    if broken:
        return f"the engine answered an error: {broken[0]}"
    if any(call.finish_reason == "abort" for call in recorded.calls):
        return "the engine aborted a call"
    return None


def refused_prompt(recorded: Recorded) -> str | None:
    """Why the engine cut the session short, or None: a prompt it refused as past its context
    length. The policy's own context ran out, so the session is truncated."""
    return f"the engine refused a prompt: {recorded.errors[0]}" if recorded.errors else None


__all__ = ["SESSION", "Gateway", "Recorded", "calls", "engine_failure", "number", "refused_prompt"]
