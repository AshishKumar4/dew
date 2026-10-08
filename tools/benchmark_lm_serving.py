"""Matched warm greedy workload for Dew's Server and vLLM.

Every backend receives the same seeded token ids, prompt/output lengths and
maximum number of outstanding requests. EOS is ignored. Output throughput
counts generated tokens over the wall time from submission to completion;
TTFT is submission to the first observed token and ITL is first-to-last time
over the remaining tokens. Client queue time is separate. Dew and vllm-engine
are called in-process; vllm includes its HTTP/streaming transport.

`--rate` adds open-loop runs (Dew and vllm-engine): `--requests` (default
six times the slots) arriving as a Poisson process at each rate, seeded per
slot count, so a request waits in the server's queue rather than the
client's. TTFT is then from the request's arrival, and the token gaps are
per token: each decoding row's time between consecutive tokens.

Start vLLM in its own environment with bf16, --generation-config vllm and
--gpu-memory-utilization 0.90. Pass the same snapshot and --vocab-limit to
all runs. Warmup uses disjoint prompts and the full measured output length.
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import dataclasses
import hashlib
import io
import json
import platform
import pstats
import re
import time
import uuid
from pathlib import Path

import numpy as np


def prompts_for(slots: int, count: int, length: int, high: int, *, warm: bool = False) -> np.ndarray:
    seed = (2000 if warm else 1000) + slots
    return np.random.default_rng(seed).integers(100, high, (count, length), dtype=np.int32)


def percentiles(values) -> dict[str, float]:
    array = np.asarray(values, float)
    return {"mean": float(array.mean()), "p50": float(np.percentile(array, 50)),
            "p90": float(np.percentile(array, 90)), "p99": float(np.percentile(array, 99))}


def metrics(rows: list[dict[str, float]], wall: float, prompt: int) -> dict[str, object]:
    generated = sum(row["tokens"] for row in rows)
    return {"requests": len(rows), "wall_seconds": wall, "output_tokens": generated,
            "output_tokens_per_second": generated / wall,
            "total_tokens_per_second": (generated + len(rows) * prompt) / wall,
            **{f"{name}_seconds": percentiles([row[name] for row in rows])
               for name in ("ttft", "itl", "latency", "client_queue")}}


def arrivals_for(slots: int, count: int, rate: float) -> np.ndarray:
    """Seconds from the start at which each of `count` requests arrives, Poisson at `rate` a second."""
    return np.cumsum(np.random.default_rng(3000 + slots).exponential(1 / rate, count))


def open_metrics(rate: float, ttft, gaps, wall: float, tokens: int) -> dict[str, object]:
    return {"rate": rate, "requests": len(ttft), "wall_seconds": wall, "output_tokens": tokens,
            "output_tokens_per_second": tokens / wall, "ttft_seconds": percentiles(ttft),
            "token_gap_seconds": percentiles(gaps)}


def dew_open(server, prompts: np.ndarray, output: int, rate: float):
    """Open-loop Dew: requests submitted as they arrive, a step whenever any is live."""
    arrivals = arrivals_for(server.slots, len(prompts), rate)
    tickets, due, gaps = [], [], []
    stamps: dict[int, float] = {}  # a decoding row's last step
    live: list[int] = []
    began = time.perf_counter()
    while len(tickets) < len(prompts) or live:
        now = time.perf_counter() - began
        while len(tickets) < len(prompts) and arrivals[len(tickets)] <= now:
            due.append(began + arrivals[len(tickets)])
            live.append(len(tickets))
            tickets.append(server.submit(prompts[len(tickets)], output, key=len(tickets)))
        if not live:
            time.sleep(max(0.0, arrivals[len(tickets)] - (time.perf_counter() - began)))
            continue
        server.step()
        stamp = time.perf_counter()
        for number in live:
            if tickets[number].first is not None:
                if number in stamps:
                    gaps.append(stamp - stamps[number])
                stamps[number] = stamp
        live = [number for number in live if not tickets[number].done()]
    wall = time.perf_counter() - began
    server.run()
    ttft = [ticket.first - arrival for ticket, arrival in zip(tickets, due, strict=True)]
    return open_metrics(rate, ttft, gaps, wall, len(prompts) * output)


def dew_run(server, prompts: np.ndarray, output: int):
    began = time.perf_counter()
    tickets, sent, active = [], [], []
    index, steps = 0, server.steps
    while index < len(prompts) or active:
        while index < len(prompts) and len(active) < server.slots:
            sent.append(time.perf_counter())
            # An integer seed, as a client sends one: the server makes the key.
            ticket = server.submit(prompts[index], output, key=index)
            tickets.append(ticket)
            active.append(ticket)
            index += 1
        server.step()
        active = [ticket for ticket in active if not ticket.done()]
    wall = time.perf_counter() - began
    server.run()
    results = [ticket.result().host() for ticket in tickets]
    rows = []
    for ticket, start, result in zip(tickets, sent, results, strict=True):
        count = int(result.lengths[0])
        assert count == output, (count, output)
        rows.append({"tokens": count, "ttft": ticket.first - start,
                     "itl": (ticket.finished - ticket.first) / max(count - 1, 1),
                     "latency": ticket.finished - start, "client_queue": start - began})
    return metrics(rows, wall, prompts.shape[1]) | {"steps": server.steps - steps}, results


def profile_decode(server, prompts: np.ndarray, output: int, steps: int, directory: Path):
    """Untraced Python costs and a warmed decode-only Dew/XProf capture."""
    import jax
    from trace_window import device_events, length, union

    import dew

    if -(-len(prompts) // server.admission) + 2 + 2 * steps * server.decode_steps >= output:
        raise ValueError("--output must cover admission, warmup and both decode profiles")
    tickets = [server.submit(prompt, output, key=index) for index, prompt in enumerate(prompts)]
    while server.queued or any(row.prefilled < len(row.prompt) for row in server._rows.values()):
        server.step()
    for _ in range(2):
        server.step()
    server._settle()
    profile = cProfile.Profile()
    began = time.perf_counter()
    profile.enable()
    for _ in range(steps):
        server.step()
    server._settle()
    profile.disable()
    host_wall = time.perf_counter() - began
    stream = io.StringIO()
    pstats.Stats(profile, stream=stream).sort_stats("tottime").print_stats(30)
    previous = set(directory.rglob("*.xplane.pb"))
    with dew.Profiler(directory) as capture:
        for _ in range(steps):
            with capture.region("serve.decode"):
                server.step()
        server._settle()
    jax.block_until_ready(server.cache)
    server.run()
    assert all(ticket.done() for ticket in tickets)
    devices = {}
    trace, = set(directory.rglob("*.xplane.pb")) - previous
    for plane, events in device_events(trace.parent)[0].items():
        if not plane.startswith("/device:GPU") or not events:
            continue
        durations, counts = {}, {}
        for event in events:
            durations[event.name] = durations.get(event.name, 0) + event.duration_ns
            counts[event.name] = counts.get(event.name, 0) + 1
        spans = union([(event.start_ns, event.start_ns + event.duration_ns) for event in events])
        busy, window = length(spans), spans[-1][1] - spans[0][0]
        patterns = {"attention": r"fmha|flash|cudnn|attention", "gemm": r"gemm|cutlass|cublas|xmma|matmul|_dot",
                    "movement": r"memcpy|copy|scatter|gather|transpose|concatenate",
                    "reduction": r"reduce|argmax|topk|sort", "random": r"random|threefry|philox"}
        classes = dict.fromkeys(patterns, 0) | {"other": 0}
        for name, duration in durations.items():
            kind = next((kind for kind, pattern in patterns.items() if re.search(pattern, name, re.I)), "other")
            classes[kind] += duration
        devices[plane] = {"window_ms": window / 1e6, "busy_ms": busy / 1e6,
                          "busy_fraction": busy / window, "kernel_count": sum(counts.values()),
                          "kernel_ms": {kind: duration / 1e6 for kind, duration in classes.items()},
                          "top": [{"name": name, "ms": duration / 1e6, "count": counts[name]}
                                  for name, duration in sorted(durations.items(), key=lambda item: -item[1])[:30]]}
    return {"directory": str(directory), "steps": steps,
            "host_wall_seconds": host_wall, "host_profile": stream.getvalue(), "devices": devices}


def dew_points(args):
    import jax

    import dew
    from dew.inference.serving import Server
    from dew.nn.kv_cache import KVCache
    from dew.sampling import Sampling

    began = time.perf_counter()
    task = dew.pipeline(args.model, dtype="bfloat16", param_dtype="bfloat16")
    task = dataclasses.replace(task, sampling=Sampling(temperature=0, eos_token_ids=None),
                               logits=None, stopping=())
    if args.vocab_limit > task.model.vocab_size:
        raise ValueError("--vocab-limit exceeds the model vocabulary")
    capacity = args.prompt + args.output
    load_seconds = time.perf_counter() - began
    print(f"loaded in {load_seconds:.2f}s", flush=True)
    hardware = {"devices": [device.device_kind for device in jax.devices()], "jax": jax.__version__,
                "load_seconds": load_seconds}
    points = []
    for slots in args.slots:
        count = args.requests or max(64, 2 * slots)
        prompts = prompts_for(slots, count, args.prompt, args.vocab_limit)
        warm = prompts_for(slots, slots, args.prompt, args.vocab_limit, warm=True)
        began = time.perf_counter()
        server = Server.from_task(task, slots=slots, capacity=capacity,
                                  admission=args.admission or None, decode_steps=args.decode_steps,
                                  kv_cache=KVCache(page_size=16 if args.kv == "paged" else None))
        print(f"slots {slots}: built in {time.perf_counter() - began:.2f}s", flush=True)
        dew_run(server, warm, args.output)
        # Each power-of-two admission width is its own program (`admission_share`).
        for rows in sorted({min(server.admission, 1 << bit) for bit in range(server.admission.bit_length())}):
            dew_run(server, warm[:rows], args.output)
        print(f"slots {slots}: warm in {time.perf_counter() - began:.2f}s", flush=True)
        compiled = server._step._cache_size()
        repeats = []
        for repeat in range(args.repeats):
            measured, generations = dew_run(server, prompts, args.output)
            repeats.append(measured)
            if args.generations and repeat == 0:
                destination = args.out.with_name(f"{args.out.stem}-slots{slots}.npz")
                np.savez(destination, tokens=np.concatenate([row.tokens[:, -args.output:] for row in generations]),
                         raw=np.concatenate([row.raw_log_probs for row in generations]),
                         behavior=np.concatenate([row.behavior_log_probs for row in generations]))
        point = {"slots": slots, "admission": server.admission, "repeats": repeats,
                 "prompt_sha256": hashlib.sha256(prompts.tobytes()).hexdigest(),
                 "measured_recompiles": server._step._cache_size() - compiled}
        if args.rate:
            opened = prompts_for(slots, args.requests or 6 * slots, args.prompt, args.vocab_limit)
            point["open_loop"] = [dew_open(server, opened, args.output, rate) for rate in args.rate]
        if args.profile:
            point["profile"] = profile_decode(server, prompts[:slots], args.output, args.profile_steps,
                                              args.out.with_name(f"{args.out.stem}-slots{slots}-profile"))
        points.append(point)
        print(json.dumps(point), flush=True)
    return hardware, points


async def async_sweep(args, request, streamed=None):
    points = []
    for slots in args.slots:
        count = args.requests or max(64, 2 * slots)
        prompts = prompts_for(slots, count, args.prompt, args.vocab_limit)
        warm = prompts_for(slots, slots, args.prompt, args.vocab_limit, warm=True)
        gate = asyncio.Semaphore(slots)
        await asyncio.gather(*(request(row, args.output, gate, time.perf_counter()) for row in warm))
        repeats = []
        for _ in range(args.repeats):
            began = time.perf_counter()
            rows = await asyncio.gather(*(request(row, args.output, gate, began) for row in prompts))
            repeats.append(metrics(rows, time.perf_counter() - began, args.prompt))
        point = {"slots": slots, "repeats": repeats,
                 "prompt_sha256": hashlib.sha256(prompts.tobytes()).hexdigest()}
        if args.rate:
            opened = prompts_for(slots, args.requests or 6 * slots, args.prompt, args.vocab_limit)
            point["open_loop"] = [await async_open(slots, opened, args.output, rate, streamed)
                                  for rate in args.rate]
        points.append(point)
        print(json.dumps(point), flush=True)
    return points


async def async_open(slots, prompts, output, rate, streamed):
    """Open-loop requests to an engine whose own scheduler queues them."""
    arrivals = arrivals_for(slots, len(prompts), rate)
    began = time.perf_counter()

    async def arrive(index):
        await asyncio.sleep(max(0.0, began + arrivals[index] - time.perf_counter()))
        stamps = await streamed(prompts[index], output)
        assert len(stamps) == output, (len(stamps), output)
        return stamps[0] - (began + arrivals[index]), np.diff(stamps)

    results = await asyncio.gather(*(arrive(index) for index in range(len(prompts))))
    wall = time.perf_counter() - began
    return open_metrics(rate, [ttft for ttft, _ in results], np.concatenate([gaps for _, gaps in results]),
                        wall, len(prompts) * output)


async def vllm_points(args):
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=args.url, api_key="none", timeout=600, max_retries=0)

    async def request(prompt, output, gate, queued):
        async with gate:
            sent = time.perf_counter()
            stream = await client.completions.create(
                model=args.model, prompt=prompt.tolist(), max_tokens=output, temperature=0, top_p=1,
                stream=True, stream_options={"include_usage": True},
                extra_body={"ignore_eos": True, "top_k": -1, "min_p": 0, "repetition_penalty": 1})
            first = last = None
            tokens = 0
            async for chunk in stream:
                if chunk.choices:
                    last = time.perf_counter()
                    first = last if first is None else first
                if chunk.usage is not None:
                    tokens = chunk.usage.completion_tokens
            assert first is not None and last is not None and tokens == output, (tokens, output)
            return {"tokens": tokens, "ttft": first - sent, "itl": (last - first) / max(tokens - 1, 1),
                    "latency": last - sent, "client_queue": sent - queued}

    try:
        points = await async_sweep(args, request)
    finally:
        await client.close()
    return {"label": args.hardware}, points


async def engine_points(args):
    """The same scheduler/stream clocks, without HTTP serialization."""
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.inputs import TokensPrompt
    from vllm.v1.engine.async_llm import AsyncLLM

    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
        model=args.model, dtype="bfloat16", max_model_len=args.prompt + args.output,
        max_num_seqs=max(args.slots), gpu_memory_utilization=0.90,
        generation_config="vllm", enable_prefix_caching=False, disable_log_stats=True))

    async def request(prompt, output, gate, queued):
        async with gate:
            sent = time.perf_counter()
            policy = SamplingParams(temperature=0, top_p=1, top_k=-1, min_p=0,
                                    repetition_penalty=1, max_tokens=output, ignore_eos=True)
            stream = engine.generate(TokensPrompt(prompt_token_ids=prompt.tolist()), policy, uuid.uuid4().hex)
            first = last = None
            tokens = 0
            async for result in stream:
                count = len(result.outputs[0].token_ids)
                if count > tokens:
                    last = time.perf_counter()
                    first = last if first is None else first
                    tokens = count
            assert first is not None and last is not None and tokens == output, (tokens, output)
            return {"tokens": tokens, "ttft": first - sent, "itl": (last - first) / max(tokens - 1, 1),
                    "latency": last - sent, "client_queue": sent - queued}

    async def streamed(prompt, output):
        """The time each of a request's tokens reached the client."""
        policy = SamplingParams(temperature=0, top_p=1, top_k=-1, min_p=0,
                                repetition_penalty=1, max_tokens=output, ignore_eos=True)
        stamps: list[float] = []
        async for result in engine.generate(TokensPrompt(prompt_token_ids=prompt.tolist()), policy,
                                            uuid.uuid4().hex):
            arrived = len(result.outputs[0].token_ids) - len(stamps)
            stamps.extend([time.perf_counter()] * arrived)
        return stamps

    try:
        points = await async_sweep(args, request, streamed)
    finally:
        engine.shutdown()
    return {"label": args.hardware}, points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("dew", "vllm", "vllm-engine"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8100/v1")
    parser.add_argument("--hardware", default="")
    parser.add_argument("--slots", default="32,64,128")
    parser.add_argument("--requests", type=int, default=0)
    parser.add_argument("--prompt", type=int, default=256)
    parser.add_argument("--output", type=int, default=128)
    parser.add_argument("--vocab-limit", type=int, default=150_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--admission", type=int, default=0)
    parser.add_argument("--decode-steps", type=int, default=1)
    parser.add_argument("--kv", choices=("dense", "paged"), default="dense")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-steps", type=int, default=20)
    parser.add_argument("--generations", action="store_true")
    parser.add_argument("--rate", default="", help="open-loop Poisson arrival rates, requests a second")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.slots = [int(value) for value in args.slots.split(",")]
    args.rate = [float(value) for value in args.rate.split(",") if value]
    if args.rate and args.backend == "vllm":
        parser.error("--rate runs Dew and vllm-engine in-process")
    if (min(args.slots) < 1 or min(args.prompt, args.output, args.repeats, args.profile_steps) < 1
            or args.requests < 0 or args.vocab_limit <= 100):
        parser.error("counts must be positive; --requests=0 chooses the default")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    run = engine_points if args.backend == "vllm-engine" else vllm_points
    hardware, points = dew_points(args) if args.backend == "dew" else asyncio.run(run(args))
    record = {"backend": args.backend, "model": args.model, "hardware": hardware,
              "prompt": args.prompt, "output": args.output, "vocab_limit": args.vocab_limit,
              "sampling": {"temperature": 0, "ignore_eos": True}, "warm_output": args.output,
              "transport": "HTTP streaming" if args.backend == "vllm" else "in-process",
              "decode_steps": args.decode_steps, "kv": args.kv, "host": platform.processor(), "sweep": points}
    args.out.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
