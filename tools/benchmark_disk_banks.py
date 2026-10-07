#!/usr/bin/env python3
"""Measure one disk-cache budget per process, without loading the resident model.

Run each cache size in a fresh process for independent peak RSS and device
allocator statistics. `read_seconds` includes mmap faults, copy, cast and
transpose; it overlaps GPU work and must not be added to end-to-end time.
The optional trace records SSD reads, callback copies and GPU kernels for
timeline attribution. The serial probe reports isolated row-read and H2D
costs, not a subtraction-based estimate of overlapping compute.
"""

from __future__ import annotations

import dataclasses
import json
import math
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tyro

import dew.nn.backbones  # noqa: F401  (registers the kind)
from dew.inference.banks import SafetensorsBanks
from dew.interop.hf_decoders import translate_config
from dew.objectives.base import merge
from dew.registry import models
from dew.training import Layout, MeshSpec


@dataclass(frozen=True)
class DiskConfig:
    checkpoint: Path
    cache_gib: float = 0.0
    dtype: str = "bfloat16"
    param_dtype: str = "auto"
    prompt: int = 8
    tokens: int = 8
    repeats: int = 2
    bank_layers: int | None = None
    probe_layers: int = 2
    read_ahead: bool = True
    warmup: bool = True
    trace: str | None = None


def resident_status():
    with open("/proc/self/status") as handle:
        wanted = {"VmRSS", "RssAnon", "RssFile", "VmHWM"}
        return {name.rstrip(":"): int(rest.split()[0]) * 1024
                for line in handle if (fields := line.split(None, 1))
                and (name := fields[0].rstrip(":")) in wanted
                for rest in fields[1:]}


def storage_reads():
    with open("/proc/self/io") as handle:
        return {name.rstrip(":"): int(count) for line in handle
                for name, count in [line.split()] if name in ("read_bytes:", "rchar:")}


def serial_probe(source, layers, device):
    read_seconds = transfer_seconds = 0.0
    read_bytes = 0
    # Drain the generation's cross-token read-ahead outside the timed
    # probe. Otherwise layer zero can already be partly/fully read when
    # its isolated read timer starts.
    source.read(0)
    for index in range(layers):
        started = time.perf_counter()
        row = source.read(index)
        read_seconds += time.perf_counter() - started
        read_bytes += sum(leaf.nbytes for leaf in jax.tree.leaves(row))
        started = time.perf_counter()
        staged = jax.block_until_ready(jax.device_put(row, device))
        transfer_seconds += time.perf_counter() - started
        for leaf in jax.tree.leaves(staged):
            leaf.delete()
        del staged, row
    return {"layers": layers, "bytes": read_bytes, "read_seconds": read_seconds,
            "h2d_seconds": transfer_seconds}


def measure(config: DiskConfig):
    if min(config.prompt, config.tokens, config.repeats) < 1 or config.probe_layers < 0:
        raise ValueError("prompt, tokens and repeats must be positive; probe_layers must be nonnegative")
    device = jax.devices()[0]
    with SafetensorsBanks(config.checkpoint, cache_bytes=int(config.cache_gib * 2**30),
                          param_dtype=config.param_dtype, read_ahead=config.read_ahead) as source:
        record = translate_config(source.config)
        record["max_seq_len"] = config.prompt + config.tokens + 1
        model = models.build("causal_transformer", {**record, "dtype": config.dtype, "attention_impl": "xla",
                                                   "scan_layers": True, "bank_layers": config.bank_layers})
        variables = source.stream(model, mesh=MeshSpec().build([device]),
                                   layout=Layout(min_shard=1, tolerance=1.0))
        print("non-decoder weights loaded", resident_status(), file=sys.stderr, flush=True)
        weight_bytes = sum(math.prod(leaf.shape) * np.dtype(leaf.dtype).itemsize
                           for leaf in jax.tree.leaves(source.shapes()))
        cache = model.apply(variables, 1, method=model.init_cache, mutable=["cache"])[1]
        prompt = jax.device_put(np.arange(config.prompt, dtype=np.int32)[None] % model.vocab_size, device)
        prefill = jax.jit(lambda read, held, tokens: model.apply(
            merge(read, held), tokens, decode=True, mutable=["cache"]))
        decode = jax.jit(lambda read, held, token: model.apply(
            merge(read, held), token, decode=True, mutable=["cache"]))

        def run():
            started = time.perf_counter()
            logits, held = jax.block_until_ready(prefill(variables, cache, prompt))
            prefill_seconds = time.perf_counter() - started
            print("prefill seconds", prefill_seconds, "reads", source.stats(), file=sys.stderr, flush=True)
            started = time.perf_counter()
            emitted = []
            for _ in range(config.tokens):
                token = jnp.argmax(logits[:, -1:], axis=-1).astype(jnp.int32)
                emitted.append(token)
                logits, held = decode(variables, held, token)
            logits, held, emitted = jax.block_until_ready((logits, held, emitted))
            print("decode seconds", time.perf_counter() - started, "reads", source.stats(),
                  file=sys.stderr, flush=True)
            return prefill_seconds, time.perf_counter() - started, np.asarray(jnp.concatenate(emitted, axis=1))

        if config.warmup:
            run()  # Also warms filesystem and retained rows.
        else:
            # Compile without an extra SSD sweep. Cache initialization has
            # already admitted retained rows; an unretained model larger
            # than RAM cannot stay warm in the filesystem cache anyway.
            prefill = prefill.lower(variables, cache, prompt).compile()
            decode = decode.lower(variables, cache, jnp.zeros((1, 1), jnp.int32)).compile()
        before = source.stats()
        before_io = storage_reads()
        if config.trace:
            jax.profiler.start_trace(config.trace)
        try:
            runs = [run() for _ in range(config.repeats)]
        finally:
            if config.trace:
                jax.profiler.stop_trace()
        after = source.stats()
        after_io = storage_reads()
        decode_seconds = sum(run[1] for run in runs)
        report = {"device": device.device_kind, "platform": device.platform,
                  "checkpoint": str(config.checkpoint.resolve()), "config": dataclasses.asdict(config),
                  "weight_bytes": weight_bytes, "cache_budget_bytes": source.cache_limit,
                  "layer_bytes": [source.layer_bytes(index) for index in range(model.num_layers)],
                  "prefill_seconds": [run[0] for run in runs], "decode_seconds": [run[1] for run in runs],
                  "tokens_per_second": config.tokens * config.repeats / decode_seconds,
                  "cache": dataclasses.asdict(after),
                  "measured_reads": {name: getattr(after, name) - getattr(before, name)
                                     for name in ("hits", "misses", "bytes_read", "read_seconds", "wait_seconds")},
                  "storage_reads": {name: after_io[name] - before_io[name] for name in before_io},
                  "tokens": [run[2].tolist() for run in runs], "host": resident_status(),
                  "device_memory": device.memory_stats(),
                  "host_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
        # Probe after recording peaks: these are isolated costs, not generation's memory accounting.
        if config.probe_layers:
            report["serial_probe"] = serial_probe(source, min(config.probe_layers, model.num_layers), device)
        print(json.dumps(report, default=str))


if __name__ == "__main__":
    measure(tyro.cli(DiskConfig))
