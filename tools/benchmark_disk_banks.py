#!/usr/bin/env python3
"""Measure one disk-cache budget per process, without loading the resident model.

    python tools/benchmark_disk_banks.py decode --checkpoint DIR [--cache-gib G]
    python tools/benchmark_disk_banks.py experts --checkpoint DIR [--experts 256 --per-step 4]

`decode` runs each cache size in a fresh process for independent peak RSS and
device allocator statistics. `read_seconds` includes the reads, copy, cast
and transpose; it overlaps GPU work and must not be added to end-to-end time.
The optional trace records SSD reads, callback copies and GPU kernels for
timeline attribution. The serial probe reports isolated row-read and H2D
costs, not a subtraction-based estimate of overlapping compute.

`experts` reads randomly chosen experts' stored bytes through the reader the
banks use (`ParallelReader`), directly, a decode step's misses at a time, beside a
sequential read of whole expert tensors from the same files: how near a MoE
decode's expert reads come to the drive's own rate. The two alternate over
`rounds`, each round reading bytes no other round read, so a drive shared
with other work slows both alike, and the ratio is the median round's.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import random
import re
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
from dew.interop.safetensors_io import read_weights
from dew.interop.streaming import ParallelReader
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
    cold: bool = False
    """Drop the checkpoint's files from the page cache before each run, as a host too small to hold
    the model finds them."""
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


def drop_cached(checkpoint: Path) -> None:
    """Give the page cache's copies of the checkpoint's weight files back to the kernel."""
    for path in checkpoint.glob("*.safetensors"):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(descriptor)


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
            if config.cold:
                drop_cached(config.checkpoint)
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


@dataclass(frozen=True)
class ExpertReads:
    checkpoint: Path
    experts: int = 64
    """How many (layer, expert) records each round reads."""
    per_step: int = 4
    """Records read together, as one layer's misses in a decode step."""
    sequential_gib: float = 1.0
    """How much of the whole expert tensors each round reads for the drive's sequential rate."""
    in_flight: int = 2
    """Steps whose reads are in flight at once, as a decode that starts the next layer's reads early."""
    rounds: int = 5
    threads: int = 32
    chunk_mib: int = 2
    seed: int = 0


def expert_records(tensors: dict[str, np.ndarray]) -> dict[tuple[int, int], list[np.ndarray]]:
    """Each (layer, expert)'s stored arrays: its slice of every tensor that stacks the experts on a
    leading axis (GPT-OSS's `experts.gate_up_proj_blocks`), or every tensor named for it alone
    (`experts.7.gate_proj.weight_packed`)."""
    records: dict[tuple[int, int], list[np.ndarray]] = {}
    for name, array in sorted(tensors.items()):
        found = re.search(r"layers\.(\d+)\..*\.experts\.(?:(\d+)\.)?", name)
        if found is None:
            continue
        layer = int(found.group(1))
        if found.group(2) is not None:
            records.setdefault((layer, int(found.group(2))), []).append(array)
        else:
            for expert in range(array.shape[0]):
                records.setdefault((layer, expert), []).append(array[expert])
    return records


def expert_reads(config: ExpertReads):
    tensors = read_weights(config.checkpoint)
    records = expert_records(tensors)
    if not records:
        raise ValueError(f"no expert tensors in {config.checkpoint}")
    keys = sorted(records)
    random.Random(config.seed).shuffle(keys)
    # Whole expert tensors cut into pieces of the round's sequential budget, never one read twice.
    piece = int(config.sequential_gib * 2**30)
    pieces = [array.reshape(-1)[start:start + piece // array.itemsize]
              for name, array in sorted(tensors.items()) if ".experts." in name
              for start in range(0, array.size, piece // array.itemsize)]
    rounds = []
    with ParallelReader(threads=config.threads, chunk=config.chunk_mib << 20, direct=True) as reader:
        for index in range(config.rounds):
            chosen = keys[index * config.experts:(index + 1) * config.experts]
            started = time.perf_counter()
            sequential = sum(array.nbytes for array in reader.load([pieces[-1 - index]]))
            sequential_seconds = time.perf_counter() - started
            started, read, pending = time.perf_counter(), 0, []
            for step in range(0, len(chosen), config.per_step):
                pending.append(reader.start(
                    [array for key in chosen[step:step + config.per_step] for array in records[key]]))
                if len(pending) == config.in_flight:
                    read += sum(array.nbytes for array in pending.pop(0).result())
            read += sum(array.nbytes for loading in pending for array in loading.result())
            seconds = time.perf_counter() - started
            rounds.append({"sequential_gb_per_second": sequential / sequential_seconds / 1e9,
                           "expert_gb_per_second": read / seconds / 1e9,
                           "ms_per_step": 1e3 * seconds / -(-len(chosen) // config.per_step)})
    ratios = sorted(round_["expert_gb_per_second"] / round_["sequential_gb_per_second"] for round_ in rounds)
    print(json.dumps({
        "checkpoint": str(config.checkpoint.resolve()), "config": dataclasses.asdict(config),
        "expert_records": len(records), "record_bytes": sum(array.nbytes for array in records[keys[0]]),
        "rounds": rounds, "median_ratio": ratios[len(ratios) // 2],
    }, default=str))


if __name__ == "__main__":
    chosen = tyro.extras.subcommand_cli_from_dict({"decode": DiskConfig, "experts": ExpertReads})
    expert_reads(chosen) if isinstance(chosen, ExpertReads) else measure(chosen)
