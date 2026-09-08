#!/usr/bin/env python3
"""What a host-resident parameter bank costs, and what it lets a card run.

Builds one `causal_transformer` twice over the same synthetic weights, once
with every bank on the device and once with the layer banks in pinned host
memory, and reports for each: the compiled memory plan (device and host
argument, output, alias and temporary sizes), the memory space the entry
computation's parameters were assigned, the resident process memory, and the
prefill and per-token decode times of a real greedy generation.

The resident case is only compiled, never run, when its arguments do not fit:
`--resident-run False` lowers it from shapes alone, which is what shows the
device memory the offload removes.

Usage:
    JAX_PLATFORMS=cpu PYTHONPATH=src python tools/benchmark_host_offload.py \\
        --depths 4 8 16 --width tiny --resident-run True
    JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src \\
        python tools/benchmark_host_offload.py --depths 144 --width candidate \\
        --dtype bfloat16 --bank-layers 16 --new-tokens 16 --resident-run False
"""

from __future__ import annotations

import dataclasses
import gc
import json
import math
import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from dew import models  # naming a registry fills it
from dew.inference.banks import SyntheticBanks, host_banked
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.registry import with_precision
from dew.training import Layout, MeshSpec
from dew.training.distributed import build_mesh

WIDTHS = {
    "tiny": {"vocab_size": 256, "emb_features": 64, "num_heads": 4, "head_dim": 16,
             "mlp_features": 128, "max_seq_len": 64},
    "candidate": {"vocab_size": 4096, "emb_features": 2048, "num_heads": 16,
                  "head_dim": 128, "mlp_features": 8192, "max_seq_len": 512},
}


@dataclass(frozen=True)
class OffloadConfig:
    """Which shapes to measure, and how far to run them."""

    depths: list[int] = field(default_factory=lambda: [4, 8, 16])
    width: str = "tiny"
    dtype: str = "float32"
    attention_impl: str = "xla"
    bank_layers: int | None = None
    batch: int = 1
    prompt: int = 16
    new_tokens: int = 4
    cache_length: int | None = None
    """max_seq_len the cache is allocated for; None keeps the width's."""
    resident_run: bool = True
    """Run the resident case as well as compiling it; False only lowers it."""
    fsdp: int = 1
    out: str | None = None


def status() -> dict[str, int]:
    """This process's resident memory, in bytes, as the kernel reports it."""
    wanted = {"VmRSS": "rss", "RssAnon": "rss_anon", "VmHWM": "rss_peak"}
    values = {}
    with open("/proc/self/status") as handle:
        for line in handle:
            name, _, rest = line.partition(":")
            if name in wanted:
                values[wanted[name]] = int(rest.split()[0]) * 1024
    return values


def plan(compiled) -> dict[str, int]:
    """The compiled memory plan, device and host sides read separately."""
    analysis = compiled.memory_analysis()
    return {name: int(getattr(analysis, f"{name}_in_bytes"))
            for name in ("argument_size", "output_size", "alias_size", "temp_size",
                         "host_argument_size", "host_output_size", "host_temp_size",
                         "peak_memory")}


def entry_spaces(compiled) -> dict[str, int]:
    """How many of the entry computation's parameters XLA put in each memory
    space, read off the compiled module's own layout."""
    head = compiled.as_text().splitlines()[0]
    body = head.partition("entry_computation_layout={(")[2].partition(")->")[0]
    counts: dict[str, int] = {}
    for entry in body.split(", "):
        space = "S(5)" if "S(5)" in entry else "device"
        counts[space] = counts.get(space, 0) + 1
    return counts


def bytes_of(tree, space: str | None = None) -> int:
    """The bytes of the leaves of `tree`, optionally only those in one memory
    space, counting each process's own shards."""
    return sum(
        leaf.nbytes for leaf in jax.tree.leaves(tree)
        if space is None or str(leaf.sharding.memory_kind) == space)


def build(config: OffloadConfig, depth: int) -> CausalTransformer:
    fields = {**WIDTHS[config.width], "num_layers": depth, "scan_layers": True,
              "bank_layers": config.bank_layers, "tie_embeddings": True,
              "mlp": "swiglu"}
    if config.cache_length is not None:
        fields["max_seq_len"] = config.cache_length
    return models.build("causal_transformer", **with_precision(
        "causal_transformer", fields, dtype=config.dtype,
        attention_impl=config.attention_impl))


def shapes_of(model: CausalTransformer, tokens) -> dict:
    """The store a plain init would produce, as shapes: what a source fills."""
    return jax.eval_shape(lambda key: model.init(key, tokens), jax.random.key(0))


def decode_step(model: CausalTransformer):
    def step(store, cache, token, position):
        logits, changed = model.apply(
            {**store, "cache": cache}, token, decode=True, mutable=["cache"],
            positions=position)
        return logits[:, -1], changed["cache"]

    return jax.jit(step)


def prefill_call(model: CausalTransformer):
    def prefill(store, cache, tokens):
        logits, changed = model.apply(
            {**store, "cache": cache}, tokens, decode=True, mutable=["cache"])
        return logits[:, -1], changed["cache"]

    return jax.jit(prefill)


def measure(config: OffloadConfig, depth: int, offload: bool) -> dict:
    """One case: the plan, and the run when the arguments fit."""
    model = build(config, depth)
    mesh = MeshSpec(fsdp=config.fsdp)
    device_mesh = build_mesh(mesh)
    tokens = jnp.zeros((config.batch, config.prompt), jnp.int32)
    layout = Layout(min_shard=1, tolerance=1.0,
                    host_parameters=("params/layers_*",) if offload else ())
    record: dict = {"depth": depth, "offload": offload, "rss_before": status()}

    started = time.perf_counter()
    shapes = shapes_of(model, tokens)
    record["shapes_seconds"] = time.perf_counter() - started
    record["parameters"] = sum(math.prod(leaf.shape) for leaf in jax.tree.leaves(shapes))
    record["parameter_bytes"] = sum(
        math.prod(leaf.shape) * leaf.dtype.itemsize for leaf in jax.tree.leaves(shapes))

    run = offload or config.resident_run
    if not run:
        # Nothing is allocated: the plan comes from the shapes and their
        # placement, which is what shows the device memory the case needs.
        placement = layout.offloaded(device_mesh, shapes)
        store = jax.tree.map(
            lambda leaf, sharding: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype,
                                                        sharding=sharding),
            shapes, placement)
        cache = jax.eval_shape(
            lambda held: model.apply(held, config.batch, method="init_cache",
                                     mutable=["cache"])[1]["cache"], store)
        started = time.perf_counter()
        compiled = prefill_call(model).lower(store, cache, tokens).compile()
        record["prefill_compile_seconds"] = time.perf_counter() - started
        record["prefill_plan"] = plan(compiled)
        record["prefill_spaces"] = entry_spaces(compiled)
        record["cache_bytes"] = sum(
            math.prod(leaf.shape) * leaf.dtype.itemsize for leaf in jax.tree.leaves(cache))
        record["rss_after"] = status()
        return record

    started = time.perf_counter()
    store = host_banked(model, SyntheticBanks(shapes), mesh=mesh, layout=layout)
    jax.block_until_ready(store)
    record["load_seconds"] = time.perf_counter() - started
    record["rss_loaded"] = status()
    record["host_bytes"] = bytes_of(store, "pinned_host")
    record["device_bytes"] = bytes_of(store, "device")
    record["banks"] = sorted(store["params"])

    started = time.perf_counter()
    cache = model.apply(store, config.batch, method="init_cache", mutable=["cache"])[1]["cache"]
    jax.block_until_ready(cache)
    record["init_cache_seconds"] = time.perf_counter() - started
    record["cache_bytes"] = bytes_of(cache)

    prefill = prefill_call(model)
    started = time.perf_counter()
    compiled = prefill.lower(store, cache, tokens).compile()
    record["prefill_compile_seconds"] = time.perf_counter() - started
    record["prefill_plan"] = plan(compiled)
    record["prefill_spaces"] = entry_spaces(compiled)

    started = time.perf_counter()
    logits, cache = jax.block_until_ready(compiled(store, cache, tokens))
    record["prefill_seconds"] = time.perf_counter() - started

    step = decode_step(model)
    token = jnp.argmax(logits, axis=-1)[:, None].astype(jnp.int32)
    position = jnp.full((config.batch, 1), config.prompt, jnp.int32)
    started = time.perf_counter()
    stepped = step.lower(store, cache, token, position).compile()
    record["decode_compile_seconds"] = time.perf_counter() - started
    record["decode_plan"] = plan(stepped)
    record["decode_spaces"] = entry_spaces(stepped)

    produced, latencies = [], []
    for index in range(config.new_tokens):
        at = time.perf_counter()
        logits, cache = jax.block_until_ready(stepped(store, cache, token, position))
        latencies.append(time.perf_counter() - at)
        token = jnp.argmax(logits, axis=-1)[:, None].astype(jnp.int32)
        produced.append(int(token[0, 0]))
        position = position + 1
    record["tokens"] = produced
    record["decode_latencies"] = latencies
    record["decode_median_seconds"] = float(np.median(latencies))
    record["decode_tokens_per_second"] = config.batch / float(np.median(latencies))
    if record["host_bytes"]:
        record["host_to_device_bytes_per_token"] = record["host_bytes"]
        record["effective_bandwidth_bytes_per_second"] = (
            record["host_bytes"] / float(np.median(latencies)))
    record["rss_after"] = status()
    record["allocator"] = {
        name: int(value) for name, value in (jax.devices()[0].memory_stats() or {}).items()
        if name in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit",
                    "largest_alloc_size")}
    del store, cache
    gc.collect()
    return record


def main(config: OffloadConfig) -> None:
    records = []
    for depth in config.depths:
        for offload in (False, True):
            record = measure(config, depth, offload)
            records.append(record)
            print(json.dumps(record, default=float))
    if config.out is not None:
        with open(config.out, "w") as handle:
            json.dump({"config": dataclasses.asdict(config), "records": records}, handle,
                      indent=1, default=float)


if __name__ == "__main__":
    main(tyro.cli(OffloadConfig))
