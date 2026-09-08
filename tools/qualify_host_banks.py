#!/usr/bin/env python3
"""One small decoder on a real device, resident and with its banks on the host.

Runs the same `CausalTransformer` and the same `dew.inference.host_banked`
paths a caller uses, in one process per residency so each process's peaks are
its own, and writes what it saw as JSON: the compiled memory plan with the
host and device sides separate, the memory space the compiled entry gave each
parameter, the bytes this process owns, the allocator's own counters, the
control group's, and the compile, prefill and decode times. The logits and the
greedy continuation go in the file too, so two runs are compared exactly
rather than described.

    systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 -- \\
        env JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src \\
        python tools/qualify_host_banks.py --kind dense --offload --out r.json
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from dew import models  # naming a registry fills it
from dew.inference.banks import HeldBanks, host_banked
from dew.registry import with_precision
from dew.sampling.text import Sampling, generate
from dew.training import Layout

# About 200 MiB of bf16 weights: small enough for a four-gigabyte cap to hold
# the source and the store at once, deep enough to bank four times.
SHAPE = dict(vocab_size=1024, emb_features=512, num_layers=24, num_heads=8,
             head_dim=64, mlp_features=2048, max_seq_len=64, tie_embeddings=True,
             scan_layers=True, bank_layers=6)
KINDS = {
    "dense": {},
    "gated_delta_net": dict(layer_types=("linear_attention",) * SHAPE["num_layers"],
                            kinds={"linear_attention": {"mixer": {"kind": "gated_delta_net"}}}),
    "latent_attention": dict(mixer={"kind": "mla", "kv_lora_rank": 64, "q_lora_rank": 64,
                                    "qk_rope_head_dim": 16, "qk_nope_head_dim": 16,
                                    "v_head_dim": 32}),
}


CAP_BYTES = 4 * 1024 ** 3
HEADROOM_BYTES = 12 * 1024 ** 3


def preflight(store_mib: int | None = None, bank_counts=None) -> dict:
    """Refuse to start unless a live budget already bounds this process tree.

    Read before the backend is opened, because a check that runs after the
    first allocation is not a check. Reported limits are not taken as a
    guard: the group has to be an active cgroup v2 memory controller with a
    finite `memory.max` no larger than the agreed cap and swap turned off, and
    the machine has to have the agreed headroom left. Anything missing,
    unlimited or unreadable exits nonzero with the reason, since running
    without enforcement is what the cap exists to prevent.
    """
    line = Path("/proc/self/cgroup").read_text().strip().splitlines()[-1]
    where = Path("/sys/fs/cgroup") / line.split(":")[-1].lstrip("/")
    if not (where / "memory.current").is_file():
        raise SystemExit(
            f"refusing to run: {where} is not an active cgroup v2 memory controller, "
            f"so nothing bounds this process tree")
    for name in ("memory.max", "memory.swap.max"):
        if not (where / name).is_file():
            raise SystemExit(f"refusing to run: {where}/{name} is not readable")
    limit = (where / "memory.max").read_text().strip()
    if limit == "max":
        raise SystemExit(
            f"refusing to run: {where}/memory.max is unlimited; start inside a scope "
            f"with MemoryMax set, at most {CAP_BYTES} bytes")
    if int(limit) > CAP_BYTES:
        raise SystemExit(
            f"refusing to run: {where}/memory.max is {limit}, over the agreed "
            f"{CAP_BYTES} byte cap")
    swap = (where / "memory.swap.max").read_text().strip()
    if swap != "0":
        raise SystemExit(
            f"refusing to run: {where}/memory.swap.max is {swap}, not 0, so the cap "
            f"can be paid for in swap")
    available = 0
    for entry in Path("/proc/meminfo").read_text().splitlines():
        if entry.startswith("MemAvailable:"):
            available = int(entry.split()[1]) * 1024
    if available < HEADROOM_BYTES:
        raise SystemExit(
            f"refusing to run: MemAvailable is {available} bytes, under the "
            f"{HEADROOM_BYTES} byte headroom this is allowed to leave")
    if store_mib is not None and not 0 < store_mib <= 512:
        raise SystemExit(
            f"refusing to run: store_mib is {store_mib}, outside the permitted "
            f"1 to 512 MiB")
    if bank_counts is not None and any(count < 1 for count in bank_counts):
        raise SystemExit(f"refusing to run: bank_counts {list(bank_counts)} is not positive")
    return {"cgroup": str(where), "memory.max": limit, "memory.swap.max": swap,
            "MemAvailable": available}


@dataclass(frozen=True)
class Case:
    kind: str = "dense"
    offload: bool = False
    dtype: str = "bfloat16"
    batch: int = 1
    prompt: int = 16
    new_tokens: int = 8
    seed: int = 0
    out: str | None = None


def cgroup_stats() -> dict:
    line = Path("/proc/self/cgroup").read_text().strip().splitlines()[-1]
    where = Path("/sys/fs/cgroup") / line.split(":")[-1].lstrip("/")
    values = {}
    for name in ("memory.max", "memory.swap.max", "memory.current", "memory.peak",
                 "memory.swap.current"):
        if (where / name).is_file():
            values[name] = (where / name).read_text().strip()
    if (where / "memory.events").is_file():
        values["memory.events"] = dict(
            line.split() for line in (where / "memory.events").read_text().splitlines())
    for source, names in ((Path("/proc/self/status"), ("VmRSS", "VmHWM", "VmSwap")),
                          (Path("/proc/self/smaps_rollup"), ("Rss", "Pss"))):
        if source.is_file():
            for line in source.read_text().splitlines():
                name, _, rest = line.partition(":")
                if name in names:
                    values[name if name.startswith("Vm") else f"smaps.{name}"] = (
                        int(rest.split()[0]) * 1024)
    return values


def plan(compiled) -> dict:
    analysis = compiled.memory_analysis()
    fields = ("argument_size", "output_size", "alias_size", "temp_size",
              "host_argument_size", "host_output_size", "host_temp_size", "peak_memory")
    head = compiled.as_text().splitlines()[0]
    body = head.partition("entry_computation_layout={(")[2].partition(")->")[0]
    entries = body.split(", ") if body else []
    return {name: int(getattr(analysis, f"{name}_in_bytes")) for name in fields} | {
        "entry_host_parameters": sum(1 for entry in entries if "S(5)" in entry),
        "entry_device_parameters": sum(1 for entry in entries if "S(5)" not in entry)}


def owned(tree, space: str | None = None) -> int:
    return sum(shard.data.nbytes for leaf in jax.tree.leaves(tree)
               for shard in leaf.addressable_shards
               if space is None or str(leaf.sharding.memory_kind) == space)


def main(case: Case) -> None:
    enforced = preflight()
    fields = {**SHAPE, **KINDS[case.kind]}
    model = models.build("causal_transformer", **with_precision(
        "causal_transformer", fields, dtype=case.dtype, attention_impl="xla"))
    plain = models.build("causal_transformer", **with_precision(
        "causal_transformer", {**fields, "scan_layers": False}, dtype=case.dtype,
        attention_impl="xla"))
    tokens = jnp.asarray(np.random.default_rng(case.seed).integers(
        1, SHAPE["vocab_size"], size=(case.batch, case.prompt)), jnp.int32)
    record: dict = {"enforced": enforced, "kind": case.kind, "offload": case.offload, "dtype": case.dtype,
                    "shape": {name: str(value) for name, value in fields.items()},
                    "devices": [str(device) for device in jax.devices()],
                    "jax": jax.__version__, "cgroup_before": cgroup_stats()}

    started = time.perf_counter()
    source = jax.tree.map(
        lambda leaf: leaf.astype(case.dtype) if jnp.issubdtype(leaf.dtype, jnp.floating)
        else leaf, plain.init(jax.random.key(case.seed), tokens))
    jax.block_until_ready(source)
    record["init_seconds"] = time.perf_counter() - started

    layout = Layout(min_shard=1, tolerance=1.0,
                    host_parameters=("params/layers_*",) if case.offload else ())
    started = time.perf_counter()
    store = host_banked(model, HeldBanks(source), layout=layout)
    record["load_seconds"] = time.perf_counter() - started
    record["banks"] = sorted(store["params"])
    record["owned_host_bytes"] = owned(store, "pinned_host")
    record["owned_device_bytes"] = owned(store, "device")
    record["memory_kinds"] = {name: sorted({str(leaf.sharding.memory_kind)
                                            for leaf in jax.tree.leaves(tree)})
                              for name, tree in store["params"].items()}
    del source
    record["cgroup_loaded"] = cgroup_stats()

    started = time.perf_counter()
    cache = jax.block_until_ready(model.apply(
        store, case.batch, method="init_cache", mutable=["cache"])[1]["cache"])
    record["init_cache_seconds"] = time.perf_counter() - started
    record["owned_cache_bytes"] = owned(cache)
    record["cache_paths"] = sorted(cache)

    prefill = jax.jit(lambda held, cached, ids: model.apply(
        {**held, "cache": cached}, ids, decode=True, mutable=["cache"]))
    started = time.perf_counter()
    compiled = prefill.lower(store, cache, tokens).compile()
    record["prefill_compile_seconds"] = time.perf_counter() - started
    record["prefill_plan"] = plan(compiled)
    started = time.perf_counter()
    logits, changed = jax.block_until_ready(compiled(store, cache, tokens))
    record["prefill_seconds"] = time.perf_counter() - started
    record["prefill_last_logits"] = np.asarray(logits[0, -1], np.float32).tolist()

    step = jax.jit(lambda held, cached, token, position: model.apply(
        {**held, "cache": cached}, token, decode=True, mutable=["cache"],
        positions=position))
    token = jnp.argmax(logits[:, -1], axis=-1)[:, None].astype(jnp.int32)
    position = jnp.full((case.batch, 1), case.prompt, jnp.int32)
    started = time.perf_counter()
    stepped = step.lower(store, changed["cache"], token, position).compile()
    record["decode_compile_seconds"] = time.perf_counter() - started
    record["decode_plan"] = plan(stepped)
    cache, produced, latencies = changed["cache"], [], []
    for _ in range(case.new_tokens):
        at = time.perf_counter()
        logits, changed = jax.block_until_ready(stepped(store, cache, token, position))
        latencies.append(time.perf_counter() - at)
        cache = changed["cache"]
        token = jnp.argmax(logits[:, -1], axis=-1)[:, None].astype(jnp.int32)
        produced.append(int(token[0, 0]))
        position = position + 1
    record["decode_tokens"] = produced
    record["decode_latencies_seconds"] = latencies
    record["decode_median_seconds"] = float(np.median(latencies))
    record["decode_tokens_per_second"] = case.batch / float(np.median(latencies))
    if record["owned_host_bytes"]:
        record["owned_host_bytes_per_token"] = record["owned_host_bytes"]
        record["implied_bytes_per_second"] = (
            record["owned_host_bytes"] / float(np.median(latencies)))

    greedy = generate(model, store, tokens, case.new_tokens, seed=case.seed,
                      sampling=Sampling(temperature=0.0))
    record["generate_tokens"] = np.asarray(greedy.tokens).tolist()
    record["allocator"] = {name: int(value) for name, value in
                           (jax.devices()[0].memory_stats() or {}).items()
                           if name in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit",
                                       "largest_alloc_size", "num_allocs")}
    record["cgroup_after"] = cgroup_stats()
    print(json.dumps(record, default=float))
    if case.out is not None:
        Path(case.out).write_text(json.dumps(record, indent=1, default=float))


if __name__ == "__main__":
    main(tyro.cli(Case))
