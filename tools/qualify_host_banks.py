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

Two processes only compare exactly if they compile the same way, and XLA's
GPU autotuner does not: it times candidate kernels and, compile to compile,
picks a different emitter for the fusion that sums the split-K partials of a
layer's output projections into the residual, which sums them in a different
order. The same graph then gives two different bf16 residuals, resident or
not. So this runs with autotuning off, and every compile of one graph is the
same compile.

    systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 -- \\
        env JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src \\
        python tools/qualify_host_banks.py --kind dense --offload --out r.json
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

# Before jax is imported: the backend parses XLA_FLAGS once, when it starts.
os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} --xla_gpu_autotune_level=0"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import tyro  # noqa: E402

from dew import models  # noqa: E402  naming a registry fills it
from dew.inference.banks import HeldBanks, host_banked  # noqa: E402
from dew.registry import with_precision  # noqa: E402
from dew.sampling.text import Sampling, generate  # noqa: E402
from dew.training import Layout, MeshSpec  # noqa: E402
from dew.training.distributed import build_mesh  # noqa: E402
from benchmark_host_offload import bytes_of, entry_spaces, plan  # noqa: E402
from probe_pinned_charge import preflight  # noqa: E402

# About 200 MiB of bf16 weights: small enough for a four-gigabyte cap to hold
# the source and the store at once, deep enough to bank four times.
SHAPE = dict(vocab_size=1024, emb_features=512, num_heads=8, head_dim=64,
             mlp_features=2048, max_seq_len=64, tie_embeddings=True,
             scan_layers=True, bank_layers=6)

BUDGET_BYTES = 512 * 1024 ** 2
"""The most host-resident weight this is allowed to retain. It is checked
against the abstract store, from the real init shapes and the layout that
would place them, before a single weight is allocated."""

# One depth per kind, chosen so the abstract store fits BUDGET_BYTES. A
# gated delta net carries more parameters per layer than a dense block at the
# same width, which is why its depth is not the dense one; the accounting
# below is what decides, not this comment.
DEPTHS = {"dense": 24, "gated_delta_net": 16, "latent_attention": 24}


def shape_of(kind: str, num_layers: int | None) -> dict:
    depth = DEPTHS[kind] if num_layers is None else num_layers
    fields = {**SHAPE, "num_layers": depth}
    if kind == "gated_delta_net":
        fields |= dict(layer_types=("linear_attention",) * depth,
                       kinds={"linear_attention": {"mixer": {"kind": "gated_delta_net"}}})
    elif kind == "latent_attention":
        fields |= dict(mixer={"kind": "mla", "kv_lora_rank": 64, "q_lora_rank": 64,
                              "qk_rope_head_dim": 16, "qk_nope_head_dim": 16,
                              "v_head_dim": 32})
    return fields


def stored_bytes(shapes, placement) -> dict[str, int]:
    """What the store would retain, per memory kind, from the shapes and the
    shardings alone: each leaf's own shard, times the devices this process
    addresses. Nothing is allocated to find out."""
    totals: dict[str, int] = {}
    leaves = jax.tree.leaves(shapes)
    for leaf, sharding in zip(leaves, jax.tree.leaves(placement), strict=True):
        shard = math.prod(sharding.shard_shape(leaf.shape))
        local = len(sharding.addressable_devices)
        kind = str(sharding.memory_kind)
        totals[kind] = totals.get(kind, 0) + shard * leaf.dtype.itemsize * local
    return totals


@dataclass(frozen=True)
class Case:
    kind: str = "dense"
    offload: bool = False
    dtype: str = "bfloat16"
    batch: int = 1
    prompt: int = 16
    new_tokens: int = 8
    seed: int = 0
    num_layers: int | None = None
    check_only: bool = False
    """Account for the store from shapes alone and stop, allocating nothing."""
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


def digest(value) -> str:
    """A hash of one array's exact bytes, in the dtype it holds.

    Two placements are compared on the values themselves rather than on a
    slice of them, and a hash is small enough to write down. Nothing is cast
    or rounded on the way, so a hash that matches means the bytes matched.
    """
    array = np.asarray(jax.block_until_ready(value))
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:32]


def leaf_digests(tree) -> dict[str, dict[str, str]]:
    """Every leaf of `tree` hashed one at a time, with where it sits.

    One leaf is fetched, hashed and dropped before the next is read, so the
    host holds one leaf of the cache and not the cache.
    """
    leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    values = {}
    for path, leaf in leaves:
        values[jax.tree_util.keystr(path)] = {
            "digest": digest(leaf), "kind": str(leaf.sharding.memory_kind),
            "shape": list(leaf.shape), "dtype": str(leaf.dtype)}
        del leaf
    return values


def main(case: Case) -> None:
    enforced = preflight()
    if min(case.batch, case.prompt, case.new_tokens) < 1:
        raise SystemExit(
            f"refusing to run: batch {case.batch}, prompt {case.prompt} and new_tokens "
            f"{case.new_tokens} all have to be positive")
    fields = shape_of(case.kind, case.num_layers)
    if case.prompt + case.new_tokens > fields["max_seq_len"]:
        raise SystemExit(
            f"refusing to run: a prompt of {case.prompt} and {case.new_tokens} new "
            f"tokens need {case.prompt + case.new_tokens} cache slots and the cache "
            f"holds {fields['max_seq_len']}")
    model = models.build("causal_transformer", **with_precision(
        "causal_transformer", fields, dtype=case.dtype, attention_impl="xla"))
    plain = models.build("causal_transformer", **with_precision(
        "causal_transformer", {**fields, "scan_layers": False}, dtype=case.dtype,
        attention_impl="xla"))
    # The store is accounted for before any weight exists: the real init
    # shapes, cast to the dtype the store holds, under the layout that would
    # offload them, whether or not this case is the offloaded one, so both
    # halves of a pair refuse a shape that overruns.
    offloaded = Layout(min_shard=1, tolerance=1.0, host_parameters=("params/layers_*",))
    abstract = jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(
            leaf.shape, jnp.dtype(case.dtype)
            if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf.dtype),
        jax.eval_shape(lambda key: plain.init(
            key, jax.ShapeDtypeStruct((case.batch, case.prompt), jnp.int32)),
            jax.random.key(case.seed)))
    accounting = stored_bytes(abstract, offloaded.offloaded(build_mesh(MeshSpec()), abstract))
    selected = accounting.get("pinned_host", 0)
    tokens = jnp.asarray(np.random.default_rng(case.seed).integers(
        1, SHAPE["vocab_size"], size=(case.batch, case.prompt)), jnp.int32)
    record: dict = {"enforced": enforced, "accounting": accounting,
                    "selected_host_bytes": selected, "budget_bytes": BUDGET_BYTES,
                    "kind": case.kind, "offload": case.offload, "dtype": case.dtype,
                    "shape": {name: str(value) for name, value in fields.items()},
                    "devices": [str(device) for device in jax.devices()],
                    "jax": jax.__version__, "xla_flags": os.environ["XLA_FLAGS"].strip(),
                    "cgroup_before": cgroup_stats()}

    if selected > BUDGET_BYTES:
        raise SystemExit(
            f"refusing to run: {case.kind} at {fields['num_layers']} layers would "
            f"retain {selected} bytes of host-resident weight, over the "
            f"{BUDGET_BYTES} byte budget, by {selected - BUDGET_BYTES} bytes. The "
            f"whole store is {accounting}. Reduce num_layers or the width")
    if case.check_only:
        print(json.dumps(record, default=float))
        if case.out is not None:
            Path(case.out).write_text(json.dumps(record, indent=1, default=float))
        return

    # Provenance before compute: the weights, the inputs and the empty cache
    # are hashed leaf by leaf, so a difference in what two residencies were
    # given is told apart from a difference in what they computed.
    started = time.perf_counter()
    source = jax.tree.map(
        lambda leaf: leaf.astype(case.dtype) if jnp.issubdtype(leaf.dtype, jnp.floating)
        else leaf, plain.init(jax.random.key(case.seed), tokens))
    jax.block_until_ready(source)
    record["init_seconds"] = time.perf_counter() - started
    record["source_digests"] = leaf_digests(source)
    record["tokens_digest"] = digest(tokens)
    record["tokens_dtype"] = str(tokens.dtype)

    layout = Layout(min_shard=1, tolerance=1.0,
                    host_parameters=("params/layers_*",) if case.offload else ())
    started = time.perf_counter()
    store = host_banked(model, HeldBanks(source), layout=layout)
    record["load_seconds"] = time.perf_counter() - started
    record["banks"] = sorted(store["params"])
    record["owned_host_bytes"] = bytes_of(store, "pinned_host")
    record["owned_device_bytes"] = bytes_of(store, "device")
    record["memory_kinds"] = {name: sorted({str(leaf.sharding.memory_kind)
                                            for leaf in jax.tree.leaves(tree)})
                              for name, tree in store["params"].items()}
    del source
    record["store_digests"] = leaf_digests(store)
    record["cgroup_loaded"] = cgroup_stats()

    forward = jax.jit(lambda held, ids: model.apply(held, ids))
    started = time.perf_counter()
    scores = jax.block_until_ready(forward(store, tokens))
    record["forward_seconds"] = time.perf_counter() - started
    record["forward_logits_digest"] = digest(scores)
    record["forward_logits_shape"] = list(scores.shape)
    del scores

    started = time.perf_counter()
    cache = jax.block_until_ready(model.apply(
        store, case.batch, method="init_cache", mutable=["cache"])[1]["cache"])
    record["init_cache_seconds"] = time.perf_counter() - started
    record["owned_cache_bytes"] = bytes_of(cache)
    record["cache_paths"] = sorted(cache)
    record["initial_cache_digests"] = leaf_digests(cache)

    prefill = jax.jit(lambda held, cached, ids: model.apply(
        {**held, "cache": cached}, ids, decode=True, mutable=["cache"]))
    started = time.perf_counter()
    compiled = prefill.lower(store, cache, tokens).compile()
    record["prefill_compile_seconds"] = time.perf_counter() - started
    record["prefill_plan"] = plan(compiled)
    record["prefill_spaces"] = entry_spaces(compiled)
    started = time.perf_counter()
    logits, changed = jax.block_until_ready(compiled(store, cache, tokens))
    record["prefill_seconds"] = time.perf_counter() - started
    record["prefill_logits_digest"] = digest(logits)
    record["prefill_logits_shape"] = list(logits.shape)
    # Kept beside the digest: a reader can see the values, not only that two
    # hashes agreed.
    record["prefill_last_logits"] = np.asarray(logits[0, -1], np.float32).tolist()
    record["cache_after_prefill"] = leaf_digests(changed["cache"])

    step = jax.jit(lambda held, cached, token, position: model.apply(
        {**held, "cache": cached}, token, decode=True, mutable=["cache"],
        positions=position))
    token = jnp.argmax(logits[:, -1], axis=-1)[:, None].astype(jnp.int32)
    position = jnp.full((case.batch, 1), case.prompt, jnp.int32)
    started = time.perf_counter()
    stepped = step.lower(store, changed["cache"], token, position).compile()
    record["decode_compile_seconds"] = time.perf_counter() - started
    record["decode_plan"] = plan(stepped)
    record["decode_spaces"] = entry_spaces(stepped)
    cache, produced, latencies, scored, steps = changed["cache"], [], [], [], []
    for _ in range(case.new_tokens):
        at = time.perf_counter()
        logits, changed = jax.block_until_ready(stepped(store, cache, token, position))
        latencies.append(time.perf_counter() - at)
        # The hash is taken after the measurement it would otherwise be
        # inside, so the timings are the step's and not the hash's.
        scored.append(logits)
        cache = changed["cache"]
        # After the latency is recorded, so the timing is the step's: one
        # step's cache hashed leaf by leaf, which is what tells a first-step
        # difference from one the whole loop accumulated.
        steps.append(leaf_digests(cache))
        token = jnp.argmax(logits[:, -1], axis=-1)[:, None].astype(jnp.int32)
        produced.append(int(token[0, 0]))
        position = position + 1
    record["decode_logits_digests"] = [digest(step) for step in scored]
    del scored
    record["cache_per_step"] = steps
    record["cache_after_decode"] = leaf_digests(cache)
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
