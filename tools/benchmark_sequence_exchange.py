"""Time the collectives a sequence axis runs, and each attention exchange over them.

`collectives` times one collective over one group of devices at a time, at
a range of message sizes: a shift to the next device (`ppermute`, what the
windowed halo and the Mamba-2 conv history and state send), the tiled
all-to-all (Ulysses' exchange) and the all-gather (the whole-K/V exchange).
The groups are two-device pairs and all devices, laid out as the mesh's
innermost axis the way `build_mesh` lays out the sequence axis, so on a box
whose pairs differ (an NVLink pair, a PCIe pair, pairs across sockets) each
line is one link class. Several groups run at once where the mesh holds
several, as the sequence axis beside a data axis does.

`attention` times attention forward and backward, bf16, causal or full, through
each exchange `sequence_parallel_attention` can pick, on a mesh of
`sequence=n` over the devices in the given order, against the same call on
one device. With `--profile-dir` each case is also traced, and
`benchmark_step.communication` splits the device time into compute, each
collective, and the collective time no kernel overlapped.

Every line is one JSON record. Times are medians over `--repeats` calls after
two warm-up calls, each call synchronized.

Usage:
    python tools/benchmark_sequence_exchange.py collectives --sizes-mib 1 16 256
    python tools/benchmark_sequence_exchange.py attention --lengths 8192 32768 \\
        --heads 16 --kv-heads 8 --head-dim 128 --orders 0,1,2,3 0,2,1,3
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import os
import sys
import time
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from benchmark_step import communication
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P, SingleDeviceSharding

from dew.nn.attention import attention_kernel, exchanged_heads_attention, gathered_keys_attention
from dew.training import MeshSpec, build_mesh

EXCHANGES: dict[str, Callable] = {
    "all_to_all": exchanged_heads_attention,
    "all_gather": gathered_keys_attention,
}
PROFILED_CALLS = 3


def median_seconds(fn: Callable, *args, repeats: int) -> float:
    jax.block_until_ready(fn(*args))
    jax.block_until_ready(fn(*args))
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def order(text: str) -> list[jax.Device]:
    devices = jax.devices()
    return [devices[int(index)] for index in text.split(",")]


def collectives(sizes_mib: Sequence[float], group: int, orders: Sequence[str],
                repeats: int) -> None:
    """One line per (device order, group size, collective, size)."""
    for text in orders:
        devices = order(text)
        mesh = Mesh(np.asarray(devices).reshape(len(devices) // group, group), ("outer", "inner"))
        for mib in sizes_mib:
            # Per device: `elements` bf16 values, a multiple of the group.
            elements = int(mib * 2 ** 20 / 2) // (group * 128) * group * 128
            data = jax.device_put(jnp.ones((len(devices), elements), jnp.bfloat16),
                                  NamedSharding(mesh, P(("outer", "inner"))))
            shift = [(r, (r + 1) % group) for r in range(group)]
            kinds = {
                "ppermute": lambda x: jax.lax.ppermute(x, "inner", shift),
                "all_to_all": lambda x: jax.lax.all_to_all(x, "inner", 1, 1, tiled=True),
                "all_gather": lambda x: jax.lax.all_gather(
                    x[:, : x.shape[1] // group], "inner", axis=1, tiled=True),
            }
            for kind, body in kinds.items():
                mapped = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=P(("outer", "inner")),
                                               out_specs=P(("outer", "inner"))))
                seconds = median_seconds(mapped, data, repeats=repeats)
                # What one device sends: all of it to its neighbour; the
                # (n-1)/n of its block that belongs elsewhere; its 1/n block
                # to each of n-1 peers.
                sent = {"ppermute": 1.0, "all_to_all": (group - 1) / group,
                        "all_gather": (group - 1) / group}[kind] * elements * 2
                print(json.dumps({
                    "mode": "collectives", "order": text, "group": group, "kind": kind,
                    "mib_per_device": mib, "seconds": seconds,
                    "sent_gb_per_s": sent / seconds / 1e9,
                    "groups": [[int(d.id) for d in row] for row in mesh.devices]}), flush=True)


def attention_case(batch: int, length: int, heads: int, kv_heads: int, head_dim: int,
                   devices: list[jax.Device], exchange: str | None, repeats: int,
                   causal: bool, profile_dir: str | None) -> dict:
    """Forward and backward of bf16 attention over `devices`, split by
    `exchange` over a sequence axis of all of them (None: one device, whole)."""
    keys = jax.random.split(jax.random.key(0), 3)
    shapes = ((batch, length, heads, head_dim), (batch, length, kv_heads, head_dim),
              (batch, length, kv_heads, head_dim))
    kernel = functools.partial(attention_kernel, implementation="auto")

    def attend(q, k, v):
        if exchange is None:
            return kernel(q, k, v, causal=causal, sliding_window=None, mask=None, bias=None)
        return EXCHANGES[exchange](kernel, q, k, v, len(devices), causal=causal,
                                   sliding_window=None, mask=None, bias=None, sinks=None)

    def loss(q, k, v):
        return jnp.sum(attend(q, k, v).astype(jnp.float32) ** 2)

    step = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2)))
    record = {"mode": "attention", "exchange": exchange or "whole", "causal": causal,
              "batch": batch,
              "length": length, "heads": heads, "kv_heads": kv_heads, "head_dim": head_dim,
              "devices": [int(d.id) for d in devices]}
    mesh = build_mesh(MeshSpec(sequence=len(devices)), devices)
    context = contextlib.nullcontext() if exchange is None else jax.set_mesh(mesh)
    try:
        with context:
            placement = (SingleDeviceSharding(devices[0]) if exchange is None
                         else NamedSharding(mesh, P(None, "sequence")))
            operands = [jax.device_put(jax.random.normal(key, shape, jnp.bfloat16), placement)
                        for key, shape in zip(keys, shapes, strict=True)]
            compiled = step.lower(*operands).compile()
            seconds = median_seconds(step, *operands, repeats=repeats)
            if profile_dir is not None:
                directory = os.path.join(
                    profile_dir, f"{record['exchange']}-{length}-"
                    f"{'-'.join(str(d) for d in record['devices'])}")
                jax.profiler.start_trace(directory)
                for _ in range(PROFILED_CALLS):
                    jax.block_until_ready(step(*operands))
                jax.profiler.stop_trace()
                record.update(communication(directory, PROFILED_CALLS))
    except Exception as error:  # an out-of-memory or a refused shape is a result
        record.update(status="failed", error=f"{type(error).__name__}: {str(error)[:300]}")
        return record
    memory = compiled.memory_analysis()
    # Two matmuls forward, five backward (the recomputed logits, dV, dP, dQ,
    # dK), 2 flops a multiply-add; a causal call computes half the logits.
    flops = 2 * 7 * batch * heads * head_dim * length * length / (2 if causal else 1)
    record.update(status="ok", seconds=seconds,
                  tflops_per_device=flops / len(devices if exchange else devices[:1]) / seconds / 1e12,
                  temp_gib=None if memory is None else memory.temp_size_in_bytes / 2 ** 30)
    return record


def attention(args) -> None:
    for text in args.orders:
        devices = order(text)
        for length in args.lengths:
            for exchange in (None, *EXCHANGES):
                if exchange is None and length > args.whole_limit:
                    continue
                record = attention_case(args.batch, length, args.heads, args.kv_heads,
                                        args.head_dim, devices, exchange, args.repeats,
                                        args.call == "causal", args.profile_dir)
                record["order"] = text
                print(json.dumps(record), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    modes = parser.add_subparsers(dest="mode", required=True)
    one = modes.add_parser("collectives")
    one.add_argument("--sizes-mib", type=float, nargs="*", default=[1, 4, 16, 64, 256])
    one.add_argument("--groups", type=int, nargs="*", default=[2, 4])
    one.add_argument("--orders", nargs="*", default=["0,1,2,3", "0,2,1,3"])
    one.add_argument("--repeats", type=int, default=20)
    two = modes.add_parser("attention")
    two.add_argument("--lengths", type=int, nargs="*", default=[4096, 16384, 65536])
    two.add_argument("--batch", type=int, default=1)
    two.add_argument("--heads", type=int, default=16)
    two.add_argument("--kv-heads", type=int, default=8)
    two.add_argument("--head-dim", type=int, default=128)
    two.add_argument("--orders", nargs="*", default=["0,1"])
    two.add_argument("--call", default="causal", choices=["causal", "full"])
    two.add_argument("--whole-limit", type=int, default=32768,
                     help="longest sequence the one-device baseline runs")
    two.add_argument("--repeats", type=int, default=10)
    two.add_argument("--profile-dir", default=None,
                     help="trace each case here and split its device time")
    args = parser.parse_args(argv)
    if args.mode == "collectives":
        for group in args.groups:
            orders = [text for text in args.orders if len(text.split(",")) % group == 0]
            collectives(args.sizes_mib, group, orders if group < len(jax.devices()) else orders[:1],
                        args.repeats)
    else:
        attention(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
