#!/usr/bin/env python3
"""How an armada gang scales: one process per container, joined over the gang's relay.

    armada map --commit=<sha> --items=gang.json -- .venv-3.12/bin/python tools/armada/gang_bench.py

with gang.json `[{"gang": 16}]`. First the relay alone: every rank serves an echo and a sink, and
streams to the next rank in a ring, all at once, after timing round trips to it. Then every rank
joins jax.distributed at rank0, and rank 0 prints one JSON line: the relay's round trip and
throughput (each rank's, and their sum), how long the join took, a barrier's round, and for each
payload the median time of a cross-process sum and the bandwidth that implies (the bytes a ring
all-reduce moves per rank, 2 (n - 1) / n of the payload, over that time), and what each way the
pool agrees on the host takes (`agreements`).
"""

import json
import os
import socket
import statistics
import sys
import threading
import time
from pathlib import Path

RANK, WORLD = int(os.environ["ARMADA_RANK"]), int(os.environ["ARMADA_WORLD"])
PAYLOADS = (4, 1 << 16, 1 << 20, 16 << 20, 64 << 20)
REPEATS = 10
RELAY_PORT, STREAM = 9100, 32 << 20


def relay() -> dict:
    """This rank's round trips and stream to the next rank, while every rank streams to its own.

    A connection to another rank always opens, since this host's relay takes it; one the peer had
    nothing listening for closes before its server's first byte, and is opened again. A rank returns
    once its predecessor is done with its server too, so no rank takes its server away early.
    """
    server = socket.create_server(("0.0.0.0", RELAY_PORT), backlog=8)
    served = threading.Event()

    def serve():
        while True:
            conn = server.accept()[0]
            with conn:
                kind = conn.recv(1)
                conn.sendall(b"r")
                while data := conn.recv(1 << 16):
                    if kind == b"e":
                        conn.sendall(data)
                if kind == b"s":
                    conn.sendall(b"d")
                    served.set()

    threading.Thread(target=serve, daemon=True).start()
    peer = f"rank{(RANK + 1) % WORLD}"
    deadline = time.monotonic() + 600

    def ready(kind: bytes) -> socket.socket:
        while True:
            conn = socket.create_connection((peer, RELAY_PORT), timeout=120)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.sendall(kind)
            if conn.recv(1) == b"r":
                return conn
            conn.close()
            if time.monotonic() > deadline:
                raise ConnectionError(f"rank {RANK}: {peer} never answered")
            time.sleep(0.5)

    def exactly(conn: socket.socket, size: int) -> None:
        got = 0
        while got < size:
            chunk = conn.recv(size - got)
            if not chunk:
                raise ConnectionError(f"rank {RANK}: {peer} closed mid-answer")
            got += len(chunk)

    with ready(b"e") as echo:
        trips = []
        for _ in range(50):
            start = time.perf_counter()
            echo.sendall(b"x" * 64)
            exactly(echo, 64)
            trips.append(time.perf_counter() - start)
    with ready(b"s") as sink:
        start = time.perf_counter()
        sink.sendall(bytes(STREAM))
        sink.shutdown(socket.SHUT_WR)
        exactly(sink, 1)
        seconds = time.perf_counter() - start
    if not served.wait(timeout=600):
        raise ConnectionError(f"rank {RANK}: rank{(RANK - 1) % WORLD} never streamed")
    return {"rtt_ms_p50": round(1000 * statistics.median(trips), 2), "MBps": round(STREAM / seconds / 1e6, 2)}


def gradient_sync(mesh) -> dict:
    """The seconds to sum tools/cluster_vs_gpu.py's dense gradients over the hosts: as the
    data-parallel step does, one all-reduce of a tuple of every parameter's buffer, and as one
    flat buffer of them all."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec

    from dew.nn.backbones import CausalTransformer

    model = CausalTransformer(emb_features=384, num_layers=6, num_heads=6, mlp_features=1536,
                              vocab_size=256, max_seq_len=257)
    shapes = jax.eval_shape(model.init, jax.random.key(0), jnp.zeros((1, 256), jnp.int32))["params"]
    rows = [jax.make_array_from_process_local_data(NamedSharding(mesh, PartitionSpec("hosts")),
                                                   np.ones((1, *leaf.shape), np.float32))
            for leaf in jax.tree.leaves(shapes)]
    replicated = NamedSharding(mesh, PartitionSpec())
    tupled = jax.jit(lambda leaves: [leaf.sum(0) for leaf in leaves], out_shardings=replicated)
    flat = jax.jit(lambda leaves: jnp.concatenate([leaf.reshape(len(leaf), -1) for leaf in leaves], 1).sum(0),
                   out_shardings=replicated)
    timed = {}
    for name, summed in (("tuple", tupled), ("flat", flat)):
        timed[f"{name}_all_reduces"] = summed.lower(rows).compile().as_text().count(" all-reduce(")
        jax.block_until_ready(summed(rows))
        times = []
        for _ in range(3):
            start = time.monotonic()
            jax.block_until_ready(summed(rows))
            times.append(time.monotonic() - start)
        timed[f"{name}_s"] = round(statistics.median(times), 3)
    return {"buffers": len(rows), "bytes": sum(4 * leaf.size for leaf in jax.tree.leaves(shapes)), **timed}


def agreements() -> dict:
    """The median seconds of each way the pool agrees on the host (`dew.coordination`), and of
    their parts: the coordination service's barrier alone, a 4-byte allgather alone, and a round of
    the service's key-value store (each rank sets a key, meets the barrier and reads every key)."""
    import numpy as np
    from jax.experimental import multihost_utils

    from dew.coordination import _client, agree_process_phase, agreed, agreed_same

    client = _client()
    rounds = iter(range(1_000_000))

    def barrier():
        client.wait_at_barrier(f"bench/barrier/{next(rounds)}", 60_000)

    def key_values():
        directory = f"bench/kv/{next(rounds)}/"
        client.key_value_set(directory + str(RANK), "0")
        client.wait_at_barrier(directory, 60_000)
        assert len(client.key_value_dir_get(directory)) == WORLD

    ways = {"barrier": barrier,
            "allgather_4B": lambda: multihost_utils.process_allgather(np.asarray(RANK, np.int32)),
            "key_value_round": key_values,
            "agree_process_phase": lambda: agree_process_phase(None, phase="bench"),
            "agreed": lambda: agreed("bench", lambda: None),
            "agreed_same": lambda: agreed_same("bench", lambda: "same")}
    timed = {}
    for name, way in ways.items():
        way()
        times = []
        for _ in range(REPEATS):
            start = time.monotonic()
            way()
            times.append(time.monotonic() - start)
        timed[f"{name}_s"] = round(statistics.median(times), 4)
    return timed


def main() -> None:
    print(f"rank {RANK} started at {time.time():.3f}", file=sys.stderr, flush=True)
    try:
        measured = relay()
        print(f"rank {RANK} relayed at {time.time():.3f}: {measured}", file=sys.stderr, flush=True)
    except Exception:
        # The relay's own account of what it could not reach.
        print(Path("/armada/relay.log").read_text()[-4000:], file=sys.stderr, flush=True)
        raise
    if os.environ.get("GANG_RELAY_ONLY"):
        return
    began = time.monotonic()
    import jax

    jax.config.update("jax_cpu_collectives_implementation", "gloo")
    jax.distributed.initialize(coordinator_address="rank0:8476", num_processes=WORLD, process_id=RANK,
                               initialization_timeout=int(os.environ.get("GANG_INIT_TIMEOUT", "900")))
    joined = time.monotonic() - began
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental import multihost_utils
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    mesh = Mesh(np.array(jax.devices()), ("hosts",))
    started = time.monotonic()
    multihost_utils.sync_global_devices("bench start")
    barrier = time.monotonic() - started
    # One row per process; the sum over the hosts axis is the all-reduce the step of a data-parallel run does.
    summed = jax.jit(lambda rows: rows.sum(axis=0), out_shardings=NamedSharding(mesh, PartitionSpec()))
    results = []
    for size in PAYLOADS:
        local = jnp.full((1, max(size // 4, 1)), RANK + 1.0, jnp.float32)
        rows = jax.make_array_from_process_local_data(NamedSharding(mesh, PartitionSpec("hosts")), local)
        total = summed(rows).block_until_ready()
        expected = WORLD * (WORLD + 1) / 2
        assert float(np.asarray(total.addressable_data(0))[0]) == expected, "the sum is wrong"
        times = []
        for _ in range(REPEATS if size < (16 << 20) else 3):
            start = time.monotonic()
            summed(rows).block_until_ready()
            times.append(time.monotonic() - start)
        median = statistics.median(times)
        moved = 2 * (WORLD - 1) / WORLD * size
        results.append({"bytes": size, "median_s": round(median, 4),
                        "bus_MBps": round(moved / median / 1e6, 2)})
    agreeing = agreements()
    synced = gradient_sync(mesh)
    relays = multihost_utils.process_allgather(np.asarray([measured["rtt_ms_p50"], measured["MBps"]]))
    multihost_utils.sync_global_devices("bench end")
    if RANK == 0:
        rtts, rates = np.asarray(relays)[:, 0], np.asarray(relays)[:, 1]
        print(json.dumps({"world": WORLD, "host": socket.gethostname(),
                          "relay": {"rtt_ms_p50": float(np.median(rtts)), "rtt_ms_max": float(rtts.max()),
                                    "MBps_min": float(rates.min()), "MBps_median": float(np.median(rates)),
                                    "MBps_sum": float(rates.sum())},
                          "join_s": round(joined, 2), "barrier_s": round(barrier, 3), "sum": results,
                          "agreements": agreeing, "gradient_sync": synced}),
              flush=True)
    jax.distributed.shutdown()


if __name__ == "__main__":
    sys.exit(main())
