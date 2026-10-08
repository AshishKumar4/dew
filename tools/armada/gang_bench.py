#!/usr/bin/env python3
"""How an armada gang scales: one process per container, joined over the gang's relay.

    armada map --commit=<sha> --items=gang.json -- .venv-3.12/bin/python tools/armada/gang_bench.py

with gang.json `[{"gang": 16}]`. First the relay alone: every rank serves an echo and a sink, and
streams to the next rank in a ring, all at once, after timing round trips to it. Then every rank
joins jax.distributed at rank0, and rank 0 prints one JSON line: the relay's round trip and
throughput (each rank's, and their sum), how long the join took, a barrier's round, and for each
payload the median time of a cross-process sum and the bandwidth that implies (the bytes a ring
all-reduce moves per rank, 2 (n - 1) / n of the payload, over that time).
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
    """This rank's round trips and stream to the next rank, while every rank streams to its own."""
    server = socket.create_server(("0.0.0.0", RELAY_PORT), backlog=8)

    def serve():
        while True:
            conn = server.accept()[0]
            with conn:
                kind = conn.recv(1)
                while data := conn.recv(1 << 16):
                    if kind == b"e":
                        conn.sendall(data)

    threading.Thread(target=serve, daemon=True).start()
    peer = f"rank{(RANK + 1) % WORLD}"
    deadline = time.monotonic() + 600
    while True:
        try:
            echo = socket.create_connection((peer, RELAY_PORT), timeout=60)
            break
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.5)
    echo.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    echo.sendall(b"e")
    trips = []
    for _ in range(50):
        start = time.perf_counter()
        echo.sendall(b"x" * 64)
        got = b""
        while len(got) < 64:
            chunk = echo.recv(64 - len(got))
            if not chunk:
                raise ConnectionError(f"rank {RANK}: {peer} closed the echo")
            got += chunk
        trips.append(time.perf_counter() - start)
    echo.close()
    sink = socket.create_connection((peer, RELAY_PORT), timeout=60)
    sink.sendall(b"s")
    start = time.perf_counter()
    sink.sendall(bytes(STREAM))
    sink.shutdown(socket.SHUT_WR)
    sink.recv(1)
    seconds = time.perf_counter() - start
    return {"rtt_ms_p50": round(1000 * statistics.median(trips), 2), "MBps": round(STREAM / seconds / 1e6, 2)}


def main() -> None:
    try:
        measured = relay()
    except Exception:
        # The relay's own account of what it could not reach.
        print(Path("/armada/relay.log").read_text()[-4000:], file=sys.stderr, flush=True)
        raise
    began = time.monotonic()
    import jax

    jax.config.update("jax_cpu_collectives_implementation", "gloo")
    jax.distributed.initialize(coordinator_address="rank0:8476", num_processes=WORLD, process_id=RANK,
                               initialization_timeout=900)
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
    relays = multihost_utils.process_allgather(np.asarray([measured["rtt_ms_p50"], measured["MBps"]]))
    multihost_utils.sync_global_devices("bench end")
    if RANK == 0:
        rtts, rates = np.asarray(relays)[:, 0], np.asarray(relays)[:, 1]
        print(json.dumps({"world": WORLD, "host": socket.gethostname(),
                          "relay": {"rtt_ms_p50": float(np.median(rtts)), "rtt_ms_max": float(rtts.max()),
                                    "MBps_min": float(rates.min()), "MBps_median": float(np.median(rates)),
                                    "MBps_sum": float(rates.sum())},
                          "join_s": round(joined, 2), "barrier_s": round(barrier, 3), "sum": results}),
              flush=True)
    jax.distributed.shutdown()


if __name__ == "__main__":
    sys.exit(main())
