#!/usr/bin/env python3
"""How an armada gang scales: one process per container, joined over the gang's relay.

    armada map --commit=<sha> --items=gang.json -- .venv-3.12/bin/python tools/armada/gang_bench.py

with gang.json `[{"gang": 16}]`. Every rank joins jax.distributed at rank0, then rank 0 prints one
JSON line: how long the join took, a barrier's round, and for each payload the median time of a
cross-process sum and the bandwidth that implies (the bytes a ring all-reduce moves per rank,
2 (n - 1) / n of the payload, over that time).
"""

import json
import os
import socket
import statistics
import sys
import time

RANK, WORLD = int(os.environ["ARMADA_RANK"]), int(os.environ["ARMADA_WORLD"])
PAYLOADS = (4, 1 << 16, 1 << 20, 16 << 20, 64 << 20)
REPEATS = 10


def main() -> None:
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
    multihost_utils.sync_global_devices("bench end")
    if RANK == 0:
        print(json.dumps({"world": WORLD, "host": socket.gethostname(), "join_s": round(joined, 2),
                          "barrier_s": round(barrier, 3), "sum": results}), flush=True)
    jax.distributed.shutdown()


if __name__ == "__main__":
    sys.exit(main())
