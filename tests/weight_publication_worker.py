"""One process of a real pool publishing sharded weights to an engine.

Driven by tests/test_rollout_servers.py. Every process loads the tiny Qwen2
fixture, splits each floating leaf over the pool's devices where its first
axis divides, doubles it and calls one `SafetensorsReload` push. The process
records whether it held only shards and what the push raised, and asserts
nothing, so the invariants stay in the test that reads the files back.

dew is imported inside `publish`, not at module scope, because a JAX backend
opened before jax.distributed.initialize() would pin the process to its own
devices.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures/hf/qwen2-tiny"


def publish(args) -> dict:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    from dew.inference import SafetensorsReload
    from dew.interop import load_pretrained

    source = load_pretrained(FIXTURE, dtype="float32")
    mesh = Mesh(np.asarray(jax.devices()), ("pool",))

    def sharded(leaf):
        value = np.asarray(leaf)
        if jnp.issubdtype(value.dtype, jnp.floating):
            value = value * 2
        split = value.ndim and value.shape[0] % mesh.size == 0
        placement = NamedSharding(mesh, PartitionSpec("pool") if split else PartitionSpec())
        return jax.make_array_from_callback(value.shape, placement, lambda index: value[index])

    variables = jax.tree.map(sharded, source.variables)
    report = {"processes": jax.process_count(),
              "sharded": any(not leaf.is_fully_addressable for leaf in jax.tree.leaves(variables)),
              "error": None}
    try:
        SafetensorsReload(source, Path(args.directory), (args.engine,), "vllm")(variables, 3)
    except BaseException as failure:
        report["error"] = f"{type(failure).__name__}: {failure}"
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--coordinator", required=True)
    parser.add_argument("--processes", type=int, required=True)
    parser.add_argument("--process-id", type=int, required=True)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--engine", required=True)
    args = parser.parse_args(argv)
    os.environ.update({
        "OMPI_MCA_orte_hnp_uri": f"0.0;tcp://{args.coordinator}",
        "OMPI_COMM_WORLD_SIZE": str(args.processes),
        "OMPI_COMM_WORLD_RANK": str(args.process_id),
        "OMPI_COMM_WORLD_LOCAL_RANK": str(args.process_id),
        "JAX_COORDINATOR_ADDRESS": args.coordinator,
    })
    from dew.training.runtime import prepare_process

    prepare_process(multi_host=True)
    args.out.write_text(json.dumps(publish(args)))


if __name__ == "__main__":
    main()
