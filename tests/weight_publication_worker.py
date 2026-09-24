"""One process of a real pool publishing sharded weights to an engine.

Driven by tests/test_rollout_servers.py. Every process loads the tiny Qwen2
fixture, splits each floating leaf over the pool's devices where its first
axis divides, doubles it and calls one push: `SafetensorsReload`, or
`NCCLPush` with its send replaced, since no engine joins an NCCL group in a
test. The push runs under JAX's transfer guard at "log_explicit", so every
device-to-host copy it makes is logged. The process records whether it held
only shards and what the push raised, and asserts nothing, so the invariants
stay in the test that reads the records back.

With --coordinator the pool is joined as Open MPI's environment describes it;
without, `dew launch` started the process. dew is imported inside `publish`,
not at module scope, because a JAX backend opened before
jax.distributed.initialize() would pin the process to its own devices.
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

    from dew.inference import NCCLPush, SafetensorsReload
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
              "error": None, "sent": None}

    class Sent(NCCLPush):
        """The push with its NCCL send replaced by a count of the leaves it would send."""

        def _push(self, served, target, ordinal, version):
            report["sent"] = len(jax.tree.leaves(served))

    if args.publication == "nccl":
        push = Sent(source, (args.engine,), args.library)
    else:
        push = SafetensorsReload(source, Path(args.directory), (args.engine,), "vllm")
    try:
        with jax.transfer_guard_device_to_host("log_explicit"):
            push(variables, 3)
    except BaseException as failure:
        report["error"] = f"{type(failure).__name__}: {failure}"
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="the record's file; without it the record is printed")
    parser.add_argument("--coordinator")
    parser.add_argument("--processes", type=int)
    parser.add_argument("--process-id", type=int)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--publication", choices=("safetensors", "nccl"), default="safetensors")
    parser.add_argument("--library", default="", help="NCCLPush's library path, never opened")
    args = parser.parse_args(argv)
    if args.coordinator:
        os.environ.update({
            "OMPI_MCA_orte_hnp_uri": f"0.0;tcp://{args.coordinator}",
            "OMPI_COMM_WORLD_SIZE": str(args.processes),
            "OMPI_COMM_WORLD_RANK": str(args.process_id),
            "OMPI_COMM_WORLD_LOCAL_RANK": str(args.process_id),
            "JAX_COORDINATOR_ADDRESS": args.coordinator,
        })
    from dew.training.runtime import prepare_process

    if args.coordinator:
        prepare_process(multi_host=True)
    else:
        prepare_process()
    report = publish(args)
    if args.out is None:
        print("report", json.dumps(report), flush=True)
    else:
        args.out.write_text(json.dumps(report))


if __name__ == "__main__":
    main()
