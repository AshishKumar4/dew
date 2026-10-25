"""One process of a real pool averaging runs (`dew.checkpoints.averaged`).

Driven by tests/test_multiprocess.py. The runs are the small decoder of
`multiprocess_worker`, trained or saved here; each mode records what it saw
into --out, and the test reads it back.

dew is imported inside the modes, as `multiprocess_worker`'s are, so the pool
joins before any JAX backend opens.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
from multiprocess_worker import TINY, dump_params, token_data, token_trainer
from process_support import serve_mode, worker_parser

from dew.data import Loading


def averaged_runs(args) -> dict:
    """Two runs of the small decoder from one start, one step and two, averaged in the pool
    (`dew.checkpoints.averaged`), its runs read onto the pool's mesh: each process takes the mean of
    the shards it holds, which every process then gathers whole."""
    import jax

    from dew.checkpoints import averaged
    from dew.training import Layout, MeshSpec

    runs = []
    for name, steps in (("one", 1), ("two", 2)):
        trainer = token_trainer(name, args.run_dir, args.fsdp_size, args.seq_len)
        trainer.fit(token_data(args, packed=False, loading=Loading(workers=0)), steps=steps, log_every=100,
                    checkpoint_every=steps)
        assert trainer.checkpoints is not None
        trainer.checkpoints.wait()
        runs.append(trainer.checkpoints.directory)
    average = averaged(runs, ema=False, mesh=MeshSpec(fsdp=args.fsdp_size), layout=Layout(min_shard=TINY))
    dump_params(args.out.with_suffix(".npz"), average.variables)
    def split(leaf) -> bool:
        # Whether this process holds only part of the leaf: its distinct blocks hold fewer values.
        blocks = {tuple((part.start, part.stop) for part in shard.index): shard.data.size
                  for shard in leaf.addressable_shards}
        return sum(blocks.values()) < leaf.size

    return {"runs": runs, "read": [list(entry) for entry in average.runs],
            "split": any(isinstance(leaf, jax.Array) and split(leaf)
                         for leaf in jax.tree.leaves(average.variables))}


def refused_runs(args) -> dict:
    """Two runs of the small decoder's start whose integer counts differ in the first entry alone, which
    the first process holds: averaged in the pool (`dew.checkpoints.averaged`), the process that holds
    the entry refuses the runs, and its peer hears it where the runs are agreed on, rather than reading
    on alone. Reports each process's refusal, by type."""
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P

    from dew.checkpoints import Checkpoints, averaged
    from dew.coordination import PeerFailure
    from dew.training import Layout, MeshSpec

    trainer = token_trainer("start", args.run_dir, args.fsdp_size, args.seq_len)
    state, _, _ = trainer.place()
    mesh = trainer.device_mesh
    count = 64 * mesh.devices.size
    sharding = NamedSharding(mesh, P("fsdp"))
    runs = []
    for name, first in (("one", 0), ("two", 1)):
        values = np.arange(count, dtype=np.int32)
        values[0] += first
        ids = jax.make_array_from_callback((count,), sharding, lambda index, values=values: values[index])
        directory = str(Path(args.run_dir) / name)
        with Checkpoints(directory) as checkpoints:
            counted = dataclasses.replace(state, variables={**state.variables, "counts": {"ids": ids}})
            checkpoints.save(0, counted, artifact=trainer.objective.inference_record())
            checkpoints.wait()
        runs.append(directory)
    with Checkpoints(runs[0]) as checkpoints:
        read = checkpoints.variables(mesh=MeshSpec(fsdp=args.fsdp_size), layout=Layout(min_shard=TINY))
    held = read["counts"]["ids"]
    split = sum(shard.data.size for shard in held.addressable_shards if shard.replica_id == 0) < held.size
    try:
        averaged(runs, ema=False, mesh=MeshSpec(fsdp=args.fsdp_size), layout=Layout(min_shard=TINY))
    except (ValueError, PeerFailure) as refused:
        return {"refused": type(refused).__name__, "split": bool(split)}
    return {"refused": None, "split": bool(split)}


MODES = {"averaged": averaged_runs, "averaged_refused": refused_runs}


def main(argv=None) -> None:
    parser = worker_parser(MODES, __doc__)
    parser.add_argument("--run-dir")
    parser.add_argument("--tokens", help="the committed token store the runs train on")
    parser.add_argument("--held-tokens")
    parser.add_argument("--seq-len", type=int, default=8)
    serve_mode(MODES, parser.parse_args(argv))


if __name__ == "__main__":
    main()
