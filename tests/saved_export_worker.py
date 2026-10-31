"""One process of a real pool writing a model whose weights span the pool through a run's export
(`dew.interop.saved_sources.export_run`).

Driven by tests/test_saved_sources.py. Every process places one small decoder over a mesh across the
pool, extends each of its layers' attention with DEX (`dew.nn.dex`), whose export folds each output
projection as it is written, and exports it as a saved run's release, the run's restore
(`Pretrained.from_saved`) standing in for the placed decoder and the release cut into small shards,
process 0 publishing and the other reading every tensor as it does (`publishes`). Each records the
files it wrote, the peak of its host allocations during the export (`tracemalloc`, which counts
NumPy's), and the model's size; outside the export's scope the write is refused. It also writes the
same weights unplaced, as one process does. Every mode records what it saw into --out and asserts
nothing.

dew is imported inside the modes, not at module scope, because a JAX backend opened before
jax.distributed.initialize() would pin the process to its own devices.
"""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path
from types import SimpleNamespace

from process_support import serve_mode, worker_parser

FIELDS = {"vocab_size": 512, "emb_features": 64, "num_layers": 8, "num_heads": 4, "num_kv_heads": 2,
          "mlp_features": 128, "max_seq_len": 8}
"""A decoder of about 1.4 MB in float32, its largest tensor (the embedding) 128 KiB."""

TINY_SHARD = 4
"""Far below the production threshold, so the pool's mesh shards every kernel."""

SHARD = "64KB"
"""The release's shard size, so it is written in a score of shards."""


def digests(directory: Path) -> dict[str, str]:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.glob("*.safetensors"))}


def tensors(directory: Path) -> dict[str, tuple[str, list[int], str]]:
    """Each tensor a directory's weight files hold, by name: its dtype, shape and bytes' digest."""
    from safetensors.numpy import load_file

    return {name: (str(array.dtype), list(array.shape), hashlib.sha256(array.tobytes()).hexdigest())
            for path in sorted(directory.glob("*.safetensors")) for name, array in load_file(path).items()}


def exported(args) -> dict:
    """The decoder exported placed across the pool, and written unplaced and outside the scope."""
    import tracemalloc

    import jax
    import jax.numpy as jnp

    from dew.inference.banks import HeldBanks
    from dew.interop.pretrained import Pretrained, PretrainedDecoder
    from dew.interop.saved_sources import export_run
    from dew.nn.dex import DifferentialExtension
    from dew.registry import models
    from dew.training import Layout, MeshSpec

    PretrainedDecoder.save = functools.partialmethod(PretrainedDecoder.save, max_shard_size=SHARD)
    model = models.build("causal_transformer", **FIELDS)
    variables = model.init(jax.random.key(0), jnp.ones((1, 4), jnp.int32))
    placed = HeldBanks(variables).place(model, mesh=MeshSpec(fsdp=args.fsdp_size),
                                        layout=Layout(min_shard=TINY_SHARD, tolerance=1.0))
    leaves = jax.tree.leaves(placed)
    extension = DifferentialExtension(tuple((layer, (0, 2)) for layer in range(FIELDS["num_layers"])),
                                      anneal_steps=10)
    bundle = PretrainedDecoder.from_model(model, placed).with_dex(extension, key=0)
    root = Path(args.run_dir)
    rank = jax.process_index()
    try:
        bundle.save(root / f"outside{rank}")
        refused = False
    except RuntimeError:
        refused = True
    Pretrained.from_saved = classmethod(lambda cls, run, **options: bundle)
    run = SimpleNamespace(objective=SimpleNamespace(saved_task=None))
    destination = root / f"release{rank}"
    tracemalloc.start()
    export_run(run, destination, publishes=rank == 0)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    alone = root / f"alone{rank}"
    PretrainedDecoder.from_model(model, variables).with_dex(extension, key=0).save(alone)
    written, own = tensors(destination), tensors(alone)
    return {"processes": jax.process_count(), "rank": rank,
            "differing": {name: [written.get(name), own.get(name)] for name in sorted({*written, *own})
                          if written.get(name) != own.get(name)} if rank == 0 else {},
            "spanning": sum(not leaf.is_fully_addressable for leaf in leaves),
            "refused_outside": refused, "release": digests(destination), "alone": digests(alone),
            "small_files": sorted(path.name for path in destination.iterdir()
                                  if path.suffix != ".safetensors"),
            "peak": peak, "whole": sum(leaf.nbytes for leaf in leaves),
            "largest_leaf": max(leaf.nbytes for leaf in leaves)}


MODES = {"exported": exported}


def main(argv=None) -> None:
    parser = worker_parser(MODES, __doc__)
    parser.add_argument("--run-dir", required=True)
    serve_mode(MODES, parser.parse_args(argv))


if __name__ == "__main__":
    main()
