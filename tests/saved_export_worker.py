"""One process of a real pool writing a model whose weights span the pool, as a run's export writes one
(`dew.interop.saved_sources.export_run`).

Driven by tests/test_saved_sources.py. Every process places one tiny decoder over a mesh across the
pool, writes it to a directory of its own inside `reading_home`, which reads each tensor home together
as the writer reads it, and records what it wrote and the largest tree it gathered; outside the scope
the write is refused. It also writes the same weights unplaced, as one process does. Every mode records
what it saw into --out and asserts nothing.

dew is imported inside the modes, not at module scope, because a JAX backend opened before
jax.distributed.initialize() would pin the process to its own devices.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from process_support import serve_mode, worker_parser

FIELDS = {"vocab_size": 32, "emb_features": 16, "num_layers": 2, "num_heads": 4, "num_kv_heads": 2,
          "mlp_features": 32, "max_seq_len": 8}
TINY_SHARD = 4
"""Far below the production threshold, so the pool's mesh shards every kernel."""


def digests(directory: Path) -> dict[str, str]:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.glob("*.safetensors"))}


def exported(args) -> dict:
    """The decoder written placed across the pool, inside and outside `reading_home`, and unplaced."""
    import jax
    import jax.numpy as jnp

    from dew import _coordination
    from dew._coordination import reading_home
    from dew.inference.banks import HeldBanks
    from dew.interop.pretrained import PretrainedDecoder
    from dew.registry import models
    from dew.training import Layout, MeshSpec

    model = models.build("causal_transformer", **FIELDS)
    variables = model.init(jax.random.key(0), jnp.ones((1, 4), jnp.int32))
    placed = HeldBanks(variables).place(model, mesh=MeshSpec(fsdp=args.fsdp_size),
                                        layout=Layout(min_shard=TINY_SHARD, tolerance=1.0))
    leaves = jax.tree.leaves(placed)
    root = Path(args.run_dir)
    rank = jax.process_index()
    outside = root / f"outside{rank}"
    try:
        PretrainedDecoder.from_model(model, placed).save(outside)
        refused = False
    except RuntimeError:
        refused = True
    gathered = []
    gather = _coordination.collective_host

    def recorded(value, **options):
        gathered.append(sum(leaf.nbytes for leaf in jax.tree.leaves(value)))
        return gather(value, **options)

    _coordination.collective_host = recorded
    pooled = root / f"pooled{rank}"
    with reading_home("pool export"):
        PretrainedDecoder.from_model(model, placed).save(pooled)
    _coordination.collective_host = gather
    alone = root / f"alone{rank}"
    PretrainedDecoder.from_model(model, variables).save(alone)
    return {"processes": jax.process_count(),
            "spanning": sum(not leaf.is_fully_addressable for leaf in leaves),
            "refused_outside": refused, "pooled": digests(pooled), "alone": digests(alone),
            "gathers": len(gathered), "largest_gathered": max(gathered, default=0),
            "largest_leaf": max(leaf.nbytes for leaf in leaves),
            "whole": sum(leaf.nbytes for leaf in leaves)}


MODES = {"exported": exported}


def main(argv=None) -> None:
    parser = worker_parser(MODES, __doc__)
    parser.add_argument("--run-dir", required=True)
    serve_mode(MODES, parser.parse_args(argv))


if __name__ == "__main__":
    main()
