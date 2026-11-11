"""The worker of the pool that recovers a lost key from a replica (`dew.checkpoints._replicas`): each
process a rank holding, or having lost, its copy.

dew is imported inside the mode, as `multiprocess_worker`'s are, so the pool starts before JAX does.
"""

from __future__ import annotations

import numpy as np
from process_support import serve_mode, worker_parser


def replica_key(args) -> dict:
    """A key of an implementation not the default (`rbg`, four words) replicated across the data axis the
    two processes split: process 1 lost its copy and holds zeros where its template places it
    (`_zeros`), a concrete key's and a shape's alike, and receives process 0's key data over the mesh
    (`_from_replicas`)."""
    import jax
    from jax.sharding import NamedSharding, PartitionSpec

    from dew.checkpoints._replicas import Grid, _from_replicas, _zeros
    from dew.distributed.placement import MeshSpec

    mesh = MeshSpec().build()
    replicated = NamedSharding(mesh, PartitionSpec())
    words = np.asarray(jax.random.key_data(jax.random.key(7, impl="rbg")))
    data = jax.make_array_from_callback(words.shape, replicated, lambda index: words[index])
    key = jax.random.wrap_key_data(data, impl="rbg")
    shape = jax.ShapeDtypeStruct(key.shape, key.dtype, sharding=replicated)
    template = {"key": key, "shape": shape}
    tree = ({"key": data, "shape": data} if jax.process_index() == 0
            else jax.tree.map(_zeros, template))
    grid = Grid.of(template)
    assert grid is not None, "a template placed by name lies on one grid"
    recovered = _from_replicas(tree, grid, frozenset({0}))

    def read(words, drawn) -> dict:
        # Each process's own copy of a value the pool replicates.
        return {"words": np.asarray(words.addressable_data(0)).tolist(),
                "draw": np.asarray(jax.random.normal(drawn, (3,)).addressable_data(0)).tolist()}

    return {name: read(value, jax.random.wrap_key_data(value, impl="rbg"))
            for name, value in recovered.items()} | {
        "expected": read(data, key),
        "zeros": [list(np.shape(leaf)) for leaf in jax.tree.leaves(jax.tree.map(_zeros, template))]}


MODES = {"replica_key": replica_key}


def main(argv=None) -> None:
    parser = worker_parser(MODES, __doc__)
    serve_mode(MODES, parser.parse_args(argv))


if __name__ == "__main__":
    main()
