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
    grids = Grid.each(template)
    assert grids is not None, "a template placed by name lies on its mesh's grid"
    recovered = _from_replicas(tree, grids, frozenset({0}))

    def read(words, drawn) -> dict:
        # Each process's own copy of a value the pool replicates.
        return {"words": np.asarray(words.addressable_data(0)).tolist(),
                "draw": np.asarray(jax.random.normal(drawn, (3,)).addressable_data(0)).tolist()}

    return {name: read(value, jax.random.wrap_key_data(value, impl="rbg"))
            for name, value in recovered.items()} | {
        "expected": read(data, key),
        "zeros": [list(np.shape(leaf)) for leaf in jax.tree.leaves(jax.tree.map(_zeros, template))]}


def replica_grids(args) -> dict:
    """Two leaves a pool replicates, each on its own mesh over the same devices (one flat, one its 2 x n
    reshape), as a pipeline's rows each lie on their own stage's: the local step records a grid a mesh
    (`Grid.each`), read back from its record as a recovery reads it (`_grids_recorded`), and process 1, which
    lost its copy and holds zeros, receives each leaf from process 0 over its own mesh (row 10814: one grid
    alone was recorded, none for leaves on two meshes, so the step was recovered from none)."""
    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    from dew.checkpoints import _grids_recorded
    from dew.checkpoints._replicas import Grid, _covered, _from_replicas, _zeros
    from dew.registry import from_record, to_record

    devices = np.asarray(jax.devices())
    meshes = [Mesh(devices, ("x",)), Mesh(devices.reshape(2, -1), ("a", "b"))]
    values = {f"row{at}": np.arange(4, dtype=np.float32) + 10 * at for at in range(2)}
    template = {name: jax.ShapeDtypeStruct((4,), np.float32, sharding=NamedSharding(mesh, PartitionSpec()))
                for name, mesh in zip(values, meshes, strict=True)}
    written = {name: jax.make_array_from_callback((4,), template[name].sharding,
                                                  lambda index, name=name: values[name][index])
               for name in values}
    tree = written if jax.process_index() == 0 else jax.tree.map(_zeros, template)
    recorded = _grids_recorded({"grids": [to_record(grid, Grid) for grid in Grid.each(template) or ()]})
    grids = tuple(from_record(Grid, grid) for grid in recorded)
    recovered = _from_replicas(tree, grids, frozenset({0}))
    return {"grids": len(grids), "covered": _covered(grids, frozenset({0})),
            "recovered": {name: np.asarray(value.addressable_data(0)).tolist()
                          for name, value in recovered.items()},
            "expected": {name: value.tolist() for name, value in values.items()}}


def replica_complex(args) -> dict:
    """A complex leaf a pool replicates (a spectral state's), process 1's copy lost and zeros where its
    template places it (`_zeros`): it receives process 0's value, its real and imaginary parts each, bit for
    bit (`_from_replicas`; row 12177: no word holds a complex number's bits, so its sum failed to trace)."""
    import jax
    from jax.sharding import NamedSharding, PartitionSpec

    from dew.checkpoints._replicas import Grid, _from_replicas, _zeros
    from dew.distributed.placement import MeshSpec

    mesh = MeshSpec().build()
    replicated = NamedSharding(mesh, PartitionSpec())
    values = (np.arange(4, dtype=np.float32) - 1.5) + 1j * (np.arange(4, dtype=np.float32) / 3)
    values = values.astype(np.complex64)
    written = jax.make_array_from_callback(values.shape, replicated, lambda index: values[index])
    template = {"spectrum": jax.ShapeDtypeStruct(values.shape, values.dtype, sharding=replicated)}
    tree = {"spectrum": written} if jax.process_index() == 0 else jax.tree.map(_zeros, template)
    grids = Grid.each(template)
    assert grids is not None, "a template placed by name lies on its mesh's grid"
    recovered = np.asarray(_from_replicas(tree, grids, frozenset({0}))["spectrum"].addressable_data(0))
    return {"bits": recovered.view(np.uint32).tolist(), "expected": values.view(np.uint32).tolist()}


MODES = {"replica_key": replica_key, "replica_grids": replica_grids, "replica_complex": replica_complex}


def main(argv=None) -> None:
    parser = worker_parser(MODES, __doc__)
    serve_mode(MODES, parser.parse_args(argv))


if __name__ == "__main__":
    main()
