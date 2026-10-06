"""Whether a placed tree is split over the parameter axes of its mesh.

A test that compares a mesh with one device proves that layout only if the
parameters are split. `Layout` replicates every parameter below its
`min_shard` elements, and a tiny model's parameters are all below the
default, so such a test can pass on data parallelism alone. A meshed
parity test checks its placed parameters with `assert_sharded`.
"""

import jax

from dew.nn.sharding import EXPERT_AXIS, FSDP_AXIS, TENSOR_AXIS

PARAMETER_AXES = (FSDP_AXIS, TENSOR_AXIS, EXPERT_AXIS)
"""The mesh axes a `Layout` splits parameters over."""


def split_axes(tree) -> set[str]:
    """Every mesh axis some leaf of `tree` is split over."""
    names: set[str] = set()
    for leaf in jax.tree.leaves(tree):
        for entry in getattr(getattr(leaf, "sharding", None), "spec", None) or ():
            names.update(entry if isinstance(entry, tuple) else () if entry is None else (entry,))
    return names


def assert_sharded(tree, mesh: jax.sharding.Mesh) -> None:
    """Every parameter axis of `mesh` larger than one splits some leaf of `tree`."""
    wanted = {axis for axis in PARAMETER_AXES if mesh.shape.get(axis, 1) > 1}
    missing = wanted - split_axes(tree)
    assert not missing, f"no parameter is split over {sorted(missing)}, so this mesh runs as data parallelism"
