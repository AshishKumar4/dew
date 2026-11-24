"""Whether values are finite, and a tree zeroed where they are not: the one statement a step's transaction
(`dew.training.transaction`) and a run that updates in its backward (`dew.nn.in_backward`) both read where a
run skips steps that are not finite (`TrainerAdvanced(skip_nonfinite=True)`)."""

import jax
import jax.numpy as jnp


def _inexact(leaf) -> bool:
    """Whether `leaf` is an array of floating or complex values, the only ones that can be non-finite."""
    return isinstance(leaf, jax.Array) and jnp.issubdtype(leaf.dtype, jnp.inexact)


def leaf_finite(leaf: jax.Array) -> jax.Array:
    """Whether `leaf` is finite throughout."""
    return jnp.isfinite(leaf).all()


def all_finite(tree) -> jax.Array:
    """Whether every floating leaf of `tree` is finite throughout, within a traced program."""
    flags = [leaf_finite(leaf) for leaf in jax.tree.leaves(tree) if _inexact(leaf)]
    return jnp.logical_and.reduce(jnp.stack(flags)) if flags else jnp.asarray(a=True)


def zeroed_unless(ok: jax.Array, tree):
    """`tree` where `ok`, its every floating leaf zero otherwise: an elementwise select against a constant,
    which the computation reading it fuses, so no second copy of the tree is held."""
    def gate(leaf):
        return jnp.where(ok, leaf, jnp.zeros((), leaf.dtype)) if _inexact(leaf) else leaf

    return jax.tree.map(gate, tree)
