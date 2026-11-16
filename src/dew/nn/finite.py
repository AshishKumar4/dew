"""Whether values are finite, and a tree zeroed where they are not: the one statement a step's transaction
(`dew.training.transaction`) and a run that updates in its backward (`dew.nn.in_backward`) both gate on."""

import jax
import jax.numpy as jnp
import numpy as np


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


MAGNITUDE_BOUND = float(np.sqrt(np.float64(jnp.finfo(jnp.bfloat16).max)))
"""The largest gradient magnitude an update is formed from finite: below it every square is finite in the
narrowest dtype a moment is stored in (bf16, a hair under float32's own), so no second moment (Adam's,
LaProp's, LAMB's: an average of past squares, never above their largest) passes it."""


def bounded(tree) -> jax.Array:
    """Whether every floating leaf of `tree` is finite with no magnitude past `MAGNITUDE_BOUND`: one
    max-reduction, which cannot overflow and is false for a value that is not a number, so it holds the
    finiteness check too (training.md, a step that is not finite)."""
    # Read in float32 at least: a half dtype's bound is its own infinity, which an infinity meets.
    flags = [jnp.max(jnp.abs(leaf)).astype(jnp.promote_types(leaf.dtype, jnp.float32)) <= MAGNITUDE_BOUND
             for leaf in jax.tree.leaves(tree) if _inexact(leaf) and leaf.size]
    return jnp.logical_and.reduce(jnp.stack(flags)) if flags else jnp.asarray(a=True)


def norm_and_bound(tree) -> tuple[jax.Array, jax.Array]:
    """The global norm of `tree`'s floating leaves, as `optax.global_norm` states it, and whether each is
    finite within `MAGNITUDE_BOUND` (`bounded`), from one reduction a leaf: its squares' sum and its largest
    magnitude reduced together, so the bound reads the gradient in the pass its norm does. Read apart, the
    two took a kernel each a leaf and the gradient's finiteness a third, 1.9 times main's kernels on the 4B
    decision step (+5.3% of its step on a GH200, UB-ATTR5)."""
    squares, within = [], jnp.asarray(a=True)
    for leaf in jax.tree.leaves(tree):
        if not (_inexact(leaf) and leaf.size):
            continue
        square = jnp.real(leaf * jnp.conj(leaf)) if jnp.iscomplexobj(leaf) else leaf * leaf
        # Read in float32 at least: a half dtype's bound is its own infinity, which an infinity meets.
        magnitude = jnp.abs(leaf).astype(jnp.promote_types(leaf.dtype, jnp.float32))
        summed, top = jax.lax.reduce((square, magnitude),
                                     (jnp.zeros((), square.dtype), jnp.zeros((), magnitude.dtype)),
                                     lambda one, other: (one[0] + other[0], jnp.maximum(one[1], other[1])),
                                     tuple(range(leaf.ndim)))
        squares.append(summed)
        within = within & (top <= MAGNITUDE_BOUND)
    return (jnp.sqrt(sum(squares)) if squares else jnp.zeros(())), within
