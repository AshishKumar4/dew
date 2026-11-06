"""A run under one layout held to the same run under another by check 1 (`layouts_agree`): two layouts
compile two programs, which XLA holds to no common bits (docs/design/testing.md)."""

import jax
import jax.numpy as jnp
import numpy as np

LAYOUT_SEEDS = 8
"""Check 1's declared seeds for a run under one layout against the same run under another."""


def x64() -> bool:
    """Whether the calling context computes in float64: a check-1 truth runs under `jax.enable_x64`,
    a thread's own setting, which `jax.config.read` does not see."""
    return jnp.zeros(()).dtype == jnp.float64


def widened_start(trainer) -> object:
    """`trainer`'s start placed, and in float64 (`x64`) the float32 run's own start, every floating leaf
    widened: a float64 truth steps the arms' start, where the objective's float literals would be drawn
    anew in float64."""
    if not x64():
        return trainer.place()[0]
    with jax.enable_x64(new_val=False):
        start = trainer.place()[0]
    return jax.tree.map(lambda leaf: leaf.astype(jnp.float64) if jnp.issubdtype(leaf.dtype, jnp.floating)
                        else leaf, start)


def floats(state) -> dict[str, np.ndarray]:
    """A state's floating leaves by path, as check 1 holds them."""
    return {jax.tree_util.keystr(path): np.asarray(leaf)
            for path, leaf in jax.tree_util.tree_leaves_with_path(state)
            if jnp.issubdtype(leaf.dtype, jnp.floating)}


def raw(leaf) -> np.ndarray:
    """A leaf's values, a key's as its data."""
    return np.asarray(jax.random.key_data(leaf) if jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key) else leaf)


def layouts_agree(run, draw, label: str, layout, reference):
    """The state `run(layout, *inputs)` steps to, on par from float64 with `run(reference, *inputs)`'s over
    the seeds' draws (check 1), the truth the reference layout's run in float64 on the draw widened; their
    other leaves (counts, steps, keys) exact at the first draw."""
    from reference_error import as_given, assert_as_exact, widened

    def arm(placed):
        return as_given(lambda *inputs: floats(run(placed, *inputs)))

    assert_as_exact(arm(layout), arm(reference), draw, label, seeds=LAYOUT_SEEDS,
                    truth_fn=as_given(lambda *inputs: floats(run(reference, *widened(inputs)))))
    ran, held = (run(placed, *draw(0)) for placed in (layout, reference))
    assert jax.tree.structure(ran) == jax.tree.structure(held)
    leaves = zip(jax.tree_util.tree_leaves_with_path(ran), jax.tree.leaves(held), strict=True)
    for (path, mine), theirs in leaves:
        if not jnp.issubdtype(mine.dtype, jnp.floating):
            np.testing.assert_array_equal(raw(mine), raw(theirs), err_msg=jax.tree_util.keystr(path))
