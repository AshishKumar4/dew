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
    """`trainer`'s start placed, and in float64 (`x64`) the float32 run's own start, drawn by `trainer`
    narrow, every floating leaf widened as a float64 truth holds it (`arm_arithmetic.float64_held`): a
    float64 truth steps the arms' start, where the objective's float literals would be drawn anew in
    float64."""
    from arm_arithmetic import float64_held

    if not x64():
        return trainer.place()[0]
    with jax.enable_x64(new_val=False):
        start = trainer.place()[0]
    # Its keys and counters as they are: a floating leaf alone widens.
    return jax.tree.map(lambda leaf: float64_held(leaf) if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf,
                        start)


def floats(state) -> dict[str, np.ndarray]:
    """A state's floating leaves by path, as check 1 holds them."""
    return {jax.tree_util.keystr(path): np.asarray(leaf)
            for path, leaf in jax.tree_util.tree_leaves_with_path(state)
            if jnp.issubdtype(leaf.dtype, jnp.floating)}


def assert_same_leaf(mine, theirs, err_msg: str = "") -> None:
    """Two leaves alike to the bit: their dtypes first, a typed PRNG key's its implementation (two may print
    one tag, `rbg` and `unsafe_rbg`), then their words, a key's its key data
    (`dew.algebra.terms.held_bits`), word for word, so neither stands for the other: a zero's sign and a
    NaN's payload apart too."""
    from dew.algebra.terms import held_bits

    (mine_dtype, mine_bits), (their_dtype, their_bits) = held_bits(mine), held_bits(theirs)
    assert mine_dtype == their_dtype, (err_msg, mine_dtype, their_dtype)
    if mine_bits.shape != their_bits.shape or mine_bits.tobytes() != their_bits.tobytes():
        # The values that differ, where they do; where none does, the words still do (row 13061).
        np.testing.assert_array_equal(mine_bits, their_bits, err_msg=err_msg)
        raise AssertionError(f"{err_msg}: equal values held in other words (a zero's sign, a NaN's payload)")


def layouts_agree(run, draw, label: str, layout, reference, seeds: int = LAYOUT_SEEDS):
    """The state `run(layout, *inputs)` steps to, on par from float64 with `run(reference, *inputs)`'s over
    `seeds` draws (check 1, the site's declared count where `LAYOUT_SEEDS` does not resolve it), the truth
    the reference layout's run in float64 on the draw widened; their other leaves (counts, steps, keys)
    exact at the first draw."""
    from reference_error import as_given, assert_as_exact, widened

    def arm(placed):
        return as_given(lambda *inputs: floats(run(placed, *inputs)))

    assert_as_exact(arm(layout), arm(reference), draw, label, seeds=seeds,
                    truth_fn=as_given(lambda *inputs: floats(run(reference, *widened(inputs)))))
    ran, held = (run(placed, *draw(0)) for placed in (layout, reference))
    assert jax.tree.structure(ran) == jax.tree.structure(held)
    leaves = zip(jax.tree_util.tree_leaves_with_path(ran), jax.tree.leaves(held), strict=True)
    for (path, mine), theirs in leaves:
        if not jnp.issubdtype(mine.dtype, jnp.floating):
            assert_same_leaf(mine, theirs, err_msg=jax.tree_util.keystr(path))
