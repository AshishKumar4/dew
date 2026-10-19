"""The summation trees a float32 reference is evaluated under (tests/reference_error.py).

Two equally exact orders of one sum cross its float32 rounding boundaries at different rates, so
at an element one tree can land on the correctly rounded value in every order while another lands
an ulp away in most. A reference measured under one tree holds Dew to that tree's luck, so the
gate takes the reference's error at an element as the worst over a declared ensemble of trees,
each summing the same terms in float32 with an error bound of the reference's own:

- "pairwise": adjacent terms added, then their sums, recursively, as a tree reduction does;
- "blocked": each block of `block` consecutive terms summed pairwise, then the blocks' sums left to
  right, as a tiled kernel accumulates its tiles.

A sequential sum is no member: its error grows with the term count (one large term and 127 ones,
uniformly placed, part from float64 by 63 on average against pairwise's 1), so it would raise the
bar past what an equally exact reference allows. A block size is one the reference's own kernels
could use, and a member is admitted only where, on such adversarial rows, it is no further from
float64 than the reference (`mean_absolute_error`). A reduction is written once with `tree_sum` or
`tree_contract` and the tree passed down, so a reference computes the same math under each. Zeros
padding a sum to whole pairs or blocks add exactly.
"""

import jax
import jax.numpy as jnp
import numpy as np

TREES = ("pairwise", "blocked")
"""The trees a reduction class declares its members from (`tree_sum`)."""


def _pairwise(x):
    """The last axis of `x` summed pairwise: adjacent terms, then their sums."""
    while x.shape[-1] > 1:
        if x.shape[-1] % 2:
            x = jnp.concatenate([x, jnp.zeros_like(x[..., :1])], axis=-1)
        x = x[..., 0::2] + x[..., 1::2]
    return x[..., 0]


def _sequential(x):
    """The last axis of `x` summed left to right."""
    def step(total, term):
        return total + term, None

    total, _ = jax.lax.scan(step, jnp.zeros_like(x[..., 0]), jnp.moveaxis(x, -1, 0))
    return total


def tree_sum(x, axis: int, tree: str, *, block: int):
    """`x` summed over `axis` in its own dtype under `tree` (TREES); "blocked" takes `block` terms
    a block."""
    x = jnp.moveaxis(x, axis, -1)
    if tree == "pairwise":
        return _pairwise(x)
    if tree != "blocked":
        raise ValueError(f"no summation tree {tree!r}; the trees are {TREES}")
    count = -(-x.shape[-1] // block)
    padded = jnp.concatenate([x, jnp.zeros((*x.shape[:-1], count * block - x.shape[-1]), x.dtype)], axis=-1)
    return _sequential(_pairwise(padded.reshape(*x.shape[:-1], count, block)))


def tree_contract(a, b, axis: int, tree: str, *, block: int):
    """The products of `a` and `b`, broadcast against each other, summed over `axis` under `tree`
    (`tree_sum`): each product rounded once, as an unfused float32 dot rounds it."""
    return tree_sum(a * b, axis, tree, block=block)


def adversarial_rows(terms: int, orders: int, seed: int = 0) -> np.ndarray:
    """`orders` float32 rows of one term of 2^24 among `terms - 1` ones, the large one at a uniform
    place in each: a sum whose float32 error depends on the tree alone, from 0 to `terms - 1`."""
    rows = np.ones((orders, terms), np.float32)
    rows[np.arange(orders), np.random.default_rng(seed).integers(0, terms, orders)] = 2.0 ** 24
    return rows


def mean_absolute_error(summed, rows: np.ndarray) -> float:
    """The mean distance from float64 of `summed(rows)`, each row's float32 sum, over the rows."""
    exact = rows.astype(np.float64).sum(axis=-1)
    return float(np.mean(np.abs(np.asarray(summed(jnp.asarray(rows)), np.float64) - exact)))
