"""The summation trees a float32 reference is evaluated under (tests/reference_error.py).

Two equally exact orders of one sum cross its float32 rounding boundaries at different rates, so
at an element one tree can land on the correctly rounded value in every order while another lands
an ulp away in most. A reference measured under one tree holds Dew to that tree's luck, so the
gate takes the reference's error at an element as the worst over a declared ensemble of trees,
each summing the same terms in float32 with an error bound of the reference's own:

- "pairwise": adjacent terms added, then their sums, recursively, as a tree reduction does;
- "blocked-<b>": each block of b consecutive terms summed pairwise, then the blocks' sums left to
  right, as a tiled kernel accumulates its tiles.

A sequential sum is no member: its error grows with the term count (one large term and 127 ones,
uniformly placed, part from float64 by 63 on average against pairwise's 1), so it would raise the
bar past what an equally exact reference allows. A block size is one the reference's own kernels
could use. A member is admitted where, on the test's own data, it is no less exact than the
reference (tests/reference_error.py), and on adversarial rows it can be checked the same way
(`mean_absolute_error`). A reduction is written once with `tree_sum` or `tree_contract` and the tree
passed down, so a reference computes the same math under each; a function of several reduction
classes takes its members' trees from `assignments`. Zeros padding a sum to whole pairs or blocks
add exactly.
"""

import itertools
from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np


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


def tree_sum(x, axis: int, tree: str):
    """`x` summed over `axis` in its own dtype under `tree`: "pairwise", or "blocked-<b>", each block
    of b terms pairwise and the blocks' sums in turn."""
    x = jnp.moveaxis(x, axis, -1)
    if tree == "pairwise":
        return _pairwise(x)
    kind, _, size = tree.partition("-")
    if kind != "blocked" or not size.isdigit():
        raise ValueError(f"no summation tree {tree!r}: a tree is 'pairwise' or 'blocked-<b>'")
    block = int(size)
    count = -(-x.shape[-1] // block)
    padded = jnp.concatenate([x, jnp.zeros((*x.shape[:-1], count * block - x.shape[-1]), x.dtype)], axis=-1)
    return _sequential(_pairwise(padded.reshape(*x.shape[:-1], count, block)))


def tree_contract(a, b, axis: int, tree: str):
    """The products of `a` and `b`, broadcast against each other, summed over `axis` under `tree`
    (`tree_sum`): each product rounded once, as an unfused float32 dot rounds it."""
    return tree_sum(a * b, axis, tree)


def assignments(classes: Mapping[str, Sequence[str]], *, bound: int = 8) -> list[dict[str, str]]:
    """The tree each member gives each of a computation's reduction classes, declared before any run:
    every combination of the classes' trees where they number at most `bound`, as a function summing
    one class's terms under one tree and another's under another can cancel where every homogeneous
    member does not (f = T_A(a) + T_B(-a)); past `bound`, each class's trees in turn with every other
    class at its first, which reaches every tree of every class and not every pairing of them."""
    names = list(classes)
    product = [dict(zip(names, chosen, strict=True))
               for chosen in itertools.product(*(classes[name] for name in names))]
    if len(product) <= bound:
        return product
    first = {name: classes[name][0] for name in names}
    return [first] + [{**first, name: tree} for name in names for tree in classes[name][1:]]


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
