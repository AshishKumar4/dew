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

import dataclasses
import itertools
from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from float64_terms import evaluated_on, walked


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
    (`tree_sum`): each product rounded once, as an unfused float32 dot rounds it. The products are
    held apart from the sum (`optimization_barrier`), so the compiler fuses no product into the add
    that takes it, which would round the pair once, as a fused multiply-add does."""
    return tree_sum(jax.lax.optimization_barrier(a * b), axis, tree)


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


ORDERED = frozenset({"cumsum", "cumprod", "cumlogsumexp", "reduce_prod", "conv_general_dilated",
                     "pallas_call", "custom_call"})
"""Primitives whose result depends on an order or a kernel the interpreter does not set: a reference it
evaluates must not hold one (`evaluated` refuses it, naming it)."""

PRODUCTS = 1 << 26
"""The most products a contraction materializes at once (`_contract`): more are taken a chunk of the
right operand's free columns at a time."""


def evaluated(function, tree: str):
    """`function` with every sum it makes under `tree` (`tree_sum`): its traced jaxpr evaluated with
    each `dot_general` a `tree_contract` and each `reduce_sum` a `tree_sum`, the jaxprs scan, while,
    cond, jit and the custom-derivative calls hold evaluated alike, and every other primitive as it is.
    A forward only: the same math as `function`, its reductions' order the tree's.

    Each dot's products are materialized before the tree sums them, an [M, N, K] array for an
    [M, K] by [K, N] product, so an evaluated function costs K times its dots' memory and runs far
    slower than the function: members are for small operators and the graph gate's nodes, not
    model-scale ensembles (a 4B decoder's forward under "pairwise" took 577 s an order on an H100
    and ran out of memory in its log-softmax)."""
    def run(*args):
        closed, shapes = jax.make_jaxpr(function, return_shape=True)(*args)
        leaves = jax.tree.leaves(args)
        with evaluated_on(leaves):
            found = walked(closed.jaxpr, closed.consts, leaves, Summed(tree))
        return jax.tree.unflatten(jax.tree.structure(shapes), found)

    return run


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Summed:
    """The walk's rule of a reference under a summation tree (`float64_terms.Rule`, `evaluated`): each
    `dot_general` a `tree_contract` and each `reduce_sum` a `tree_sum` under `tree`, a primitive of an order
    the tree does not set refused (`ORDERED`), every other one as it is."""

    tree: str = dataclasses.field(metadata={"static": True})

    def at(self, index: int) -> "Summed":
        return self

    def held(self, dtype):
        return dtype

    def primitive(self, equation, values: list) -> list:
        name, params = equation.primitive.name, equation.params
        # Integers and booleans sum exactly in any order: a position's cumulative count binds as it is.
        exact = not any(jnp.issubdtype(value.dtype, jnp.inexact)
                        for value in values if hasattr(value, "dtype"))
        if name in ORDERED and not exact:
            raise NotImplementedError(f"{name} sums or calls in an order the tree does not set; evaluate a "
                                      "reference without it (an XLA attention, no fused kernel)")
        if not exact and name == "dot_general":
            return [_contract(*values, params["dimension_numbers"], params.get("preferred_element_type"),
                              self.tree)]
        if not exact and name == "reduce_sum":
            (operand,), axes = values, tuple(params["axes"])
            kept = [axis for axis in range(operand.ndim) if axis not in axes]
            moved = jnp.transpose(operand, (*kept, *axes))
            return [tree_sum(moved.reshape(*moved.shape[:len(kept)], -1), -1, self.tree)]
        found = equation.primitive.bind(*values, **equation.primitive.get_bind_params(params))
        return list(found) if equation.primitive.multiple_results else [found]


def _contract(lhs, rhs, dimension_numbers, preferred, tree):
    """`lax.dot_general` with its contraction under `tree`: each product rounded once in at least
    float32 and summed by `tree_sum`, the result in the dtype the dot gives. Batch dimensions, then
    the left operand's free ones, then the right's, as `dot_general` orders them."""
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
    work = jnp.promote_types(jnp.promote_types(lhs.dtype, rhs.dtype), jnp.float32)
    out = preferred or jnp.result_type(lhs, rhs)
    lhs_free = [axis for axis in range(lhs.ndim) if axis not in (*lhs_contract, *lhs_batch)]
    rhs_free = [axis for axis in range(rhs.ndim) if axis not in (*rhs_contract, *rhs_batch)]
    batch = [lhs.shape[axis] for axis in lhs_batch]
    left_shape, right_shape = [lhs.shape[axis] for axis in lhs_free], [rhs.shape[axis] for axis in rhs_free]
    terms = int(np.prod([lhs.shape[axis] for axis in lhs_contract], dtype=np.int64))
    left = jnp.transpose(lhs, (*lhs_batch, *lhs_free, *lhs_contract)).astype(work)
    right = jnp.transpose(rhs, (*rhs_batch, *rhs_free, *rhs_contract)).astype(work)
    left = left.reshape(*batch, int(np.prod(left_shape, dtype=np.int64)), terms)
    right = right.reshape(*batch, int(np.prod(right_shape, dtype=np.int64)), terms)
    rows, columns = left.shape[-2], right.shape[-2]
    chunk = max(1, min(columns, PRODUCTS // max(1, int(np.prod(batch, dtype=np.int64)) * rows * terms)))
    count = -(-columns // chunk)
    padding = count * chunk - columns
    right = jnp.concatenate([right, jnp.zeros((*batch, padding, terms), work)], axis=-2)
    pieces = jnp.moveaxis(right.reshape(*batch, count, chunk, terms), -3, 0)

    def piece(columns_of):
        return tree_contract(left[..., :, None, :], columns_of[..., None, :, :], -1, tree)

    found = jax.lax.map(piece, pieces) if count > 1 else piece(pieces[0])[None]
    found = jnp.moveaxis(found, 0, -2).reshape(*batch, rows, count * chunk)[..., :columns]
    return found.reshape(*batch, *left_shape, *right_shape).astype(out)
