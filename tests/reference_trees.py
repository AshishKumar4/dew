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
from jax.extend.core import ClosedJaxpr, Literal

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
    (`tree_sum`): each product rounded once, as an unfused float32 dot rounds it. The products are
    held apart from the sum (`optimization_barrier`), so the compiler fuses no product into the add
    that takes it, which would round the pair once, as a fused multiply-add does."""
    return tree_sum(jax.lax.optimization_barrier(a * b), axis, tree, block=block)


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


def evaluated(function, tree: str, *, block: int):
    """`function` with every sum it makes under `tree` (`tree_sum`): its traced jaxpr evaluated with
    each `dot_general` a `tree_contract` and each `reduce_sum` a `tree_sum`, the jaxprs scan, while,
    cond, jit and the custom-derivative calls hold evaluated alike, and every other primitive as it is.
    A forward only: the same math as `function`, its reductions' order the tree's."""
    def run(*args):
        closed, shapes = jax.make_jaxpr(function, return_shape=True)(*args)
        found = _evaluate(closed.jaxpr, closed.consts, jax.tree.leaves(args), tree, block)
        return jax.tree.unflatten(jax.tree.structure(shapes), found)

    return run


def _evaluate(jaxpr, consts, args, tree, block):
    """`jaxpr` on `consts` and `args` under `tree`, its outputs."""
    env = dict(zip(jaxpr.constvars, consts, strict=True)) | dict(zip(jaxpr.invars, args, strict=True))

    def read(var):
        return var.val if isinstance(var, Literal) else env[var]

    for equation in jaxpr.eqns:
        found = _apply(equation, [read(var) for var in equation.invars], tree, block)
        env.update(zip(equation.outvars, found if equation.primitive.multiple_results else [found],
                       strict=True))
    return [read(var) for var in jaxpr.outvars]


def _closed(closed, values, tree, block):
    """A closed jaxpr's outputs on `values` under `tree`."""
    return _evaluate(closed.jaxpr, closed.consts, list(values), tree, block)


def _apply(equation, values, tree, block):
    """One equation under `tree`: a reduction as the tree sums it, a jaxpr-holding one evaluated
    within, any other bound as it is."""
    name, params = equation.primitive.name, equation.params
    if name in ORDERED:
        raise NotImplementedError(f"{name} sums or calls in an order the tree does not set; evaluate a "
                                  "reference without it (an XLA attention, no fused kernel)")
    if name == "dot_general":
        return _contract(*values, params["dimension_numbers"], params.get("preferred_element_type"),
                         tree, block)
    if name == "reduce_sum":
        (operand,), axes = values, tuple(params["axes"])
        kept = [axis for axis in range(operand.ndim) if axis not in axes]
        moved = jnp.transpose(operand, (*kept, *axes))
        flat = moved.reshape(*moved.shape[:len(kept)], -1)
        return tree_sum(flat, -1, tree, block=block)
    if name in ("pjit", "jit", "closed_call", "core_call", "remat", "checkpoint", "custom_jvp_call",
                "custom_vjp_call", "custom_vjp_call_jaxpr", "custom_lin"):
        held = next(params[key] for key in ("jaxpr", "call_jaxpr", "fun_jaxpr") if key in params)
        if not hasattr(held, "consts"):
            held = ClosedJaxpr(held, ())
        return _closed(held, values, tree, block)
    if name == "scan":
        consts, carry, xs = (list(part) for part in params["ft_in"].update(values).unpack())

        def step(carried, sliced):
            found = _closed(params["jaxpr"], (*consts, *carried, *sliced), tree, block)
            return tuple(list(part) for part in params["ft_out"].update(found).unpack())

        carried, ys = jax.lax.scan(step, carry, xs, length=params["length"], reverse=params["reverse"])
        return [*carried, *ys]
    if name == "while":
        cond, body = params["cond_jaxpr"], params["body_jaxpr"]
        cond_consts = values[:params["cond_nconsts"]]
        body_consts = values[params["cond_nconsts"]:params["cond_nconsts"] + params["body_nconsts"]]
        init = values[params["cond_nconsts"] + params["body_nconsts"]:]
        return jax.lax.while_loop(lambda carried: _closed(cond, (*cond_consts, *carried), tree, block)[0],
                                  lambda carried: _closed(body, (*body_consts, *carried), tree, block),
                                  list(init))
    if name == "cond":
        index, *operands = values
        branches = [lambda *ops, branch=branch: _closed(branch, ops, tree, block)
                    for branch in params["branches"]]
        return jax.lax.switch(index, branches, *operands)
    return equation.primitive.bind(*values, **params)


def _contract(lhs, rhs, dimension_numbers, preferred, tree, block):
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
        return tree_contract(left[..., :, None, :], columns_of[..., None, :, :], -1, tree, block=block)

    found = jax.lax.map(piece, pieces) if count > 1 else piece(pieces[0])[None]
    found = jnp.moveaxis(found, 0, -2).reshape(*batch, rows, count * chunk)[..., :columns]
    return found.reshape(*batch, *left_shape, *right_shape).astype(out)
