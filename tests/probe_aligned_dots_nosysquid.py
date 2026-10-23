"""Independently widen a named product's primal or either transpose."""
import functools

import jax
import jax.numpy as jnp
from jax.custom_derivatives import linear_call

from dew.nn.kernels import recurrence_algebra as a


def wide_dot(left, right):
    return jnp.matmul(left.astype(jnp.float64), right.astype(jnp.float64),
                      precision="highest").astype(left.dtype)


def install(selected):
    selected = set(selected.split(","))
    if "all" in selected:
        selected = {f"{name}.{side}" for name in ("merge", "intra", "scores", "local", "apply_w",
                                                  "apply_u", "corrected", "memory", "write")
                    for side in ("p", "l", "r", "e")}
    original = a.portable
    original_factor = a.factor
    original_paired = a.paired

    @functools.lru_cache(None)
    def product(name):
        @jax.custom_jvp
        def dot(left, right):
            return (wide_dot(left, right) if name + ".p" in selected
                    else original(name, left, right))

        @dot.defjvp
        def tangent(primals, tangents):
            left, right = primals
            dleft, dright = tangents

            def lhs_transpose(right, cot):
                return (wide_dot(cot, jnp.swapaxes(right, -1, -2)) if name + ".l" in selected
                        else original(name, cot, jnp.swapaxes(right, -1, -2)))

            def rhs_transpose(left, cot):
                return (wide_dot(jnp.swapaxes(left, -1, -2), cot) if name + ".r" in selected
                        else original(name, jnp.swapaxes(left, -1, -2), cot))

            lhs = linear_call(lambda right, dx: original(name, dx, right), lhs_transpose, right, dleft)
            rhs = linear_call(lambda left, dy: original(name, left, dy), rhs_transpose, left, dright)
            return dot(left, right), lhs + rhs

        return dot

    def issue(name, left, right):
        if name.endswith(".entered"):
            return (wide_dot(left, right) if name.removesuffix(".entered") + ".e" in selected
                    else original(name, left, right))
        return product(name)(left, right)

    for name in ("factor", "paired", "intra", "apply", "carry", "readout"):
        fn = getattr(a, name)
        fn.__defaults__ = (*fn.__defaults__[:-1], issue)

    @functools.partial(jax.custom_jvp, nondiff_argnums=(2, 3))
    def remembered(left, right, issuer=issue, name="memory"):
        return issuer(name, left, right)

    @remembered.defjvp
    def remembered_jvp(issuer, name, primals, tangents):
        left, right = primals
        dleft, dright = tangents

        def lhs_transpose(right, cot):
            rhs = jnp.swapaxes(right, -1, -2)
            return wide_dot(cot, rhs) if name + ".l" in selected else original(name, cot, rhs)

        def rhs_transpose(left, cot):
            lhs = jnp.swapaxes(jnp.flip(left, -2), -1, -2)
            rhs = jnp.flip(cot, -2)
            return wide_dot(lhs, rhs) if name + ".e" in selected else original(name, lhs, rhs)

        lhs = linear_call(lambda right, dx: original(name, dx, right), lhs_transpose, right, dleft)
        rhs = linear_call(lambda left, dy: original(name, left, dy), rhs_transpose, left, dright)
        return remembered(left, right, issuer, name), lhs + rhs

    a.remembered = remembered

    def einsum(expression, *xs, precise):
        if precise:
            dtype = xs[0].dtype
            return jnp.einsum(expression, *(x.astype(jnp.float64) for x in xs),
                              precision="highest").astype(dtype)
        return jnp.einsum(expression, *xs, precision="highest")

    @functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
    def vector(name, left, right, decay):
        return einsum("...sd,...td,...std->...st", left, right, decay, precise=name + ".p" in selected)

    @vector.defjvp
    def vector_jvp(name, primals, tangents):
        left, right, decay = primals
        dl, dr, dd = tangents

        def tl(res, dx):
            right, decay = res
            return einsum("...sd,...td,...std->...st", dx, right, decay, precise=False)

        def tr(res, dy):
            left, decay = res
            return einsum("...sd,...td,...std->...st", left, dy, decay, precise=False)

        def td(res, dp):
            left, right = res
            return einsum("...sd,...td,...std->...st", left, right, dp, precise=False)

        def pl(res, ct):
            right, decay = res
            return einsum("...st,...td,...std->...sd", ct, right, decay,
                          precise=name + ".l" in selected)

        def pr(res, ct):
            left, decay = res
            return einsum("...st,...sd,...std->...td", ct, left, decay,
                          precise=name + ".r" in selected)

        def pd(res, ct):
            left, right = res
            return einsum("...st,...sd,...td->...std", ct, left, right,
                          precise=name + ".d" in selected)

        tangent = (linear_call(tl, pl, (right, decay), dl)
                   + linear_call(tr, pr, (left, decay), dr)
                   + linear_call(td, pd, (left, right), dd))
        return vector(name, left, right, decay), tangent

    def paired(name, left, right, decay, issue=issue):
        return (original_paired(name, left, right, decay, issue) if decay.shape[-1] == 1
                else vector(name, left, right, decay))

    a.paired = paired

    if "inverse.t" in selected or "inverse.32" in selected:
        @jax.custom_jvp
        def inverse(matrix):
            return original_factor(matrix, issue)

        @inverse.defjvp
        def inverse_jvp(primals, tangents):
            (matrix,), (dmatrix,) = primals, tangents
            inv = inverse(matrix)
            strict = jnp.tril(jnp.ones(matrix.shape[-2:], bool), -1)

            def solve(inv, dm):
                return original("merge", original("merge", inv, jnp.where(strict, dm, 0.0)), inv)

            def transpose(inv, cot):
                it = jnp.swapaxes(inv, -1, -2)
                if "inverse.t" in selected:
                    cot = cot.astype(jnp.float64)
                    it = it.astype(jnp.float64)
                    result = (it @ cot @ it).astype(inv.dtype)
                else:
                    result = original("merge", original("merge", it, cot), it)
                return jnp.where(strict, result, 0.0)

            return inv, linear_call(solve, transpose, inv, dmatrix)

        a.factor = lambda matrix, issue=issue: inverse(matrix)
