import jax.numpy as jnp
from jax.experimental import pallas as pl

from dew.nn.kernels import fla_gdn
from dew.nn.kernels import recurrence_algebra as algebra
from dew.nn.kernels.chunked_launch import _apart, _gates, _rows, read_rows, write_rows


def doubling_solve_chunk(k_ref, g_ref, gl_ref, beta_ref, a_ref, start, taken, *, chunk: int, sub: int, bk: int,
                         exact: bool):
    """The port's kkt-solve with its inverse taken by the algebra's doubling (`recurrence_algebra.factor`)
    in place of fla's forward substitution and merges; interpreted only."""
    width = k_ref.shape[-1]
    count = chunk // sub

    def part(ref, i, *rest):
        return read_rows(_rows(ref, start + i * sub, sub, *rest), taken - i * sub)

    gates = [_gates(g_ref, start, taken, sub, i * sub) for i in range(count)]
    lows = [_gates(gl_ref, start, taken, sub, i * sub) for i in range(count)]
    betas = [part(beta_ref, i).astype(jnp.float32) for i in range(count)]
    index = jnp.arange(sub)
    strict = index[:, None] > index[None, :]

    def keys(i, c):
        return part(k_ref, i, pl.ds(c * bk, bk))

    def block(i, j):
        product = sum(fla_gdn._dot(keys(i, c), keys(j, c).T, exact=exact) for c in range(width // bk))
        decay = jnp.exp2(_apart(gates[i], lows[i], gates[j], lows[j]))
        product = product * (jnp.where(strict, decay, 0.0) if i == j else decay)
        return product * betas[i][:, None]

    rows = [jnp.concatenate([block(i, j) if j <= i else jnp.zeros((sub, sub), jnp.float32)
                             for j in range(count)], axis=1) for i in range(count)]
    inverse = algebra.factor(-jnp.concatenate(rows, axis=0))
    for i in range(count):
        for j in range(count):
            write_rows(_rows(a_ref, start + i * sub, sub, pl.ds(j * sub, sub)), taken - i * sub,
                       inverse[i * sub:(i + 1) * sub, j * sub:(j + 1) * sub])


