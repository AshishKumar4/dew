"""Fresh SSD inputs and exact head, channel and state-coordinate orbits."""

import jax
import jax.numpy as jnp
import numpy as np
from reference_error import with_gradients
from tool_support import load

from dew.nn.mixers.mamba2 import chunk_ssd, recurrent_ssd

_mamba_ssm = load("mamba_ssd_reference")


def ssd_orbit(arrays, seed):
    """Permute heads with their groups and parameters, P channels and N coordinates together.

    Cotangents follow their primal's coordinates; neither outputs nor gradients are unpermuted.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    heads, width = arrays["x"].shape[2:]
    groups, state = arrays["B"].shape[2:]
    group_order = rng.permutation(groups)
    per_group = heads // groups
    head_order = np.concatenate([group * per_group + rng.permutation(per_group) for group in group_order])
    channels, states = rng.permutation(width), rng.permutation(state)
    axes = {
        "x": ((2, head_order), (3, channels)), "dy": ((2, head_order), (3, channels)),
        "dt": ((2, head_order),), "a": ((2, head_order),),
        "A": ((0, head_order),), "D": ((0, head_order),), "dt_bias": ((0, head_order),),
        "B": ((2, group_order), (3, states)), "C": ((2, group_order), (3, states)),
        "h0": ((1, head_order), (2, channels), (3, states)),
        "dfinal": ((1, head_order), (2, channels), (3, states)),
    }
    result = {}
    for name, value in arrays.items():
        value = np.asarray(value)
        for axis, order in axes.get(name, ()):
            value = np.take(value, order, axis=axis)
        result[name] = value
    return result


def mamba2_draw(seed, shape=(2, 70, 4, 6, 2, 5)):
    """The transformers scan fixture's distribution and geometry, with a fresh input and orbit seed.

    `dt` is raw, before softplus and its per-head bias, as the upstream reference takes it.
    """
    batch, length, heads, width, groups, state = shape
    rng = np.random.default_rng(seed)

    def normal(shape):
        return rng.standard_normal(shape).astype(np.float32)

    return ssd_orbit({
        "x": normal((batch, length, heads, width)), "dt": normal((batch, length, heads)),
        "A": -np.exp(normal((heads,))), "B": normal((batch, length, groups, state)),
        "C": normal((batch, length, groups, state)), "D": normal((heads,)),
        "dt_bias": normal((heads,)), "h0": normal((batch, heads, width, state)),
    }, seed)


def mamba_ssd_operands(rng, shape, dtype, documents):
    """The upstream recorder's native operands without torch, including its bf16-to-fp32 storage."""
    return _mamba_ssm.operands(rng, *shape, lambda value: value.astype(dtype).astype(np.float32), documents)


def mamba_ssd_draw(seed, shape, dtype, documents):
    """Regenerate a seeded mamba_ssm recording's native inputs and exact coordinate orbit."""
    return ssd_orbit(mamba_ssd_operands(np.random.default_rng(seed), shape, dtype, documents), seed)


def mamba2_run(site, arrays):
    """Dew's scan or selective step on the native inputs recorded for transformers."""
    x, dt, A, B, C, D, bias, state = (jnp.asarray(arrays[name]) for name in
                                     ("x", "dt", "A", "B", "C", "D", "dt_bias", "h0"))
    dt = jax.nn.softplus(dt + bias)
    if site == "step":
        output, final = recurrent_ssd(x[:, :1], dt[:, :1], A, B[:, :1], C[:, :1], D, state)
        output = output[:, 0]
    else:
        output, final = chunk_ssd(x, dt, A, B, C, D, state if site == "carried" else None, 32)
    return {"output": output, "final": final}


def ssd_value_and_grad(scan, operands, dy, dfinal):
    """A scan's output, final state and all seven operand gradients under both cotangents, each cast to
    its output's dtype, pulled back through the gate (`reference_error.with_gradients`)."""
    out, final = jax.eval_shape(scan, *operands)
    (out, final), *grads = with_gradients(scan)(*operands, (dy.astype(out.dtype), dfinal.astype(final.dtype)))
    return {"y": out, "final": final, **dict(zip(("x", "dt", "A", "B", "C", "D", "h0"),
                                               grads, strict=True))}


def mamba_ssd_xla(arrays, chunk, dtype=None):
    """The recorder's XLA sizing proxy and float64 cross-check, with optional native input casts."""
    operands = tuple(jnp.asarray(arrays[name], dtype if name in ("x", "B", "C") else None)
                     for name in ("x", "dt", "A", "B", "C", "D", "h0"))
    seq = arrays.get("seq")
    starts = (None if seq is None else
              jnp.concatenate([seq[:, :1] > 0, seq[:, 1:] > seq[:, :-1]], axis=1))
    return ssd_value_and_grad(lambda *args: chunk_ssd(*args, chunk_size=chunk, starts=starts), operands,
                              jnp.asarray(arrays["dy"], dtype), jnp.asarray(arrays["dfinal"]))


