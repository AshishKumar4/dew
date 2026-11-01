"""Fresh SSD inputs and exact head, channel and state-coordinate orbits, and the recurrence's float64
oracle (`stepwise_scan`)."""

import jax
import jax.numpy as jnp
import numpy as np
from reference_error import with_gradients
from tool_support import load

from dew.nn.mixers.mamba2 import Mamba2, chunk_ssd, recurrent_ssd

_mamba_ssm = load("mamba_ssd_reference")

LAYER_SHAPES = {"in_proj.weight": (72, 12), "conv1d.weight": (44, 1, 4), "conv1d.bias": (44,),
                "dt_bias": (4,), "A_log": (4,), "D": (4,), "norm.weight": (24,), "out_proj.weight": (12, 24),
                "hidden": (2, 70, 12)}
"""One transformers `Mamba2Mixer`'s weights (torch's names and layouts) and hidden states at the layer
fixture's geometry (tests/fixtures/mamba2/config.json: a hidden width of 12, four heads of six, two groups of
five states, a 4-wide conv), two rows of 70 tokens."""

LAYER_GEOMETRY = {"hidden_size": 12, "num_heads": 4, "head_dim": 6, "state_size": 5, "n_groups": 2,
                  "conv_kernel": 4, "chunk_size": 32, "layer_norm_epsilon": 1e-5}
"""The layer fixture's geometry (tests/fixtures/mamba2/config.json's)."""


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


def stepwise_scan(x, b, c, a, state):
    """The recurrence one step at a time, no chunks: `h_t = exp(a_t) h_{t-1}
    + x_t b_t^T`, `y_t = h_t c_t`. Written apart from both chunked paths, so
    in float64 it is an oracle neither of them shares a line with."""
    def step(h, inputs):
        x, b, c, a = inputs
        h = jnp.exp(a)[..., None, None] * h + x[..., :, None] * b[..., None, :]
        return h, jnp.einsum('bhpn,bhn->bhp', h, c)

    final, ys = jax.lax.scan(step, state, tuple(jnp.moveaxis(t, 1, 0) for t in (x, b, c, a)))
    return jnp.moveaxis(ys, 0, 1), final


def mamba2_layer_draw(seed):
    """A fresh `Mamba2Mixer` (`LAYER_SHAPES`): its weights and hidden states standard normal, the weights
    halved, as the layer fixture draws them (tools/mamba2_reference.py's `layer_fixture`)."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    return {name: (rng.standard_normal(shape) * (1.0 if name == "hidden" else 0.5)).astype(np.float32)
            for name, shape in LAYER_SHAPES.items()}


def mamba2_layer(dtype=jnp.float32) -> Mamba2:
    """Dew's Mamba-2 mixer at the layer fixture's geometry (`LAYER_GEOMETRY`) computing in `dtype`."""
    g = LAYER_GEOMETRY
    return Mamba2(emb_features=g["hidden_size"], num_heads=g["num_heads"], head_dim=g["head_dim"],
                  state_size=g["state_size"], n_groups=g["n_groups"], conv_kernel=g["conv_kernel"],
                  chunk_size=g["chunk_size"], norm_eps=g["layer_norm_epsilon"], dtype=dtype)


def mamba2_layer_params(arrays) -> dict:
    """A transformers `Mamba2Mixer`'s weights (torch's names) under Dew's names: kernels transposed from
    torch's [out, in], every other weight as stored."""
    return {"params": {
        "in_proj": {"kernel": jnp.asarray(arrays["in_proj.weight"]).T},
        "conv1d": {"weight": jnp.asarray(arrays["conv1d.weight"]),
                   "bias": jnp.asarray(arrays["conv1d.bias"])},
        "A_log": jnp.asarray(arrays["A_log"]), "dt_bias": jnp.asarray(arrays["dt_bias"]),
        "D": jnp.asarray(arrays["D"]), "norm": {"weight": jnp.asarray(arrays["norm.weight"])},
        "out_proj": {"kernel": jnp.asarray(arrays["out_proj.weight"]).T}}}


def mamba2_layer_run(site, arrays):
    """Dew's mixer on a drawn layer (`mamba2_layer_draw`): 'layer' in the arrays' dtype, 'layer_bf16'
    computing in bfloat16 on fp32 weights and bf16 hidden states, its output in fp32."""
    hidden = jnp.asarray(arrays["hidden"])
    if site == "layer_bf16":
        out = mamba2_layer(jnp.bfloat16).apply(mamba2_layer_params(arrays), hidden.astype(jnp.bfloat16))
        return {"output": out.astype(jnp.float32)}
    return {"output": mamba2_layer(hidden.dtype).apply(mamba2_layer_params(arrays), hidden)}
