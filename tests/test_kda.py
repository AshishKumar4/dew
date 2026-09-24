"""Kimi Delta Attention against a float64 NumPy oracle of the reference.

`oracle_chunk` is `chunk_kimi_delta_attention` (modeling_glm5_next.py:482-578)
transcribed into NumPy at float64, the sequential row correction loop
included, and `oracle_recurrent` is `recurrent_kimi_delta_attention`
(modeling_glm5_next.py:428-478); `oracle_layer` composes
`Glm5NextTextLinearAttention.forward` (modeling_glm5_next.py:628-733) over
them with the forget gate (319-335) and the gated norm (346-358). Nothing
here imports torch, so the module is held to the reference's math and not
to another implementation's rounding.

Observed on CPU, fp32 against the oracle: the chunked rule to 8.1e-08
scaled with two chunks of four and a carried state, the recurrent rule to
1.7e-07, the whole layer to 3.1e-07, its gradients to 1.1e-06 scaled against
central differences of the oracle; the bound is 1e-4 on each. A per-head
scalar decay in place of the per-dimension one moves the layer by 1.06.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.nn.kda import (
    KimiDeltaAttention,
    KimiDeltaAttentionMixer,
    chunk_kimi_delta_rule,
    recurrent_kimi_delta_rule,
)
from dew.registry import mixers

BOUND = 1e-4
B, S, H, D, E, K = 2, 7, 2, 4, 8, 4  # batch, tokens, heads, head dim, model width, conv kernel


def l2norm(x, eps=1e-6):
    return x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + eps)


def oracle_recurrent(query, key, value, g, beta, state):
    """modeling_glm5_next.py:428-478, operands `[B, S, H, D]`, `g` `[B, S, H, Dk]`."""
    query, key = l2norm(query), l2norm(key)
    query = query / np.sqrt(query.shape[-1])
    out = np.zeros_like(value)
    s = state.copy()
    for i in range(query.shape[1]):
        s = s * np.exp(g[:, i])[..., None]
        kv_mem = (s * key[:, i][..., None]).sum(-2)
        delta = (value[:, i] - kv_mem) * beta[:, i][..., None]
        s = s + key[:, i][..., None] * delta[..., None, :]
        out[:, i] = (s * query[:, i][..., None]).sum(-2)
    return out, s


def oracle_chunk(query, key, value, g, beta, state, chunk_size):
    """modeling_glm5_next.py:482-578 with its row correction loop."""
    query, key, value, beta, g = (np.swapaxes(x, 1, 2) for x in (query, key, value, beta, g))
    query, key = l2norm(query), l2norm(key)
    batch, heads, length, dk = key.shape
    dv = value.shape[-1]
    scale = 1 / np.sqrt(dk)
    pad = (chunk_size - length % chunk_size) % chunk_size
    total = length + pad
    query = np.pad(query, ((0, 0), (0, 0), (0, pad), (0, 0))) * scale
    key, value, g = (np.pad(x, ((0, 0), (0, 0), (0, pad), (0, 0))) for x in (key, value, g))
    beta = np.pad(beta, ((0, 0), (0, 0), (0, pad)))
    v_beta = value * beta[..., None]
    k_beta = key * beta[..., None]
    query, key, value, g, k_beta, v_beta = (
        x.reshape(batch, heads, -1, chunk_size, x.shape[-1]) for x in (query, key, value, g, k_beta, v_beta))
    g = np.cumsum(g, axis=-2)
    mask = np.triu(np.ones((chunk_size, chunk_size), bool), 0)
    decay_mask = np.exp(g[..., :, None, :] - g[..., None, :, :])
    attn = -(k_beta[..., :, None, :] * key[..., None, :, :] * decay_mask).sum(-1)
    attn[..., mask] = 0
    for i in range(1, chunk_size):
        row = attn[..., i, :i].copy()
        sub = attn[..., :i, :i].copy()
        attn[..., i, :i] = row + (row[..., None] * sub).sum(-2)
    attn = attn + np.eye(chunk_size)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * np.exp(g))
    s = state.copy()
    core = np.zeros_like(value)
    mask = np.triu(np.ones((chunk_size, chunk_size), bool), 1)
    for i in range(total // chunk_size):
        q_i, k_i, v_i, g_i = query[:, :, i], key[:, :, i], value[:, :, i], g[:, :, i]
        attn_inter = (q_i * np.exp(g_i)) @ s
        attn_intra = (q_i[..., :, None, :] * k_i[..., None, :, :] * decay_mask[:, :, i]).sum(-1)
        attn_intra[..., mask] = 0
        v_new = v_i - k_cumdecay[:, :, i] @ s
        core[:, :, i] = attn_inter + attn_intra @ v_new
        s = s * np.exp(g_i[:, :, -1])[..., None] + np.swapaxes(k_i * np.exp(g_i[:, :, -1:] - g_i), -1, -2) @ v_new
    core = core.reshape(batch, heads, -1, dv)[:, :, :length]
    return np.swapaxes(core, 1, 2), s


def oracle_layer(x, p, lower_bound, eps, rule):
    """`Glm5NextTextLinearAttention.forward` at float64 over `rule`."""
    batch, length, _ = x.shape
    mixed = np.concatenate([x @ p["q_proj"], x @ p["k_proj"], x @ p["v_proj"]], axis=-1)
    taps = np.concatenate([p[name][:, 0, :] for name in ("q_conv1d", "k_conv1d", "v_conv1d")])
    padded = np.pad(np.swapaxes(mixed, 1, 2), ((0, 0), (0, 0), (K - 1, 0)))
    conv = sum(padded[:, :, k:k + length] * taps[None, :, k, None] for k in range(K))
    conv = conv / (1 + np.exp(-conv))  # silu, the conv activation (hidden_act)
    q, k, v = (part.reshape(batch, length, H, D) for part in np.split(np.swapaxes(conv, 1, 2), 3, axis=-1))
    gate = ((x @ p["f_a_proj"]) @ p["f_b_proj"] + p["dt_bias"]).reshape(batch, length, H, D)
    rate = np.exp(p["A_log"])[None, None, :, None]
    if lower_bound is not None:
        g = lower_bound / (1 + np.exp(-(rate * gate)))
    else:
        g = -rate * np.where(gate > 20.0, gate, np.log(1.0 + np.exp(np.minimum(gate, 20.0))))
    beta = 1 / (1 + np.exp(-(x @ p["b_proj"])))
    core, _ = rule(q, k, v, g, beta, np.zeros((batch, H, D, D)))
    out_gate = ((x @ p["g_a_proj"]) @ p["g_b_proj"]).reshape(batch, length, H, D)
    normed = core / np.sqrt(np.mean(core * core, axis=-1, keepdims=True) + eps) * p["o_norm"]
    normed = normed / (1 + np.exp(-out_gate))
    return normed.reshape(batch, length, H * D) @ p["o_proj"]


def scaled(actual, wanted) -> float:
    return float(np.max(np.abs(np.asarray(actual, np.float64) - wanted)) / np.max(np.abs(wanted)))


@pytest.fixture
def operands():
    rng = np.random.RandomState(0)
    q, k, v = (rng.randn(B, S, H, D) for _ in range(3))
    g = -np.abs(rng.randn(B, S, H, D)) * 0.7
    beta = 1 / (1 + np.exp(-rng.randn(B, S, H)))
    state = rng.randn(B, H, D, D) * 0.3
    return q, k, v, g, beta, state


def as_f32(*arrays):
    return tuple(jnp.asarray(array, jnp.float32) for array in arrays)


def test_the_chunked_rule_matches_the_oracle_across_chunks_with_a_carried_state(operands):
    """Two chunks of four with three tokens of padding and a nonzero
    initial state, so the inter-chunk write, the padding and the carried
    memory all take part."""
    q, k, v, g, beta, state = operands
    wanted, wanted_state = oracle_chunk(q, k, v, g, beta, state, chunk_size=4)
    ql, kl = jnp.asarray(q, jnp.float32), jnp.asarray(k, jnp.float32)
    from dew.nn.linear import l2norm

    out, final = chunk_kimi_delta_rule(l2norm(ql), l2norm(kl), *as_f32(v, g, beta, state), chunk_size=4)

    assert scaled(out, wanted) < BOUND
    assert scaled(final, wanted_state) < BOUND


def test_the_recurrent_rule_matches_the_oracle_and_the_chunked_rule(operands):
    q, k, v, g, beta, state = operands
    wanted, wanted_state = oracle_recurrent(q, k, v, g, beta, state)
    from dew.nn.linear import l2norm

    ql, kl = l2norm(jnp.asarray(q, jnp.float32)), l2norm(jnp.asarray(k, jnp.float32))
    out, final = recurrent_kimi_delta_rule(ql, kl, *as_f32(v, g, beta, state))
    chunked, chunked_state = chunk_kimi_delta_rule(ql, kl, *as_f32(v, g, beta, state), chunk_size=4)

    assert scaled(out, wanted) < BOUND
    assert scaled(final, wanted_state) < BOUND
    assert scaled(chunked, np.asarray(out)) < BOUND
    assert scaled(chunked_state, np.asarray(final)) < BOUND


def layer_and_params(lower_bound):
    module = KimiDeltaAttention(emb_features=E, num_heads=H, head_dim=D, conv_kernel=K,
                                lower_bound=lower_bound, chunk_size=4, norm_eps=1e-5)
    rng = np.random.RandomState(3)
    dense = lambda rows, cols: rng.randn(rows, cols) / np.sqrt(rows)
    params = {
        "q_proj": dense(E, H * D), "k_proj": dense(E, H * D), "v_proj": dense(E, H * D),
        **{name: rng.randn(H * D, 1, K) * 0.5 for name in ("q_conv1d", "k_conv1d", "v_conv1d")},
        "f_a_proj": dense(E, D), "f_b_proj": dense(D, H * D) * 3,
        "dt_bias": rng.randn(H * D) * 0.5, "A_log": rng.randn(H) * 0.3,
        "b_proj": dense(E, H), "g_a_proj": dense(E, D), "g_b_proj": dense(D, H * D) * 2,
        "o_norm": 1 + rng.randn(D) * 0.2, "o_proj": dense(H * D, E),
    }
    variables = {"params": {
        **{name: {"kernel": jnp.asarray(params[name], jnp.float32)}
           for name in ("q_proj", "k_proj", "v_proj", "f_a_proj", "f_b_proj", "b_proj",
                        "g_a_proj", "g_b_proj", "o_proj")},
        **{name: {"weight": jnp.asarray(params[name], jnp.float32)}
           for name in ("q_conv1d", "k_conv1d", "v_conv1d")},
        "dt_bias": jnp.asarray(params["dt_bias"], jnp.float32),
        "A_log": jnp.asarray(params["A_log"], jnp.float32),
        "o_norm": {"weight": jnp.asarray(params["o_norm"], jnp.float32)},
    }}
    return module, params, variables


@pytest.mark.parametrize("lower_bound", [-5.0, None], ids=["safe_gate", "softplus"])
def test_the_layer_matches_the_oracle_in_both_forget_gate_forms(lower_bound):
    module, params, variables = layer_and_params(lower_bound)
    x = np.random.RandomState(4).randn(B, S, E)
    wanted = oracle_layer(x, params, lower_bound, 1e-5, lambda *a: oracle_chunk(*a, chunk_size=4))

    out = module.apply(variables, jnp.asarray(x, jnp.float32))

    assert scaled(out, wanted) < BOUND
    assert module.init(jax.random.key(0), jnp.asarray(x, jnp.float32))["params"].keys() == variables["params"].keys()


def test_a_per_head_scalar_decay_is_a_different_layer():
    """The decay is a vector over the key dimensions; GDN's per-head scalar
    over the same weights is not KDA, and the oracle separates the two."""
    module, params, variables = layer_and_params(-5.0)
    x = np.random.RandomState(4).randn(B, S, E)

    def scalar_decay(q, k, v, g, beta, state):
        return oracle_recurrent(q, k, v, np.broadcast_to(g.mean(-1, keepdims=True), g.shape), beta, state)

    wanted = oracle_layer(x, params, -5.0, 1e-5, scalar_decay)
    assert scaled(module.apply(variables, jnp.asarray(x, jnp.float32)), wanted) > 1e-2


def test_the_layer_gradients_match_central_differences_of_the_oracle():
    """Into the input and every parameter, through the conv, the forget
    gate, the chunked rule across a chunk boundary and the gated norm."""
    module, params, variables = layer_and_params(-5.0)
    x = np.random.RandomState(4).randn(B, S, E)
    cotangent = np.random.RandomState(5).randn(B, S, E)

    def loss_oracle(x, params):
        return float(np.sum(oracle_layer(x, params, -5.0, 1e-5,
                                         lambda *a: oracle_chunk(*a, chunk_size=4)) * cotangent))

    def loss_module(x, variables):
        return jnp.sum(jnp.asarray(module.apply(variables, x)) * jnp.asarray(cotangent, jnp.float32))

    grad_x, grad_params = jax.grad(loss_module, argnums=(0, 1))(jnp.asarray(x, jnp.float32), variables)
    ours = {"x": grad_x, **{name: (leaf["kernel"] if "kernel" in leaf else leaf["weight"])
                            if isinstance(leaf, dict) else leaf
                            for name, leaf in grad_params["params"].items()}}
    arguments = {"x": x, **params}
    step = 1e-6
    for name, value in arguments.items():
        wanted = np.zeros_like(value)
        for index in np.ndindex(value.shape):
            bumped = {**arguments, name: value.copy()}
            bumped[name][index] += step
            up = loss_oracle(bumped["x"], {key: bumped[key] for key in params})
            bumped[name][index] -= 2 * step
            down = loss_oracle(bumped["x"], {key: bumped[key] for key in params})
            wanted[index] = (up - down) / (2 * step)
        assert scaled(ours[name], wanted) < BOUND, name


def test_prefill_then_token_steps_reproduce_the_parallel_layer():
    """The decode cache: a prefill of three tokens, then one token at a time
    through the conv and recurrent states, equals the layer over the whole
    sequence; the allocation call leaves no state behind."""
    module, _, variables = layer_and_params(-5.0)
    x = jnp.asarray(np.random.RandomState(6).randn(B, S, E), jnp.float32)
    full = module.apply(variables, x)

    _, allocated = module.apply(variables, x[:, :1], decode=True, mutable=["cache"])
    assert float(jnp.abs(jnp.asarray(allocated["cache"]["recurrent_state"])).max()) == 0.0
    out, state = module.apply({**variables, **allocated}, x[:, :3], decode=True, mutable=["cache"])
    steps = [out]
    for position in range(3, S):
        out, state = module.apply({**variables, **state}, x[:, position:position + 1], decode=True, mutable=["cache"])
        steps.append(out)

    assert scaled(jnp.concatenate(steps, axis=1), np.asarray(full)) < BOUND


def test_padded_rows_preserve_the_memory_and_the_history():
    """A padded slot writes nothing: the layer over a packed row with a gap
    equals the layer over the same tokens contiguous, on the real tokens."""
    from dew.nn.inputs import AttentionMetadata

    module, _, variables = layer_and_params(-5.0)
    x = jnp.asarray(np.random.RandomState(7).randn(1, 6, E), jnp.float32)
    valid = jnp.array([[True, True, False, True, True, False]])
    compact = x[:, jnp.array([0, 1, 3, 4])]

    gapped = module.apply(variables, x, attention_metadata=AttentionMetadata(valid=valid))
    wanted = module.apply(variables, compact)

    assert scaled(np.asarray(gapped)[:, [0, 1, 3, 4]], np.asarray(wanted)) < BOUND


def test_the_kind_builds_from_the_configs_fields():
    mixer = mixers.from_record({"kind": "kimi_delta_attention", "linear_num_heads": 3,
                               "linear_head_dim": 8, "linear_conv_kernel_dim": 4, "linear_lower_bound": -5.0})
    assert isinstance(mixer, KimiDeltaAttentionMixer)
    assert mixer.linear_num_heads == 3 and mixer.linear_head_dim == 8
