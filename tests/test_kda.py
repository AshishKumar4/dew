"""Kimi Delta Attention against Transformers' published rules and their JAX transcription.

`oracle_chunk` is `chunk_kimi_delta_attention` (modeling_glm5_next.py:482-578)
transcribed into JAX at any precision, the sequential row correction loop
included, and `oracle_recurrent` is `recurrent_kimi_delta_attention`
(modeling_glm5_next.py:428-478), both in tools/kda_reference.py; `oracle_layer`
composes `Glm5NextTextLinearAttention.forward` (modeling_glm5_next.py:628-733)
over them with the forget gate (319-335) and the gated norm (346-358).
tools/kda_reference.py also executes the pinned upstream torch functions on CPU
over check 1's seeds, in float32 and float64, and records each rule's outputs
and gradients; the transcription in float64 is the truth the recording is held
to. Every output and gradient is on par with the upstream rules' or the
transcription's at float32 from the float64 transcription, over drawn operands
and inputs (check 1, tests/reference_error.py); a per-head scalar decay in
place of the per-dimension one moves the layer by 1.06.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from model_support import decode_through
from reference_error import (
    assert_as_exact,
    declared_seeds,
    indexed_recording,
    recording,
    widened,
    with_gradients,
)

from dew.nn.inputs import MixerCall
from dew.nn.kda import KimiDeltaAttention
from dew.nn.kernels.recurrence_chunk import chunk_gated_delta_rule, recurrent_delta_rule
from dew.nn.precision import l2norm as jax_l2norm
from tools.kda_reference import (
    RECORDED,
    oracle_chunk,
    oracle_recurrent,
    oracle_rule,
    rule_gradients,
    rule_inputs,
)

B, S, H, D, E, K = 2, 7, 2, 4, 8, 4  # batch, tokens, heads, head dim, model width, conv kernel
SEEDS = {'chunk oracle': 64, 'recurrent oracle': 64, 'safe_gate layer': 32, 'softplus layer': 8,
         'layer gradients': 8, 'decode': 32, 'padded rows': 64}
"""Check 1's declared seeds for each site of the file, sized by `seeds_for` (on armada)."""
ROWS = 64
"""A layer draw's rows: one layer's few outputs a draw leave its error to a handful of values."""
GROUPS = 32
"""The batches a gradient draw holds, each a gradient of its own."""


def oracle_layer(x, p, lower_bound, eps, rule):
    """`Glm5NextTextLinearAttention.forward` in JAX at `x`'s dtype over `rule`, its heads and head
    width those of `p`."""
    batch, length, _ = x.shape
    H, D = p["A_log"].shape[-1], p["o_norm"].shape[-1]
    mixed = jnp.concatenate([x @ p["q_proj"], x @ p["k_proj"], x @ p["v_proj"]], axis=-1)
    taps = jnp.concatenate([p[name][:, 0, :] for name in ("q_conv1d", "k_conv1d", "v_conv1d")])
    padded = jnp.pad(jnp.swapaxes(mixed, 1, 2), ((0, 0), (0, 0), (K - 1, 0)))
    conv = sum(padded[:, :, k:k + length] * taps[None, :, k, None] for k in range(K))
    conv = conv / (1 + jnp.exp(-conv))  # silu, the conv activation (hidden_act)
    q, k, v = (part.reshape(batch, length, H, D) for part in jnp.split(jnp.swapaxes(conv, 1, 2), 3, axis=-1))
    gate = ((x @ p["f_a_proj"]) @ p["f_b_proj"] + p["dt_bias"]).reshape(batch, length, H, D)
    rate = jnp.exp(p["A_log"])[None, None, :, None]
    if lower_bound is not None:
        g = lower_bound / (1 + jnp.exp(-(rate * gate)))
    else:
        g = -rate * jnp.where(gate > 20.0, gate, jnp.log(1.0 + jnp.exp(jnp.minimum(gate, 20.0))))
    beta = 1 / (1 + jnp.exp(-(x @ p["b_proj"])))
    core, _ = rule(q, k, v, g, beta, jnp.zeros((batch, H, D, D), x.dtype))
    out_gate = ((x @ p["g_a_proj"]) @ p["g_b_proj"]).reshape(batch, length, H, D)
    normed = core / jnp.sqrt(jnp.mean(core * core, axis=-1, keepdims=True) + eps) * p["o_norm"]
    normed = normed / (1 + jnp.exp(-out_gate))
    return normed.reshape(batch, length, H * D) @ p["o_proj"]


def oracle_forward(x, p, *, eps):
    """`oracle_layer`'s output with the safe gate over the chunked transcription: the forward whose
    gradients the gradient sites hold (`with_gradients`)."""
    return oracle_layer(x, p, -5.0, eps, partial(oracle_chunk, chunk_size=4))


def oracle_params(variables) -> dict:
    """The layer's parameters as `oracle_layer` takes them, at their own dtype."""
    return {name: (leaf["kernel"] if "kernel" in leaf else leaf["weight"]) if isinstance(leaf, dict) else leaf
            for name, leaf in variables["params"].items()}


def operands(seed: int):
    """A seed's float32 rule operands: decays below zero, gates in (0, 1), a nonzero initial state."""
    rng = np.random.RandomState(seed)
    q, k, v = (rng.randn(B, S, H, D) for _ in range(3))
    g = -np.abs(rng.randn(B, S, H, D)) * 0.7
    beta = 1 / (1 + np.exp(-rng.randn(B, S, H)))
    state = rng.randn(B, H, D, D) * 0.3
    return tuple(np.asarray(x, np.float32) for x in (q, k, v, g, beta, state))


def dew_rule(rule: str):
    """Dew's `rule` as the oracle's signature takes it, its queries and keys normalized first as
    the layer normalizes them."""
    def run(q, k, v, g, beta, state):
        q, k = jax_l2norm(q), jax_l2norm(k)
        if rule == "chunk":
            return chunk_gated_delta_rule(q, k, v, g, beta, state, chunk_size=4)
        return recurrent_delta_rule(q, k, v, g, beta, state)
    return run


@pytest.mark.parametrize("rule", ["chunk", "recurrent"])
def test_the_rule_and_all_operand_gradients_match_the_executed_torch_reference(rule):
    """Each rule's output, final state and six operand gradients on par with the published torch
    rule's in float32 from its float64 run, recorded over the declared seeds (tools/kda_reference.py),
    the truth the JAX transcription in float64, held to the recording's sketch."""
    label = f"{rule} rule"
    seeds = declared_seeds(RECORDED)[label]
    draw, recorded, truth = indexed_recording(recording(RECORDED), label, seeds, inputs_of=rule_inputs,
                                              truth_fn=jax.jit(partial(rule_gradients, oracle_rule(rule))))
    dew = jax.jit(partial(rule_gradients, dew_rule(rule)))
    assert_as_exact(lambda index, inputs: dew(inputs), recorded, draw, label, seeds=seeds, truth_fn=truth)


def test_the_chunked_rule_matches_the_oracle_across_chunks_with_a_carried_state():
    """Two chunks of four with three tokens of padding and a nonzero
    initial state, so the inter-chunk write, the padding and the carried
    memory all take part: on par with the transcription from its float64
    evaluation over drawn operands (check 1)."""
    oracle = jax.jit(partial(oracle_chunk, chunk_size=4))
    assert_as_exact(jax.jit(dew_rule("chunk")), oracle, operands, "chunk oracle", seeds=SEEDS['chunk oracle'])
    # The transcriptions compute at their operands' dtype, float32 ones in float32 whatever x64 says.
    with jax.enable_x64(new_val=True):
        for rule in (oracle, oracle_recurrent):
            assert {value.dtype for value in rule(*operands(0))} == {np.dtype(np.float32)}


def test_the_recurrent_rule_matches_the_oracle_and_the_chunked_rule():
    """The recurrent rule on par with the recurrent transcription from its float64 evaluation over
    drawn operands (check 1); the chunked rule is held to the chunked one, the two transcriptions
    one function."""
    assert_as_exact(jax.jit(dew_rule("recurrent")), jax.jit(oracle_recurrent), operands, "recurrent oracle",
                    seeds=SEEDS['recurrent oracle'])


def layer_and_params(lower_bound, seed=3, *, E=E, H=H, D=D, eps=1e-5):
    """The layer of width `E`, `H` heads of `D` and the gated norm's `eps`, its parameters drawn from
    `seed` and its float32 variables."""
    module = KimiDeltaAttention(emb_features=E, num_heads=H, head_dim=D, conv_kernel=K,
                                lower_bound=lower_bound, chunk_size=4, norm_eps=eps)
    rng = np.random.RandomState(seed)
    def dense(rows, cols):
        return rng.randn(rows, cols) / np.sqrt(rows)
    params = {
        "q_proj": dense(E, H * D), "k_proj": dense(E, H * D), "v_proj": dense(E, H * D),
        **{name: rng.randn(H * D, 1, K) * 0.5 for name in ("q_conv1d", "k_conv1d", "v_conv1d")},
        "f_a_proj": dense(E, D), "f_b_proj": dense(D, H * D) * 3,
        "dt_bias": rng.randn(H * D) * 0.5, "A_log": rng.randn(H) * 0.3,
        "b_proj": dense(E, H), "g_a_proj": dense(E, D), "g_b_proj": dense(D, H * D) * 2,
        "o_norm": 1 + rng.randn(D) * 0.2, "o_proj": dense(H * D, E),
    }
    return module, params, variables_of(jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float32), params))


def variables_of(p: dict) -> dict:
    """The layer's variables holding `oracle_layer`'s parameters `p`, as they are."""
    return {"params": {
        **{name: {"kernel": p[name]} for name in ("q_proj", "k_proj", "v_proj", "f_a_proj", "f_b_proj",
                                                  "b_proj", "g_a_proj", "g_b_proj", "o_proj")},
        **{name: {"weight": p[name]} for name in ("q_conv1d", "k_conv1d", "v_conv1d")},
        "dt_bias": p["dt_bias"], "A_log": p["A_log"], "o_norm": {"weight": p["o_norm"]},
    }}


@pytest.mark.parametrize("lower_bound", [-5.0, None], ids=["safe_gate", "softplus"])
def test_the_layer_matches_the_oracle_in_both_forget_gate_forms(lower_bound):
    """The layer on par with `oracle_layer` over the chunked transcription from its float64
    evaluation over drawn inputs (check 1), on the layer's own float32 parameters."""
    module, _, variables = layer_and_params(lower_bound)
    layer = jax.jit(module.apply)
    oracle = jax.jit(lambda x, p: oracle_layer(x, p, lower_bound, 1e-5, partial(oracle_chunk, chunk_size=4)))
    params = oracle_params(variables)

    def draw(seed):
        return (np.random.RandomState(seed).randn(ROWS, S, E).astype(np.float32), params)

    label = f"{'safe_gate' if lower_bound is not None else 'softplus'} layer"
    assert_as_exact(lambda x, p: layer(variables, x), oracle, draw, label, seeds=SEEDS[label])
    x = jnp.ones((B, S, E), jnp.float32)
    assert module.init(jax.random.key(0), x)["params"].keys() == variables["params"].keys()


def test_the_layer_gradients_match_the_oracle():
    """Into the input and every parameter, through the conv, the forget
    gate, the chunked rule across a chunk boundary and the gated norm: on
    par with the transcription's gradients from their float64 evaluation
    over drawn inputs and cotangents (check 1)."""
    # Four heads of eight on a width of sixteen, and a gated norm of eps 1e-2: through the norm of a near
    # zero row, a gradient's error grows as the row's inverse square, a tail no count of seeds resolves,
    # and rows of eight values come near zero far more rarely than rows of four.
    wide = {"E": 16, "H": 4, "D": 8, "eps": 1e-2}
    module, _, _ = layer_and_params(-5.0, **wide)

    @jax.jit
    @jax.vmap
    def ours(x, p, cotangent):
        _, grad_params, grad_x = with_gradients(module.apply)(variables_of(p), x, cotangent)
        return {"x": grad_x, **oracle_params(grad_params)}

    @jax.jit
    @jax.vmap
    def theirs(x, p, cotangent):
        _, grad_x, grad_p = with_gradients(partial(oracle_forward, eps=1e-2))(x, p, cotangent)
        return {"x": grad_x, **grad_p}

    params = oracle_params(layer_and_params(-5.0, **wide)[2])
    stacked = jax.tree.map(lambda leaf: np.broadcast_to(leaf, (GROUPS, *leaf.shape)), params)

    def draw(seed):
        # GROUPS batches a draw, each its own inputs and cotangent, each gradient a row of its own.
        x, cotangent = np.random.RandomState(seed).randn(2, GROUPS, 16, S, wide["E"]).astype(np.float32)
        return x, stacked, cotangent

    assert_as_exact(ours, theirs, draw, "layer gradients", seeds=SEEDS['layer gradients'])


BOUND = 1e-4
"""The existing bound (`test_default_eps_gradients_hold_their_prior_bound`), its gradients'
largest error over their largest value."""


def scaled(actual, wanted) -> float:
    return float(np.max(np.abs(np.asarray(actual, np.float64) - wanted)) / np.max(np.abs(wanted)))


def test_default_eps_gradients_hold_their_prior_bound():
    """An existing bound (ledger row 1348): at the gated norm's own eps 1e-5, a gradient through the norm
    of a near-zero row errs as the row's inverse square, a tail check 1 cannot resolve, so the layer's
    gradients there keep the prior bound, their largest error within BOUND of their largest value from
    the transcription's float64 gradients (its autodiff, where central differences were), until the
    zero-element certificate (tests/zero_elements.py) judges those rows."""
    module, _, variables = layer_and_params(-5.0)
    x = jnp.asarray(np.random.RandomState(4).randn(B, S, E), jnp.float32)
    cotangent = jnp.asarray(np.random.RandomState(5).randn(B, S, E), jnp.float32)

    _, grad_params, grad_x = with_gradients(module.apply)(variables, x, cotangent)
    ours = {"x": grad_x, **oracle_params(grad_params)}
    with jax.enable_x64(new_val=True):
        _, wanted_x, wanted = with_gradients(partial(oracle_forward, eps=1e-5))(
            *widened((x, oracle_params(variables), cotangent)))
        wanted = {"x": wanted_x, **wanted}
    for name, value in ours.items():
        assert scaled(value, np.asarray(wanted[name])) < BOUND, name


def test_prefill_then_token_steps_reproduce_the_parallel_layer():
    """The decode cache: a prefill of three tokens, then one token at a time
    through the conv and recurrent states, on par with the layer over the
    whole sequence from the transcription's float64 evaluation over drawn
    inputs (check 1); the allocation call leaves no state behind."""
    module, _, variables = layer_and_params(-5.0)
    params = oracle_params(variables)

    def draw(seed):
        return (np.random.RandomState(seed).randn(ROWS, S, E).astype(np.float32),)

    _, allocated, _ = decode_through(module, variables, jnp.asarray(draw(0)[0]), prefill=3)
    assert not np.asarray(allocated["recurrent_state"]).any()
    assert_as_exact(lambda x: decode_through(module, variables, x, prefill=3)[0], jax.jit(
        lambda x: module.apply(variables, x)), draw, "decode", seeds=SEEDS['decode'],
        truth_fn=lambda x: oracle_layer(x, jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64), params),
                                        -5.0, 1e-5, partial(oracle_chunk, chunk_size=4)))


def test_padded_rows_preserve_the_memory_and_the_history():
    """A padded slot writes nothing: the layer over a packed row with a gap
    is on par, on the real tokens, with the layer over the same tokens
    contiguous, from the transcription's float64 evaluation over drawn
    inputs (check 1)."""
    from dew.batch import coordinates
    from dew.nn.inputs import AttentionMetadata

    module, _, variables = layer_and_params(-5.0)
    params = oracle_params(variables)
    valid = jnp.tile(jnp.array([[True, True, False, True, True, False]]), (ROWS, 1))
    real = jnp.array([0, 1, 3, 4])

    def draw(seed):
        return (np.random.RandomState(seed).randn(ROWS, 6, E).astype(np.float32),)

    def gapped(x):
        out = module.apply(variables, x, MixerCall(AttentionMetadata(coordinates=coordinates(valid=valid))))
        return out[:, real]

    assert_as_exact(jax.jit(gapped), jax.jit(lambda x: module.apply(variables, x[:, real])), draw,
                    "padded rows", seeds=SEEDS['padded rows'],
                    truth_fn=lambda x: oracle_layer(x[:, real], jax.tree.map(
                        lambda leaf: jnp.asarray(leaf, jnp.float64), params), -5.0, 1e-5,
                        partial(oracle_chunk, chunk_size=4)))
