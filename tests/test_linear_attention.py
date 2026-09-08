"""Gated DeltaNet against the transformers 5.16.1 reference, at fp32.

tools/qwen_linear_reference.py runs `torch_chunk_gated_delta_rule`,
`torch_recurrent_gated_delta_rule`, `F.conv1d` and one `Qwen3_5GatedDeltaNet`
layer (modeling_qwen3_5.py) on fixed-seed operands and writes what they
produced; the tests here run dew's forms on the same operands and the same
weights and hold them to the reference.

Tolerances and the differences actually observed, fp32 on CPU:

- chunked rule           : output 4.5e-07, state 1.2e-07, tolerance 1e-5
- recurrent rule         : output 8.9e-08, state 6.0e-08, tolerance 1e-5
- carried initial state  : the same numbers for both forms, tolerance 1e-5
- chunked vs recurrent   : output 4.2e-07, state 7.5e-08, tolerance 1e-5
- a sequence cut at 30   : output 4.7e-07, state 5.7e-07, tolerance 1e-5
- causal conv            : 2.4e-07, tolerance 1e-5
- the whole layer        : 6.8e-05 on outputs of magnitude up to 9.9 over
  70 tokens (key heads 2, value heads 4, key dim 12, value dim 16),
  tolerance 5e-4; its decode path 3.4e-05 against its parallel forward

The two bugs the port hit on the way are the mutations these tests are known
to catch, each measured by applying it and reading the assertion: an
inclusive instead of a strictly lower triangle in the in-chunk operator (the
reference's masked_fill zeroes the diagonal) and reading the memory as `S q`
instead of `q^T S` in the recurrent form; the numbers are in the tests that
catch them.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn

from dew.nn.inputs import AttentionMetadata
from dew.nn.linear import (
    GatedDeltaNet, _masked_conv1d, causal_conv1d, chunk_gated_delta_rule, l2norm,
    recurrent_gated_delta_rule,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "linear_attention"


@pytest.fixture(scope="module")
def reference():
    return dict(np.load(FIXTURES / "gated_delta_net.npz"))


@pytest.fixture(scope="module")
def geometry():
    return json.loads((FIXTURES / "config.json").read_text())


def operands(reference, normalised: bool = True):
    """The rule's operands as the module hands them to the rule: the reference
    normalises q and k inside the rule (use_qk_l2norm_in_kernel=True), dew
    normalises them before it."""
    query, key = reference["rule.query"], reference["rule.key"]
    if normalised:
        query, key = l2norm(query), l2norm(key)
    return (jnp.asarray(query), jnp.asarray(key), jnp.asarray(reference["rule.value"]),
            jnp.asarray(reference["rule.g"]), jnp.asarray(reference["rule.beta"]))


def largest(left, right) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def test_the_chunked_rule_matches_the_reference(reference):
    """`torch_chunk_gated_delta_rule` on 70 tokens: a full chunk and a padded
    one, so both the in-chunk operator and the cross-chunk memory are read.
    Largest observed difference 4.5e-07 on the output, 1.2e-07 on the state.
    With the in-chunk triangle made inclusive (`tril(..., 0)` in place of
    `tril(..., -1)`) the output moves by 1.5e-01."""
    out, state = chunk_gated_delta_rule(*operands(reference))

    assert largest(out, reference["chunk.output"]) < 1e-5
    assert largest(state, reference["chunk.state"]) < 1e-5


def test_the_recurrent_rule_matches_the_reference(reference):
    """`torch_recurrent_gated_delta_rule`, one token at a time. Largest
    observed difference 8.9e-08 on the output, 6.0e-08 on the state. With
    the memory read as `S q` (q contracted against the value axis) instead
    of `q^T S` the output moves by 4.7e-01."""
    out, state = recurrent_gated_delta_rule(*operands(reference))

    assert largest(out, reference["recurrent.output"]) < 1e-5
    assert largest(state, reference["recurrent.state"]) < 1e-5


def test_the_chunked_and_recurrent_forms_agree(reference):
    """The two forms are one computation: a model trains with the first and
    decodes with the second, so they have to produce the same numbers and
    leave the same memory behind. Largest observed difference 4.2e-07 on
    the output, 7.5e-08 on the state."""
    chunked, chunked_state = chunk_gated_delta_rule(*operands(reference))
    recurrent, recurrent_state = recurrent_gated_delta_rule(*operands(reference))

    assert largest(chunked, recurrent) < 1e-5
    assert largest(chunked_state, recurrent_state) < 1e-5


def test_strong_decay_has_the_same_finite_gradient_in_both_forms():
    """Masked positive log differences must never reach exp in the chunk rule.

    At g=-4 a 64-token chunk has finite outputs but exp of its unused upper
    triangle overflows; masking afterwards leaves a 0*inf NaN in backward.
    The recurrent form stays finite and supplies the behavioral reference.
    """
    query = jnp.asarray(np.random.default_rng(99).normal(size=(1, 64, 1, 4)) * .1, jnp.float32)
    key, value = query * .7, query * .3
    beta, decay = jnp.full((1, 64, 1), .2), jnp.full((1, 64, 1), -4.)

    def differentiate(rule):
        return jax.jit(jax.value_and_grad(lambda g: rule(query, key, value, g, beta)[0].sum()))(decay)

    expected, expected_grad = differentiate(recurrent_gated_delta_rule)
    actual, actual_grad = differentiate(chunk_gated_delta_rule)
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-5)
    np.testing.assert_allclose(actual_grad, expected_grad, atol=1e-7, rtol=1e-5)



def test_an_initial_state_is_carried_by_both_forms(reference):
    """Starting from a memory, as every decode step past the first does,
    matches the reference started from the same memory: 4.5e-07 chunked,
    8.9e-08 recurrent on the output."""
    initial = jnp.asarray(reference["rule.initial_state"])

    out, state = chunk_gated_delta_rule(*operands(reference), initial)
    assert largest(out, reference["chunk_carried.output"]) < 1e-5
    assert largest(state, reference["chunk_carried.state"]) < 1e-5

    out, state = recurrent_gated_delta_rule(*operands(reference), initial)
    assert largest(out, reference["recurrent_carried.output"]) < 1e-5
    assert largest(state, reference["recurrent_carried.state"]) < 1e-5


def test_a_sequence_split_in_two_carries_its_memory_across_the_cut(reference):
    """The property a prefill followed by a continuation depends on: the
    state the first 30 tokens leave, fed to the last 40, gives the numbers
    the whole 70 give at once. The cut is inside a chunk, so it is the
    carried state and the padding that are tested, not a chunk boundary.
    Largest observed difference 4.7e-07 on the output, 5.7e-07 on the
    state."""
    query, key, value, g, beta = operands(reference)
    whole, whole_state = chunk_gated_delta_rule(query, key, value, g, beta)

    cut = 30
    head, state = chunk_gated_delta_rule(
        query[:, :cut], key[:, :cut], value[:, :cut], g[:, :cut], beta[:, :cut])
    tail, final = chunk_gated_delta_rule(
        query[:, cut:], key[:, cut:], value[:, cut:], g[:, cut:], beta[:, cut:], state)

    assert largest(jnp.concatenate([head, tail], axis=1), whole) < 1e-5
    assert largest(final, whole_state) < 1e-5


def test_the_causal_conv_matches_f_conv1d(reference):
    """`F.conv1d(padding=K - 1, groups=D)[..., :S]` then silu, the reference's
    short mixer (modeling_qwen3_next.py:345-365): position s reads s-K+1..s.
    Largest observed difference 2.4e-07."""
    taps = jnp.asarray(reference["conv.weight"][:, 0, :])
    out = causal_conv1d(jnp.asarray(reference["conv.input"]), taps)

    assert largest(out, reference["conv.output"]) < 1e-5


def test_the_conv_reads_no_future_column(reference):
    """Causality of the short mixer, stated as a property: changing the last
    column moves nothing before it."""
    x = jnp.asarray(reference["conv.input"])
    taps = jnp.asarray(reference["conv.weight"][:, 0, :])
    moved = x.at[:, :, -1].set(x[:, :, -1] + 3.0)

    assert jnp.array_equal(causal_conv1d(x, taps)[..., :-1], causal_conv1d(moved, taps)[..., :-1])
    assert not jnp.allclose(causal_conv1d(x, taps)[..., -1], causal_conv1d(moved, taps)[..., -1])


def layer_params(reference):
    """The reference layer's state_dict under dew's names: Linear weights
    transposed to [in, out], everything else as it stands."""
    leaf = lambda name: jnp.asarray(reference[f"layer.{name}"])
    return {"params": {
        "in_proj_qkv": {"kernel": leaf("in_proj_qkv.weight").T},
        "in_proj_z": {"kernel": leaf("in_proj_z.weight").T},
        "in_proj_b": {"kernel": leaf("in_proj_b.weight").T},
        "in_proj_a": {"kernel": leaf("in_proj_a.weight").T},
        "out_proj": {"kernel": leaf("out_proj.weight").T},
        "conv1d": {"weight": leaf("conv1d.weight")},
        "norm": {"weight": leaf("norm.weight")},
        "A_log": leaf("A_log"),
        "dt_bias": leaf("dt_bias"),
    }}


def layer(geometry) -> GatedDeltaNet:
    return GatedDeltaNet(
        emb_features=geometry["hidden_size"],
        num_k_heads=geometry["linear_num_key_heads"],
        num_v_heads=geometry["linear_num_value_heads"],
        head_k_dim=geometry["linear_key_head_dim"],
        head_v_dim=geometry["linear_value_head_dim"],
        conv_kernel=geometry["linear_conv_kernel_dim"],
        norm_eps=geometry["rms_norm_eps"])


def test_the_layer_matches_qwen3_5_gated_delta_net(reference, geometry):
    """The whole mixer on the reference's own weights: projections, the
    conv, the gates, the rule with value heads outnumbering key heads, the
    gated norm and the output projection. Largest observed difference
    6.8e-05 over 70 tokens, on outputs of magnitude up to 9.9; the bare
    rule above holds to 1e-5, so the residue is fp32 accumulation through
    the projections and the norm."""
    module = layer(geometry)
    variables = layer_params(reference)
    template = jax.eval_shape(module.init, jax.random.key(0), jnp.zeros((1, 4, geometry["hidden_size"])))
    assert jax.tree.map(jnp.shape, variables) == jax.tree.map(jnp.shape, template)

    out = module.apply(variables, jnp.asarray(reference["layer.hidden"]))

    assert largest(out, reference["layer.output"]) < 5e-4


def test_the_layer_decodes_as_it_prefills(reference, geometry):
    """A prefill against the cache and single-token steps after it reproduce
    the parallel forward: the conv tail and the recurrent memory both have
    to cross the step boundary. Largest observed difference 3.4e-05 against
    the parallel forward on the same 70 tokens."""
    module = layer(geometry)
    variables = layer_params(reference)
    hidden = jnp.asarray(reference["layer.hidden"])
    parallel = module.apply(variables, hidden)

    cache = module.apply(variables, hidden[:, :1], decode=True, mutable=["cache"])[1]["cache"]
    prefill = 5
    out, mutated = module.apply({**variables, "cache": cache}, hidden[:, :prefill],
                                decode=True, mutable=["cache"])
    steps = [out]
    for position in range(prefill, hidden.shape[1]):
        out, mutated = module.apply({**variables, **mutated}, hidden[:, position:position + 1],
                                    decode=True, mutable=["cache"])
        steps.append(out)

    assert largest(jnp.concatenate(steps, axis=1), parallel) < 5e-4


def scan_masked_conv1d(x, kernel, valid, state=None):
    """The token-by-token form the compacted convolution replaced.

    One window per physical slot, the history advancing only on a real
    token. It stays here as the reference the compacted form reproduces.
    """
    batch, channels, _ = x.shape
    width = kernel.shape[1] - 1
    if state is None:
        state = jnp.zeros((batch, channels, width), x.dtype)

    def step(history, inputs):
        token, active = inputs
        window = jnp.concatenate([history, token[:, :, None]], axis=-1)
        output = nn.silu(jnp.sum(window * kernel[None, :, :], axis=-1))
        history = jnp.where(active[:, None, None], window[:, :, 1:], history)
        return history, jnp.where(active[:, None], output, 0.0)

    state, output = jax.lax.scan(step, state, (jnp.moveaxis(x, 2, 0), valid.T))
    return jnp.moveaxis(output, 0, 2), state


def masked_operands(rows, kernel_width, *, channels=3, history=True, seed=0):
    """Input, taps, initial history, validity and a cotangent for one case."""
    valid = np.asarray(rows, bool)
    batch, length = valid.shape
    keys = jax.random.split(jax.random.key(seed), 4)
    x = jax.random.normal(keys[0], (batch, channels, length), jnp.float32)
    kernel = jax.random.normal(keys[1], (channels, kernel_width), jnp.float32)
    state = (jax.random.normal(keys[2], (batch, channels, kernel_width - 1), jnp.float32)
             if history else None)
    cotangent = jax.random.normal(keys[3], (batch, channels, length), jnp.float32)
    return x, kernel, state, jnp.asarray(valid), cotangent


MASKED_ROWS = {
    "right padding": ([[1] * 5 + [0] * 3, [1] * 3 + [0] * 5], 4),
    "left padding": ([[0] * 3 + [1] * 5, [0] * 1 + [1] * 7], 4),
    "interior holes": ([[1, 1, 0, 1, 1, 0, 0, 1, 1], [1, 1, 1, 0, 1, 1, 1, 0, 1]], 4),
    "alternating slots": ([[1, 0] * 6, [0, 1] * 6], 4),
    "a paused row beside a full one": ([[0] * 7, [1] * 7], 4),
    "kernel 2": ([[1, 0, 1, 1, 0, 1, 1]], 2),
    "kernel 8": ([[1, 1, 0, 1, 1, 1, 0, 1, 1, 1]], 8),
    "the kernel's own length": ([[1, 0, 1, 1]], 4),
    "one token past a chunk": ([[1] * 60 + [0] * 3 + [1] * 2], 4),
}


@pytest.mark.parametrize("case", list(MASKED_ROWS))
@pytest.mark.parametrize("history", [False, True])
def test_the_masked_conv_matches_the_token_scan(case, history):
    """The compacted convolution is the scan it replaced: the same outputs,
    the same history to carry, and the same gradients into the input, the
    taps and the incoming history.

    Largest difference observed over these cases, fp32 on CPU: 4.8e-07,
    tolerance 1e-5, which is the tolerance the rest of this file holds
    fp32 agreement to.
    """
    x, kernel, state, valid, cotangent = masked_operands(*MASKED_ROWS[case], history=history)

    def scored(implementation):
        def inner(x, kernel, state):
            out, carried = implementation(x, kernel, valid, state)
            return jnp.sum(out * cotangent) + jnp.sum(carried ** 2), (out, carried)
        return jax.value_and_grad(inner, argnums=(0, 1, 2), has_aux=True)

    (_, (out, carried)), grads = scored(_masked_conv1d)(x, kernel, state)
    (_, (want_out, want_carried)), want_grads = scored(scan_masked_conv1d)(x, kernel, state)

    assert largest(out, want_out) < 1e-5
    assert largest(carried, want_carried) < 1e-5
    for gradient, wanted in zip(grads, want_grads):
        if wanted is not None:
            assert largest(gradient, wanted) < 1e-5


def test_the_masked_conv_reads_across_a_gap():
    """A real token reads the real tokens before it, not the padding between
    them, so the row's outputs are the plain convolution of the row with its
    padding taken out.

    The mutation this states the invariant against is the tempting one:
    masking the input and convolving the physical slots. It agrees on
    trailing padding and moves the slots after an interior gap by 5.0 here.
    """
    rows, width = MASKED_ROWS["interior holes"]
    x, kernel, state, valid, _ = masked_operands(rows, width, history=False, seed=3)
    out, _ = _masked_conv1d(x, kernel, valid)

    for row in range(x.shape[0]):
        kept = np.flatnonzero(np.asarray(valid[row]))
        compacted = causal_conv1d(x[row:row + 1, :, kept], kernel)
        assert largest(out[row:row + 1, :, kept], compacted) < 1e-5

    physical = jnp.where(valid[:, None, :], causal_conv1d(
        jnp.where(valid[:, None, :], x, 0), kernel), 0)
    assert largest(physical, out) > 1.0


@pytest.mark.parametrize("case", list(MASKED_ROWS))
def test_a_padded_slot_carries_no_output_and_no_gradient(case):
    """Padding neither emits nor learns: its outputs are zero, its inputs
    take exactly zero gradient, and a row with no real token at all hands
    back the history it was given, unchanged."""
    x, kernel, state, valid, cotangent = masked_operands(*MASKED_ROWS[case], seed=7)
    out, carried = _masked_conv1d(x, kernel, valid, state)
    grad = jax.grad(lambda value: jnp.sum(
        _masked_conv1d(value, kernel, valid, state)[0] * cotangent))(x)

    assert jnp.array_equal(jnp.where(valid[:, None, :], 0.0, out), jnp.zeros_like(out))
    assert jnp.array_equal(jnp.where(valid[:, None, :], 0.0, grad), jnp.zeros_like(grad))
    paused = ~jnp.any(valid, axis=1)
    assert state is not None
    assert jnp.array_equal(carried[paused], state[paused])


def test_the_masked_conv_splits_at_a_call_boundary():
    """A prefill and a continuation see one stream of real tokens: the
    history the first call leaves is what makes the second one agree with a
    single call over the whole row, with the cut falling inside a gap."""
    rows, width = MASKED_ROWS["interior holes"]
    x, kernel, state, valid, _ = masked_operands(rows, width, seed=11)
    whole, carried = _masked_conv1d(x, kernel, valid, state)

    cut = 6
    head, history = _masked_conv1d(x[:, :, :cut], kernel, valid[:, :cut], state)
    tail, final = _masked_conv1d(x[:, :, cut:], kernel, valid[:, cut:], history)

    assert largest(jnp.concatenate([head, tail], axis=2), whole) < 1e-5
    assert largest(final, carried) < 1e-5


def test_the_layer_reads_a_holed_row_as_the_row_without_its_holes(reference, geometry):
    """The mixer's own contract, through its public call: the states of the
    real tokens of a row with padded slots between them are the states of
    those tokens on their own. Observed difference 0.0, and the padded slots
    themselves come back zero."""
    module = layer(geometry)
    variables = layer_params(reference)
    hidden = jnp.asarray(reference["layer.hidden"])[:, :6]
    kept = jnp.asarray([0, 1, 4, 5, 7, 8])
    spread = jnp.zeros((hidden.shape[0], 9, hidden.shape[2]), hidden.dtype).at[:, kept].set(hidden)
    spread = spread.at[:, jnp.asarray([2, 3, 6])].set(
        jax.random.normal(jax.random.key(5), (hidden.shape[0], 3, hidden.shape[2])))
    valid = jnp.zeros((hidden.shape[0], 9), bool).at[:, kept].set(True)

    alone = module.apply(variables, hidden)
    holed = module.apply(variables, spread,
                         attention_metadata=AttentionMetadata(valid=valid))

    assert largest(holed[:, kept], alone) < 1e-5
    assert jnp.array_equal(holed[:, jnp.asarray([2, 3, 6])],
                           jnp.zeros_like(holed[:, jnp.asarray([2, 3, 6])]))
