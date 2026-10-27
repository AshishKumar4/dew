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

import functools
import itertools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from delta_rule_support import fused_step, recurrence, through
from flax import linen as nn
from model_support import decode_through
from reference_error import assert_as_exact, assert_as_exact_as_the_reference, widened
from tuned import taken

from dew.batch import coordinates
from dew.nn.inputs import AttentionMetadata, MixerCall
from dew.nn.kernels import tuning
from dew.nn.kernels.recurrence_algebra import decays, factor
from dew.nn.kernels.recurrence_chunk import chunk_delta_rule, chunk_gated_delta_rule, fla_rule
from dew.nn.linear import (
    GatedDeltaNet,
    _masked_conv1d,
    causal_conv1d,
    held_conv1d,
    l2norm,
    recurrent_gated_delta_rule,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "linear_attention"

SHARDED_SEEDS = 128
"""The per-shard contract's declared draws (check 1, `sharded.assert_runs_per_shard`), sized by
`seeds_for`: the fewest for which one device compared with itself resolves the factor."""


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


def assert_matches(outputs, reference, name: str):
    """A form's output and final state within 1e-5 of the reference's recording `name`."""
    out, state = outputs
    assert largest(out, reference[f"{name}.output"]) < 1e-5
    assert largest(state, reference[f"{name}.state"]) < 1e-5


def decode_operands(rng, rows: int, heads: int, width: int):
    """A decode token's operands from `rng`: unit queries and keys, values, a log decay in (-3, 0], a
    write gate in [0, 1) and a memory of scale 0.1, each fp32."""
    unit = [rng.normal(size=(rows, 1, heads, width)) for _ in range(2)]
    query, key = (jnp.asarray(x / np.linalg.norm(x, axis=-1, keepdims=True), jnp.float32) for x in unit)
    value = jnp.asarray(rng.normal(size=(rows, 1, heads, width)), jnp.float32)
    g = jnp.asarray(-rng.random((rows, 1, heads)) * 3, jnp.float32)
    beta = jnp.asarray(rng.random((rows, 1, heads)), jnp.float32)
    state = jnp.asarray(rng.normal(size=(rows, heads, width, width)) * 0.1, jnp.float32)
    return query, key, value, g, beta, state


def test_the_chunked_rule_matches_the_reference(reference):
    """`torch_chunk_gated_delta_rule` on 70 tokens: a full chunk and a padded
    one, so both the in-chunk operator and the cross-chunk memory are read.
    Largest observed difference 4.5e-07 on the output, 1.2e-07 on the state.
    With the in-chunk triangle made inclusive (`tril(..., 0)` in place of
    `tril(..., -1)`) the output moves by 1.5e-01."""
    out, state = chunk_gated_delta_rule(*operands(reference))

    assert_matches((out, state), reference, "chunk")


def portable(query, key, value, g, beta, state=None):
    """The gated rule on the portable math (`chunk_delta_rule`), the port of
    `torch_chunk_gated_delta_rule` a GPU does not run: it runs fla's kernels,
    held to fla's numbers (tests/test_fla_gdn.py)."""
    return chunk_delta_rule(query, key, value, g[..., None], beta, state)


def test_the_chunked_rule_rounds_as_close_to_float64_as_the_reference(reference):
    """The same 70 tokens against the chunked rule evaluated in float64: Dew's
    fp32 output and final state within twice the RMS distance of
    `torch_chunk_gated_delta_rule`'s own fp32 ones (tests/reference_error.py),
    at fp32 products, as the reference's were. Observed ratios 0.24 (output)
    and 1.05 (state)."""
    rule_inputs = operands(reference)
    with jax.default_matmul_precision("highest"):
        out, state = portable(*rule_inputs)
    cpu = jax.devices("cpu")[0]
    with jax.enable_x64(new_val=True), jax.default_device(cpu):
        truth, truth_state = portable(*(jax.device_put(np.asarray(x, np.float64), cpu) for x in rule_inputs))

    assert_as_exact_as_the_reference(out, reference["chunk.output"], truth, "output")
    assert_as_exact_as_the_reference(state, reference["chunk.state"], truth_state, "state")


def test_the_recurrent_rule_matches_the_reference(reference):
    """`torch_recurrent_gated_delta_rule`, one token at a time. Largest
    observed difference 8.9e-08 on the output, 6.0e-08 on the state. With
    the memory read as `S q` (q contracted against the value axis) instead
    of `q^T S` the output moves by 4.7e-01."""
    out, state = recurrent_gated_delta_rule(*operands(reference))

    assert_matches((out, state), reference, "recurrent")


def test_the_chunked_and_recurrent_forms_agree(reference):
    """The two forms are one computation: a model trains with the first and
    decodes with the second, so they have to produce the same numbers and
    leave the same memory behind. Largest observed difference 4.2e-07 on
    the output, 7.5e-08 on the state."""
    chunked, chunked_state = chunk_gated_delta_rule(*operands(reference))
    recurrent, recurrent_state = recurrent_gated_delta_rule(*operands(reference))

    assert largest(chunked, recurrent) < 1e-5
    assert largest(chunked_state, recurrent_state) < 1e-5


PACKED_RULE_SEEDS = {(1, False): 32, (1, True): 128, (8, False): 16, (8, True): 128}
"""Declared seeds for check 1, sized by `seeds_for` on armada, 2026-10-20: the separate documents'
chunked rule against the float64 recurrence, for each decay width and strength."""


@pytest.mark.parametrize("strong", [False, True], ids=["drawn decay", "strong decay"])
@pytest.mark.parametrize("decay_width", [1, 8], ids=["per head", "per key dimension"])
def test_the_chunked_rule_runs_packed_documents_as_each_runs_alone(decay_width, strong):
    """Packed documents in the chunked rule, one starting on a chunk's first token, one four tokens before a
    chunk's end and one spanning chunks: each document's output, the final memory and every gradient
    on par with the chunked rule on each document alone, both measured from the float64 recurrence.
    The first document reads the row's memory and every other starts from zero. Each draw has fresh
    operands and cotangents, with the key and value dimensions consistently permuted, an exact
    symmetry of the rule. Decay is per head (the gated delta net) or per key dimension (Kimi delta
    attention); at a strong decay (-4 a token) an earlier document sits 96 below the last one's,
    past exp's fp32 range."""
    batch, length, heads, width, chunk = 1, 160, 2, 8, 32
    begins = (0, 32, 60, 110)
    starts = np.zeros((batch, length), bool)
    starts[0, list(begins[1:])] = True

    def draw(seed):
        keys = jax.random.split(jax.random.key(23 + seed), 8)
        q = l2norm(jax.random.normal(keys[0], (batch, length, heads, width)))
        k = l2norm(jax.random.normal(keys[1], (batch, length, heads, width)))
        v = jax.random.normal(keys[2], (batch, length, heads, width))
        g = (jnp.full((batch, length, heads, decay_width), -4.0) if strong else
             -jnp.exp(jax.random.normal(keys[3], (batch, length, heads, decay_width)) - 2.0))
        beta = jax.nn.sigmoid(jax.random.normal(keys[4], (batch, length, heads)))
        h0 = 0.1 * jax.random.normal(keys[5], (batch, heads, width, width))
        do = jax.random.normal(keys[6], v.shape)
        dht = 0.1 * jax.random.normal(keys[7], h0.shape)
        rng = np.random.default_rng(seed)
        key_order, value_order = rng.permutation(width), rng.permutation(width)
        q, k = (x[..., key_order] for x in (q, k))
        v, do = (x[..., value_order] for x in (v, do))
        h0, dht = (x[..., key_order, :][..., value_order] for x in (h0, dht))
        if decay_width > 1:
            g = g[..., key_order]
        return q, k, v, g, beta, h0, do, dht

    def packed(*args):
        return chunk_delta_rule(*args, chunk, starts=jnp.asarray(starts))

    def chunked(*args):
        return chunk_delta_rule(*args, chunk)

    def evaluated(rule, *inputs):
        with jax.default_matmul_precision("highest"):
            return through(rule, *inputs[6:])(*inputs[:6])

    dew = jax.jit(functools.partial(evaluated, packed))
    reference = jax.jit(functools.partial(
        evaluated, functools.partial(recurrence, cuts=(begins,), rule=chunked)))
    truth = jax.jit(functools.partial(evaluated, functools.partial(recurrence, cuts=(begins,))))
    label = f"packed delta rule {decay_width=} {strong=}"
    assert_as_exact(dew, reference, draw, label, seeds=PACKED_RULE_SEEDS[decay_width, strong], truth_fn=truth)


@pytest.mark.parametrize("per_key", [False, True], ids=["gated delta net", "Kimi delta attention"])
def test_the_port_takes_each_packed_tokens_output_and_gradients_as_its_document_alone_does(per_key):
    """The port (`fla_rule`, interpreted off a GPU) runs each packed document as a sequence of
    its own, so each packed token's output and its gradients of the token's query, key, value, decay
    and beta are bitwise those of the same row with every other document's tokens zero: no token's
    value or gradient carries another document's, the decay's rounding included. The same program
    on the same shapes, so any difference is information crossing a document boundary. Documents of
    40, 60, 1, 49, 40, 10 and 56 tokens, one of a single token, resets two apart in a chunk, and on
    and off the chunk grid. The port's arithmetic is held against fla's own packed sequences (test_fla_gdn,
    test_fla_kda); a row of 32 two-token documents once lost the decays of its later ones to an fp32
    prefix that every earlier document's reset had shifted."""
    length, heads, width = 256, 2, 64
    begins = (0, 40, 100, 101, 150, 190, 200, length)
    keys = jax.random.split(jax.random.key(29), 6)
    q = 0.3 * jax.random.normal(keys[0], (1, length, heads, width))
    k = l2norm(jax.random.normal(keys[1], (1, length, heads, width)))
    v = jax.random.normal(keys[2], (1, length, heads, width))
    g = -0.1 * jax.nn.softplus(jax.random.normal(keys[3], (1, length, heads, *(width,) * per_key)))
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (1, length, heads)))
    do = jax.random.normal(keys[5], v.shape)

    def taken(starts):
        def run(*operands):
            *rule, do = operands

            def loss(*rule):
                out = fla_rule(*rule, starts=starts)[0]
                return jnp.sum(out * do), out
            (_, out), grads = jax.value_and_grad(loss, argnums=tuple(range(5)), has_aux=True)(*rule)
            return {"out": out, **dict(zip(("q", "k", "v", "g", "beta"), grads, strict=True))}
        return jax.jit(run)

    run = taken(jnp.asarray(np.isin(np.arange(length), begins[1:-1])[None]))
    packed = run(q, k, v, g, beta, do)
    for begin, end in itertools.pairwise(begins):
        mine = (begin <= np.arange(length)) & (np.arange(length) < end)

        def alone(x, mine=mine):
            return jnp.where(mine.reshape(1, length, *(1,) * (x.ndim - 2)), x, 0)

        found = run(*(alone(x) for x in (q, k, v, g, beta, do)))
        for name, value in packed.items():
            np.testing.assert_array_equal(np.asarray(value)[:, mine], np.asarray(found[name])[:, mine],
                                          err_msg=f"{name} of the document at {begin}")


def test_the_port_keeps_each_row_to_itself_at_its_rows_tightest_document_bound():
    """Two right-padded rows of two packed documents each (segment 0 after them) at the tightest
    bound their coordinates give (`Coordinates.max_segments`, two): each document's output on the
    port (`fla_rule`) is bitwise the document run alone, a row's first from the row's memory.
    A row's padding runs as a sequence of its own, so a bound without it ran the second row's last
    document on from its first."""
    from dew.batch import coordinates

    segments = jnp.asarray([[1] * 70 + [2] * 50 + [0] * 8, [1] * 50 + [2] * 50 + [0] * 28])
    tokens = coordinates(segments=segments, max_segments=2, right_padded=True)
    keys = jax.random.split(jax.random.key(31), 5)
    q, k, v = (0.3 * jax.random.normal(key, (2, 128, 1, 64)) for key in keys[:3])
    g = -0.1 * jax.nn.softplus(jax.random.normal(keys[3], (2, 128, 1)))
    beta = jnp.full((2, 128, 1), 0.5)
    h0 = 0.1 * jax.random.normal(keys[4], (2, 1, 64, 64))
    packed = jax.jit(lambda *a: fla_rule(*a, starts=tokens.starts, max_segments=tokens.max_segments))
    out, _ = packed(q, k, v, g, beta, h0)
    rule = jax.jit(fla_rule)
    for row, cuts in ((0, (0, 70, 120)), (1, (0, 50, 100))):
        for begin, end in itertools.pairwise(cuts):
            alone, _ = rule(*(x[row:row + 1, begin:end] for x in (q, k, v, g, beta)),
                            h0[row:row + 1] if begin == 0 else None)
            np.testing.assert_array_equal(np.asarray(out[row:row + 1, begin:end]), np.asarray(alone),
                                          err_msg=f"row {row} from {begin}")


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
    assert_matches((out, state), reference, "chunk_carried")

    out, state = recurrent_gated_delta_rule(*operands(reference), initial)
    assert_matches((out, state), reference, "recurrent_carried")


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
    # The reference's tensors are channel-major, [B, D, S]; Dew's time-major.
    x = jnp.swapaxes(jnp.asarray(reference["conv.input"]), 1, 2)
    out = causal_conv1d(x, taps)

    assert largest(jnp.swapaxes(out, 1, 2), reference["conv.output"]) < 1e-5
    moved = x.at[:, -1].set(x[:, -1] + 3.0)
    changed = causal_conv1d(moved, taps)
    assert jnp.array_equal(out[:, :-1], changed[:, :-1])
    assert not jnp.allclose(out[:, -1], changed[:, -1])


def layer_params(reference):
    """The reference layer's state_dict under dew's names: Linear weights
    transposed to [in, out], everything else as it stands."""
    def leaf(name):
        return jnp.asarray(reference[f"layer.{name}"])
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

    decoded, _, _ = decode_through(module, variables, hidden, prefill=5)
    assert largest(decoded, parallel) < 5e-4


def test_a_bf16_layer_decodes_its_state_in_bf16_within_its_prefills_error(reference, geometry):
    """A bf16 layer at the default matmul precision holds its decode state in
    bf16, as vLLM does, and runs the rule in fp32. Over 65 single-token steps
    after a prefill its outputs stay within twice the bf16 parallel forward's
    error (which keeps its chunk state in fp32), both measured from the fp32
    forward, whose error is far below bf16's (tests/reference_error.py's
    rule); the conv state, the projections' own bf16 values, is held as they
    are. Asked for the highest precision, as this lane runs, it holds fp32."""
    module = layer(geometry)
    variables = layer_params(reference)
    hidden = jnp.asarray(reference["layer.hidden"])
    truth = np.asarray(module.apply(variables, hidden), np.float64)
    narrow = module.clone(dtype=jnp.bfloat16)
    exact = narrow.apply(variables, hidden[:, :1], MixerCall(decode=True), mutable=["cache"])[1]["cache"]
    assert {leaf.dtype for leaf in jax.tree.leaves(exact)} == {jnp.dtype(jnp.float32)}

    with jax.default_matmul_precision("default"):
        parallel = np.asarray(narrow.apply(variables, hidden), np.float64)
        stepped, cache, _ = decode_through(narrow, variables, hidden, prefill=5)
        assert {leaf.dtype for leaf in jax.tree.leaves(cache)} == {jnp.dtype(jnp.bfloat16)}
    decoded = np.asarray(stepped, np.float64)

    def rms(x):
        return float(np.sqrt(np.mean((x - truth) ** 2)))

    assert rms(decoded) <= 2 * rms(parallel), (rms(decoded), rms(parallel))


def scan_masked_conv1d(x, kernel, valid, state=None):
    """The token-by-token form the compacted convolution replaced.

    One window per physical slot, the history advancing only on a real
    token. It stays here as the reference the compacted form reproduces.
    """
    batch, _, channels = x.shape
    width = kernel.shape[1] - 1
    if state is None:
        state = jnp.zeros((batch, width, channels), x.dtype)

    def step(history, inputs):
        token, active = inputs
        window = jnp.concatenate([history, token[:, None, :]], axis=1)
        output = nn.silu(jnp.sum(window * kernel.T[None], axis=1))
        history = jnp.where(active[:, None, None], window[:, 1:], history)
        return history, jnp.where(active[:, None], output, 0.0)

    state, output = jax.lax.scan(step, state, (jnp.moveaxis(x, 1, 0), valid.T))
    return jnp.moveaxis(output, 0, 1), state


def masked_operands(rows, kernel_width, *, channels=3, history=True, seed=0):
    """Input, taps, initial history, validity and a cotangent for one case."""
    valid = np.asarray(rows, bool)
    batch, length = valid.shape
    keys = jax.random.split(jax.random.key(seed), 4)
    x = jax.random.normal(keys[0], (batch, length, channels), jnp.float32)
    kernel = jax.random.normal(keys[1], (channels, kernel_width), jnp.float32)
    state = (jax.random.normal(keys[2], (batch, kernel_width - 1, channels), jnp.float32)
             if history else None)
    cotangent = jax.random.normal(keys[3], (batch, length, channels), jnp.float32)
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
    for gradient, wanted in zip(grads, want_grads, strict=True):
        if wanted is not None:
            assert largest(gradient, wanted) < 1e-5

    assert jnp.array_equal(jnp.where(valid[:, :, None], 0.0, out), jnp.zeros_like(out))
    assert jnp.array_equal(jnp.where(valid[:, :, None], 0.0, grads[0]), jnp.zeros_like(grads[0]))
    if state is not None:
        paused = ~jnp.any(valid, axis=1)
        assert jnp.array_equal(carried[paused], state[paused])
    if case == "interior holes" and not history:
        rows, width = MASKED_ROWS[case]
        gap_x, gap_kernel, _state, gap_valid, _ = masked_operands(rows, width, history=False, seed=3)
        gap_out, _ = _masked_conv1d(gap_x, gap_kernel, gap_valid)
        for row in range(gap_x.shape[0]):
            kept = np.flatnonzero(np.asarray(gap_valid[row]))
            compacted = causal_conv1d(gap_x[row:row + 1, kept], gap_kernel)
            assert largest(gap_out[row:row + 1, kept], compacted) < 1e-5
        physical = jnp.where(gap_valid[:, :, None], causal_conv1d(
            jnp.where(gap_valid[:, :, None], gap_x, 0), gap_kernel), 0)
        assert largest(physical, gap_out) > 1.0


def test_the_masked_conv_splits_at_a_call_boundary():
    """A prefill and a continuation see one stream of real tokens: the
    history the first call leaves is what makes the second one agree with a
    single call over the whole row, with the cut falling inside a gap."""
    rows, width = MASKED_ROWS["interior holes"]
    x, kernel, state, valid, _ = masked_operands(rows, width, seed=11)
    whole, carried = _masked_conv1d(x, kernel, valid, state)

    cut = 6
    head, history = _masked_conv1d(x[:, :cut], kernel, valid[:, :cut], state)
    tail, final = _masked_conv1d(x[:, cut:], kernel, valid[:, cut:], history)

    assert largest(jnp.concatenate([head, tail], axis=1), whole) < 1e-5
    assert largest(final, carried) < 1e-5


def test_a_packed_prefill_leaves_only_its_last_documents_conv_history():
    """A row packing two documents, the last two tokens long, leaves the history its continuation
    reads: the last document's two tokens and zeros where the first document's would be, so the
    next call's tokens convolve as that document's would on its own."""
    kernel = jnp.asarray(np.random.default_rng(3).normal(size=(6, 4)), jnp.float32)
    x = jnp.asarray(np.random.default_rng(4).normal(size=(1, 7, 6)), jnp.float32)
    segments = jnp.asarray([[1, 1, 1, 1, 1, 2, 2]])
    valid = jnp.ones((1, 7), bool)
    _, history = _masked_conv1d(x, kernel, valid, segments=segments)
    np.testing.assert_array_equal(np.asarray(history[:, 0]), 0)
    np.testing.assert_array_equal(np.asarray(history[:, 1:]), np.asarray(x[:, 5:]))
    following = jnp.asarray(np.random.default_rng(5).normal(size=(1, 3, 6)), jnp.float32)
    continued, _ = _masked_conv1d(following, kernel, jnp.ones((1, 3), bool), history)
    alone, _ = _masked_conv1d(jnp.concatenate([x[:, 5:], following], axis=1), kernel, jnp.ones((1, 5), bool))
    np.testing.assert_allclose(np.asarray(continued), np.asarray(alone[:, 2:]), rtol=0, atol=1e-6)


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
                         MixerCall(AttentionMetadata(coordinates=coordinates(valid=valid))))

    assert largest(holed[:, kept], alone) < 1e-5
    assert jnp.array_equal(holed[:, jnp.asarray([2, 3, 6])],
                           jnp.zeros_like(holed[:, jnp.asarray([2, 3, 6])]))


RIGHT_PADDED = {
    "rows of five and three": [[1] * 5 + [0] * 3, [1] * 3 + [0] * 5],
    "a full row beside an empty one": [[1] * 8, [0] * 8],
    "fewer real tokens than the kernel's history": [[1] * 2 + [0] * 6, [1] + [0] * 7],
}


@pytest.mark.parametrize("case", list(RIGHT_PADDED))
@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_a_right_padded_row_convolves_as_the_masked_conv(case, history, packed):
    """Declared right-padded, a row convolves as it stands, with no gather:
    the masked conv's outputs, padding zeroed, the history it carries and
    the gradients into the input, the taps and the incoming history, with
    its real tokens one document or packed as two, the second from the
    third token on, the padding a document of its own."""
    x, kernel, state, valid, cotangent = masked_operands(RIGHT_PADDED[case], 4, history=history, seed=3)
    segments = (jnp.where(valid, 1 + (jnp.arange(valid.shape[1]) >= 2), 0).astype(jnp.int32) if packed
                else None)

    def scored(*, right_padded):
        def inner(x, kernel, state):
            out, carried = held_conv1d(x, kernel, valid, state, segments=segments, right_padded=right_padded)
            return jnp.sum(out * cotangent) + jnp.sum(carried ** 2), (out, carried)
        return jax.value_and_grad(inner, argnums=(0, 1, 2), has_aux=True)

    (_, (out, carried)), grads = scored(right_padded=True)(x, kernel, state)
    (_, (want_out, want_carried)), want_grads = scored(right_padded=False)(x, kernel, state)

    assert largest(out, want_out) < 1e-6
    assert largest(carried, want_carried) < 1e-6
    for gradient, wanted in zip(grads, want_grads, strict=True):
        if wanted is not None:
            assert largest(gradient, wanted) < 1e-6


def test_a_right_padded_layer_keeps_its_real_tokens_states(reference, geometry):
    """The mixer told its rows are right-padded gives the masked call's
    outputs, zero at the padding: as close
    to float64 as the masked call is (tests/reference_error.py), the two
    being separate programs, which a GPU rounds apart (1.7e-6 on an H100)."""
    module = layer(geometry)
    variables = layer_params(reference)
    hidden = jnp.asarray(reference["layer.hidden"])[:, :9]
    valid = jnp.arange(9)[None, :] < jnp.asarray([[6], [9]])

    def call(metadata, variables=variables):
        return jax.jit(lambda hidden: module.apply(variables, hidden, MixerCall(metadata)))

    masked = AttentionMetadata(coordinates=coordinates(valid=valid))
    declared = AttentionMetadata(coordinates=coordinates(valid=valid, right_padded=True))
    mine, theirs = call(declared)(hidden), call(masked)(hidden)

    assert jnp.array_equal(mine[0, 6:], jnp.zeros_like(hidden[0, 6:]))
    cpu = jax.devices("cpu")[0]
    with jax.enable_x64(new_val=True), jax.default_device(cpu):
        wide = jax.tree.map(lambda x: jax.device_put(np.asarray(x, np.float64), cpu), variables)
        truth = call(AttentionMetadata(coordinates=coordinates(valid=jax.device_put(np.asarray(valid), cpu))),
                     wide)(jax.device_put(np.asarray(hidden, np.float64), cpu))
    assert_as_exact_as_the_reference(mine, theirs, truth, "right-padded output")


@pytest.mark.parametrize("chunk", [4, 16, 64])
def test_the_chunk_inverse_is_exact_where_its_series_cancels(chunk):
    """Every key the same and beta one make A every -1 below the diagonal:
    (I - A)^-1 is I minus the subdiagonal, entries of at most 1, while the
    series' powers of A grow as binomials (to 1.8e17 at 64), which summed
    in fp32 left Qwen3.5-0.8B's fifth delta-rule layer 1e24 off on a
    256-token prompt and its logits NaN."""
    a = -jnp.tril(jnp.ones((2, chunk, chunk), jnp.float32), -1)
    want = np.eye(chunk) - np.eye(chunk, k=-1)
    np.testing.assert_allclose(factor(a), np.broadcast_to(want, a.shape), atol=1e-6)


def test_a_decay_deep_in_a_chunk_keeps_its_own_precision():
    """Each pairwise decay's exponent is a difference of two compensated
    cumulative sums, rounded twice: within 2u of its own range's magnitude,
    plus the compensated sums' own error (u^2 per term of the chunk's
    magnitude) and exp's rounding, however far into the chunk the pair
    sits. A difference of plain cumulative sums carries the rounding of the
    whole accumulated magnitude instead: at g near -2 over 64 positions,
    ~128 u on a decay between neighbours, which the KDA gradient amplified
    to twice the reference's distance from float64 (tests/test_kimi_linear.py)."""
    g = np.random.default_rng(7).uniform(-3, -1, (2, 64, 4)).astype(np.float32)
    _, decay = decays(jnp.asarray(g))
    wide = np.asarray(g, np.float64)
    rows, cols = np.tril_indices(64)
    ranges = [wide[:, c + 1:r + 1] for r, c in zip(rows, cols, strict=True)]
    exponent = np.array([part.sum(1) for part in ranges])
    magnitude = np.array([np.abs(part).sum(1) for part in ranges])
    u = np.finfo(np.float32).eps / 2
    got = np.asarray(decay, np.float64)[:, rows, cols].transpose(1, 0, 2)
    kept = got > np.finfo(np.float32).tiny
    error = np.abs(np.log(got[kept]) - exponent[kept])
    chunk = np.broadcast_to(np.abs(wide).sum(1), magnitude.shape)[kept]
    bound = 2 * u * magnitude[kept] + 64 * u * u * chunk + 2 * u
    assert np.all(error <= bound), float(np.max(error / bound))
    above = ~np.tril(np.ones((64, 64), bool))
    assert not np.asarray(decay)[:, above].any()


def test_the_decay_gradient_sums_only_the_pairs_spanning_each_position():
    """The decay's gradient, `sum_{s >= i > t} cotangent[s, t] decay[s, t]` at position i, as exact
    as the sum of those pairs alone: within two sequential sums' rounding of their magnitudes
    against float64, where the prefix difference's gradient summed every pair after i into it
    with opposite signs and cancelled. Forward mode takes the same tangent: its pairing with a
    cotangent is the gradient's."""
    rng = np.random.default_rng(11)
    g = rng.uniform(-3, -1, (2, 64, 4)).astype(np.float32)
    cotangent = rng.standard_normal((2, 64, 64, 4)).astype(np.float32)

    def gradient(g, cotangent):
        _, pull = jax.vjp(lambda g: decays(g)[1], g)
        return pull(cotangent)[0]

    got = np.asarray(gradient(jnp.asarray(g), jnp.asarray(cotangent)), np.float64)
    with jax.enable_x64(new_val=True):
        wide = widened((g, cotangent))
        want = np.asarray(gradient(*wide))
        weighed = np.abs(np.asarray(wide[1] * decays(wide[0])[1]))
    spans = np.array([weighed[:, i:, :i].sum((1, 2)) for i in range(64)]).transpose(1, 0, 2)
    bound = 2 * 64 * (np.finfo(np.float32).eps / 2) * spans
    assert np.all(np.abs(got - want) <= bound), float(np.max(np.abs(got - want) / np.maximum(bound, 1e-30)))
    tangent = rng.standard_normal(g.shape).astype(np.float32)
    _, forward = jax.jvp(lambda g: decays(g)[1], (jnp.asarray(g),), (jnp.asarray(tangent),))
    paired = float(jnp.vdot(forward, cotangent))
    np.testing.assert_allclose(paired, float(np.vdot(tangent, got)), rtol=1e-5)


def test_the_chunked_rule_holds_with_aligned_keys():
    """Keys nearly aligned within a chunk, beta near one and slow decay, the
    case real checkpoints reach: the chunked math still gives the
    token-by-token recurrence's output."""
    rng = np.random.default_rng(0)
    B, S, H, D = 1, 128, 2, 16
    base = rng.normal(size=(1, 1, H, D))
    key = base + 0.05 * rng.normal(size=(B, S, H, D))
    key = key / np.linalg.norm(key, axis=-1, keepdims=True)
    query = rng.normal(size=(B, S, H, D))
    query = query / np.linalg.norm(query, axis=-1, keepdims=True)
    value = rng.normal(size=(B, S, H, D))
    g = -0.01 * rng.random((B, S, H))
    beta = 0.99 + 0.01 * rng.random((B, S, H))
    query, key, value, g, beta = (jnp.asarray(x, jnp.float32) for x in (query, key, value, g, beta))
    want, want_state = recurrent_gated_delta_rule(query, key, value, g, beta)
    got, state = portable(query, key, value, g, beta)
    np.testing.assert_allclose(got, want, atol=2e-4)
    np.testing.assert_allclose(state, want_state, atol=2e-4)


@pytest.mark.parametrize("held", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("rows, heads, width", [
    pytest.param(3, 2, 64, id="small"),
    pytest.param(3, 16, 128, marks=pytest.mark.gpu(reason="the 0.8B's widths are a GPU's"), id="0.8b-3"),
    pytest.param(32, 16, 128, marks=pytest.mark.gpu(reason="the 0.8B's widths are a GPU's"), id="0.8b-32"),
])
def test_the_decode_kernel_steps_as_the_recurrence_does(rows, heads, width, held):
    """One decode token through `dew.nn.kernels.recurrence.step` (one read and one
    write of the state) gives the recurrence's output and state at the
    launch's config: an fp32 state's within the float64 gate of the fp32
    recurrence's, at Qwen3.5-0.8B's widths (16 heads, 128 wide) on a GPU and
    interpreted at a small width. A state held in bf16 is read and stepped in
    fp32 and written back rounded once, stochastically: within one bf16 step
    of the fp32 one."""
    from dew.nn.kernels.recurrence import StepLaunch
    from dew.nn.linear import decode_gated_delta_rule

    rng = np.random.default_rng(0)
    unit = [rng.normal(size=(rows, 1, heads, width)) for _ in range(2)]
    query, key = (jnp.asarray(x / np.linalg.norm(x, axis=-1, keepdims=True), jnp.float32) for x in unit)
    value = jnp.asarray(rng.normal(size=(rows, 1, heads, width)), jnp.float32)
    g = jnp.asarray(-rng.random((rows, 1, heads)) * 3, jnp.float32)
    beta = jnp.asarray(rng.random((rows, 1, heads)), jnp.float32)
    state = jnp.asarray(rng.normal(size=(rows, heads, width, width)) * 0.1, held)
    operands = (query, key, value, g, beta, state.astype(jnp.float32))
    want, want_state = recurrent_gated_delta_rule(*operands)
    with jax.enable_x64(new_val=True):
        truth = [np.asarray(x) for x in recurrent_gated_delta_rule(*widened(operands))]
    [config] = taken(StepLaunch, decode_gated_delta_rule, query, key, value, g, beta, state)
    got, got_state = jax.jit(decode_gated_delta_rule)(query, key, value, g, beta, state)
    assert got_state.dtype == held
    assert_as_exact_as_the_reference(np.asarray(got), np.asarray(want), truth[0], f"{config} out")
    if held == jnp.float32:
        assert_as_exact_as_the_reference(np.asarray(got_state), np.asarray(want_state), truth[1],
                                         f"{config} state")
    else:
        np.testing.assert_allclose(np.asarray(got_state, np.float32), want_state, atol=2e-6, rtol=2 ** -7)


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("axis, split", [("data", 0), ("tensor", 2)])
def test_the_decode_kernel_keeps_the_kernel_contract_on_a_mesh(axis, split):
    """One decode token through `dew.nn.kernels.recurrence.step` over rows split two
    ways, or heads split two ways, idle rows among the drawing ones, runs per
    shard (`sharded.assert_runs_per_shard`): its output and state within the
    float64 gate of one device's, from the recurrence's in float64 (an idle
    row's output zeros and its state as it was), and no operand or state moved
    between devices. Off a GPU the kernel interprets."""
    from jax.sharding import Mesh, PartitionSpec
    from sharded import assert_runs_per_shard

    from dew.nn.linear import decode_gated_delta_rule

    rows, H, D = 4, 4, 32
    active = jnp.asarray([True, False, True, False])

    def draw(seed):
        rng = np.random.default_rng(2 + seed)
        query, key, value, g, beta, state = decode_operands(rng, rows, H, D)
        return query, key, value, g, beta, state, active

    def exact(*operands):
        """The recurrence in float64, an idle row's output zero and its state as it was."""
        out, final = recurrent_gated_delta_rule(*operands[:6])
        drawing = np.asarray(operands[6])[:, None, None, None]
        return np.where(drawing, np.asarray(out), 0.0), np.where(drawing, np.asarray(final), operands[5])

    def spec(rank: int, at: int):
        return PartitionSpec(*(axis if dim == at else None for dim in range(rank)))

    heads_at = {0: 0, 2: 1}[split]
    specs = [spec(4, split)] * 3 + [spec(3, split)] * 2 + [spec(4, heads_at), spec(1, split)]
    mesh = Mesh(np.asarray(jax.devices()[:2]), (axis,))
    # The kernel aliases its state into its output, and serving donates it (`donate`).
    assert_runs_per_shard(decode_gated_delta_rule, draw, specs, mesh, f"delta decode split over {axis}",
                          seeds=SHARDED_SEEDS, truth_fn=exact, donate=(5,))


@pytest.mark.gpu(reason="the kernel runs on CUDA")
def test_the_decode_kernel_leaves_an_idle_row_alone():
    """A serving step holds every slot, drawing or not. A row the step marks
    idle keeps its state bit for bit, unread and unwritten, and outputs
    zeros; the drawing rows step exactly as they do with every row active."""
    from dew.nn.linear import decode_gated_delta_rule

    rng = np.random.default_rng(1)
    rows, H, D = 8, 16, 128
    query, key, value, g, beta, state = decode_operands(rng, rows, H, D)
    active = jnp.asarray([True, False, True, True, False, False, True, False])
    every, every_state = jax.jit(decode_gated_delta_rule)(query, key, value, g, beta, state)
    got, got_state = jax.jit(decode_gated_delta_rule)(query, key, value, g, beta, state, active)
    drawing = np.asarray(active)
    np.testing.assert_array_equal(np.asarray(got)[drawing], np.asarray(every)[drawing])
    np.testing.assert_array_equal(np.asarray(got_state)[drawing], np.asarray(every_state)[drawing])
    np.testing.assert_array_equal(np.asarray(got)[~drawing], 0)
    np.testing.assert_array_equal(np.asarray(got_state)[~drawing], np.asarray(state)[~drawing])


def fused_case(heads, width, rows, dtype, held):
    """A gated delta net of `heads` value heads over half as many key heads, `width` wide, and its
    cache after a 5-token prefill of `rows` rows in `dtype` with its recurrent state `held`, and the
    next token: the layer's own conv, gates, update and gated norm unfused, and the decode token's
    inputs as the fused step takes them (`fused_step`)."""
    layer = GatedDeltaNet(emb_features=64, num_k_heads=heads // 2, num_v_heads=heads, head_k_dim=width,
                          head_v_dim=width, dtype=dtype)
    hidden = jax.random.normal(jax.random.key(7), (rows, 6, 64), jnp.float32).astype(dtype)
    variables = layer.init(jax.random.key(8), hidden[:, :1])
    variables = {**variables, "params": {**variables["params"], "A_log": jnp.linspace(-1.0, 1.0, heads),
                                         "dt_bias": jnp.linspace(0.5, 1.5, heads)}}
    step = MixerCall(decode=True)
    allocated = layer.apply(variables, hidden[:, :1], step, mutable=["cache"])[1]
    cache = layer.apply({**variables, **allocated}, hidden[:, :5], step, mutable=["cache"])[1]["cache"]
    cache = {**cache, "recurrent_state": cache["recurrent_state"].astype(held)}
    return layer, variables, cache, hidden[:, 5:6]


def fused_inputs(layer, variables, cache, token):
    """The fused step's operands for `token` over `cache`, from the layer's own projections, as the
    layer hands them over on one device (`_fused_token`)."""
    mixed, z, b, a = layer.apply(variables, token, method=GatedDeltaNet._project)
    params = variables["params"]
    return (mixed[:, 0], cache["conv_state"], params["conv1d"]["weight"][:, 0, :], a[:, 0], b[:, 0],
            params["A_log"], params["dt_bias"], z[:, 0].reshape(z.shape[0], layer.num_v_heads, -1),
            params["norm"]["weight"], cache["recurrent_state"])


@pytest.mark.parametrize("held", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("heads, width, rows", [
    pytest.param(4, 32, 3, id="small"),
    pytest.param(16, 128, 4, marks=pytest.mark.gpu(reason="the 0.8B's widths are a GPU's"), id="0.8b"),
])
def test_the_fused_decode_step_computes_the_layers_token(heads, width, rows, held):
    """A gated delta net's decode token in one launch (`fused_step`: the conv over the
    held history, the gates, the update and the gated norm) at its config: its output
    through the layer's projection, its history and its state, fp32 or bf16, within
    tests/reference_error.py's rule of the layer's own unfused token, holding the same state, against
    the float64 layer's. Compiled on a GPU, interpreted elsewhere."""
    from dew.nn.kernels.recurrence import StepLaunch

    layer, variables, cache, token = fused_case(heads, width, rows, jnp.float32, held)
    step = MixerCall(decode=True)
    fp32 = {**cache, "recurrent_state": cache["recurrent_state"].astype(jnp.float32)}
    want, after = layer.apply({**variables, "cache": fp32}, token, step, mutable=["cache"])
    unfused = layer.apply({**variables, "cache": cache}, token, step, mutable=["cache"])[1]["cache"]
    with jax.enable_x64(new_val=True):
        wide = layer.clone(dtype=jnp.float64)
        exact, exact_after = wide.apply({**widened(variables), "cache": widened(fp32)}, widened(token),
                                        step, mutable=["cache"])

    def fused(*operands):
        out, history, state = fused_step(*operands, epsilon=layer.norm_eps,
                                                    activation=layer.gate_activation, dtype=jnp.float32)
        flat = out.reshape(out.shape[0], 1, -1)
        projected = layer.apply(variables, flat, method=lambda module, out: module.out_proj(out))
        return projected, history, state

    operands = fused_inputs(layer, variables, cache, token)
    [config] = taken(StepLaunch, fused, *operands)
    out, history, state = jax.jit(fused)(*operands)
    assert_as_exact_as_the_reference(np.asarray(out), np.asarray(want), np.asarray(exact), f"{config} out")
    np.testing.assert_array_equal(np.asarray(history), np.asarray(after["cache"]["conv_state"]))
    assert state.dtype == held
    assert_as_exact_as_the_reference(np.asarray(state), np.asarray(unfused["recurrent_state"]),
                                     np.asarray(exact_after["cache"]["recurrent_state"]), f"{config} state")


def test_a_cached_layer_traced_for_a_jvp_decodes_through_the_recurrence():
    """A cached gated delta net on an A100, whose row's step kernel decodes its tokens, traced for
    `jax.jvp` (`forward_mode_attention`, as MeanFlow and consistency training trace it), decodes its
    token through the differentiable recurrence, which gives the tangent the in-place kernels do not,
    and reports that route with its stable code; the fused step chosen for such a token
    (`generation.chosen`) raises, naming it (`selection.KernelRefused`)."""
    from dew.nn.attention import forward_mode_attention
    from dew.nn.kernels.generation import assumed, chosen, routes
    from dew.nn.kernels.selection import KernelRefused

    layer, variables, cache, token = fused_case(4, 32, 2, jnp.float32, jnp.float32)

    def step(token):
        return layer.apply({**variables, "cache": cache}, token, MixerCall(decode=True), mutable=["cache"])[0]

    def traced():
        tangent = jax.random.normal(jax.random.key(9), token.shape)
        jax.make_jaxpr(lambda token, tangent: jax.jvp(step, (token,), (tangent,)))(token, tangent)

    with assumed("sm80"), forward_mode_attention(), routes() as found:
        traced()
    (decode,) = [route for route in found if route.choice == "delta_decode"]
    assert (decode.kernel, decode.preferred, decode.refusal.code) == ("xla", "step", "forward_mode")
    with assumed("sm80"), chosen(delta_decode="fused"), forward_mode_attention(), pytest.raises(
            KernelRefused) as refused:
        traced()
    assert [refusal.code for refusal in refused.value.refusals] == ["forward_mode"]


def test_the_fused_decode_step_leaves_an_idle_row_alone():
    """A row the fused step marks idle: its history and state as they were, bitwise, and its output
    zeros, beside an active row that moves."""

    layer, variables, cache, token = fused_case(4, 32, 2, jnp.float32, jnp.float32)
    operands = fused_inputs(layer, variables, cache, token)
    out, history, state = jax.jit(functools.partial(
        fused_step, epsilon=layer.norm_eps, activation=layer.gate_activation,
        dtype=jnp.float32, active=jnp.asarray([False, True])))(*operands)
    np.testing.assert_array_equal(np.asarray(out[0]), 0)
    np.testing.assert_array_equal(np.asarray(history[0]), np.asarray(cache["conv_state"][0]))
    np.testing.assert_array_equal(np.asarray(state[0]), np.asarray(cache["recurrent_state"][0]))
    assert np.any(np.asarray(state[1]) != np.asarray(cache["recurrent_state"][1]))
    assert np.any(np.asarray(history[1]) != np.asarray(cache["conv_state"][1]))


@pytest.mark.gpu(reason="the fused step runs on CUDA")
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_a_layer_choosing_the_fused_step_decodes_as_the_unfused_one(dtype):
    """A gated delta net of 4 value heads over 2 key heads 32 wide decoding 65 tokens after a 5-token
    prefill with its generation's choice 'delta_decode' set to 'fused' (`generation.chosen`): the
    decode tokens run the fused step, and the outputs are within tests/reference_error.py's rule of
    the unfused layer's against the float64 parallel forward."""
    from dew.nn.kernels.generation import chosen
    from dew.training._lowering import triton_kernels

    module = GatedDeltaNet(emb_features=64, num_k_heads=2, num_v_heads=4, head_k_dim=32, head_v_dim=32,
                           dtype=dtype)
    hidden = jax.random.normal(jax.random.key(11), (2, 70, 64), jnp.float32)
    variables = module.init(jax.random.key(12), hidden[:, :1])
    variables = {**variables, "params": {**variables["params"], "A_log": jnp.linspace(-1.0, 1.0, 4),
                                         "dt_bias": jnp.linspace(0.5, 1.5, 4)}}
    with jax.enable_x64(new_val=True):
        truth = np.asarray(module.clone(dtype=jnp.float64).apply(widened(variables), widened(hidden)),
                           np.float64)
    hidden = hidden.astype(dtype)
    unfused = np.asarray(decode_through(module, variables, hidden, prefill=5)[0], np.float64)
    with chosen(delta_decode="fused"):
        fused = np.asarray(decode_through(module, variables, hidden, prefill=5)[0], np.float64)
        _, _, cache = decode_through(module, variables, hidden[:, :6], prefill=5)
        step = jax.jit(lambda cache, token: module.apply({**variables, "cache": cache}, token,
                                                         MixerCall(decode=True), mutable=["cache"]))
        assert "recurrence_step" in triton_kernels(step.trace(cache, hidden[:, 6:7]).lower())
    assert_as_exact_as_the_reference(fused, unfused, truth, f"{jnp.dtype(dtype).name} decode")


def test_a_layer_choosing_the_fused_step_allocates_its_cache_as_the_unfused_one():
    """A gated delta net choosing the fused step, as on an A100, initialized straight into decode mode:
    the allocation-only call allocates the caches the unfused layer does, reading nothing the fused
    step needs before the first token."""
    from dew.nn.kernels.generation import assumed, chosen

    module = GatedDeltaNet(emb_features=64, num_k_heads=2, num_v_heads=4, head_k_dim=32, head_v_dim=32,
                           dtype=jnp.bfloat16)
    x = jnp.zeros((2, 1, 64), jnp.bfloat16)
    unfused = jax.eval_shape(lambda: module.init(jax.random.key(0), x, MixerCall(decode=True)))
    with assumed("sm80"), chosen(delta_decode="fused"):
        fused = jax.eval_shape(lambda: module.init(jax.random.key(0), x, MixerCall(decode=True)))
    def described(tree):
        return jax.tree.map(lambda leaf: (leaf.shape, leaf.dtype), tree)

    assert described(fused) == described(unfused)


@pytest.mark.gpu(reason="the fused step runs on CUDA")
@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("axis", ["data", "tensor"])
def test_a_layer_choosing_the_fused_step_decodes_on_a_mesh_as_on_one_device(axis):
    """The fused step over rows or heads split two ways (`_fused_token`: each shard's channels its
    heads' queries, keys and values): 9 decode tokens after a 3-token prefill within
    tests/reference_error.py's rule of the same layer on one device against the float64 parallel
    forward."""
    from jax.sharding import NamedSharding, PartitionSpec

    from dew.nn.kernels.generation import chosen
    from dew.training import MeshSpec

    module = GatedDeltaNet(emb_features=64, num_k_heads=2, num_v_heads=4, head_k_dim=32, head_v_dim=32,
                           dtype=jnp.float32)
    hidden = jax.random.normal(jax.random.key(13), (2, 12, 64), jnp.float32)
    variables = module.init(jax.random.key(14), hidden[:, :1])
    with jax.enable_x64(new_val=True):
        truth = np.asarray(module.clone(dtype=jnp.float64).apply(widened(variables), widened(hidden)))
    with chosen(delta_decode="fused"):
        alone = np.asarray(decode_through(module, variables, hidden, prefill=3)[0])
        # The data axis takes the devices the others leave.
        mesh = MeshSpec(**({} if axis == 'data' else {axis: 2})).build(jax.devices()[:2])
        with jax.set_mesh(mesh):
            placed = jax.device_put(hidden, NamedSharding(mesh, PartitionSpec()))
            split = np.asarray(decode_through(module, variables, placed, prefill=3)[0])
    assert_as_exact_as_the_reference(split, alone, truth, f"{axis} decode")


@pytest.mark.parametrize("target", ["sm80", "sm90", "sm120"])
def test_the_fused_step_compiles_on_tritons_compiler_within_each_gpus_block(target):
    """The fused decode step at Qwen3.5-0.8B's widths (16 value heads over 16 key heads of 128, a
    conv of 4 taps over an fp32 history, a bf16 state) and at Qwen3.5-9B's value heads over half as
    many key heads, at every config of its space, lowered for each GPU on this CPU and compiled by
    Triton within its block's shared memory."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_kernels, failures

    from dew.nn.kernels.recurrence import Launch, StepLaunch

    def case(key_heads, value_heads, warps):
        rows, width = 8, 128
        channels = (2 * key_heads + value_heads) * width
        specs = (jax.ShapeDtypeStruct((rows, channels), jnp.bfloat16),
                 jax.ShapeDtypeStruct((rows, 3, channels), jnp.float32),
                 jax.ShapeDtypeStruct((channels, 4), jnp.bfloat16),
                 *(jax.ShapeDtypeStruct((rows, value_heads), jnp.bfloat16),) * 2,
                 *(jax.ShapeDtypeStruct((value_heads,), jnp.float32),) * 2,
                 jax.ShapeDtypeStruct((rows, value_heads, width), jnp.bfloat16),
                 jax.ShapeDtypeStruct((width,), jnp.float32),
                 jax.ShapeDtypeStruct((rows, value_heads, width, width), jnp.bfloat16))

        def run(*operands):
            with tuning.pinned({StepLaunch: Launch(0, warps)}):
                return fused_step(*operands, epsilon=1e-6, activation="silu", dtype=jnp.bfloat16)
        return run, specs

    cases = [KernelCase(f"fused/{key}x{value}/warps{warps}",
                        lambda key=key, value=value, warps=warps: case(key, value, warps))
             for key, value in ((16, 16), (16, 32)) for warps in (4, 8, 16)]
    found = compile_kernels(cases, target)
    assert len(found) == len(cases)
    assert not failures(found), "\n".join(failures(found))


def test_a_decode_token_reports_the_fused_step_as_its_route_where_its_target_takes_it():
    """A gated delta net's decode token traced for an A100 on this CPU, the generation's choice
    'delta_decode' set to 'fused', reports the fused step as its route; with no choice, the step
    kernel, the selector's preference (no row read), and at widths the step kernel's blocks do not take the
    portable recurrence with the refusal's code, where the fused step chosen raises, naming its own
    (`selection.KernelRefused`); and with the rule's math chosen, the portable recurrence, so an
    unfused reference decodes unfused."""
    from dew.nn.kernels.generation import assumed, chosen, routes
    from dew.nn.kernels.selection import KernelRefused

    def decode_routes(width):
        module = GatedDeltaNet(emb_features=64, num_k_heads=2, num_v_heads=4, head_k_dim=width,
                               head_v_dim=width, dtype=jnp.bfloat16)
        x = jnp.zeros((2, 1, 64), jnp.bfloat16)
        step = MixerCall(decode=True)
        with assumed("sm80"):
            params = jax.eval_shape(lambda: module.init(jax.random.key(0), x))
            cache = jax.eval_shape(lambda v: module.apply(v, x, step, mutable=["cache"])[1], params)
            with routes() as found:
                jax.eval_shape(lambda v: module.apply(v, x, step, mutable=["cache"]), {**params, **cache})
        return [(found_route.kernel, found_route.preferred, found_route.refusal and found_route.refusal.code)
                for found_route in found if found_route.choice == "delta_decode"]

    assert decode_routes(32)[-1] == ("step", "step", None)
    assert decode_routes(24)[-1] == ("xla", "step", "widths")
    with chosen(gated_delta_rule="xla"):
        assert decode_routes(32)[-1] == ("xla", "xla", None)
    with chosen(delta_decode="fused"):
        assert decode_routes(32)[-1] == ("fused", "fused", None)
        with pytest.raises(KernelRefused) as refused:
            decode_routes(24)
    assert [refusal.code for refusal in refused.value.refusals] == ["widths"]


@pytest.mark.gpu(reason="launch timing and occupancy vary on a GPU")
def test_the_fused_decode_step_gives_the_same_bits_however_its_launches_are_timed():
    """The fused decode step at its config, run 20 times, each run beside a large matmul
    that holds most of the GPU or alone, gives the same output, history and state bits every run: no
    value head reads a history or state another program is writing, whatever order they run in."""
    from dew.nn.kernels.recurrence import StepLaunch

    layer, variables, cache, token = fused_case(16, 128, 4, jnp.bfloat16, jnp.bfloat16)
    operands = fused_inputs(layer, variables, cache, token)
    load = jnp.ones((8192, 8192), jnp.bfloat16)
    busy = jax.jit(lambda a: a @ a)

    def fused(*operands):
        return fused_step(*operands, epsilon=layer.norm_eps, activation=layer.gate_activation,
                                     dtype=jnp.bfloat16)

    [config] = taken(StepLaunch, fused, *operands)
    run = jax.jit(fused)
    first = jax.tree.map(np.asarray, run(*operands))
    for repeat in range(20):
        if repeat % 2:
            busy(load)
        for mine, theirs in zip(jax.tree.leaves(run(*operands)), jax.tree.leaves(first), strict=True):
            np.testing.assert_array_equal(np.asarray(mine), theirs, err_msg=f"{config}, run {repeat}")
