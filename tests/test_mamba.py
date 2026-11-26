"""Mamba's selective scan and a `MambaForCausalLM` checkpoint against
transformers 5.16.1: tools/mamba_reference.py writes tests/fixtures/hf/mamba-tiny."""

import functools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from forks_support import LAYOUTS, assert_forks_as_own_rows
from model_support import as_stepped, decode_through, float64_twin
from reference_error import FEWEST_SEEDS, assert_as_exact, with_gradients
from selective_support import (
    DIFFERENTIATED,
    FAINT_BESIDE_THE_LARGEST,
    NEAR_THE_LARGEST,
    RATES_PAST_THE_RANGE,
    faint_beside_the_largest,
    near_the_largest,
    rate_past_the_range,
    recurred,
    second_beside_the_largest,
    starts_near_the_largest,
    stepwise,
)
from sequence_split_support import SPLIT_ROWS, assert_split_as_whole

from dew.interop import Pretrained
from dew.nn.mixers.mamba import Mamba, selective_scan
from dew.nn.ssm import UNROLLED, affine_states

ROOT = Path(__file__).parent / 'fixtures' / 'hf'
TINY = ROOT / 'mamba-tiny'


def load():
    return Pretrained.load(TINY, dtype='float32')


def operands(length=11, channels=16, state=4, seed=0):
    keys = jax.random.split(jax.random.key(seed), 6)
    return (jax.random.normal(keys[0], (2, length, channels)),
            jax.nn.softplus(jax.random.normal(keys[1], (2, length, channels))),
            -jnp.exp(jax.random.normal(keys[2], (channels, state))),
            jax.random.normal(keys[3], (2, length, state)), jax.random.normal(keys[4], (2, length, state)),
            jax.random.normal(keys[5], (channels,)))


@pytest.mark.parametrize('chunk', [1, 4, 64])
def test_the_chunked_scan_is_the_recurrence_from_any_state(chunk):
    """Each chunk size, one that divides the sequence and one that pads it,
    computes the token-by-token recurrence and leaves its state."""
    x, dt, A, B, C, D = operands()
    held = jax.random.normal(jax.random.key(7), (2, 16, 4))
    out, final = selective_scan(x, dt, A, B, C, D, state=held, chunk_size=chunk)
    expected, last = stepwise(x, dt, A, B, C, D, held)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(final, last, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('chunk', [16, 64])
@pytest.mark.parametrize('case', list(NEAR_THE_LARGEST))
def test_the_chunked_scan_near_float32s_largest_is_as_exact_as_the_recurrence(case, chunk):
    """The chunked scan near float32's largest (`near_the_largest`), its output and final state, and every
    input's gradient where a float32 recurrence holds them (`DIFFERENTIATED`), on par from float64 with the
    recurrence a token at a time (`stepwise`, check 1, which refuses a value not finite where float64's is):
    its spans summed the writes from a zero state past float32's largest, inf (row 8465), its cotangents'
    spans did so too, and a halving of them lost a faint write (row 8486)."""
    starts = starts_near_the_largest(case)

    def scan(x, dt, A, B, C, D, state):
        return selective_scan(x, dt, A, B, C, D, state=state, starts=starts, chunk_size=chunk)

    def reference(x, dt, A, B, C, D, state):
        return stepwise(x, dt, A, B, C, D, state, starts)

    def draw(seed):
        *inputs, cotangents = near_the_largest(case, seed)
        return (*inputs, cotangents) if case in DIFFERENTIATED else tuple(inputs)

    arm = with_gradients if case in DIFFERENTIATED else (lambda fn: fn)
    exact = jax.jit(arm(reference))
    # A gradient's leaves resolve over more draws than the states alone (its skip's at 40 was inconclusive).
    assert_as_exact(jax.jit(arm(scan)), exact, draw,
                    f"the chunked scan near float32's largest, {case}, chunks of {chunk}",
                    seeds=160 if case in DIFFERENTIATED else FEWEST_SEEDS, truth_fn=exact)


@pytest.mark.parametrize('axis', [1, -2])
@pytest.mark.parametrize('case', FAINT_BESIDE_THE_LARGEST)
def test_an_affine_scans_faint_values_beside_float32s_largest_are_as_exact_as_the_recurrence(case, axis):
    """Mamba's affine scan (`ssm.affine_states`) along its tokens, named by a positive or a negative axis,
    over faint values beside sums past float32's largest (`faint_beside_the_largest`): its states, and with
    a faint cotangent every gradient, on par from float64 with the recurrence a token at a time (`recurred`,
    check 1): a recurrence halved where it reached the largest flushed its faint half (row 8486), and its
    transpose reversed a negative axis JAX had not made positive (the head review of 29231f794)."""
    def scan(decay, write, entering):
        return affine_states(decay, write, entering, axis=axis)

    def reference(decay, write, entering):
        return recurred(decay, write, entering, axis=1)

    differentiated = case == "a faint cotangent"

    def draw(seed):
        *inputs, cotangent = faint_beside_the_largest(case, seed)
        return (*inputs, cotangent) if differentiated else tuple(inputs)

    arm = with_gradients if differentiated else (lambda fn: fn)
    exact = jax.jit(arm(reference))
    assert_as_exact(jax.jit(arm(scan)), exact, draw,
                    f"an affine scan beside float32's largest, {case}, along axis {axis}", seeds=FEWEST_SEEDS,
                    truth_fn=exact)


def test_an_affine_scans_second_derivative_beside_float32s_largest_is_as_exact_as_the_recurrence():
    """The decays' gradient of Mamba's affine scan's writes' cotangent along a direction, under a cotangent of
    2^127 beside a reset (`second_beside_the_largest`): on par from float64 with the recurrence's a token at
    a time (`recurred`, check 1), every order of the scan its own ranged recurrence, where autodiff of a
    halving's scaling doubled the cotangent past the largest, NaN (the head review of 29231f794)."""
    def second(fn):
        def run(decay, write, entering, cotangent, direction):
            def along(decay):
                _, pull = jax.vjp(lambda write: fn(decay, write, entering, axis=1), write)
                return jnp.sum(pull(cotangent)[0] * direction)

            return jax.grad(along)(decay)

        return run

    exact = jax.jit(second(recurred))
    assert_as_exact(jax.jit(second(affine_states)), exact, second_beside_the_largest,
                    "an affine scan's second derivative beside float32's largest", seeds=FEWEST_SEEDS,
                    truth_fn=exact)


def decayed(fn):
    """`fn(x, dt, A, B, C, D, state)` as `run(x, dt, A, B, C, D, state, cotangent)`, which returns its output
    and the gradients of the decay's step and rate `(dt, A)` under the cotangent: the operands a decay's form
    reaches."""
    def run(x, dt, A, B, C, D, state, cotangent):
        out, pull = jax.vjp(lambda dt, A: fn(x, dt, A, B, C, D, state), dt, A)
        return out, *pull(cotangent)

    return run


def channelled(fn, leaf):
    """`fn(x, dt, A, B, C, D, state)` as `run(x, dt, A, B, C, D, state, cotangent)`, which returns the
    gradient of `B` or of `C` (`leaf`) under the cotangent alone: each a sum over the channels of its own
    transpose, measured apart (row 8563)."""
    def run(x, dt, A, B, C, D, state, cotangent):
        moving = {"B": B, "C": C}

        def scanned(value):
            given = {**moving, leaf: value}
            return fn(x, dt, A, given["B"], given["C"], D, state)

        _, pull = jax.vjp(scanned, moving[leaf])
        return pull(cotangent)[0]

    return run


@pytest.mark.parametrize('leaf', ['B', 'C'])
@pytest.mark.parametrize('case', RATES_PAST_THE_RANGE)
def test_a_scans_b_and_c_gradients_sum_their_channels_as_the_recurrence(case, leaf):
    """The gradient of B and of C, each apart (`channelled`), of the chunked scan at a rate past float32's
    range (`rate_past_the_range`, 128 channels): on par from float64 with the recurrence a token at a time
    (`stepwise`, check 1), their channel sums paired as its autodiff's. C's: where the readout's transpose
    summed 128 channels in a dot, red here (row 8563). B's: a parity guard on the write's spread, which
    XLA:CPU's reduce of a broadcast's transpose held before too."""
    def scan(x, dt, A, B, C, D, state):
        return selective_scan(x, dt, A, B, C, D, state=state, chunk_size=16)

    exact = jax.jit(channelled(stepwise, leaf))
    assert_as_exact(jax.jit(channelled(scan, leaf)), exact, functools.partial(rate_past_the_range, case),
                    f"the chunked scan's {leaf} gradient at a rate past float32's range, {case}",
                    seeds=FEWEST_SEEDS, truth_fn=exact)


def test_a_wide_cpu_readouts_c_gradient_sums_its_channels_before_it_rounds(monkeypatch):
    """C's gradient of the chunked scan under Dew's wide CPU arithmetic, two channels each holding `2^100`
    in one state through a token that neither decays nor writes, read under cotangents `2^30` and `-2^30`:
    zero, the channels' products summed in float64 and rounded once, where C's cotangent rounded each
    channel's product to float32 first, `inf - inf`, NaN (review of fb050ba39)."""
    from dew.nn.precision import WIDE_CPU_ARITHMETIC

    monkeypatch.setenv(WIDE_CPU_ARITHMETIC, "1")
    # On the CPU, at a precision that states no fp32 sums: where the wide sum runs.
    with jax.default_device(jax.devices("cpu")[0]), jax.default_matmul_precision("highest"):
        x = dt = jnp.zeros((1, 1, 2), jnp.float32)
        A, D = jnp.zeros((2, 2), jnp.float32), jnp.zeros((2,), jnp.float32)
        B = C = jnp.zeros((1, 1, 2), jnp.float32)
        state = jnp.asarray([[[2.0 ** 100, 0.0], [2.0 ** 100, 0.0]]], jnp.float32)
        _, pull = jax.vjp(lambda C: selective_scan(x, dt, A, B, C, D, state=state, chunk_size=1), C)
        (gradient,) = pull((jnp.asarray([[[2.0 ** 30, -(2.0 ** 30)]]], jnp.float32), jnp.zeros_like(state)))
    np.testing.assert_array_equal(np.asarray(gradient), np.zeros((1, 1, 2), np.float32))


@pytest.mark.mesh(devices=2)
def test_a_scan_whose_states_an_explicit_axis_splits_scans_as_one_unsplit():
    """The chunked scan's output and B's and C's gradients with its rate and entering state split over the
    states by an Explicit axis of two devices, C replicated: on par from float64 with the recurrence a token
    at a time (check 1), where the readout's spread C placed as the states whole split both contracted
    operands, which JAX refuses (review of fb050ba39)."""
    from jax.sharding import AxisType, NamedSharding, PartitionSpec

    mesh = jax.make_mesh((2,), ("state",), axis_types=(AxisType.Explicit,), devices=jax.devices()[:2])

    def gradients(fn, placed=False):
        def run(x, dt, A, B, C, D, state, cotangent):
            out, pull = jax.vjp(lambda B, C: fn(x, dt, A, B, C, D, state), B, C)
            if placed:
                # Each cotangent placed as its output is: the final state's split over the states.
                cotangent = jax.tree.map(lambda c, o: jax.sharding.reshard(c, jax.typeof(o).sharding),
                                         cotangent, out)
            return out, *pull(cotangent)
        return run

    def split(x, dt, A, B, C, D, state):
        A = jax.sharding.reshard(A, NamedSharding(mesh, PartitionSpec(None, "state")))
        state = jax.sharding.reshard(state, NamedSharding(mesh, PartitionSpec(None, None, "state")))
        return selective_scan(x, dt, A, B, C, D, state=state, chunk_size=16)

    def draw(seed):
        return rate_past_the_range("vanishing decays", seed)

    exact = jax.jit(gradients(stepwise))
    with jax.set_mesh(mesh):
        assert_as_exact(jax.jit(gradients(split, placed=True)), exact, draw,
                        "the chunked scan with its states split by an Explicit axis", seeds=FEWEST_SEEDS,
                        truth_fn=exact)


def test_a_scan_of_no_channels_takes_zero_b_and_c_gradients():
    """The chunked scan over no channels gives B and C zero gradients of their own shapes, where a channel
    sum that indexed an empty axis raised (row 8563's review)."""
    x = jnp.zeros((1, 8, 0), jnp.float32)
    B = C = jnp.ones((1, 8, 4), jnp.float32)
    A = jnp.zeros((0, 4), jnp.float32)

    def loss(B, C):
        out, final = selective_scan(x, x, A, B, C, jnp.zeros((0,), jnp.float32), chunk_size=4)
        return jnp.sum(out) + jnp.sum(final)

    for gradient in jax.grad(loss, argnums=(0, 1))(B, C):
        assert gradient.shape == (1, 8, 4) and not np.any(np.asarray(gradient))


@pytest.mark.parametrize('case', RATES_PAST_THE_RANGE)
@pytest.mark.parametrize('chunk', [16, 64])
def test_a_rate_past_float32s_range_scans_as_the_recurrence(chunk, case):
    """A scan whose rate `A log2(e)` passes float32's range over a ragged chunk (`rate_past_the_range`): its
    output, final state and its decay's step's and rate's gradients (`decayed`) on par from float64 with the
    recurrence a token at a time (`stepwise`, check 1), where the rate taken to base 2 before the step was
    -inf and a padded token's zero step made its decay NaN (row 8491), and under steps of 2^-126 each decay
    e^-3, which no clamp of that rate gives (row 8609). Its B's and C's gradients there are their own cell's
    (`test_a_scans_b_and_c_gradients_sum_their_channels_as_the_recurrence`)."""
    def scan(x, dt, A, B, C, D, state):
        return selective_scan(x, dt, A, B, C, D, state=state, chunk_size=chunk)

    exact = jax.jit(decayed(stepwise))
    assert_as_exact(jax.jit(decayed(scan)), exact, functools.partial(rate_past_the_range, case),
                    f"the chunked scan at a rate past float32's range, {case}, chunks of {chunk}",
                    seeds=FEWEST_SEEDS, truth_fn=exact)


def test_a_document_start_drops_the_state_entering_it():
    x, dt, A, B, C, D = operands()
    starts = jnp.zeros((2, 11), bool).at[:, 5].set(True)
    out, _ = selective_scan(x, dt, A, B, C, D, starts=starts, chunk_size=4)
    second, _ = selective_scan(x[:, 5:], dt[:, 5:], A, B[:, 5:], C[:, 5:], D, chunk_size=4)
    np.testing.assert_allclose(out[:, 5:], second, rtol=1e-5, atol=1e-5)




DECODE_SEEDS = 8
"""The layer's cached decode's declared check-1 seeds against its parallel forward at each step's width,
sized by `seeds_for` on armada."""


def test_forked_documents_in_one_chunk_read_none_of_each_others_operands():
    """Two one-token documents in one chunk of a forked call (`selective_scan`'s `forked`, walked in place
    a segment a document), each entering a zero memory: the first's infinite write (B = inf) reaches no
    value of the second's, whose output and final state are its own token's, 1, and whose output's
    gradient along its own operands is finite (its token's x, dt, B and C each 1). A segment's other tokens
    take zero operands before any arithmetic (`mamba._forked_row`): its step zero alone left `dt x B` at
    0 * inf, NaN (head review of cadac1879). The first's own operands take its infinite state times a
    zero cotangent in its own segment's pullback, NaN, as the documents laid from chunks' edges took it."""
    from dew.nn.kernels.chunked_launch import Forked

    ones = jnp.ones((1, 2, 1), jnp.float32)
    B = jnp.asarray([[[jnp.inf], [1.0]]], jnp.float32)
    A, D = -jnp.ones((1, 1), jnp.float32), jnp.zeros((1,), jnp.float32)
    firsts, slots = jnp.asarray([[0, 1]], jnp.int32), jnp.zeros((1, 2), jnp.int32)
    forked = Forked(firsts, slots, slots, None, 0)
    given = jnp.zeros((1, 1, 1, 1), jnp.float32)

    def second(x, dt, B, C):
        out, final = selective_scan(x, dt, A, B, C, D, given, forked=forked)
        return out[0, 1, 0], final

    (out, final) = second(ones, ones, B, ones)
    assert float(out) == 1.0 and float(final[0, 0, 0]) == 1.0, (out, final)
    grads = jax.grad(lambda *operands: second(*operands)[0], argnums=(0, 1, 2, 3))(ones, ones, B, ones)
    assert [float(grad[0, 1, 0]) for grad in grads] == [1.0] * 4, grads


@pytest.mark.parametrize('chunk', [0, UNROLLED + 1])
def test_a_chunk_outside_its_straight_line_block_is_refused(chunk):
    """A chunk of no tokens, or past the tokens `ssm.affine_states` runs straight-line (`ssm.UNROLLED`, row
    8796), refused by a Mamba layer as it is built and by its scan (`selective_scan`) called alone."""
    mixer = Mamba(emb_features=8, intermediate_size=16, state_size=4, time_step_rank=2, chunk_size=chunk)
    with pytest.raises(ValueError):
        mixer.init(jax.random.key(0), jnp.zeros((1, 4, 8)))
    x = jnp.zeros((1, 2 * UNROLLED + 2, 4))
    with pytest.raises(ValueError):
        selective_scan(x, x, jnp.zeros((4, 2)), x[..., :2], x[..., :2], jnp.zeros((4,)), chunk_size=chunk)


def test_the_layer_decodes_as_it_prefills():
    """Over drawn layers and inputs, the layer's single-token steps through its cache on par with its
    parallel forward at each step's width (`as_stepped`) from the float64 layer's forward (check 1,
    ledger 2739), its cache holding a state per channel and state dimension."""
    mixer = Mamba(emb_features=8, intermediate_size=16, state_size=4, time_step_rank=2, chunk_size=4)

    def draw(seed):
        key, inputs = jax.random.split(jax.random.key(seed))
        x = jax.random.normal(inputs, (2, 9, 8))
        return mixer.init(key, x), x

    wide = float64_twin(mixer)
    assert_as_exact(lambda variables, x: decode_through(mixer, variables, x, prefill=1)[0],
                    lambda variables, x: as_stepped(lambda held: mixer.apply(variables, held), x, 1),
                    draw, "mamba decode", seeds=DECODE_SEEDS, truth_fn=wide.apply)
    variables, x = draw(0)
    assert decode_through(mixer, variables, x, prefill=1)[2]['ssm_state'].shape == (2, 16, 4)


SPLIT_SEEDS = {"all_valid": 64, "right_padded": 64, "left_padded": 64, "gaps": 64, "packed": 64,
               "empty_shard": 64, "empty_shard_packed": 64}
"""Each row's declared draws."""


@pytest.mark.mesh
@pytest.mark.parametrize("row", sorted(SPLIT_ROWS))
def test_a_layer_split_over_the_sequence_runs_padded_and_packed_rows_as_whole_ones(row):
    """Under a sequence axis of 4, the layer runs rows with padding and packed documents as `held_scan`
    runs them on one device (`assert_split_as_whole`): each shard's conv reads the real tokens before it
    from the earlier shards, and its scan enters with the state they leave it, each channel's and state's
    decay its own (`linear.sequence_entered`, rows 1710, 7423)."""
    mixer = Mamba(emb_features=8, intermediate_size=16, state_size=4, time_step_rank=2, chunk_size=4)
    variables = jax.tree.map(np.asarray, mixer.init(jax.random.key(1), jnp.zeros((1, 32, 8))))
    assert_split_as_whole(mixer, variables, row, SPLIT_SEEDS[row], f"mamba split over the sequence, {row}",
                          width=8)



@pytest.mark.network
def test_state_spaces_mamba_130m_computes_the_references_logits():
    """state-spaces/mamba-130m-hf in fp32 on CPU, in Dew and in
    transformers 5.16.1, on two prompts cut to a common length."""
    import torch
    import transformers
    source = json.loads((ROOT / 'mamba-130m-hf' / 'source.json').read_text())
    tokenizer = transformers.AutoTokenizer.from_pretrained(source['repo'], revision=source['revision'])
    prompts = ["The capital of France is Paris, and the capital of Germany is",
               "def fibonacci(n):\n    if n < 2:\n        return n\n    return"]
    rows = [tokenizer(prompt)["input_ids"] for prompt in prompts]
    ids = np.asarray([row[:min(map(len, rows))] for row in rows], np.int32)
    reference = transformers.MambaForCausalLM.from_pretrained(
        source['repo'], revision=source['revision'], dtype=torch.float32).eval()
    with torch.no_grad():
        expected = reference(torch.from_numpy(ids.astype(np.int64))).logits.numpy()
    loaded = Pretrained.load(source['repo'], revision=source['revision'], dtype='float32')
    logits = np.asarray(loaded.model.apply(loaded.variables, ids), np.float32)
    assert float(np.max(np.abs(logits - expected))) < 1e-3
    assert np.array_equal(logits.argmax(-1), expected.argmax(-1))


@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_documents_that_continue_another_read_its_state_and_history_as_their_own_rows_do(layout):
    """The Mamba layer over forks (`dew.batch.Forks`) gives each document's tokens, and the gradients
    through them, what its own row of its ancestors and itself gives (check 1; `forks_support`): its
    selective scan enters each document's own state (`recurrence_chunk.selective_scan`'s `entered`)."""
    mixer = Mamba(emb_features=8, intermediate_size=16, state_size=4, time_step_rank=2, chunk_size=4)
    variables = jax.jit(mixer.init)(jax.random.key(0), jnp.ones((1, 4, 8), jnp.float32))
    assert_forks_as_own_rows(mixer, variables, 8, layout, "Mamba")

