"""The seam between a template's written component and its generated, checked body (`dew.algebra.generated`,
docs/design/algebra.md §6.11): under the default choice the written math runs and nothing is counted; under
"generated" a declared case runs the generated body, bit for bit its module's, and any other call the written
math, its reason counted."""

import generated_matrix
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import FEWEST_SEEDS, assert_as_exact

from dew.algebra import attention_cases_derived, generated, recurrence_cases_derived
from dew.nn.kernels import generation, recurrence_algebra
from dew.nn.kernels.templates import attention_algebra

DECAY = "dew.algebra.recurrence_cases.selective_decay"
MERGE = "dew.algebra.attention_cases.split_kv_merge"
READ = "dew.algebra.recurrence_cases.selective_read"


def _decay_operands(size: int):
    rng = np.random.default_rng(0)
    return (rng.uniform(0, 2, size).astype(np.float32), -rng.uniform(0, 2, size).astype(np.float32))


def _states(rows: int, width: int):
    rng = np.random.default_rng(1)

    def state():
        return attention_algebra.Partial(rng.standard_normal((rows, width)).astype(np.float32),
                                         rng.standard_normal(rows).astype(np.float32),
                                         (np.abs(rng.standard_normal(rows)) + 1).astype(np.float32))
    return state(), state()


def test_the_written_math_runs_and_nothing_is_counted_by_default():
    step, rate = _decay_operands(128)
    with generated.taken() as counts:
        assert generated.body(DECAY, (step, rate)) is None
        written = recurrence_algebra.selective_decay(step, rate)
    assert not counts
    with generation.chosen(generated="written"), generation.assumed("sm90"):
        np.testing.assert_array_equal(recurrence_algebra.selective_decay(step, rate), written)


def test_a_declared_case_runs_its_generated_body_and_is_counted():
    step, rate = _decay_operands(128)
    first, second = _states(32, 64)
    with generation.chosen(generated="generated"), generation.assumed("sm90"), generated.taken() as counts:
        decay = jax.jit(recurrence_algebra.selective_decay)(step, rate)
        merged = attention_algebra.merge(attention_algebra.GENERATED_MERGE, first, second)
    assert counts == {(DECAY, generated.GENERATED): 1, (MERGE, generated.GENERATED): 1}
    np.testing.assert_array_equal(decay, recurrence_cases_derived.selective_decay(step, rate))
    for found, expected in zip(merged[:3], attention_cases_derived.split_kv_merge(*first[:3], *second[:3]),
                               strict=True):
        np.testing.assert_array_equal(found, expected)


def test_an_undeclared_call_runs_its_written_math_and_counts_why():
    step, rate = _decay_operands(96)
    first, second = _states(32, 64)
    natural = attention_algebra.Forward("full", 2, ids=False, scale=0.5, natural=True)
    written = (recurrence_algebra.selective_decay(step, rate),
               attention_algebra.merge(natural, first, second))
    with generation.chosen(generated="generated"), generation.assumed("sm90"), generated.taken() as counts:
        decay = recurrence_algebra.selective_decay(step, rate)
        merged = attention_algebra.merge(natural, first, second)
    assert {outcome for (_, outcome) in counts} >= {"another static program"}
    assert [name for name, outcome in counts if outcome.startswith("no declared case")] == [DECAY]
    assert generated.GENERATED not in {outcome for (_, outcome) in counts}
    np.testing.assert_array_equal(decay, written[0])
    for found, expected in zip(jax.tree.leaves(merged), jax.tree.leaves(written[1]), strict=True):
        np.testing.assert_array_equal(found, expected)


def test_a_compensated_or_scaled_state_keeps_the_written_merge():
    first, second = _states(32, 64)
    scaled = first._replace(scale=attention_algebra.Scale(jnp.zeros(32), jnp.ones(32)))
    with generation.chosen(generated="generated"), generation.assumed("sm90"), generated.taken() as counts:
        attention_algebra.merge(attention_algebra.GENERATED_MERGE, scaled, second)
    assert counts == {(MERGE, "another static program"): 1}


def test_the_generated_merge_takes_an_empty_slice_as_its_identity():
    """A slice of no keys (a decode split's empty tail: its maximum -inf, its sums zero) merged either side of
    a state gives that state back, and two give an empty state, finite: the generated body floors each
    maximum as the written merge does, so no -inf - -inf reaches its exponentials."""
    state, _ = _states(32, 64)
    empty = attention_algebra.Partial(np.zeros((32, 64), np.float32), np.full(32, -np.inf, np.float32),
                                      np.zeros(32, np.float32))
    spec = attention_algebra.GENERATED_MERGE
    with generation.chosen(generated="generated"), generation.assumed("sm90"), generated.taken() as counts:
        merges = [attention_algebra.merge(spec, empty, state), attention_algebra.merge(spec, state, empty),
                  attention_algebra.merge(spec, empty, empty)]
    assert counts == {(MERGE, generated.GENERATED): 3}
    for merged in merges[:2]:
        np.testing.assert_array_equal(merged.acc, state.acc)
        np.testing.assert_array_equal(merged.m, state.m)
        np.testing.assert_array_equal(merged.total, state.total)
    assert np.all(np.isfinite(merges[2].acc)) and not np.any(merges[2].acc)
    assert not np.any(merges[2].total)


@pytest.mark.parametrize("case", generated_matrix.CASES, ids="-".join)
def test_each_generated_site_is_on_par_with_its_written_body(case):
    """The generated bodies' matrix cells (tests/generated_matrix.py) on the CPU, as their generation."""
    with generation.assumed(case[0]):
        generated_matrix.a_generated_body_is_on_par(case)


def _interpreted_scan(kernel: str):
    """The selective template interpreted as an sm90 runs it, the generated choice `kernel`."""
    from dew.nn.kernels.templates import bundles
    from dew.nn.kernels.templates.recurrence import selective

    def scan(*values):
        with generation.assumed("sm90"), generation.chosen(generated=kernel):
            return selective.selective_scan(*values, family=bundles.of("sm90", jnp.bfloat16), interpret=True)
    return scan


def _scan_draw(seed):
    rng = np.random.default_rng([6110, seed])

    def normal(*shape):
        return rng.standard_normal(shape).astype(np.float32)

    rows, length, channels, states = 1, 32, 128, 16
    return (normal(rows, length, channels).astype(jnp.bfloat16),
            np.logaddexp(0.0, normal(rows, length, channels) - 3.0).astype(jnp.bfloat16),
            -np.exp(0.5 * normal(channels, states)), normal(rows, length, states).astype(jnp.bfloat16),
            normal(rows, length, states).astype(jnp.bfloat16), normal(channels))


def _scan_truth(*values):
    from dew.nn.mixers.mamba import selective_scan

    with jax.enable_x64(new_val=True):
        return selective_scan(*(jnp.asarray(value, jnp.float64) for value in values))


def test_the_interpreted_selective_scan_reads_its_states_by_the_generated_body():
    """The selective template interpreted as an sm90 runs it, under the generated choice: its readout reaches
    the declared case at the scan's own operands (a block's lane of each of 16 states, C a scalar a state,
    `recurrence_cases.selective_read`) and its decays theirs, and runs the generated bodies, on par with the
    written choice from the float64 scan (check 1)."""
    with generated.taken() as counts:
        assert_as_exact(jax.jit(_interpreted_scan("generated")), jax.jit(_interpreted_scan("written")),
                        _scan_draw, "the interpreted selective scan, generated", seeds=FEWEST_SEEDS,
                        truth_fn=_scan_truth)
    assert counts[READ, generated.GENERATED] >= 1, counts
    assert counts[DECAY, generated.GENERATED] >= 1, counts


def test_a_python_number_operand_runs_the_written_math_and_counts_why():
    """A Python number where the declared case holds a float32 scalar (`kept` of the decay) keys no generated
    body: the written decay runs, the refusal counted, nothing raised."""
    step, rate = _decay_operands(128)
    with generation.chosen(generated="generated"), generation.assumed("sm90"), generated.taken() as counts:
        decayed = recurrence_algebra.selective_decay(step, rate, 0.0)
    np.testing.assert_array_equal(decayed, np.zeros(128, np.float32))
    ((name, outcome),) = counts
    assert name == DECAY and "not an array" in outcome


def test_a_block_within_another_counts_into_both_and_the_outer_keeps_counting():
    step, rate = _decay_operands(128)
    with generation.chosen(generated="generated"), generation.assumed("sm90"), generated.taken() as outer:
        with generated.taken() as inner:
            recurrence_algebra.selective_decay(step, rate)
        recurrence_algebra.selective_decay(step, rate)
    assert inner == {(DECAY, generated.GENERATED): 1}
    assert outer == {(DECAY, generated.GENERATED): 2}


def test_a_grouped_scores_readout_runs_the_written_tile_and_counts_why():
    """The scores tile read apart from its group's writes is no declared program: the written tile runs and
    the static refusal is counted."""
    rng = np.random.default_rng(3)
    products, decays = (rng.standard_normal((32, 32)).astype(np.float32) for _ in range(2))
    scales = rng.standard_normal(32).astype(np.float32)
    written = recurrence_algebra.scores_tile(products, decays, scales, 2)
    with generation.chosen(generated="generated"), generation.assumed("sm90"), generated.taken() as counts:
        found = recurrence_algebra.scores_tile(products, decays, scales, 2)
    np.testing.assert_array_equal(found, written)
    assert counts == {("dew.algebra.recurrence_cases.scores", "another static program"): 1}
