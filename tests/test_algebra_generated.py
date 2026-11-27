"""The seam between a template's written component and its generated, checked body (`dew.algebra.generated`,
docs/design/algebra.md §6.11): under the default choice the written math runs and nothing is counted; under
"generated" a declared case runs the generated body, bit for bit its module's, and any other call the written
math, its reason counted."""

import jax
import jax.numpy as jnp
import numpy as np

from dew.algebra import component_cases_derived, generated
from dew.nn.kernels import generation, recurrence_algebra
from dew.nn.kernels.templates import attention_algebra

DECAY = "dew.algebra.component_cases.selective_decay"
MERGE = "dew.algebra.component_cases.split_kv_merge"


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
    np.testing.assert_array_equal(decay, component_cases_derived.selective_decay(step, rate))
    for found, expected in zip(merged[:3], component_cases_derived.split_kv_merge(*first[:3], *second[:3]),
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
