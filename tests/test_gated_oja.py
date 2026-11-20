"""Oja's rule with a gate a value dimension (`recurrence_chunk.chunk_gated_oja_rule`, fla's gated_oja_rule)
on the derived chunk forms of its transposed memory, read along its values: its output, final memory and
gradients on par with the rule a token at a time from float64 (check 1)."""

import functools

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec as P
from test_recurrence_chunk import (
    DOCUMENTS as FORKED_DOCUMENTS,
    GIVEN,
    OJA,
    assert_forked_on_par,
    assert_packed_on_par,
    entering_documents,
    hosted,
    sigmoid,
    unit,
)

from dew.nn.kernels import generation
from dew.nn.kernels.recurrence_chunk import Chunk, chunk, chunk_gated_oja_rule, recurrent_gated_oja_rule

ROWS, LENGTH, KEY_HEADS, VALUE_HEADS, KEY_WIDTH, VALUE_WIDTH = 2, 70, 1, 2, 16, 8
"""The values narrower than the keys: the transposed memory's read along its values is the keys' width."""
CHUNK = 16
DOCUMENTS = ((0, 23, 40), (9, 48))
"""Each row's packed documents' first tokens: the first row's memory zeroed at token 0, the second's entering
its first document, 48 a chunk's first token."""
SEEDS = {False: 16, True: 32}
"""Each site's declared seeds (one sequence, packed), sized by `seeds_for` on armada."""

CHUNKED = functools.partial(chunk_gated_oja_rule, chunk_size=CHUNK)
"""The chunked rule at the test's chunk."""


def draw(seed: int):
    """The rule's fp32 operands from `seed` (`hosted`), as fla's test draws them, and a cotangent of its
    output and final memory: fresh queries and keys, values of unit length, a head's write gate a sigmoid's, a
    log decay a value dimension a log sigmoid's, a memory entering."""
    normal = np.random.default_rng([seed, 1]).standard_normal
    query, key = (normal((ROWS, LENGTH, KEY_HEADS, KEY_WIDTH)) for _ in range(2))
    value = unit(normal((ROWS, LENGTH, VALUE_HEADS, VALUE_WIDTH)))
    gv = -np.logaddexp(0.0, -normal((ROWS, LENGTH, VALUE_HEADS, VALUE_WIDTH)))
    beta = sigmoid(normal((ROWS, LENGTH, VALUE_HEADS)))
    state = 0.3 * normal((ROWS, VALUE_HEADS, KEY_WIDTH, VALUE_WIDTH))
    cotangent = (normal((ROWS, LENGTH, VALUE_HEADS, VALUE_WIDTH)),
                 normal((ROWS, VALUE_HEADS, KEY_WIDTH, VALUE_WIDTH)))
    return *(hosted(t) for t in (query, key, value, gv, beta, state)), jax.tree.map(hosted, cotangent)


@pytest.mark.parametrize('packed', (False, True))
def test_the_chunked_rule_is_as_exact_as_the_rule_a_token_at_a_time(packed):
    """The chunked rule's output, final memory and every operand's gradient on par with the rule a token at a
    time from float64, one sequence and packed documents (check 1): the delta rule's derived forms on the
    transposed memory, its readout along the memory's values."""
    assert_packed_on_par(CHUNKED, recurrent_gated_oja_rule, draw, 'gated oja', packed=packed,
                         documents=DOCUMENTS, length=LENGTH, seeds=SEEDS[packed])


def forked_draw(seed: int):
    """`draw`'s operands a forked call's (`assert_forked_on_par`): a table of `GIVEN` memories a row and an
    entry write a document of two terms (`Chunk.entry_key`), the keys' heads and widths the values' own."""
    query, key, value, gv, beta, _, cotangent = draw(seed)
    normal = np.random.default_rng([seed, 2]).standard_normal
    documents = len(FORKED_DOCUMENTS[0])
    given = 0.3 * normal((ROWS, GIVEN, VALUE_HEADS, KEY_WIDTH, VALUE_WIDTH))
    operands = Chunk(query, key, value, log_decay=gv, beta=beta, state=hosted(given),
                     entry_key=hosted(0.3 * normal((ROWS, documents, 2, KEY_HEADS, KEY_WIDTH))),
                     entry_value=hosted(0.3 * normal((ROWS, documents, 2, VALUE_HEADS, VALUE_WIDTH))),
                     entry_weight=hosted(np.random.default_rng([seed, 3]).uniform(0, 1, (ROWS, documents,
                                                                                         VALUE_HEADS))))
    return operands, cotangent


def test_a_forked_calls_entry_writes_are_as_exact_as_each_document_alone():
    """A forked call of Oja's rule (`Chunk.loads`) whose documents each add an entry write to the memory they
    enter (`Chunk.entry_key`), the keys wider than the values and a key head serving two value heads: its
    output, last memory and every gradient against each document a sequence of its own a token at a time
    from float64 (check 1), the write laid on the transposed memory as its keys' and values' outer
    products."""
    assert_forked_on_par('xla', OJA, forked_draw, 'forked gated oja', seeds=16)


@pytest.mark.parametrize('named', ('fla', 'mosaic_gpu'))
def test_a_kernel_named_for_the_delta_rule_refuses_no_oja_call(named):
    """A forked call of Oja's rule, and one inside a shard_map that checks varying axes, under a native kernel
    named for the delta rule (`generation.chosen`): each handed to the portable math by its erase along the
    values before the named kernel's other needs are asked (`recurrence_chunk.TRANSPOSED`), so neither the
    fork nor the varying operands the kernel refuses stops its trace."""
    operands, _ = forked_draw(0)
    query, key, value, gv, beta, state, _ = draw(0)
    rows = P('data')
    mapped = jax.shard_map(lambda *xs: chunk_gated_oja_rule(*xs, chunk_size=CHUNK)[0],
                           mesh=Mesh(np.asarray(jax.devices()[:2]), ('data',)), in_specs=(rows,) * 6,
                           out_specs=rows, check_vma=True)
    with generation.chosen(gated_delta_rule=named):
        out, final = jax.eval_shape(lambda given: chunk(OJA, given, chunk_size=CHUNK),
                                    entering_documents(operands, FORKED_DOCUMENTS))
        varying = jax.eval_shape(mapped, query, key, value, gv, beta, state)
    assert (out.shape, final.shape, varying.shape) == (value.shape, state.shape, value.shape)
