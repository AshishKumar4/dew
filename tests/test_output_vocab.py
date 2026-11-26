"""A head scoring fewer tokens than its table holds rows (`CausalTransformer.output_vocab`, Inkling 2's 200058
of 201024; theme M71)."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from model_support import TINY_DECODER

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.protocols import logits_width

ROWS, SCORED = 64, 48


def _decoder(*, tied: bool, output_vocab: int | None = None) -> CausalTransformer:
    return CausalTransformer(**{**TINY_DECODER, "vocab_size": ROWS, "dtype": "float32",
                                "tie_embeddings": tied, "output_vocab": output_vocab})


@pytest.mark.parametrize("tied", [True, False], ids=["tied", "untied"])
def test_a_head_scores_its_first_rows_alone(tied):
    """A head scoring the first 48 of its table's 64 rows keeps every row as stored (its parameters the
    full head's) and gives the full head's first 48 logits, bit for bit, its table (`output_table`) cut
    alike; a head scoring more rows than its table holds is refused."""
    narrow, full = _decoder(tied=tied, output_vocab=SCORED), _decoder(tied=tied)
    ids = jnp.asarray(np.random.default_rng(0x0B0).integers(1, SCORED, (2, 9)), jnp.int32)
    variables = nn.unbox(full.init(jax.random.key(0), ids))
    assert jax.tree.map(jnp.shape, nn.unbox(narrow.init(jax.random.key(0), ids))) == jax.tree.map(
        jnp.shape, variables)
    logits = np.asarray(jax.jit(narrow.apply)(variables, ids))
    assert logits.shape[-1] == SCORED == logits_width(narrow)
    np.testing.assert_array_equal(logits, np.asarray(jax.jit(full.apply)(variables, ids))[..., :SCORED])
    table = narrow.bind(variables).output_table()
    assert table.rows().shape[0] == SCORED
    with pytest.raises(ValueError):
        _decoder(tied=tied, output_vocab=ROWS + 1)


@pytest.mark.parametrize("tied", [True, False], ids=["tied", "untied"])
def test_a_chunked_loss_over_a_narrow_head_reads_its_scored_rows_alone(tied):
    """The chunked loss over a head scoring 48 of its 64 rows (its table the stored parameter itself,
    `OutputTable.scored`) is, bit for bit, the loss over a table holding those rows alone; its
    gradient leaves every unscored row of the table exactly zero, and a table scoring every row reads its
    matrix whole."""
    from dew.objectives.lm.chunked import chunked_cross_entropy

    narrow = _decoder(tied=tied, output_vocab=SCORED)
    ids = jnp.asarray(np.random.default_rng(0x0B1).integers(1, SCORED, (2, 9)), jnp.int32)
    variables = nn.unbox(narrow.init(jax.random.key(0), ids))
    hidden = narrow.apply(variables, ids, method="hidden_states")
    targets = jnp.roll(ids, -1, axis=1)
    whole = _decoder(tied=tied).bind(variables).output_table()
    assert whole.scored is None and whole.vocab == ROWS

    def loss(variables):
        table = narrow.bind(variables).output_table()
        assert table.scored == SCORED and any(table.matrix is leaf for leaf in jax.tree.leaves(variables))
        return jnp.sum(chunked_cross_entropy(hidden, table, targets, 2)[0])
    # The same loss over a table holding the scored rows alone, as the head computed it before.
    table = narrow.bind(variables).output_table()
    cut = dataclasses.replace(table, scored=None, matrix=table.matrix[:SCORED] if table.vocab_major
                              else table.matrix[:, :SCORED])
    np.testing.assert_array_equal(np.asarray(loss(variables)),
                                  np.asarray(jnp.sum(chunked_cross_entropy(hidden, cut, targets, 2)[0])))
    gradient = jax.grad(loss)(variables)["params"]
    table = gradient["embed_tokens"]["embedding"] if tied else gradient["lm_head"]["kernel"].T
    assert not np.any(np.asarray(table[SCORED:])) and np.any(np.asarray(table[:SCORED]))


def test_an_unscored_row_enters_no_product_of_the_head():
    """A head scoring 48 of its 64 rows, an unscored row holding fp32's largest values, which round to
    bf16's infinity in its bf16 product: its logits and their pullback to the states are finite, the row
    entering neither, where a product over every row multiplied the discarded columns' zero cotangents by
    it."""
    from dew.nn.protocols import OutputTable

    rng = np.random.default_rng(0x0B2)
    largest = float(np.finfo(np.float32).max)
    matrix = jnp.asarray(rng.normal(size=(ROWS, 16)), jnp.float32).at[SCORED + 3].set(largest)
    table = OutputTable(matrix, vocab_major=True, compute=jnp.bfloat16, scored=SCORED,
                        precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32)
    hidden = jnp.asarray(rng.normal(size=(2, 5, 16)), jnp.float32)
    logits, pull = jax.vjp(table.logits, hidden)
    (gradient,) = pull(jnp.ones_like(logits))
    assert logits.shape[-1] == SCORED
    assert bool(jnp.all(jnp.isfinite(logits))) and bool(jnp.all(jnp.isfinite(gradient)))
