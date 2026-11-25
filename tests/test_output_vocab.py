"""A head scoring fewer tokens than its table holds rows (`CausalTransformer.output_vocab`, Inkling 2's 200058
of 201024; theme M71)."""

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
