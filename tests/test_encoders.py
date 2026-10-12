"""The conditioning seam: an encoder's value reaches the model under one
keyword, and the text mask it carries is what the pooling weighs by, so a
padded row moves nothing.
"""


import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.inputs import Condition, Field, InputSpec
from dew.nn.dit import ConditioningEmbed, TextContext, masked_mean

############################################################################################################
# Text pooling reads the mask (T23)
############################################################################################################

def test_the_text_conditioning_projects_each_row_once_not_each_token(rng):
    """The text reaches the adaLN vector only as the masked mean of its
    projected tokens. Projecting every token cost layout_parity's DiT (77
    text tokens of 1024 features for 16 image tokens) most of its step's
    matmul FLOPs, and a layout that splits no rows (tensor, sequence)
    repeated all of it on every device: 2.63 times one device's FLOPs on
    four 3090s. The vector is the same, a trained bias and an empty row
    included, and a longer text adds only its pooling's few operations per
    element, not a projection's."""
    embed = ConditioningEmbed(emb_features=64, mlp_ratio=1)
    hidden = jax.random.normal(rng, (3, 77, 32))
    mask = jnp.array([[1] * 20 + [0] * 57, [0] * 77, [1] * 77])
    temb = jnp.array([1.0, 2.0, 3.0])
    params = embed.init(rng, temb, TextContext(hidden, mask))
    projection = params["params"]["text_context_proj"]
    projection["bias"] = projection["bias"] + 0.5

    got = embed.apply(params, temb, TextContext(hidden, mask))
    tokens = hidden @ projection["kernel"] + projection["bias"]
    want = embed.apply(params, temb, None) + masked_mean(tokens, mask)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)




def test_padded_rows_do_not_move_the_conditioning_vector():
    """The tokenizer pads every prompt to 77 slots; what the padded rows hold
    must not reach the adaLN vector. Before the mask, they were averaged in
    and moved it."""
    embed = ConditioningEmbed(emb_features=16, mlp_ratio=1)
    key = jax.random.PRNGKey(0)
    hidden = jax.random.normal(jax.random.fold_in(key, 1), (2, 6, 8))
    mask = jnp.array([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]])
    temb = jnp.array([10.0, 20.0])
    params = embed.init(key, temb, TextContext(hidden, mask))

    baseline = embed.apply(params, temb, TextContext(hidden, mask))
    garbage = hidden.at[0, 3:].set(hidden[0, 3:] + 50.0)
    moved = embed.apply(params, temb, TextContext(garbage, mask))
    assert jnp.allclose(moved, baseline, atol=1e-5)

    # the same rows do count once the mask says they are real
    full = jnp.ones_like(mask)
    assert not jnp.allclose(embed.apply(params, temb, TextContext(garbage, full)),
                            embed.apply(params, temb, TextContext(hidden, full)), atol=1e-2)


def test_two_conditions_cannot_share_one_batch_field():
    """The objective reads batch[condition.field] under each keyword, so two
    conditions on one field would tokenize twice into one key and the second
    writing would win. A run with two text towers names a field per tower."""
    from dew.inputs import CharTable

    conditions = {
        "clip": Condition(CharTable.from_pretrained(seed=0), field="text"),
        "t5": Condition(CharTable.from_pretrained(seed=1), field="text"),
    }
    with pytest.raises(ValueError):
        InputSpec(sample=Field("image", (8, 8, 3)), conditions=conditions)


def test_condition_roundtrip_preserves_nondefault_character_embeddings():
    from dew.inputs import CharTable

    condition = Condition(CharTable.from_pretrained(
        seed=7, dtype="bfloat16", tokens=4, features=2, vocab=8))
    restored = Condition.from_json(condition.to_json())
    contexts = [value.encoder.encode(value.encoder.params, value.encoder.tokenize(["ab", ""]))
                for value in (condition, restored)]
    np.testing.assert_array_equal(contexts[0].hidden, contexts[1].hidden)
    np.testing.assert_array_equal(contexts[0].mask, contexts[1].mask)
    assert contexts[1].hidden.dtype == jnp.bfloat16
