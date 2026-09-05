"""The conditioning seam: an encoder's value reaches the model under one
keyword, and the text mask it carries is what the pooling weighs by, so a
padded row moves nothing.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.inputs import CLIPText, Condition, InputSpec, Field, unit_range
from dew.nn.dit import ConditioningEmbed, TextContext, masked_mean
from dew.registry import encoders

CLIP_TINY = Path(__file__).resolve().parent / "fixtures" / "clip" / "tiny"


def test_unit_range_is_the_one_pixel_convention():
    pixels = jnp.array([0, 127, 128, 255], jnp.uint8)
    assert jnp.allclose(unit_range(pixels), jnp.array([-1.0, -0.5 / 127.5, 0.5 / 127.5, 1.0]))


def test_field_shape_is_a_tuple_of_ints():
    assert Field("image", [8, 8, 3]).shape == (8, 8, 3)
    assert InputSpec(Field("image", (8, 8, 3))).conditions == {}


############################################################################################################
# Text pooling reads the mask (T23)
############################################################################################################

def test_masked_mean_ignores_padded_rows():
    hidden = jnp.arange(2 * 4 * 3, dtype=jnp.float32).reshape(2, 4, 3)
    mask = jnp.array([[1, 1, 0, 0], [1, 1, 1, 1]])
    pooled = masked_mean(hidden, mask)
    assert jnp.allclose(pooled[0], jnp.mean(hidden[0, :2], axis=0))
    assert jnp.allclose(pooled[1], jnp.mean(hidden[1], axis=0))


def test_an_empty_mask_row_contributes_nothing(rng):
    """A row with no real tokens pools to exactly zero, the same vector the
    model gets with no text at all, rather than 0/0 = NaN."""
    embed = ConditioningEmbed(emb_features=16, mlp_ratio=1)
    hidden = jax.random.normal(rng, (2, 6, 8))
    mask = jnp.array([[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]])
    temb = jnp.array([10.0, 20.0])
    params = embed.init(rng, temb, TextContext(hidden, mask))
    pooled = embed.apply(params, temb, TextContext(hidden, mask))
    assert jnp.all(jnp.isfinite(pooled))
    assert jnp.allclose(pooled[0], embed.apply(params, temb, None)[0], atol=1e-6)


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


def test_the_text_encoder_hands_the_model_its_mask():
    """The CLIP encoder's TextContext carries the tokenizer's mask, so a short
    prompt's padding is excluded from the pooling by construction."""
    encoder = CLIPText.from_pretrained(str(CLIP_TINY))
    tokens = encoder.tokenize(["a red bird", ""])
    context = encoder.encode(encoder.params, tokens)
    assert context.hidden.shape[:2] == context.mask.shape == (2, 77)
    # BOS, four word pieces, EOS for the prompt; BOS and EOS for the empty one
    assert int(context.mask[1].sum()) == 2
    assert int(context.mask[0].sum()) > 2


def test_two_conditions_cannot_share_one_batch_field():
    """The objective reads batch[condition.field] under each keyword, so two
    conditions on one field would tokenize twice into one key and the second
    writing would win. A run with two text towers names a field per tower."""
    from dew.inputs import CharTable

    conditions = {
        "clip": Condition(CharTable.from_pretrained(seed=0), field="text"),
        "t5": Condition(CharTable.from_pretrained(seed=1), field="text"),
    }
    with pytest.raises(ValueError, match="field"):
        InputSpec(sample=Field("image", (8, 8, 3)), conditions=conditions)
