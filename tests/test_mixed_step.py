"""A serving step's mixed cached call (`dew.nn.inputs.Admitted`): one forward
over the decoding rows' tokens and the admitted prompts computes what a
decode call and a separate prefill compute."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.inputs import Admitted

VOCAB, ROWS, WIDTH = 50, 4, 6


def tiny(**overrides):
    config = {"vocab_size": VOCAB, "emb_features": 32, "num_layers": 2, "num_heads": 4, "num_kv_heads": 2,
              "mlp_features": 64, "max_seq_len": 16}
    return CausalTransformer(**{**config, **overrides})


def cache_of(model, params, rows):
    return model.apply(params, rows, method=CausalTransformer.init_cache, mutable=["cache"])[1]["cache"]


def prefilled(model, params):
    """Rows 0 and 1 hold prompts of 3 and 5 tokens, rows 2 and 3 nothing."""
    prompts = jnp.asarray(np.random.default_rng(0).integers(1, VOCAB, (ROWS, 5)), jnp.int32)
    valid = jnp.zeros((ROWS, 5), bool).at[0, 2:].set(True).at[1].set(True)
    _, updated = model.apply({**params, "cache": cache_of(model, params, ROWS)}, prompts,
                             attention_mask=valid, decode=True, mutable=["cache"])
    return updated["cache"]


@pytest.mark.parametrize("continuing", [True, False])
def test_a_mixed_call_decodes_and_prefills_as_the_separate_calls_do(continuing):
    """Rows 0 and 1 decode a token each while row 3 is admitted with a
    four-token prompt (left-padded to the piece's width, as admission pads)
    and a padding piece rides along: the logits and every cache row match a
    decode call over the rows and a prefill of row 3 alone, whether the
    piece reads its row's cache or, starting the row, its own keys."""
    model = tiny()
    params = model.init(jax.random.key(0), jnp.zeros((1, 4), jnp.int32))
    cache = prefilled(model, params)
    fed = jnp.asarray([7, 9, 0, 0], jnp.int32)
    feeding = jnp.asarray([True, True, False, False])
    prompt = jnp.asarray([0, 0, 11, 12, 13, 14], jnp.int32)
    tokens = jnp.concatenate([fed, prompt, jnp.zeros(WIDTH, jnp.int32)])[None]
    valid = jnp.concatenate([feeding, prompt > 0, jnp.zeros(WIDTH, bool)])[None]
    picked = jnp.asarray([[0, 1, 2, 3, ROWS + WIDTH - 1, ROWS + 2 * WIDTH - 1]])
    (_, logits), mixed = model.apply(
        {**params, "cache": cache}, tokens, picked, attention_mask=valid, decode=True, mutable=["cache"],
        admitted=Admitted(slots=jnp.asarray([3, ROWS]), cursors=jnp.asarray([0, 0]), continuing=continuing),
        method=CausalTransformer.states_and_logits_at)

    decoded, stepped = model.apply({**params, "cache": cache}, fed[:, None], attention_mask=feeding[:, None],
                                   decode=True, mutable=["cache"])
    alone, admitted = model.apply({**params, "cache": cache_of(model, params, 1)}, prompt[None],
                                  attention_mask=(prompt > 0)[None], decode=True, mutable=["cache"])
    np.testing.assert_allclose(logits[0, :2], decoded[:2, 0], atol=2e-5)
    np.testing.assert_allclose(logits[0, 4], alone[0, -1], atol=2e-5)
    for layer, held in mixed["cache"].items():
        attention = held["self_attn"]
        np.testing.assert_array_equal(attention["cache_index"], [4, 6, 0, 4])
        for name in ("cached_key", "cached_value"):
            written = np.asarray(attention[name])
            decoded_rows = np.asarray(stepped["cache"][layer]["self_attn"][name])[:2]
            np.testing.assert_allclose(written[:2], decoded_rows, atol=2e-6)
            alone_row = np.asarray(admitted["cache"][layer]["self_attn"][name])[0, :4]
            np.testing.assert_allclose(written[3, :4], alone_row, atol=2e-6)
            np.testing.assert_array_equal(attention[name][2], cache[layer]["self_attn"][name][2])


@pytest.mark.parametrize("overrides,named", [
    ({"layer_types": ("local", "global"), "kinds": {"local": {"window": 4}}}, "sliding window"),
    ({"attention_sinks": True}, "attention sinks")])
def test_a_layer_a_mixed_call_cannot_serve_names_why(overrides, named):
    """A layer that reads more than its own dense, causal row refuses the
    mixed call by name, so the server can say why it keeps two forwards."""
    model = tiny(**overrides)
    params = model.init(jax.random.key(0), jnp.zeros((1, 4), jnp.int32))
    with pytest.raises(ValueError, match=named):
        tokens = jnp.zeros((1, ROWS + WIDTH), jnp.int32)
        model.apply({**params, "cache": cache_of(model, params, ROWS)}, tokens,
                    jnp.zeros((1, 1), jnp.int32), attention_mask=jnp.ones(tokens.shape, bool), decode=True,
                    mutable=["cache"], admitted=Admitted(slots=jnp.asarray([3]), cursors=jnp.asarray([0])),
                    method=CausalTransformer.states_and_logits_at)
