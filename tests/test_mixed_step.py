"""A serving step's mixed cached call (`dew.nn.inputs.Admitted`): one forward
over the decoding rows' tokens and the admitted prompts computes what a
decode call and a separate prefill compute."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from model_support import TINY_DECODER

from dew.batch import coordinates
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.inputs import Admitted
from dew.nn.kv_cache import KVCache
from dew.nn.mixers import AttentionMixer

VOCAB, ROWS, WIDTH = 50, 4, 6


def tiny(**overrides):
    config = {"vocab_size": VOCAB, **TINY_DECODER, "num_kv_heads": 2}
    return CausalTransformer(**{**config, **overrides})


def cache_of(model, params, rows):
    return model.apply(params, rows, method=CausalTransformer.init_cache, mutable=["cache"])[1]["cache"]


def tables_of(cache, rows):
    """The page-table rows of `rows` in a paged cache's first layer."""
    first = next(iter(cache.values()))["self_attn"]
    return first["page_table"][jnp.asarray(rows)]


def prefilled(model, params):
    """Rows 0 and 1 hold prompts of 3 and 5 tokens, rows 2 and 3 nothing."""
    prompts = jnp.asarray(np.random.default_rng(0).integers(1, VOCAB, (ROWS, 5)), jnp.int32)
    valid = jnp.zeros((ROWS, 5), bool).at[0, 2:].set(True).at[1].set(True)
    _, updated = model.apply({**params, "cache": cache_of(model, params, ROWS)}, prompts,
                             attention_mask=valid, decode=True, mutable=["cache"])
    return updated["cache"]


FEATURES = {"plain": {}, "no qk norm": {"qk_norm": False}, "projection norm": {"qk_norm_scope": "projection"},
            "softcap": {"attn_logit_softcap": 5.0}, "output gate": {"output_gate": True},
            "attention scale": {"attention_scale": 0.3}, "partial rope": {"partial_rotary_factor": 0.5},
            "exclusive, no rope": {"mixer": AttentionMixer(nope=True, exclusive_self_attention=True)}}


@pytest.mark.parametrize("paged", [False, True], ids=["dense", "paged"])
@pytest.mark.parametrize("continuing", [True, False])
@pytest.mark.parametrize("feature", list(FEATURES))
def test_a_mixed_call_decodes_and_prefills_as_the_separate_calls_do(continuing, feature, paged):
    """Rows 0 and 1 decode a token each while row 3 is admitted with a
    four-token prompt (left-padded to the piece's width, as admission pads)
    and a padding piece rides along: the logits and every cache row match a
    decode call over the rows and a prefill of row 3 alone, whether the
    piece reads its row's cache or, starting the row, its own keys, under
    each attention feature the mixed call takes, over a dense cache or a
    page pool (row 3 writing through its own pages)."""
    pool = {"kv_cache": KVCache(page_size=4, pages=ROWS * 4)} if paged else {}
    model = tiny(**FEATURES[feature], **pool)
    params = model.init(jax.random.key(0), jnp.zeros((1, 4), jnp.int32))
    cache = prefilled(model, params)
    fed = jnp.asarray([7, 9, 0, 0], jnp.int32)
    feeding = jnp.asarray([True, True, False, False])
    prompt = jnp.asarray([0, 0, 11, 12, 13, 14], jnp.int32)
    tokens = jnp.concatenate([fed, prompt, jnp.zeros(WIDTH, jnp.int32)])[None]
    valid = jnp.concatenate([feeding, prompt > 0, jnp.zeros(WIDTH, bool)])[None]
    picked = jnp.asarray([[0, 1, 2, 3, ROWS + WIDTH - 1, ROWS + 2 * WIDTH - 1]])
    (_, logits), mixed = model.apply(
        {**params, "cache": cache}, tokens, picked, coordinates=coordinates(valid.shape, valid=valid),
        decode=True, mutable=["cache"],
        admitted=Admitted(slots=jnp.asarray([3, ROWS]), cursors=jnp.asarray([0, 0]), continuing=continuing,
                          tables=tables_of(cache, [3, 0]) if paged else None),
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
        if paged:
            continue  # the pool's pages: the logits read them, and `cache_index` is checked
        for name in ("cached_key", "cached_value"):
            written = np.asarray(attention[name])
            decoded_rows = np.asarray(stepped["cache"][layer]["self_attn"][name])[:2]
            np.testing.assert_allclose(written[:2], decoded_rows, atol=2e-6)
            alone_row = np.asarray(admitted["cache"][layer]["self_attn"][name])[0, :4]
            np.testing.assert_allclose(written[3, :4], alone_row, atol=2e-6)
            np.testing.assert_array_equal(attention[name][2], cache[layer]["self_attn"][name][2])


REFUSALS = {"plain": {}, "alibi": {"position_embedding": "alibi"}, "sinks": {"attention_sinks": True},
            "shared keys": {"kv_shared_layers": (1,)}, "int8 cache": {"kv_cache": KVCache(quantized="int8")},
            "window": {"layer_types": ("local", "global"),
                       "kinds": {"local": LayerKind(window=4), "global": LayerKind()}}}


@pytest.mark.parametrize("case", list(REFUSALS))
def test_a_model_says_whether_its_layers_run_the_mixed_call_as_they_do(case):
    """The model answers for every layer it builds, unbound and untraced
    (`mixed_admission_refusal`): it names a reason exactly where a layer's
    mixed call refuses to run, and None where the call runs. ALiBi, sinks,
    shared keys, a quantized cache and a sliding window are refused; plain
    attention runs."""
    model = tiny(**REFUSALS[case])
    refusal = model.mixed_admission_refusal()
    params = model.init(jax.random.key(0), jnp.zeros((1, 4), jnp.int32))
    cache = prefilled(model, params)
    tokens = jnp.zeros((1, ROWS + WIDTH), jnp.int32)
    valid = jnp.zeros((1, ROWS + WIDTH), bool).at[0, :2].set(True)

    def mixed():
        return model.apply({**params, "cache": cache}, tokens, jnp.asarray([[0, 1]]),
                           coordinates=coordinates(valid.shape, valid=valid), decode=True, mutable=["cache"],
                           admitted=Admitted(slots=jnp.asarray([ROWS]), cursors=jnp.asarray([0])),
                           method=CausalTransformer.states_and_logits_at)

    assert (refusal is None) == (case == "plain")
    if refusal is None:
        mixed()
    else:
        with pytest.raises(ValueError):
            mixed()


def test_a_mixed_call_carrying_a_pairwise_mask_is_refused_by_name():
    """A mixed call's rows read their caches as a decode step does, which is
    its whole visibility, so a pairwise mask beside it, which a cached call
    would apply, is refused by name rather than run without it."""
    model = tiny()
    params = model.init(jax.random.key(0), jnp.zeros((1, 4), jnp.int32))
    cache = prefilled(model, params)
    tokens = jnp.zeros((1, ROWS + WIDTH), jnp.int32)
    valid = jnp.zeros((1, ROWS + WIDTH), bool).at[0, :2].set(True)
    with pytest.raises(ValueError):
        model.apply({**params, "cache": cache}, tokens, jnp.asarray([[0, 1]]),
                    coordinates=coordinates(valid.shape, valid=valid),
                    attention_pairwise_mask=jnp.ones((1, ROWS + WIDTH, ROWS + WIDTH), bool), decode=True,
                    mutable=["cache"], admitted=Admitted(slots=jnp.asarray([ROWS]), cursors=jnp.asarray([0])),
                    method=CausalTransformer.states_and_logits_at)
