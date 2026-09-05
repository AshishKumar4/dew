"""The mixer seam: one declared value naming a layer's mixer, by kind.

The backbone takes a `mixer` value from the `mixers` registry — None for
today's grouped-query causal attention — and each kind builds its own
`DecoderBlock` factory from the layer's context. A new kind registers its
value and plugs in with no branch on the backbone, which `test_scale` here
proves by being one: a second member that exists only in this file, yet
builds, runs and refuses unknown fields through the same paths a reference
kind will. Records (`mixer={"kind": ...}`) and values agree, and an unknown
kind or field raises naming what was asked for.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.mixers import AttentionMixer, MixerBase, MixerContext, mixers
from dew.registry import models

VOCAB = 37


def tiny(**overrides):
    config = dict(vocab_size=VOCAB, emb_features=32, num_layers=2, num_heads=4,
                  mlp_features=64, max_seq_len=16)
    return CausalTransformer(**{**config, **overrides})


class ScaleMixerModule(nn.Module):
    """A token mixer with no parameters: scale and pass through."""

    scale: float = 2.0

    @nn.compact
    def __call__(self, x, decode: bool = False, positions=None, segment_ids=None):
        del decode, positions, segment_ids
        return x * self.scale


@mixers("test_scale")
@dataclasses.dataclass(frozen=True)
class ScaleMixer(MixerBase):
    """The seam's proof of pluggability: a kind from outside the backbone."""

    scale: float = 2.0

    def build(self, ctx: MixerContext):
        del ctx
        return functools.partial(ScaleMixerModule, scale=self.scale)


def test_the_default_mixer_is_todays_attention():
    """None builds the grouped-query attention tree, unchanged."""
    params = tiny().init(jax.random.key(0), jnp.ones((1, 8), jnp.int32))

    assert tiny().mixer is None
    assert params["params"]["layers_0"]["self_attn"]["q_proj"]["kernel"].shape == (32, 32)


def test_none_and_the_attention_value_build_the_same_tree():
    """The default and an explicit attention value agree exactly."""
    ids = jnp.ones((1, 8), jnp.int32)
    default = tiny().init(jax.random.key(0), ids)
    explicit = tiny(mixer=AttentionMixer()).init(jax.random.key(0), ids)

    assert jax.tree.structure(default) == jax.tree.structure(explicit)
    for left, right in zip(jax.tree.leaves(default), jax.tree.leaves(explicit)):
        assert jnp.array_equal(left, right)


def test_a_record_and_a_value_agree():
    """`mixer={"kind": ...}` from a config and the dataclass from code agree."""
    assert tiny(mixer={"kind": "attention"}).mixer == AttentionMixer()
    assert tiny(mixer={"kind": "test_scale", "scale": 3.0}).mixer == ScaleMixer(3.0)


def test_models_build_takes_mixer_records():
    """The CLI path builds the same values through `models.build`."""
    config = dict(vocab_size=VOCAB, emb_features=32, num_layers=2, num_heads=4,
                  mlp_features=64, max_seq_len=16)

    assert models.build("causal_transformer", **config).mixer is None
    assert models.build(
        "causal_transformer", **{**config, "mixer": {"kind": "test_scale", "scale": 3.0}}
    ).mixer == ScaleMixer(3.0)


def test_a_registered_kind_builds_and_runs():
    """The test kind lands in the block as self_attn and runs the forward."""
    model = tiny(mixer=ScaleMixer(scale=0.0))
    ids = jnp.ones((2, 8), jnp.int32)
    params = model.init(jax.random.key(0), ids)
    bound = model.bind(params)

    assert isinstance(bound.layers[0].self_attn, ScaleMixerModule)
    assert bound.layers[0].self_attn.scale == 0.0
    assert "q_proj" not in str(jax.tree.map(jnp.shape, params))
    logits = model.apply(params, ids)
    assert logits.shape == (2, 8, VOCAB)
    assert jnp.all(jnp.isfinite(logits))


def test_an_unknown_mixer_kind_is_refused():
    with pytest.raises(KeyError, match="no mixer named 'nope'"):
        tiny(mixer={"kind": "nope"})


def test_a_mixer_record_without_a_kind_is_refused():
    with pytest.raises(ValueError, match="names its kind"):
        tiny(mixer={"scale": 3.0})


def test_a_mixer_field_no_kind_declares_is_refused():
    with pytest.raises(ValueError, match="has no field for"):
        tiny(mixer={"kind": "attention", "scale": 3.0})
    with pytest.raises(ValueError, match="has no field for"):
        tiny(mixer={"kind": "test_scale", "kv_lora_rank": 4})


def test_something_that_is_neither_a_value_nor_a_record_is_refused():
    with pytest.raises(ValueError, match="not 'test_scale'"):
        tiny(mixer="test_scale")


def test_the_registry_builds_kinds_by_name():
    assert mixers["attention"] is AttentionMixer
    assert mixers["test_scale"] is ScaleMixer
    assert mixers.build("test_scale", scale=3.0) == ScaleMixer(3.0)
    with pytest.raises(KeyError, match="no mixer named 'nope'"):
        mixers.build("nope")
    with pytest.raises(ValueError, match="has no field for"):
        mixers.build("test_scale", nope=1.0)


def test_a_kind_without_a_build_is_refused_loudly():
    """A registered value that builds nothing fails at setup, not silently."""

    @mixers("test_empty")
    @dataclasses.dataclass(frozen=True)
    class EmptyMixer(MixerBase):
        pass

    try:
        with pytest.raises(NotImplementedError, match="builds no mixer"):
            tiny(mixer=EmptyMixer()).init(jax.random.key(0), jnp.ones((1, 8), jnp.int32))
    finally:
        del mixers._members["test_empty"]


def hybrid(**overrides):
    """Two layer types: full_attention rides the model, linear names its own."""
    return tiny(layer_types=("full_attention", "linear"), **overrides)


def test_a_kind_names_its_own_mixer():
    """One kind's layers build from its value, the other kind from the model's."""
    from dew.nn.backbones.causal_transformer import CausalSelfAttention, LayerKind

    model = hybrid(kinds={"linear": LayerKind(mixer=ScaleMixer(scale=0.0))})
    bound = model.bind(model.init(jax.random.key(0), jnp.ones((1, 8), jnp.int32)))

    assert isinstance(bound.layers[0].self_attn, CausalSelfAttention)
    assert isinstance(bound.layers[1].self_attn, ScaleMixerModule)
    logits = model.apply(model.init(jax.random.key(0), jnp.ones((2, 8), jnp.int32)),
                         jnp.ones((2, 8), jnp.int32))
    assert logits.shape == (2, 8, VOCAB)


def test_a_kind_mixer_beats_the_model_mixer():
    """The kind's value wins where set; elsewhere the model's applies."""
    from dew.nn.backbones.causal_transformer import CausalSelfAttention, LayerKind

    model = hybrid(mixer=ScaleMixer(scale=0.0),
                   kinds={"linear": LayerKind(mixer={"kind": "attention"})})
    bound = model.bind(model.init(jax.random.key(0), jnp.ones((1, 8), jnp.int32)))

    assert isinstance(bound.layers[0].self_attn, ScaleMixerModule)
    assert isinstance(bound.layers[1].self_attn, CausalSelfAttention)


def test_a_kind_record_coerces_like_the_model_record():
    """LayerKind takes its mixer as a record from a config or a value."""
    from dew.nn.backbones.causal_transformer import LayerKind

    assert LayerKind(mixer={"kind": "test_scale", "scale": 3.0}).mixer == ScaleMixer(3.0)
    assert LayerKind(mixer=ScaleMixer(3.0)).mixer == ScaleMixer(3.0)
    assert LayerKind().mixer is None
    with pytest.raises(ValueError, match="kind's mixer"):
        LayerKind(mixer="test_scale")
    with pytest.raises(KeyError, match="no mixer named 'nope'"):
        LayerKind(mixer={"kind": "nope"})


def test_models_build_takes_kind_mixer_records():
    """The CLI path builds per-kind mixers through `models.build`."""
    config = dict(vocab_size=VOCAB, emb_features=32, num_layers=2, num_heads=4,
                  mlp_features=64, max_seq_len=16,
                  layer_types=("full_attention", "linear"),
                  kinds={"linear": {"mixer": {"kind": "test_scale", "scale": 3.0}}})

    model = models.build("causal_transformer", **config)
    assert model.kind_of("linear").mixer == ScaleMixer(3.0)
    assert model.kind_of("full_attention").mixer is None
