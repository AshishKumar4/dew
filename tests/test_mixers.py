"""The mixer seam: one declared value naming a layer's mixer, by kind.

The backbone takes a `mixer` value from the `mixers` registry (None for
today's grouped-query causal attention), and each kind builds its own
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

from dew.nn.backbones.causal_transformer import CausalTransformer, LayerKind
from dew.nn.mixers import AttentionMixer, MixerBase, MixerContext, mixers
from dew.registry import models

VOCAB = 37
TINY = dict(vocab_size=VOCAB, emb_features=32, num_layers=2, num_heads=4,
            mlp_features=64, max_seq_len=16)


def tiny(**overrides):
    return CausalTransformer(**{**TINY, **overrides})


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


def test_none_and_the_attention_value_build_the_same_tree():
    """The default and an explicit attention value agree exactly."""
    ids = jnp.ones((1, 8), jnp.int32)
    default = tiny().init(jax.random.key(0), ids)
    explicit = tiny(mixer=AttentionMixer()).init(jax.random.key(0), ids)

    assert jax.tree.structure(default) == jax.tree.structure(explicit)
    for left, right in zip(jax.tree.leaves(default), jax.tree.leaves(explicit)):
        assert jnp.array_equal(left, right)


def test_mixer_records_and_values_compute_the_same_logits():
    expected = tiny(mixer=ScaleMixer(3.0))
    ids = jnp.asarray([[1, 2, 3, 4]], jnp.int32)
    params = expected.init(jax.random.key(0), ids)
    logits = expected.apply(params, ids)
    record = {"kind": "test_scale", "scale": 3.0}
    for model in (tiny(mixer=record),
                  models.build("causal_transformer", **TINY, mixer=record)):
        assert jnp.array_equal(model.apply(params, ids), logits)
    assert not jnp.allclose(tiny(mixer=ScaleMixer(0.0)).apply(params, ids), logits)


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


def test_the_mixer_registry_refuses_unknown_kinds_and_fields():
    with pytest.raises(KeyError, match="no mixer named 'nope'"):
        mixers.build("nope")
    with pytest.raises(ValueError, match="has no field for"):
        mixers.build("test_scale", nope=1.0)


def test_a_kind_without_a_build_is_refused_loudly():
    """A registered value that builds nothing raises NotImplementedError at setup."""

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


def test_a_kind_selects_the_mixer_its_layers_run():
    """A kind's mixer value builds that kind's layers and the model's value
    the rest, observed through the forward pass: a ScaleMixer of 0 on one
    kind zeroes that kind's mixing, so the logits move when the assignment
    flips and the two kinds are told apart by what they compute."""
    ids = jnp.ones((2, 8), jnp.int32)
    key = jax.random.key(0)
    per_kind = hybrid(kinds={"linear": LayerKind(mixer=ScaleMixer(scale=0.0))})
    swapped = hybrid(mixer=ScaleMixer(scale=0.0), kinds={"linear": LayerKind(mixer={"kind": "attention"})})
    plain = hybrid()

    kind_logits = per_kind.apply(per_kind.init(key, ids), ids)
    swapped_logits = swapped.apply(swapped.init(key, ids), ids)
    plain_logits = plain.apply(plain.init(key, ids), ids)

    assert not jnp.allclose(kind_logits, plain_logits)
    assert not jnp.allclose(swapped_logits, plain_logits)
    assert not jnp.allclose(kind_logits, swapped_logits)


def test_invalid_kind_mixer_records_are_refused():
    with pytest.raises(ValueError, match="kind's mixer"):
        LayerKind(mixer="test_scale")
    with pytest.raises(KeyError, match="no mixer named 'nope'"):
        LayerKind(mixer={"kind": "nope"})


def test_models_build_takes_kind_mixer_records():
    """The CLI path builds per-kind mixers through `models.build`."""
    model = models.build(
        "causal_transformer", **TINY, layer_types=("full_attention", "linear"),
        kinds={"linear": {"mixer": {"kind": "test_scale", "scale": 3.0}}})
    expected = hybrid(kinds={"linear": LayerKind(mixer=ScaleMixer(3.0))})
    ids = jnp.asarray([[1, 2, 3, 4]], jnp.int32)
    params = expected.init(jax.random.key(0), ids)
    assert jnp.array_equal(model.apply(params, ids), expected.apply(params, ids))
