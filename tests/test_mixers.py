"""The mixer seam: one declared value naming a layer's mixer, by kind.

The backbone takes a `mixer` value from the `mixers` registry (None for
today's grouped-query causal attention), and each kind builds its own
`DecoderBlock` factory from the layer's context. A new kind needs only its
value and plugs in with no branch on the backbone, which `ScaleMixer` here
proves by being one: a member that exists only in this file, named by its
import path, yet builds, runs and refuses unknown fields through the same
paths a reference kind will. Records (`mixer={"class": ..., "fields": ...}`)
and values agree, and an unknown kind or field raises naming what was asked for.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from flax import linen as nn
from model_support import TINY_DECODER

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.mixers import MixerBase, MixerContext
from dew.registry import models

VOCAB = 37
TINY = {"vocab_size": VOCAB, **TINY_DECODER}


def tiny(**overrides):
    return CausalTransformer(**{**TINY, **overrides})


class ScaleMixerModule(nn.Module):
    """A token mixer with no parameters: scale and pass through."""

    scale: float = 2.0

    @nn.compact
    def __call__(self, x, decode: bool = False, positions=None, segment_ids=None):
        del decode, positions, segment_ids
        return x * self.scale


@dataclasses.dataclass(frozen=True)
class ScaleMixer(MixerBase):
    """The seam's proof of pluggability: a kind from outside the backbone."""

    scale: float = 2.0

    def build(self, ctx: MixerContext):
        del ctx
        return functools.partial(ScaleMixerModule, scale=self.scale)


SCALE = f"{__name__}:ScaleMixer"


def test_mixer_records_and_values_compute_the_same_logits():
    expected = tiny(mixer=ScaleMixer(3.0))
    ids = jnp.asarray([[1, 2, 3, 4]], jnp.int32)
    params = expected.init(jax.random.key(0), ids)
    logits = expected.apply(params, ids)
    record = {"class": SCALE, "fields": {"scale": 3.0}}
    for model in (tiny(mixer=record),
                  models.build("causal_transformer", **TINY, mixer=record)):
        assert jnp.array_equal(model.apply(params, ids), logits)
    assert not jnp.allclose(tiny(mixer=ScaleMixer(0.0)).apply(params, ids), logits)


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
    swapped = hybrid(mixer=ScaleMixer(scale=0.0),
                     kinds={"linear": LayerKind(mixer={"class": "attention", "fields": {}})})
    plain = hybrid()

    kind_logits = per_kind.apply(per_kind.init(key, ids), ids)
    swapped_logits = swapped.apply(swapped.init(key, ids), ids)
    plain_logits = plain.apply(plain.init(key, ids), ids)

    assert not jnp.allclose(kind_logits, plain_logits)
    assert not jnp.allclose(swapped_logits, plain_logits)
    assert not jnp.allclose(kind_logits, swapped_logits)


def test_models_build_takes_kind_mixer_records():
    """The CLI path builds per-kind mixers through `models.build`."""
    model = models.build(
        "causal_transformer", **TINY, layer_types=("full_attention", "linear"),
        kinds={"linear": {"mixer": {"class": SCALE, "fields": {"scale": 3.0}}}})
    expected = hybrid(kinds={"linear": LayerKind(mixer=ScaleMixer(3.0))})
    ids = jnp.asarray([[1, 2, 3, 4]], jnp.int32)
    params = expected.init(jax.random.key(0), ids)
    assert jnp.array_equal(model.apply(params, ids), expected.apply(params, ids))
