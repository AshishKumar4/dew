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
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from model_support import TINY_DECODER
from reference_error import assert_leaves_as_exact

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.inputs import coordinates
from dew.nn.mixers import MixerBase, MixerContext
from dew.registry import mixers, models

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


DOCUMENTS = (5, 7)
"""Two packed documents' lengths: the first shorter than a conv's history would carry, both uneven."""

PADDING = 3
"""Segment-0 slots after the documents, the packer's tail."""


def packed_and_alone(model, variables, tokens, cotangent):
    """Logits and parameter gradients of the documents' `tokens` `[1, sum(DOCUMENTS)]` run packed
    (segment ids and positions restarting per document, then PADDING segment-0 slots whose logits
    the loss ignores) and of each document run alone, the latter concatenated."""
    total = sum(DOCUMENTS)
    documents = [jnp.full((1, length), i + 1, jnp.int32) for i, length in enumerate(DOCUMENTS)]
    segments = jnp.concatenate([*documents, jnp.zeros((1, PADDING), jnp.int32)], axis=1)
    positions = jnp.concatenate([*(jnp.arange(length)[None] for length in DOCUMENTS),
                                 jnp.zeros((1, PADDING), jnp.int32)], axis=1)
    row = jnp.concatenate([tokens, jnp.zeros((1, PADDING), tokens.dtype)], axis=1)

    def packed(variables):
        return model.apply(variables, row, coordinates=coordinates(
            segments=segments, positions=positions, max_segments=len(DOCUMENTS) + 1))[:, :total]

    def alone(variables):
        starts = (0, *itertools.accumulate(DOCUMENTS))
        return jnp.concatenate([model.apply(variables, tokens[:, begin:end])
                                for begin, end in itertools.pairwise(starts)], axis=1)

    def run(forward):
        logits, pull = jax.vjp(forward, variables)
        return logits, pull(cotangent.astype(logits.dtype))[0]

    assert tokens.shape == (1, total)
    return run(packed), run(alone)


KIND_FIELDS = {"mlp": ({"intermediate_size": 64}, {}), "llama4": ({}, {"qk_norm": False}),
               "deepseek_v4": ({"rope_head_dim": 8}, {"kinds": {"full_attention": LayerKind(window=8)}})}
"""The fields a kind needs beyond its defaults to build in `tiny`: its own, then the model's."""

REFUSED = {"kpool_sparse_attention": "a packed row of documents"}
"""Kinds that refuse packed rows by name, and the name."""


@pytest.mark.parametrize("kind", sorted(mixers.paths))
def test_a_packed_row_computes_what_its_documents_compute_alone(kind):
    """Every mixer in the registry: a row packing two documents (segment ids, positions restarting)
    gives each document's logits and the parameters' gradients its documents give run one at a time,
    within the float64 gate, so no layer carries a conv history, a recurrent state or attention across
    a document boundary, nor between the documents and the segment-0 padding after them. The
    float64 truth is the documents run alone in float64, the fp32 runs at IEEE products, as the
    rule's references take theirs."""
    if jax.default_backend() == "gpu":
        pytest.skip("a GPU's XLA picks GEMM algorithms by shape, so a packed row and its documents "
                    "alone sum in different orders and one draw of each measures luck; isolation on a "
                    "GPU is test_no_token_reaches_another_document's, bitwise")
    fields, overrides = KIND_FIELDS.get(kind, ({}, {}))
    model = tiny(mixer=mixers.build(kind, **fields), max_seq_len=32, **overrides)
    tokens = jax.random.randint(jax.random.key(3), (1, sum(DOCUMENTS)), 0, VOCAB)
    variables = model.init(jax.random.key(4), tokens)
    cotangent = jax.random.normal(jax.random.key(5), (1, sum(DOCUMENTS), VOCAB))
    if kind in REFUSED:
        with pytest.raises(ValueError, match=REFUSED[kind]):
            packed_and_alone(model, variables, tokens, cotangent)
        return
    with jax.default_matmul_precision("highest"):
        (packed, packed_grads), (alone, alone_grads) = packed_and_alone(model, variables, tokens, cotangent)
    with jax.enable_x64(new_val=True):
        wide = jax.tree.map(lambda x: jnp.asarray(np.asarray(x, np.float64)), variables)
        _, (truth, truth_grads) = packed_and_alone(model.clone(dtype=jnp.float64), wide, tokens,
                                                   jnp.asarray(cotangent, jnp.float64))
    assert_leaves_as_exact({"logits": packed, "grads": packed_grads}, {"logits": alone, "grads": alone_grads},
                           {"logits": truth, "grads": truth_grads}, reductions=sum(DOCUMENTS))


@pytest.mark.parametrize("kind", sorted(mixers.paths))
def test_no_token_reaches_another_document(kind):
    """Every mixer in the registry, on every backend: in a packed row (two documents, then segment-0
    padding), changing the first document's tokens and the padding's leaves the second document's
    logits, and the gradients of a loss on them alone, bitwise as they were. The same program on the
    same shapes, so any difference is information crossing a document boundary."""
    if kind in REFUSED:
        pytest.skip(f"{kind} refuses packed rows")
    fields, overrides = KIND_FIELDS.get(kind, ({}, {}))
    model = tiny(mixer=mixers.build(kind, **fields), max_seq_len=32, **overrides)
    first, second = DOCUMENTS
    total = first + second + PADDING
    segments = jnp.asarray([[1] * first + [2] * second + [0] * PADDING], jnp.int32)
    positions = jnp.asarray([[*range(first), *range(second)] + [0] * PADDING], jnp.int32)
    tokens = jax.random.randint(jax.random.key(3), (1, total), 0, VOCAB)
    changed = tokens.at[:, :first].set((tokens[:, :first] + 7) % VOCAB)
    changed = changed.at[:, first + second:].set((changed[:, first + second:] + 5) % VOCAB)
    variables = model.init(jax.random.key(4), tokens)
    weight = jax.random.normal(jax.random.key(5), (1, total, VOCAB))
    weight = weight.at[:, :first].set(0).at[:, first + second:].set(0)
    tokens_at = coordinates(segments=segments, positions=positions, max_segments=3)

    def second_document(row):
        def loss(variables):
            logits = model.apply(variables, row, coordinates=tokens_at)
            return jnp.sum(logits * weight), logits[:, first:first + second]

        (_, logits), grads = jax.value_and_grad(loss, has_aux=True)(variables)
        return logits, grads

    for (path, mine), theirs in zip(jax.tree_util.tree_leaves_with_path(second_document(changed)),
                                    jax.tree.leaves(second_document(tokens)), strict=True):
        np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs),
                                      err_msg=jax.tree_util.keystr(path))


POLICY_VALUES = {"force_fp32_for_softmax": False, "attn_logit_softcap": 30.0, "attention_sinks": True,
                 "value_scale": 0.5, "o_proj_bias": True, "attention_dropout_rate": 0.1}
"""A value each attention policy (`MixerContext.policies`) asks with."""

KIND_CONTEXT = {"llama4": {"qk_norm": False}, "deepseek_v4": {"sliding_window": 8}}
"""Layer fields a kind needs to build at all, beyond the test's context."""

ATTENDS_NOTHING = {"mlp", "gated_delta_net", "kimi_delta_attention", "mamba", "mamba2"}
"""The registry's kinds that mix tokens without attention, whose layers' attention policies are not theirs."""


@pytest.mark.parametrize("kind", sorted(mixers.paths))
@pytest.mark.parametrize("policy", sorted(POLICY_VALUES))
def test_every_attention_mixer_implements_or_refuses_each_policy_it_is_asked(kind, policy):
    """A layer asking an attention policy builds a mixer that takes it as an argument, or refuses
    it by name; no attention-family kind builds without it."""
    if kind in ATTENDS_NOTHING:
        pytest.skip(f"{kind} attends nothing")
    context = MixerContext(emb_features=64, num_heads=4, num_kv_heads=2, head_dim=64, max_seq_len=32,
                           **KIND_CONTEXT.get(kind, {}))
    asked = dataclasses.replace(context, **{policy: POLICY_VALUES[policy]})
    assert asked.policies()[policy] and not context.policies()[policy]
    value = mixers.build(kind)
    value.build(context)
    try:
        factory = value.build(asked)
    except ValueError as refused:
        assert policy in str(refused)
        return
    assert factory.keywords.get(policy) == POLICY_VALUES[policy]
