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
from reference_error import assert_leaves_as_exact, float64_terms

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
    def __call__(self, x, decode: bool = False):
        del decode
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


DOCUMENTS = (5, 70, 7)
"""Three packed documents' lengths, all uneven: the first shorter than a conv's history would carry,
the second across the chunked rules' 64-token chunk boundary, the third starting mid-chunk after it."""

PADDING = 3
"""Segment-0 slots after the documents, the packer's tail."""


def documents_alone(model, tokens):
    """The logits of each document of `tokens` `[1, sum(DOCUMENTS)]` run alone, concatenated, as a
    function of the variables."""
    starts = (0, *itertools.accumulate(DOCUMENTS))

    def alone(variables):
        return jnp.concatenate([model.apply(variables, tokens[:, begin:end])
                                for begin, end in itertools.pairwise(starts)], axis=1)

    return alone


def pulled(forward, variables, cotangent):
    """`forward`'s logits and the variables' gradient under `cotangent`."""
    logits, pull = jax.vjp(forward, variables)
    return logits, pull(cotangent.astype(logits.dtype))[0]


def packed_forward(model, tokens, *, segmented=True):
    """The forward of the documents' `tokens` `[1, sum(DOCUMENTS)]` packed in one row (positions
    restarting per document, then PADDING slots whose logits the loss ignores), their segment ids
    given, or with `segmented` False the same row and positions without them."""
    total = sum(DOCUMENTS)
    assert tokens.shape == (1, total)
    documents = [jnp.full((1, length), i + 1, jnp.int32) for i, length in enumerate(DOCUMENTS)]
    segments = jnp.concatenate([*documents, jnp.zeros((1, PADDING), jnp.int32)], axis=1)
    positions = jnp.concatenate([*(jnp.arange(length)[None] for length in DOCUMENTS),
                                 jnp.zeros((1, PADDING), jnp.int32)], axis=1)
    row = jnp.concatenate([tokens, jnp.zeros((1, PADDING), tokens.dtype)], axis=1)
    tokens_at = (coordinates(segments=segments, positions=positions, max_segments=len(DOCUMENTS) + 1)
                 if segmented else coordinates(positions=positions))

    def forward(variables):
        return model.apply(variables, row, coordinates=tokens_at)[:, :total]

    return forward


def packed_and_alone(model, variables, tokens, cotangent):
    """Logits and parameter gradients of the documents' `tokens` run packed (`packed_forward`) and
    of each document run alone, the latter concatenated (`documents_alone`)."""
    def run(forward):
        return jax.jit(functools.partial(pulled, forward))(variables, cotangent)

    return run(packed_forward(model, tokens)), run(documents_alone(model, tokens))


KIND_FIELDS = {"mlp": ({"intermediate_size": 64}, {}), "llama4": ({}, {"qk_norm": False}),
               "deepseek_v4": ({"rope_head_dim": 8}, {"kinds": {"full_attention": LayerKind(window=8)}}),
               "kimi_delta_attention": ({"linear_num_heads": 2, "linear_head_dim": 16}, {}),
               "gated_delta_net": ({"linear_num_key_heads": 2, "linear_num_value_heads": 4,
                                    "linear_key_head_dim": 16, "linear_value_head_dim": 16}, {})}
"""The fields a kind needs beyond its defaults to build in `tiny`: its own, then the model's. The
recurrent kinds default to their models' heads (Kimi Linear's 64 of 128, Qwen3.5's 32 of 128), whose
per-key pairwise decays alone take 134 MB a chunk in KDA's portable math."""

REFUSED = {"kpool_sparse_attention"}
"""Kinds that refuse packed rows."""


@pytest.mark.parametrize("kind", sorted(mixers.paths))
def test_a_packed_row_computes_what_its_documents_compute_alone(kind):
    """Every mixer in the registry: a row packing three documents (segment ids, positions restarting),
    one across a chunk of the chunked rules, gives each document's logits and the parameters' gradients
    its documents give run one at a time, within the float64 gate, so no layer carries a conv history,
    a recurrent state or attention across a document boundary, nor between the documents and the
    segment-0 padding after them. The float64 truth is the documents run alone in float64, the fp32
    runs at IEEE products, as the rule's references take theirs; each leaf's floor is its own float64
    terms' (`float64_terms`), as a gradient summed over a document's tokens rounds with the partial
    sums it adds: Kimi delta attention's second layer's A_log, its terms 2.7 times its size, rounds
    to 6.6e-06 packed under its floor of 7.3e-06, where the documents alone round to 5.6e-07 at their
    own chunk phase and 6.6e-06 at the packed one."""
    if jax.default_backend() == "gpu":
        pytest.skip("a GPU's XLA picks GEMM algorithms by shape, so a packed row and its documents "
                    "alone sum in different orders and one draw of each measures luck; isolation on a "
                    "GPU is test_no_token_reaches_another_document's, bitwise")
    fields, overrides = KIND_FIELDS.get(kind, ({}, {}))
    model = tiny(mixer=mixers.build(kind, **fields), max_seq_len=128, **overrides)
    tokens = jax.random.randint(jax.random.key(3), (1, sum(DOCUMENTS)), 0, VOCAB)
    variables = model.init(jax.random.key(4), tokens)
    cotangent = jax.random.normal(jax.random.key(5), (1, sum(DOCUMENTS), VOCAB))
    if kind in REFUSED:
        unsegmented = functools.partial(pulled, packed_forward(model, tokens, segmented=False))
        jax.block_until_ready(jax.jit(unsegmented)(variables, cotangent))
        with pytest.raises(ValueError):
            packed_and_alone(model, variables, tokens, cotangent)
        return
    with jax.default_matmul_precision("highest"):
        (packed, packed_grads), (alone, alone_grads) = packed_and_alone(model, variables, tokens, cotangent)
    with jax.enable_x64(new_val=True):
        wide = jax.tree.map(lambda x: jnp.asarray(np.asarray(x, np.float64)), variables)
        alone64 = documents_alone(model.clone(dtype=jnp.float64), tokens)
        (truth, truth_grads), (terms, terms_grads) = float64_terms(
            functools.partial(pulled, alone64), wide, jnp.asarray(cotangent, jnp.float64))
    assert_leaves_as_exact({"logits": packed, "grads": packed_grads}, {"logits": alone, "grads": alone_grads},
                           {"logits": truth, "grads": truth_grads}, reductions=sum(DOCUMENTS),
                           terms={"logits": terms, "grads": terms_grads})


def embedded(module, tokens):
    """The scaled token embeddings the model's first layer reads, which `input_embeddings` replace."""
    return module.scaled_embeddings(module.token_embeddings(tokens))


@pytest.mark.parametrize("kind", sorted(mixers.paths))
def test_no_token_reaches_another_document(kind):
    """Every mixer in the registry, on every backend: in a packed row (DOCUMENTS' first two, then
    segment-0 padding), changing the first document's tokens and the padding's leaves the second's
    logits, the parameters' gradients of a loss on them alone and its gradient at the second's
    embedded tokens bitwise as they were. The same program on the same shapes, so any difference is
    information crossing a document boundary.

    The loss is differentiated at the embedded tokens, not through the lookup: the table's gradient
    adds each position's by a scatter, which a GPU's deterministic XLA (`--xla_gpu_deterministic_ops`)
    sums in an order every id in the row decides, so new ids in the first document reorder the
    second's sums though nothing crosses."""
    if kind in REFUSED:
        pytest.skip(f"{kind} refuses packed rows")
    fields, overrides = KIND_FIELDS.get(kind, ({}, {}))
    model = tiny(mixer=mixers.build(kind, **fields), max_seq_len=128, **overrides)
    first, second = DOCUMENTS[:2]
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
        def loss(variables, inputs):
            logits = model.apply(variables, row, input_embeddings=inputs, coordinates=tokens_at)
            return jnp.sum(logits * weight), logits[:, first:first + second]

        inputs = model.apply(variables, row, method=embedded)
        (_, logits), (grads, at_inputs) = jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))(
            variables, inputs)
        return logits, grads, at_inputs[:, first:first + second]

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
    """A layer asking an attention policy builds a mixer that takes it, as an argument or in the
    context it holds (`CausalSelfAttention.ctx`), or refuses it by name; no attention-family kind
    builds without it."""
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
    held = factory.keywords.get("ctx")
    taken = factory.keywords.get(policy) if held is None else getattr(held, policy)
    assert taken == POLICY_VALUES[policy]


@pytest.mark.parametrize("kind", ["kimi_delta_attention", "mamba", "mamba2", "gated_delta_net"])
def test_a_remat_policy_keeps_every_residual_a_recurrent_mixer_names(kind, monkeypatch, capsys):
    """A decoder skips the remat rungs that keep none of its mixers' names (`MixerBase.residuals`), so
    a kind may name only what its layer marks: under `jax.checkpoint` with a policy saving all of them,
    a tiny decoder of that kind saves each one for its backward. A delta rule's decay and inverse are
    its port's (`row_names_port`, interpreted off a GPU)."""
    from jax.ad_checkpoint import print_saved_residuals

    from dew.nn import linear
    from dew.nn.backbones.decoder_block import RematPolicy

    monkeypatch.setattr(linear, "measured_kernel",
                        lambda choice, default: "fla" if choice == "gated_delta_rule" else default)
    value = mixers.build(kind, **KIND_FIELDS.get(kind, ({}, {}))[0])
    assert value.residuals is not None
    model = tiny(mixer=value, max_seq_len=64, **KIND_FIELDS.get(kind, ({}, {}))[1])
    tokens = jax.random.randint(jax.random.key(3), (1, 64), 0, VOCAB)
    variables = model.init(jax.random.key(4), tokens)
    policy = RematPolicy(save=value.residuals).checkpoint_policy()

    def loss(variables):
        return jnp.sum(model.apply(variables, tokens) ** 2)

    capsys.readouterr()
    print_saved_residuals(jax.checkpoint(loss, policy=policy), variables)
    saved = capsys.readouterr().out
    assert [name for name in value.residuals if f"named '{name}'" not in saved] == []
