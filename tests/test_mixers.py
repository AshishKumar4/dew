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
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from delta_rule_support import DECAY_SUMS, port_formula
from flax import linen as nn
from model_support import TINY_DECODER
from reference_error import assert_leaves_as_exact, float64_terms

from dew.nn import linear
from dew.nn.attention import alibi_bias
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import BlockWiring, DecoderBlock
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.inputs import TOKENWISE, AttentionMetadata, MixerCall, coordinates
from dew.nn.kernels.generation import chosen
from dew.nn.mixers import AttentionMixer, MixerBase, MixerContext
from dew.nn.mixers.gated_delta_net import GatedDeltaNetMixer
from dew.registry import mixers, models

VOCAB = 37
TINY = {"vocab_size": VOCAB, **TINY_DECODER}


def tiny(**overrides):
    return CausalTransformer(**{**TINY, **overrides})


class ScaleMixerModule(nn.Module):
    """A token mixer with no parameters: scale and pass through, token by token."""

    reads: ClassVar[frozenset[str]] = TOKENWISE

    scale: float = 2.0

    @nn.compact
    def __call__(self, x, call: MixerCall):
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


def test_a_layer_runs_no_mask_its_mixer_does_not_read():
    """A pairwise mask says which keys each query reads. An attention layer reads it, so a mask
    that keeps only each token's own key moves its output; a gated delta net reads every earlier
    token of its document and no mask, so its layer refuses the call rather than run it as if
    unmasked. A model asked for a mask refuses it at the layer that cannot read it."""
    x = jax.random.normal(jax.random.key(1), (1, 4, 32))
    own = AttentionMetadata(pairwise_mask=jnp.eye(4, dtype=bool)[None])
    context = MixerContext(emb_features=32, num_heads=4, num_kv_heads=4, head_dim=8, max_seq_len=16)
    delta = GatedDeltaNetMixer(linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8,
                               linear_value_head_dim=8)
    for kind, reads in ((AttentionMixer(), True), (delta, False)):
        layer = DecoderBlock(mixer=kind.build(context), feedforward=None, emb_features=32,
                             wiring=BlockWiring())
        params = layer.init(jax.random.key(0), x)
        if reads:
            assert not jnp.allclose(layer.apply(params, x, attention_metadata=own), layer.apply(params, x))
        else:
            with pytest.raises(ValueError):
                layer.apply(params, x, attention_metadata=own)
    recurrent = hybrid(kinds={"linear": LayerKind(mixer=delta)})
    ids = jnp.asarray([[1, 2, 3, 4]], jnp.int32)
    params = recurrent.init(jax.random.key(0), ids)
    with pytest.raises(ValueError):
        recurrent.apply(params, ids, attention_pairwise_mask=own.pairwise_mask)


def test_alibi_biases_rotated_attention_by_its_keys_places():
    """ALiBi and rotation are independent: a rotary layer with `alibi` attends
    as the same rotary layer handed ALiBi's bias at its keys' places as its
    position bias, bitwise, and not as the rotary layer alone does."""
    x = jax.random.normal(jax.random.key(1), (2, 6, 32))
    context = MixerContext(emb_features=32, num_heads=4, num_kv_heads=2, head_dim=8, max_seq_len=16)
    with_alibi, rotary = (DecoderBlock(mixer=kind.build(context), feedforward=None, emb_features=32,
                                       wiring=BlockWiring())
                          for kind in (AttentionMixer(alibi=True), AttentionMixer()))
    params = rotary.init(jax.random.key(0), x)
    bias = AttentionMetadata(position_bias=alibi_bias(jnp.arange(6), 4, dtype=x.dtype))
    biased = with_alibi.apply(params, x)
    np.testing.assert_array_equal(np.asarray(biased),
                                  np.asarray(rotary.apply(params, x, attention_metadata=bias)))
    assert not np.allclose(biased, rotary.apply(params, x))


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


DELTA_RULES = frozenset({"gated_delta_net", "kimi_delta_attention"})
"""The registry's kinds on the delta rule, which a GPU runs on its port (`linear.fla_rule`)."""

@pytest.mark.parametrize("kind", sorted(mixers.paths))
def test_no_token_reaches_another_document(kind, monkeypatch):
    """Every mixer in the registry, on every backend: in a packed row (DOCUMENTS' first two, then
    segment-0 padding), changing the first document's tokens and the padding's leaves the second's
    logits, the parameters' gradients of a loss on them alone and its gradient at the second's
    embedded tokens bitwise as they were. The same program on the same shapes, so any difference is
    information crossing a document boundary. The delta rule's port runs a row as one sequence, each
    document's decay prefix shifted by the one before (`linear.FORGET`), so there the second
    document's values move by that prefix's rounding: they are held to float64 within the distance of
    the same changed row with the port's formula in the portable math (`port_formula`), whose ranges
    round as the port's do, and which a token crossing the boundary would miss by orders of magnitude,
    but the per-head decay parameters (`DECAY_SUMS`). The unchanged row is no reference there: the
    first document's new decays redraw the second's.

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

    def second_document(row, model=model, variables=variables):
        def loss(variables, inputs):
            logits = model.apply(variables, row, input_embeddings=inputs, coordinates=tokens_at)
            return jnp.sum(logits * weight), logits[:, first:first + second]

        inputs = model.apply(variables, row, method=embedded)
        (_, logits), (grads, at_inputs) = jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))(
            variables, inputs)
        return logits, grads, at_inputs[:, first:first + second]

    moved = second_document(changed)
    if kind in DELTA_RULES and linear.measured_kernel("gated_delta_rule", "xla") == "fla":
        with monkeypatch.context() as patched:
            patched.setattr(linear, "fla_rule", port_formula)
            formula = second_document(changed)
        with jax.enable_x64(new_val=True):
            widened = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64), variables)
            truth = second_document(tokens, model.clone(dtype=jnp.float64), widened)

        def held(tree):
            return {name: leaf for name, leaf in ((jax.tree_util.keystr(path), leaf)
                                                  for path, leaf in jax.tree_util.tree_leaves_with_path(tree))
                    if not any(decay in name for decay in DECAY_SUMS)}

        assert_leaves_as_exact(held(moved), held(formula), held(truth))
        return
    for (path, mine), theirs in zip(jax.tree_util.tree_leaves_with_path(moved),
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


@pytest.mark.parametrize("kind, rungs", [
    ("attention", [None, "minimal", "save_dot_except_mlp", "full"]),
    ("mamba2", [None, "minimal_with_conv", "minimal", "full"]),
    ("gated_delta_net", [None, "all_but_delta_memory", "minimal_with_conv", "minimal", "full"]),
])
def test_a_decoder_climbs_only_the_remat_rungs_its_layers_mark(kind, rungs):
    """`recompute_more` passes a rung that keeps the same of the decoder's marked names as the rung
    below or above it: 'all_but_delta_memory', which recomputes a delta rule's memory alone, is a
    rung only for a decoder with a delta rule, and the ladder of any other is the one before it (a
    recurrent decoder's 'save_dot_except_mlp' keeps no more of its names than 'full')."""
    fields, overrides = KIND_FIELDS.get(kind, ({}, {}))
    model = tiny(mixer=mixers.build(kind, **fields), max_seq_len=64, **overrides)
    climbed = [model]
    while (stronger := climbed[-1].recompute_more()) is not None:
        climbed.append(stronger)
    assert [rung.recompute_record() for rung in climbed] == rungs


@pytest.mark.parametrize("kind", ["kimi_delta_attention", "mamba", "mamba2", "gated_delta_net"])
@chosen(gated_delta_rule="fla")
def test_a_remat_policy_keeps_every_residual_a_recurrent_mixer_names(kind, capsys):
    """A decoder skips the remat rungs that keep none of its mixers' names (`MixerBase.residuals`), so
    a kind may name only what its layer marks: under `jax.checkpoint` with a policy saving all of them,
    a tiny decoder of that kind saves each one for its backward. A delta rule's decay and inverse are
    its port's (`row_names_port`, interpreted off a GPU)."""
    from jax.ad_checkpoint import print_saved_residuals

    from dew.nn.backbones.decoder_block import RematPolicy

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


def one_tap(kind):
    """A recurrent mixer of `kind` whose causal conv takes one tap, at small widths."""
    from dew.nn.kda import KimiDeltaAttention
    from dew.nn.mixers.mamba import Mamba
    from dew.nn.mixers.mamba2 import Mamba2

    return {"gated_delta_net": lambda: linear.GatedDeltaNet(emb_features=64, num_k_heads=2, num_v_heads=4,
                                                            head_k_dim=16, head_v_dim=16, conv_kernel=1),
            "kimi_delta_attention": lambda: KimiDeltaAttention(emb_features=64, num_heads=2, head_dim=16,
                                                               conv_kernel=1),
            "mamba": lambda: Mamba(emb_features=16, intermediate_size=32, state_size=4, time_step_rank=2,
                                   chunk_size=4, conv_kernel=1),
            "mamba2": lambda: Mamba2(emb_features=64, num_heads=4, head_dim=16, state_size=16, chunk_size=8,
                                     conv_kernel=1)}[kind]()


@pytest.mark.parametrize("kind", ["gated_delta_net", "kimi_delta_attention", "mamba", "mamba2"])
def test_a_recurrent_mixer_whose_conv_takes_one_tap_decodes_as_its_whole_pass(kind):
    """A one-tap causal conv is a scale per channel with no history to hold: each recurrent mixer
    builds with one, and its decode (a 3-token prefill, then one token a step) is within
    tests/reference_error.py's rule of its whole pass against the float64 whole pass."""
    from model_support import decode_through
    from reference_error import assert_as_exact_as_the_reference

    module = one_tap(kind)
    hidden = jax.random.normal(jax.random.key(4), (2, 9, module.emb_features), jnp.float32)
    variables = module.init(jax.random.key(5), hidden)
    whole = np.asarray(module.apply(variables, hidden), np.float64)
    with jax.enable_x64(new_val=True):
        wide = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64), variables)
        truth = np.asarray(module.clone(dtype=jnp.float64).apply(wide, jnp.asarray(hidden, jnp.float64)))
    decoded = np.asarray(decode_through(module, variables, hidden, prefill=3)[0], np.float64)
    assert_as_exact_as_the_reference(decoded, whole, truth, f"{kind} decode")
