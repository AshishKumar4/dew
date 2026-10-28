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
import pickle
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from model_support import TINY_DECODER
from reference_error import assert_as_exact, widened, with_gradients
from residual_orders import orders, permuted, residual_width

from dew.batch import coordinates
from dew.nn import linear
from dew.nn.attention import alibi_bias
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import BlockWiring, DecoderBlock
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.cross_attention import CrossAttention
from dew.nn.inputs import TOKENWISE, AttentionMetadata, MixerCall
from dew.nn.kernels.generation import chosen
from dew.nn.memory import Gate, InContextMemory, MLPMemory
from dew.nn.mixer_base import MLP_RESIDUALS, MixerRefused
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


OWN_KEYS = {"pairwise_mask": AttentionMetadata(pairwise_mask=jnp.eye(4, dtype=bool)[None]),
            "key_runs": AttentionMetadata(key_runs=jnp.stack([jnp.arange(4), jnp.arange(4) + 1],
                                                             axis=-1)[None, :, None])}
"""A call whose every token reads only its own key, as a pairwise mask and as key runs."""


@pytest.mark.parametrize("own", OWN_KEYS.values(), ids=list(OWN_KEYS))
def test_a_layer_runs_no_mask_its_mixer_does_not_read(own):
    """A pairwise mask, or key runs, says which keys each query reads. An attention layer reads it, so
    a call that keeps only each token's own key moves its output; a gated delta net reads every earlier
    token of its document and no mask, so its layer refuses the call rather than run it as if
    unmasked. A model asked for a mask refuses it at the layer that cannot read it."""
    x = jax.random.normal(jax.random.key(1), (1, 4, 32))
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
        recurrent.apply(params, ids, attention_pairwise_mask=own.pairwise_mask,
                        attention_key_runs=own.key_runs)


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


def documents_alone(model, tokens, documents=DOCUMENTS):
    """The logits of each document of `tokens` `[1, sum(documents)]` run alone, concatenated, as a
    function of the variables."""
    starts = (0, *itertools.accumulate(documents))

    def alone(variables):
        return jnp.concatenate([model.apply(variables, tokens[:, begin:end])
                                for begin, end in itertools.pairwise(starts)], axis=1)

    return alone


def pulled(forward, variables, cotangent):
    """`forward`'s logits and the variables' gradient under `cotangent`, at the logits' dtype, pulled
    back through the gate (`reference_error.with_gradients`)."""
    logits_dtype = jax.eval_shape(forward, variables).dtype
    return with_gradients(forward)(variables, cotangent.astype(logits_dtype))


def packed_forward(model, tokens, *, documents=DOCUMENTS, segmented=True):
    """The forward of the documents' `tokens` `[1, sum(documents)]` packed in one row (positions
    restarting per document, then PADDING slots whose logits the loss ignores), their segment ids
    given, or with `segmented` False the same row and positions without them."""
    total = sum(documents)
    assert tokens.shape == (1, total)
    segments = [jnp.full((1, length), i + 1, jnp.int32) for i, length in enumerate(documents)]
    segments = jnp.concatenate([*segments, jnp.zeros((1, PADDING), jnp.int32)], axis=1)
    positions = jnp.concatenate([*(jnp.arange(length)[None] for length in documents),
                                 jnp.zeros((1, PADDING), jnp.int32)], axis=1)
    row = jnp.concatenate([tokens, jnp.zeros((1, PADDING), tokens.dtype)], axis=1)
    tokens_at = (coordinates(segments=segments, positions=positions, max_segments=len(documents) + 1)
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
               "kimi_delta_attention": ({"linear_num_heads": 2, "linear_head_dim": 32}, {}),
               "mamba2": ({"num_heads": 8, "head_dim": 32, "state_size": 32, "n_groups": 4,
                           "chunk_size": 64}, {}),
               "gated_delta_net": ({"linear_num_key_heads": 2, "linear_num_value_heads": 4,
                                    "linear_key_head_dim": 16, "linear_value_head_dim": 16}, {}),
               "memory": ({"memory": InContextMemory(
                   network=MLPMemory(hidden=(16,)), heads=2, head_features=8, chunk=4, normalize=True,
                   learning_rate=Gate(scale=0.1), momentum=Gate(), retention=Gate(bias=-2.0))}, {})}
"""The fields a kind needs beyond its defaults to build in `tiny`: its own, then the model's. The
recurrent kinds default to their models' heads (Kimi Linear's 64 of 128, Qwen3.5's 32 of 128), whose
per-key pairwise decays alone take 134 MB a chunk in KDA's portable math."""

REFUSED = {"kpool_sparse_attention"}
"""Kinds that refuse packed rows."""

SEEDS = {"attention": 64, "deepseek_v4": 256, "gated_delta_net": 512, "kimi_delta_attention": 512,
         "llama4": 128, "mamba": 32, "mamba2": 256, "memory": 256, "mla": 64, "mlp": 64}
"""Each kind's declared seeds for check 1 (`assert_as_exact`), sized by `seeds_for`: the fewest for which
documents run alone compared with themselves resolve PARITY in every logit and parameter gradient (on
armada). Mamba2's 8 heads of 32 over a state of 32 in 64-token chunks (`KIND_FIELDS`) take 256 at 0.04 s a
seed, where its defaults (128 heads of 64 over 128) took 256 at 30 minutes a run. Kimi delta attention's
heads are 32 wide (`KIND_FIELDS`): at 16 the reference compared with itself did not resolve within 2048
seeds, one seed carrying up to 68% of a leaf's summed error, near-collinear keys' erases in fp32; at 32 it
carries at most 5%. 16 is no path of its own: the portable math this site runs has no width-specific code,
and the port pads every key width below 64 to 64 (`recurrence_chunk._blocked_rule`), which
tests/test_fla_gdn.py's any-width site holds at 32. The packed row no longer runs a 16-wide Kimi layer or
Mamba2's defaults (128 heads of 64 over a state of 128 in 256-token chunks, where these documents sit in
one chunk); the scan's packed resets within a chunk stay held by tests/test_mamba2.py and
tests/test_mamba_ssd.py at their own geometries."""


@pytest.mark.parametrize("kind", sorted(mixers.paths))
def test_a_packed_row_computes_what_its_documents_compute_alone(kind):
    """Every mixer in the registry: a row packing documents (segment ids, positions restarting),
    one across a chunk of the chunked rules, gives each document's logits and the parameters' gradients
    its documents give run one at a time, within the float64 gate, so no layer carries a conv history,
    a recurrent state or attention across a document boundary, nor between the documents and the
    segment-0 padding after them. Each draw has fresh tokens, parameters and cotangents, with the
    residual channels consistently permuted (`residual_orders`). Both fp32 runs use IEEE products;
    the float64 truth runs each document alone under the same order."""
    fields, overrides = KIND_FIELDS.get(kind, ({}, {}))
    documents = DOCUMENTS
    total = sum(documents)
    model = tiny(mixer=mixers.build(kind, **fields), max_seq_len=max(128, total + PADDING), **overrides)
    tokens = jax.random.randint(jax.random.key(3), (1, total), 0, VOCAB)
    variables = model.init(jax.random.key(4), tokens)
    cotangent = jax.random.normal(jax.random.key(5), (1, total, VOCAB))
    if kind in REFUSED:
        unsegmented = functools.partial(pulled, packed_forward(model, tokens, segmented=False))
        jax.block_until_ready(jax.jit(unsegmented)(variables, cotangent))
        with pytest.raises(ValueError):
            packed_and_alone(model, variables, tokens, cotangent)
        return
    initialize = jax.jit(model.init)
    width = residual_width(variables)

    def draw(seed):
        tokens = jax.random.randint(jax.random.key(3 + 3 * seed), (1, total), 0, VOCAB)
        variables = initialize(jax.random.key(4 + 3 * seed), tokens)
        cotangent = jax.random.normal(jax.random.key(5 + 3 * seed), (1, total, VOCAB))
        order = orders(width, 2, seed)[1]
        return permuted(variables, order), tokens, cotangent

    def evaluated(model, variables, tokens, cotangent, *, packed):
        forward = (packed_forward(model, tokens, documents=documents) if packed
                   else documents_alone(model, tokens, documents))
        with jax.default_matmul_precision("highest"):
            logits, grads = pulled(forward, variables, cotangent)
        return {"logits": logits, "grads": grads}

    dew = jax.jit(functools.partial(evaluated, model, packed=True))
    reference = jax.jit(functools.partial(evaluated, model, packed=False))
    truth = jax.jit(functools.partial(evaluated, model.clone(dtype=jnp.float64), packed=False))
    assert_as_exact(dew, reference, draw, f"packed {kind}", seeds=SEEDS[kind], truth_fn=truth)


def embedded(module, tokens):
    """The scaled token embeddings the model's first layer reads, which `input_embeddings` replace."""
    return module.scaled_embeddings(module.token_embeddings(tokens))


@pytest.mark.parametrize("kind", sorted(mixers.paths))
def test_no_token_reaches_another_document(kind):
    """Every mixer in the registry, on every backend: in a packed row (DOCUMENTS' first two, then
    segment-0 padding), changing the first document's tokens and the padding's leaves the second's
    logits, the parameters' gradients of a loss on them alone and its gradient at the second's
    embedded tokens bitwise as they were. The same program on the same shapes, so any difference is
    information crossing a document boundary. The delta rule's port runs each document as a sequence
    of its own, so it holds bitwise too.

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
    for (path, mine), theirs in zip(jax.tree_util.tree_leaves_with_path(moved),
                                    jax.tree.leaves(second_document(tokens)), strict=True):
        np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs),
                                      err_msg=jax.tree_util.keystr(path))


POLICY_VALUES = {"force_fp32_for_softmax": False, "attn_logit_softcap": 30.0, "attention_sinks": True,
                 "value_scale": 0.5, "o_proj_bias": True, "attention_dropout_rate": 0.1}
"""A value each attention policy (`MixerContext.policies`) asks with."""

KIND_CONTEXT = {"llama4": {"qk_norm": False}, "deepseek_v4": {"sliding_window": 8}}
"""Layer fields a kind needs to build at all, beyond the test's context."""

ATTENDS_NOTHING = {"mlp", "gated_delta_net", "kimi_delta_attention", "mamba", "mamba2", "memory"}
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


def test_a_mixers_refusal_names_its_kind_and_entries_across_a_process_boundary():
    """`MixerContext.refuse` raises `MixerRefused`, a ValueError naming the mixer, the context entries
    the layer set and why; pickled, as a process carries an exception across, it is the same refusal."""
    context = MixerContext(emb_features=64, num_heads=4, num_kv_heads=2, head_dim=64, max_seq_len=32)
    with pytest.raises(MixerRefused) as refused:
        context.refuse("mla", {"kv_cache": True, "attention_sinks": False}, "its latents are its cache")
    again = pickle.loads(pickle.dumps(refused.value))
    assert type(again) is MixerRefused and again.args == refused.value.args
    assert (again.kind, again.names, again.why) == ("mla", ("kv_cache",), "its latents are its cache")


@pytest.mark.parametrize("kind, rungs", [
    ("attention", [None, "minimal", "save_dot_except_mlp", "full"]),
    ("mamba2", [None, "minimal_with_scan_and_conv", "minimal_with_conv", "minimal", "full"]),
    ("gated_delta_net", [None, "all_but_delta_memory", "minimal_with_scan_and_conv", "minimal_with_conv",
                         "minimal", "full"]),
])
def test_a_decoder_climbs_only_the_remat_rungs_its_layers_mark(kind, rungs):
    """`recompute_more` passes a rung that keeps the same of the decoder's marked names as the rung
    below or above it: 'all_but_delta_memory', which recomputes a delta rule's memory alone, is a
    rung only for a decoder with a delta rule, 'minimal_with_scan_and_conv', which keeps a
    recurrence's scan, only for a decoder with a recurrence, so attention's ladder is the one before
    either (a recurrent decoder's 'save_dot_except_mlp' keeps no more of its names than 'full')."""
    fields, overrides = KIND_FIELDS.get(kind, ({}, {}))
    model = tiny(mixer=mixers.build(kind, **fields), max_seq_len=64, **overrides)
    climbed = [model]
    while (stronger := climbed[-1].recompute_more()) is not None:
        climbed.append(stronger)
    assert [rung.recompute_record() for rung in climbed] == rungs


def test_a_cross_attention_feed_forward_keeps_its_rungs_where_the_trunk_has_none():
    """A decoder with no trunk MLP, Mamba-2's layout, marks its gated cross-attention's feed-forward
    (`marked_residuals`): 'minimal', which keeps that MLP's projections, climbs on to 'full', which
    recomputes them, whether it starts there or a record restores it."""
    fields, overrides = KIND_FIELDS["mamba2"]
    model = tiny(mixer=mixers.build("mamba2", **fields), max_seq_len=64, mlp_features=0,
                 cross_attention=CrossAttention(layer_ids=(0,), context_length=4, context_features=16,
                                                intermediate_size=32), **overrides)
    assert set(MLP_RESIDUALS) <= model.marked_residuals()
    minimal = model.restore_recompute(model.clone(remat="minimal").recompute_record())
    climbed = minimal.recompute_more()
    assert climbed is not None and climbed.recompute_record() == "full"


@pytest.mark.parametrize("kind", ["kimi_delta_attention", "mamba", "mamba2", "gated_delta_net"])
@chosen(gated_delta_rule="fla")
def test_a_remat_policy_keeps_every_residual_a_recurrent_mixer_names(kind, capsys):
    """A decoder skips the remat rungs that keep none of its mixers' names (`MixerBase.marks`), so
    a kind may name only what its layer marks: under `jax.checkpoint` with a policy saving each one
    alone, a tiny decoder of that kind saves it for its backward. A recurrence's output is both its
    'context' and its 'scan', either of which keeps it. A delta rule's decay and inverse are its
    port's (`row_names_port`, interpreted off a GPU)."""
    from jax.ad_checkpoint import print_saved_residuals

    from dew.nn.backbones.decoder_block import RematPolicy

    value = mixers.build(kind, **KIND_FIELDS.get(kind, ({}, {}))[0])
    assert value.marks() is not None
    model = tiny(mixer=value, max_seq_len=64, **KIND_FIELDS.get(kind, ({}, {}))[1])
    tokens = jax.random.randint(jax.random.key(3), (1, 64), 0, VOCAB)
    variables = model.init(jax.random.key(4), tokens)

    def loss(variables):
        return jnp.sum(model.apply(variables, tokens) ** 2)

    def saves(name):
        capsys.readouterr()
        policy = RematPolicy(save=(name,)).checkpoint_policy()
        print_saved_residuals(jax.checkpoint(loss, policy=policy), variables)
        return f"named '{name}'" in capsys.readouterr().out

    assert [name for name in value.marks() or () if not saves(name)] == []


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


ONE_TAP_SEEDS = {"gated_delta_net": 8, "kimi_delta_attention": 8, "mamba": 8, "mamba2": 8}
"""Each one-tap mixer's declared draws for its decode against its whole pass (check 1)."""


@pytest.mark.parametrize("kind", ["gated_delta_net", "kimi_delta_attention", "mamba", "mamba2"])
def test_a_recurrent_mixer_whose_conv_takes_one_tap_decodes_as_its_whole_pass(kind):
    """A one-tap causal conv is a scale per channel with no history to hold: each recurrent mixer
    builds with one, and its decode (a 3-token prefill, then one token a step) is on par with its
    whole pass from the float64 whole pass over drawn inputs (check 1)."""
    from model_support import decode_through

    module = one_tap(kind)
    shape = (2, 9, module.emb_features)
    variables = module.init(jax.random.key(5), jnp.zeros(shape, jnp.float32))

    def draw(seed):
        return (jax.random.normal(jax.random.key(seed), shape, jnp.float32),)

    def truth(hidden):
        return module.clone(dtype=jnp.float64).apply(widened(variables), hidden)

    assert_as_exact(lambda hidden: decode_through(module, variables, hidden, prefill=3)[0],
                    lambda hidden: module.apply(variables, hidden), draw, f"{kind} decode",
                    seeds=ONE_TAP_SEEDS[kind], truth_fn=truth)
