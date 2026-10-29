"""Cross-attention through each layer's own attention (`dew.nn.cross_attention.SharedCrossAttention`):
the layer reads the conditioning sequence with its own query, key, value and output projections and
norms, and joins the read through one tanh gate, the only weight the form adds."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from model_support import flat_tree
from reference_error import assert_as_exact
from test_cross_attention import (
    MODEL,
    SEQ,
    VOCAB,
    B,
    _without_blocks,
    assert_a_cached_context_decodes_as_the_full_forward,
    float64,
    fresh_against_base,
    logits,
    model_draw,
)

from dew.batch import FROZEN
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.cross_attention import CrossAttention, SharedCrossAttention, crossing
from dew.nn.inputs import CONTEXT, CONTEXT_LENGTHS
from dew.nn.mixers.mamba2 import Mamba2Mixer

SHARED = SharedCrossAttention(layer_ids=(0, 2), context_length=6)
SHARED_MODEL = MODEL.clone(cross_attention=SHARED)
"""The shared form over a decoder whose attention norms its queries and keys (qk_norm), which its reads
of a context norm alike."""
TIED = CrossAttention(layer_ids=(0, 2), context_length=6)
"""IDEFICS's block at the decoder's own width, which computes the shared form over a decoder whose
attention norms no query or key where its attention is the layer's, its input norm the layer's and its
feed-forward's gate zero (`tied`)."""
UNNORMED = MODEL.clone(qk_norm=False)

SHARED_SEEDS = {"shared fresh": 8, "shared idefics": 8, "shared decode": 8, "shared scan": 8,
                "alternating scan": 8}
"""Check 1's declared seeds for each site of the file, sized paired by `seeds_for` on armada at
edf94b090 (job IGXSIZE) and, for the IDEFICS read and the alternating scan, at db0cc076a (IGXSIZE3)."""


def test_a_fresh_shared_read_computes_the_decoder_it_was_attached_to():
    """Drawn at zero gates, the decoder reading a context through its layers' own attention computes the
    decoder without the reads (check 1), and its gates are the only weights the form adds."""
    fresh_against_base(SHARED_MODEL, "shared fresh", seeds=SHARED_SEEDS["shared fresh"])
    variables, *_ = model_draw(SHARED_MODEL, gated=False)(0)
    added = {path for path in flat_tree(variables["params"]) if crossing(tuple(path.split(".")))}
    assert {path.rsplit(".", 1)[-1] for path in added} == {"alpha_cross_attn"}
    assert len(added) == len(SHARED.layer_ids)


TIED_MODEL = UNNORMED.clone(cross_attention=TIED)
SHARED_UNNORMED = UNNORMED.clone(cross_attention=SHARED)

SHARED_LEAVES = {f"cross_attention.{name}.kernel": f"self_attn.{name}.kernel"
                 for name in ("q_proj", "k_proj", "v_proj", "o_proj")} | {
    "cross_attention.input_layernorm.scale": "input_layernorm.scale",
    "cross_attention.alpha_cross_attn": "cross_attention.alpha_cross_attn"}
"""Each IDEFICS block leaf (`TIED`) the shared model's own leaf it takes, in the same layer."""


def tied(variables):
    """The shared model's `variables` as IDEFICS's blocks (`TIED`) hold them: each block's attention
    the layer's own projections, its input norm the layer's and its read's gate the shared one
    (`SHARED_LEAVES`), its feed-forward's gate zero, which leaves the feed-forward's maps out, and
    every other leaf the shared model's own."""
    own = flat_tree(variables["params"])
    shapes = flat_tree(jax.eval_shape(TIED_MODEL.init, jax.random.key(0),
                                      jnp.zeros((1, 1), jnp.int32))["params"])

    def leaf(path, shape):
        if path in own:
            return own[path]
        layer, _, rest = path.partition(".cross_attention.")
        name = f"cross_attention.{rest}"
        if name in SHARED_LEAVES:
            return own[f"{layer}.{SHARED_LEAVES[name]}"]
        filled = jnp.zeros if rest == "alpha_dense" else jnp.ones
        return filled(shape.shape, shape.dtype)

    params = {}
    for path, shape in shapes.items():
        *parents, last = path.split(".")
        node = params
        for part in parents:
            node = node.setdefault(part, {})
        node[last] = leaf(path, shape)
    return {**variables, "params": params}




def read(model: CausalTransformer, held=lambda variables: variables):
    """`model`'s logits over a draw reading its context and their pullback into the context under the
    draw's cotangent, from its variables as `held` gives them, compiled."""
    run = logits(model)

    @jax.jit
    def pulled(variables, ids, context, lengths, cotangent):
        out, pull = jax.vjp(lambda context: run(held(variables), ids, context, lengths), context)
        return out, *pull(cotangent)
    return pulled


def read_draw(model: CausalTransformer):
    """`model_draw`'s draw with a cotangent on the logits."""
    draw = model_draw(model)

    def drawn(seed):
        rng = np.random.default_rng([seed, 1])
        return (*draw(seed), rng.standard_normal((B, SEQ, VOCAB)).astype(np.float32))
    return drawn


def test_a_shared_read_is_idefics_block_over_the_layers_own_attention():
    """Gated, each layer's read through its own attention is IDEFICS's gated block whose attention is
    the layer's own projections and whose input norm is the layer's, its feed-forward gated off: the
    layer's query over its normed residual and its keys and values over the context, unrotated and
    uncausal, each row's real context tokens, then its output projection. The logits and their pullback
    into the context on par with IDEFICS's block from its float64 (check 1), over a decoder whose
    attention norms no query or key, which IDEFICS's block does not either; a decoder that norms them
    norms the read's alike, through the projections' own code (`context_query`, `context_keys`)."""
    assert_as_exact(read(SHARED_UNNORMED), read(TIED_MODEL, tied), read_draw(SHARED_UNNORMED),
                    "shared idefics", seeds=SHARED_SEEDS["shared idefics"],
                    truth_fn=read(float64(TIED_MODEL), tied))
    variables, ids, context, lengths = model_draw(SHARED_MODEL)(1)
    conditioning = {CONTEXT: context, CONTEXT_LENGTHS: lengths}
    moved = np.asarray(SHARED_MODEL.apply(variables, ids, conditioning=conditioning))
    base = dataclasses.replace(SHARED_MODEL, cross_attention=None)
    assert not np.array_equal(moved[lengths > 0],
                              np.asarray(base.apply(_without_blocks(variables), ids))[lengths > 0])
    gradients = jax.grad(lambda params: jnp.sum(SHARED_MODEL.apply(
        {**variables, "params": params}, ids, conditioning=conditioning)))(variables["params"])
    gates = [np.asarray(leaf) for path, leaf in flat_tree(gradients).items() if "alpha_" in path]
    assert len(gates) == len(SHARED.layer_ids)
    assert all(np.isfinite(gate).all() and np.all(gate != 0) for gate in gates)


def test_a_cached_context_decodes_through_the_layers_own_attention_as_the_full_forward():
    """A prefill keeps the context's keys and values its layers' own projections give, and each later
    step reads them without it (check 1, from the float64 forward)."""
    assert_a_cached_context_decodes_as_the_full_forward(SHARED_MODEL, "shared decode",
                                                        seeds=SHARED_SEEDS["shared decode"])


def test_a_scanned_stack_reads_through_each_layers_own_attention_as_the_plain_one():
    """With a read on every layer the layers scan as one run, on par with the plain loop (check 1)."""
    every = SHARED_MODEL.clone(cross_attention=SharedCrossAttention((0, 1, 2), 6))
    assert_as_exact(logits(every.clone(scan_layers=True)), logits(every), model_draw(every), "shared scan",
                    seeds=SHARED_SEEDS["shared scan"], truth_fn=logits(float64(every)))


def test_a_layer_with_no_attention_of_its_own_is_refused():
    """A layer whose mixer reads no context through its own attention (a state space mixer) or projects
    no keys of its own (a KV-sharing layer) is refused as the decoder that names it is built."""
    ids = jnp.ones((1, 4), jnp.int32)
    hybrid = CausalTransformer(
        vocab_size=37, emb_features=8, num_layers=2, num_heads=2, num_kv_heads=1, head_dim=4, mlp_features=16,
        layer_types=("ssm", "attention"),
        kinds={"ssm": LayerKind(mixer=Mamba2Mixer(num_heads=2, head_dim=4, state_size=3, n_groups=1,
                                                   norm_groups=1, chunk_size=1)),
               "attention": LayerKind()})
    hybrid.clone(cross_attention=SharedCrossAttention((1,), 4)).init(jax.random.key(0), ids)
    with pytest.raises(ValueError):
        hybrid.clone(cross_attention=SharedCrossAttention((0,), 4)).init(jax.random.key(0), ids)
    for fields in ({"layer_ids": ()}, {"layer_ids": (1, 1)}, {"context_length": 0}):
        with pytest.raises(ValueError):
            SharedCrossAttention(**{"layer_ids": (0,), "context_length": 4, **fields})


def test_cross_flops_count_each_layers_own_attention_over_the_context():
    """`cross_flops` counts each read's query and output maps over the tokens, its key and value maps
    over the context and the products of the two, and no feed-forward."""
    lengths, context = np.array([5, 3]), np.array([6, 0])
    tokens, context_tokens, pairs = 8, 6, 5 * 6
    width, queries, keys = 32, 4 * 8, 2 * 8
    projections = 2 * 2 * width * queries * tokens + 2 * 2 * width * keys * context_tokens
    per_read = projections + 2 * 4 * (8 + 8) * pairs
    assert SHARED_MODEL.cross_flops(lengths, context) == len(SHARED.layer_ids) * per_read
    # The projections are the layer's own matrices, which train as its trunk does; IDEFICS's block holds
    # none of the decoder's.
    assert SHARED_MODEL.held_cross_flops(lengths, context) == len(SHARED.layer_ids) * projections
    assert MODEL.held_cross_flops(lengths, context) == 0


def test_a_frozen_trunk_counts_no_weight_gradient_for_its_reads_through_its_own_matrices():
    """Attached frozen to a source, the shared form trains its gates alone: the training count is a fully
    trained decoder's less every trunk and head matrix's weight gradient and the weight gradient of the
    reads' own projections (`held_cross_flops`), each a frozen layer's own matrix (row 2413)."""
    from test_cross_attention import LLAMA

    from dew.interop.pretrained import Pretrained
    from dew.nn.inputs import ModelInputs
    from dew.objectives.lm import LMObjective

    source = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference")
    attached = source.with_cross_attention(SharedCrossAttention((1,), 3), key=0)
    width = attached.model.context_width
    rng = np.random.default_rng(0)
    batch = {"text": ModelInputs(jnp.asarray(rng.integers(0, 256, (2, 8)), jnp.int32), conditioning={
        CONTEXT: jnp.asarray(rng.standard_normal((2, 3, width)), jnp.float32),
        CONTEXT_LENGTHS: jnp.asarray([3, 1], jnp.int32)})}
    frozen, trained = LMObjective(attached, 7), LMObjective(attached.model, 7)
    rows, targets = frozen.real_lengths(batch), np.full(2, 7, np.int64)
    held = attached.model.held_cross_flops(rows, np.asarray([3, 1]))
    assert held > 0
    weights = attached.model.parameter_flops(rows) + attached.model.logits_flops(targets) + held
    assert trained.training_flops(None, batch) - frozen.training_flops(None, batch) == weights


def test_a_row_without_context_sends_no_gradient_into_the_layers_projections():
    """A row with no context reads its first slot so its softmax stays finite, and its block joins
    nothing: the layer's key and value projections take the same gradient with that row's context
    anything at all."""
    variables, ids, context, lengths = model_draw(SHARED_MODEL)(3)
    lengths = np.asarray(lengths).copy()
    lengths[0] = 0

    def projections(context):
        def loss(params):
            return jnp.sum(SHARED_MODEL.apply({**variables, "params": params}, ids,
                                              conditioning={CONTEXT: context, CONTEXT_LENGTHS: lengths}))
        grads = flat_tree(jax.grad(loss)(variables["params"]))
        return {path: np.asarray(leaf) for path, leaf in grads.items()
                if "k_proj" in path or "v_proj" in path}

    moved = np.asarray(context).copy()
    moved[0] = 1e3
    plain, other = projections(context), projections(moved)
    assert plain
    for path, leaf in plain.items():
        np.testing.assert_array_equal(leaf, other[path], err_msg=path)


def test_the_gates_train_alone_over_a_frozen_source():
    """Attached to a source, the gates are the only weights that train, the source frozen beside them,
    and the decoder's record rebuilds the shared form."""
    from test_cross_attention import LLAMA

    from dew.interop.pretrained import Pretrained

    source = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference")
    attached = source.with_cross_attention(SharedCrossAttention((1,), 3), key=0)
    assert FROZEN in attached.variables
    trained = flat_tree(attached.variables["params"])
    assert trained and all(path.endswith("alpha_cross_attn") for path in trained)
    from dew.registry import from_record, to_record

    model = attached.model
    assert isinstance(model, CausalTransformer)
    rebuilt = from_record(CausalTransformer, to_record(model, CausalTransformer))
    assert rebuilt.cross_attention == SharedCrossAttention((1,), 3)


def test_a_context_reads_logits_enter_the_layers_qk_clip_maxima():
    """The QK-Clip bounds the logits of a layer's own projections wherever they run: a layer that reads
    a context through them sows one maximum a head, its own sequence's or the context's, whichever is
    larger, over the keys it sees. At zero gates the reads leave every layer's own sequence as the base
    decoder's, so each layer's maxima are the base's or more, the base's alike where it reads none and
    on a row with no context, whatever its padding holds, and a context scaled up raises a reading
    layer's past them."""
    variables, ids, context, lengths = model_draw(SHARED_UNNORMED, gated=False)(2)
    base = dataclasses.replace(SHARED_UNNORMED, cross_attention=None)

    def maxima(model, variables, conditioning=None):
        _, held = model.apply(variables, ids, conditioning=conditioning, mutable=["qk"])
        named = ((jax.tree_util.keystr(path), leaf)
                 for path, leaf in jax.tree_util.tree_leaves_with_path(held))
        return {path: np.asarray(leaf) for path, leaf in named if "['max_logits']" in path}

    own = maxima(base, _without_blocks(variables))
    plain, scaled = (maxima(SHARED_UNNORMED, variables, {CONTEXT: scale * context, CONTEXT_LENGTHS: lengths})
                     for scale in (1.0, 100.0))
    assert own and set(plain) == set(scaled) == set(own)
    for path, leaf in scaled.items():
        reads = any(f"['layers_{index}']" in path for index in SHARED.layer_ids)
        assert np.all(plain[path] >= own[path]) and np.all(leaf >= own[path]), path
        assert lengths[0] == 0
        np.testing.assert_array_equal(leaf[0], own[path][0], err_msg=path)
        if reads:
            assert np.any(leaf > own[path]), path
        else:
            np.testing.assert_array_equal(leaf, own[path], err_msg=path)
            np.testing.assert_array_equal(plain[path], own[path], err_msg=path)


@dataclasses.dataclass(frozen=True)
class Alternating(CrossAttention):
    """IDEFICS's block whose feed-forward's activation alternates by layer: a configuration of its own
    (`CrossBlocks`), whose layers' blocks differ though their shapes do not."""

    def block(self, decoder, index):
        return CrossAttention.block(dataclasses.replace(self, activation=("swiglu", "geglu")[index % 2]),
                                    decoder, index)

    def layer_key(self, decoder, index):
        return CrossAttention.layer_key(self, decoder, index), index % 2


@dataclasses.dataclass(frozen=True)
class Unkeyed(SharedCrossAttention):
    """The shared form whose layers' key is None, which a configuration may give."""

    def layer_key(self, decoder, index):
        return None


def test_a_layer_keyed_none_still_reads_its_context():
    """A layer's key is whatever its configuration gives (`CrossBlocks.layer_key`), None among them:
    every named layer builds its block, and only those do."""
    model = MODEL.clone(cross_attention=Unkeyed((0, 2), 6))
    gates = {path for path in flat_tree(model_draw(model, gated=False)(0)[0]["params"]) if "alpha_" in path}
    assert {path.split(".", 1)[0] for path in gates} == {"layers_0", "layers_2"}


def test_a_configuration_of_its_own_scans_each_layers_own_block():
    """Any conforming configuration builds the blocks (`CrossBlocks`): one whose adjacent layers' blocks
    differ in what they compute, not their shapes, scans each layer's own block as the plain loop does
    (check 1, from float64), its layers' keys apart (`CrossBlocks.layer_key`)."""
    every = MODEL.clone(cross_attention=Alternating(layer_ids=(0, 1, 2), context_length=6,
                                                    context_features=24))
    assert_as_exact(logits(every.clone(scan_layers=True)), logits(every), model_draw(every),
                    "alternating scan", seeds=SHARED_SEEDS["alternating scan"],
                    truth_fn=logits(float64(every)))


def test_an_earlier_record_of_idefics_block_reads_as_its_class_record():
    """A run recorded before a decoder's cross-attention was any configuration wrote IDEFICS's block by its
    fields alone; the saved run's reader (`earlier_shapes`) reads it as the class record Dew writes now, and
    the decoder rebuilds from it."""
    from dew.inference.tasks import earlier_shapes
    from dew.registry import from_record, to_record

    record = to_record(MODEL, CausalTransformer)
    fields = dict(record["fields"]) if "fields" in record else dict(record)
    cross = fields["cross_attention"]
    assert cross["class"] == "dew.nn.cross_attention:CrossAttention"
    earlier = {**fields, "cross_attention": dict(cross["fields"])}
    model = {"class": "dew.nn.backbones.causal_transformer:CausalTransformer", "fields": earlier}
    current = earlier_shapes({"model": model})["model"]
    assert current["fields"]["cross_attention"] == cross
    assert from_record(CausalTransformer, current).cross_attention == MODEL.cross_attention
