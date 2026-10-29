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
TIED = CrossAttention(layer_ids=(0, 2), context_length=6)
"""IDEFICS's block at the decoder's own width, which computes the shared form where its attention is the
layer's, its input norm the layer's and its feed-forward's gate zero (`tied`)."""

SHARED_SEEDS = {"shared fresh": 16, "shared idefics": 16, "shared decode": 16, "shared scan": 16}
"""Check 1's declared seeds for each site of the file, sized by `seeds_for` on armada."""


def test_a_fresh_shared_read_computes_the_decoder_it_was_attached_to():
    """Drawn at zero gates, the decoder reading a context through its layers' own attention computes the
    decoder without the reads (check 1), and its gates are the only weights the form adds."""
    fresh_against_base(SHARED_MODEL, "shared fresh", seeds=SHARED_SEEDS["shared fresh"])
    variables, *_ = model_draw(SHARED_MODEL, gated=False)(0)
    added = {path for path in flat_tree(variables["params"]) if crossing(tuple(path.split(".")))}
    assert {path.rsplit(".", 1)[-1] for path in added} == {"alpha_cross_attn"}
    assert len(added) == len(SHARED.layer_ids)


TIED_MODEL = MODEL.clone(cross_attention=TIED)

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




def test_a_shared_read_is_idefics_block_over_the_layers_own_attention():
    """Gated, each layer's read through its own attention is IDEFICS's gated block whose attention is
    the layer's own projections and whose input norm is the layer's, its feed-forward gated off (check 1,
    from the shared decoder in float64): the layer's query over its normed residual and its keys and
    values over the context, unrotated and uncausal, each row's real context tokens, then its output
    projection."""
    tied_logits = logits(TIED_MODEL)
    assert_as_exact(logits(SHARED_MODEL),
                    lambda variables, *inputs: tied_logits(tied(variables), *inputs),
                    model_draw(SHARED_MODEL), "shared idefics", seeds=SHARED_SEEDS["shared idefics"],
                    truth_fn=logits(float64(SHARED_MODEL)))
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
    """A layer whose mixer projects no keys of its own (a state space mixer, a KV-sharing layer) has no
    attention to read a context through, so the decoder that names it is refused as it is built."""
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
    per_read = (2 * 2 * width * queries * tokens + 2 * 2 * width * keys * context_tokens
                + 2 * 4 * (8 + 8) * pairs)
    assert SHARED_MODEL.cross_flops(lengths, context) == len(SHARED.layer_ids) * per_read


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
