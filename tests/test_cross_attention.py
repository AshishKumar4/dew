"""Gated cross-attention (`dew.nn.cross_attention`) on par with IDEFICS's layer in transformers 5.16.1.

The block is checked against `IdeficsGatedCrossAttentionLayer` (modeling_idefics.py:678-803), run
live in float32 and, as the truth, in float64 with its float32 pins widened (check 1,
tests/reference_error.py): its output, and the gradients into the residual, the context and every
parameter, the gates drawn away from zero. What zero gates and an empty context leave of the
residual is exact. The decoder carries the blocks through its forward, its decode cache and its
scanned stack, each on par with the plain forward from float64. A loaded bundle trains the blocks
beside a LoRA adapter's factors with the source frozen.
"""

import contextlib
import dataclasses
import functools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import torch
from model_support import TINY_DECODER, flat_tree
from reference_error import assert_as_exact, torch_with_gradients, with_gradients

from dew.data.dataset import Loader
from dew.interop.pretrained import Pretrained
from dew.lora import LoRA
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import decoder_norm
from dew.nn.cross_attention import CROSS_LENGTHS, CrossAttention, GatedCrossAttention, crossing
from dew.nn.inputs import CONTEXT, CONTEXT_LENGTHS, CrossContext, ModelInputs
from dew.nn.mlp import GatedMLP
from dew.nn.ngpt import NGPT
from dew.objectives.base import FROZEN, thaw
from dew.objectives.lm import LMObjective
from dew.training import Layout, MeshSpec, Trainer

ROOT = Path(__file__).resolve().parents[1]
LLAMA = ROOT / "tests" / "fixtures" / "hf" / "llama-tiny"

B, S, D, H, C, FEATURES, WIDTH, EPS = 4, 5, 32, 4, 7, 48, 40, 1e-6
BLOCK = GatedCrossAttention(
    norm=decoder_norm("rms", epsilon=EPS, bias=False, scale_offset=False, scale_after_cast=False, dtype=None),
    feedforward=functools.partial(GatedMLP, hidden_features=WIDTH, out_features=D),
    emb_features=D, context_features=FEATURES, context_length=C, heads=H, kv_heads=H, head_dim=D // H)

VOCAB, SEQ, PROMPT = 37, 10, 4
CROSS = CrossAttention(layer_ids=(0, 2), context_length=6, context_features=24)
MODEL = CausalTransformer(**{"vocab_size": VOCAB, **TINY_DECODER, "num_layers": 3, "num_kv_heads": 2,
                             "cross_attention": CROSS})

SEEDS = {"block": 64, "base": 16, "decode": 16, "scan": 16}
"""Check 1's declared seeds for each site of the file, sized by `seeds_for` on armada."""


def block_draw(seed):
    """The residual, a context, each row's real context tokens (one or more), the block's parameters
    with its gates away from zero, and a cotangent on its output, drawn from `seed` in float32."""
    rng = np.random.default_rng(seed)
    shapes = jax.eval_shape(BLOCK.init, jax.random.key(0), jnp.zeros((B, S, D)),
                            CrossContext(jnp.zeros((B, C, FEATURES)), jnp.ones((B,), jnp.int32)))["params"]

    def leaf(path, shape):
        name = path[-1].key
        if name == "kernel":
            return rng.standard_normal(shape.shape) / np.sqrt(shape.shape[0])
        if name == "scale":
            return 1 + 0.2 * rng.standard_normal(shape.shape)
        return rng.uniform(-1, 1, shape.shape)  # a gate, tanh's argument

    params = jax.tree_util.tree_map_with_path(leaf, shapes)
    drawn = (rng.standard_normal((B, S, D)), rng.standard_normal((B, C, FEATURES)), params,
             rng.standard_normal((B, S, D)))
    x, context, params, cotangent = jax.tree.map(lambda value: np.asarray(value, np.float32), drawn)
    return x, context, params, rng.integers(1, C + 1, B).astype(np.int32), cotangent


@jax.jit
def dew_block(x, context, params, lengths, cotangent):
    """The block's output and its gradients into the residual, the context and the parameters."""
    def block(x, context, params):
        return BLOCK.apply({"params": params}, x, CrossContext(context, lengths))
    return with_gradients(block)(x, context, params, cotangent)


def idefics_block(wide: bool):
    """`IdeficsGatedCrossAttentionLayer` over Dew's parameters, transposed to torch's layout inside the
    arm so their gradients come back in Dew's, in float32 or, where `wide`, in float64: a row's context
    past its length is masked by the dtype's minimum, as IDEFICS's image attention mask is."""
    from transformers.models.idefics.configuration_idefics import IdeficsConfig
    from transformers.models.idefics.modeling_idefics import IdeficsGatedCrossAttentionLayer

    from tools.diffusers_wan_reference import float64

    def forward(x, context, params, *, lengths):
        with float64() if wide else contextlib.nullcontext():
            config = IdeficsConfig(hidden_size=D, num_attention_heads=H, intermediate_size=WIDTH,
                                   rms_norm_eps=EPS, hidden_act="silu", vision_config={"embed_dim": FEATURES})
            config._attn_implementation = "eager"
            layer = IdeficsGatedCrossAttentionLayer(config).to(x.dtype)
            mlp = params["mlp"]
            weights = {
                "input_layernorm.weight": params["input_layernorm"]["scale"],
                "post_attention_layernorm.weight": params["post_attention_layernorm"]["scale"],
                **{f"cross_attn.{name}.weight": params[name]["kernel"].T
                   for name in ("q_proj", "k_proj", "v_proj", "o_proj")},
                **{f"mlp.{name}.weight": mlp[name]["kernel"].T
                   for name in ("gate_proj", "up_proj", "down_proj")},
                "alpha_cross_attn": params["alpha_cross_attn"], "alpha_dense": params["alpha_dense"]}
            visible = torch.arange(C)[None, :] < torch.as_tensor(lengths)[:, None]
            mask = torch.where(visible, 0.0, torch.finfo(x.dtype).min).to(x.dtype)[:, None, None, :]
            return torch.func.functional_call(layer, weights, (x,), {
                "image_hidden_states": context, "image_attention_mask": mask,
                "cross_attention_gate": torch.ones(x.shape[:2], dtype=x.dtype)})

    def run(x, context, params, lengths, cotangent):
        arm = torch_with_gradients(functools.partial(forward, lengths=lengths))
        return arm(x, context, params, cotangent)
    return run


def test_the_block_computes_and_differentiates_as_idefics():
    """Through both gated branches, the output and the gradients into the residual, the context and
    every parameter on par with IDEFICS's float32 (check 1), over contexts of every length."""
    assert_as_exact(dew_block, idefics_block(wide=False), block_draw, "gated cross-attention",
                    seeds=SEEDS["block"], truth_fn=idefics_block(wide=True))


def test_zero_gates_and_an_empty_context_leave_the_residual_as_it_is():
    """At zero gates the block adds tanh(0) times its branches, an exact zero, so its output is its
    residual bit for bit while the gates' own gradients move them. A row with no context tokens adds
    nothing at any gate, its gradients finite."""
    x, context, params, lengths, cotangent = block_draw(0)
    shut = {**params, "alpha_cross_attn": np.zeros(1, np.float32), "alpha_dense": np.zeros(1, np.float32)}
    out, _, _, gradients = dew_block(x, context, shut, lengths, cotangent)
    np.testing.assert_array_equal(np.asarray(out), x)
    assert np.all(np.asarray(gradients["alpha_cross_attn"]) != 0)
    assert np.all(np.asarray(gradients["alpha_dense"]) != 0)

    empty = lengths.copy()
    empty[1] = 0
    out, into_x, into_context, gradients = dew_block(x, context, params, empty, cotangent)
    np.testing.assert_array_equal(np.asarray(out)[1], x[1])
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves((into_x, into_context, gradients)))
    assert not np.array_equal(np.asarray(out)[0], x[0]), "a row with a context moves"


def model_draw(model: CausalTransformer, *, gated: bool = True):
    """Check 1's draw for `model`: a seed's variables, their gates drawn away from zero where
    `gated`, and its tokens, a context and each row's context tokens, a row of none among them."""
    init = jax.jit(model.init)

    def draw(seed):
        rng = np.random.default_rng(seed)
        ids = rng.integers(0, VOCAB, (B, SEQ)).astype(np.int32)
        context = rng.standard_normal((B, CROSS.context_length, CROSS.context_features)).astype(np.float32)
        lengths = rng.integers(0, CROSS.context_length + 1, B).astype(np.int32)
        lengths[0] = 0
        variables = init(jax.random.key(seed), ids)
        if gated:
            variables = jax.tree_util.tree_map_with_path(
                lambda path, leaf: (rng.uniform(-1, 1, leaf.shape).astype(np.float32)
                                    if path[-1].key.startswith("alpha_") else leaf), variables)
        return variables, ids, context, lengths
    return draw


def logits(model):
    """`model`'s logits over a draw's tokens reading its context, compiled."""
    @jax.jit
    def run(variables, ids, context, lengths):
        return model.apply(variables, ids, conditioning={CONTEXT: context, CONTEXT_LENGTHS: lengths})
    return run


def float64(model):
    """`model` in float64 on the reference kernel, which keeps float64 where a fused one narrows."""
    return model.clone(dtype=jnp.float64, attention_impl="reference")


def test_a_fresh_block_computes_the_decoder_it_was_attached_to():
    """Drawn at zero gates, the decoder with its blocks computes the decoder without them, on par
    with its float32 from the float64 decoder (check 1): the same function in two programs."""
    base = dataclasses.replace(MODEL, cross_attention=None)

    def base_logits(model):
        run = jax.jit(lambda variables, ids: model.apply(variables, ids))
        return lambda variables, ids, context, lengths: run(_without_blocks(variables), ids)

    assert_as_exact(logits(MODEL), base_logits(base), model_draw(MODEL, gated=False), "fresh blocks",
                    seeds=SEEDS["base"], truth_fn=base_logits(float64(base)))


def _without_blocks(variables):
    """`variables` with every cross-attention block's leaves left out."""
    def kept(tree, path=()):
        if not isinstance(tree, dict):
            return tree
        return {name: kept(child, (*path, name)) for name, child in tree.items()
                if not crossing((*path, name))}
    return kept(dict(variables))


@jax.jit
def _decoded(variables, ids, context, lengths):
    """The prompt prefilled with its context into a fresh cache, then each later token a step with
    none: the logits from the prompt's last position on."""
    _, made = MODEL.apply(variables, ids.shape[0], method=CausalTransformer.init_cache, mutable=["cache"])
    out, held = MODEL.apply({**variables, **made}, ids[:, :PROMPT], decode=True, mutable=["cache"],
                            conditioning={CONTEXT: context, CONTEXT_LENGTHS: lengths})
    steps = [out[:, -1]]
    for position in range(PROMPT, SEQ):
        out, held = MODEL.apply({**variables, **held}, ids[:, position:position + 1], decode=True,
                                mutable=["cache"])
        steps.append(out[:, -1])
    return jnp.stack(steps, axis=1)


def test_a_cached_context_decodes_as_the_full_forward():
    """The context a prefill carries stays in the cache, and each later step attends over it without
    it: the steps' logits on par with the full forward's from the float64 forward (check 1)."""
    def full(model):
        run = logits(model)
        return lambda variables, ids, context, lengths: run(variables, ids, context, lengths)[:, PROMPT - 1:]

    draw = model_draw(MODEL)
    assert_as_exact(_decoded, full(MODEL), draw, "cached context", seeds=SEEDS["decode"],
                    truth_fn=full(float64(MODEL)))
    variables, ids, context, lengths = draw(0)
    _, made = MODEL.apply(variables, B, method=CausalTransformer.init_cache, mutable=["cache"])
    _, held = MODEL.apply({**variables, **made}, ids[:, :PROMPT], decode=True, mutable=["cache"],
                          conditioning={CONTEXT: context, CONTEXT_LENGTHS: lengths})
    kept = [leaf for path, leaf in flat_tree(held["cache"]).items() if path.endswith(CROSS_LENGTHS)]
    assert len(kept) == len(CROSS.layer_ids)
    for leaf in kept:
        np.testing.assert_array_equal(np.asarray(leaf), lengths)


def test_a_scanned_stack_runs_its_blocks_as_the_plain_one():
    """With a block on every layer the layers scan as one run, its blocks' leaves stacked, on par with
    the plain loop from its float64 (check 1)."""
    every = dataclasses.replace(MODEL, cross_attention=dataclasses.replace(CROSS, layer_ids=(0, 1, 2)))
    assert_as_exact(logits(every.clone(scan_layers=True)), logits(every), model_draw(every), "scanned blocks",
                    seeds=SEEDS["scan"], truth_fn=logits(float64(every)))


def test_a_decoder_refuses_what_its_blocks_cannot_read():
    """A decoder without blocks takes no conditioning, a cached call keeps at most the context length,
    and the spec and the decoder refuse what they cannot build."""
    variables, ids, context, _ = model_draw(MODEL, gated=False)(0)
    base = dataclasses.replace(MODEL, cross_attention=None)
    with pytest.raises(TypeError):
        base.apply(_without_blocks(variables), ids, conditioning={CONTEXT: context})
    with pytest.raises(ValueError):
        MODEL.apply(variables, ids, conditioning={"pixel_values": context})
    _, made = MODEL.apply(variables, B, method=CausalTransformer.init_cache, mutable=["cache"])
    longer = np.concatenate([context, context], axis=1)
    with pytest.raises(ValueError):
        MODEL.apply({**variables, **made}, ids, decode=True, mutable=["cache"],
                    conditioning={CONTEXT: longer})
    for fields in ({"layer_ids": ()}, {"layer_ids": (1, 1)}, {"context_length": 0}):
        with pytest.raises(ValueError):
            CrossAttention(**{"layer_ids": (0,), "context_length": 4, **fields})
    with pytest.raises(ValueError):
        MODEL.clone(cross_attention=CrossAttention((3,), 4)).init(jax.random.key(0), ids)
    normalized = base.clone(ngpt=NGPT(std=0.02, alpha=0.05))
    normalized.init(jax.random.key(0), ids)
    with pytest.raises(ValueError):
        normalized.clone(cross_attention=CROSS).init(jax.random.key(0), ids)


def test_cross_flops_count_the_blocks_over_the_context():
    """`cross_flops` counts each block's query and output maps and feed-forward over the tokens, its key
    and value maps over the context, and the products of the two, which `model_flops` leaves out."""
    lengths, context = np.array([5, 3]), np.array([6, 0])
    tokens, context_tokens, pairs = 8, 6, 5 * 6
    width, queries, keys, mlp = 32, 4 * 8, 2 * 8, 64
    per_block = (2 * 2 * width * queries * tokens  # the query and output maps
                 + 2 * 2 * CROSS.context_features * keys * context_tokens  # the key and value maps
                 + 2 * 4 * (8 + 8) * pairs  # each head's QK and AV
                 + 3 * 2 * width * mlp * tokens)  # the gated feed-forward
    assert MODEL.cross_flops(lengths, context) == len(CROSS.layer_ids) * per_block
    assert dataclasses.replace(MODEL, cross_attention=None).cross_flops(lengths, context) == 0


@pytest.fixture(scope="module")
def source():
    return Pretrained.load(LLAMA, dtype="float32", attention_impl="reference")


def test_the_blocks_train_beside_an_adapter_with_the_source_frozen(source, tmp_path):
    """Attached to a loaded source, the blocks train and the source stays frozen; attached beside a
    LoRA adapter in either order, the blocks and the factors train together. An objective over the
    bundle reads the batch's context and moves the gates off zero, every frozen leaf as it was; the
    source's format holds no blocks, so the bundle refuses to export."""
    cross = CrossAttention(layer_ids=(1,), context_length=3, context_features=8)
    attached = source.with_cross_attention(cross, key=0)
    assert all(crossing(path) for path in flat_tree(attached.variables["params"]))
    assert FROZEN in attached.variables
    lora = LoRA(rank=2, modules=("q_proj",))
    before, after = attached.adapt(lora, key=1), source.adapt(lora, key=1).with_cross_attention(cross, key=0)
    assert set(flat_tree(before.variables["params"])) == set(flat_tree(after.variables["params"]))
    assert any("lora_A" in path for path in flat_tree(before.variables["params"]))
    with pytest.raises(ValueError):
        attached.export()
    with pytest.raises(ValueError):
        attached.with_cross_attention(cross, key=0)

    rng = np.random.default_rng(0)
    rows = 2 * jax.device_count()
    inputs = ModelInputs(jnp.asarray(rng.integers(0, 256, (rows, 8)), jnp.int32), conditioning={
        CONTEXT: jnp.asarray(rng.standard_normal((rows, 3, 8)), jnp.float32),
        CONTEXT_LENGTHS: jnp.asarray(rng.integers(1, 4, rows), jnp.int32)})
    trainer = Trainer(LMObjective(before, 7), optax.sgd(0.1), key=jax.random.key(3), mesh=MeshSpec(),
                      layout=Layout(min_shard=2**30))
    initial = trainer.initial_state()
    state = trainer.fit(Loader(train=lambda partition: iter([{"text": inputs}] * 2), val=None, records=rows,
                               batch=rows), steps=2, log_every=2)
    for before_leaf, after_leaf in zip(jax.tree.leaves(initial.variables[FROZEN]),
                                       jax.tree.leaves(state.variables[FROZEN]), strict=True):
        np.testing.assert_array_equal(np.asarray(before_leaf), np.asarray(after_leaf))
    gates = {path: np.asarray(leaf) for path, leaf in flat_tree(thaw(state.variables)["params"]).items()
             if "alpha_" in path}
    assert gates and all(np.all(gate != 0) for gate in gates.values())
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(state.variables["params"]))
