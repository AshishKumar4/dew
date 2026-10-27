"""Gated cross-attention (`dew.nn.cross_attention`) on par with IDEFICS's layer in transformers 5.16.1.

The block is checked against `IdeficsGatedCrossAttentionLayer` (modeling_idefics.py:678-803), run
live in float32 and, as the truth, in float64 with its float32 pins widened (check 1,
tests/reference_error.py): its output, and the gradients into the residual, the context and every
parameter, the gates drawn away from zero. What zero gates and an empty context leave of the
residual is exact. The decoder carries the blocks through its forward, its decode cache and its
scanned stack, each on par with the plain forward from float64, and in every form of residual it
carries. A loaded bundle trains the blocks beside a LoRA adapter's factors with the source frozen,
and its run's checkpoint restores them.
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
from model_support import TINY_DECODER, Digits, decoder, flat_tree
from reference_error import assert_as_exact, torch_with_gradients, with_gradients
from test_serving import assert_same_generation

from dew.batch import FROZEN, coordinates, thaw
from dew.checkpoints import Checkpoints
from dew.data.dataset import Loader
from dew.inference import RunProcessor, TextGeneration
from dew.inference.serving import Server
from dew.interop.pretrained import Pretrained
from dew.lora import LoRA
from dew.nn.attention_residuals import AttentionResiduals
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import decoder_norm
from dew.nn.cross_attention import (
    CROSS_LENGTHS,
    CrossAttention,
    GatedCrossAttention,
    crossing,
    encode_context,
    gated_join,
)
from dew.nn.gemma3n import AltUp
from dew.nn.hyper_connections import HyperConnections
from dew.nn.inputs import CONTEXT, CONTEXT_LENGTHS, CrossContext, ModelInputs
from dew.nn.kv_cache import KVCache
from dew.nn.mlp import GatedMLP
from dew.nn.ngpt import NGPT
from dew.objectives.lm import LMObjective
from dew.sampling import Sampling
from dew.training import Layout, MeshSpec, Trainer

ROOT = Path(__file__).resolve().parents[1]
LLAMA = ROOT / "tests" / "fixtures" / "hf" / "llama-tiny"

B, S, D, H, C, FEATURES, WIDTH, EPS = 4, 5, 32, 4, 7, 48, 40, 1e-6
BLOCK = GatedCrossAttention(
    norm=decoder_norm("rms", epsilon=EPS, bias=False, scale_offset=False, scale_after_cast=True, dtype=None),
    feedforward=functools.partial(GatedMLP, hidden_features=WIDTH, out_features=D),
    emb_features=D, context_features=FEATURES, context_length=C, heads=H, kv_heads=H, head_dim=D // H)

VOCAB, SEQ, PROMPT = 37, 10, 4
CROSS = CrossAttention(layer_ids=(0, 2), context_length=6, context_features=24)
MODEL = CausalTransformer(**{"vocab_size": VOCAB, **TINY_DECODER, "num_layers": 3, "num_kv_heads": 2,
                             "cross_attention": CROSS})

FORMS = {
    "altup": {"altup": AltUp(num_inputs=2)},
    "mhc": {"hyper_connections": HyperConnections(hc_mult=2)},
    "single-pass-mhc": {"hyper_connections": HyperConnections(hc_mult=2, head="carried", single_pass=True)},
    "depth": {"attention_residuals": AttentionResiduals(block_size=2)},
    "ngpt": {"ngpt": NGPT(std=0.02, alpha=0.05), "pre_norms": False, "qk_norm": False},
}
"""Each residual a decoder carries beside the plain stream, which the blocks read and join in its form."""

SEEDS = {"block float32": 512, "block bfloat16": 512, "base": 16, "decode": 16, "scan": 16, "fresh altup": 16,
         "fresh mhc": 32, "fresh single-pass-mhc": 16, "fresh depth": 16, "fresh ngpt": 8}
"""Check 1's declared seeds for each site of the file, sized by `seeds_for` on armada."""


def block_draw(seed, dtype=np.float32):
    """The residual, a context, each row's real context tokens (one or more), the block's parameters
    with its gates away from zero, and a cotangent on its output, drawn from `seed` in `dtype`."""
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
    x, context, params, cotangent = jax.tree.map(lambda value: np.asarray(value, dtype), drawn)
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

    from tools.torch_precision import float64

    def forward(x, context, params, *, lengths):
        with float64() if wide else contextlib.nullcontext():
            config = IdeficsConfig(hidden_size=D, num_attention_heads=H, intermediate_size=WIDTH,
                                   rms_norm_eps=EPS, hidden_act="silu", vision_config={"embed_dim": FEATURES},
                                   attn_implementation="eager")
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


@pytest.mark.parametrize("dtype", [np.float32, jnp.bfloat16], ids=["float32", "bfloat16"])
def test_the_block_computes_and_differentiates_as_idefics(dtype):
    """Through both gated branches, the output and the gradients into the residual, the context and
    every parameter on par with IDEFICS's in float32 and in bfloat16 (check 1), over contexts of
    every length; the gates' products are at least float32."""
    label = f"block {np.dtype(dtype).name}"
    assert_as_exact(dew_block, idefics_block(wide=False), functools.partial(block_draw, dtype=dtype), label,
                    seeds=SEEDS[label], truth_fn=idefics_block(wide=True))


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


def fresh_against_base(model: CausalTransformer, label: str):
    """Check 1: `model` drawn at zero gates, reading a context, on par with the decoder without its
    blocks from that decoder's float64: the same function in two programs."""
    base = dataclasses.replace(model, cross_attention=None)

    def base_logits(model):
        run = jax.jit(lambda variables, ids: model.apply(variables, ids))
        return lambda variables, ids, context, lengths: run(_without_blocks(variables), ids)

    assert_as_exact(logits(model), base_logits(base), model_draw(model, gated=False), label,
                    seeds=SEEDS[label], truth_fn=base_logits(float64(base)))


def test_a_fresh_block_computes_the_decoder_it_was_attached_to():
    """Drawn at zero gates, the decoder with its blocks computes the decoder without them (check 1)."""
    fresh_against_base(MODEL, "base")


@pytest.mark.parametrize("form", FORMS)
def test_every_residual_form_takes_the_blocks(form):
    """AltUp's active copy, mHC's streams (Single-Pass's too), the depth state's partial block and
    nGPT's unit stream each take the blocks: fresh, the decoder computes itself (check 1); gated, its
    logits move, and every gate takes a finite gradient that moves it."""
    model = MODEL.clone(**FORMS[form])
    fresh_against_base(model, f"fresh {form}")
    variables, ids, context, lengths = model_draw(model)(1)
    conditioning = {CONTEXT: context, CONTEXT_LENGTHS: lengths}
    base = dataclasses.replace(model, cross_attention=None)
    moved = np.asarray(model.apply(variables, ids, conditioning=conditioning))
    assert not np.array_equal(moved[lengths > 0], np.asarray(base.apply(_without_blocks(variables), ids))[
        lengths > 0])
    gradients = jax.grad(lambda params: jnp.sum(model.apply({**variables, "params": params}, ids,
                                                            conditioning=conditioning)))(variables["params"])
    gates = [np.asarray(leaf) for path, leaf in flat_tree(gradients).items() if "alpha_" in path]
    assert len(gates) == 2 * len(CROSS.layer_ids)
    assert all(np.isfinite(gate).all() and np.all(gate != 0) for gate in gates)


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
    bundle reads the batch's context and moves the gates off zero, every frozen leaf as it was, and
    the run's checkpoint restores the model with its blocks; the source's own format holds no
    blocks, so the bundle refuses to export to it."""
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
    checkpoints = Checkpoints(str(tmp_path / "run"))
    trainer = Trainer(LMObjective(before, 7), optax.sgd(0.1), key=jax.random.key(3), mesh=MeshSpec(),
                      layout=Layout(min_shard=2**30), checkpoints=checkpoints)
    initial = trainer.initial_state()
    state = trainer.fit(Loader(train=lambda partition: iter([{"text": inputs}] * 2), val=None, records=rows,
                               batch=rows), steps=2, log_every=2, checkpoint_every=2)
    checkpoints.wait()
    for before_leaf, after_leaf in zip(jax.tree.leaves(initial.variables[FROZEN]),
                                       jax.tree.leaves(state.variables[FROZEN]), strict=True):
        np.testing.assert_array_equal(np.asarray(before_leaf), np.asarray(after_leaf))
    gates = {path: np.asarray(leaf) for path, leaf in flat_tree(thaw(state.variables)["params"]).items()
             if "alpha_" in path}
    assert gates and all(np.all(gate != 0) for gate in gates.values())
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(state.variables["params"]))
    restored = Pretrained.from_run(tmp_path / "run")
    assert restored.model.cross_attention == cross
    kept = flat_tree(thaw(restored.variables))
    for path, leaf in flat_tree(thaw(state.variables)).items():
        np.testing.assert_array_equal(np.asarray(kept[path]), np.asarray(leaf), err_msg=path)


def test_the_blocks_train_beside_a_whole_source_where_it_is_not_frozen(source):
    """With `frozen=False` the blocks join an unsplit source's weights, which all train beside them, a
    full fine-tune: the source's leaves are its own, the blocks' gates zero, so the bundle computes
    what the source does, and two steps move the source's weights and the gates both."""
    cross = CrossAttention(layer_ids=(1,), context_length=3, context_features=8)
    attached = source.with_cross_attention(cross, key=0, frozen=False)
    assert FROZEN not in attached.variables
    held, own = flat_tree(attached.variables["params"]), flat_tree(thaw(source.variables)["params"])
    assert {path for path in held if not crossing(path)} == set(own)
    for path, leaf in own.items():
        np.testing.assert_array_equal(np.asarray(held[path]), np.asarray(leaf), err_msg=path)
    tokens = jnp.arange(8, dtype=jnp.int32)[None] % 256
    np.testing.assert_array_equal(np.asarray(attached.model.apply(attached.variables, tokens)),
                                  np.asarray(source.model.apply(source.variables, tokens)))

    rng = np.random.default_rng(0)
    rows = 2 * jax.device_count()
    inputs = ModelInputs(jnp.asarray(rng.integers(0, 256, (rows, 8)), jnp.int32), conditioning={
        CONTEXT: jnp.asarray(rng.standard_normal((rows, 3, 8)), jnp.float32),
        CONTEXT_LENGTHS: jnp.asarray(rng.integers(1, 4, rows), jnp.int32)})
    trainer = Trainer(LMObjective(attached, 7), optax.sgd(0.1), key=jax.random.key(3), mesh=MeshSpec(),
                      layout=Layout(min_shard=2**30))
    initial = flat_tree(thaw(trainer.initial_state().variables)["params"])
    state = trainer.fit(Loader(train=lambda partition: iter([{"text": inputs}] * 2), val=None, records=rows,
                               batch=rows), steps=2, log_every=2)
    trained = flat_tree(thaw(state.variables)["params"])
    assert FROZEN not in state.variables
    moved = {path for path, leaf in trained.items() if not np.array_equal(np.asarray(leaf), initial[path])}
    assert any(not crossing(path) for path in moved) and any("alpha_" in path for path in moved)


SERVED = CrossAttention(layer_ids=(0,), context_length=4, context_features=6)
PROMPTS = [[1, 2], [5], [1, 2, 3, 4, 5, 6, 7], [9, 8], [3, 2, 1]]
BUDGETS = [5, 9, 3, 12, 7]
CONTEXT_TOKENS = [2, 4, 3, 0, 1]
"""Each request's context tokens; the fourth request carries none."""


@functools.cache
def served_task() -> TextGeneration:
    """model_support's one-layer decoder with a block before its layer, its gates drawn away from
    zero, as a greedy task."""
    model = decoder().clone(max_seq_len=128, cross_attention=SERVED)
    params = model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    params = jax.tree_util.tree_map_with_path(
        lambda path, leaf: jnp.full_like(leaf, 0.8) if path[-1].key.startswith("alpha_") else leaf, params)
    return TextGeneration(model, params, RunProcessor(Digits()), sampling=Sampling(temperature=0,
                                                                                    eos_token_ids=12))


def request(index: int, length: int | None = None) -> ModelInputs:
    """Request `index`'s prompt with a context of `length` tokens (its own count where None), drawn
    from the request's index."""
    length = CONTEXT_TOKENS[index] if length is None else length
    context = np.random.default_rng([11, index, length]).standard_normal((1, SERVED.context_length, 6))
    return ModelInputs(jnp.asarray([PROMPTS[index]], jnp.int32), conditioning={
        CONTEXT: jnp.asarray(context, jnp.float32), CONTEXT_LENGTHS: jnp.asarray([length], jnp.int32)})


@pytest.mark.parametrize("options", [
    {},
    {"kv_cache": KVCache(page_size=4, pages=40)},
    {"kv_cache": KVCache(page_size=4, pages=40), "chunk": 3},
    {"kv_cache": KVCache(page_size=4, pages=40), "chunk": 3, "prefix_cache": True},
], ids=["dense", "paged", "chunked", "prefix"])
def test_a_served_request_reads_its_own_context(options):
    """Requests of their own contexts, one with none, served together over a dense cache, a pool of
    pages, pieces of three tokens and a prefix cache, each draw what the request draws alone; so does
    a repeat of one, whose prefix pages it shares under the same context and not under another. The
    admitting step prefills apart from the decoding rows, whose one mixed row seats no context."""
    bound = served_task()
    alone = [bound(request(index), budget, key=index) for index, budget in enumerate(BUDGETS)]
    other = bound(request(2, 1), BUDGETS[2], key=2)
    assert not np.array_equal(np.asarray(other.host().raw_log_probs),
                              np.asarray(alone[2].host().raw_log_probs)), "the context is read"
    server = Server.from_task(bound, slots=4, capacity=128, admission=2, **options)
    assert server.mixed_refusal is not None
    tickets = [server.submit(request(index), budget, key=index) for index, budget in enumerate(BUDGETS)]
    server.run()
    again = server.submit(request(2), BUDGETS[2], key=2)
    server.run()
    hits = server.prefix_hits
    elsewhere = server.submit(request(2, 1), BUDGETS[2], key=2)
    server.run()
    for ticket, lone in zip([*tickets, again, elsewhere], [*alone, alone[2], other], strict=True):
        assert_same_generation(ticket.result(), lone)
    # "1234567" keeps its last token to prefill: one full page of four is shared, under its context alone.
    assert (hits, server.prefix_hits) == ((4, 4) if options.get("prefix_cache") else (0, 0))


def test_a_server_refuses_a_context_its_layers_cannot_read():
    """A context wider than the layers read, longer than they keep, or counting tokens it does not
    hold, an int64 count the cache's int32 would wrap among them, is refused at submit; a decoder
    without cross-attention takes none."""
    server = Server.from_task(served_task(), slots=2, capacity=128)
    wide = request(0)
    with pytest.raises(ValueError):
        server.submit(dataclasses.replace(wide, conditioning={CONTEXT: jnp.zeros((1, 4, 7))}), 2)
    with pytest.raises(ValueError):
        server.submit(dataclasses.replace(wide, conditioning={CONTEXT: jnp.zeros((1, 5, 6))}), 2)
    for length in (-1, 3):
        with pytest.raises(ValueError):
            server.submit(dataclasses.replace(wide, conditioning={
                CONTEXT: jnp.zeros((1, 2, 6)), CONTEXT_LENGTHS: jnp.asarray([length], jnp.int32)}), 2)
    model = decoder().clone(max_seq_len=128)
    plain = TextGeneration(model, model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32)),
                           RunProcessor(Digits()), sampling=Sampling(temperature=0))
    with pytest.raises(ValueError):
        Server.from_task(plain, slots=2, capacity=128).submit(request(0), 2)
    with jax.enable_x64(new_val=True):
        wrapped = np.asarray([2 ** 32 + 1], np.int64)
        with pytest.raises(ValueError):
            server.submit(dataclasses.replace(wide, conditioning={
                CONTEXT: jnp.zeros((1, 2, 6)), CONTEXT_LENGTHS: wrapped}), 2)


def test_a_zero_gate_leaves_an_ngpt_stream_whatever_its_length():
    """Under nGPT a zero gate moves the stream by nothing, tiny, large and zero rows alike, bit for
    bit, where a step normed back to the stream's length would shrink a tiny one; the gradients stay
    finite, a zero row's included. A stream whose squares overflow float32 (1e20), joined to a finite
    branch, stays itself too (`gated_join`): its length is taken over its largest element."""
    block = BLOCK.clone(norm=None, ngpt=True)
    x, context, params, lengths, cotangent = block_draw(2)
    x = x * np.asarray([1e-4, 1.0, 30.0, 0.0], np.float32)[:, None, None]
    shut = {**params, "alpha_cross_attn": np.zeros(1, np.float32), "alpha_dense": np.zeros(1, np.float32)}

    def block_out(x, params):
        return block.apply({"params": params}, x, CrossContext(context, lengths))

    out, into_x, gradients = jax.jit(with_gradients(block_out))(x, shut, cotangent)
    np.testing.assert_array_equal(np.asarray(out), x)
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves((into_x, gradients)))
    large = x * np.float32(1e20 / 30)
    zero = jnp.zeros(1, jnp.float32)
    np.testing.assert_array_equal(np.asarray(gated_join(large, x, zero, ngpt=True)), large)
    moved = np.asarray(gated_join(large, x, jnp.ones(1, jnp.float32), ngpt=True))
    assert np.isfinite(moved).all() and not np.array_equal(moved, large)


def test_freeze_keeps_what_a_split_tree_trains():
    """`freeze` over a tree already split keeps its trained leaves training beside those it is asked
    to, and its frozen ones frozen; over an unsplit tree it splits by the filter alone, and freezing
    a split tree by the filter it was split by changes nothing."""
    from dew.batch import freeze

    tree = {"params": {"a": np.ones(2), "b": np.ones(3), "c": np.ones(4)}, "constants": {"k": np.ones(1)}}
    first = freeze(tree, lambda path: path[-1] == "a")
    assert set(first["params"]) == {"a"} and set(first[FROZEN]) == {"b", "c"}
    second = freeze(first, lambda path: path[-1] == "b")
    assert set(second["params"]) == {"a", "b"} and set(second[FROZEN]) == {"c"}
    assert second["constants"] is tree["constants"]
    again = freeze(first, lambda path: path[-1] == "a")
    assert jax.tree.structure(again) == jax.tree.structure(first)
    with pytest.raises(ValueError):
        freeze(first, lambda path: True)


def test_an_objective_with_prediction_depths_scores_and_counts_a_conditioned_batch():
    """An LM objective over a decoder with blocks and a prediction depth scores a batch that carries a
    context, the depth reading the conditioned trunk's states, through the model's own scoring
    methods too, and counts the blocks' work over the context beside the trunk's (`cross_flops`);
    conditioning of another kind has no count."""
    from dew.nn.flops import MissingFlopsCount
    from dew.objectives.base import Step

    model = MODEL.clone(num_nextn_predict_layers=1)
    variables, ids, context, lengths = model_draw(model)(4)
    objective = LMObjective(model, SEQ - 1, mtp_weight=0.3)
    inputs = ModelInputs(jnp.asarray(ids), conditioning={CONTEXT: jnp.asarray(context),
                                                         CONTEXT_LENGTHS: jnp.asarray(lengths)})
    loss, _ = objective.scalar_loss(variables, {"text": inputs},
                                    Step(step=jnp.zeros((), jnp.int32), key=jax.random.key(0), ema=None))
    assert np.isfinite(float(loss))
    hidden = model.apply(variables, jnp.asarray(ids), method="hidden_states",
                         conditioning=inputs.conditioning)
    depths = model.apply(variables, hidden, jnp.asarray(ids), method="mtp_logits",
                         conditioning=inputs.conditioning)
    assert len(depths) == 1 and np.isfinite(np.asarray(depths[0])).all()
    plain = objective.model_flops(None, {"text": ModelInputs(jnp.asarray(ids))})
    rows = np.full(B, SEQ - 1, np.int64)
    crossed = model.cross_flops(rows, lengths.astype(np.int64))
    assert objective.model_flops(None, {"text": inputs}) == plain + crossed
    with pytest.raises(MissingFlopsCount):
        media = dataclasses.replace(inputs, conditioning={"pixel_values": context})
        objective.model_flops(None, {"text": media})


def test_the_blocks_attach_to_a_decoder_whose_own_feed_forward_they_do_not_run():
    """GPT OSS's feed-forward is its experts' clamped SwiGLU, which no dense block runs; its blocks
    run their own activation, attach and read a context."""
    source = Pretrained.load(ROOT / "tests" / "fixtures" / "hf" / "gpt-oss-tiny", dtype="float32",
                             attention_impl="reference")
    attached = source.with_cross_attention(CrossAttention((0,), context_length=2, context_features=4), key=0)
    conditioning = {CONTEXT: jnp.ones((1, 2, 4)), CONTEXT_LENGTHS: jnp.asarray([2], jnp.int32)}
    logits = attached.model.apply(thaw(attached.variables), jnp.asarray([[1, 2, 3]]),
                                  conditioning=conditioning)
    assert np.isfinite(np.asarray(logits)).all()


def test_a_wider_context_enters_in_the_decoders_context_dtype():
    """With x64 on, a float64 context enters a float32 decoder as float32 (`context_dtype`), as a
    server holds it: no product of the call, the cross-attention's key and value maps included, takes
    a float64 operand as an accelerator runs it (a CPU's own float64 sums of float32 products are
    `dew.nn.precision.accumulated`'s), and the cached keys and values are float32."""
    from reference_error import equations

    variables, ids, context, lengths = model_draw(MODEL)(5)
    with jax.enable_x64(new_val=True):
        conditioning = {CONTEXT: np.asarray(context, np.float64), CONTEXT_LENGTHS: lengths}
        traced = jax.make_jaxpr(lambda variables: MODEL.apply(variables, ids, conditioning=conditioning))(
            variables)
        wide = [equation for equation in equations(traced, platform="cuda") if equation.primitive.name ==
                "dot_general" and any(variable.aval.dtype == np.float64 for variable in equation.invars)]
        assert not wide
        _, made = MODEL.apply(variables, B, method=CausalTransformer.init_cache, mutable=["cache"])
        _, held = MODEL.apply({**variables, **made}, ids[:, :PROMPT], decode=True, mutable=["cache"],
                              conditioning=conditioning)
    kept = [leaf for path, leaf in flat_tree(held["cache"]).items()
            if path.endswith(("cross_key", "cross_value"))]
    assert kept and all(leaf.dtype == np.float32 for leaf in kept)


def test_a_frozen_trunk_counts_the_weight_gradients_of_its_blocks_alone():
    """A source with blocks attached trains the blocks alone: its training count is a fully trained
    decoder's less every trunk and head matrix's weight gradient (`parameter_flops`, `logits_flops`),
    the blocks' three passes kept."""
    source = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference")
    cross = CrossAttention(layer_ids=(1,), context_length=3, context_features=8)
    attached = source.with_cross_attention(cross, key=0)
    rng = np.random.default_rng(0)
    batch = {"text": ModelInputs(jnp.asarray(rng.integers(0, 256, (2, 8)), jnp.int32), conditioning={
        CONTEXT: jnp.asarray(rng.standard_normal((2, 3, 8)), jnp.float32),
        CONTEXT_LENGTHS: jnp.asarray([3, 1], jnp.int32)})}
    frozen = LMObjective(attached, 7)
    trained = LMObjective(attached.model, 7)
    rows, targets = frozen.real_lengths(batch), np.full(2, 7, np.int64)
    weights = attached.model.parameter_flops(rows) + attached.model.logits_flops(targets)
    assert trained.training_flops(None, batch) - frozen.training_flops(None, batch) == weights


def test_zero_gates_leave_a_pretrained_ngpt_decoder_bit_for_bit():
    """Blocks attached to a pretrained nGPT decoder (Dory, looped), their gates at zero, return its
    logits bit for bit over a context; drawn away from zero, the gates move them."""
    dory = Pretrained.load(ROOT / "tests" / "fixtures" / "hf" / "dory-tiny", dtype="float32",
                           attention_impl="reference")
    width = dory.model.emb_features
    attached = dory.with_cross_attention(CrossAttention((0,), context_length=3, intermediate_size=32), key=0)
    rng = np.random.default_rng(6)
    ids = jnp.asarray(rng.integers(0, dory.model.vocab_size, (2, 9)), jnp.int32)
    conditioning = {CONTEXT: jnp.asarray(rng.standard_normal((2, 3, width)), jnp.float32),
                    CONTEXT_LENGTHS: jnp.asarray([3, 2], jnp.int32)}
    variables = thaw(attached.variables)
    base = np.asarray(dory.model.apply(thaw(dory.variables), ids))
    fresh = np.asarray(attached.model.apply(variables, ids, conditioning=conditioning))
    np.testing.assert_array_equal(fresh, base)
    gated = jax.tree_util.tree_map_with_path(
        lambda path, leaf: jnp.full_like(leaf, 0.5) if path[-1].key.startswith("alpha_") else leaf, variables)
    moved = np.asarray(attached.model.apply(gated, ids, conditioning=conditioning))
    assert np.isfinite(moved).all() and not np.array_equal(moved, base)


@pytest.mark.parametrize("loop", [None, (1, 3)], ids=["whole", "partial"])
def test_cross_flops_count_each_pass_of_a_looped_layer(loop):
    """A looped decoder runs a block on every pass of its layer: `cross_flops` counts it that many
    times (`layer_passes`), a layer outside a partial loop once."""
    from dew.nn.backbones.decoder_stack import Loop

    lengths, context = np.array([5, 3]), np.array([6, 2])
    looped = MODEL.clone(loop=Loop(steps=3, layers=loop))
    alone = [MODEL.clone(cross_attention=dataclasses.replace(CROSS, layer_ids=(index,)))
             for index in CROSS.layer_ids]
    blocks = [model.cross_flops(lengths, context) for model in alone]
    passes = [looped.layer_passes(index) for index in CROSS.layer_ids]
    assert passes == ([3, 3] if loop is None else [1, 3])
    assert looped.cross_flops(lengths, context) == sum(p * b for p, b in zip(passes, blocks, strict=True))


TOKEN_CONTEXT = CrossAttention(layer_ids=(0, 2), context_length=6,
                               context_features=TINY_DECODER["emb_features"])
"""A decoder's blocks over contexts of its own tokens, at its own width."""


def test_a_decoder_reads_a_context_of_its_own_tokens_by_its_table_or_its_own_pass():
    """A decoder with blocks reads a context of token ids as `encode_context` says: `table`, each
    token's row of its own table at its compute dtype; `backbone`, its final states over each context
    alone, its real tokens one document, a row's padding never reaching them. The states keep the
    dtype they are computed in, a bf16 decoder's bf16 where its blocks take fp32, and its open blocks
    read them as the same states handed in at their own dtype, a context that moves its logits. A
    decoder without blocks, and another encoder, are refused."""
    model = CausalTransformer(**{"vocab_size": VOCAB, **TINY_DECODER, "num_layers": 3, "num_kv_heads": 2,
                                 "cross_attention": TOKEN_CONTEXT, "dtype": "bfloat16"})
    rng = np.random.default_rng(0)
    ids = jnp.asarray(rng.integers(0, VOCAB, (B, SEQ)), jnp.int32)
    tokens = jnp.asarray(rng.integers(0, VOCAB, (B, TOKEN_CONTEXT.context_length)), jnp.int32)
    lengths = jnp.asarray(rng.integers(1, TOKEN_CONTEXT.context_length + 1, B), jnp.int32)
    # Open gates, so a block reads its context.
    variables = jax.tree_util.tree_map_with_path(
        lambda path, leaf: (rng.uniform(-1, 1, leaf.shape).astype(np.float32)
                            if path[-1].key.startswith("alpha_") else leaf),
        model.init(jax.random.key(0), ids))
    table = jnp.asarray(variables["params"]["embed_tokens"]["embedding"])
    encoded = encode_context(model, variables, tokens, lengths, encoder="table")
    assert encoded[CONTEXT].dtype == jnp.bfloat16 and model.context_dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(encoded[CONTEXT], np.float32),
                                  np.asarray(table[tokens].astype(jnp.bfloat16), np.float32))
    np.testing.assert_array_equal(np.asarray(encoded[CONTEXT_LENGTHS]), np.asarray(lengths))
    real = jnp.arange(tokens.shape[1])[None, :] < lengths[:, None]
    passed = encode_context(model, variables, tokens, lengths, encoder="backbone")[CONTEXT]
    alone = model.apply(variables, tokens, method="hidden_states", coordinates=coordinates(
        segments=real.astype(jnp.int32), max_segments=1, right_padded=True))
    np.testing.assert_array_equal(np.asarray(passed, np.float32), np.asarray(alone, np.float32))
    noise = jnp.where(real, tokens, (tokens + 1) % VOCAB)
    repadded = encode_context(model, variables, noise, lengths, encoder="backbone")[CONTEXT]
    np.testing.assert_array_equal(np.asarray(passed, np.float32)[np.asarray(real)],
                                  np.asarray(repadded, np.float32)[np.asarray(real)])
    unread = model.apply(variables, ids, conditioning={CONTEXT: encoded[CONTEXT],
                                                       CONTEXT_LENGTHS: jnp.zeros_like(lengths)})
    for states in (encoded[CONTEXT], passed):
        native = model.apply(variables, ids, conditioning={CONTEXT: states, CONTEXT_LENGTHS: lengths})
        widened = model.apply(variables, ids, conditioning={
            CONTEXT: jnp.asarray(states, model.context_dtype), CONTEXT_LENGTHS: lengths})
        np.testing.assert_array_equal(np.asarray(native, np.float32), np.asarray(widened, np.float32))
        assert not np.array_equal(np.asarray(native, np.float32), np.asarray(unread, np.float32))
    with pytest.raises(TypeError):
        encode_context(model.with_cross_attention(None), variables, tokens, lengths, encoder="table")
    with pytest.raises(ValueError):
        encode_context(model, variables, tokens, lengths, encoder="other")
