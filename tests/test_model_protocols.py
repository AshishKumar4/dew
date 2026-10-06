"""Every token model's reads give what its own forward gives.

A loss, a probe or a server reads a token model through the capabilities in
`dew.nn.protocols`: `logits`, `hidden_states`, `logits_from_hidden` and
`output_table`. Each read stands for a forward the model already has, and
these tests hold it to that forward: the logits to the call's, the head over
the states to the logits, the table's product (the matrix a tiled loss
contracts, `dew.objectives.lm.chunked.head_logits`) or else the exact head to
the logits, the gradients to the call's, and no read writing a cache.

The models are every family Dew loads. `dew.registry.models` holds classes,
and a class is not a family: a `CausalTransformer` is Qwen 3 or Gemma 2 only
once a config fills its fields, and its defaults build a 512-wide model.
The decoder family registry (`dew.interop.hf_decoders.families`) maps each
family's `model_type` to that config, and every family keeps a tiny random
checkpoint under tests/fixtures/hf for its parity tests. The cases are those
checkpoints: each one whose config names a registered family, itself or as
the decoder of a wrapper (`text_config`), loaded by `Pretrained.load` into
the model the loader builds for it, a `CausalTransformer`, a
`MultimodalTransformer` or a `DiffusionGemma`. A source config names its
`architectures` or nests its decoder, so the two DiffusionGemma reference
files that keep only a text config beside the released wrapper's tensors
(read by tests/test_block_diffusion.py through the wrapper) are not
checkpoints. A checkpoint whose `model_type` is the registry name of a Dew
model of its own (DiffusionGemma's) is that model's bundle, whose call
denoises a canvas; its reads stand for its causal encoder's forward, which
the DiffusionGemma tests below hold them to. Every other checkpoint is a
decoder or a wrapper around one, and its call is its logits.

Every comparison is bitwise, at both checkpoint dtypes, because the read and
the forward run the same operations on the same operands: the reads run
eagerly, operation by operation, as the forward they are compared with does.
Two programs compiled whole need not agree to the bit even then, since XLA
fuses each program as a whole: one that also returns the statistics the
layers sow (`mutable=True`) moved the tiny Gemma 3's logits (up to 9.2) by
3.3e-6 against the same forward compiled alone.

Beside the families: DiffusionGemma's layer-scalar preparation, a LoRA
adapter on the head, media through the multimodal reads, the JEPA encoders'
representations (test_architectures.py's cases), and the torchax fallback.
"""

import functools
import json
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
from test_architectures import CASES as ARCHITECTURES, MASK, TEXT_FEATURES, TEXT_TOKENS, VOCAB

from dew.interop import Pretrained, hf_decoders
from dew.lora import LoRA
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.dit import TextContext
from dew.nn.inputs import ModelInputs
from dew.nn.protocols import OutputTable, ProjectionSites, Recomputing
from dew.objectives.lm.chunked import head_logits
from dew.registry import models

FIXTURES = Path(__file__).parent / "fixtures" / "hf"
DTYPES = ("float32", "bfloat16")
ROWS, LENGTH = 2, 9


def _config(name: str) -> dict:
    return json.loads((FIXTURES / name / "config.json").read_text())


def _checkpoint(directory: Path) -> bool:
    """Whether `directory` is a tiny source of a registered decoder family."""
    if not (directory / "config.json").is_file() or not (directory / "model.safetensors").is_file():
        return False
    config, families = _config(directory.name), hf_decoders.families()
    text = config.get("text_config") or {}
    return ("architectures" in config or bool(text)) and (
        config.get("model_type") in families or text.get("model_type") in families)


CHECKPOINTS = sorted(directory.name for directory in FIXTURES.iterdir() if _checkpoint(directory))
OWN = [name for name in CHECKPOINTS if _config(name)["model_type"] in models]
"""The checkpoints of a Dew model of their own, DiffusionGemma's."""
DECODERS = [name for name in CHECKPOINTS if name not in OWN]


@functools.cache
def loaded(name: str, dtype: str = "float32") -> Pretrained:
    return Pretrained.load(FIXTURES / name, dtype=dtype, attention_impl="reference")


def same(actual, expected) -> None:
    """Every leaf of `actual` the same bits as `expected`'s."""
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want), strict=True)


class Reads(NamedTuple):
    tokens: jax.Array
    logits: jax.Array
    hidden: jax.Array
    table: OutputTable | None


@functools.cache
def reads(name: str, dtype: str) -> Reads:
    source = loaded(name, dtype)
    model, variables = source.model, source.variables
    tokens = jnp.asarray(np.random.RandomState(0).randint(0, model.vocab_size, (ROWS, LENGTH)), jnp.int32)
    return Reads(tokens, model.apply(variables, tokens, method="logits"),
                 model.apply(variables, tokens, method="hidden_states"),
                 model.apply(variables, method="output_table"))


def _cotangent(logits: jax.Array) -> jax.Array:
    return jax.random.normal(jax.random.key(1), logits.shape, logits.dtype)


def gradient(model, variables, tokens, cotangent, **apply) -> dict:
    """The gradient of `<model's output, cotangent>` in every floating
    parameter, by path; integer parameters are held as they are."""
    flat = flatten_dict(variables["params"])
    floats = {path: leaf for path, leaf in flat.items() if jnp.issubdtype(leaf.dtype, jnp.inexact)}
    held = {path: leaf for path, leaf in flat.items() if path not in floats}

    def loss(trained):
        params = unflatten_dict({**held, **trained})
        return jnp.sum(model.apply({**variables, "params": params}, tokens, **apply) * cotangent)

    return jax.grad(loss)(floats)


dtypes = pytest.mark.parametrize("dtype", DTYPES)


@dtypes
@pytest.mark.parametrize("name", DECODERS)
def test_logits_are_the_forwards(name, dtype):
    source, read = loaded(name, dtype), reads(name, dtype)
    same(read.logits, source.model.apply(source.variables, read.tokens))


@dtypes
@pytest.mark.parametrize("name", CHECKPOINTS)
def test_the_head_over_the_hidden_states_is_the_logits(name, dtype):
    source, read = loaded(name, dtype), reads(name, dtype)
    same(source.model.apply(source.variables, read.hidden, method="logits_from_hidden"), read.logits)


@dtypes
@pytest.mark.parametrize("name", CHECKPOINTS)
def test_the_head_a_loss_contracts_is_the_logits(name, dtype):
    """The table's product, bias and softcap included, at the table's
    precision; or, where the model gives no table, the exact head."""
    source, read = loaded(name, dtype), reads(name, dtype)
    table = read.table
    if table is None:
        scored = source.model.apply(source.variables, read.hidden, method="logits_from_hidden")
    else:
        scored = head_logits(read.hidden, table.matrix, softcap=table.softcap, precision=table.precision,
                             vocab_major=table.vocab_major, bias=table.bias)
    same(scored, read.logits)


@dtypes
@pytest.mark.parametrize("name", CHECKPOINTS)
def test_the_reads_write_no_cache_and_leave_an_allocated_one_alone(name, dtype):
    """With every collection mutable, neither read writes a `cache`. With a
    causal model's decode cache allocated beside the parameters (a
    bidirectional model has none), both leave it as it was and compute what
    they computed without it. (A mutable `qk` collection has the attention
    layers sow their statistics, which moves some families' logits by a
    rounding, so the reads are compared under the same collections.)"""
    source, read = loaded(name, dtype), reads(name, dtype)
    model, variables = source.model, source.variables
    cache = (model.apply(variables, ROWS, method="init_cache", mutable=["cache"])[1]["cache"]
             if model.causal else None)
    for method in ("logits", "hidden_states"):
        without, written = model.apply(variables, read.tokens, method=method, mutable=True)
        assert "cache" not in written, method
        if cache is not None:
            beside, written = model.apply({**variables, "cache": cache}, read.tokens, method=method,
                                          mutable=True)
            same(written["cache"], cache)
            same(beside, without)


@dtypes
@pytest.mark.parametrize("name", DECODERS)
def test_gradients_through_the_logits_are_the_forwards(name, dtype):
    source, read = loaded(name, dtype), reads(name, dtype)
    model, variables = source.model, source.variables
    cotangent = _cotangent(read.logits)
    same(gradient(model, variables, read.tokens, cotangent, method="logits"),
         gradient(model, variables, read.tokens, cotangent))


# DiffusionGemma: the clean read is the causal encoder over the shared tree.


def _text(variables):
    """DiffusionGemma's text tree, in every collection that holds one."""
    return {collection: tree["text"] for collection, tree in variables.items() if "text" in tree}


@dtypes
@pytest.mark.parametrize("name", OWN)
def test_the_clean_read_is_the_causal_encoders_forward_over_the_shared_tree(name, dtype):
    """The encoder the canvas shares its tree with, run over the whole
    sequence: the same logits and states as the text model's own forward
    over the text tree, and the gradient its forward gives that tree, with
    none for the parameters only the canvas reads."""
    source, read = loaded(name, dtype), reads(name, dtype)
    model, variables = source.model, source.variables
    text = _text(variables)
    same(read.logits, model.text.apply(text, read.tokens))
    same(read.hidden, model.text.apply(text, read.tokens, method="hidden_states"))
    cotangent = _cotangent(read.logits)
    clean = gradient(model, variables, read.tokens, cotangent, method="logits")
    encoder = gradient(model.text, text, read.tokens, cotangent)
    same({path[1:]: value for path, value in clean.items() if path[0] == "text"}, encoder)
    others = [value for path, value in clean.items() if path[0] != "text"]
    assert others
    assert not any(np.any(np.asarray(value)) for value in others)


@pytest.mark.parametrize("name", OWN)
def test_encode_writes_the_cache_the_clean_read_leaves_alone(name):
    """`encode` is still the cached commit: over the same tokens it moves
    the cache the clean read left as it was."""
    source, read = loaded(name), reads(name, "float32")
    model, variables = source.model, source.variables
    cache = model.apply(variables, ROWS, method="init_cache", mutable=["cache"])[1]["cache"]
    _, written = model.apply({**variables, "cache": cache}, read.tokens, method="encode", mutable=["cache"])
    moved = [not np.array_equal(np.asarray(after), np.asarray(before))
             for after, before in zip(jax.tree.leaves(written["cache"]), jax.tree.leaves(cache), strict=True)]
    assert any(moved)


@pytest.mark.parametrize("name", OWN)
def test_trainable_layer_scalars_read_the_frozen_tree_moved_without_a_copy(name):
    """The published SFT's model reads each layer's scalar as a parameter,
    where a frozen source keeps it among the constants: the same arrays,
    each scalar moved under its layer's params and every other leaf where
    it was, give the same logits, and the scalars then take a gradient."""
    source, read = loaded(name), reads(name, "float32")
    model, variables = source.model, source.variables
    trainable = model.with_trainable_layer_scalars()
    moved = model.trainable_variables(variables)
    before = flatten_dict(variables)
    after = flatten_dict(moved)
    scalars = {path for path in before if path[0] == "constants" and path[-1] == "layer_scalar"}
    assert scalars
    expected = {(("params", *path[1:]) if path in scalars else path): leaf for path, leaf in before.items()}
    assert after.keys() == expected.keys()
    assert all(after[path] is leaf for path, leaf in expected.items())
    same(trainable.apply(moved, read.tokens, method="logits"), read.logits)
    learned = gradient(trainable, moved, read.tokens, _cotangent(read.logits), method="logits")
    assert all(np.any(np.asarray(learned[path[1:]])) for path in scalars)
    assert trainable.trainable_variables(moved) is moved


@pytest.mark.parametrize("name", OWN)
def test_layer_scalar_preparation_refuses_what_it_cannot_move(name):
    source = loaded(name)
    model, variables = source.model, source.variables
    with pytest.raises(ValueError, match="frozen source unexpectedly contains a trainable layer_scalar"):
        model.trainable_variables(model.trainable_variables(variables) | {
            "constants": variables["constants"]})
    scalarless = model.clone(text=model.text.clone(layer_scalar=None))
    with pytest.raises(ValueError, match="no layer scalars to train"):
        scalarless.with_trainable_layer_scalars()
    with pytest.raises(ValueError, match="no layer scalars to move"):
        scalarless.trainable_variables(variables)


# A LoRA adapter on the head.


def _adapted(modules):
    """A tiny untied decoder whose head adds a bias and caps, under an adapter
    on `modules` whose B factors are drawn, so the adapter counts."""
    model = CausalTransformer(vocab_size=97, emb_features=24, num_layers=1, num_heads=3, mlp_features=32,
                              max_seq_len=16, tie_embeddings=False, head_bias=True, final_logit_softcap=5.0,
                              attention_impl="reference")
    tokens = jnp.asarray(np.random.RandomState(0).randint(0, 97, (ROWS, LENGTH)), jnp.int32)
    variables = model.init(jax.random.key(0), tokens)
    variables["params"]["head_bias"] = jax.random.normal(jax.random.key(2), (97,)) / 7
    adapter = LoRA(rank=2, modules=modules).apply(model, variables, key=3)
    params = jax.tree_util.tree_map_with_path(
        lambda path, leaf: jax.random.normal(jax.random.key(4), leaf.shape, leaf.dtype) / 3
        if jax.tree_util.keystr(path).endswith("['lora_B']") else leaf, adapter.variables["params"])
    return model, variables, adapter.model, {**adapter.variables, "params": params}, tokens


def test_an_adapter_on_the_head_leaves_no_table_and_the_exact_head_scores_it():
    """The adapted head is `x W + scale (x A) B`, which no stored matrix is,
    so the model gives no table; its exact head over its states is its
    logits, which the adapter moved off the base model's."""
    model, variables, adapted, adapted_variables, tokens = _adapted(("lm_head",))
    assert adapted.apply(adapted_variables, method="output_table") is None
    logits = adapted.apply(adapted_variables, tokens, method="logits")
    hidden = adapted.apply(adapted_variables, tokens, method="hidden_states")
    same(adapted.apply(adapted_variables, hidden, method="logits_from_hidden"), logits)
    same(logits, adapted.apply(adapted_variables, tokens))
    assert not np.array_equal(np.asarray(logits), np.asarray(model.apply(variables, tokens)))


def test_an_adapter_off_the_head_keeps_the_table():
    """An adapter on attention leaves the head a stored matrix: the table,
    read through the adapted model's frozen tree, contracts to its logits."""
    model, variables, adapted, adapted_variables, tokens = _adapted(("q_proj",))
    table = adapted.apply(adapted_variables, method="output_table")
    assert table is not None
    logits = adapted.apply(adapted_variables, tokens, method="logits")
    hidden = adapted.apply(adapted_variables, tokens, method="hidden_states")
    same(head_logits(hidden, table.matrix, softcap=table.softcap, precision=table.precision,
                     vocab_major=table.vocab_major, bias=table.bias), logits)
    assert not np.array_equal(np.asarray(logits), np.asarray(model.apply(variables, tokens)))


# Media through the multimodal reads.

MEDIA = [name for name in CHECKPOINTS
         if (FIXTURES / name / "prompts.json").is_file() and (FIXTURES / name / "raw_images.npy").is_file()]
"""The checkpoints tools/multimodal_reference.py wrote prompts and images
for, beside the reference's own encoding of them."""


@pytest.mark.parametrize("name", MEDIA)
def test_media_reach_the_logits_through_the_fused_reads(name):
    """The processor's request, images placed as the writer placed them (one
    in the first row, two in the second, `_encode`): the reads fuse the
    images as the call does, and blank pixels move the logits."""
    pytest.importorskip("torchvision", reason="the vision extra supplies the actual processors")
    source = loaded(name)
    model, variables = source.model, source.variables
    assert source.processor is not None
    images = np.load(FIXTURES / name / "raw_images.npy")
    inputs = source.processor(json.loads((FIXTURES / name / "prompts.json").read_text()),
                              images=[[images[0]], [images[1], images[2]]])
    fields = inputs.kwargs()
    logits = model.apply(variables, inputs.tokens, method="logits", **fields)
    same(logits, model.apply(variables, inputs.tokens, **fields))
    hidden = model.apply(variables, inputs.tokens, method="hidden_states", **fields)
    same(model.apply(variables, hidden, method="logits_from_hidden"), logits)
    conditioning = fields["conditioning"]
    blank = {**fields, "conditioning": {**conditioning,
                                        "pixel_values": jnp.zeros_like(conditioning["pixel_values"])}}
    valid = np.asarray(inputs.token_fields["attention_mask"])
    moved = np.asarray(model.apply(variables, inputs.tokens, method="logits", **blank)) != np.asarray(logits)
    assert moved[valid].any()


# The JEPA encoders' representations.


JEPA = [case for case in ARCHITECTURES if case.is_jepa]


@pytest.mark.parametrize("case", JEPA, ids=[case.name for case in JEPA])
def test_a_jepa_encoders_hidden_states_are_its_representation(case):
    """Each encoder test_architectures.py trains, at its size there: the
    states are its call's normed tokens, of the whole input and of a
    context subset."""
    encoder = models.build(case.architecture, **case.config)
    sample = jax.random.normal(jax.random.key(0), (ROWS, *case.sample_shape))
    context = jnp.broadcast_to(jnp.arange(MASK.num_context, dtype=jnp.int32), (ROWS, MASK.num_context))
    variables = encoder.init(jax.random.key(1), sample, context)
    same(encoder.apply(variables, sample, method="hidden_states"), encoder.apply(variables, sample))
    same(encoder.apply(variables, sample, method="hidden_states", token_idx=context),
         encoder.apply(variables, sample, context))


# The torchax fallback.


def test_a_torchax_models_reads_are_its_forward(tmp_path):
    """A tiny random GPT-NeoX through transformers' own forward: the logits
    read is the call's, its plain head's stored weight contracts to them,
    it writes no cache, and every field past the tokens is refused by name."""
    pytest.importorskip("torchax")
    import torch
    import transformers

    torch.manual_seed(0)
    config = transformers.GPTNeoXConfig(num_hidden_layers=2, hidden_size=64, num_attention_heads=4,
                                        intermediate_size=256, vocab_size=256, max_position_embeddings=64)
    transformers.GPTNeoXForCausalLM(config).eval().save_pretrained(tmp_path)
    with pytest.warns(UserWarning, match="tier 3"):
        source = Pretrained.load(tmp_path, fallback="torchax", dtype="float32")
    model, variables = source.model, source.variables
    tokens = jnp.asarray(np.random.RandomState(0).randint(0, 256, (ROWS, LENGTH)), jnp.int32)
    logits, written = model.apply(variables, tokens, method="logits", mutable=True)
    assert "cache" not in written
    same(logits, model.apply(variables, tokens))
    table = model.apply(variables, method="output_table")
    assert table is not None
    hidden = model.apply(variables, tokens, method="hidden_states")
    same(head_logits(hidden, table.matrix, softcap=table.softcap, precision=table.precision,
                     vocab_major=table.vocab_major, bias=table.bias), logits)
    for method in ("logits", "hidden_states"):
        with pytest.raises(ValueError, match=r"takes no \['positions'\]"):
            model.apply(variables, tokens, method=method, positions=tokens)
    assert model.causal


# The hooks a task, a server and the trainer read off a model: the same
# model at another cache capacity or remat rung, the projections its decode
# step reads packed, and a request continued into its response.


STEPS = 3
CAPACITY = LENGTH + STEPS
"""Room for the prompt and the greedy steps, below every checkpoint's own capacity."""


def _greedy(model, variables, tokens, **call):
    """The cache `init_cache` allocates, the `STEPS` greedy draws after
    `tokens` and the logits of every call, through `call` (the decode mode,
    or DiffusionGemma's `encode`, the clean commits a canvas reads)."""
    cache = model.apply(variables, ROWS, method="init_cache", mutable=["cache"])[1]["cache"]
    logits, held = model.apply({**variables, "cache": cache}, tokens, mutable=["cache"], **call)
    drawn, scores = [], [logits]
    for _ in range(STEPS):
        drawn.append(jnp.argmax(logits[:, -1], axis=-1).astype(jnp.int32))
        logits, held = model.apply({**variables, "cache": held["cache"]}, drawn[-1][:, None],
                                   mutable=["cache"], **call)
        scores.append(logits)
    return cache, jnp.stack(drawn, axis=1), scores


def _decoding(name: str) -> dict:
    return {"method": "encode"} if name in OWN else {"decode": True}


@pytest.mark.parametrize("name", CHECKPOINTS)
def test_a_model_at_a_cache_capacity_decodes_what_the_model_does_within_it(name):
    """The resized model reads the same variables: its whole-sequence
    logits are the model's, and a causal one decodes into a cache with
    `CAPACITY` slots where the model's had `max_seq_len` (an axis a cache
    derives from it shrinks with it) and draws the same tokens. Its
    attention reduces over the shorter cache, so the cached logits agree to
    rounding."""
    source, read = loaded(name), reads(name, "float32")
    model, variables = source.model, source.variables
    sized = model.with_cache_capacity(CAPACITY)
    same(sized.apply(variables, read.tokens, method="logits"), read.logits)
    if not model.causal:
        return
    cache, drawn, scores = _greedy(model, variables, read.tokens, **_decoding(name))
    sized_cache, sized_drawn, sized_scores = _greedy(sized, variables, read.tokens, **_decoding(name))
    assert jax.tree.structure(sized_cache) == jax.tree.structure(cache)
    for before, after in zip(jax.tree.leaves(cache), jax.tree.leaves(sized_cache), strict=True):
        assert all(old == new or (old, new) == (model.max_seq_len, CAPACITY) or new < old
                   for old, new in zip(before.shape, after.shape, strict=True)), (before.shape, after.shape)
    assert any(CAPACITY in after.shape for after in jax.tree.leaves(sized_cache)) or not any(
        model.max_seq_len in before.shape for before in jax.tree.leaves(cache))
    np.testing.assert_array_equal(np.asarray(sized_drawn), np.asarray(drawn))
    for got, want in zip(sized_scores, scores, strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-5)


def _call(model, variables, inputs, **apply) -> jax.Array:
    """A training forward: a token model's logits, a denoiser's output."""
    return model.apply(variables, *inputs, train=True, rngs={"dropout": jax.random.key(2)}, **apply)


def _stepped(model, variables, inputs, **apply) -> tuple[jax.Array, dict]:
    """A training step's forward and its gradient in every floating parameter."""
    output = _call(model, variables, inputs, **apply)
    cotangent = jax.random.normal(jax.random.key(1), output.shape, output.dtype)
    flat = flatten_dict(variables["params"])
    floats = {path: leaf for path, leaf in flat.items() if jnp.issubdtype(leaf.dtype, jnp.inexact)}
    held = {path: leaf for path, leaf in flat.items() if path not in floats}

    def loss(trained):
        params = unflatten_dict({**held, **trained})
        return jnp.sum(_call(model, {**variables, "params": params}, inputs, **apply) * cotangent)

    return output, jax.grad(loss)(floats)


def _rungs(model) -> list:
    """The model, then every rung `recompute_more` climbs to."""
    rungs = [model]
    while (stronger := rungs[-1].recompute_more()) is not None:
        rungs.append(stronger)
        assert len(rungs) <= 16, "the ladder does not end"
    return rungs


def _climbs_and_restores(model, variables, inputs, **apply) -> None:
    """Each rung holds less than the one below it, and steps the same
    forward and backward bitwise: the step runs eagerly, so each recomputed
    block replays the operations its forward ran. A rung's record restores
    it from any rung below it and moves no rung above it, and the rung
    runs over the model's variables, every collection where it was."""
    rungs = _rungs(model)
    records = [rung.recompute_record() for rung in rungs]
    assert len(rungs) > 1
    assert all(records[index] not in records[:index] for index in range(len(records)))
    for low, rung in enumerate(rungs):
        for high in range(len(rungs)):
            restored = rung.restore_recompute(records[high]).recompute_record()
            assert restored == records[max(low, high)], (records[low], records[high], restored)
    assert all(rung.restore_recompute("no rung").recompute_record() == record
               for rung, record in zip(rungs, records, strict=True))
    output, gradient_tree = _stepped(model, variables, inputs, **apply)
    written = model.apply(variables, *inputs, mutable=True, **apply)[1]
    for rung in rungs[1:]:
        stepped = _stepped(rung, variables, inputs, **apply)
        same(stepped[0], output)
        same(stepped[1], gradient_tree)
        assert rung.apply(variables, *inputs, mutable=True, **apply)[1].keys() == written.keys()


@pytest.mark.parametrize("name", CHECKPOINTS)
def test_a_checkpoints_remat_rungs_step_as_it_does(name):
    source, read = loaded(name), reads(name, "float32")
    _climbs_and_restores(source.model, source.variables, (read.tokens,), method="logits")


RECOMPUTED = [case for case in ARCHITECTURES
              if not case.is_jepa and isinstance(models.build(case.architecture, **case.config), Recomputing)]


def _inputs(case) -> tuple:
    """A batch of what the case's model takes, at test_architectures.py's sizes."""
    if case.is_lm:
        return (jnp.asarray(np.random.RandomState(0).randint(0, VOCAB, (ROWS, case.seq_len)), jnp.int32),)
    sample = jax.random.normal(jax.random.key(0), (ROWS, *case.sample_shape))
    text = jax.random.normal(jax.random.key(3), (ROWS, TEXT_TOKENS, TEXT_FEATURES))
    return sample, jnp.asarray([0.3, 0.7]), TextContext(text, jnp.ones((ROWS, TEXT_TOKENS), bool))


@pytest.mark.parametrize("case", RECOMPUTED, ids=[case.name for case in RECOMPUTED])
def test_an_architectures_remat_rungs_step_as_it_does(case):
    """Every architecture test_architectures.py trains that climbs a remat
    ladder, decoders and DiT stacks, at its size there."""
    model = models.build(case.architecture, **case.config)
    inputs = _inputs(case)
    _climbs_and_restores(model, model.init(jax.random.key(0), *inputs), inputs)


def _packed(variables, groups) -> dict:
    """`variables` with each group's members concatenated, in order, on
    their output axis into its packed projection, as `ProjectionGroup`
    describes the layout its module reads."""
    params = flatten_dict(variables["params"])
    for group in groups:
        fields = {path[-1] for path in params if path[:-1] == (*group.path, group.members[0])}
        for field in fields:
            params[(*group.path, group.packed, field)] = jnp.concatenate(
                [params.pop((*group.path, member, field)) for member in group.members], axis=-1)
    return {**variables, "params": unflatten_dict(params)}


@pytest.mark.parametrize("name", DECODERS)
def test_packed_projections_serve_the_models_numbers(name):
    """Every group a decoder declares packs: the packed variables still
    declare the same groups, and give the same logits and the same cached
    decode. One wider product can round apart from three (falcon-tiny's
    logits moved by 1.2e-7, Kimi K2's by 7.2e-7), so the logits agree to
    rounding and the drawn tokens exactly. Each module the forward runs
    declares no group the model leaves out, and a bidirectional decoder,
    which serves no decode step, declares none."""
    source, read = loaded(name), reads(name, "float32")
    model, variables = source.model, source.variables
    groups = model.inference_projection_groups(variables)
    if not model.causal:
        assert groups == ()
        return
    widths = flatten_dict(variables["params"])
    for group in groups:
        assert all(widths[(*group.path, member, "kernel")].shape[-1] == width
                   for member, width in zip(group.members, group.widths, strict=True))
    packed = _packed(variables, groups)
    assert model.inference_projection_groups(packed) == groups
    _, drawn, scores = _greedy(model, variables, read.tokens, decode=True)
    _, packed_drawn, packed_scores = _greedy(model, packed, read.tokens, decode=True)
    same(packed_drawn, drawn)
    for got, want in zip([model.apply(packed, read.tokens), *packed_scores], [read.logits, *scores],
                         strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-5)

    declared, busy = set(), []

    def declaring(next_fun, args, kwargs, context):
        if not busy and context.method_name == "__call__" and isinstance(context.module, ProjectionSites):
            busy.append(context.module)
            try:
                declared.update(context.module.projection_groups())
            finally:
                busy.pop()
        return next_fun(*args, **kwargs)

    with nn.intercept_methods(declaring):
        model.apply(variables, read.tokens)
    assert declared <= set(groups) and bool(declared) == bool(groups)


def _request() -> ModelInputs:
    """Two left-padded prompts, the second a document reset at slot 3 with
    an image read at slots 3 and 4, and three-axis rotary coordinates."""
    valid = jnp.asarray([[False, False, True, True, True], [False, True, True, True, True]])
    rotary = jnp.asarray([[[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 1], [2, 2, 2]],
                          [[0, 0, 0], [0, 0, 0], [1, 1, 1], [2, 3, 4], [3, 4, 4]]], jnp.int32)
    return ModelInputs(
        jnp.asarray([[0, 0, 5, 6, 7], [0, 8, 9, 10, 11]], jnp.int32),
        {"attention_mask": valid,
         "positions": jnp.asarray([[0, 0, 0, 1, 2], [0, 0, 1, 0, 1]], jnp.int32),
         "segment_ids": jnp.asarray([[0, 0, 1, 1, 1], [0, 1, 1, 2, 2]], jnp.int32),
         "rotary_positions": rotary,
         "image_indices": jnp.asarray([[-1, -1, -1, -1, -1], [-1, -1, -1, 0, 1]], jnp.int32),
         "image_groups": jnp.asarray([[-1, -1, -1, -1, -1], [-1, -1, -1, 0, 0]], jnp.int32)},
        {"pixel_values": jnp.ones((2, 1, 4, 4, 3))})


def test_a_request_continues_into_its_response_by_each_fields_meaning():
    """The response's slots read no media and sit in no image group (-1),
    are real in rows with a real prompt token, count logical positions on
    from the last real token and rotary ones past the largest coordinate,
    stay in the last document, and keep each field's dtype; the prompt's
    slots and the media are as they were."""
    request = _request()
    response = jnp.full((2, 2), 3, jnp.int32)
    extended = request.extended(response)
    fields = extended.token_fields
    same(extended.tokens, jnp.concatenate([request.tokens, response], axis=1))
    for name, value in request.token_fields.items():
        same(fields[name][:, :5], value)
    same(fields["image_indices"][:, 5:], jnp.full((2, 2), -1, jnp.int32))
    same(fields["image_groups"][:, 5:], jnp.full((2, 2), -1, jnp.int32))
    same(fields["attention_mask"][:, 5:], jnp.ones((2, 2), bool))
    same(fields["positions"][:, 5:], jnp.asarray([[3, 4], [2, 3]], jnp.int32))
    same(fields["segment_ids"][:, 5:], jnp.asarray([[1, 1], [2, 2]], jnp.int32))
    same(fields["rotary_positions"][:, 5:],
         jnp.asarray([[[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]], jnp.int32))
    same(extended.conditioning, request.conditioning)
    extended.validate()
    same(jax.jit(lambda inputs: inputs.extended(response))(request), extended)


def test_a_row_without_a_real_token_gets_no_real_response_slot():
    request = ModelInputs(jnp.zeros((2, 3), jnp.int32),
                          {"attention_mask": jnp.asarray([[False] * 3, [True] * 3])})
    same(request.extended(jnp.ones((2, 2), jnp.int32)).token_fields["attention_mask"][:, 3:],
         jnp.asarray([[False, False], [True, True]]))


def test_a_field_with_no_extension_rule_is_refused_and_a_named_rule_extends_it():
    """A routing replay has no value for slots the model has not routed, so
    extending it is refused until the caller names a rule, which also
    replaces a built-in one."""
    tokens = jnp.zeros((2, 3), jnp.int32)
    request = ModelInputs(tokens, {"routed": jnp.ones((2, 3), bool),
                                   "positions": jnp.zeros((2, 3), jnp.int32)})
    with pytest.raises(ValueError, match=r"\['routed'\] have no rule"):
        request.extended(jnp.ones((2, 2), jnp.int32))

    def unrouted(value, valid, width):
        return jnp.zeros((value.shape[0], width), bool)

    def zeros(value, valid, width):
        return jnp.zeros((value.shape[0], width), value.dtype)

    extended = request.extended(jnp.ones((2, 2), jnp.int32), rules={"routed": unrouted, "positions": zeros})
    same(extended.token_fields["routed"][:, 3:], jnp.zeros((2, 2), bool))
    same(extended.token_fields["positions"][:, 3:], jnp.zeros((2, 2), jnp.int32))
    with pytest.raises(ValueError, match="a response is"):
        request.extended(jnp.ones((3, 2), jnp.int32), rules={"routed": unrouted})
