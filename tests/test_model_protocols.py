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
from flax.traverse_util import flatten_dict, unflatten_dict
from test_architectures import CASES as ARCHITECTURES, MASK

from dew.interop import Pretrained, hf_decoders
from dew.lora import LoRA
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.protocols import OutputTable
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
