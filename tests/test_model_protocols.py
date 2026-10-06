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
The last section holds the denoisers and autoencoders to what they declare:
every registered model built tiny by test_precision_policy.py's `build_model`
and `tiny_inputs`, which that file's registry-wide parametrization needs.
"""

import dataclasses
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
from reference_error import assert_as_exact_as_the_reference, widened
from test_architectures import CASES as ARCHITECTURES, MASK, TEXT_FEATURES, TEXT_TOKENS, VOCAB
from test_precision_policy import build_model, tiny_inputs

from dew.interop import Pretrained, hf_decoders
from dew.lora import LoRA
from dew.nn.autoencoders.dc_ae import DCAE, DCAutoencoder
from dew.nn.autoencoders.flux2 import Flux2Autoencoder
from dew.nn.autoencoders.kl import AutoencoderKL, posterior_latent
from dew.nn.autoencoders.sd_vae import StableDiffusionVAE
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.dit import TextContext
from dew.nn.inputs import ModelInputs
from dew.nn.protocols import IntervalModel, OutputTable, Recomputing, RequiresText, TimeScaled
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


def contracted(hidden: jax.Array, table: OutputTable, variables) -> jax.Array:
    """`head_logits` over `table`, whose matrix is one of `variables`' own
    arrays: the stored parameter, which a loss keeps, and no copy of it."""
    assert any(table.matrix is leaf for leaf in jax.tree.leaves(variables))
    return head_logits(hidden, table.matrix, softcap=table.softcap, precision=table.precision,
                       vocab_major=table.vocab_major, bias=table.bias)


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
    same(source.model.apply(source.variables, read.hidden, method="logits_from_hidden")
         if read.table is None else contracted(read.hidden, read.table, source.variables), read.logits)


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
    none for the parameters only the canvas reads. Packed rows reach it as
    they reach the text model: two documents a row, the second's states
    moved off the unpacked row's. `encode`, the cached commit, still moves
    the cache the clean read leaves alone."""
    source, read = loaded(name, dtype), reads(name, dtype)
    model, variables = source.model, source.variables
    text = _text(variables)
    same(read.logits, model.text.apply(text, read.tokens))
    same(read.hidden, model.text.apply(text, read.tokens, method="hidden_states"))
    boundary, places = LENGTH // 2, jnp.arange(LENGTH)
    packing = {"segment_ids": jnp.broadcast_to((places >= boundary).astype(jnp.int32), read.tokens.shape),
               "positions": jnp.broadcast_to(jnp.where(places >= boundary, places - boundary, places),
                                             read.tokens.shape)}
    packed = model.apply(variables, read.tokens, method="hidden_states", **packing)
    same(packed, model.text.apply(text, read.tokens, method="hidden_states", **packing))
    assert not np.array_equal(np.asarray(packed[:, boundary:]), np.asarray(read.hidden[:, boundary:]))
    cotangent = _cotangent(read.logits)
    clean = gradient(model, variables, read.tokens, cotangent, method="logits")
    encoder = gradient(model.text, text, read.tokens, cotangent)
    same({path[1:]: value for path, value in clean.items() if path[0] == "text"}, encoder)
    others = [value for path, value in clean.items() if path[0] != "text"]
    assert others
    assert not any(np.any(np.asarray(value)) for value in others)
    cache = model.apply(variables, ROWS, method="init_cache", mutable=["cache"])[1]["cache"]
    _, written = model.apply({**variables, "cache": cache}, read.tokens, method="encode", mutable=["cache"])
    pairs = zip(jax.tree.leaves(written["cache"]), jax.tree.leaves(cache), strict=True)
    assert not all(np.array_equal(np.asarray(after), np.asarray(before)) for after, before in pairs)


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
    same(contracted(hidden, table, adapted_variables), logits)
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
    same(contracted(hidden, table, variables), logits)
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


def _greedy(model, variables, tokens, forced=None, **call):
    """The cache `init_cache` allocates, the `STEPS` greedy draws after
    `tokens` (or the `forced` ones) and the logits of every call, through
    `call` (the decode mode, or DiffusionGemma's `encode`, the clean commits
    a canvas reads)."""
    cache = model.apply(variables, ROWS, method="init_cache", mutable=["cache"])[1]["cache"]
    logits, held = model.apply({**variables, "cache": cache}, tokens, mutable=["cache"], **call)
    drawn, scores = [], [logits]
    for step in range(STEPS):
        drawn.append(jnp.argmax(logits[:, -1], axis=-1).astype(jnp.int32) if forced is None
                     else forced[:, step])
        logits, held = model.apply({**variables, "cache": held["cache"]}, drawn[-1][:, None],
                                   mutable=["cache"], **call)
        scores.append(logits)
    return cache, jnp.stack(drawn, axis=1), scores


def _decoding(name: str) -> dict:
    return {"method": "encode"} if name in OWN else {"decode": True}


@pytest.mark.parametrize("name", CHECKPOINTS)
def test_a_model_at_a_cache_capacity_decodes_what_the_model_does_within_it(name):
    """The resized model reads the same variables to the model's logits, and
    a causal one decodes into a cache of `CAPACITY` slots and draws the same
    tokens; its attention reduces over the shorter cache, so the cached
    logits agree to rounding."""
    source, read = loaded(name), reads(name, "float32")
    model, variables = source.model, source.variables
    sized = model.with_cache_capacity(CAPACITY)
    same(sized.apply(variables, read.tokens, method="logits"), read.logits)
    if not model.causal:
        return
    cache, drawn, scores = _greedy(model, variables, read.tokens, **_decoding(name))
    sized_cache, sized_drawn, sized_scores = _greedy(sized, variables, read.tokens, **_decoding(name))
    assert jax.tree.structure(sized_cache) == jax.tree.structure(cache)
    assert (any(CAPACITY in leaf.shape for leaf in jax.tree.leaves(sized_cache))
            == any(model.max_seq_len in leaf.shape for leaf in jax.tree.leaves(cache)))
    np.testing.assert_array_equal(np.asarray(sized_drawn), np.asarray(drawn))
    for got, want in zip(sized_scores, scores, strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-5)


def _float64(model):
    """`model` computing in float64, its own `dtype` and every module it
    holds: the truth a float32 run's rounding is measured from."""
    fields = [field.name for field in dataclasses.fields(model) if field.name not in ("parent", "name")]
    held = {name: _float64(getattr(model, name)) for name in fields
            if isinstance(getattr(model, name), nn.Module)}
    return model.clone(**held, **({"dtype": jnp.float64} if "dtype" in fields else {}))


def _flat(tree) -> np.ndarray:
    return np.concatenate([np.ravel(np.asarray(leaf, np.float64)) for leaf in jax.tree.leaves(tree)])


def as_exact(dew, reference, truth, label: str) -> None:
    """`dew` the reference's bits, or where they differ as exact as the
    reference (tests/reference_error.py) from `truth()`, the float64 twin."""
    if all(np.array_equal(np.asarray(mine), np.asarray(theirs))
           for mine, theirs in zip(jax.tree.leaves(dew), jax.tree.leaves(reference), strict=True)):
        return
    with jax.enable_x64(new_val=True):
        wanted = truth()
    assert_as_exact_as_the_reference(_flat(dew), _flat(reference), _flat(wanted), label)


def _call(model, variables, inputs, **apply) -> jax.Array:
    """A forward without dropout, whose float64 twin would draw other masks:
    a token model's logits, a denoiser's output."""
    return model.apply(variables, *inputs, **apply)


def _stepped(model, variables, inputs, **apply) -> tuple[jax.Array, dict]:
    """A step's forward and its gradient in every floating parameter."""
    output = _call(model, variables, inputs, **apply)
    cotangent = jax.random.normal(jax.random.key(1), output.shape, jnp.float32).astype(output.dtype)
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
    """Each rung holds less than the one below it and steps the same forward,
    bitwise. Its backward recomputes, which can round apart (gemma2-tiny's
    gradients by up to 1.2e-4, kimi-k3-tiny's 8.3e-3 of 10.8; test_decoder_remat.py
    found the same), so the gradients are the model's bits or as exact as them.
    A rung's record restores it from any rung below it and moves no rung
    above it, and the rung runs over the model's variables, every collection
    where it was."""
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
    truth = functools.cache(
        lambda: _stepped(_float64(model), widened(variables), widened(inputs), **apply)[1])
    written = model.apply(variables, *inputs, mutable=True, **apply)[1]
    for rung in rungs[1:]:
        stepped = _stepped(rung, variables, inputs, **apply)
        same(stepped[0], output)
        as_exact(stepped[1], gradient_tree, truth, f"gradient at {rung.recompute_record()}")
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
    logits moved by 1.2e-7, Kimi K2's by 7.2e-7, qwen3-next-tiny's by 1.8e-5
    on another CPU), so the drawn tokens are the model's and the logits its
    bits or as exact as them. A bidirectional decoder, which serves no
    decode step, declares none."""
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

    def truth():
        wide, wide_variables = _float64(model), widened(variables)
        return [wide.apply(wide_variables, read.tokens, method="logits"),
                *_greedy(wide, wide_variables, read.tokens, forced=drawn, decode=True)[2]]

    as_exact([model.apply(packed, read.tokens, method="logits"), *packed_scores], [read.logits, *scores],
             truth, f"{name} packed")


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


# Autoencoders and denoisers: the KL posterior an objective trains an
# autoencoder through, the text a denoiser cannot run without, and the
# interval and time scale it reads its time in.


def kl_module(dtype=jnp.float32) -> AutoencoderKL:
    return AutoencoderKL(channels=(8, 16), latent_channels=4, blocks_per_level=1, norm_groups=4, dtype=dtype)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["float32", "bfloat16"])
def test_a_kl_autoencoder_gives_the_posterior_end_to_end_tuning_trains_through(dtype):
    """REPA-E's step (objective.py's `_end_to_end_latents`) read the
    posterior and decoded its draw by applying the autoencoder's module
    itself. `moments` and `decode_raw` give the same arrays and the same
    gradients, bit for bit, and `encode_batch`'s latent is a draw from that
    posterior, which is what lets the step renormalize the draw itself."""
    module = kl_module(dtype)
    params = module.init(jax.random.PRNGKey(5), jnp.zeros((1, 8, 8, 3)))["params"]
    autoencoder = StableDiffusionVAE(model=module, params=params, dtype=dtype, latent_shift=0.1,
                                     latent_scale=0.8)
    images = jax.random.uniform(jax.random.PRNGKey(1), (8, 8, 8, 3), minval=-1.0, maxval=1.0)
    key = jax.random.PRNGKey(2)

    def module_applied(tree):
        weights = {"params": tree}
        moments = module.apply(weights, images, method=module.moments)
        return moments, module.apply(weights, posterior_latent(moments, key), method=module.decode)

    def asked(tree):
        moments = autoencoder.moments(tree, images)
        return moments, autoencoder.decode_raw(tree, posterior_latent(moments, key))

    def loss(read):
        return lambda tree: sum(jnp.sum(jnp.square(part.astype(jnp.float32))) for part in read(tree))

    for transform in (lambda read: read, lambda read: jax.grad(loss(read))):
        got, expected = (jax.jit(transform(read))(params) for read in (asked, module_applied))
        for leaf, reference in zip(jax.tree.leaves(got), jax.tree.leaves(expected), strict=True):
            np.testing.assert_array_equal(leaf, reference)
    drawn = jax.jit(lambda tree: posterior_latent(autoencoder.moments(tree, images), key))(params)
    np.testing.assert_array_equal(autoencoder.encode_batch(params, images, key), drawn)


@pytest.mark.parametrize("autoencoder", [
    Flux2Autoencoder(model=kl_module(), params={}, mean=np.zeros(16), variance=np.ones(16), epsilon=1e-4),
    DCAutoencoder(model=DCAE(), params={}, latent_scale=1.0),
], ids=lambda autoencoder: type(autoencoder).__name__)
def test_an_autoencoder_whose_latent_is_no_kl_draw_refuses_by_name(autoencoder):
    """Two autoencoders that inherit the base refusal. FLUX.2's latent folds
    2x2 blocks of its AutoencoderKL's draw and normalizes them by batch-norm
    statistics, and a DC-AE encodes without a posterior: neither latent is a
    draw an objective may renormalize, so each refuses the posterior and its
    decode, naming itself."""
    name = type(autoencoder).__name__
    with pytest.raises(TypeError, match=name):
        autoencoder.moments(autoencoder.params, jnp.zeros((1, 8, 8, 3)))
    with pytest.raises(TypeError, match=name):
        autoencoder.decode_raw(autoencoder.params, jnp.zeros((1, 2, 2, 4)))


def denoiser(architecture: str, rng):
    """The registered `architecture` at a tiny size in float32, its sample,
    time and the rest of its tiny inputs; None where those are not a batch
    of samples and one time per row, a denoiser's."""
    inputs = tiny_inputs(architecture, rng)
    (sample, *args), conditions = inputs if isinstance(inputs[0], tuple) else (inputs, {})
    if not args or sample.ndim < 4 or jnp.shape(args[0]) != sample.shape[:1]:
        return None
    return build_model(architecture, "float32"), sample, args[0], tuple(args[1:]), conditions


def perturbed(variables, rng):
    """`variables` with noise on every parameter, so a zero-initialized head
    hides nothing a change of input does."""
    leaves, tree = jax.tree.flatten(variables["params"])
    noisy = [leaf + 0.1 * jax.random.normal(key, leaf.shape, leaf.dtype)
             for leaf, key in zip(leaves, jax.random.split(rng, len(leaves)), strict=True)]
    return {**variables, "params": jax.tree.unflatten(tree, noisy)}


@pytest.mark.parametrize("architecture", [architecture for architecture in sorted(models)
                                          if denoiser(architecture, jax.random.PRNGKey(0))])
def test_a_denoiser_reads_the_text_interval_and_time_it_declares(architecture, rng):
    """What the diffusion recipe relies on, held to every registered denoiser's declarations.

    An unconditional run builds and calls its denoiser on the sample and
    time alone: one declaring `RequiresText` refuses that, naming the keyword
    its text arrives under, and runs on that keyword's text; every other
    runs. An interval process hands its model each step's `duration`: cloned
    with `interval` set, an `IntervalModel` embeds it, a missing one as the
    zero duration, and cloned without refuses one, as every other denoiser
    does. MeanFlow and sCM slow a `TimeScaled` denoiser's time features
    through `time_scale`, which is the time's unit: at 16 on t and d it
    computes bit for bit what it computes at 2 on 8t and 8d, powers of two
    keeping both sides exact.
    """
    model, sample, time, rest, conditions = denoiser(architecture, rng)
    if isinstance(model, RequiresText):
        variables = model.init(rng, sample, time, *rest, **conditions)
        with pytest.raises(TypeError, match=model.text_keyword):
            model.apply(variables, sample, time)
        text = conditions.get(model.text_keyword, rest[0] if rest else None)
        denoised = model.apply(variables, sample, time, **{model.text_keyword: text})
    else:
        denoised = model.apply(model.init(rng, sample, time), sample, time)
    assert denoised.shape == sample.shape and bool(jnp.all(jnp.isfinite(denoised)))

    def built(model, scale=None):
        model = model if scale is None else model.clone(time_scale=scale)
        variables = perturbed(model.init(rng, sample, time, *rest, **conditions), rng)
        return lambda at, **duration: model.apply(variables, sample, at, *rest, **duration, **conditions)

    duration = jnp.full_like(time, 0.25)
    if isinstance(model, IntervalModel):
        with pytest.raises(ValueError, match="duration"):
            built(model.clone(interval=False))(time, duration=duration)
        model = model.clone(interval=True)
        call = built(model)
        np.testing.assert_array_equal(call(time), call(time, duration=jnp.zeros_like(time)))
        assert not np.array_equal(call(time), call(time, duration=duration))
    else:
        with pytest.raises(TypeError, match="duration"):
            built(model)(time, duration=duration)
    if isinstance(model, TimeScaled):
        spanned = isinstance(model, IntervalModel)
        fast, slow = (built(model, scale)(unit * time, **({"duration": unit * duration} if spanned else {}))
                      for scale, unit in ((16.0, 1.0), (2.0, 8.0)))
        assert not np.array_equal(fast, jnp.zeros_like(fast))
        np.testing.assert_array_equal(fast, slow)
