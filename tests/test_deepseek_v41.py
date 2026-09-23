"""DeepSeek-V4.1-Flash against the release's own inference code.

deepseek-v41-tiny comes from tools/deepseek_v41_reference.py: the release's
inference/model.py at dba1be0a (DeepSeek-V4.1-Flash) run in fp32 over the
torch stand-ins for its kernels, which match the tilelang kernels bit for bit
on a GPU (the tool's docstring). The architecture outputs are taken with the
quantizers off, since across that many forwards some value always sits
within fp32 noise of a rounding boundary; `qat_logits` is the one forward
with them on, at a seed whose margins keep it clear of every boundary.

Observed on CPU in fp32, tolerance 1e-4: logits 8.0e-6 without and 5.4e-6
with quantization-aware rounding, prefill 3.9e-6 and teacher-forced decode
5.4e-6, DSpark draft logits 5.7e-6 and confidence 8.7e-6 with identical
draft ids, greedy ids exact.
"""

import dataclasses
import json
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.interop import load_pretrained
from dew.interop.hf_decoders import _FAMILIES, _flatten, translate_config
from dew.nn.engram import Engram
from dew.nn.fake_quant import fake_quant_fp4, fake_quant_fp8
from dew.nn.inputs import ModelInputs
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective
from dew.registry import models
from dew.sampling import Sampling, generate

ROOT = Path(__file__).parent / "fixtures" / "hf"
TINY = ROOT / "deepseek-v41-tiny"
RELEASED = ROOT / "deepseek-v41-flash"
TOLERANCE = 1e-4


def unquantized(model):
    """The model with its quantization-aware rounding off, as the reference's
    architecture outputs were taken."""
    kinds = {name: dataclasses.replace(kind, mixer=dataclasses.replace(kind.mixer, kv_qat=False))
             for name, kind in model.kinds.items()}
    return model.clone(kinds=flax.core.freeze(kinds))


@pytest.fixture(scope="module")
def source():
    loaded = load_pretrained(TINY, dtype="float32", attention_impl="reference")
    return loaded, unquantized(loaded.model), np.load(TINY / "reference.npz")


def close(actual, expected):
    actual = np.asarray(actual)
    np.testing.assert_allclose(actual, expected, atol=TOLERANCE, rtol=0)
    np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))


def test_the_quantizers_round_as_the_release_kernels():
    """FP8 over 32 channels under power-of-two scales, FP4 over 16 under E4M3
    scales and over 32 under power-of-two scales, bit for bit against the
    kernels' torch stand-ins, ties, all-zero blocks and saturation included,
    compiled, on whichever backend runs the test."""
    torch = pytest.importorskip("torch")
    from tools import deepseek_v41_kernels as kernels

    rng = np.random.default_rng(0)
    blocks = [rng.standard_normal((64, 64)) * scale * rng.random((64, 1)) * 3
              for scale in (1e-3, 0.3, 3.0, 40.0, 1e4)]
    ties = np.tile(np.array([0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6, 6]), 4)
    ties[0] = 6.0  # the block's amax, so its scale is exactly one
    x = np.concatenate([*blocks, np.stack([ties, -ties])]).astype(np.float32)
    x[3, :32] = 0
    for jax_quant, torch_quant in (
            (lambda v: fake_quant_fp8(v, 32), lambda v: kernels.act_quant(v, 32, "ue8m0", None, True)),
            (lambda v: fake_quant_fp4(v, 16, True),
             lambda v: kernels.fp4_act_quant(v, 16, True, torch.float8_e4m3fn)),
            (lambda v: fake_quant_fp4(v, 32, False), lambda v: kernels.fp4_act_quant(v, 32, True))):
        expected = torch_quant(torch.from_numpy(x.copy())).numpy()
        # under jit, where XLA GPU would delete a convert-pair rounding
        np.testing.assert_array_equal(np.asarray(jax.jit(jax_quant)(jnp.asarray(x))), expected)


def test_the_quantizers_pass_their_gradient_straight_through():
    x = jnp.linspace(-3.0, 3.0, 64).reshape(2, 32)
    for quant in (lambda v: fake_quant_fp8(v, 32), lambda v: fake_quant_fp4(v, 16, True)):
        np.testing.assert_array_equal(jax.grad(lambda v: jnp.sum(quant(v) * v))(x),
                                      quant(x) + x)
    # bf16 input whose E4M3 scale saturates: the forward value is the
    # rounded one exactly, not a bf16 re-rounding of the correction
    big = jnp.asarray([[1e4] + [3.0] * 15], jnp.bfloat16)
    rounded = fake_quant_fp4(big.astype(jnp.float32), 16, True).astype(jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(fake_quant_fp4(big, 16, True)), np.asarray(rounded))


def test_the_engram_hash_is_the_releases_int64_arithmetic():
    """At the release's sizes (a compressed vocabulary of 99092, buckets
    above 16M, 8 heads over 2- to 4-grams) the limb arithmetic equals
    int64 multiply, XOR and modulo, whose products reach 2**63."""
    released = json.loads((RELEASED / "config.json").read_text())["text_config"]
    spec = Engram(layer_ids=released["engram_layer_ids"], num_embeddings=released["engram_num_embeddings"],
                  max_ngram_size=released["engram_max_ngram_size"], vocab_size=released["engram_vocab_size"],
                  n_heads=released["engram_n_heads"], head_dim=released["engram_head_dim"],
                  compressed_vocab_size=released["engram_compressed_vocab_size"])
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, spec.compressed_vocab_size, (3, 40, spec.max_ngram_size))
    tokens[0, 0] = spec.compressed_vocab_size - 1
    blocked = rng.random(tokens.shape) < 0.1
    pad = 7
    ids = np.asarray(spec.hash_ids(jnp.asarray(tokens, jnp.int32), jnp.asarray(blocked), pad))
    values = np.where(blocked, pad, tokens).astype(np.int64)
    products = values[:, :, None, :] * spec.multipliers  # [B, S, layers, n]
    primes = np.asarray(spec.primes, np.int64)
    offsets = np.concatenate([np.zeros((len(spec.layer_ids), 1), np.int64),
                              np.cumsum(primes.reshape(len(spec.layer_ids), -1), axis=1)[:, :-1]], axis=1)
    rolling, expected = products[..., 0], []
    for size in range(1, spec.max_ngram_size):
        rolling = np.bitwise_xor(rolling, products[..., size])
        expected.append(rolling[..., None] % primes[:, size - 1])
    np.testing.assert_array_equal(ids, np.concatenate(expected, -1) + offsets)


def test_every_released_tensor_lands_on_one_leaf_of_the_released_tree():
    """The pinned weight index's 96085 tensors, their FP8/FP4 `.scale`
    partners aside, map onto the tree the released config builds, and
    together they cover it; the vision tower, its aligner, the image span
    embeddings and the routers' image-token bias are the vision half, which
    the text model retains by name."""
    config = json.loads((RELEASED / "config.json").read_text())
    fields = translate_config(config)
    family = _FAMILIES["deepseek_v41"]
    text = config["text_config"]
    names = []
    for name in json.loads((RELEASED / "tensor_names.json").read_text())["names"]:
        if ".experts.K." in name:
            experts = (text["dspark_n_routed_experts"] if name.startswith("mtp.")
                       else text["n_routed_experts"])
            names += [name.replace(".K.", f".{index}.") for index in range(experts)]
        else:
            names.append(name)
    assert len(names) == len(set(names))
    placeholders = {name: np.zeros((2, 4, 2) if name.endswith("wo_a.weight") else (1,)) for name in names}
    placeholders.update({name.removesuffix("wo_a.weight") + "wq_b.weight": np.zeros((8, 1))
                         for name in names if name.endswith("wo_a.weight")})
    prepared = family.prepare_weights(placeholders)
    paths = {name: family.weight_path(name, fields) for name in prepared}
    retained = {name for name, path in paths.items() if path is None}
    assert retained and all(name.startswith(("vision.", "aligner.", "image_")) or name.endswith("bias_vl")
                            for name in retained)
    bound = [path for path in paths.values() if path is not None]
    assert len(set(bound)) == len(bound)
    # One tensor per expert stacks into one leaf per projection.
    bound = {tuple(part for index, part in enumerate(path)
                   if not (index and path[index - 1] == 'experts' and part.isdigit()))
             for path in bound}
    model = models.build("causal_transformer", **fields)
    shapes = jax.eval_shape(lambda: model.init(jax.random.key(0), jnp.zeros((1, 4), jnp.int32)))
    tree = {tuple(name.split(".")) for name in _flatten(dict(shapes))}
    tree.discard(("constants", "engram_hashes", "token_map"))
    assert bound == tree
    assert model.num_layers == 40 and model.dspark.stages == 3 and model.mixture.experts == 384


def test_the_forward_matches_the_reference(source):
    loaded, plain, reference = source
    ids = jnp.asarray(reference["input_ids"])
    close(jax.jit(plain.apply)(loaded.variables, ids), reference["logits"])


def test_the_quantization_aware_forward_matches_the_reference(source):
    loaded, _, reference = source
    ids = jnp.asarray(reference["input_ids"])
    close(jax.jit(loaded.model.apply)(loaded.variables, ids), reference["qat_logits"])


def test_the_candidate_pool_decides_the_logits(source):
    """Dropping the pool's restriction on the Reindex layers moves the logits
    past the tolerance, so the fixture exercises the hierarchical indexer."""
    loaded, plain, reference = source
    ids = jnp.asarray(reference["input_ids"])
    kinds = dict(plain.kinds)
    kinds["csa2_ratio_1_reindex"] = dataclasses.replace(
        kinds["csa2_ratio_1_reindex"], mixer=dataclasses.replace(
            kinds["csa2_ratio_1_reindex"].mixer, candidates=None, candidate_blocks=None,
            candidate_block_size=None))
    unpooled = plain.clone(kinds=flax.core.freeze(kinds))
    assert np.max(np.abs(np.asarray(unpooled.apply(loaded.variables, ids)) - reference["logits"])) > TOLERANCE


def test_the_update_exports_and_decodes_as_the_reference(source, tmp_path):
    """Loss, one SGD step with every gradient (the indexer's are zero, as the
    reference's selection passes none), the source-layout export read back
    bit for bit, and greedy decoding after the fixture's prompt."""
    loaded, plain, reference = source
    ids = jnp.asarray(reference["input_ids"])
    inputs = ModelInputs(ids)
    objective = LMObjective(plain, ids.shape[1] - 1, pretrained=loaded.variables, ema_decay=None)
    step = Step(step=jnp.int32(0), key=jax.random.key(0), ema=None)

    def loss(params):
        statistics, _ = objective.loss({**loaded.variables, "params": params}, {"text": inputs}, step)
        return objective.reduce_loss(statistics)[0]

    value, gradient = jax.jit(jax.value_and_grad(loss))(loaded.variables["params"])
    np.testing.assert_allclose(value, reference["loss"], atol=1e-5, rtol=0)
    indexer = [np.max(np.abs(leaf)) for path, leaf in _flatten(gradient).items() if ".indexer." in f".{path}."]
    assert indexer and max(indexer) == 0
    variables = {**loaded.variables, "params": jax.tree.map(
        lambda weight, grad: weight - reference["learning_rate"] * grad, loaded.variables["params"], gradient)}
    loaded.save(tmp_path, variables=variables)
    restored = load_pretrained(tmp_path, dtype="float32", attention_impl="reference")
    held, again = _flatten(variables), _flatten(restored.variables)
    assert held.keys() == again.keys()
    for name, leaf in again.items():
        np.testing.assert_array_equal(np.asarray(leaf), np.asarray(held[name]), err_msg=name)
    close(unquantized(restored.model).apply(restored.variables, ids), reference["updated_logits"])
    prompt = int(reference["decode_prompt"])
    generated = generate(plain, loaded.variables, ModelInputs(ids[:, :prompt]),
                         reference["generated"].shape[1], key=jax.random.key(1),
                         sampling=Sampling(temperature=0))
    np.testing.assert_array_equal(np.asarray(generated.tokens)[:, -reference["generated"].shape[1]:],
                                  reference["generated"])


def cached_run(model, variables, ids, prompt):
    """Prefill `prompt` tokens, then one token per step; the prompt's logits,
    each step's logits, and DSpark's drafts after each step."""
    rows = ids.shape[0]
    cache = model.apply(variables, rows, method=model.init_cache, mutable=["cache"])[1]
    drafts = model.apply(variables, rows, method=model.init_draft_cache, mutable=["cache"])[1]
    (hidden, prompt_logits), cache = model.apply(
        {**variables, **cache}, ids[:, :prompt], decode=True, method=model.states_and_logits, mutable=["cache"])
    _, drafts = model.apply({**variables, **drafts}, hidden, None, method=model.draft, mutable=["cache"])
    steps, drafted = [], []
    for position in range(prompt, ids.shape[1]):
        (hidden, logits), cache = model.apply(
            {**variables, **cache}, ids[:, position:position + 1], decode=True,
            method=model.states_and_logits, mutable=["cache"])
        steps.append(np.asarray(logits[:, 0]))
        out, drafts = model.apply({**variables, **drafts}, hidden, jnp.argmax(logits[:, -1], -1),
                                  method=model.draft, mutable=["cache"])
        drafted.append([np.asarray(value) for value in out])
    return np.asarray(prompt_logits), np.stack(steps, 1), [np.stack(value, 1) for value in zip(*drafted, strict=True)]


def test_the_decode_cache_and_the_drafter_match_the_reference(source):
    """The cached prefill and teacher-forced steps cover the window ring,
    ratio-2 windows closing across calls, the shared entries, index keys,
    selections and candidate pool, and the engram history; DSpark's windows
    seed on the prompt and draft after every step."""
    loaded, plain, reference = source
    ids = jnp.asarray(reference["input_ids"])
    prompt_logits, steps, (draft_ids, draft_logits, confidence) = cached_run(
        plain, loaded.variables, ids, int(reference["decode_prompt"]))
    close(prompt_logits, reference["prompt_logits"])
    close(steps, reference["decode_logits"])
    close(draft_logits, reference["draft_logits"])
    np.testing.assert_allclose(confidence, reference["draft_confidence"], atol=TOLERANCE, rtol=0)
    np.testing.assert_array_equal(draft_ids, reference["draft_ids"])
