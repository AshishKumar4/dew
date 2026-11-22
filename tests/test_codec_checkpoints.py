"""Every quantized checkpoint fixture through Dew's public load and save.

Each one computes its format's own reference forward over the same files,
saves back to the bytes it was read from, and reloads to the same logits:
ModelOpt's NVFP4 and its per-layer FP8 and NVFP4 table, compressed-tensors'
NVFP4 with input QDQ under either local-scale rounding, and weight-only FP8
and NVFP4 storage. The weight readers, activation quantizers and gradients
the forwards compose are each format's own file's: tests/test_modelopt_nvfp4.py,
tests/test_modelopt_mixed.py, tests/test_nvfp4_qdq.py and
tests/test_nvfp4_storage.py.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from interop_support import NVFP4_QDQ_CASES, assert_same_stored_tensors
from recorded_runs import assert_arms_as_recorded

from dew.interop import Pretrained
from dew.interop.safetensors_io import read_weights
from dew.registry import ModelConfig

CODECS = Path(__file__).parent / "fixtures" / "codecs"
CHECKPOINTS = ("modelopt/tiny", "modelopt/mixed", *(f"nvfp4_qdq/{kind}" for kind in NVFP4_QDQ_CASES),
               "storage/fp8", "storage/nvfp4")
RECORDED = ("modelopt/tiny", "modelopt/mixed", "storage/fp8", "storage/nvfp4",
            *(f"nvfp4_qdq/{kind}" for kind in NVFP4_QDQ_CASES))
"""Every checkpoint, recorded over drawn ids by its author's own CPU forward, where Dew's arm runs:
compressed-tensors' reader (tools/codec_worker.py) and ModelOpt's (tools/modelopt_worker.py, rows 4741 and
5244)."""
CODEC_SEEDS = dict.fromkeys(RECORDED, 8)
"""Each checkpoint's declared seeds, sized by `seeds_for` paired as its recording confirmed them."""


def site(name: str) -> str:
    """tools/venv_reference.py's site of checkpoint `name`: `codec-<family>-<kind>`."""
    return "codec-" + name.replace("/", "-")


def drawing(name: str):
    """`draw(seed)` of checkpoint `name`: two rows of ids over its vocabulary, as long as its stored ids."""
    directory = CODECS / name
    vocab = json.loads((directory / "config.json").read_text())["vocab_size"]
    with np.load(directory / "reference.npz") as reference:
        shape = reference["ids"].shape

    def draw(seed: int) -> dict[str, np.ndarray]:
        return {"ids": np.random.default_rng([47, seed]).integers(0, vocab, shape).astype(np.int32)}
    return draw


def computing(name: str):
    """`dew(dtype, inputs)` of checkpoint `name`: its logits at `dtype` through Dew's public load, the model
    rebuilt in float64 for a float64 run (`ModelConfig.with_dtype`) over the same decoded weights."""
    loaded = Pretrained.load(CODECS / name, dtype="float32", attention_impl="reference")
    wide = ModelConfig.from_model(loaded.model).with_dtype("float64").build()

    def dew(dtype, inputs) -> dict:
        model = loaded.model if dtype == jnp.float32 else wide
        variables = jax.tree.map(lambda leaf: jnp.asarray(leaf, dtype) if jnp.issubdtype(
            jnp.asarray(leaf).dtype, jnp.floating) else leaf, loaded.variables)
        return {"logits": model.apply(variables, inputs["ids"])}
    return dew


def pipeline_arms(name: str):
    """tools/venv_reference.py's site of checkpoint `name`: its recording, draw, Dew run and leaf."""
    return CODECS / name / "recording.npz", drawing(name), computing(name), ("logits",)


@pytest.mark.parametrize("name", RECORDED)
def test_a_quantized_checkpoint_computes_its_reference(name):
    """The reference is the format author's own CPU forward of the same files, and Dew's logits are on par
    with it over drawn ids (check 1). Float64 widens the weights the author's reader decodes exactly
    (weight-only NVFP4 rounds them to bf16, FP8 to its stored scale's dtype), so the format's quantization is
    common to both forwards and the check measures only their rounding."""
    assert_arms_as_recorded(pipeline_arms(name), site(name), CODEC_SEEDS[name], f"{name} logits")


@pytest.mark.parametrize("name", CHECKPOINTS)
def test_a_quantized_checkpoint_saves_its_own_bytes(name, tmp_path):
    """The checkpoint held in its own storage (`param_dtype='auto'`) saves its own bytes; held in float32
    masters it saves each dense weight the source stored narrower at the masters' width, the same values,
    and every other tensor's bytes. Both reload to the same logits."""
    directory = CODECS / name
    with np.load(directory / "reference.npz") as reference:
        ids = reference["ids"].astype(np.int32)
    loaded = Pretrained.load(directory, dtype="float32", attention_impl="reference")
    actual = np.asarray(loaded.model.apply(loaded.variables, ids))
    stored = read_weights(directory)
    own = tmp_path / "own"
    Pretrained.load(directory, dtype="float32", param_dtype="auto", attention_impl="reference").save(own)
    assert_same_stored_tensors(read_weights(own), stored)
    masters = tmp_path / "masters"
    loaded.save(masters)
    written = read_weights(masters)
    assert set(written) == set(stored)
    for tensor, value in stored.items():
        if written[tensor].dtype == value.dtype:
            assert_same_stored_tensors({tensor: written[tensor]}, {tensor: value})
        else:
            assert jnp.issubdtype(value.dtype, jnp.floating) and written[tensor].dtype == np.float32, tensor
            np.testing.assert_array_equal(written[tensor], value.astype(np.float32), err_msg=tensor)
    for saved in (own, masters):
        reloaded = Pretrained.load(saved, dtype="float32", attention_impl="reference")
        np.testing.assert_array_equal(reloaded.model.apply(reloaded.variables, ids), actual)


E2M1_VALUES = np.asarray([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)
"""Each E2M1 code's value, its sign bit 3, code 8 read as +0 (ModelOpt's reader's)."""

PACKED_MLP = ("gate_proj", "up_proj", "down_proj")
"""The Linears `_w4a16_qwen3` stores as ModelOpt W4A16_NVFP4."""

PACKED_ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")
"""The attention's Linears `_w4a16_qwen3` stores as W4A16_NVFP4 too where asked, beside their biases."""


def _w4a16_qwen3(directory: Path, attention: bool = False) -> Path:
    """A one-layer Qwen3 whose MLP's Linears ModelOpt stores as W4A16_NVFP4 (`quant_algo`, the layers whose
    scales are stored): U8 pairs of drawn codes, drawn E4M3 block scales and a float32 global scale, each
    depth one NVFP4 stage (256, `formats.NVFP4_STAGE`); its attention bf16, or with `attention` W4A16 too
    and biased (`attention_bias`, Qwen2's), its norms and tied embedding bf16."""
    import ml_dtypes
    from safetensors.numpy import save_file

    width, inner, heads, kv, head = 256, 256, 4, 2, 64
    rng = np.random.default_rng(0x8550)
    bf16 = ml_dtypes.bfloat16

    def dense(*shape):
        return (rng.standard_normal(shape) * 0.05).astype(bf16)

    layer = "model.layers.0."
    tensors = {"model.embed_tokens.weight": dense(96, width), "model.norm.weight": np.ones(width, bf16),
               layer + "input_layernorm.weight": np.ones(width, bf16),
               layer + "post_attention_layernorm.weight": np.ones(width, bf16),
               layer + "self_attn.q_norm.weight": np.ones(head, bf16),
               layer + "self_attn.k_norm.weight": np.ones(head, bf16),
               layer + "self_attn.q_proj.weight": dense(heads * head, width),
               layer + "self_attn.k_proj.weight": dense(kv * head, width),
               layer + "self_attn.v_proj.weight": dense(kv * head, width),
               layer + "self_attn.o_proj.weight": dense(width, heads * head)}
    packed = {f"mlp.{name}": shape for name, shape in zip(
        PACKED_MLP, ((inner, width), (inner, width), (width, inner)), strict=True)}
    if attention:
        packed |= {f"self_attn.{name}": tensors[f"{layer}self_attn.{name}.weight"].shape
                   for name in PACKED_ATTENTION}
        tensors |= {f"{layer}self_attn.{name}.bias": dense(packed[f"self_attn.{name}"][0])
                    for name in PACKED_ATTENTION}
    for name, (outputs, inputs) in packed.items():
        stem = f"{layer}{name}.weight"
        tensors[stem] = rng.integers(0, 256, (outputs, inputs // 2)).astype(np.uint8)
        scales = rng.uniform(0.25, 4, (outputs, inputs // 16))
        tensors[stem + "_scale"] = scales.astype(ml_dtypes.float8_e4m3fn)
        tensors[stem + "_scale_2"] = np.asarray(rng.uniform(0.002, 0.02), np.float32)
    config = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": width,
              "intermediate_size": inner, "num_hidden_layers": 1, "num_attention_heads": heads,
              "num_key_value_heads": kv, "head_dim": head, "hidden_act": "silu", "vocab_size": 96,
              "max_position_embeddings": 64, "rms_norm_eps": 1e-6, "tie_word_embeddings": True,
              "attention_bias": attention, "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
              "layer_types": ["full_attention"], "dtype": "bfloat16",
              "quantization_config": {"quant_method": "modelopt", "quant_algo": "W4A16_NVFP4",
                                      "producer": {"name": "modelopt", "version": "0.47.0"}}}
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps(config))
    save_file(tensors, str(directory / "model.safetensors"), metadata={"format": "pt"})
    return directory


def test_a_modelopt_sources_w4a16_linears_served_packed_are_on_par_with_them_decoded(tmp_path):
    """A ModelOpt source's W4A16_NVFP4 Linears loaded packed (`Pretrained.load`'s `packed_weights`, row
    8550): each kernel an NVFP4 matrix in the `QUANTIZATION` collection, no parameter beside it, holding
    exactly the values its codes, block scales and global scale state (decoded, bitwise, against the
    stored tensors read here); the packed model's logits on par with the same model over those values
    decoded (check 1 in bf16, the truth the decoded model in float64); and its export the source's own
    bytes."""
    from flax.traverse_util import flatten_dict
    from reference_error import assert_as_exact

    from dew.nn.kernels.formats import NvFp4
    from dew.quantization import QUANTIZATION, decoded_banks

    source = _w4a16_qwen3(tmp_path / "source")
    packed = Pretrained.load(source, dtype="bfloat16", attention_impl="reference", packed_weights=True)
    banks = {path: bank for path, bank in flatten_dict(
        dict(packed.variables[QUANTIZATION]), is_leaf=lambda _, leaf: isinstance(leaf, NvFp4)).items()
        if isinstance(bank, NvFp4)}
    assert sorted(path[-2] for path in banks) == sorted(PACKED_MLP)
    assert all(path[-1] == "kernel" for path in banks)
    params = flatten_dict(dict(packed.variables["params"]))
    assert not set(banks) & set(params)
    stored = {name: np.asarray(value) for name, value in read_weights(source).items()}
    dense = decoded_banks(packed.variables)
    held = flatten_dict(dict(dense["params"]))
    for path in banks:
        stem = f"model.layers.0.mlp.{path[-2]}.weight"
        codes = stored[stem]
        values = E2M1_VALUES[np.stack([codes & 15, codes >> 4], axis=-1).reshape(codes.shape[0], -1)]
        scales = np.repeat(stored[stem + "_scale"].astype(np.float32), 16, axis=-1)
        expected = (values * scales * stored[stem + "_scale_2"]).T
        np.testing.assert_array_equal(np.asarray(held[path], np.float32).view(np.uint32),
                                      expected.view(np.uint32), err_msg=stem)

    def draw(seed: int):
        return (jax.random.randint(jax.random.key(seed), (2, 8), 0, 96),)

    wide = ModelConfig.from_model(packed.model).with_dtype("float64").build()

    def truth(tokens):
        variables = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64) if jnp.issubdtype(
            jnp.asarray(leaf).dtype, jnp.floating) else leaf, dense)
        return wide.apply(variables, tokens)

    assert_as_exact(jax.jit(lambda tokens: packed.model.apply(packed.variables, tokens)),
                    jax.jit(lambda tokens: packed.model.apply(dense, tokens)), draw,
                    "a ModelOpt W4A16 model served packed", seeds=8, truth_fn=truth)
    exported = packed.export()
    for name in PACKED_MLP:
        stem = f"model.layers.0.mlp.{name}.weight"
        parts = (stem, stem + "_scale", stem + "_scale_2")
        assert_same_stored_tensors({part: np.asarray(exported[part]) for part in parts},
                                   {part: stored[part] for part in parts})


def test_a_modelopt_w4a16_load_without_its_values_holds_the_packed_tree_and_refuses_an_adapter(tmp_path):
    """A ModelOpt W4A16 source with its biased attention packed too (Qwen2's `attention_bias`, row 8550's
    review): its load without values (`weights=False`, a resume's) holds the tree a load of its values holds,
    every packed kernel's leaves at their shapes and dtypes; served, its attention's projections pack into
    no group (each member's kernel is packed, its bias a parameter), its logits finite; and an adapter over
    a packed kernel is refused by name."""
    from dew.lora import LoRA
    from dew.nn.projections import projection_groups

    source = _w4a16_qwen3(tmp_path / "source", attention=True)
    packed, shaped = (Pretrained.load(source, dtype="bfloat16", attention_impl="reference",
                                      packed_weights=True, weights=read) for read in (True, False))
    assert jax.tree.structure(shaped.variables) == jax.tree.structure(packed.variables)
    for held, shape in zip(jax.tree.leaves(packed.variables), jax.tree.leaves(shaped.variables), strict=True):
        assert (np.shape(held), np.dtype(held.dtype)) == (tuple(shape.shape), np.dtype(shape.dtype))
    assert projection_groups(packed.model, packed.variables) == ()
    ids = jnp.asarray(np.random.default_rng(3).integers(0, 96, (2, 8)), jnp.int32)
    assert np.isfinite(np.asarray(packed.model.apply(packed.variables, ids), np.float32)).all()
    with pytest.raises(ValueError, match="packed"):
        packed.adapt(LoRA(rank=2, modules=("q_proj",)), key=0)

