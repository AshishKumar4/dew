"""Routed experts held in an MXFP4 checkpoint's own bytes and decoded on the device.

`Pretrained.load(..., expert_storage='mxfp4')` keeps GPT OSS's and Kimi K3's
experts as uint8 E2M1 codes beside E8M0 exponents (`dew.nn.moe.MXFP4Experts`),
and the expert projection decodes them with `decode_e2m1_device`. MXFP4 values
are exact in bfloat16 and float32, so each check here is bit for bit: the
decode against the host codec's, every expert projection against the same
projection of the decoded matrices, and the loaded models' logits and greedy
tokens against the same checkpoint loaded with its experts decoded.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from model_support import flat_tree
from reference_error import ieee_fixture
from test_hf_decoders import fixture_config

from dew.interop import Pretrained
from dew.interop.codecs import (
    MXFP4,
    PACKED_MXFP4,
    decode_e2m1,
    decode_e2m1_device,
    dequantize_mxfp4,
    quantize_mxfp4,
    quantize_packed_mxfp4,
)
from dew.nn.gpt_oss import GptOssMLP
from dew.nn.inputs import ModelInputs
from dew.nn.moe import MXFP4Experts, SparseMLP, expert_projection
from dew.sampling import Sampling, generate
from dew.training import Layout, MeshSpec

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"
GPT_OSS = FIXTURES / "gpt-oss-tiny"
KIMI_K3 = FIXTURES / "kimi-k3-tiny"


def assert_bitwise(actual, expected) -> None:
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape, (actual.dtype, actual.shape)
    mismatched = np.flatnonzero(actual.reshape(-1).view(np.uint8).reshape(actual.size, -1)
                                != expected.reshape(-1).view(np.uint8).reshape(expected.size, -1))
    assert not mismatched.size, f"{mismatched.size} bytes differ, the first in element {mismatched[0]}"


def mxfp4_pair(kernel) -> tuple[dict[str, jax.Array], jax.Array]:
    """A float `[exp, in, out]` kernel through GPT OSS's encoder: its MXFP4
    parts, and the bf16 matrices they decode to on the host."""
    blocks, scales = quantize_mxfp4(np.asarray(kernel, np.float32))
    experts, outputs = scales.shape[:2]
    parts = {"codes": jnp.asarray(blocks.reshape(experts, outputs, -1)), "exponents": jnp.asarray(scales)}
    return parts, jnp.asarray(dequantize_mxfp4(blocks, scales), jnp.bfloat16)


# --------------------------------------------------------------------------
# The decode: decode_e2m1_device against the host codec
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dtype, bits", [(jnp.bfloat16, np.uint16), (jnp.float32, np.uint32)])
def test_the_device_decode_is_the_host_decode_bit_for_bit(dtype, bits):
    """Every code byte under every exponent byte: the subnormals of bytes 0
    and 1 (0.5 * 2 ** -127 is 2 ** -128), the products past float32's range
    under 253 and 254 (infinities), and byte 255's NaN, which XLA's CPU
    backend would flush or round were the values multiplied."""
    codes = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
    exponents = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 16, axis=1)
    with np.errstate(over="ignore"):
        expected = decode_e2m1(codes, exponents).astype(dtype)
    actual = np.asarray(jax.jit(decode_e2m1_device, static_argnums=2)(codes, exponents, dtype))
    assert np.isnan(expected[255]).all() and np.isinf(expected[254]).any() and (expected[0] != 0).any()
    np.testing.assert_array_equal(actual.view(bits), expected.view(bits))


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_both_layouts_decode_on_the_device_as_their_codecs_decode_them(dtype):
    """GPT OSS's `[E, out, groups, 16]` blocks and compressed-tensors'
    `[out, in / 2]` pairs, held as the loader keeps them
    (`SourceQuantization.stored`): the device's matrices are the codec's."""
    rng = np.random.default_rng(0)
    blocks = rng.integers(0, 256, (3, 40, 2, 16), dtype=np.uint8)
    scales = rng.integers(100, 140, (3, 40, 2), dtype=np.uint8)
    gpt_oss = {"experts_blocks": blocks, "experts_scales": scales}
    stored = MXFP4.stored(gpt_oss, "experts")
    assert stored.shape == (3, 64, 40) and stored.codes.shape == (3, 40, 32)
    held = MXFP4Experts(jnp.asarray(stored.codes), jnp.asarray(stored.exponents))
    assert held.shape == stored.shape
    assert_bitwise(held.decoded(dtype), MXFP4.decode(gpt_oss, "experts").astype(dtype))

    packed, exponents = quantize_packed_mxfp4(rng.standard_normal((24, 96)).astype(np.float32))
    linear = {"w.weight_packed": packed, "w.weight_scale": exponents}
    stored = PACKED_MXFP4.stored(linear, "w.weight")
    assert stored.shape == (24, 96)
    assert_bitwise(decode_e2m1_device(jnp.asarray(stored.codes), jnp.asarray(stored.exponents), dtype),
                   PACKED_MXFP4.decode(linear, "w.weight").astype(dtype))


# --------------------------------------------------------------------------
# The projection: MXFP4 experts against their decoded matrices
# --------------------------------------------------------------------------

@pytest.mark.parametrize("implementation", ["xla", "pallas"])
def test_mxfp4_experts_project_what_their_decoded_matrices_project(implementation):
    """24 rows over 8 experts, two of them idle, 96 inputs (a whole and a
    half 64-input tile, so the kernel masks a partial tile of groups) to 40
    outputs. 'xla' decodes every expert for `jax.lax.ragged_dot`; 'pallas',
    interpreted on the CPU, decodes each tile in the grouped matmul. Either
    gives the decoded matrices' rows bit for bit, and an input's gradient
    through them to within bf16's rounding of it."""
    rng = np.random.default_rng(1)
    parts, matrices = mxfp4_pair(rng.standard_normal((8, 96, 40)) / np.sqrt(96))
    held = MXFP4Experts(parts["codes"], parts["exponents"])
    sizes = jnp.asarray([5, 0, 3, 7, 0, 2, 4, 3], jnp.int32)
    x = jnp.asarray(rng.standard_normal((24, 96)), jnp.bfloat16)
    probe = jnp.asarray(rng.standard_normal((24, 40)), jnp.float32)

    def project(kernel, x):
        return jnp.asarray(expert_projection(x, kernel, sizes, jnp.bfloat16, implementation, None))

    def gradient(kernel):
        return jax.grad(lambda x: jnp.sum(project(kernel, x).astype(jnp.float32) * probe))(x)

    assert_bitwise(jax.jit(project)(held, x), jax.jit(project)(matrices, x))
    np.testing.assert_allclose(np.asarray(gradient(held), np.float32),
                               np.asarray(gradient(matrices), np.float32), rtol=1e-2, atol=1e-6)
    assert float(jnp.max(jnp.abs(gradient(held).astype(jnp.float32)))) > 0


def held_and_decoded(variables):
    """A layer's variables twice, every expert matrix through MXFP4: once as
    its parts, once as the bf16 matrices they decode to; every other leaf bf16."""
    held, decoded = {}, {}
    for name, value in variables.items():
        if isinstance(value, dict):
            held[name], decoded[name] = held_and_decoded(value)
        elif value.ndim == 3:
            held[name], decoded[name] = mxfp4_pair(value)
        else:
            held[name] = decoded[name] = value.astype(jnp.bfloat16)
    return held, decoded


@pytest.mark.mesh
@pytest.mark.parametrize("dispatch", ["exchange", "global"])
@pytest.mark.parametrize("experts", ["gpt_oss", "mixture"])
def test_mxfp4_experts_dispatch_over_a_mesh_as_their_decoded_matrices(experts, dispatch):
    """GPT OSS's fused experts and `SparseMLP`'s, split over an expert axis of
    4 and an fsdp axis of 2: each part of an MXFP4 matrix is placed by its
    matrix's expert and output axes (`moe.mxfp4_axes`), and the layer's
    output is the decoded layer's, bit for bit."""
    width, hidden = 64, 96
    if experts == "gpt_oss":
        layers = [GptOssMLP(width, hidden, 8, 2, dispatch=dispatch, expert_storage=storage)
                  for storage in ("mxfp4", "float")]
    else:
        layers = [SparseMLP(num_experts=8, top_k=2, hidden_features=hidden, out_features=width,
                            dispatch=dispatch, expert_storage=storage) for storage in ("mxfp4", "float")]
    x = jax.random.normal(jax.random.key(0), (4, 16, width), jnp.float32)
    held, decoded = held_and_decoded(layers[1].init(jax.random.key(1), x)["params"])
    shapes = jax.eval_shape(layers[0].init, jax.random.key(1), x)["params"]
    assert jax.tree.map(np.shape, held) == jax.tree.map(np.shape, shapes)
    with jax.set_mesh(MeshSpec(expert=4, fsdp=2).build()), nn.logical_axis_rules(()):
        outputs = [jax.jit(layer.apply)({"params": params}, x)
                   for layer, params in zip(layers, (held, decoded), strict=True)]
    assert_bitwise(*outputs)


# --------------------------------------------------------------------------
# The loaded models: GPT OSS and Kimi K3 against their decoded loads
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gpt_oss_mxfp4(tmp_path_factory) -> Path:
    """gpt-oss-tiny as the release ships it: uint8 blocks and scales in place
    of each expert matrix, under `quant_method: mxfp4`."""
    from safetensors.numpy import load_file, save_file

    packed = {}
    for name, tensor in load_file(str(GPT_OSS / "model.safetensors")).items():
        if name.endswith(("gate_up_proj", "down_proj")):
            packed[name + "_blocks"], packed[name + "_scales"] = quantize_mxfp4(tensor)
        else:
            packed[name] = tensor
    directory = tmp_path_factory.mktemp("gpt-oss-mxfp4")
    save_file(packed, str(directory / "model.safetensors"))
    (directory / "config.json").write_text(json.dumps(
        {**fixture_config("gpt-oss-tiny"), "quantization_config": {"quant_method": "mxfp4"}}))
    return directory


def greedy(loaded: Pretrained, inputs) -> tuple[np.ndarray, np.ndarray]:
    generated = generate(loaded.model, loaded.variables, inputs, 4, key=jax.random.key(1),
                         sampling=Sampling(temperature=0))
    return np.asarray(generated.tokens), np.asarray(generated.raw_log_probs)


@pytest.mark.parametrize("dtype, param_dtype", [("bfloat16", "bfloat16"), ("float32", "auto")])
def test_gpt_oss_held_in_mxfp4_gives_the_decoded_logits_and_tokens(gpt_oss_mxfp4, dtype, param_dtype):
    """The same packed checkpoint loaded twice, its experts decoded on the
    host and held as its bytes: the logits over the fixture's ids and four
    greedy steps, tokens and log probabilities, are the same bits. The held
    parts are the checkpoint's own bytes, and the held load exports what the
    decoded load exports."""
    from safetensors.numpy import load_file

    loads = [Pretrained.load(str(gpt_oss_mxfp4), dtype=dtype, param_dtype=param_dtype, attention_impl="xla",
                             expert_storage=storage) for storage in ("float", "mxfp4")]
    flat = flat_tree(loads[1].variables["params"])
    shipped = load_file(str(gpt_oss_mxfp4 / "model.safetensors"))
    blocks = shipped["model.layers.1.mlp.experts.down_proj_blocks"]
    assert_bitwise(flat["layers_1.mlp.experts.down_proj.codes"], blocks.reshape(*blocks.shape[:2], -1))
    assert_bitwise(flat["layers_1.mlp.experts.down_proj.exponents"],
                   shipped["model.layers.1.mlp.experts.down_proj_scales"])
    assert loads[1].model.mixture.expert_storage == "mxfp4"

    ids = jnp.asarray(np.load(GPT_OSS / "input_ids.npy"), jnp.int32)
    logits = [jax.jit(loaded.model.apply)(loaded.variables, ids) for loaded in loads]
    assert_bitwise(*logits)
    (tokens, log_probs), (held_tokens, held_log_probs) = (greedy(loaded, ids) for loaded in loads)
    assert_bitwise(held_tokens, tokens)
    assert_bitwise(held_log_probs, log_probs)
    exported, held_export = (loaded.export() for loaded in loads)
    assert set(held_export) == set(exported)
    for name, tensor in exported.items():
        assert_bitwise(held_export[name], tensor)


@pytest.mark.mesh
def test_gpt_oss_held_in_mxfp4_loads_onto_a_mesh_and_gives_the_decoded_logits(gpt_oss_mxfp4):
    """Placed one shard at a time over expert 4 x fsdp 2 and run under that
    mesh, the held load's logits are the decoded load's."""
    mesh = MeshSpec(expert=4, fsdp=2)
    # Every leaf the mesh divides is split, as tests/test_streaming_load.py splits them.
    loads = [Pretrained.load(str(gpt_oss_mxfp4), dtype="bfloat16", param_dtype="auto", attention_impl="xla",
                             mesh=mesh, layout=Layout(min_shard=1, tolerance=1.0), expert_storage=storage)
             for storage in ("float", "mxfp4")]
    codes = flat_tree(loads[1].variables["params"])["layers_0.mlp.experts.gate_up_proj.codes"]
    assert codes.dtype == np.uint8 and codes.sharding.shard_shape(codes.shape)[0] == 1
    ids = jnp.asarray(np.load(GPT_OSS / "input_ids.npy"), jnp.int32)
    with jax.set_mesh(mesh.build()):
        logits = [jax.jit(loaded.model.apply)(loaded.variables, ids) for loaded in loads]
    assert_bitwise(*logits)


def kimi_k3_inputs() -> ModelInputs:
    reference = ieee_fixture(KIMI_K3 / "reference.npz")
    return ModelInputs(jnp.asarray(reference["input_ids"], jnp.int32),
                       {"attention_mask": jnp.asarray(reference["attention_mask"], bool)})


def kimi_k3_logits(loaded: Pretrained, inputs: ModelInputs) -> jax.Array:
    return jax.jit(lambda variables: loaded.model.apply(variables, inputs.tokens, **inputs.kwargs()))(
        loaded.variables)


def test_kimi_k3_held_in_mxfp4_gives_the_decoded_logits_tokens_and_export():
    """K3's compressed-tensors experts, one `[out, in / 2]` pair per expert,
    stack into each layer's parts. Logits over the fixture's padded rows and
    four greedy steps are the decoded load's bits, and the export writes the
    tensors the decoded load writes."""
    inputs = kimi_k3_inputs()
    loads = [Pretrained.load(KIMI_K3, dtype="float32", attention_impl="reference", expert_storage=storage)
             for storage in ("float", "mxfp4")]
    codes = flat_tree(loads[1].variables["params"])["layers_1.mlp.experts.gate_proj.kernel.codes"]
    assert codes.dtype == np.uint8 and codes.shape[0] == loads[1].model.mixture.experts

    assert_bitwise(*(kimi_k3_logits(loaded, inputs) for loaded in loads))
    (tokens, log_probs), (held_tokens, held_log_probs) = (greedy(loaded, inputs) for loaded in loads)
    assert_bitwise(held_tokens, tokens)
    assert_bitwise(held_log_probs, log_probs)
    exported, held_export = (loaded.export() for loaded in loads)
    assert set(held_export) == set(exported)
    for name, tensor in exported.items():
        assert_bitwise(held_export[name], tensor)


def test_mxfp4_weights_outside_the_routed_experts_decode_on_the_host(tmp_path):
    """K3 with its latent up projection, a Linear beside the routed experts,
    also shipped as an MXFP4 pair: the held load keeps the routed experts'
    bytes and decodes the up projection as the default load does, and the
    two loads' logits are the same bits."""
    from shutil import copytree

    from safetensors.numpy import load_file, save_file

    directory = copytree(KIMI_K3, tmp_path / "kimi-k3")
    tensors = load_file(str(directory / "model.safetensors"))
    stem = "language_model.model.layers.1.block_sparse_moe.routed_expert_up_proj"
    tensors[stem + ".weight_packed"], tensors[stem + ".weight_scale"] = quantize_packed_mxfp4(
        tensors.pop(stem + ".weight"))
    save_file(tensors, str(directory / "model.safetensors"))
    loads = [Pretrained.load(directory, dtype="float32", attention_impl="reference", expert_storage=storage)
             for storage in ("float", "mxfp4")]
    flat = flat_tree(loads[1].variables["params"])
    assert flat["layers_1.mlp.routed_expert_up_proj.kernel"].dtype == np.float32
    assert flat["layers_1.mlp.experts.up_proj.kernel.codes"].dtype == np.uint8
    inputs = kimi_k3_inputs()
    assert_bitwise(*(kimi_k3_logits(loaded, inputs) for loaded in loads))


def test_dew_pipeline_holds_mxfp4_experts_where_the_fused_kernel_was_measured(gpt_oss_mxfp4, monkeypatch):
    """`dew.pipeline`'s default: the checkpoint's MXFP4 bytes on a generation
    where `KERNELS['mxfp4_grouped_matmul']` measured the fused kernel, the
    decoded experts elsewhere. Here the CPU stands for both; with no GPU the
    held experts run XLA, which `ran_kernel` logs, and the two tasks' logits
    are the same bits."""
    import dew
    from dew.nn.kernels import KERNELS

    tasks = []
    for measured in ({}, {"cpu": "pallas"}):
        monkeypatch.setitem(KERNELS, "mxfp4_grouped_matmul", measured)
        tasks.append(dew.pipeline(str(gpt_oss_mxfp4), dtype="bfloat16"))
    assert [task.model.mixture.expert_storage for task in tasks] == ["float", "mxfp4"]
    ids = jnp.asarray(np.load(GPT_OSS / "input_ids.npy"), jnp.int32)
    assert_bitwise(*(jax.jit(task.model.apply)(task.variables, ids) for task in tasks))


def test_a_trainer_refuses_experts_held_in_mxfp4(gpt_oss_mxfp4):
    """Held experts serve: an optimizer would move their uint8 codes as
    numbers, so the trainer refuses them where it first reads the tree, and
    the same checkpoint loaded with float experts builds its state."""
    import optax

    from dew.objectives.lm import LMObjective
    from dew.training import Trainer

    def state(storage: str):
        loaded = Pretrained.load(str(gpt_oss_mxfp4), dtype="float32", attention_impl="xla",
                                 expert_storage=storage)
        trainer = Trainer(LMObjective(loaded, 7, ema_decay=None, pad_id=0), optax.adamw(1e-3),
                          key=jax.random.key(0))
        return jax.eval_shape(trainer.initial_state)

    with pytest.raises(ValueError, match=r"layers_0/mlp/experts/gate_up_proj is held as the checkpoint's "
                                         r"bytes .* load with expert_storage='float' to train"):
        state("mxfp4")
    experts = state("float").variables["params"]["layers_0"]["mlp"]["experts"]
    assert experts["gate_up_proj"].dtype == np.float32


def test_a_source_without_mxfp4_experts_is_refused():
    with pytest.raises(ValueError, match="ships none"):
        Pretrained.load(str(GPT_OSS), dtype="float32", attention_impl="xla", expert_storage="mxfp4")
