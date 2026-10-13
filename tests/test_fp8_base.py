"""A LoRA run's frozen base multiplied in FP8 (`LoRA.fp8_base`).

The sites a record names (`InputQuantization.frozen_fp8`) hold their frozen
kernels packed once, from the base the adapter freezes, as E4M3 rows for the
forward product and for the input gradient (`frozen_fp8_leaves`); each such
Dense computes its output from them inside the adapter's branch, its input's
rows cast by the same FP8 block cast (`dew.interop.codecs`). The product is
held to the float64 product of those exact operands, every site of a model
is checked to take it, and a run with an FP8 base resumes, scans and shards
as one with a bf16 base does. Where FP8 pays is a GPU's measurement
(`measured_fp8_sites`), recorded with the generation it was taken on.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn
from flax.traverse_util import flatten_dict
from model_support import TINY_DECODER
from reference_error import assert_leaves_as_exact

from dew.data.dataset import Loader
from dew.interop.codecs import dequantize_fp8_blocks, quantize_fp8_blocks
from dew.lora import LoRA
from dew.nn.kernels.generation import assumed, device_generation, gpu_target
from dew.objectives.base import FROZEN
from dew.objectives.lm import LMObjective
from dew.registry import models, wrapper_layers
from dew.training import Checkpoints, Layout, MeshSpec, Trainer
from dew.training.quantization import (
    FROZEN_FP8,
    QUANTIZATION,
    InputQuantization,
    InputQuantizer,
    frozen_fp8_leaves,
    frozen_fp8_product,
    measured_fp8_sites,
)

VOCAB, SEQ, BATCH = 64, 16, 8
FEATURES, MLP = 64, 128
FP8_SITES = ("q_proj", "gate_proj", "up_proj", "down_proj")


def decoder(**overrides):
    fields = {**TINY_DECODER, "vocab_size": VOCAB, "emb_features": FEATURES, "mlp_features": MLP,
              "dtype": "bfloat16"}
    return models.build("causal_transformer", **{**fields, **overrides})


def ids():
    return jnp.asarray(np.random.default_rng(0).integers(0, VOCAB, (BATCH, SEQ)), jnp.int32)


def record(variables, generation=None) -> InputQuantization:
    """The FP8 sites of the tiny decoder as a hand-written record, measured
    on this process's own backend, where it multiplies them."""
    kernels = flatten_dict(variables["params"], sep="/")
    return InputQuantization(
        frozen_fp8=tuple(path.removesuffix("/kernel") for path in kernels
                         if path.endswith("/kernel") and path.split("/")[-2] in FP8_SITES),
        measured_on=generation or device_generation())


def sites(adapter) -> tuple[str, ...]:
    """The frozen FP8 sites of the record `adapter`'s model computes under."""
    return adapter.model._dew_wrapped._dew_wrapper().frozen_fp8


def adapters(model=None):
    """One adapter over a bf16 base and one over the same base in FP8."""
    model = decoder() if model is None else model
    variables = model.init(jax.random.key(0), ids())
    spec = LoRA(rank=4, modules=("q_proj", "v_proj"))
    return (spec.apply(model, variables, key=1),
            spec.apply(model, variables, key=1, fp8_base=record(variables)))


def calls(model, variables, inputs):
    """Every Dense call of `model`'s forward: its path, the module unbound, its input and its output."""
    seen = {}

    def capture(next_fun, args, kwargs, context):
        output = next_fun(*args, **kwargs)
        if context.method_name == "__call__" and isinstance(context.module, nn.Dense):
            seen["/".join(context.module.path)] = (context.module.clone(parent=None), args[0], output)
        return output

    with nn.intercept_methods(capture):
        model.apply(variables, inputs)
    return seen


def operands(x, rows, scales):
    """`x`'s rows through the host FP8 block cast, and the packed rows, in float64."""
    codes, row_scales = quantize_fp8_blocks(np.asarray(x, np.float32), (1, x.shape[1]))
    packed = dequantize_fp8_blocks(np.asarray(rows), np.asarray(scales), (1, rows.shape[1]))
    cast = dequantize_fp8_blocks(codes, row_scales, (1, x.shape[1]))
    return cast.astype(np.float64), packed.astype(np.float64)


def test_the_frozen_product_multiplies_the_cast_rows_exactly_and_trains_no_kernel():
    """Forward and input gradient are the float64 products of the FP8 block
    cast's rows to float32's order of summation; neither the packed kernel
    nor its scales take a gradient."""
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.standard_normal((48, 96)) * np.exp2(rng.integers(-6, 6, (48, 1))), jnp.bfloat16)
    kernel = rng.standard_normal((96, 80)).astype(np.float32) * 0.05
    leaves = [jnp.asarray(value) for value in frozen_fp8_leaves(kernel).values()]
    cotangent = jnp.asarray(rng.standard_normal((48, 80)), jnp.float32)
    output, pullback = jax.vjp(frozen_fp8_product, x, *leaves)
    gradients = pullback(cotangent)
    forward, forward_scale, dgrad, dgrad_scale = leaves
    rows, packed = operands(x, forward, forward_scale)
    cotangent_rows, packed_dgrad = operands(cotangent, dgrad, dgrad_scale)
    truth = {"output": rows @ packed.T, "input": cotangent_rows @ packed_dgrad.T}
    dew = {"output": np.asarray(output), "input": np.asarray(gradients[0], np.float32)}
    assert_leaves_as_exact(dew["output"], truth["output"], truth["output"], reductions=96)
    # The input gradient rounds once more, to the input's bf16.
    assert_leaves_as_exact(
        {"input": dew["input"]}, {"input": np.asarray(jnp.asarray(truth["input"], jnp.bfloat16), np.float64)},
        {"input": truth["input"]}, reductions=80)
    assert gradients[0].dtype == x.dtype and all(gradient is None or not np.any(gradient)
                                                 for gradient in gradients[1:])


def test_every_named_site_computes_from_its_packed_kernel_inside_the_branch():
    """A fresh adapter's branch adds zero (B starts at 0), so each named
    site's output is the frozen product of its own input, the adapted ones
    included, and every other Dense computes as the bf16 base's does."""
    plain, fp8 = adapters()
    seen, base = calls(fp8.model, fp8.variables, ids()), calls(plain.model, plain.variables, ids())
    named = set(sites(fp8))
    assert named and named <= set(seen)
    packed = flatten_dict(fp8.variables[QUANTIZATION], sep="/")
    frozen = fp8.variables[FROZEN]
    for path, (module, x, output) in seen.items():
        if path in named:
            leaves = [packed[f"{path}/{leaf}"] for leaf in FROZEN_FP8]
            expected = frozen_fp8_product(x.reshape(-1, x.shape[-1]), *leaves).astype(output.dtype)
        else:
            own = frozen
            for name in path.split("/"):
                own = own[name]
            expected = module.apply({"params": own}, x)
        np.testing.assert_array_equal(np.asarray(output), np.asarray(expected).reshape(output.shape))
    assert set(base) == set(seen)


def fit(adapter, steps, directory=None, mesh=None, devices=1):
    objective = LMObjective(adapter.model, SEQ, variables=adapter.variables)
    batches = np.random.default_rng(2).integers(0, VOCAB, (8, BATCH, SEQ + 1)).astype(np.int32)

    class Stream:
        def __init__(self):
            self.index = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.index += 1
            return {"text": batches[(self.index - 1) % len(batches)]}

        def get_state(self):
            return str(self.index).encode()

        def set_state(self, state):
            self.index = int(state)

    mesh = mesh or MeshSpec()
    trainer = Trainer(objective, optax.adam(1e-2), key=jax.random.key(0), mesh=mesh,
                      layout=Layout(min_shard=1, tolerance=1.0),
                      checkpoints=None if directory is None else Checkpoints(str(directory)))
    trainer.device_mesh = mesh.build(jax.devices()[:devices])
    return trainer.fit(Loader(lambda partition: Stream(), None, None, BATCH), steps=steps,
                       checkpoint_every=None if directory is None else 2)


def test_lora_packs_the_frozen_base_once_and_training_moves_neither():
    """The packed leaves are the frozen kernels' (`frozen_fp8_leaves`), and a
    run trains the factors alone: the frozen kernels and their packing come
    out as they went in."""
    _, fp8 = adapters()
    frozen = flatten_dict(fp8.variables[FROZEN], sep="/")
    packed = flatten_dict(fp8.variables[QUANTIZATION], sep="/")
    for site in sites(fp8):
        for leaf, value in frozen_fp8_leaves(frozen[f"{site}/kernel"]).items():
            np.testing.assert_array_equal(np.asarray(packed[f"{site}/{leaf}"]).view(np.uint8),
                                          np.asarray(value).view(np.uint8))
    state = fit(fp8, 3)
    for tree, before in ((state.variables[FROZEN], frozen), (state.variables[QUANTIZATION], packed)):
        for path, value in flatten_dict(tree, sep="/").items():
            np.testing.assert_array_equal(np.asarray(value).view(np.uint8),
                                          np.asarray(before[path]).view(np.uint8))


def test_a_resumed_fp8_base_run_trains_what_an_uninterrupted_one_does(tmp_path):
    _, fp8 = adapters()
    whole = fit(fp8, 4)
    fit(fp8, 2, tmp_path)
    resumed = fit(fp8, 4, tmp_path)
    for got, want in zip(jax.tree.leaves(resumed.variables), jax.tree.leaves(whole.variables), strict=True):
        np.testing.assert_array_equal(np.asarray(got).view(np.uint8), np.asarray(want).view(np.uint8))


def test_the_scanned_stack_reads_each_layers_own_packed_kernel():
    _, fp8 = adapters()
    _, scanned = adapters(decoder(scan_layers=True))
    looped = np.asarray(fp8.model.apply(fp8.variables, ids()), np.float32)
    np.testing.assert_allclose(np.asarray(scanned.model.apply(fp8.variables, ids()), np.float32), looped,
                               rtol=1e-6, atol=1e-6)


@pytest.mark.mesh(devices=2)
def test_fsdp_trains_an_fp8_base_as_one_device_does():
    _, fp8 = adapters()
    split = fit(fp8, 2, mesh=MeshSpec(fsdp=2), devices=2)
    whole = fit(fp8, 2)
    split_params, whole_params = (jax.tree.leaves(state.variables["params"]) for state in (split, whole))
    for got, want in zip(split_params, whole_params, strict=True):
        np.testing.assert_allclose(np.asarray(got, np.float32), np.asarray(want, np.float32),
                                   rtol=1e-5, atol=1e-6)


def test_a_checkpoints_input_quantization_and_an_fp8_base_share_one_wrapper():
    """A model computing under its checkpoint's input quantization takes the
    FP8 base into the same record, so one interceptor serves both: the site
    rounds its input first, then multiplies in FP8."""
    model = decoder()
    variables = model.init(jax.random.key(0), ids())
    site = "layers_0/self_attn/q_proj"
    loaded = InputQuantization(sites=((site, "fp8"),)).apply(model)
    variables = {**variables, QUANTIZATION: {"layers_0": {"self_attn": {"q_proj": {
        "fp8_input_scale": jnp.asarray(0.01, jnp.float32)}}}}}
    # The record's fp8 input scale is ModelOpt's amax / 448.
    adapter = LoRA(rank=4, modules=("v_proj",)).apply(loaded, variables, key=1, fp8_base=record(variables))
    _, layers = wrapper_layers(type(adapter.model))
    wrappers = [layer._dew_wrapper() for layer in layers]
    joined = [wrapper for wrapper in wrappers if isinstance(wrapper, InputQuantization)]
    assert len(joined) == 1 and joined[0].sites == ((site, "fp8"),) and site in joined[0].frozen_fp8
    _, x, output = calls(adapter.model, adapter.variables, ids())[site]
    rounded = InputQuantizer.fp8(0.01).qdq(x)
    leaves = [flatten_dict(adapter.variables[QUANTIZATION], sep="/")[f"{site}/{leaf}"] for leaf in FROZEN_FP8]
    expected = frozen_fp8_product(rounded.reshape(-1, x.shape[-1]), *leaves).astype(output.dtype)
    np.testing.assert_array_equal(np.asarray(output), np.asarray(expected.reshape(output.shape)))


def test_a_record_runs_on_the_generation_it_was_measured_on_alone():
    _, fp8 = adapters()
    with assumed("sm80"), pytest.raises(ValueError, match=r"measured on \w+, and this is sm80"):
        jax.eval_shape(lambda: fp8.model.apply(fp8.variables, ids()))


FP8_CORES = gpu_target(device_generation()).fp8_peak is not None


@pytest.mark.skipif(FP8_CORES, reason="this GPU has FP8 tensor cores")
def test_a_device_without_fp8_tensor_cores_measures_no_site():
    variables = decoder().init(jax.random.key(0), ids())
    with pytest.raises(ValueError, match="no FP8 tensor cores"):
        measured_fp8_sites(variables, record(variables).frozen_fp8, 8192)


@pytest.mark.gpu(reason="FP8 tensor cores and cuBLASLt's FP8 GEMM")
@pytest.mark.skipif(not FP8_CORES, reason="this GPU has no FP8 tensor cores")
def test_a_wide_site_measures_faster_in_fp8_and_compiles_to_an_fp8_gemm():
    """On a GPU with FP8 tensor cores the wide projection measures faster in
    FP8 and the narrow one does not (H100: 1.47x and 0.70x), and the frozen
    product's forward and input gradient compile to cuBLASLt FP8 GEMMs."""
    variables = {"params": {"wide": {"kernel": jnp.zeros((4096, 12288), jnp.float32)},
                            "narrow": {"kernel": jnp.zeros((1024, 1024), jnp.float32)}}}
    measured = measured_fp8_sites(variables, ("wide", "narrow"), 8192)
    assert measured.frozen_fp8 == ("wide",) and measured.measured_on == device_generation()
    leaves = [jnp.asarray(value) for value in frozen_fp8_leaves(np.ones((256, 512), np.float32)).values()]
    text = jax.jit(jax.value_and_grad(lambda x: jnp.sum(frozen_fp8_product(x, *leaves)))).lower(
        jnp.ones((1024, 256), jnp.bfloat16)).compile().as_text()
    assert text.count("f8e4m3fn") and text.count("__cublas$lt$matmul$f8") == 2
