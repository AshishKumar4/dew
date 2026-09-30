"""Quantized training through Qwix: the value, the wrapping, the training.

Qwix-dependent tests skip without the package; the value's refusals run
everywhere, since construction never imports it.
"""

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew import models  # noqa: F401  registers the models
from dew.config import OptimConfig, _rebuild
from dew.nn.sharding import pipeline_microbatches
from dew.objectives.base import Step, scalar_loss
from dew.objectives.lm import LMObjective
from dew.training.distributed import Layout, MeshSpec, build_mesh, shard_batch
from dew.training.optim import build_optimizer
from dew.training.quantization import Quantization, apply_quantization, quantize_for_serving

VOCAB = 64
SEQ_LEN = 8
BATCH = 4
TINY = dict(vocab_size=VOCAB, emb_features=32, num_layers=2, num_heads=4,
            num_kv_heads=2, mlp_features=64, max_seq_len=16)


def tiny(**overrides):
    return models.build("causal_transformer", **{**TINY, **overrides})


def token_batch():
    rng = np.random.default_rng(0)
    return {"text": rng.integers(0, VOCAB, size=(BATCH, SEQ_LEN + 1)).astype(np.int32)}


def test_a_value_that_says_nothing_is_refused():
    with pytest.raises(ValueError, match="no patterns"):
        Quantization(patterns=())
    with pytest.raises(ValueError, match="does not compile"):
        Quantization(patterns=(".*(unclosed",))
    with pytest.raises(ValueError, match="calibration"):
        Quantization(calibration="percentile,99")
    with pytest.raises(ValueError, match="tile_size"):
        Quantization(tile_size=0)
    with pytest.raises(ValueError, match="bwd_qtype"):
        Quantization(bwd_qtype="int4")
    with pytest.raises(ValueError, match="bwd_stochastic_rounding"):
        Quantization(bwd_stochastic_rounding="gaussian")
    with pytest.raises(ValueError, match="int8 or fp8"):
        Quantization(dtype="int4")


def test_the_value_round_trips_through_json():
    spec = Quantization(dtype="fp8", patterns=(".*mlp.*", ".*self_attn.*"),
                        calibration="absmax,0.8", tile_size=32,
                        bwd_qtype="int8", bwd_stochastic_rounding="uniform")
    record = json.loads(json.dumps(dataclasses.asdict(spec)))
    assert _rebuild(Quantization, record) == spec


def quantized_forward(spec, **overrides):
    model = tiny(**overrides)
    variables = model.init(jax.random.key(0), jnp.ones((1, SEQ_LEN), jnp.int32))
    qmodel = apply_quantization(model, spec)
    ids = jnp.asarray(token_batch()["text"][:, :SEQ_LEN])
    return model, qmodel, variables, ids


def test_the_wrapped_forward_matches_qwixs_own_call():
    """Dew's wrapping against the provider built by hand: identical logits,
    and both away from the fp32 forward. The gap is the assertion: rules
    that never reached a matmul would leave the fp32 numerics bitwise.
    Observed on CPU: wrapped and hand-built bitwise equal, 1.2e-01 from
    fp32 on logits of order 3."""
    qwix = pytest.importorskip("qwix")
    model, qmodel, variables, ids = quantized_forward(Quantization())
    rules = [qwix.QtRule(module_path=".*", weight_qtype=jnp.int8,
                         act_qtype=jnp.int8)]
    reference = qwix.quantize_model(model, qwix.QtProvider(rules))
    plain = model.apply(variables, ids)
    wrapped = qmodel.apply(variables, ids)
    manual = reference.apply(variables, ids)
    assert float(jnp.max(jnp.abs(wrapped - manual))) == 0.0
    assert float(jnp.max(jnp.abs(wrapped - plain))) > 1e-2


def test_a_pattern_that_matches_nothing_quantizes_nothing():
    """A narrowed pattern is selective: a regex no module path matches runs
    the fp32 numerics bitwise. A pattern that matched everything would move
    the logits here, so the equality fails it. Observed on CPU: bitwise
    equal."""
    pytest.importorskip("qwix")
    model, qmodel, variables, ids = quantized_forward(
        Quantization(patterns=(".*does_not_exist.*",)))
    assert float(jnp.max(jnp.abs(
        qmodel.apply(variables, ids) - model.apply(variables, ids)))) == 0.0


def test_an_int8_trunk_trains_down():
    """Five adamw steps through the int8 trunk: each loss below the one
    before, which quantized gradients pointing the wrong way or drowned in
    rounding noise would not manage. Observed on CPU: 5.17 down to 4.07 in
    steps of about 0.25, against 1e-7 of int8 rounding."""
    pytest.importorskip("qwix")
    _, qmodel, variables, _ = quantized_forward(Quantization())
    objective = LMObjective(qmodel, SEQ_LEN)
    batch = token_batch()
    solver = build_optimizer(OptimConfig(learning_rate=1e-3), 5)
    params, opt_state = variables["params"], solver.init(variables["params"])

    @jax.jit
    def train(params, opt_state, key):
        (loss, _), grads = jax.value_and_grad(
            lambda p: scalar_loss(objective, {**variables, "params": p}, batch,
            Step(step=jnp.zeros((), jnp.int32), key=key, ema=None)),
            has_aux=True)(params)
        updates, opt_state = solver.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    losses = []
    key = jax.random.key(0)
    for _ in range(5):
        key, subkey = jax.random.split(key)
        params, opt_state, loss = train(params, opt_state, subkey)
        losses.append(float(loss))
    assert all(later < earlier for earlier, later in zip(losses, losses[1:])), losses


def test_a_scanned_quantized_stack_scores_as_the_plain_one():
    """Quantization composes with the scan. Both stacks compiled as a
    training step compiles them, the wrapped scan agrees with the wrapped
    plain loop while staying away from its fp32 twin; the distance is the
    assertion that the rules reached under the scan. Observed on logits of
    order 3: scan against plain 0.0 on CPU and 1.5e-02 on the RTX 4080,
    where the scan body and the unrolled layers lower to different fusions
    and int8 rounding flips under the reordered reductions; scan against
    fp32 1.2e-01 on both. (Eager the two differ by 3.6e-02, fusion order in
    the uncompiled matmuls, so both sides compile here.)"""
    pytest.importorskip("qwix")
    model, qmodel, variables, ids = quantized_forward(
        Quantization(), scan_layers=True)
    plain_wrapped = apply_quantization(tiny(), Quantization())
    scanned = jax.jit(qmodel.apply)(variables, ids)
    plain = jax.jit(plain_wrapped.apply)(variables, ids)
    reference = jax.jit(model.apply)(variables, ids)
    assert float(jnp.max(jnp.abs(scanned - reference))) > 1e-2
    assert float(jnp.max(jnp.abs(scanned - plain))) < 5e-2


@pytest.mark.skipif(jax.default_backend() == "gpu",
                    reason="Dew refuses a grouped quantized convolution on a GPU; "
                           "test_a_gpu_refuses_grouped_quantized_convolutions covers it")
@pytest.mark.parametrize("group_width", [1, 4])
def test_a_grouped_convolution_quantizes_each_group_on_its_own_range(group_width):
    """A grouped convolution never adds one group's inputs into another's
    outputs, so each group's activations take their own int8 scale. Over 64
    channels whose ranges span 1e-3 to 1, every output channel lands within
    int8 rounding of float (depthwise and 4 channels per group). One scale
    per example for all channels rounds the small groups to zero instead:
    observed on CPU before the fix, the worst channel 100% (depthwise) and
    107% (4 per group) off. Observed after: 1.0% and 1.2%."""
    pytest.importorskip("qwix")
    from dew.nn.conv import Conv

    features = 64
    conv = Conv(features=features, kernel_size=(3, 3), padding="SAME",
                feature_group_count=features // group_width, use_bias=False)
    ranges = jnp.logspace(-3, 0, features)
    x = jax.random.normal(jax.random.key(0), (2, 8, 8, features)) * ranges
    variables = conv.init(jax.random.key(1), x)
    plain = conv.apply(variables, x)
    quantized = apply_quantization(conv, Quantization()).apply(variables, x)
    error = (jnp.sqrt(jnp.sum((quantized - plain) ** 2, axis=(0, 1, 2)))
             / jnp.sqrt(jnp.sum(plain ** 2, axis=(0, 1, 2))))
    assert float(error.max()) < 0.02, np.asarray(error)
    assert float(error.min()) > 0.0


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="needs a GPU")
@pytest.mark.parametrize("dtype", ["int8", "fp8"])
@pytest.mark.parametrize("group_width", [1, 4])
def test_a_gpu_refuses_grouped_quantized_convolutions(group_width, dtype):
    """XLA:GPU computes a grouped convolution with int8 or fp8 activations
    wrongly or not at all, depending on the GPU, so quantizing one raises
    naming the ways around it, for training and for serving alike; weight-only
    quantization, one of them, leaves the convolution in float."""
    pytest.importorskip("qwix")
    from dew.nn.conv import Conv

    features = 64
    conv = Conv(features=features, kernel_size=(3, 3), padding="SAME",
                feature_group_count=features // group_width, use_bias=False)
    x = jax.random.normal(jax.random.key(0), (2, 8, 8, features))
    variables = conv.init(jax.random.key(1), x)
    with pytest.raises(ValueError, match="spatial_fusion"):
        apply_quantization(conv, Quantization(dtype=dtype)).apply(variables, x)
    with pytest.raises(ValueError, match="spatial_fusion"):
        quantize_for_serving(conv, variables, Quantization(dtype=dtype), x)
    served, served_variables = quantize_for_serving(conv, variables, Quantization(dtype=dtype, weight_only=True), x)
    np.testing.assert_array_equal(served.apply(served_variables, x), conv.apply(variables, x))


def test_serving_stores_int8_kernels_and_computes_what_training_quantized():
    """Served weights are int8 values with their scales, and the served
    projections reproduce the quantized-training forward they were trained
    under, away from fp32. Observed on CPU: served against wrapped 0.0 on
    logits of order 3, 1.2e-01 from fp32. (Qwix's serving and training
    providers quantize attention's matmuls of two activations differently:
    with those quantized too, the two forwards differ by 3e-02.)"""
    pytest.importorskip("qwix")
    spec = Quantization(patterns=(".*_proj",))
    model, qmodel, variables, ids = quantized_forward(spec)
    served, served_variables = quantize_for_serving(model, variables, spec, ids)
    dtypes = {leaf.dtype for leaf in jax.tree.leaves(served_variables["params"])}
    assert jnp.dtype(jnp.int8) in dtypes
    logits = served.apply(served_variables, ids)
    assert float(jnp.max(jnp.abs(logits - qmodel.apply(variables, ids)))) < 1e-4
    assert float(jnp.max(jnp.abs(logits - model.apply(variables, ids)))) > 1e-2


@pytest.mark.parametrize("weight_only", [False, True])
def test_a_served_language_model_generates_with_its_quantized_kernels(weight_only):
    """Generation enters a language model through methods besides
    `__call__` (`states_and_logits_at` for the prefill, `states_and_logits`
    for each step), and they compute with the stored quantized kernels as
    `__call__` does. Before, they ran outside Qwix's interception and a
    stored kernel reached a Dense raw (`TypeError: expected number, got
    WithAux`)."""
    pytest.importorskip("qwix")
    from dew.sampling import Sampling, generate

    model = tiny()
    prompt = token_batch()["text"][:, :4]
    variables = model.init(jax.random.key(0), prompt)
    served, served_variables = quantize_for_serving(model, variables, Quantization(weight_only=weight_only), prompt)
    logits = served.apply(served_variables, prompt)
    _, stepped = served.apply(served_variables, prompt, method="states_and_logits")
    np.testing.assert_array_equal(stepped, logits)
    assert float(jnp.max(jnp.abs(logits - model.apply(variables, prompt)))) > 1e-3
    tokens = generate(served, served_variables, prompt, 3, seed=0, sampling=Sampling(temperature=0)).host().tokens
    assert tokens.shape == (BATCH, 7)


@pytest.mark.parametrize("weight_only", [False, True])
def test_a_served_bf16_module_computes_in_bf16(weight_only):
    """A quantized kernel follows the module's compute dtype as its float
    kernel would: a bf16 Dense stays bf16 through its matmul. Before, the
    fp32 scales of the stored kernel promoted it to fp32."""
    pytest.importorskip("qwix")
    from flax import linen as nn

    dense = nn.Dense(16, dtype=jnp.bfloat16)
    x = jax.random.normal(jax.random.key(0), (4, 32), jnp.bfloat16)
    variables = dense.init(jax.random.key(1), x)
    served, served_variables = quantize_for_serving(dense, variables, Quantization(weight_only=weight_only), x)
    assert served.apply(served_variables, x).dtype == jnp.bfloat16


def test_a_bf16_int8_matmul_scales_its_int32_products_in_float32():
    """A served bf16 Dense multiplies its int8 matmul's int32 accumulator by
    both scales in float32 and rounds once to bf16. Qwix 0.1.8 rounded the
    accumulator to bf16 before the scales multiplied in, and XLA:TPU then
    compiled the int8 matmuls of the bf16 176M text-to-image model with bf16
    results, which sampled NaN images on a v6e."""
    pytest.importorskip("qwix")
    from flax import linen as nn
    from qwix._src.core import qarray

    dense = nn.Dense(64, use_bias=False, dtype=jnp.bfloat16)
    x = jax.random.normal(jax.random.key(0), (8, 256), jnp.bfloat16)
    variables = dense.init(jax.random.key(1), x)
    served, served_variables = quantize_for_serving(dense, variables, Quantization(), x)
    kernel = served_variables["params"]["kernel"].array.astype(jnp.bfloat16)
    activations = qarray.quantize(x, qarray.HowToQuantize(qtype=jnp.int8, channelwise_axes=(0,)))
    products = jax.lax.dot(activations.qvalue, kernel.qvalue, preferred_element_type=jnp.int32)
    scaled = products * activations.scale.astype(jnp.float32) * kernel.scale.astype(jnp.float32)
    np.testing.assert_array_equal(served.apply(served_variables, x), scaled.astype(jnp.bfloat16))


def test_serving_leaves_complex_matmuls_in_float():
    """Qwix quantizes real values only; a complex matmul, like the S5 scan's
    in the hybrid DiT, runs unquantized beside the quantized Dense, within
    int8 rounding of float. Observed on CPU: 4e-03 relative."""
    pytest.importorskip("qwix")
    from flax import linen as nn

    class Rotated(nn.Module):
        @nn.compact
        def __call__(self, x):
            h = nn.Dense(8)(x).astype(jnp.complex64)
            rotation = jnp.exp(1j * jnp.arange(64.0).reshape(8, 8))
            mixed = jnp.einsum("bf,fg->bg", h, rotation)
            return jnp.real(jax.lax.dot_general(mixed, rotation, (((1,), (0,)), ((), ()))))

    model = Rotated()
    x = jax.random.normal(jax.random.key(0), (4, 16))
    variables = model.init(jax.random.key(1), x)
    served, served_variables = quantize_for_serving(model, variables, Quantization(), x)
    plain = model.apply(variables, x)
    error = jnp.linalg.norm(served.apply(served_variables, x) - plain) / jnp.linalg.norm(plain)
    assert 0.0 < float(error) < 0.02


@pytest.mark.mesh
def test_a_quantized_pipeline_has_finite_loss_and_gradients():
    """The int8 trunk over two stages on the eight simulated devices: finite
    loss and gradients, and the loss within int8 rounding of the plain
    wrapped loop's, the way the bf16 pipeline test holds its policy's
    rounding. Observed on CPU: 1.6e-03 on a loss of order 5."""
    pytest.importorskip("qwix")
    model = tiny(num_layers=4)
    variables = model.init(jax.random.key(0), jnp.ones((1, SEQ_LEN), jnp.int32))
    qmodel = apply_quantization(model, Quantization())
    rows = np.random.default_rng(0).integers(0, VOCAB, size=(8, SEQ_LEN + 1)).astype(np.int32)
    batch = {"text": rows}

    def run(spec):
        objective = LMObjective(qmodel, SEQ_LEN)
        mesh = build_mesh(spec)
        placed = jax.device_put(
            variables, Layout(min_shard=256).shardings(mesh, variables))
        placed_batch = shard_batch(mesh, batch)
        info = Step(step=jnp.zeros((), jnp.int32), key=jax.random.key(3),
                    ema=None)

        def loss(params):
            return scalar_loss(objective, {**variables, "params": params},
                                  placed_batch, info)

        with jax.set_mesh(mesh), pipeline_microbatches(spec.microbatches):
            (value, _), grads = jax.jit(
                jax.value_and_grad(loss, has_aux=True))(placed["params"])
        return float(value), jax.tree.map(np.asarray, grads)

    loss, grads = run(MeshSpec(fsdp=8))
    piped, piped_grads = run(MeshSpec(fsdp=4, stage=2, microbatches=2))
    assert np.isfinite(loss) and np.isfinite(piped)
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(piped_grads))
    assert abs(loss - piped) < 1e-2 * max(1.0, abs(loss)), (loss, piped)


def test_stochastic_rounding_draws_from_its_own_stream():
    """Test rounding without the decoder embedding scatter's GPU reduction order."""
    pytest.importorskip("qwix")
    from flax import linen as nn

    model = nn.Dense(16)
    values = jax.random.normal(jax.random.key(3), (32, 8))
    variables = model.init(jax.random.key(2), values)
    quantized = apply_quantization(
        model, Quantization(bwd_qtype="int8", bwd_stochastic_rounding="uniform"))

    def loss(params, key):
        result = quantized.apply({"params": params}, values,
                                 rngs={"stochastic_rounding": key})
        return jnp.mean(result ** 2)

    differentiate = jax.jit(jax.grad(loss))
    gradients = differentiate(variables["params"], jax.random.key(0))
    repeated = differentiate(variables["params"], jax.random.key(0))
    other = differentiate(variables["params"], jax.random.key(1))
    for first, second in zip(jax.tree.leaves(gradients), jax.tree.leaves(repeated), strict=True):
        np.testing.assert_array_equal(first, second)
    assert max(float(jnp.max(jnp.abs(first - second)))
               for first, second in zip(jax.tree.leaves(gradients),
                                        jax.tree.leaves(other), strict=True)) > 0.0


# --------------------------------------------------------------------------
# The trainer's knob
# --------------------------------------------------------------------------

RES = 8
RUN_BATCH = 8
"""A whole-run batch: one record per simulated device, so the default mesh
splits it evenly."""


def image_batches(batch):
    def stream():
        rng = np.random.RandomState(0)
        while True:
            yield {"image": rng.randint(0, 256, (batch, RES, RES, 3), np.uint8)}
    return stream


def diffusion_run(directory, batch=RUN_BATCH, **trainer):
    """The smallest unconditional diffusion run: a tiny DiT over 8-pixel
    images, one step, nothing written but the record."""
    from dew.config import ModelConfig, TrainerConfig
    from dew.data import OxfordFlowers
    from dew.objectives.diffusion import DiffusionRunConfig

    return DiffusionRunConfig(
        model=ModelConfig("simple_dit", {"patch_size": 4, "emb_features": 16,
                                         "num_layers": 1, "num_heads": 2}, dtype="float32"),
        data=OxfordFlowers(image_size=RES), text=None, guidance=None,
        sampling_steps=2, val_metrics=(),
        trainer=TrainerConfig(name="quantized", checkpoint_dir=str(directory), batch_size=batch,
                              steps=1, eval_every=None, checkpoint_every=None,
                              compilation_cache_dir=None, **trainer))


def test_the_trainer_knob_quantizes_the_objective_a_run_trains(tmp_path):
    """`--trainer.quantization` is the one place a run names quantization:
    `RunConfig.train` wraps the module the objective trains before anything
    initialises it, and the run steps on the quantized forward. The distance
    from the fp32 forward on the trained weights is the assertion that the
    rules reached the matmuls. Observed on CPU: 2.7e-05 on activations of
    order 1e-2, against 0.0 for a run trained without the knob."""
    pytest.importorskip("qwix")
    from dew.data import Dataset

    batch = RUN_BATCH
    config = diffusion_run(tmp_path, batch, quantization=Quantization())
    objective = config.build()
    plain = config.build()

    state = config.train(objective, Dataset(lambda partition: image_batches(batch)(), None, None, batch),
                         name="quantized")

    assert int(state.step) == 1
    image = jnp.ones((1, RES, RES, 3), jnp.float32)
    noise_level = jnp.ones((1,), jnp.float32)
    quantized_out = objective.model.apply(state.params, image, noise_level)
    plain_out = plain.model.apply(state.params, image, noise_level)
    assert float(jnp.max(jnp.abs(quantized_out - plain_out))) > 0.0


def test_an_objective_that_trains_no_single_model_is_refused(tmp_path):
    """The wrap needs a module to wrap; an objective that keeps none is
    refused by name, before the run writes anything."""
    from dew.data import Dataset
    from dew.objectives.base import Aux, Objective

    class Modelless(Objective):
        def init(self, key, variables=None):
            return {"params": {}}

        def loss(self, params, batch, step):
            return jnp.zeros(()), Aux({})

    config = diffusion_run(tmp_path, quantization=Quantization())
    with pytest.raises(ValueError, match="Modelless keeps no `model`"):
        config.train(Modelless(), Dataset(lambda partition: image_batches(RUN_BATCH)(), None, None, RUN_BATCH),
                     name="quantized")
    assert not (tmp_path / "quantized").exists()
