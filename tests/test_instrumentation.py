"""Throughput accounting, divergence detection and the profiler hook.

None of the performance work is evaluable without these numbers, so they get
the same treatment as the training maths.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn
from jax.sharding import NamedSharding, PartitionSpec as P

from dew.inputs import DiffusionInputConfig
from dew.nn.backbones.dit import SimpleDiT
from dew.diffusion.transforms import get_diffusion_preset
from dew.objectives.lm import LMObjective
from dew.registry import build_model
from dew.training import ObjectiveTrainer
from dew.training.distributed import DevicePrefetchIterator
from dew.telemetry.instrumentation import (
    compiled_flops, enable_compilation_cache, hlo_flops, model_flops_utilization,
    step_flops,
)

RES = 8
BATCH = 8


def make_trainer(tmp_path, **kwargs):
    train_schedule, _, transform = get_diffusion_preset("edm")
    return ObjectiveTrainer(
        model=SimpleDiT(patch_size=4, emb_features=16, num_layers=1, num_heads=2, mlp_ratio=1),
        optimizer=optax.adam(1e-3),
        noise_schedule=train_schedule,
        model_output_transform=transform,
        input_config=DiffusionInputConfig(
            sample_data_key="image", sample_data_shape=(RES, RES, 3), conditions=[]),
        rngs=jax.random.PRNGKey(0),
        name="instr",
        wandb_config=None,
        distributed_training=False,
        checkpoint_base_path=str(tmp_path),
        **kwargs,
    )


def batches():
    images = np.tile(np.linspace(0, 255, RES, dtype=np.float32)[None, :, None, None],
                     (BATCH, 1, RES, 3))
    while True:
        yield {"image": images}


def data_dict():
    return {"train": batches, "train_len": BATCH * 8,
            "local_batch_size": BATCH, "global_batch_size": BATCH}


def token_batches(batch, seq, vocab):
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, vocab, size=(batch, seq + 1)).astype(np.int32)
    while True:
        yield {"text": tokens}


def lm_trainer(tmp_path, *, vocab, width, layers, heads, ratio, batch, seq):
    """The language-model trainer, on one device so the step is the whole batch."""
    config = {"vocab_size": vocab, "emb_features": width, "num_layers": layers,
              "num_heads": heads, "mlp_ratio": ratio, "max_seq_len": seq}
    model = build_model('causal_transformer', config)
    trainer = ObjectiveTrainer(
        model=model,
        optimizer=optax.adam(1e-3),
        input_config=None,
        objective=LMObjective(model, seq, vocab_size=vocab),
        rngs=jax.random.PRNGKey(0),
        name="instr-lm",
        wandb_config=None,
        distributed_training=False,
        checkpoint_base_path=str(tmp_path),
    )
    trainer.global_batch_size = batch
    return trainer


def hlo_module(*instructions: str) -> str:
    """One entry computation holding the given instruction lines."""
    body = "\n".join(f"  {line}" for line in instructions)
    return f"HloModule counted\n\nENTRY %main (p: f32[]) -> f32[] {{\n{body}\n}}\n"


# Recorded from this repo's own GPU: a tied vocabulary projection of a
# causal_transformer step, a 3x3 UNet convolution in each of the three kinds
# cuDNN is given, and fused attention forward and backward. Operand shapes are
# the ones the calls were emitted with, which is where their arithmetic is.
CUBLAS_MATMUL = hlo_module(
    '%lhs = bf16[768,2304]{1,0} parameter(0)',
    '%rhs = bf16[8192,768]{1,0} parameter(1)',
    'ROOT %matmul = (bf16[2304,8192]{0,1}, s8[4194304]{0}) custom-call(%lhs, %rhs), '
    'custom_call_target="__cublas$lt$matmul", '
    'backend_config={"gemm_backend_config":{"alpha_real":1,"beta":0,'
    '"dot_dimension_numbers":{"lhs_batch_dimensions":[],'
    '"lhs_contracting_dimensions":["0"],"rhs_batch_dimensions":[],'
    '"rhs_contracting_dimensions":["1"]},"epilogue":"DEFAULT"}}',
)
CUDNN_CONV_FORWARD = hlo_module(
    '%images = bf16[4,32,32,64]{3,2,1,0} parameter(0)',
    '%kernel = bf16[32,3,3,64]{3,2,1,0} parameter(1)',
    '%bias = bf16[32]{0} parameter(2)',
    'ROOT %conv = (bf16[4,32,32,32]{3,2,1,0}, u8[0]{0}) '
    'custom-call(%images, %kernel, %bias), window={size=3x3 pad=1_1x1_1}, '
    'dim_labels=b01f_o01i->b01f, '
    'custom_call_target="__cudnn$convBiasActivationForward"',
)
CUDNN_CONV_BACKWARD_INPUT = hlo_module(
    '%grad = bf16[4,32,32,32]{3,2,1,0} parameter(0)',
    '%kernel = bf16[32,3,3,64]{3,2,1,0} parameter(1)',
    'ROOT %conv = (bf16[4,32,32,64]{3,2,1,0}, u8[0]{0}) custom-call(%grad, %kernel), '
    'window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f, '
    'custom_call_target="__cudnn$convBackwardInput"',
)
CUDNN_CONV_BACKWARD_FILTER = hlo_module(
    '%images = bf16[4,32,32,64]{3,2,1,0} parameter(0)',
    '%grad = bf16[4,32,32,32]{3,2,1,0} parameter(1)',
    'ROOT %conv = (bf16[32,3,3,64]{3,2,1,0}, u8[0]{0}) custom-call(%images, %grad), '
    'window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f, '
    'custom_call_target="__cudnn$convBackwardFilter"',
)


def attention_module(target, query, key, results):
    """A fused-attention call over `[batch, sequence, heads, width]` operands."""
    return hlo_module(
        f'%query = bf16{query} parameter(0)',
        f'%key = bf16{key} parameter(1)',
        f'%value = bf16{key} parameter(2)',
        f'ROOT %attention = ({results}) custom-call(%query, %key, %value), '
        f'custom_call_target="{target}"',
    )


CUDNN_ATTENTION_FORWARD = attention_module(
    '__cudnn$fmhaSoftmax', '[3,128,8,64]{3,2,1,0}', '[3,128,8,64]{3,2,1,0}',
    'bf16[3,8,128,64]{3,1,2,0}, f32[3,8,128]{2,1,0}, u8[0]{0}')
CUDNN_ATTENTION_BACKWARD = attention_module(
    '__cudnn$fmhaSoftmaxBackward', '[3,128,8,64]{3,2,1,0}', '[3,128,8,64]{3,2,1,0}',
    'bf16[3,8,128,64]{3,1,2,0}, bf16[3,8,128,64]{3,1,2,0}, '
    'bf16[3,8,128,64]{3,1,2,0}, u8[1585280]{0}')
CUDNN_ATTENTION_GROUPED = attention_module(
    '__cudnn$fmhaSoftmax', '[3,128,8,64]{3,2,1,0}', '[3,128,2,64]{3,2,1,0}',
    'bf16[3,8,128,64]{3,1,2,0}, f32[3,8,128]{2,1,0}, u8[0]{0}')
CUDNN_ATTENTION_CROSS = attention_module(
    '__cudnn$fmhaSoftmax', '[3,128,8,64]{3,2,1,0}', '[3,64,8,64]{3,2,1,0}',
    'bf16[3,8,128,64]{3,1,2,0}, f32[3,8,128]{2,1,0}, u8[0]{0}')

# A loop XLA states no trip count for: the body holds a matmul, so counting it
# once would be a number that belongs to no run.
UNBOUNDED_LOOP = """HloModule unbounded

%body (carry: (f32[8,32], f32[32,32])) -> (f32[8,32], f32[32,32]) {
  %carry = (f32[8,32]{1,0}, f32[32,32]{1,0}) parameter(0)
  %values = f32[8,32]{1,0} get-tuple-element(%carry), index=0
  %kernel = f32[32,32]{1,0} get-tuple-element(%carry), index=1
  %product = f32[8,32]{1,0} dot(%values, %kernel), lhs_contracting_dims={1}, \
rhs_contracting_dims={0}
  ROOT %next = (f32[8,32]{1,0}, f32[32,32]{1,0}) tuple(%product, %kernel)
}

%condition (carry.1: (f32[8,32], f32[32,32])) -> pred[] {
  %carry.1 = (f32[8,32]{1,0}, f32[32,32]{1,0}) parameter(0)
  ROOT %again = pred[] constant(true)
}

ENTRY %main (start: (f32[8,32], f32[32,32])) -> (f32[8,32], f32[32,32]) {
  %start = (f32[8,32]{1,0}, f32[32,32]{1,0}) parameter(0)
  ROOT %loop = (f32[8,32]{1,0}, f32[32,32]{1,0}) while(%start), \
condition=%condition, body=%body
}
"""


def test_step_flops_reports_a_positive_count(tmp_path):
    trainer = make_trainer(tmp_path)
    step = trainer._define_train_step(batch_size=BATCH)
    source = DevicePrefetchIterator(batches(), trainer.batch_sharding)
    flops = step_flops(step, trainer.state, trainer.rngstate, next(source))
    assert flops is not None and flops > 0


def test_throughput_metrics_are_consistent(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.global_batch_size = 64
    metrics = trainer._throughput_metrics(elapsed=2.0, steps=10)
    assert metrics["train/step_time_ms"] == pytest.approx(200.0)
    assert metrics["train/samples_per_sec"] == pytest.approx(320.0)


def test_throughput_metrics_ignore_a_zero_interval(tmp_path):
    assert make_trainer(tmp_path)._throughput_metrics(elapsed=0.0, steps=0) == {}


def test_mfu_is_skipped_on_unknown_hardware():
    # CPU is deliberately absent from the peak-FLOPs table
    assert model_flops_utilization(1e12, 1.0) is None


def test_mfu_uses_the_per_device_flop_count(monkeypatch):
    from dew.telemetry import instrumentation
    monkeypatch.setitem(instrumentation.PEAK_FLOPS_PER_DEVICE,
                        jax.devices()[0].device_kind, 100.0)
    assert instrumentation.model_flops_utilization(50.0, 1.0) == pytest.approx(0.5)
    assert instrumentation.model_flops_utilization(50.0, 2.0) == pytest.approx(0.25)


def test_compiled_flops_is_per_device_under_spmd():
    devices = jax.devices()
    mesh = jax.make_mesh((len(devices),), ("data",), devices=devices)
    split = NamedSharding(mesh, P("data"))
    replicated = NamedSharding(mesh, P())
    batch, width = len(devices), 32
    x = jax.device_put(jnp.ones((batch, width)), split)
    weight = jax.device_put(jnp.ones((width, width)), replicated)
    executable = jax.jit(
        lambda values, kernel: values @ kernel,
        in_shardings=(split, replicated), out_shardings=split,
    ).lower(x, weight).compile()

    whole_batch_flops = 2 * batch * width * width
    assert compiled_flops(executable) == pytest.approx(
        whole_batch_flops / len(devices))


# --------------------------------------------------------------------------
# What the FLOP count counts
# --------------------------------------------------------------------------

def test_compiled_flops_counts_a_convolution():
    """A 3x3 convolution costs one multiply-add per output element, kernel tap
    and input feature: 2 B Ho Wo Co Kh Kw Ci. XLA's own cost analysis reports
    less, because it drops the taps that land on the padding."""
    batch, size, in_features, out_features, kernel = 4, 16, 8, 16, 3
    model = nn.Conv(out_features, (kernel, kernel), use_bias=False)
    images = jnp.ones((batch, size, size, in_features))
    params = model.init(jax.random.PRNGKey(0), images)
    executable = jax.jit(model.apply).lower(params, images).compile()

    analytic = (2 * batch * size * size * out_features
                * kernel * kernel * in_features)
    assert compiled_flops(executable) == pytest.approx(analytic, rel=0.01)


def test_compiled_flops_counts_a_matmul_and_its_gradient():
    """2 M K N for the forward matmul, and the same again for the one weight
    gradient a loss over its output needs."""
    rows, contracted, columns = 128, 64, 256
    values = jnp.ones((rows, contracted))
    kernel = jnp.ones((contracted, columns))
    forward = jax.jit(lambda a, b: a @ b).lower(values, kernel).compile()
    with_gradient = jax.jit(jax.grad(lambda b, a: jnp.sum((a @ b) ** 2))).lower(
        kernel, values).compile()

    analytic = 2 * rows * contracted * columns
    assert compiled_flops(forward) == pytest.approx(analytic, rel=0.01)
    assert compiled_flops(with_gradient) == pytest.approx(2 * analytic, rel=0.01)


def test_compiled_flops_matches_the_transformer_flop_formula(tmp_path):
    """The whole language-model step against the closed form for it.

    With B the batch, S the sequence, T = B S tokens, d the width, f the MLP
    width, L the layers, V the vocabulary, H heads of width D:

        N_matmul = d V + L (4 d^2 + 3 d f)
        F_step   = 6 T N_matmul + 12 L B S^2 H D

    Six FLOPs per parameter per token is the forward matmul plus its two
    gradients; the attention term is the two score matmuls forward and their
    four backward products. The vocabulary head is tied, so it is one d by V
    matrix. Nothing elementwise is in here: no optimizer, no EMA, no softmax,
    no loss reduction (docs/research/benchmark-parity.md:44-61).
    """
    width, layers, heads, vocab, ratio = 128, 2, 4, 512, 4
    batch, seq = 4, 64
    head_dim, hidden = width // heads, ratio * width
    trainer = lm_trainer(tmp_path, vocab=vocab, width=width, layers=layers,
                         heads=heads, ratio=ratio, batch=batch, seq=seq)
    step = trainer._define_train_step(batch_size=batch)
    source = DevicePrefetchIterator(token_batches(batch, seq, vocab),
                                    trainer.batch_sharding)
    executable = trainer._compiled_step(
        step, trainer.state, trainer.rngstate, next(source))

    per_layer = 4 * width * width + 3 * width * hidden
    matmuls = width * vocab + layers * per_layer
    analytic = (6 * batch * seq * matmuls
                + 12 * layers * batch * seq * seq * heads * head_dim)
    assert compiled_flops(executable) == pytest.approx(analytic, rel=0.05)


def test_compiled_flops_counts_every_iteration_of_a_scanned_body():
    """A scanned stack runs its body once per iteration. The count has to be
    the run's, not the body's: cost analysis reports the body's."""
    rows, width, steps = 8, 32, 6

    def scanned(kernel, values):
        return jax.lax.scan(lambda carry, _: (carry @ kernel, None),
                            values, None, length=steps)[0]

    executable = jax.jit(scanned).lower(
        jnp.ones((width, width)), jnp.ones((rows, width))).compile()
    assert compiled_flops(executable) == pytest.approx(
        steps * 2 * rows * width * width)


def test_no_flops_are_reported_for_a_loop_of_unknown_length():
    """A body counted once when it runs an unknown number of times is a wrong
    number, so nothing is reported and the run logs no utilisation."""
    assert hlo_flops(UNBOUNDED_LOOP) is None


def test_gpu_matmul_custom_calls_are_counted_from_their_shapes():
    """cuBLAS and cuDNN calls carry their arithmetic in their operands, and a
    GPU backend hands most of a step to them. These are the calls XLA emitted
    on this repo's own GPU for a UNet step, a language-model step and
    `jax.nn.dot_product_attention`; the arithmetic each one stands for is the
    ordinary formula for that operation.
    """
    tokens, width, vocab_shard = 2304, 768, 8192
    assert hlo_flops(CUBLAS_MATMUL) == pytest.approx(
        2 * tokens * vocab_shard * width)

    batch, size, in_features, out_features, kernel = 4, 32, 64, 32, 3
    convolution = (2 * batch * size * size * out_features
                   * kernel * kernel * in_features)
    # The three kinds cost the same multiply-adds as the forward shape.
    assert hlo_flops(CUDNN_CONV_FORWARD) == pytest.approx(convolution)
    assert hlo_flops(CUDNN_CONV_BACKWARD_INPUT) == pytest.approx(convolution)
    assert hlo_flops(CUDNN_CONV_BACKWARD_FILTER) == pytest.approx(convolution)

    batch, heads, seq, head_dim = 3, 8, 128, 64
    scores = batch * heads * seq * seq * head_dim
    # Q K^T and P V forward, then dV, dP, dQ and dK backward: 12 per layer,
    # which is the attention term of the transformer formula above.
    assert hlo_flops(CUDNN_ATTENTION_FORWARD) == pytest.approx(4 * scores)
    assert hlo_flops(CUDNN_ATTENTION_BACKWARD) == pytest.approx(8 * scores)
    # Grouped-query heads share their keys, so the query heads set the cost.
    assert hlo_flops(CUDNN_ATTENTION_GROUPED) == pytest.approx(4 * scores)
    # Cross attention counts its own key length.
    assert hlo_flops(CUDNN_ATTENTION_CROSS) == pytest.approx(
        4 * batch * heads * seq * (seq // 2) * head_dim)

def test_epoch_loss_accumulates_bfloat16_losses_in_float32(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path, log_every=1000)

    def step(state, rng_state, batch):
        del batch
        loss = jnp.array(1.5, jnp.bfloat16)
        return state, loss, {}, rng_state, jnp.array(True)

    monkeypatch.setattr(trainer, "_compiled_step", lambda *_: step)
    steps = 400
    epoch_loss, *_ = trainer.train_loop(
        trainer.state, object(), iter([None] * steps), steps, 0, trainer.rngstate)

    assert epoch_loss.dtype == jnp.float32
    assert float(epoch_loss / steps) == pytest.approx(1.5)


def test_fit_reports_throughput_to_wandb(tmp_path):
    """The logging tick must actually carry the numbers, not just the loss."""
    logged = []

    class FakeWandb:
        def log(self, payload, step=None):
            logged.append(payload)

        def define_metric(self, *args, **kwargs):
            pass

    trainer = make_trainer(tmp_path, log_every=1)
    trainer.wandb = FakeWandb()
    trainer.fit(data_dict(), training_steps_per_epoch=3, epochs=1, val_steps_per_epoch=0)

    ticks = [p for p in logged if "train/samples_per_sec" in p]
    assert ticks, "no throughput was logged"
    assert all(p["train/step_time_ms"] > 0 for p in ticks)
    assert all(p["train/samples_per_sec"] > 0 for p in ticks)


def test_compilation_cache_directory_is_configured(tmp_path):
    path = str(tmp_path / "xla-cache")
    enable_compilation_cache(path)
    assert os.path.isdir(path)
    assert jax.config.jax_compilation_cache_dir == path


def test_profiler_writes_a_trace(tmp_path, monkeypatch):
    """The window has to open after the warmup: a trace that starts at step 0
    is mostly compilation, and reports its occupancy instead of the loop's."""
    trainer = make_trainer(tmp_path, profile_steps=2, profile_warmup_steps=2)
    started_at = []
    real_start = jax.profiler.start_trace

    def spy(*args, **kwargs):
        # The train state is replaced after every step, so its counter is the
        # number of steps that have run by the time the trace opens.
        started_at.append(int(trainer.state.step))
        return real_start(*args, **kwargs)

    monkeypatch.setattr(jax.profiler, "start_trace", spy)
    trainer.fit(data_dict(), training_steps_per_epoch=5, epochs=1, val_steps_per_epoch=0)

    assert started_at == [2], "the trace did not open at the configured step"
    assert os.path.isdir(trainer.profile_path())
    assert any(files for _, _, files in os.walk(trainer.profile_path()))


def test_an_unfinished_profile_window_is_still_closed(tmp_path):
    """A window wider than the epoch has to close anyway: a trace left running
    takes the next one down with it."""
    trainer = make_trainer(tmp_path / "long", profile_steps=8, profile_warmup_steps=1)
    trainer.fit(data_dict(), training_steps_per_epoch=3, epochs=1, val_steps_per_epoch=0)
    assert any(files for _, _, files in os.walk(trainer.profile_path()))

    second = make_trainer(tmp_path / "short", profile_steps=1, profile_warmup_steps=0)
    second.fit(data_dict(), training_steps_per_epoch=2, epochs=1, val_steps_per_epoch=0)
    assert any(files for _, _, files in os.walk(second.profile_path()))
def test_profiler_runs_only_once_across_epochs(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path, profile_steps=1, profile_warmup_steps=0)
    starts = []
    stops = []
    monkeypatch.setattr(jax.profiler, "start_trace", lambda *a, **k: starts.append(1))
    monkeypatch.setattr(jax.profiler, "stop_trace", lambda: stops.append(1))

    trainer.fit(data_dict(), training_steps_per_epoch=1, epochs=3,
                val_steps_per_epoch=0)

    assert len(starts) == 1
    assert len(stops) == 1


def test_profiler_warmup_can_cross_an_epoch_boundary(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path, profile_steps=1, profile_warmup_steps=2)
    started_at = []
    monkeypatch.setattr(
        jax.profiler, "start_trace",
        lambda *a, **k: started_at.append(int(trainer.state.step)))
    monkeypatch.setattr(jax.profiler, "stop_trace", lambda: None)

    trainer.fit(data_dict(), training_steps_per_epoch=1, epochs=3,
                val_steps_per_epoch=0)

    assert started_at == [2]


def test_the_training_step_is_compiled_once_per_run(tmp_path, monkeypatch):
    """Reading the cost analysis used to compile the step a second time, which
    doubled the startup cost of every fit()."""
    trainer = make_trainer(tmp_path)
    compiles = []
    real_compile = jax.stages.Lowered.compile

    def counting_compile(lowered, *args, **kwargs):
        compiles.append(lowered)
        return real_compile(lowered, *args, **kwargs)

    monkeypatch.setattr(jax.stages.Lowered, "compile", counting_compile)

    jitted = []
    real_define = trainer._define_train_step

    def capture(**kwargs):
        step = real_define(**kwargs)
        jitted.append(step)
        return step

    monkeypatch.setattr(trainer, "_define_train_step", capture)
    trainer.fit(data_dict(), training_steps_per_epoch=3, epochs=2, val_steps_per_epoch=0)

    assert len(compiles) == 1, "the training step was compiled more than once"
    # Both epochs ran on that one executable; a jit call would have compiled
    # its own and left it in the jit cache.
    assert jitted[0]._cache_size() == 0, "the loop went through the jit path too"
    assert trainer.flops_per_step and trainer.flops_per_step > 0


# --------------------------------------------------------------------------
# Divergence
# --------------------------------------------------------------------------

def test_sustained_non_finite_loss_stops_the_run(tmp_path):
    trainer = make_trainer(
        tmp_path, log_every=1, max_bad_loss_steps=3,
        loss_fn=lambda pred, target: jnp.full_like(pred, jnp.nan))
    with pytest.raises(RuntimeError, match="non-finite"):
        trainer.fit(data_dict(), training_steps_per_epoch=8, epochs=1, val_steps_per_epoch=0)


def test_healthy_run_does_not_trip_the_detector(tmp_path):
    trainer = make_trainer(tmp_path, log_every=1, max_bad_loss_steps=3)
    trainer.fit(data_dict(), training_steps_per_epoch=6, epochs=1, val_steps_per_epoch=0)
    assert int(trainer.state.step) == 6
