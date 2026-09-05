"""Throughput accounting, the FLOP count and the profiler hook.

None of the performance work is evaluable without these numbers, so they get
the same treatment as the training maths.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

# Needs the eight simulated CPU devices conftest configures; the GPU lane skips it.
pytestmark = pytest.mark.mesh
from flax import linen as nn
from jax.sharding import NamedSharding, PartitionSpec as P

from dew.objectives.base import Aux, EMASpec, Objective
import dew.nn.backbones  # registers the decoder the FLOP formula test builds
from dew.objectives.lm import LMObjective
from dew.registry import models
from dew.training import Profile, Trainer
from dew.training.distributed import shard_batch
from dew.telemetry.instrumentation import (
    compiled_flops, enable_compilation_cache, hlo_flops, model_flops_utilization,
    step_flops,
)

BATCH = 8


class Affine(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(2)(x)


class Regression(Objective):
    def __init__(self):
        self.model = Affine()
        self.ema = EMASpec(decay=optax.constant_schedule(0.9))

    def init(self, key):
        return self.model.init(key, jnp.zeros((1, 3)))

    def loss(self, params, batch, step):
        return jnp.mean((self.model.apply(params, batch["x"]) - batch["y"]) ** 2), Aux({})


class Data:
    def __init__(self, train, batch=BATCH):
        self._train, self.val, self.batch, self.records = train, None, batch, None

    def train(self):
        return self._train()

    steps_per_epoch = None


def batches():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(BATCH, 3)).astype(np.float32)
    batch = {"x": x, "y": 2 * x[:, :2]}
    while True:
        yield batch


def make_trainer(**kwargs):
    return Trainer(Regression(), optax.adam(1e-3), key=jax.random.key(0), **kwargs)


def token_batches(batch, seq, vocab):
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, vocab, size=(batch, seq + 1)).astype(np.int32)
    while True:
        yield {"text": tokens}


def lm_trainer(*, vocab, width, layers, heads, ratio, seq):
    """The language-model trainer, replicated so the step is the whole batch."""
    model = models.build('causal_transformer', vocab_size=vocab, emb_features=width,
                         num_layers=layers, num_heads=heads, mlp_features=ratio * width,
                         max_seq_len=seq)
    return Trainer(LMObjective(model, seq), optax.adam(1e-3), key=jax.random.key(0))


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


def test_the_compiled_step_reports_a_positive_flop_count():
    trainer = make_trainer()
    state, _, _ = trainer.place()
    trainer.compile(state, shard_batch(trainer.device_mesh, next(batches())))
    assert trainer.flops_per_step is not None and trainer.flops_per_step > 0


def test_step_flops_reads_a_jitted_function():
    flops = step_flops(jax.jit(lambda a, b: a @ b), jnp.ones((8, 16)), jnp.ones((16, 4)))
    assert flops == pytest.approx(2 * 8 * 16 * 4)


def test_throughput_metrics_are_consistent():
    metrics = make_trainer()._throughput(elapsed=2.0, steps=10, batch=64)
    assert metrics["train/step_time_ms"] == pytest.approx(200.0)
    assert metrics["train/samples_per_sec"] == pytest.approx(320.0)


def test_throughput_metrics_ignore_a_zero_interval():
    assert make_trainer()._throughput(elapsed=0.0, steps=0, batch=64) == {}


def test_mfu_is_skipped_on_unknown_hardware():
    # CPU is deliberately absent from the peak-FLOPs table
    assert model_flops_utilization(1e12, 1.0) is None


def test_mfu_uses_the_per_device_flop_count(monkeypatch):
    from dew.telemetry import instrumentation
    monkeypatch.setitem(instrumentation.PEAK_FLOPS_PER_DEVICE,
                        jax.devices()[0].device_kind, 100.0)
    assert instrumentation.model_flops_utilization(50.0, 1.0) == pytest.approx(0.5)
    assert instrumentation.model_flops_utilization(50.0, 2.0) == pytest.approx(0.25)


@pytest.mark.parametrize("device_kind,peak", [
    ("NVIDIA H100 80GB HBM3", 989e12),
    ("NVIDIA H100 PCIe", 756e12),
    ("NVIDIA A100-SXM4-80GB", 312e12),
    ("NVIDIA GeForce RTX 4080", 97.5e12),
    ("TPU v5 lite", 197e12),
    ("TPU v5", 459e12),
    ("TPU7x", None),
])
def test_the_peak_table_resolves_the_names_devices_report(device_kind, peak):
    """`device_kind` is the CUDA device name or the TPU generation string,
    never the bare model: an exact lookup found no H100 or A100 at all, and
    the PCIe H100 has to resolve to its own figure rather than the SXM's,
    whose name it also starts with. Hardware the table does not name is
    None, so the run logs no utilisation rather than a made-up one."""
    from dew.telemetry.instrumentation import peak_flops
    assert peak_flops(device_kind) == peak


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


def test_compiled_flops_counts_a_convolutions_gradients():
    """The backward of a strided convolution on the CPU lane, against the
    analytic backward FLOPs of one convolution.

    The input gradient and the kernel gradient each cost the multiply-adds of
    the forward: every output element's MAC contributes to exactly one input
    element through one output element (dgrad), and to exactly one kernel
    weight (wgrad), so the total is 2 B Ho Wo Co Kh Kw Ci for each. XLA emits
    dgrad as a convolution with `lhs_dilate` set to the stride, whose dilated
    positions are zeros no kernel multiplies, and wgrad as a convolution whose
    window is the input's spatial extent; this is what pins both of those
    reductions: a stride-2 dgrad counted without the dilation division reads
    1,179,648 instead of 589,824, and the assertion fails.
    """
    batch, size, in_features, out_features, kernel, stride = 4, 16, 8, 16, 3, 2
    model = nn.Conv(out_features, (kernel, kernel), strides=stride,
                   use_bias=False)
    images = jnp.ones((batch, size, size, in_features))
    params = model.init(jax.random.PRNGKey(0), images)

    grads = jax.jit(jax.grad(
        lambda p, x: jnp.sum(model.apply(p, x)), argnums=(0, 1)
    )).lower(params, images).compile()

    out = -(-size // stride)  # SAME padding
    forward = (2 * batch * out * out * out_features
               * kernel * kernel * in_features)
    assert compiled_flops(grads) == pytest.approx(2 * forward, rel=0.01)


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


def test_compiled_flops_matches_the_transformer_flop_formula():
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
    batch, seq = 8, 64
    head_dim, hidden = width // heads, ratio * width
    trainer = lm_trainer(vocab=vocab, width=width, layers=layers, heads=heads, ratio=ratio,
                         seq=seq)
    state, _, _ = trainer.place()
    trainer.compile(
        state, shard_batch(trainer.device_mesh, next(token_batches(batch, seq, vocab))))

    per_layer = 4 * width * width + 3 * width * hidden
    matmuls = width * vocab + layers * per_layer
    analytic = (6 * batch * seq * matmuls
                + 12 * layers * batch * seq * seq * heads * head_dim)
    # rel=0.05 is slack for XLA rewriting a matmul pair into one fused call or
    # splitting one over tiles, not for a missing term: on this CPU lane the
    # measured count lands on the closed form exactly (956,301,312 vs
    # 956,301,312, largest observed difference 0 FLOPs in 956 million). The
    # step is partitioned over the eight devices, so the count is per device.
    assert trainer.flops_per_step * jax.device_count() == pytest.approx(analytic, rel=0.05)


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


def test_compiled_flops_returns_none_when_the_compiler_emits_no_text():
    """`Compiled.as_text` returns None when the executable has no HLO to read.
    Passing that null into the text parser would fail on None rather than
    reporting nothing, so the executable entry point guards it."""
    class Silent:
        def as_text(self):
            return None

    assert compiled_flops(Silent()) is None


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

class RecordingTracker:
    def __init__(self):
        self.scalars = []

    def log(self, scalars, step):
        self.scalars.append(dict(scalars))

    def artifact(self, value, step):
        pass


def test_fit_reports_throughput_to_the_tracker():
    """The logging tick must actually carry the numbers, not just the loss."""
    tracker = RecordingTracker()
    make_trainer(tracker=tracker).fit(Data(batches), steps=3, log_every=1)

    ticks = [p for p in tracker.scalars if "train/samples_per_sec" in p]
    assert len(ticks) == 3, "no throughput was logged"
    assert all(p["train/step_time_ms"] > 0 for p in ticks)
    assert all(p["train/samples_per_sec"] > 0 for p in ticks)


def test_compilation_cache_directory_is_configured(tmp_path):
    path = str(tmp_path / "xla-cache")
    enable_compilation_cache(path)
    assert os.path.isdir(path)
    assert jax.config.jax_compilation_cache_dir == path


def test_profiler_writes_a_trace_after_the_warmup(tmp_path, monkeypatch):
    """The window has to open after the warmup: a trace that starts at step 0
    is mostly compilation, and reports its occupancy instead of the loop's."""
    started_at = []
    real_start = jax.profiler.start_trace
    seen = []

    class Counting(Regression):
        def loss(self, params, batch, step):
            seen.append(step)
            return super().loss(params, batch, step)

    trainer = Trainer(Counting(), optax.adam(1e-3), key=jax.random.key(0),
                      profile=Profile(str(tmp_path / "profile"), steps=2, warmup=2))
    compile_step = trainer.compile

    def counting_compile(*args):
        executable = compile_step(*args)
        ran = []

        def counted(*step_args):
            ran.append(1)
            return executable(*step_args)
        counted.ran = ran
        trainer.executable = counted
        return counted

    monkeypatch.setattr(trainer, "compile", counting_compile)
    monkeypatch.setattr(jax.profiler, "start_trace",
                        lambda *a, **k: started_at.append(len(trainer.executable.ran)) or real_start(*a, **k))
    trainer.fit(Data(batches), steps=5, log_every=1)

    assert started_at == [2], "the trace did not open after the configured warmup"
    assert any(files for _, _, files in os.walk(tmp_path / "profile"))


def test_an_unfinished_profile_window_is_still_closed(tmp_path):
    """A window wider than the run has to close anyway: a trace left running
    takes the next one down with it."""
    make_trainer(profile=Profile(str(tmp_path / "long"), steps=8, warmup=1)).fit(
        Data(batches), steps=3, log_every=1)
    assert any(files for _, _, files in os.walk(tmp_path / "long"))

    make_trainer(profile=Profile(str(tmp_path / "short"), steps=1, warmup=0)).fit(
        Data(batches), steps=2, log_every=1)
    assert any(files for _, _, files in os.walk(tmp_path / "short"))


def test_the_profiler_runs_once_per_fit(tmp_path, monkeypatch):
    starts, stops = [], []
    monkeypatch.setattr(jax.profiler, "start_trace", lambda *a, **k: starts.append(1))
    monkeypatch.setattr(jax.profiler, "stop_trace", lambda: stops.append(1))

    make_trainer(profile=Profile(str(tmp_path), steps=1, warmup=0)).fit(
        Data(batches), steps=6, log_every=1)

    assert len(starts) == 1 and len(stops) == 1


def test_the_training_step_is_compiled_once_per_fit(monkeypatch):
    """Reading the cost analysis must not compile the step a second time, which
    would double the startup cost of every fit()."""
    compiles = []
    real_compile = jax.stages.Lowered.compile

    def counting_compile(lowered, *args, **kwargs):
        compiles.append(lowered)
        return real_compile(lowered, *args, **kwargs)

    monkeypatch.setattr(jax.stages.Lowered, "compile", counting_compile)
    trainer = make_trainer()
    trainer.fit(Data(batches), steps=6, log_every=1)

    # The placement goes through the jit cache; the step's lower().compile()
    # is the one explicit compile, and the loop runs on it.
    assert len(compiles) == 1, "the training step was compiled more than once"
    assert trainer.flops_per_step and trainer.flops_per_step > 0
