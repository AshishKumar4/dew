"""Throughput accounting, the FLOP count and the profiler hook.

None of the performance work is evaluable without these numbers, so they get
the same treatment as the training maths.
"""

import importlib.metadata
import itertools
import os
import threading
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from affine_run import BATCH, FEATURES, TimingRegression as Regression, batches as batches
from flax import linen as nn
from jax.sharding import NamedSharding, PartitionSpec as P
from recording import RecordingTracker

from dew.cache import enable_compilation_cache
from dew.data.dataset import Loader
from dew.objectives.lm import LMObjective
from dew.registry import models
from dew.telemetry.instrumentation import compiled_flops
from dew.training import Trainer
from dew.training.distributed import shard_batch

# Needs the eight simulated CPU devices conftest configures; the GPU lane skips it.
pytestmark = pytest.mark.mesh


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


def test_a_remats_recomputed_forward_counts_once_in_the_models_flops():
    """A block under `jax.checkpoint` runs its forward again in the backward:
    the hardware FLOPs count both runs, and the model's count the step as it
    would run without the remat, matmul for matmul."""
    from dew.telemetry.instrumentation import step_flops

    def loss(weights, values, block):
        hidden = values
        for weight in weights:
            hidden = block(hidden, weight)
        return jnp.sum(hidden ** 2)

    def layer(hidden, weight):
        return jnp.tanh(hidden @ weight)

    weights = [jnp.full((64, 64), 0.01) for _ in range(3)]
    values = jnp.ones((32, 64))
    plain = jax.jit(jax.grad(lambda w, x: loss(w, x, layer))).lower(weights, values).compile()
    rematted = jax.jit(jax.grad(lambda w, x: loss(w, x, jax.checkpoint(layer)))).lower(
        weights, values).compile()

    once, twice = step_flops(plain), step_flops(rematted)
    assert once.model == once.hardware
    assert twice.model == pytest.approx(once.hardware)
    assert twice.hardware > twice.model


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
    assert trainer.step_flops is not None
    assert trainer.step_flops.hardware * jax.device_count() == pytest.approx(analytic, rel=0.05)


@pytest.mark.gpu("cuDNN's fused attention runs on a CUDA GPU alone")
@pytest.mark.mesh(devices=1)
@pytest.mark.parametrize("window", [None, 64])
def test_a_fused_attention_counts_the_pairs_its_mask_computes(window):
    """A causal kernel computes about half the scores, a sliding window fewer: the forward's two
    products and the backward's four count one multiply-add per pair it reaches, head and width.
    On an H100 FlashAttention-3 counted the whole square of a causal call, twice its work."""
    batch, length, heads, width = 1, 256, 2, 128
    shape = jax.ShapeDtypeStruct((batch, length, heads, width), jnp.bfloat16)
    spans = {} if window is None else {"local_window_size": (window, 0)}

    def attend(q, k, v):
        return jax.nn.dot_product_attention(q, k, v, implementation="cudnn", is_causal=True, **spans)

    def loss(q, k, v):
        return attend(q, k, v).astype(jnp.float32).sum()

    pairs = sum(min(row, length if window is None else window) + 1 for row in range(length))
    forward = compiled_flops(jax.jit(attend).lower(shape, shape, shape).compile())
    both = compiled_flops(jax.jit(jax.grad(loss, argnums=(0, 1, 2))).lower(shape, shape, shape).compile())
    assert forward == 2 * 2 * batch * heads * pairs * width
    assert both == (2 + 4) * 2 * batch * heads * pairs * width


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


def test_fit_reports_throughput_to_the_tracker():
    """The logging tick carries the throughput numbers along with the loss."""
    tracker = RecordingTracker()
    make_trainer(tracker=tracker).fit(
        Loader(lambda partition: batches(), None, batch=BATCH, records=None), steps=3, log_every=1)

    ticks = [p for _, p in tracker.scalars if "train/samples_per_sec" in p]
    assert len(ticks) == 3, "no throughput was logged"
    assert all(p["train/step_time_ms"] > 0 for p in ticks)
    assert all(p["train/samples_per_sec"] > 0 for p in ticks)


class TickGatedTracker(RecordingTracker):
    """Marks each step's tick, so an input can start a batch's delay only
    when the loop is about to ask for it: the loop reads a step's log once
    the next step is dispatched, just before it asks for the batch after."""

    def __init__(self, steps):
        super().__init__()
        self.ticked = [threading.Event() for _ in range(steps + 1)]

    def log(self, scalars, step):
        super().log(scalars, step)
        if "train/step_time_ms" in scalars:
            self.ticked[step].set()


def test_fit_reports_the_wait_on_input_and_a_later_compile():
    """An input slower than the step shows as train/data_wait_ms, and a batch
    of a shape the loop has not met compiles inside its tick, which reports the
    compile as train/compile_s; the ticks that compiled nothing report none.

    Each batch's 100 ms delay starts at the tick of the step two before the
    one it feeds, read as the step before is dispatched, when the loop is
    about to ask for it, so the loop waits nearly all of it however long its
    own work between reads takes. A delay that ran beside the loop's work
    would leave a wait of 100 ms less that work, which a loaded runner
    stretched past 58 ms."""
    steps = 10
    tracker = TickGatedTracker(steps)

    def slow():
        rng = np.random.default_rng(0)
        for index, rows in enumerate(itertools.chain([BATCH] * 2, itertools.repeat(2 * BATCH))):
            if 1 < index <= steps:
                tracker.ticked[index - 1].wait()
            time.sleep(0.1)
            x = rng.normal(size=(rows, FEATURES)).astype(np.float32)
            yield {"x": x, "y": 2 * x[:, :2]}

    make_trainer(tracker=tracker).fit(
        Loader(lambda partition: slow(), None, batch=BATCH, records=None), steps=steps, log_every=1)

    ticks = [p for _, p in tracker.scalars if "train/step_time_ms" in p]
    assert len(ticks) == 10
    waits = [p["train/data_wait_ms"] for p in ticks]
    assert min(waits[-4:]) >= 50, waits
    assert [("train/compile_s" in p) for p in ticks] == [False, False, True] + [False] * 7
    assert ticks[2]["train/compile_s"] > 0


def test_the_gpu_autotuners_picks_persist_beside_the_executables_for_this_xla(tmp_path, monkeypatch):
    """A later compile of the same fusions takes the picks without measuring
    them; a directory the run names is its own."""
    previous = jax.config.jax_compilation_cache_dir
    try:
        monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_autotune_level=4")
        enable_compilation_cache(str(tmp_path / "cache"))
        picks = tmp_path / "cache" / "autotune" / f"jaxlib-{importlib.metadata.version('jaxlib')}"
        assert os.environ["XLA_FLAGS"].split() == [
            "--xla_gpu_autotune_level=4", f"--xla_gpu_per_fusion_autotune_cache_dir={picks}"]
        assert picks.is_dir()
        named = "--xla_gpu_per_fusion_autotune_cache_dir=/elsewhere"
        monkeypatch.setenv("XLA_FLAGS", named)
        enable_compilation_cache(str(tmp_path / "cache"))
        assert os.environ["XLA_FLAGS"] == named
    finally:
        jax.config.update("jax_compilation_cache_dir", previous)


@pytest.mark.mesh(devices=8)
def test_the_collectives_of_a_compiled_step_are_counted_with_the_bytes_they_move():
    """A step over eight devices that gathers a bf16 array whole and sums an f32 one across them:
    the shared parser names both collectives (`collective`) and the bytes each result holds, and
    the cluster diagnostic counts each once, an async pair by its `-done`, with those bytes."""
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from tool_support import load

    from dew.telemetry.instrumentation import collective, hlo_instructions

    mesh = Mesh(np.asarray(jax.devices()[:8]), ("data",))

    def step(gathered, summed):
        whole = jax.lax.all_gather(gathered, "data", tiled=True)
        return whole, jax.lax.psum(summed, "data")

    mapped = jax.shard_map(step, mesh=mesh, in_specs=(P("data"), P("data")), out_specs=(P(), P()))
    gathered = jax.device_put(jnp.ones((64, 32), jnp.bfloat16), NamedSharding(mesh, P("data")))
    summed = jax.device_put(jnp.ones((8, 16), jnp.float32), NamedSharding(mesh, P("data")))
    text = jax.jit(mapped).lower(gathered, summed).compile().as_text()
    instructions = hlo_instructions(text).values()
    found = {collective(instruction.op) for instruction in instructions} - {None}
    assert found == {"all-gather", "all-reduce"}
    counted = load("cluster_vs_gpu").collectives(text)
    assert {kind: (entry["ops"], entry["bytes"]) for kind, entry in counted.items()} == {
        "all-gather": (1, 64 * 32 * 2), "all-reduce": (1, 16 * 4)}
