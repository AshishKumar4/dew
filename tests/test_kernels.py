"""Which kernel the 'auto' rule reaches for, and the XLA flags a run passes
to the backend.

'auto' asks two predicates per trace: `cudnn_runs` for the fused cudnn kernel
and then `tpu_runs` for the pallas splash kernel. Both read the backend, so
both selections are testable anywhere by monkeypatching `jax.default_backend`
and tracing the call without running it. What actually executes on a CUDA
device skips elsewhere; run that half with
`JAX_PLATFORMS=cuda python -m pytest tests/test_kernels.py`. The splash
kernel's own numbers are tests/test_attention_splash.py.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.nn.attention import (
    SPLASH_LANES,
    attention_kernel,
    cudnn_runs,
    scaled_dot_product_attention,
    tpu_runs,
)
from dew.telemetry.devices import apply_xla_flags, deterministic_ops_requested, xla_flag
from dew.training import MeshSpec, build_mesh

on_gpu = pytest.mark.skipif(jax.default_backend() != 'gpu',
                            reason="needs a cuda device")


def qkv(shape, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    return tuple(jax.random.normal(key, shape, jnp.bfloat16) for key in keys)


@pytest.fixture
def without_deterministic_ops(monkeypatch):
    """The cuda lane runs the whole suite under --xla_gpu_deterministic_ops,
    which Dew refuses cudnn attention under (openxla/xla#46500). XLA read the
    variable when it opened the backend, so taking the flag back out of the
    environment leaves this executable's reductions as they are and lets a
    kernel test ask for the kernel it measures."""
    kept = [flag for flag in os.environ.get("XLA_FLAGS", "").split()
            if not flag.startswith("--xla_gpu_deterministic_ops")]
    monkeypatch.setenv("XLA_FLAGS", " ".join(kept))


def value_and_grads(implementation, query, key, value, **kwargs):
    def loss(q, k, v):
        out = scaled_dot_product_attention(q, k, v, implementation=implementation, **kwargs)
        return jnp.sum(out.astype(jnp.float32) ** 2), out

    (_, out), grads = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True))(
        query, key, value)
    return [np.asarray(x, np.float32) for x in (out, *grads)]


@on_gpu
@pytest.mark.parametrize("q_len, kv_len, causal", [
    (1024, 77, False),   # every cross-attention over CLIP's 77 text tokens
    (9, 7, False),       # short enough that one attended pad key would move an eighth of the mass
    (333, 333, True),    # the concatenated text-plus-image sequence, causal
])
def test_cudnn_trains_odd_lengths_and_agrees_with_xla(q_len, kv_len, causal,
                                                      without_deterministic_ops):
    """cudnn's kernel has no backward pass for an odd length; the padded call
    trains, and what it computes for the real rows is what the xla kernel
    computes: within two bf16 ulps of the output scale, forward and backward,
    which is also how far the two kernels sit apart at an even length. A pad
    key left unmasked shifts every output by 1/(kv+1) of the value mass, 12%
    at 7 keys, and a pad query row left in the output changes its shape."""
    query, _, _ = qkv((2, q_len, 4, 64))
    _, key, value = qkv((2, kv_len, 4, 64), seed=1)

    fused = value_and_grads('cudnn', query, key, value, causal=causal)
    reference = value_and_grads('xla', query, key, value, causal=causal)

    for got, want in zip(fused, reference):
        assert got.shape == want.shape
        assert np.abs(got - want).max() <= 2 ** -6 * np.abs(want).max()


def test_flags_are_appended_to_what_the_environment_already_carries(monkeypatch):
    """The test suite itself sets a flag, and a run's own flags have to add to
    it, not replace it."""
    monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
    apply_xla_flags("--xla_gpu_autotune_level=4")
    assert os.environ["XLA_FLAGS"] == (
        "--xla_force_host_platform_device_count=8 --xla_gpu_autotune_level=4")


@pytest.mark.parametrize("flags", [None, ""])
def test_no_flags_leaves_the_environment_alone(monkeypatch, flags):
    monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
    apply_xla_flags(flags)
    assert os.environ["XLA_FLAGS"] == "--xla_force_host_platform_device_count=8"


def test_flags_reach_an_environment_that_had_none(monkeypatch):
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    apply_xla_flags("--xla_gpu_triton_gemm_any=true")
    assert os.environ["XLA_FLAGS"] == "--xla_gpu_triton_gemm_any=true"


@pytest.mark.parametrize("flags, value", [
    ("--xla_gpu_deterministic_ops=true", "true"),
    ("--xla_gpu_deterministic_ops", "true"),
    ("--xla_gpu_deterministic_ops=false", "false"),
    ("--xla_force_host_platform_device_count=8", None),
    ("--xla_gpu_deterministic_ops_level=2", None),
    ("--xla_gpu_deterministic_ops=true --xla_gpu_deterministic_ops=false", "false"),
    ("--xla_gpu_deterministic_ops=false --xla_gpu_autotune_level=4 "
     "--xla_gpu_deterministic_ops=true", "true"),
])
def test_xla_flag_reads_the_value_the_run_asked_for(monkeypatch, flags, value):
    """A bare flag is on, a repeated one keeps its last value, and a longer
    flag name that begins the same way is another flag."""
    monkeypatch.setenv("XLA_FLAGS", flags)
    assert xla_flag("xla_gpu_deterministic_ops") == value


def test_xla_flag_is_none_when_the_variable_is_unset(monkeypatch):
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    assert xla_flag("xla_gpu_deterministic_ops") is None


@pytest.mark.parametrize("flags, requested", [
    ("--xla_gpu_deterministic_ops=true", True),
    ("--xla_gpu_deterministic_ops=TRUE", True),
    ("--xla_gpu_deterministic_ops=1", True),
    ("--xla_gpu_deterministic_ops", True),
    ("--xla_gpu_deterministic_ops=false", False),
    ("--xla_gpu_deterministic_ops=0", False),
    ("--xla_force_host_platform_device_count=8", False),
])
def test_deterministic_ops_requested_reads_the_flag(monkeypatch, flags, requested):
    monkeypatch.setenv("XLA_FLAGS", flags)
    assert deterministic_ops_requested() is requested


def test_deterministic_ops_keep_the_auto_rule_off_cudnn(monkeypatch):
    """Everything else the predicate reads holds (a gpu backend, bf16 inputs,
    a 64-wide head, no softcap), so the flag is what decides."""
    query, _, _ = qkv((1, 8, 2, 64))
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
    assert cudnn_runs(query)
    monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=8 "
                                    "--xla_gpu_deterministic_ops=true")
    assert not cudnn_runs(query)


def test_auto_computes_what_xla_computes_under_deterministic_ops(monkeypatch):
    """What the selection did is in the output: on a gpu backend 'auto' would
    fuse this call, and under the flag it returns the xla kernel's numbers
    bit for bit."""
    query, key, value = qkv((1, 8, 2, 64))
    expected = scaled_dot_product_attention(query, key, value, implementation='xla')
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_deterministic_ops=true")
    routed = scaled_dot_product_attention(query, key, value, implementation='auto')
    assert np.array_equal(np.asarray(routed, np.float32), np.asarray(expected, np.float32))


def cudnn_shape(query, key, value):
    """Trace an explicit cudnn call without running it, so what the selection
    decided is readable on a host that has no cudnn kernel to run."""
    return jax.eval_shape(
        lambda q, k, v: scaled_dot_product_attention(q, k, v, implementation='cudnn'),
        query, key, value)


def test_an_explicit_cudnn_call_is_refused_under_deterministic_ops(monkeypatch):
    """XLA cannot execute cudnn attention's backward pass under the flag
    (openxla/xla#46500), so the request is refused by name while the call is
    traced, not at the training step that would have crashed."""
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_deterministic_ops=true")
    with pytest.raises(ValueError, match="xla_gpu_deterministic_ops"):
        cudnn_shape(*qkv((1, 8, 2, 64)))


def test_an_explicit_cudnn_call_stands_without_the_flag(without_deterministic_ops):
    """The refusal belongs to the flag, not to the implementation: the same
    call traces to its output shape when the run asked for nothing."""
    assert cudnn_shape(*qkv((1, 8, 2, 64))).shape == (1, 8, 2, 64)


@pytest.fixture
def tpu_backend(monkeypatch):
    """A tpu backend for the selection rules. The pallas kernels lower at
    compile time, not at trace time, so a traced call under this fixture
    names the kernel the rule picked without needing the device that runs
    it, the way `cudnn_shape` reads the cudnn selection off a host with no
    cudnn."""
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")


def kernel_chosen(query, key, value, **kwargs):
    """The names of the kernels one traced dispatch holds."""
    text = jax.make_jaxpr(
        lambda q, k, v: attention_kernel(q, k, v, **kwargs))(query, key, value)
    printed = text.pretty_print(use_color=False)
    return {name for name in ('splash', 'flash_attention') if name in printed}


@pytest.mark.parametrize("structure", [
    {}, {"causal": True}, {"sliding_window": 128},
])
def test_auto_picks_the_splash_kernel_on_a_tpu_backend(tpu_backend, structure):
    """The shapes a decoder trains at meet the kernel's constraints, so a run
    that asked for nothing gets the block-sparse kernel instead of XLA's
    attention, which is the whole point of the rule."""
    query, key, value = qkv((2, 512, 8, 128))
    assert tpu_runs(query, key, **structure)
    assert kernel_chosen(query, key, value, implementation='auto', **structure) == {'splash'}


@pytest.mark.parametrize("shape, reason", [
    ((2, 200, 8, 128), "a length the mask blocking has no block size for"),
    ((2, 1, 8, 128), "a decode step's single query row"),
])
def test_auto_stays_on_xla_where_splash_cannot_tile_the_sequence(tpu_backend, shape, reason):
    """Every length splash takes is a multiple of 128; the lengths that are
    not fall back rather than failing at compile time."""
    query, key, value = qkv(shape)
    assert not tpu_runs(query, key), reason
    assert kernel_chosen(query, key, value, implementation='auto', causal=True) == set()


def test_auto_stays_on_xla_where_the_call_carries_a_bias(tpu_backend):
    """Splash has no bias argument at all, so T5's relative position table
    keeps XLA's attention under 'auto' and the older flash kernel under an
    explicit 'tpu'."""
    query, key, value = qkv((2, 512, 8, 128))
    bias = jnp.zeros((2, 8, 512, 512), jnp.bfloat16)
    assert not tpu_runs(query, key, bias=bias)
    assert kernel_chosen(query, key, value, implementation='auto', bias=bias) == set()
    assert kernel_chosen(query, key, value, implementation='tpu',
                         bias=bias) == {'flash_attention'}


def test_auto_stays_on_xla_where_the_mask_is_a_value_of_the_trace(tpu_backend):
    """A decode mask over cache slots exists only inside the trace, and the
    descriptor is built while the executable is."""
    query, key, value = qkv((2, 512, 8, 128))

    def traced(q, k, v, mask):
        assert not tpu_runs(q, k, mask=mask)
        return attention_kernel(q, k, v, implementation='auto', mask=mask)

    printed = jax.make_jaxpr(traced)(
        query, key, value, jnp.ones((1, 1, 512, 512), bool)).pretty_print(use_color=False)
    assert 'splash' not in printed


@pytest.mark.parametrize("dtype, taken", [
    (jnp.bfloat16, True), (jnp.float32, True), (jnp.float16, False),
])
def test_tpu_runs_reads_the_dtypes_a_tpu_matmul_takes(tpu_backend, dtype, taken):
    """fp16 has no MXU path, so it is the one dtype the rule turns down."""
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    query, key = (jax.random.normal(k, (2, 512, 8, 128), dtype) for k in keys)
    assert tpu_runs(query, key) is taken


def test_tpu_runs_needs_the_tpu_backend(monkeypatch):
    """Everything else the predicate reads holds; only the backend says no,
    because the interpreter that runs splash elsewhere is far slower than the
    XLA attention 'auto' falls back to."""
    query, key, _ = qkv((2, SPLASH_LANES, 8, 128))
    assert not tpu_runs(query, key)
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    assert tpu_runs(query, key)


@pytest.mark.mesh
def test_auto_stays_on_xla_where_the_mesh_splits_the_sequence(tpu_backend):
    """Splash states its sharding as a shard_map spec, and Dew's sequence
    parallelism is GSPMD constraints around a whole-sequence kernel: a pallas
    call with no partitioning rule would have the queries gathered back to
    serve it, undoing the split. An unmasked call is the only one that could
    reach splash under that mesh at all, because a causal one arrives with a
    striped mask that is a value of the trace."""
    query, key, _ = qkv((8, 512, 8, 128))
    assert tpu_runs(query, key)
    with jax.set_mesh(build_mesh(MeshSpec(fsdp=4, sequence=2))):
        assert not tpu_runs(query, key)


def test_a_softcapped_call_never_reaches_either_tpu_kernel(tpu_backend):
    """No fused kernel has Gemma 2's tanh between the scaling and the
    softmax, so 'auto' resolves it to xla and an explicit 'tpu' refuses by
    name."""
    query, key, value = qkv((2, 512, 8, 128))
    assert not tpu_runs(query, key, 30.0)
    assert kernel_chosen(query, key, value, implementation='auto', softcap=30.0) == set()
    with pytest.raises(ValueError, match="'tpu'"):
        jax.eval_shape(
            lambda q, k, v: attention_kernel(q, k, v, implementation='tpu', softcap=30.0),
            query, key, value)
