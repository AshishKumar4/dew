"""The upstream bugs Dew works around, each reproduced as it behaves on the
installed jax and jaxlib.

Each test asserts the right behaviour and is marked xfail(strict=True). While
the bug stands the test fails, and the suite passes; the release that fixes it
makes the test pass, which fails the suite and names the workaround to delete.
Only the bug's own symptom counts as the expected failure (`assert_fixed`):
any other outcome raises an error that no xfail excuses. A repro that can
crash its process runs in a fresh interpreter, where a crash is an outcome.

The CPU repro runs in CI. The CUDA repros run on a cuda lane
(`JAX_PLATFORMS=cuda python -m pytest tests/test_upstream_bugs.py`), the TPU
one on a TPU VM, and elsewhere they skip.

jax-ml/jax#40940 has its repro in
tests/test_distribution.py::test_a_pools_second_run_loads_what_its_first_compiled.
The suite runs it on the patched jax constraints.txt installs, where it must
pass. CI's package job runs it on PyPI's jax with DEW_UPSTREAM_JAX=1, where it
is expected to fail until a release carries the fix.
"""

import os
import re
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

CUDA = pytest.mark.skipif(jax.default_backend() != "gpu", reason="a CUDA bug, which a cuda lane reproduces")
TPU = pytest.mark.skipif(jax.default_backend() != "tpu", reason="a TPU bug, which a TPU VM reproduces")


def assert_fixed(said: str, right: str, bug: str) -> None:
    """Assert that a repro said `right`. Where it said neither that nor something
    the regex `bug`, the bug's own symptom, matches, raise a RuntimeError
    instead, which the tests' xfail marks do not excuse."""
    if said != right and not re.search(bug, said):
        raise RuntimeError(f"the repro said {said!r}: neither the right {right!r} nor the bug's {bug!r}")
    assert said == right


def outcome(script: str, **env: str) -> str:
    """The last `outcome: ...` line `script` printed in a fresh interpreter
    under `env`, so a crash after a line leaves that line the outcome."""
    done = subprocess.run([sys.executable, "-c", script], env={**os.environ, **env}, capture_output=True,
                          text=True, timeout=600)
    said = [line.removeprefix("outcome: ") for line in done.stdout.splitlines()
            if line.startswith("outcome: ")]
    if not said:
        raise RuntimeError(f"the repro printed no outcome (exit {done.returncode}):\n"
                           f"{done.stdout[-2000:]}{done.stderr[-4000:]}")
    return said[-1]


CHECK_BEFORE_SHARD_MAP = """
import contextlib, functools
import jax, jax.numpy as jnp
from jax.experimental import checkify
from jax.sharding import AxisType, PartitionSpec as P

mesh = jax.make_mesh((2, 2), ("a", "b"), axis_types=(AxisType.Auto, AxisType.Auto))

def program(manual, context):
    spec = P(manual)
    def f(x):
        checkify.check(jnp.all(x > -1.0), "x must be above -1")
        mapped = functools.partial(jax.shard_map, axis_names=set(manual), in_specs=spec, out_specs=spec)
        return (mapped if context else functools.partial(mapped, mesh=mesh))(lambda y: y * 2.0)(x)
    return f

x = jnp.ones((8,), jnp.float32)
failed = []
for context in (True, False):
    for manual in (("b",), ("a", "b")):
        try:
            with jax.set_mesh(mesh) if context else contextlib.nullcontext():
                error, out = jax.jit(checkify.checkify(program(manual, context)))(x)
            error.throw()
        except Exception as failure:
            failed.append(f"context mesh {context}, manual {list(manual)}: "
                          + str(failure).splitlines()[0][:200])
print("outcome: " + ("; ".join(failed) or "every check carried"))
"""


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "jax-ml/jax#40907: when this passes, a device check can stand before a shard_map: "
    "dew.nn.kv_cache.refuse_unassigned's shape check, dew.inference.serving_kernel's "
    "model-before-draw step and dew.sampling.text._refuse_exchange can go"))
def test_a_check_carries_into_a_shard_map():
    """checkify carries a check made before a `shard_map` into it, with the
    mesh set as the context or passed, whether the map makes some mesh axes
    manual or all. On jax 0.11.2 the context mesh fails to match the aval's
    ("should match the aval mesh"), and a partly manual map without one
    fails to lower ("Cannot lower jaxpr with verifier errors")."""
    said = outcome(CHECK_BEFORE_SHARD_MAP, JAX_PLATFORMS="cpu",
                   XLA_FLAGS="--xla_force_host_platform_device_count=4")
    assert_fixed(said, "every check carried",
                 r"should match the aval mesh|Cannot lower jaxpr with verifier errors")


@CUDA
@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "openxla/xla#49380, fixed by openxla/xla#49498: when this passes, dew.nn.scatter.DROPPED "
    "can give way to an index at its axis size"))
def test_a_deterministic_scatter_drops_an_index_at_its_axis_size():
    """A scatter with mode='drop' drops a column index equal to its axis size.
    The cuda lane's --xla_gpu_deterministic_ops linearizes the index and drops
    only a position past the whole operand, so row 0's dropped (0, 4) lands
    on (1, 0) and overwrites row 1's 3 with 2."""
    if "--xla_gpu_deterministic_ops=true" not in os.environ.get("XLA_FLAGS", ""):
        pytest.skip("the bug is the deterministic scatter's, which the cuda lane's flags select")
    scatter = jax.jit(lambda buffer, rows, columns, values: buffer.at[rows, columns].set(values, mode="drop"))
    written = scatter(jnp.zeros((2, 4), jnp.int32), jnp.arange(2)[:, None], jnp.array([[0, 4], [0, 4]]),
                      jnp.array([[1, 2], [3, 4]], jnp.int32))
    assert_fixed(str(written.tolist()), "[[1, 0, 0, 0], [3, 0, 0, 0]]",
                 re.escape("[[1, 0, 0, 0], [2, 0, 0, 0]]"))


@CUDA
@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "openxla/xla#49299: when this passes, dew.diffusion.schedules.source_grids may interpolate "
    "by jnp.interp's bisection rather than comparing against the whole grid"))
def test_a_scan_of_one_row_gathers_at_a_bisected_offset():
    """A scan over one row reads a table at an index a bisection (a while
    loop) found. XLA:GPU's DynamicSliceAnnotator evaluates the slice offset
    without running the loop and fails the compile ("Failed to evaluate
    instruction"); two rows, or the same program on XLA:CPU, compile."""
    table = jnp.asarray(np.linspace(1.0, 0.0, 9), jnp.float32)
    steps = jnp.asarray(np.arange(8, dtype=np.float32))

    def walk(x):
        def body(x, t):
            index = jnp.searchsorted(jnp.arange(9.0), jnp.full(x.shape[:1], t))
            return x * table[index][:, None], None
        return jax.lax.scan(body, x, steps)[0]

    try:
        said = f"{float(jax.jit(walk)(jnp.ones((1, 4)))[0, 0]):.6g}"
    except Exception as failure:
        said = str(failure).splitlines()[0]
    two_rows = f"{float(jax.jit(walk)(jnp.ones((2, 4)))[0, 0]):.6g}"
    assert_fixed(said, two_rows, "Failed to evaluate instruction")


TWO_CUDNN_BACKWARDS = """
import jax, jax.numpy as jnp

def loss(q, k, v):
    x = jax.nn.dot_product_attention(q, k, v, implementation="cudnn")
    x = jax.nn.dot_product_attention(x, k, v, implementation="cudnn")
    return x.astype(jnp.float32).sum()

q = k = v = jnp.ones((1, 128, 4, 64), jnp.bfloat16)
step = jax.jit(jax.grad(loss, argnums=(0, 1, 2))).lower(q, k, v).compile()
print("outcome: compiled", flush=True)
jax.block_until_ready(step(q, k, v))
print("outcome: ran", flush=True)
"""


@CUDA
@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "openxla/xla#46500: when this passes, dew.nn.attention may run cudnn under "
    "--xla_gpu_deterministic_ops (cudnn_runs and the explicit refusal)"))
def test_two_identical_cudnn_attention_backwards_run_under_deterministic_ops():
    """One executable holding two structurally identical cudnn attention
    backward calls, as every multi-layer model does, runs under
    --xla_gpu_deterministic_ops. XLA keys the second call's cudnn graph
    without the flag and fails it at execution, after the compile succeeds."""
    if float(jax.devices()[0].compute_capability) < 8.0:
        pytest.skip("cudnn's bf16 attention needs sm80 or later")
    said = outcome(TWO_CUDNN_BACKWARDS, JAX_PLATFORMS="cuda", XLA_PYTHON_CLIENT_PREALLOCATE="false",
                   XLA_FLAGS="--xla_gpu_deterministic_ops=true")
    assert_fixed(said, "ran", "^compiled$")


A_FREED_POOL = """
import jax, jax.numpy as jnp

limit = jax.devices()[0].memory_stats()["bytes_limit"]
first = jnp.zeros(int(0.45 * limit) // 4, jnp.float32).block_until_ready()
del first
try:
    jnp.zeros(int(0.6 * limit) // 4, jnp.float32).block_until_ready()
    print("outcome: allocated", flush=True)
except Exception as failure:
    print("outcome: " + str(failure).splitlines()[0], flush=True)
"""


@CUDA
@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "openxla/xla#50052: when this passes, a growing pool returns the regions it freed, so a "
    "process that compiles a step and one that loads it find the same memory, and "
    "dew.training.rungs need not record the rung"))
def test_a_growing_pool_holds_a_larger_buffer_once_the_smaller_is_freed():
    """With XLA_PYTHON_CLIENT_PREALLOCATE=false the allocator pool grows a
    region for an allocation no free chunk holds and never gives a free region
    back, so a buffer of 0.6 of the limit fails once one of 0.45 was freed,
    with nothing in use. The limit is a fifth of the device, beside what the
    suite's own process holds."""
    said = outcome(A_FREED_POOL, JAX_PLATFORMS="cuda", XLA_PYTHON_CLIENT_PREALLOCATE="false",
                   XLA_PYTHON_CLIENT_MEM_FRACTION="0.2")
    assert_fixed(said, "allocated", "RESOURCE_EXHAUSTED")


@TPU
@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "Dew #1 (openxla/xla#49635, in libtpu 0.0.50, which pairs with the next jax): when this "
    "passes, UniPC's corrector may gate on the step counter again "
    "(dew.sampling.solvers.unipc carries the order instead)"))
def test_a_scan_counter_that_starts_at_zero_is_not_above_zero():
    """Inside a jitted scan whose counter starts at a compile-time 0, the
    gate `i > 0` is closed at the first step. XLA:TPU's while-loop simplifier
    folds the compare to true, so a TPU v6e adds one at every step."""
    @jax.jit
    def gated(x):
        def body(carry, _):
            i, x = carry
            x = jnp.where(i > 0, x + 1.0, x)
            return (i + 1, x), x
        return jax.lax.scan(body, (jnp.int32(0), x), None, length=3)[1]

    assert_fixed(str(gated(jnp.float32(0)).tolist()), "[0.0, 1.0, 2.0]", re.escape("[1.0, 2.0, 3.0]"))
