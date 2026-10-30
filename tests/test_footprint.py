"""The residual storage a pullback names before compiling, not a claim about XLA's scheduled peak."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from dew.training.footprint import device_bytes, gradient_peak, residual_bytes


def test_resident_device_bytes_do_not_include_host_state():
    """A state leaf placed on pinned host memory is not HBM, even though it has the same global
    shape and partition as a device leaf. Counting the two as resident would charge offload twice."""
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("data",))
    sharding = NamedSharding(mesh, P())
    device = jax.ShapeDtypeStruct((8, 16), jnp.float32, sharding=sharding)
    host = jax.ShapeDtypeStruct((8, 16), jnp.float32, sharding=sharding.with_memory_kind("pinned_host"))
    assert device_bytes({"params": device, "optimizer": host}) == 8 * 16 * 4


def test_a_pullback_counts_new_residuals_not_input_storage():
    """On a CPU the exponential needed to differentiate exp is new storage, an expensive value XLA keeps.
    A dot product needs the other operand, already resident as an input, which must not count again as
    residual storage; nor does the cosine sine's derivative needs, which XLA recomputes from its input
    in the backward's fusion, a sine's cosine being cheap (XLA's `InstructionFusion::IsExpensive`)."""
    shape = jax.ShapeDtypeStruct((8, 16), jnp.float32)

    def exponential(params):
        return jnp.exp(params).sum(), ()

    def sine(params):
        return jnp.sin(params).sum(), ()

    def product(params, values):
        return jnp.vdot(params, values), ()

    assert residual_bytes(exponential, shape, (), platform="cpu") == 8 * 16 * 4
    assert residual_bytes(sine, shape, (), platform="cpu") == 0
    assert residual_bytes(product, shape, (shape,), platform="cpu") == 0


def test_a_pullback_counts_a_cheap_chain_as_the_product_it_is_recomputed_from():
    """Residuals XLA recomputes in the backward's fusions, each a cheap elementwise value (a select, a
    multiply) of a product, count as that product, once: a rectified, scaled product squared keeps the
    product's bytes, not the two cheap values JAX's pullback closes over."""
    weights, rows = jax.ShapeDtypeStruct((16, 32), jnp.float32), jax.ShapeDtypeStruct((8, 16), jnp.float32)

    def loss(weights, rows):
        return jnp.sum(jnp.square(jnp.maximum(rows @ weights, 0.0) * 2.0 + 1.0)), ()

    assert residual_bytes(loss, weights, (rows,), platform="cpu") == 8 * 32 * 4


def test_a_custom_pullback_counts_one_buffer_shared_by_two_residual_slots():
    """A custom VJP retains the same intermediate twice and its input once. Only the one new
    intermediate's storage counts; neither repeating its reference nor retaining an input doubles it."""
    @jax.custom_vjp
    def summed(params):
        return jnp.sin(params).sum()

    def forward(params):
        # An exponential, which XLA keeps (a cosine of real values it would recompute from the input).
        exponential = jnp.exp(params)
        return summed(params), (exponential, exponential, params)

    def backward(saved, cotangent):
        first, second, inputs = saved
        return ((first + second) * cotangent / 2 + inputs * 0,)

    summed.defvjp(forward, backward)

    def loss(params):
        return summed(params), ()

    shape = jax.ShapeDtypeStruct((8, 16), jnp.float32)
    assert residual_bytes(loss, shape, (), platform="cpu") == 8 * 16 * 4


def test_a_gpu_pullback_counts_a_transcendental_chain_as_the_product_it_is_recomputed_from():
    """On a GPU XLA's priority fusion recomputes every elementwise value in its consumers, a softmax's
    exponentials and quotients among them, from the product they are computed from: an attention's
    probabilities keep its scores' bytes there, and on a CPU, whose fusion keeps the expensive values
    (`InstructionFusion::IsExpensive`), the exponentials and quotients as well."""
    keys, queries = jax.ShapeDtypeStruct((16, 32), jnp.float32), jax.ShapeDtypeStruct((8, 32), jnp.float32)

    def attended(keys, queries):
        scores = queries @ keys.T
        weights = jnp.exp(scores - jnp.max(scores, axis=-1, keepdims=True))
        return jnp.sum((weights / jnp.sum(weights, axis=-1, keepdims=True)) @ keys), ()

    scores = 8 * 16 * 4
    gpu = residual_bytes(attended, keys, (queries,), platform="gpu")
    assert scores <= gpu < 2 * scores < residual_bytes(attended, keys, (queries,), platform="cpu")


def test_a_gradients_peak_holds_the_gradient_beside_the_cotangent_each_residual_freed_after_its_read():
    """The most a gradient program holds at once, in its traced order: through two products, the second
    weight's gradient and the first layer's cotangent, then the first weight's gradient beside them, each
    residual (the first layer's output, the second's) freed after its last read, which comes before."""
    rows = jax.ShapeDtypeStruct((8, 16), jnp.float32)
    params = {"first": jax.ShapeDtypeStruct((16, 64), jnp.float32),
              "second": jax.ShapeDtypeStruct((64, 16), jnp.float32)}

    def loss(params, rows):
        return jnp.sum(jnp.exp((rows @ params["first"]) @ params["second"])), ()

    # Both gradients, 64 x 16 floats each, and the first layer's cotangent, 8 x 64.
    assert gradient_peak(loss, params, (rows,), platform="gpu") == (2 * 64 * 16 + 8 * 64) * 4


@pytest.mark.parametrize(("generation", "workspace"), [("sm80", 4 << 20), ("sm90", 32 << 20)])
def test_a_derived_step_holds_the_gemm_workspace_xla_gives_its_gpu(generation, workspace):
    """A derived step on a CUDA GPU holds beside its gradient program's peak the workspace XLA hands each
    cuBLAS matmul (gemm_rewriter's `GemmWorkspaceRewriteVisitor`): 32 MiB from Hopper on, 4 MiB before
    (`GemmConfig::kHopperWorkspace`, `kDefaultWorkspace`); a CPU's matmuls take none."""
    from dew.training.ahead import compile_only_gpus
    from dew.training.footprint import Footprint
    from dew.training.memory import derived_headroom

    capacity = 1 << 30
    step = Footprint(residuals=1 << 20, gradients=1 << 20, held=0, peak=2 << 20)
    (gpu,) = compile_only_gpus(generation, 1, capacity)
    empty = Footprint(residuals=0, gradients=0, held=0, peak=0)
    # The workspace is one buffer beside every step, whatever its peak.
    assert derived_headroom(empty, [gpu], capacity) == capacity - workspace
    assert derived_headroom(step, [gpu], capacity) == capacity - workspace - (2 << 20)
