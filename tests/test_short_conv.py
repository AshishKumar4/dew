"""The short conv's kernels (dew.nn.kernels.short_conv) against `causal_conv1d`'s XLA arithmetic.

Host tests interpret the kernels at small shapes, where they hold the runs a program walks, the pieces
it writes and each gradient to `xla_conv1d`, with the float64 conv as the truth. The kernel lab's GPU
tier runs them on the GPU, with the training shapes' tests: there each output's and gradient's float64
distance is held to XLA's own by tests/reference_error.py's rule.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fla_gdn_support import KernelCase
from kernel_compile import compile_kernels
from reference_error import assert_as_exact_as_the_reference, assert_rounds_where_the_reference_does, within

from dew.nn.kernels import generation, short_conv
from dew.nn.linear import causal_conv1d, conv_refusal, xla_conv1d

pytestmark = pytest.mark.kernel

DELTA = ((32, 16), (32, 16), (32, None))
"""A delta rule's queries, keys and values on two 16-wide heads: the keys and queries normalised."""


def operands(batch, length, parts, *, taps=4, dtype=jnp.bfloat16, bias=False, valid=False, seed=0):
    rng = np.random.default_rng(seed)
    channels = sum(width for width, _ in parts)
    x = jnp.asarray(rng.normal(size=(batch, length, channels)), dtype)
    weights = jnp.asarray(rng.normal(size=(channels, taps)) * 0.5, jnp.float32)
    shift = jnp.asarray(rng.normal(size=(channels,)) * 0.1, jnp.float32) if bias else None
    # Right-padded rows: the second keeps half its tokens.
    kept = jnp.asarray(np.arange(length)[None, :] < np.array([length, length // 2] * batch)[:batch, None])
    return x, weights, shift, kept if valid else None


def documents(batch, length, seed=2):
    """Packed rows `[batch, length]`: documents numbered from 1 of 1 to a few tens of tokens, so some
    are shorter than a conv's window and others cross a forward tile and a backward run."""
    rng = np.random.default_rng(seed)
    starts = rng.random((batch, length)) < rng.choice([0.03, 0.3], size=(batch, length))
    starts[:, 0] = True
    return jnp.asarray(np.cumsum(starts, axis=1), jnp.int32)


def cotangents(x, parts, dtype, seed=1):
    rng = np.random.default_rng(seed)
    return tuple(jnp.asarray(rng.normal(size=(*x.shape[:2], width)), dtype) for width, _ in parts)


def run(conv, x, weights, shift, valid, parts, dtype, grads, activation=True, segments=None):
    """`conv`'s pieces and its gradients for `x`, the taps and the bias under `grads`."""
    def loss(x, weights, shift):
        out = conv(x, weights, activation, shift, valid=valid, segments=segments, parts=parts, dtype=dtype)
        return sum(jnp.sum(piece.astype(jnp.float32) * grad.astype(jnp.float32))
                   for piece, grad in zip(out, grads, strict=True)), out

    (_, out), found = jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True)(x, weights, shift)
    return out, found


def kernel_conv(x, weights, activation, shift, *, valid, segments, parts, dtype):
    return short_conv.conv(x, weights, shift, None if valid is None else valid.astype(jnp.int8), segments,
                           parts, activation=activation, eps=1e-6, dtype=jnp.dtype(dtype),
                           interpret=jax.default_backend() != "gpu")


def truth(x, weights, shift, valid, parts, grads, activation=True, segments=None):
    """The float64 conv and its gradients, the same cotangents widened."""
    with jax.enable_x64(new_val=True):
        wide = jax.tree.map(lambda a: None if a is None else jnp.asarray(a, jnp.float64), (x, weights, shift))
        return run(xla_conv1d, *wide, valid, parts, jnp.float64,
                   tuple(jnp.asarray(grad, jnp.float64) for grad in grads), activation, segments)


def assert_follows_xla(got, want, exact, label):
    """A piece or gradient as XLA computes it: a bf16 one rounded where XLA rounds it, apart only where
    their fp32 sums straddle a rounding (`assert_rounds_where_the_reference_does`), an fp32 one within
    fp32 reassociation of the largest value."""
    assert got.dtype == want.dtype, (label, got.dtype, want.dtype)
    if want.dtype == jnp.bfloat16:
        assert_rounds_where_the_reference_does(got, want, exact, label)
        return
    apart, scale = (float(np.abs(np.asarray(a, np.float64)).max()) for a in (got - want, exact))
    assert within(apart, 1e-5 * scale), (label, apart, scale)


@pytest.mark.parametrize("dtype, out", [(jnp.bfloat16, jnp.bfloat16), (jnp.bfloat16, jnp.float32),
                                         (jnp.float32, jnp.float32)])
@pytest.mark.parametrize("shape, parts, taps, bias, valid, packed", [
    ((2, 37), DELTA, 4, False, False, False),
    ((2, 37), DELTA, 4, False, True, False),
    ((2, 37), DELTA, 4, False, False, True),
    ((1, 700), ((64, None),), 4, True, False, False),
    ((1, 700), ((64, None),), 4, True, False, True),
    ((2, 19), ((32, 32), (32, 32), (64, None)), 3, True, True, False),
    ((2, 19), ((32, 32), (32, 32), (64, None)), 3, True, True, True),
    ((2, 9), ((48, None),), 2, True, False, False),
])
def test_the_kernels_compute_causal_conv1d(dtype, out, shape, parts, taps, bias, valid, packed):
    """Each piece and the gradients of the projection, the taps and the bias, as XLA computes them:
    bf16 values rounded where XLA rounds them, fp32 ones within fp32 reassociation, at lengths that
    leave runs and a tile partly past the end, and over packed documents that start inside a window."""
    x, weights, shift, kept = operands(*shape, parts, taps=taps, dtype=dtype, bias=bias, valid=valid)
    segments = documents(*shape) if packed else None
    grads = cotangents(x, parts, out)
    got, got_grads = run(kernel_conv, x, weights, shift, kept, parts, out, grads, segments=segments)
    want, want_grads = run(xla_conv1d, x, weights, shift, kept, parts, out, grads, segments=segments)
    exact, exact_grads = truth(x, weights, shift, kept, parts, grads, segments=segments)
    for number, (piece, reference, value) in enumerate(zip(got, want, exact, strict=True)):
        assert piece.dtype == jnp.dtype(out)
        assert_follows_xla(piece, reference, value, f"piece {number}")
    for name, grad, reference, value in zip(("x", "taps", "bias"), got_grads, want_grads, exact_grads,
                                            strict=True):
        if grad is not None:
            assert_follows_xla(grad, reference, value, f"d{name}")


def test_padding_tokens_are_zero_and_pass_no_gradient():
    """A right-padded row's padding slots come out zero and their cotangents reach nothing."""
    x, weights, _, kept = operands(2, 24, DELTA, valid=True)
    grads = cotangents(x, DELTA, jnp.bfloat16)
    out, (dx, _, _) = run(kernel_conv, x, weights, None, kept, DELTA, jnp.bfloat16, grads)
    for piece in out:
        assert not jnp.any(piece[1, 12:])
    moved = tuple(grad.at[1, 12:].set(7.0) for grad in grads)
    _, (moved_dx, _, _) = run(kernel_conv, x, weights, None, kept, DELTA, jnp.bfloat16, moved)
    assert jnp.array_equal(dx, moved_dx)


def test_a_gpu_runs_the_kernels_both_ways():
    """`causal_conv1d` on a GPU that runs Triton is the two kernels, in reverse mode too; forward mode
    and a mesh-split operand keep XLA's."""
    x, weights, _, _ = operands(1, 16, DELTA)

    def loss(x):
        return sum(jnp.sum(piece.astype(jnp.float32)) for piece in causal_conv1d(x, weights, parts=DELTA))

    def kernels(fn, *args):
        text = str(jax.make_jaxpr(fn)(*args))
        return {name for name in ("short_conv_forward", "short_conv_backward") if name in text}

    with generation.assumed("sm80"):
        assert kernels(jax.grad(loss), x) == {"short_conv_forward", "short_conv_backward"}
        assert not kernels(lambda x: jax.jvp(loss, (x,), (x,)), x)
    with generation.assumed("cpu"):
        assert not kernels(jax.grad(loss), x)


@pytest.mark.parametrize("dtype, out, parts, taps, reason", [
    (jnp.float64, jnp.float64, DELTA, 4, "float64"),
    (jnp.float32, jnp.float64, DELTA, 4, "into float64"),
    (jnp.bfloat16, jnp.bfloat16, ((48, 48), (48, None)), 4, "its block"),
    (jnp.bfloat16, jnp.bfloat16, ((32, 16), (40, 32)), 4, "its block"),
    # One key head of 16 and a value head of 8: no 16-channel block holds the values.
    (jnp.bfloat16, jnp.bfloat16, ((16, 16), (16, 16), (8, None)), 4, "its block"),
    (jnp.bfloat16, jnp.bfloat16, DELTA, 12, "12 taps"),
])
def test_the_kernels_name_what_they_do_not_take(dtype, out, parts, taps, reason):
    with jax.enable_x64(new_val=jnp.float64 in (dtype, out)):
        x = jnp.zeros((1, 8, sum(width for width, _ in parts)), dtype)
        with generation.assumed("sm80"):
            refused = conv_refusal(x, jnp.zeros((x.shape[-1], taps), jnp.float32), parts, out)
    assert refused is not None and reason in refused


TRAINING_SHAPES = {
    # Qwen3.5-0.8B: 16 key and 16 value heads of 128.
    "0.8b": ((8, 3072), ((2048, 128), (2048, 128), (2048, None))),
    # Qwen3.5-9B: 16 key heads, 32 value heads.
    "9b": ((4, 3072), ((2048, 128), (2048, 128), (4096, None))),
}


@pytest.mark.gpu(reason="the training shapes' float64 distances are a GPU's")
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("model", sorted(TRAINING_SHAPES))
def test_the_kernels_are_as_exact_as_xla_at_the_training_shapes(model, dtype, packed):
    """A delta net's conv at its training width and length, bf16 projections into the bf16 operands
    fla's rule takes, and fp32 into fp32, whole rows and packed ones: each piece and gradient as exact as
    XLA's own conv (`assert_as_exact_as_the_reference`). Two rows: a row's tokens are the distances'
    terms, and the 0.8b's eight held 28 GiB of an H100 at the float64 truth, two 7."""
    (_, length), parts = TRAINING_SHAPES[model]
    x, weights, _, _ = operands(2, length, parts, dtype=dtype)
    segments = documents(2, length) if packed else None
    grads = cotangents(x, parts, dtype)
    got, got_grads = run(kernel_conv, x, weights, None, None, parts, dtype, grads, segments=segments)
    want, want_grads = run(xla_conv1d, x, weights, None, None, parts, dtype, grads, segments=segments)
    exact, exact_grads = truth(x, weights, None, None, parts, grads, segments=segments)
    labels = ("q", "k", "v", "dx", "dtaps")
    # The gradients of the projection and the taps; there is no bias.
    for label, mine, theirs, value in zip(labels, (*got, *got_grads[:2]), (*want, *want_grads[:2]),
                                          (*exact, *exact_grads[:2]), strict=True):
        assert_as_exact_as_the_reference(mine, theirs, value, f"{model} {label}")


def training_cases():
    """Both kernels, through the conv's VJP, at each training shape: a delta net's bf16 conv into bf16
    pieces over right-padded rows and over packed ones, and Mamba's fp32 output with a bias and no norm
    over whole rows and packed ones."""
    def case(name, shape, parts, out, bias, valid, packed):
        def call():
            x = jax.ShapeDtypeStruct((*shape, sum(width for width, _ in parts)), jnp.bfloat16)
            weights = jax.ShapeDtypeStruct((x.shape[-1], 4), jnp.float32)
            shift = jax.ShapeDtypeStruct((x.shape[-1],), jnp.float32) if bias else None
            kept = jax.ShapeDtypeStruct(shape, jnp.int8) if valid else None
            segments = jax.ShapeDtypeStruct(shape, jnp.int32) if packed else None

            def grads(x, weights, shift, kept, segments):
                def loss(x, weights, shift):
                    out_ = short_conv.conv(x, weights, shift, kept, segments, parts, activation=True,
                                           eps=1e-6, dtype=jnp.dtype(out), interpret=False)
                    return sum(jnp.sum(piece.astype(jnp.float32)) for piece in out_)
                return jax.grad(loss, argnums=(0, 1, 2))(x, weights, shift)
            return grads, (x, weights, shift, kept, segments)
        return KernelCase(name, call)

    for model, (shape, parts) in sorted(TRAINING_SHAPES.items()):
        yield case(f"{model}/delta", shape, parts, jnp.bfloat16, bias=False, valid=True, packed=False)
        yield case(f"{model}/packed", shape, parts, jnp.bfloat16, bias=False, valid=False, packed=True)
    for packed in (False, True):
        yield case("mamba/packed" if packed else "mamba", (8, 3072), ((3072, None),), jnp.float32, bias=True,
                   valid=False, packed=packed)


@pytest.mark.parametrize("target", ["sm80", "sm89", "sm120"])
def test_the_kernels_compile_on_tritons_compiler_within_each_gpus_block(target):
    """Each training case lowered as on the GPU and compiled by pip's Triton for its capability on this
    CPU (tests/kernel_compile.py), within the compile bound and the block's shared memory."""
    pytest.importorskip("triton")
    found = compile_kernels(training_cases(), target)
    assert len(found) == 2 * len(TRAINING_SHAPES) + 2
    failures = [f"{record['name']}: {record['error']}" if "error" in record else
                f"{record['name']}: {record['shared']} shared bytes, limit {record['limit']}"
                for record in found if "error" in record or not record["shared"] <= record["limit"]]
    assert not failures, "\n".join(failures)


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("axis", ["data", "fsdp"])
def test_the_conv_keeps_the_kernel_contract_on_a_mesh_that_splits_rows(axis):
    """`causal_conv1d` over rows split two ways, as data or fsdp parallelism splits them, runs per shard
    (`sharded.assert_runs_per_shard`): its pieces, the l2-normalised queries and keys of a delta rule
    over padded rows of packed documents, and the gradients of the input, the taps and the bias within
    the float64 gate of one device's, and no operand moved between devices. On a GPU each shard is the
    fused kernel."""
    from jax.sharding import Mesh, PartitionSpec
    from sharded import assert_runs_per_shard

    x, weights, shift, valid = operands(4, 48, DELTA, dtype=jnp.float32, bias=True, valid=True)
    segments = documents(4, 48)
    grads = cotangents(x, DELTA, jnp.float32)

    def gradients(x, weights, shift, valid, segments, *grads):
        return run(causal_conv1d, x, weights, shift, valid, DELTA, jnp.float32, grads, segments=segments)

    expected = jax.tree.map(np.asarray, truth(x, weights, shift, valid, DELTA, grads, segments=segments))
    mesh = Mesh(np.asarray(jax.devices()[:2]), (axis,))
    rows, whole = PartitionSpec(axis), PartitionSpec()
    assert_runs_per_shard(gradients, (x, weights, shift, valid, segments, *grads),
                          [rows, whole, whole, rows, rows, rows, rows, rows], mesh, expected)
