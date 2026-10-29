"""The short conv's kernels (dew.nn.kernels.short_conv) against `causal_conv1d`'s XLA arithmetic.

Host tests interpret the kernels at small shapes, where they hold the runs a program walks, the pieces
it writes and each gradient to `xla_conv1d`, with the float64 conv as the truth. The kernel lab's GPU
tier runs them on the GPU, with the training shapes' tests: there each output's and gradient's float64
distance is held to XLA's own by tests/reference_error.py's rule.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fla_gdn_support import CONV_SHAPES
from reference_error import (
    FACTOR,
    assert_as_exact,
    assert_as_exact_as_the_reference,
    assert_rounds_where_the_reference_does,
    chain_roundings,
    distance,
    widened,
    with_gradients,
    within,
)

from dew.nn.kernels import generation, short_conv
from dew.nn.linear import causal_conv1d, xla_conv1d

pytestmark = pytest.mark.kernel

DELTA = ((32, 16), (32, 16), (32, None))
SHARDED_SEEDS = 8
"""The per-shard contract's declared draws (check 1, `sharded.assert_runs_per_shard`), sized by
`seeds_for`: the fewest for which one device compared with itself resolves the factor."""
REFUSED_SEEDS = 8
"""The declared draws of a conv the kernels do not take against XLA's conv (check 1), sized by `seeds_for`
paired with the kernels' arm on armada: the most any of its 16-bit geometries took."""
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
    def pieces(x, weights, shift):
        return conv(x, weights, activation, shift, valid=valid, segments=segments, parts=parts, dtype=dtype)

    # Each piece pulls back its cotangent cast through float32 to the piece's dtype, as a float32 sum of
    # their products pulls it back, through the gate (`reference_error.with_gradients`).
    shapes = jax.eval_shape(pieces, x, weights, shift)
    cotangents = tuple(grad.astype(jnp.float32).astype(shape.dtype)
                       for grad, shape in zip(grads, shapes, strict=True))
    out, *found = with_gradients(pieces)(x, weights, shift, cotangents)
    return out, tuple(found)


def kernel_conv(x, weights, activation, shift, *, valid, segments, parts, dtype):
    return short_conv.conv(x, weights, shift, None if valid is None else valid.astype(jnp.int8), segments,
                           parts, activation=activation, eps=1e-6, dtype=jnp.dtype(dtype),
                           interpret=jax.default_backend() != "gpu")


@pytest.mark.parametrize("staged", ["jit", "remat"])
def test_the_kernels_take_forward_mode_however_a_caller_stages_them(staged):
    """`jax.jvp` of the conv on the kernels (interpreted off a GPU), a normalised piece among its pieces,
    staged by a jit or by a remat as a layer's `nn.remat` stages it: its tangent XLA's conv's
    (`short_conv.conv`'s `generation.linear_tangent`), on par with XLA's from float64. Its selector once
    routed by whether it saw a JVP tracer, which a remat hides, and there the kernels' custom VJP raised
    (row 2395)."""
    parts = ((64, 32), (64, None))
    x, weights, shift, _ = operands(1, 40, parts, dtype=jnp.float32, bias=True)
    tangents = tuple(jnp.asarray(np.random.default_rng(3 + index).normal(size=t.shape), t.dtype)
                     for index, t in enumerate((x, weights, shift)))

    def kernels(x, weights, shift):
        return kernel_conv(x, weights, activation=True, shift=shift, valid=None, segments=None, parts=parts,
                           dtype=x.dtype)

    def xla(x, weights, shift):
        return xla_conv1d(x, weights, activation=True, bias=shift, parts=parts, dtype=x.dtype)

    def tangent(pieces, *primals_and_tangents):
        rule = jax.jit(pieces) if staged == "jit" else jax.checkpoint(pieces)
        return jax.jvp(rule, primals_and_tangents[:3], primals_and_tangents[3:])[1]

    mine = tangent(kernels, x, weights, shift, *tangents)
    theirs = tangent(xla, x, weights, shift, *tangents)
    with jax.enable_x64(new_val=True):
        exact = tangent(xla, *widened((x, weights, shift, *tangents)))
    for index, (found, reference, value) in enumerate(zip(mine, theirs, exact, strict=True)):
        assert_as_exact_as_the_reference(found, reference, value, f"{staged} piece {index}")


def truth(x, weights, shift, valid, parts, grads, activation=True, segments=None):
    """The float64 conv and its gradients, the same cotangents widened."""
    with jax.enable_x64(new_val=True):
        wide = jax.tree.map(lambda a: None if a is None else jnp.asarray(a, jnp.float64), (x, weights, shift))
        return run(xla_conv1d, *wide, valid, parts, jnp.float64,
                   tuple(jnp.asarray(grad, jnp.float64) for grad in grads), activation, segments)


def row_truth(x, weights, parts, grads, segments):
    """`truth`'s pieces and gradients of `x` and the taps, on the host, a row at a time: each row's pieces and
    input gradient its own, the taps' gradient the rows' summed in float64; a row's float64 conv the device
    holds at once, not the batch's."""
    rows = [jax.device_get(truth(x[i:i + 1], weights, None, None, parts,
                                 tuple(grad[i:i + 1] for grad in grads),
                                 segments=None if segments is None else segments[i:i + 1]))
            for i in range(x.shape[0])]
    pieces = tuple(np.concatenate([row[0][k] for row in rows]) for k in range(len(rows[0][0])))
    dx = np.concatenate([row[1][0] for row in rows])
    dtaps = functools.reduce(np.add, (np.asarray(row[1][1], np.float64) for row in rows))
    return pieces, (dx, dtaps, *rows[0][1][2:])


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


@pytest.mark.parametrize("dtype, out, parts, taps", [
    (jnp.float64, jnp.float64, DELTA, 4),
    (jnp.float32, jnp.float64, DELTA, 4),
    (jnp.bfloat16, jnp.bfloat16, ((48, 48), (48, None)), 4),
    # One key head of 16 and a value head of 8: no 16-channel block holds the values.
    (jnp.bfloat16, jnp.bfloat16, ((16, 16), (16, 16), (8, None)), 4),
    (jnp.bfloat16, jnp.bfloat16, DELTA, 12),
])
def test_a_conv_the_kernels_do_not_take_runs(dtype, out, parts, taps):
    """On a GPU whose row names the kernels, a conv they do not take (`short_conv.CONV`: float64, a piece
    off its block, more taps than they hold) runs: its value and its gradients, in their operands'
    dtypes, on par with XLA's conv over the draws (check 1), truth XLA's conv in float64. A float64 conv
    has no truth finer than itself, so it is held by the oracle rule on a draw: each leaf within its own
    dtype's rounding over the conv's chain of roundings of XLA's conv on the inputs widened."""
    width, channels = sum(width for width, _ in parts), taps

    def draw(seed):
        keys = jax.random.split(jax.random.fold_in(jax.random.key(89), seed), 3)
        x = jax.random.normal(keys[0], (1, 8, width)).astype(dtype)
        weights = 0.3 * jax.random.normal(keys[1], (width, channels))
        probe = tuple(jax.random.normal(key, (1, 8, piece)) for key, (piece, _) in zip(
            jax.random.split(keys[2], len(parts)), parts, strict=True))
        return x, weights, probe

    def loss(conv, out=out):
        def run(x, weights, probe):
            def scored(x, weights):
                return sum(jnp.sum(piece.astype(weight.dtype) * weight) for piece, weight in zip(
                    conv(x, weights, parts=parts, dtype=out), probe, strict=True))
            return jax.value_and_grad(scored, argnums=(0, 1))(x, weights)
        return run

    with jax.enable_x64(new_val=jnp.float64 in (dtype, out)), generation.assumed("sm80"):
        x, weights, probe = draw(0)
        _, gradients = jax.jit(loss(causal_conv1d))(x, weights, probe)
        assert [gradient.dtype for gradient in gradients] == [x.dtype, weights.dtype]
        if out == jnp.float64:
            found = jax.jit(loss(causal_conv1d))(x, weights, probe)
            oracle = loss(xla_conv1d)(*widened((x, weights, probe)))
            roundings = chain_roundings(jax.make_jaxpr(loss(causal_conv1d))(x, weights, probe))
            for mine, exact in zip(jax.tree.leaves(found), jax.tree.leaves(oracle), strict=True):
                scale = float(np.sqrt(np.mean(np.square(np.asarray(exact, np.float64)))))
                bound = FACTOR * roundings * float(jnp.finfo(mine.dtype).eps) * scale
                assert within(distance(mine, exact), bound), (distance(mine, exact), bound)
            return
        assert_as_exact(jax.jit(loss(causal_conv1d)), jax.jit(loss(xla_conv1d)), draw,
                        f"a conv the kernels do not take: {jnp.dtype(dtype)}, {parts}, {taps} taps",
                        seeds=REFUSED_SEEDS, truth_fn=loss(xla_conv1d, out=jnp.float64))


TRAINING_SEEDS = {(model, dtype, packed): 8 for model in CONV_SHAPES for dtype in ("bfloat16", "float32")
                  for packed in (False, True)}
"""The training shapes' declared draws against XLA's conv (check 1), by model, dtype and packing: sized by
`seeds_for` paired with the kernels on a GPU."""


@pytest.mark.gpu(reason="the training shapes' float64 distances are a GPU's")
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("model", sorted(CONV_SHAPES))
def test_the_kernels_are_as_exact_as_xla_at_the_training_shapes(model, dtype, packed):
    """A delta net's conv at its training width and length, bf16 projections into the bf16 operands
    fla's rule takes, and fp32 into fp32, whole rows and packed ones: each piece and the gradients of the
    projection and the taps on par (check 1) with XLA's own conv from float64, at each launch's config at
    that model's shapes, over drawn rows, taps and cotangents. Two rows: a row's tokens are the distances'
    terms, and the 0.8b's eight held 28 GiB of an H100 at the float64 truth, two 7."""
    (_, length), parts = CONV_SHAPES[model]
    segments = documents(2, length) if packed else None

    def draw(seed):
        x, weights, _, _ = operands(2, length, parts, dtype=dtype, seed=[1724, seed])
        return x, weights, cotangents(x, parts, dtype, seed=[1725, seed])

    def conv(fn):
        def pieces_and_gradients(x, weights, grads):
            # The gradients of the projection and the taps; there is no bias.
            out, found = run(fn, x, weights, None, None, parts, dtype, grads, segments=segments)
            return out, found[:2]
        return pieces_and_gradients

    def exact(x, weights, grads):
        # A row at a time on the host (`row_truth`): the 9b's packed rows peaked at 9.2 GiB of an A100, past
        # a packed job's 8.9 GiB share where main's GPU tier ran them (row 1724,
        # SE-conv1724-a100-fd8508b5a), 8.85 with the arrays on the host alone (SE-conv1724-a100-9419b90c1).
        pieces, gradients = row_truth(x, weights, parts, grads, segments)
        return pieces, gradients[:2]

    assert_as_exact(conv(kernel_conv), conv(xla_conv1d), draw,
                    f"{model} {jnp.dtype(dtype).name}{' packed' if packed else ''}",
                    seeds=TRAINING_SEEDS[model, jnp.dtype(dtype).name, packed], truth_fn=exact)


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("axis", ["data", "fsdp"])
def test_the_conv_keeps_the_kernel_contract_on_a_mesh_that_splits_rows(axis):
    """`causal_conv1d` over rows split two ways, as data or fsdp parallelism splits them, runs per shard
    (`sharded.assert_runs_per_shard`): its pieces, the l2-normalised queries and keys of a delta rule
    over padded rows of packed documents, and the gradients of the input, the taps and the bias on par
    with one device's over the site's draws, and no operand moved between devices. On a GPU each shard
    is the fused kernel."""
    from jax.sharding import Mesh, PartitionSpec
    from sharded import assert_runs_per_shard

    def draw(seed):
        # Each role's stream its own: a draw's cotangents never replay another draw's input.
        x, weights, shift, valid = operands(4, 48, DELTA, dtype=jnp.float32, bias=True, valid=True,
                                            seed=3 * seed)
        return (x, weights, shift, valid, documents(4, 48, seed=3 * seed + 2),
                *cotangents(x, DELTA, jnp.float32, seed=3 * seed + 1))

    def gradients(x, weights, shift, valid, segments, *grads):
        return run(causal_conv1d, x, weights, shift, valid, DELTA, jnp.float32, grads, segments=segments)

    def exact(x, weights, shift, valid, segments, *grads):
        return truth(x, weights, shift, valid, DELTA, grads, segments=segments)

    mesh = Mesh(np.asarray(jax.devices()[:2]), (axis,))
    rows, whole = PartitionSpec(axis), PartitionSpec()
    assert_runs_per_shard(gradients, draw, [rows, whole, whole, rows, rows, rows, rows, rows], mesh,
                          f"conv split over {axis}", seeds=SHARDED_SEEDS, truth_fn=exact)
