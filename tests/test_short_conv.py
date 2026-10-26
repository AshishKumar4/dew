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
import tokamax
from fla_gdn_support import CONV_SHAPES
from reference_error import (
    assert_as_exact_as_the_reference,
    assert_rounds_where_the_reference_does,
    equations,
    with_gradients,
    within,
)
from tuned import each_config, launches, timed

from dew.nn.kernels import generation, short_conv, tuning
from dew.nn.linear import causal_conv1d, conv_refusal, xla_conv1d

pytestmark = pytest.mark.kernel

DELTA = ((32, 16), (32, 16), (32, None))
SHARDED_SEEDS = 8
"""The per-shard contract's declared draws (check 1, `sharded.assert_runs_per_shard`), sized by
`seeds_for`: the fewest for which one device compared with itself resolves the factor."""
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


def configs(op) -> list:
    """Every config of `op`'s space, as its key declares it for the CPU tests' delta conv
    (`each_config` reads them from a lowering, which a CPU's interpreted kernels describe too)."""
    x, weights, _, _ = operands(2, 37, DELTA)
    grads = cotangents(x, DELTA, jnp.bfloat16)
    return each_config(op, lambda x, weights: run(kernel_conv, x, weights, None, None, DELTA, jnp.bfloat16,
                                                  grads), x, weights)


def tuned_step():
    """A right-padded delta conv's pieces and gradients as one jitted step, its operands, and the
    validity and cotangents it closes over."""
    x, weights, _, kept = operands(2, 37, DELTA, valid=True)
    grads = cotangents(x, DELTA, jnp.bfloat16)
    step = jax.jit(lambda x, weights: run(kernel_conv, x, weights, None, kept, DELTA, jnp.bfloat16, grads))
    return step, (x, weights), kept, grads


def test_the_tuner_times_each_conv_launch_on_real_tokens():
    """Each launch the tuner reads from the conv's step is timed on rows whose every token is real
    (`KernelOp.operands`), so every token's output and the taps' gradients are written, where the
    default operands' zero validity masks every token: a timing of work the step does."""
    step, arrays, _, _ = tuned_step()
    found = launches(step, *arrays)
    assert found and all(isinstance(ba.op, tuning.KernelOp) for ba in found)
    for ba in found:
        written = timed(ba) if type(ba.op) is short_conv.ConvForward else timed(ba)[1:2]
        for array in written:
            assert np.all(np.any(np.asarray(array, np.float32) != 0, axis=-1)), type(ba.op).__name__


TUNED = """
import json, os, sys
import jax
import numpy as np
from test_short_conv import tuned_step
from tuned import launches
from dew import cache
from dew.nn.kernels import tuning
step, arrays, _, _ = tuned_step()
lowered, launched = tuning.tune(lambda: step.lower(*arrays), measure=sys.argv[1] == "measure")
pieces, grads = lowered.compile()(*arrays)
np.savez(sys.argv[2], *(np.asarray(array, np.float32) for array in (*pieces, *grads[:2])))
path = cache.kernel_picks_path(jax.devices()[0].device_kind)
taken = [repr(ba.op.chosen(ba)) for ba in launches(step, *arrays)]
print(json.dumps([path, os.stat(path).st_mtime_ns, tuning.settled(launched), taken]))
"""
"""A process tuning the conv's step, measuring where its first argument says so, its pieces and gradients
saved to its second: the store's path and time, whether every launch is picked, and each launch's config."""


@pytest.mark.gpu(reason="the tuner measures on a GPU")
def test_a_conv_step_launches_the_configs_its_store_holds_as_exact_as_xla(tmp_path):
    """The tuner's explicit pass over a right-padded delta conv's step (`tuning.tune(..., measure=True)`,
    on a GPU the process has to itself) measures each launch over every config of its space, stores the
    fastest and launches it. A fresh process beside other work, which measures nothing
    (AUTOTUNE_CACHE_MODE_READ), finds every launch picked, leaves the store as it was, launches the
    stored configs and computes the same pieces and gradients bitwise; and where the store holds a
    config of each launch's space other than its heuristic, as another pass may leave, the next process
    launches that. Each process's pieces and gradients are as exact as XLA's conv, from float64."""
    from process_support import READ_ONLY, fresh, keeping_picks

    inherited = keeping_picks()
    if READ_ONLY in inherited:
        pytest.skip("measuring needs the GPU to itself: gpu-run without -n")
    cache_dir, reading = tmp_path / "cache", f"{inherited} {READ_ONLY}".strip()
    step, (x, weights), kept, grads = tuned_step()
    keyed = [ba.op.keyed(ba) for ba in launches(step, x, weights)]
    distinct = {(type(ba.op), ba.autotuning_cache_key): ba for ba in keyed}

    def taking(configs) -> list[str]:
        """The config each launch of the step takes, from each launch's by key."""
        return [repr(configs[type(ba.op), ba.autotuning_cache_key]) for ba in keyed]

    path, stamp, settled, taken = fresh(TUNED, "measure", str(tmp_path / "measured.npz"),
                                        cache_dir=cache_dir, flags=inherited)
    stored = {(type(ba.op), ba.autotuning_cache_key): timings
              for ba, timings in tuning._read(path)[0].values()}
    assert settled is True and stored.keys() == distinct.keys()
    for key, timings in stored.items():
        assert len(timings) > 1, key[0]
        assert all(data.median_evaluation_time_ms > 0 for data in timings.values())
        assert timings.fastest_config in distinct[key].autotuning_configs
    fastest = {key: timings.fastest_config for key, timings in stored.items()}
    assert taken == taking(fastest)
    assert fresh(TUNED, "read", str(tmp_path / "read.npz"), cache_dir=cache_dir,
                 flags=reading) == [path, stamp, True, taken]
    with np.load(tmp_path / "measured.npz") as first, np.load(tmp_path / "read.npz") as second:
        for name in first.files:
            np.testing.assert_array_equal(second[name], first[name])

    # A config of each space other than its heuristic, stored as the tuner stores its picks.
    chosen = {key: next(config for config in sorted(ba.autotuning_configs, key=repr)
                        if config != ba.heuristics_config) for key, ba in distinct.items()}
    timing = tokamax.BenchmarkData(compile_time_ms=1.0, lower_time_ms=1.0, evaluation_times_ms=(1.0,),
                                   metadata={})
    with open(path, "w") as file:
        file.write(tuning.dumps_picks(jax.devices()[0].device_kind, tuple(
            (ba, tuning.Timings({chosen[key]: timing})) for key, ba in distinct.items())))
    _, _, settled, taken = fresh(TUNED, "read", str(tmp_path / "chosen.npz"), cache_dir=cache_dir,
                                 flags=reading)
    assert settled is True and taken == taking(chosen)

    want, want_grads = run(xla_conv1d, x, weights, None, kept, DELTA, jnp.bfloat16, grads)
    exact, exact_grads = truth(x, weights, None, kept, DELTA, grads)
    labels = [f"piece {number}" for number in range(len(want))] + ["dx", "dtaps"]
    for run_name in ("read", "chosen"):
        with np.load(tmp_path / f"{run_name}.npz") as saved:
            got = [saved[name] for name in saved.files]
        for label, mine, theirs, value in zip(labels, got, (*want, *want_grads[:2]),
                                              (*exact, *exact_grads[:2]), strict=True):
            assert_as_exact_as_the_reference(jnp.asarray(mine, theirs.dtype), theirs, value,
                                             f"{run_name} {label}")


@pytest.mark.parametrize("op", [short_conv.ConvForward, short_conv.ConvBackward])
def test_every_config_of_the_kernels_computes_causal_conv1d(op):
    """Each config of each launch's space (`tuning.pinned`) gives each piece and the gradients of the
    projection and the taps as XLA computes them, over packed documents at a length that leaves runs and
    a tile partly past the end: the configs order the tap gradients' sums, and every one is within the
    gate the heuristic is."""
    x, weights, _, _ = operands(2, 37, DELTA)
    segments = documents(2, 37)
    grads = cotangents(x, DELTA, jnp.bfloat16)
    want, want_grads = run(xla_conv1d, x, weights, None, None, DELTA, jnp.bfloat16, grads, segments=segments)
    exact, exact_grads = truth(x, weights, None, None, DELTA, grads, segments=segments)
    found = configs(op)
    assert len(found) > 1
    for config in found:
        with tuning.pinned({op: config}):
            got, got_grads = run(kernel_conv, x, weights, None, None, DELTA, jnp.bfloat16, grads,
                                 segments=segments)
        for label, mine, theirs, value in zip(("q", "k", "v", "dx", "dtaps"), (*got, *got_grads[:2]),
                                              (*want, *want_grads[:2]), (*exact, *exact_grads[:2]),
                                              strict=True):
            assert_follows_xla(mine, theirs, value, f"{config} {label}")


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
        return {str(equation.params["name"]) for equation in equations(jax.make_jaxpr(fn)(*args))
                if equation.primitive.name == "pallas_call"}

    with generation.assumed("sm80"):
        assert kernels(jax.grad(loss), x) == {"short_conv_forward", "short_conv_backward"}
        assert not kernels(lambda x: jax.jvp(loss, (x,), (x,)), x)
    with generation.assumed("cpu"):
        assert not kernels(jax.grad(loss), x)


@pytest.mark.parametrize("dtype, out, parts, taps", [
    (jnp.float64, jnp.float64, DELTA, 4),
    (jnp.float32, jnp.float64, DELTA, 4),
    (jnp.bfloat16, jnp.bfloat16, ((48, 48), (48, None)), 4),
    # One key head of 16 and a value head of 8: no 16-channel block holds the values.
    (jnp.bfloat16, jnp.bfloat16, ((16, 16), (16, 16), (8, None)), 4),
    (jnp.bfloat16, jnp.bfloat16, DELTA, 12),
])
def test_a_conv_the_kernels_do_not_take_runs_as_the_xla_conv(dtype, out, parts, taps):
    """On a GPU whose row names the kernels, a conv they do not take (`conv_refusal`: float64, a piece
    off its block, more taps than they hold) runs XLA's conv: the same values and gradients, and no
    kernel launch."""
    with jax.enable_x64(new_val=jnp.float64 in (dtype, out)):
        keys = jax.random.split(jax.random.key(89), 3)
        x = jax.random.normal(keys[0], (1, 8, sum(width for width, _ in parts))).astype(dtype)
        taps = 0.3 * jax.random.normal(keys[1], (x.shape[-1], taps))
        probe = [jax.random.normal(key, (1, 8, width)) for key, (width, _) in zip(
            jax.random.split(keys[2], len(parts)), parts, strict=True)]

        def loss(conv):
            def run(x, taps):
                return sum(jnp.sum(piece.astype(jnp.float32) * weight) for piece, weight in zip(
                    conv(x, taps, parts=parts, dtype=out), probe, strict=True))
            return jax.value_and_grad(run, argnums=(0, 1))

        with generation.assumed("sm80"):
            assert conv_refusal(x, taps, parts, out) is not None
            launched = [equation for equation in equations(jax.make_jaxpr(loss(causal_conv1d))(x, taps))
                        if equation.primitive.name == "pallas_call"]
            got = jax.jit(loss(causal_conv1d))(x, taps)
        want = jax.jit(loss(xla_conv1d))(x, taps)
    assert not launched
    for mine, theirs in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs))


@pytest.mark.gpu(reason="the training shapes' float64 distances are a GPU's")
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("model", sorted(CONV_SHAPES))
def test_the_kernels_are_as_exact_as_xla_at_the_training_shapes(model, dtype, packed):
    """A delta net's conv at its training width and length, bf16 projections into the bf16 operands
    fla's rule takes, and fp32 into fp32, whole rows and packed ones: each piece and gradient as exact as
    XLA's own conv (`assert_as_exact_as_the_reference`), at each launch's heuristic and at every config
    of its space at that model's shapes (`tuning.pinned`), the space the tuner picks from there. Two
    rows: a row's tokens are the distances' terms, and the 0.8b's eight held 28 GiB of an H100 at the
    float64 truth, two 7."""
    (_, length), parts = CONV_SHAPES[model]
    x, weights, _, _ = operands(2, length, parts, dtype=dtype)
    segments = documents(2, length) if packed else None
    grads = cotangents(x, parts, dtype)

    def kernels(x, weights):
        return run(kernel_conv, x, weights, None, None, parts, dtype, grads, segments=segments)

    # The reference's and the truth's leaves on the host as each is computed, and the truth a row at a time
    # (`row_truth`): the 9b's packed rows peaked at 9.2 GiB of an A100, past a packed job's 8.9 GiB share
    # where main's GPU tier ran them (row 1724, SE-conv1724-a100-fd8508b5a), 8.85 with the arrays on the
    # host alone (SE-conv1724-a100-9419b90c1).
    want, want_grads = jax.device_get(run(xla_conv1d, x, weights, None, None, parts, dtype, grads,
                                          segments=segments))
    exact, exact_grads = row_truth(x, weights, parts, grads, segments)
    pins = [{}] + [{op: config} for op in (short_conv.ConvForward, short_conv.ConvBackward)
                   for config in each_config(op, kernels, x, weights)]
    for pin in pins:
        with tuning.pinned(pin):
            got, got_grads = kernels(x, weights)
        labels = ("q", "k", "v", "dx", "dtaps")
        # The gradients of the projection and the taps; there is no bias.
        for label, mine, theirs, value in zip(labels, (*got, *got_grads[:2]), (*want, *want_grads[:2]),
                                              (*exact, *exact_grads[:2]), strict=True):
            assert_as_exact_as_the_reference(mine, theirs, value, f"{model} {pin} {label}")


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
