"""The float64 replay of a float32 program (tests/reference_error.py, `float64_replay`): a draw's
replay is its own bits' transform in float64, a program of bits replays to itself, the replay reads
the float32 program whatever its caller's mode, and an operation it cannot replay is refused."""

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from float64_replay import float64_replay
from reference_error import assert_as_exact


def _widened_uniform(key, shape, lo, hi):
    """JAX's uniform written out over the key's own bits: the mantissa trick in uint32 and the
    bitcast, exact, then its shift into [lo, hi) in float64 (`jax._src.random.core._uniform`)."""
    bits = jax.random.bits(key, shape, jnp.uint32)
    floats = jax.lax.bitcast_convert_type((bits >> np.uint32(9)) | np.uint32(0x3F800000), jnp.float32)
    with jax.enable_x64(new_val=True):
        lo, hi = jnp.float64(lo), jnp.float64(hi)
        return jnp.maximum(lo, (floats.astype(jnp.float64) - 1.0) * (hi - lo) + lo)


def _widened_draw(name, key, shape):
    """`jax.random`'s float32 draw `name` written out in float64 over the same bits, its constants
    the program's float32 ones widened."""
    sqrt2 = np.float64(np.float32(np.sqrt(2)))
    if name == "uniform":
        return _widened_uniform(key, shape, 0.0, 1.0)
    if name == "normal":
        u = _widened_uniform(key, shape, np.nextafter(np.float32(-1), np.float32(0)), 1.0)
        with jax.enable_x64(new_val=True):
            return sqrt2 * jax.lax.erf_inv(u)
    with jax.enable_x64(new_val=True):
        a, b = (jax.lax.erf(jnp.float64(bound) / sqrt2) for bound in (-2.0, 2.0))
    u = _widened_uniform(key, shape, a, b)
    with jax.enable_x64(new_val=True):
        drawn = sqrt2 * jax.lax.erf_inv(u)
        low, high = (np.float64(np.nextafter(np.float32(bound), np.float32(toward)))
                     for bound, toward in ((-2.0, np.inf), (2.0, -np.inf)))
        return jnp.minimum(high, jnp.maximum(low, drawn))


DRAWS = {"uniform": lambda key: jax.random.uniform(key, (4, 97), jnp.float32),
         "normal": lambda key: jax.random.normal(key, (4, 97), jnp.float32),
         "truncated_normal": lambda key: jax.random.truncated_normal(key, -2.0, 2.0, (4, 97), jnp.float32)}


@pytest.mark.parametrize("name", sorted(DRAWS))
def test_a_float32_draws_replay_is_its_bits_transform_in_float64(name):
    """A float32 draw's float64 replay is the same random bits' uniform, exact, through the draw's
    transform in float64, written out directly: bit for bit, for several keys."""
    for seed in range(3):
        key = jax.random.key(seed)
        replayed = np.asarray(float64_replay(DRAWS[name])(key))
        direct = np.asarray(_widened_draw(name, key, (4, 97)))
        assert replayed.dtype == np.float64
        np.testing.assert_array_equal(replayed.view(np.uint64), direct.view(np.uint64))


def _integer_jvp():
    @jax.custom_jvp
    def shifted(bits):
        return bits >> np.uint32(3)

    shifted.defjvp(lambda primals, tangents: (shifted(*primals), tangents[0]))
    return shifted


BIT_PROGRAMS = {
    "draw": lambda key: jax.random.bits(jax.random.fold_in(key, 3), (8, 5), jnp.uint32) >> np.uint32(3),
    # Traced under its own Threefry setting, which the replay restores for each equation.
    "unpartitioned": lambda key: _unpartitioned_bits(key),
    "custom_jvp": lambda key: _integer_jvp()(jax.random.bits(key, (8, 5), jnp.uint32)),
}


def _unpartitioned_bits(key):
    with jax.threefry_partitionable(new_val=False):
        return jax.random.bits(key, (8, 5), jnp.uint32)


@pytest.mark.parametrize("name", sorted(BIT_PROGRAMS))
def test_a_program_of_bits_replays_to_itself(name):
    """A program with no float in it, a custom derivative's integer call or a draw under its own
    Threefry setting among them, replays to its own result, bit for bit."""
    key = jax.random.key(5)
    np.testing.assert_array_equal(float64_replay(BIT_PROGRAMS[name])(key), BIT_PROGRAMS[name](key))


def test_a_replay_reads_the_float32_program_whatever_its_callers_mode():
    """Check 1 hands its truth widened arguments under x64; the replay still reads the float32
    program the caller runs, its arguments rounded to their traced dtypes, so its truth is the same
    bit for bit as called plainly."""
    x = np.random.default_rng(1).uniform(0.5, 2.0, (6, 7)).astype(np.float32)
    plain = np.asarray(float64_replay(_arithmetic)(x))
    with jax.enable_x64(new_val=True):
        widened_call = np.asarray(float64_replay(_arithmetic)(x.astype(np.float64)))
        drawn = np.asarray(float64_replay(DRAWS["normal"])(jax.random.key(2)))
    np.testing.assert_array_equal(widened_call.view(np.uint64), plain.view(np.uint64))
    direct = np.asarray(_widened_draw("normal", jax.random.key(2), (4, 97)))
    np.testing.assert_array_equal(drawn.view(np.uint64), direct.view(np.uint64))
    def seeded(seed):
        return DRAWS["normal"](jax.random.key(seed))

    assert_as_exact(seeded, seeded, lambda seed: (np.int32(seed),), "normal draw", seeds=8,
                    truth_fn=float64_replay(seeded))


def _arithmetic(x):
    return jnp.exp(jnp.sqrt(x * 0.5 + 2.0)) - jnp.log(x + 3.0) / 4.0


@jax.custom_vjp
def _custom(x):
    return _arithmetic(x)


_custom.defvjp(lambda x: (_custom(x), x), lambda x, cotangent: (cotangent,))

PROGRAMS = {
    "plain": _arithmetic,
    "jit": jax.jit(_arithmetic),
    "custom_vjp": _custom,
    "scan": lambda x: jax.lax.scan(lambda carry, row: (carry + _arithmetic(row), carry * row), x[0], x)[1],
}


@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_a_float32_program_replays_to_itself_run_in_float64(name):
    """A float32 program without random bits, through a jit, a custom derivative's call or a scan,
    replays to that program run in float64, bit for bit, each run an operation at a time (a
    compiled program may contract a product and a sum into one rounding)."""
    x = np.random.default_rng(0).uniform(0.5, 2.0, (6, 7)).astype(np.float32)
    replayed = np.asarray(float64_replay(PROGRAMS[name])(x))
    # A scan's body compiles in both; every other program runs an operation at a time in both.
    with jax.enable_x64(new_val=True), contextlib.nullcontext() if name == "scan" else jax.disable_jit():
        direct = np.asarray(PROGRAMS[name](jnp.asarray(x, jnp.float64)))
    np.testing.assert_array_equal(replayed.view(np.uint64), direct.view(np.uint64))


def test_what_a_programs_results_do_not_read_is_not_replayed():
    """A program whose result drops a value it computed, as a module's init drops its forward pass,
    replays without it: a float operation the replay has no rule for (`cos`) runs nowhere when no
    result reads it, inside a jit as well, and the result is its program's in float64."""
    x = np.random.default_rng(3).uniform(0.5, 2.0, (6, 7)).astype(np.float32)

    def dropping(x):
        return (jnp.cos(x), _arithmetic(x))[1]

    replayed = np.asarray(float64_replay(jax.jit(dropping))(x))
    with jax.enable_x64(new_val=True), jax.disable_jit():
        direct = np.asarray(_arithmetic(jnp.asarray(x, jnp.float64)))
    np.testing.assert_array_equal(replayed.view(np.uint64), direct.view(np.uint64))


REFUSED = {
    "float": lambda x: jnp.cos(x),
    "complex": lambda x: jnp.cos(x.astype(jnp.complex64)).real,
    # A call whose integer boundary hides float arithmetic inside.
    "hidden": lambda x: jax.lax.cond(x[0] > 0, lambda x: jnp.cos(x).astype(jnp.int32),
                                     lambda x: x.astype(jnp.int32), x),
}


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_a_float_operation_without_a_float64_rule_refuses(name):
    """A float operation the replay has no rule for refuses, never runs as traced: a complex one,
    and one inside a call it does not open, too."""
    with pytest.raises(NotImplementedError):
        float64_replay(REFUSED[name])(np.ones(3, np.float32))

