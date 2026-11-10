"""The arithmetic a check's arms and truths run in (tests/reference_error.py): the scope each arm runs in
(`summed_as_run`), the programs a reference and a truth run (`unrewritten`), the float64 truth's inputs
(`widened`) and its compiled program (`Truth`)."""

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import primitives

from dew.nn import programs
from dew.nn.precision import FP32_SUMS


def widened(tree):
    """`tree` in float64, its complex leaves in complex128 and its integer leaves as they are: the inputs
    of a float64 truth. A leaf an accelerator holds is widened where it lies, so its truth runs on that
    device's FP64 units with no host round trip (a decode step's keys and values, a hundred megabytes a
    draw to the host and back); any other is a host array."""
    def wide(leaf):
        to = (np.complex128 if jnp.issubdtype(leaf.dtype, jnp.complexfloating)
              else np.float64 if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf.dtype)
        if isinstance(leaf, jax.Array) and to != leaf.dtype and all(
                device.platform != "cpu" for device in leaf.devices()):
            with jax.enable_x64(new_val=True):
                return leaf.astype(to)
        return np.asarray(leaf, to)

    return jax.tree.map(wide, tree)


EXACT = frozenset({jax.lax.neg_p, jax.lax.abs_p, jax.lax.max_p, jax.lax.min_p, jax.lax.select_n_p,
    jax.lax.copy_p, jax.lax.iota_p, jax.lax.sign_p, jax.lax.floor_p, jax.lax.ceil_p, jax.lax.round_p,
    jax.lax.reduce_max_p, jax.lax.reduce_min_p, jax.lax.argmax_p, jax.lax.argmin_p, jax.lax.eq_p,
    jax.lax.ne_p, jax.lax.lt_p, jax.lax.le_p, jax.lax.gt_p, jax.lax.ge_p, jax.lax.and_p, jax.lax.or_p,
    jax.lax.not_p, jax.lax.xor_p, jax.lax.clamp_p, jax.lax.is_finite_p, jax.lax.stop_gradient_p,
    jax.lax.optimization_barrier_p, jax.lax.cummax_p, jax.lax.cummin_p, jax.lax.sort_p, jax.lax.top_k_p,
    jax.lax.shift_left_p,
    jax.lax.shift_right_logical_p, jax.lax.shift_right_arithmetic_p, jax.lax.population_count_p,
    jax.lax.rem_p, primitives.name_p, jax.lax.nextafter_p, primitives.random_bits_p, primitives.random_seed_p,
    primitives.random_wrap_p, primitives.random_unwrap_p, primitives.threefry2x32_p, jax.lax.scatter_p,
    jax.lax.stack_p, jax.lax.unstack_p, jax.lax.bitcast_convert_type_p, jax.lax.device_put_p,
    jax.lax.sharding_constraint_p, primitives.pvary_p, primitives.reshard_p, jax.lax.reduce_and_p,
    jax.lax.reduce_or_p, primitives.empty2_p, jax.lax.all_gather_p, primitives.all_gather_invariant_p,
    jax.lax.all_to_all_p, jax.lax.ppermute_p, primitives.pbroadcast_p, jax.lax.axis_index_p})
"""Operations whose float results are their operands' values selected, or exact: no rounding. An operation
that only moves entries is exact too (`dew.nn.programs.rearranges`, read beside these, `rounds_nothing`).
An integer result rounds nothing either, nor does an array made to be written over (`empty2`); a collective
that moves blocks between devices (a gather, an all-to-all, a permutation) is a move too. The one list of
them: the rounding model reads it (tests/rounding_bound.py), and the truth rule
(`reference_error._read_through_floats`)."""


def rounds_nothing(primitive) -> bool:
    """Whether `primitive` rounds nothing: a move of entries, or one of `EXACT`."""
    return programs.rearranges(primitive) or primitive in EXACT


FORMATTED = frozenset({jax.lax.bitcast_convert_type_p, jax.lax.nextafter_p, jax.lax.reduce_precision_p})
"""The operations whose meaning is a float format's own, whatever width a program computes in: a value's
bits read as another type, its neighbour on its format's grid, its rounding to a format's digits. JAX states
no such fact of a primitive, so they are named here, once, by their objects: the float64 replay runs them at
the traced dtypes (tests/float64_replay.py), and a float64 truth may hold their narrow results
(`reference_error._read_through_floats`)."""


def stored_as(value, dtype):
    """`value`, computed in a truth's float64, as the format `dtype` stores it: rounded once to it and read
    back at its own width. A truth's declaration that a value is a format's own (a dequantized weight, a
    decoded code, a quotient the format defines in its own arithmetic): the truth computes it in float64
    and states the format's rounding here, a cast, which the truth rule keeps
    (`reference_error._read_through_floats`), where the same arithmetic run in the format is refused as a
    second fp32 run. One sum, difference, product, quotient or square root of the format's values so
    declared is the format's own result bit for bit: float64's 53 bits round twice to float32's 24, or to
    fewer, as once."""
    wide = jnp.result_type(value)
    return jnp.asarray(value).astype(dtype).astype(wide)


def float64_held(tree):
    """`tree` widened (`widened`) as arrays made under x64, so a program keeps them float64: the variables
    a float64 truth compiled once for every draw reads, where a layer that indexes a weight by its ids
    (DeepSeek V4's hash routing, Gemma 4's per-expert scales) indexes an array in a program (row 3654)."""
    with jax.enable_x64(new_val=True):
        return jax.tree.map(jnp.asarray, widened(tree))


def summed_as_run():
    """The scope a check runs each arm in (`_drawn`), the truth and its terms outside it: on the CPU lane,
    the process's default backend (`jax.default_backend()`), fp32 products summed in fp32
    (`dew.nn.precision.FP32_SUMS`), as an accelerator sums them and as a CPU does in any process but
    the suite's, whose wide arithmetic (`WIDE_CPU_ARITHMETIC`) sums Dew's products in float64, so the
    gate measures each arm as its users run it; elsewhere the process's own precision. A Dew arm that
    sums wide only under the suite's setting is no nearer float64 here than its users' runs are, and
    no route is kept off a CPU for summing narrower than that setting. The preset is JAX's default dot
    algorithm in it: every dot of no stated precision takes float32 operands, a bfloat16's exactly,
    and sums them in float32, as a GPU's fp32-accumulated product does, and a loop's running sum of
    them is the plain one (`compensates_loops`); the setting's widened reductions, which are not
    product sums, stay."""
    if jax.default_backend() == "cpu":
        return jax.default_matmul_precision(FP32_SUMS)
    return contextlib.nullcontext()


def unrewritten(fn):
    """`fn` run with the algebra pass off (`Unrewritten`), the one scope of every reference arm
    (`reference_error._compiled`) and float64 truth (`reference_error._truth_program`, `Truth`) the gate
    runs: the programs they trace or run are Dew's as written, never a rewrite the candidate's check judges
    (docs/design/algebra.md §3). A recorded arm, which replays values and runs no program, is as it is, and
    so is one unrewritten already."""
    if fn is None or getattr(fn, "recorded", False) or isinstance(fn, Unrewritten):
        return fn
    return Unrewritten(fn)


class Unrewritten:
    """`wrapped`, called or lowered (`lower`, where it has one: a compiled arm's) with the algebra pass off
    (`dew.algebra.passes(on=False)`), its other attributes `wrapped`'s (an arm's marks: `as_given`,
    `pilots`). The switch is in JAX's trace context, which jit's and Dew's programs key their traces and
    executables on: a reference's program never stands in for the candidate's, and a function's own, wrapped
    once a site, is the same program at every site of it."""

    def __init__(self, wrapped):
        self.wrapped = wrapped

    def __call__(self, *inputs, **options):
        from dew.algebra import passes

        with passes(on=False):
            return self.wrapped(*inputs, **options)

    def __getattr__(self, name):
        found = getattr(self.wrapped, name)
        if name not in ("lower", "trace"):
            return found

        # A compiled arm's trace and lowering, under the switch its calls run under; an arm of neither has
        # none.
        def staged(*inputs, **options):
            from dew.algebra import passes

            with passes(on=False):
                return found(*inputs, **options)
        return staged


class Truth:
    """A check's float64 truth as the gate runs it (`reference_error._truth_program`): its own function
    jitted, which every site of it shares, run unrewritten (`unrewritten`), compiled from its first call and
    never run eagerly, so a seed's truth is the same whichever seed a process draws first (a map task's,
    DRAWS) and a large truth pays no eager run. A truth that does host work on its values (NumPy, torch, a
    fetch to the host) states it, `as_given`, and the gate runs it so; unstated, its trace fails at its
    first seed, naming it (ledger 3198)."""

    def __init__(self, fn):
        self.fn, self.run = fn, unrewritten(jax.jit(fn))

    def __call__(self, *inputs):
        return self.run(*inputs)

    @property
    def trace(self):
        """Its program's trace (JAX's `jit(...).trace`), the one its calls run."""
        return self.run.trace
