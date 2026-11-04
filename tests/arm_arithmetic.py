"""The arithmetic a check's arms and truths run in (tests/reference_error.py): the scope each arm runs in
(`summed_as_run`), the programs a reference and a truth run (`unrewritten`), the float64 truth's inputs
(`widened`) and its compiled program (`Truth`)."""

import contextlib

import jax
import jax.numpy as jnp
import numpy as np

from dew.nn.precision import FP32_SUMS


def widened(tree):
    """`tree` in float64, its integer leaves as they are: the inputs of a float64 truth. A leaf an
    accelerator holds is widened where it lies, so its truth runs on that device's FP64 units with no host
    round trip (a decode step's keys and values, a hundred megabytes a draw to the host and back); any other
    is a host array."""
    def wide(leaf):
        if isinstance(leaf, jax.Array) and jnp.issubdtype(leaf.dtype, jnp.floating) and all(
                device.platform != "cpu" for device in leaf.devices()):
            with jax.enable_x64(new_val=True):
                return leaf.astype(jnp.float64)
        return np.asarray(leaf, np.float64 if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf.dtype)

    return jax.tree.map(wide, tree)


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
    (docs/design/algebra.md §3). A recorded arm, which replays values and runs no program, is as it is."""
    return fn if fn is None or getattr(fn, "recorded", False) else Unrewritten(fn)


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
        if name != "lower":
            return found

        # A compiled arm's lowering, under the switch its calls run under; an arm of no lowering has none.
        def lower(*inputs, **options):
            from dew.algebra import passes

            with passes(on=False):
                return found(*inputs, **options)
        return lower


class Truth:
    """A check's float64 truth as the gate runs it (`reference_error._truth_program`): compiled from its
    first call, or, where it cannot be traced and returns host values run as it is given (a NumPy truth), as
    given. A truth that returns JAX values and cannot be traced does host work on its inputs (a widening or a
    conversion on the host): it fails at its first seed with its trace's error, naming itself; it passes
    `as_given`."""

    def __init__(self, fn):
        self.fn, self.run = fn, None

    def __call__(self, *inputs):
        if self.run is not None:
            return self.run(*inputs)
        # Its own function jitted, which every site of it shares, run unrewritten (`unrewritten`).
        compiled = unrewritten(jax.jit(self.fn))
        try:
            # Compiled from the first call on, never run eagerly: a seed's truth is the same whichever seed
            # a process draws first (a map task's, DRAWS), and a large truth pays no eager run.
            values = compiled(*inputs)
        except Exception as untraced:
            first = unrewritten(self.fn)(*inputs)
            if all(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(first)):
                raise untraced
            self.run = unrewritten(self.fn)
            return first
        self.run = compiled
        return values
