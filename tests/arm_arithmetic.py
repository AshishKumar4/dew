"""The arithmetic a check's arms and truths run in (tests/reference_error.py): the scope each arm runs in
(`summed_as_run`), the programs a reference and a truth run (`unrewritten`), the device a draw makes its
arrays on (`drawn_on_the_host`) and a float64 truth computes on (`float64_host`), its inputs (`widened`)
and its compiled program (`Truth`)."""

import contextlib
import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import eval_jaxpr
from jax.extend import core as jcore
from jax.extend.core import primitives

from dew.nn import programs
from dew.nn.precision import FP32_SUMS

__accuracy_gate__ = True
"""The accuracy gate's own: its floats compared within their bounds (tools/lint_slop.py's SLOP020)."""


def float64_host() -> jax.Device | None:
    """The device this process's float64 truths compute on where its accelerators only emulate float64: a
    TPU's host CPU. A TPU keeps float64 as a dtype but computes it as pairs of float32s, within float32's
    range (1/3 is 0.33333333333333304 on a v5e), so a truth there is no float64's: a splash cell's dV truth
    gave every seed's error NaN, its arms finite (row 7226). None elsewhere, where float64 runs where its
    values lie: a GPU's FP64 units, a CPU's."""
    return jax.devices("cpu")[0] if jax.default_backend() == "tpu" else None


def on_float64_device():
    """The scope a float64 truth, its terms or its evaluation computes in: on `float64_host` where one is
    named (a TPU's process), every array it makes placed there; else where its values lie. Check 2's
    certification of native kernels (`certified`) runs on the lane's own device, outside it."""
    host = float64_host()
    return contextlib.nullcontext() if host is None else jax.default_device(host)


def on_float64_host(tree):
    """`tree`'s arrays put on `float64_host` where one is named (a truth's inputs and the constants its
    program holds, so it computes there whatever device made them), as they are elsewhere."""
    host = float64_host()
    if host is None:
        return tree
    with jax.enable_x64(new_val=True):
        def put(leaf):
            return jax.device_put(leaf, host) if isinstance(leaf, (jax.Array, np.ndarray)) else leaf

        return jax.tree.map(put, tree)


def drawn_on_the_host[S, R](inputs_of: Callable[[S], R], seed: S, *, held: bool = False) -> R:
    """`inputs_of(seed)` drawn on the host's CPU, as a recording's inputs were, whatever device the test runs
    on: jax.random's draws are the same bits on every device but its normal's float transform is not (a
    recorded site's seed drawn on an H100 had another digest than on the CPU, the recording's, in 157 of the
    recurrence GPU flow's cells, draw-digest at 996fa29f1), so a draw on the test's device would not be the
    recorded inputs; and a GPU tier part's prepare, whose default device is a compile-only GPU, computes
    nothing there (a draw or a weight load made there left its cell unprepared, rows 8074, 8635, 10552). The
    arrays, committed to no device, then go where the run reads them; with `held`, fetched to the host as
    NumPy's (`jax.device_get`), as a matrix a cell's every program closes over or a flow's prepare pickles
    is held (row 10831: its inline copies)."""
    with jax.default_device(jax.devices("cpu")[0]):
        made = inputs_of(seed)
    return jax.device_get(made) if held else made


def widened(tree):
    """`tree` in float64, its complex leaves in complex128 and its integer leaves as they are: the inputs
    of a float64 truth. A leaf an accelerator holds is widened where it lies, so its truth runs on that
    device's FP64 units with no host round trip (a decode step's keys and values, a hundred megabytes a
    draw to the host and back); any other is a host array, as is every leaf on a process whose float64
    computes on its host (`float64_host`, a TPU's): a truth computes there (`on_float64_device`, `Truth`),
    and one run as given reads host values (a torch truth's `torch.from_numpy`). The accelerator's leaves
    widen in one program a tree's shapes (`_widened_on_device`): a conversion a leaf, each traced and
    compiled, held a GPU idle 200 times a recurrence tier file (row 7675)."""
    leaves, structure = jax.tree.flatten(tree)
    held = [index for index, leaf in enumerate(leaves)
            if isinstance(leaf, jax.Array) and _wide(leaf.dtype) != leaf.dtype and float64_host() is None
            and all(device.platform != "cpu" for device in leaf.devices())]
    # One program a sharding: a jit takes its arguments on one ordered assignment of devices (a truth run as
    # given may hold leaves apart, each on its own GPU, or on meshes of one device set in two orders).
    grouped: dict[jax.sharding.Sharding, list[int]] = {}
    for index in held:
        grouped.setdefault(leaves[index].sharding, []).append(index)
    with jax.enable_x64(new_val=True):
        for group in grouped.values():
            for index, value in zip(group, _widened_on_device([leaves[index] for index in group]),
                                    strict=True):
                leaves[index] = value
    placed = set(held)
    return jax.tree.unflatten(structure, [leaf if index in placed else np.asarray(leaf, _wide(leaf.dtype))
                                          for index, leaf in enumerate(leaves)])


def _wide(dtype) -> np.dtype:
    """`dtype` widened (`widened`): a complex one to complex128, a float to float64, any other as it is."""
    return np.dtype(np.complex128 if jnp.issubdtype(dtype, jnp.complexfloating)
                    else np.float64 if jnp.issubdtype(dtype, jnp.floating) else dtype)


@jax.jit
def _widened_on_device(leaves: list) -> list:
    """`leaves`, arrays an accelerator holds, each widened where it lies (`widened`, under x64): one
    program a list of shapes."""
    return [leaf.astype(_wide(leaf.dtype)) for leaf in leaves]


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
    declared is the format's own correctly rounded result bit for bit, inside the format's normal range:
    float64's 53 bits round twice to float32's 24, or to fewer, as once. A backend's own operation need
    not be that one: XLA divides by a constant as a product with its rounded reciprocal, and a GPU's
    division is not IEEE's (`dew.nn.fake_quant`'s E2M1 quotient), so a format whose values a backend's
    plain operation defines is declared by what the format's definition says, not by this identity."""
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
    where the context places a call (`generation.placed_platform`: a GPU tier part's prepare, a CPU process,
    places its arms on its compile-only GPU and takes the GPU's own arithmetic, as its run does; row 8580's
    cuBLASLt FP8 arm, traced under the CPU's preset in that prepare, compiled without its fast
    accumulation), fp32 products summed in fp32
    (`dew.nn.precision.FP32_SUMS`), as an accelerator sums them and as a CPU does in any process but
    the suite's, whose wide arithmetic (`WIDE_CPU_ARITHMETIC`) sums Dew's products in float64, so the
    gate measures each arm as its users run it; elsewhere the process's own precision. A Dew arm that
    sums wide only under the suite's setting is no nearer float64 here than its users' runs are, and
    no route is kept off a CPU for summing narrower than that setting. The preset is JAX's default dot
    algorithm in it: every dot of no stated precision takes float32 operands, a bfloat16's exactly,
    and sums them in float32, as a GPU's fp32-accumulated product does, and a loop's running sum of
    them is the plain one (`compensates_loops`); the setting's widened reductions, which are not
    product sums, stay."""
    from dew.nn.kernels.generation import placed_platform

    if placed_platform() == "cpu":
        return jax.default_matmul_precision(FP32_SUMS)
    return contextlib.nullcontext()


AS_WRITTEN = {"cpu": "algsimp,constant_folding,reduction-splitter,multi_output_fusion",
              "gpu": "algsimp,constant_folding,multi_output_fusion,horizontal-loop-fusion,"
                     "horizontal-input-fusion",
              "tpu": "algsimp,constant_folding,multi_output_fusion"}
"""XLA's passes a reference's twin compiles without (`as_written`), by backend: its algebraic simplifier,
which rewrites (A/B)/C as A/(B*C) and divides by a broadcast through its reciprocal; its constant folding,
which computes at compile time what the program computes when it runs (and, without the simplifier,
XLA:CPU's raises SIGFPE on an operation of no elements: an MXFP8 step over an empty group); and a
reduction's split, which reorders its sum. With them a reference carries the rewrite Dew's arm, compiled by
the same XLA, carries too, and check 1 holds the two on par wherever both are wrong (a merge of +-2^127 and a
least normal: 3.6e30 where the truth is 1.2e-38, row 6909). A fusion's one effect on a value, summing a
product unrounded beside another term (a fused multiply-add: so equal and opposite terms do not cancel), is
held by its products' barriers (`products_held`), the fusions kept: a twin compiled without them ran every
operation as a kernel of its own, a delta net's nvfp4 gradient's 1,929 s and past 10 GiB where main's
reference ran in 42 s (test_joined, train 1099). The GPU's names are its passes', read on the GPU tier; the
TPU's, the open pipeline's, on the TPU tier."""

UNFUSED = {"cpu": "fusion,cpu-instruction-fusion",
           "gpu": "fusion,priority-fusion",
           "tpu": "fusion"}
"""The fusion passes a twin also compiles without where the gate cannot hold its products (`as_written`'s
`held`: an arm running Dew's own programs, whose compile point compiles them, `arm_forms._with_options`)."""

PRODUCTS = frozenset({jax.lax.mul_p, jax.lax.square_p, jax.lax.integer_pow_p})
"""The operations whose float result a fusion may sum unrounded beside another term (a fused multiply-add):
XLA:CPU contracts every product a fused addition reads, which none of its options turns off (armada,
2026-11-20: `xla_cpu_enable_fast_math` and the `xla_cpu_fast_math_honor_*` options each left a fused
`a * b + c` contracted, as a `reduce_precision` to the product's own format and an added zero did, both
elided); a product behind an optimization barrier stays rounded, its readers a fusion of their own."""


def as_written(platform: str | None = None, *, held: bool = True) -> dict:
    """The compiler options that compile a program as written on `platform` (`AS_WRITTEN`; where none is
    named, the platform the context places a call on, `generation.placed_platform`: a GPU tier part's prepare
    places its programs on its compile-only GPU; a device's `platform` where a program runs on one): each of
    its operations as it is, none rewritten with another; its fusions kept where its products are `held`
    (`products_held`), else compiled without them too (`UNFUSED`)."""
    from dew.nn.kernels.generation import placed_platform

    platform = platform or placed_platform()
    backend = "gpu" if platform in ("cuda", "rocm", "gpu") else platform
    passes = AS_WRITTEN[backend] if held else f"{AS_WRITTEN[backend]},{UNFUSED[backend]}"
    return {"xla_disable_hlo_passes": passes}


def products_held(fn):
    """`fn` computing as it traces, each float product (`PRODUCTS`) its program and the programs within it
    compute (a call's, a loop's, a branch's) behind an optimization barrier, so a fusion sums none unrounded
    beside another term: what a twin as written compiles, its fusions kept (`AS_WRITTEN`)."""
    @functools.wraps(fn)
    def held(*inputs, **options):
        closed, shapes = jax.make_jaxpr(functools.partial(fn, **options), return_shape=True)(*inputs)
        return jax.tree.unflatten(jax.tree.structure(shapes), eval_jaxpr(
            _products_held(closed.jaxpr), closed.consts, *jax.tree.leaves(inputs)))
    return held


def _products_held(jaxpr: jcore.Jaxpr) -> jcore.Jaxpr:
    """`jaxpr` with each float product's result (`PRODUCTS`) read through an optimization barrier, at every
    depth (`dew.nn.programs.with_bodies`); a native kernel's body, which XLA never sees, as it is
    (`programs.opaque`); `jaxpr` itself where it computes no product."""
    equations, changed = [], False
    for equation in jaxpr.eqns:
        if not programs.opaque(equation):
            params = programs.with_bodies(equation.params, _products_held)
            if any(params[name] is not value for name, value in equation.params.items()):
                equation, changed = equation.replace(params=params), True
        floats = [var for var in equation.outvars if not isinstance(var, jcore.DropVar)
                  and jnp.issubdtype(var.aval.dtype, jnp.floating)]
        if equation.primitive in PRODUCTS and floats:
            (made,) = equation.outvars
            product = jcore.Var(made.aval)
            equations += [equation.replace(outvars=[product]),
                          jcore.new_jaxpr_eqn([product], [made], jax.lax.optimization_barrier_p, {},
                                              jcore.no_effects, equation.source_info)]
            changed = True
            continue
        equations.append(equation)
    return jaxpr.replace(eqns=equations) if changed else jaxpr


def unrewritten(fn):
    """`fn` run with the algebra pass off (`Unrewritten`), the one scope of every reference arm
    (`reference_error._compiled`) and float64 truth (`reference_error._truth_program`, `Truth`) the gate
    runs: the programs they trace or run are Dew's as written, never a rewrite the candidate's check judges
    (docs/design/algebra.md §3). A recorded arm, which replays values and runs no program, is as it is, and
    so is one unrewritten already. A window loop (`over_windows`) stays the outer layer, the arm it runs
    unrewritten inside it, so what the gate reads of that arm (its `trace`, `_of_float64`) is the arm's."""
    from arm_forms import _Windowed

    if fn is None or getattr(fn, "recorded", False) or isinstance(fn, Unrewritten):
        return fn
    if isinstance(fn, _Windowed):
        return _Windowed(unrewritten(fn.fn))
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


class Written:
    """`fn` compiled by `compile` (`jax.jit`, or an arm's pairing's: `arm_forms.without_ynn`) as written
    (`as_written`) for the platform each call runs on (`_platform_of`: where its inputs lie, a GPU-committed
    draw's under a CPU default included), once a platform: an accuracy claim's reference as written
    (`arm_forms._written_twin`). Its `lower` and `trace` are the program an uncommitted call runs, for the
    platform the context places a call on."""

    def __init__(self, fn, compile: Callable = jax.jit):
        self.fn, self.compile, self._made = fn, compile, {}

    def program(self, platform: str):
        if platform not in self._made:
            self._made[platform] = self.compile(products_held(self.fn), compiler_options=as_written(platform))
        return self._made[platform]

    def __call__(self, *inputs, **options):
        return self.program(_platform_of(inputs))(*inputs, **options)

    @property
    def placed(self):
        """Its program for the platform the context places a call on (`generation.placed_platform`)."""
        from dew.nn.kernels.generation import placed_platform

        return self.program(placed_platform())

    def lower(self, *inputs, **options):
        return self.placed.lower(*inputs, **options)

    @property
    def trace(self):
        return self.placed.trace


class Truth:
    """A check's float64 truth as the gate runs it (`reference_error._truth_program`): its own function
    jitted, which every site of it shares, run unrewritten (`unrewritten`), compiled from its first call and
    never run eagerly, so a seed's truth is the same whichever seed a process draws first (a map task's,
    DRAWS) and a large truth pays no eager run. A truth that does host work on its values (NumPy, torch, a
    fetch to the host) states it, `as_given`, and the gate runs it so; unstated, its trace fails at its
    first seed, naming it (ledger 3198). On `device` where one is named (a GPU run's sizing, whose truths its
    host computes as a prepare's do, the GPU's memory left to the arms)."""

    def __init__(self, fn, device=None):
        self.fn, self.run, self.device = fn, unrewritten(jax.jit(fn)), device

    def __call__(self, *inputs):
        # On `device` where one is named: a GPU run's sizing computes its truths on the host, as a prepare
        # does; a process whose float64 is its host's (`float64_host`) computes them there.
        device = self.device or float64_host()
        return self.run(*(inputs if device is None else jax.device_put(inputs, device)))

    @property
    def trace(self):
        """Its program's trace (JAX's `jit(...).trace`), the one its calls run."""
        return self.run.trace


def _platform_of(inputs) -> str:
    """The platform a call on `inputs` runs on: their committed arrays' devices' (a GPU's for a draw an
    accelerator holds), else the one the context places a call on (`generation.placed_platform`)."""
    from dew.nn.kernels.generation import placed_platform

    platforms = {device.platform for leaf in jax.tree.leaves(inputs)
                 if isinstance(leaf, jax.Array) and getattr(leaf, "committed", False)
                 for device in leaf.devices()}
    return platforms.pop() if len(platforms) == 1 else placed_platform()
