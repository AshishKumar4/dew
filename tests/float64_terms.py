"""The float64 evaluation of a program the gate's checks read (tests/reference_error.py): a model's forward
in float64 (`float64_forward`), and a program evaluated op by op with each output's terms
(`float64_terms`), the size check 1 sizes a row that cancels by and the one-draw rule sizes a row by. Its
own module, a seam of the gate's: it reads a program's jaxpr and nothing of a check."""

from __future__ import annotations

import contextlib
import contextvars
import weakref
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np
from arm_arithmetic import on_float64_device, on_float64_host, unrewritten
from jax.extend.core import primitives
from jax.interpreters import partial_eval as pe
from store_wire import COMPILED as STORE, running

from dew.nn import programs
from dew.nn.precision import correctly_rounded_divide_p

SUMS = frozenset({jax.lax.dot_general_p, primitives.ragged_dot_general_p, jax.lax.reduce_sum_p,
    jax.lax.conv_general_dilated_p, jax.lax.cumsum_p, jax.lax.scatter_add_p})
"""The operations that sum terms: a leaf's terms are those its own sums add,
taken at their float64 values (`float64_terms`). A ragged dot, an expert
layer's grouped matmul, sums as a dot does within each group, and its
transpose, which the kernel's gradient runs, is one too."""

CARRIES = frozenset({jax.lax.add_p, primitives.add_jaxvals_p, jax.lax.sub_p, jax.lax.neg_p,
    jax.lax.convert_element_type_p, jax.lax.reshape_p, jax.lax.transpose_p, jax.lax.broadcast_in_dim_p,
    jax.lax.squeeze_p, jax.lax.copy_p, jax.lax.reduce_precision_p, jax.lax.slice_p, jax.lax.dynamic_slice_p,
    jax.lax.concatenate_p, jax.lax.stack_p, jax.lax.unstack_p, jax.lax.pad_p, jax.lax.select_n_p,
    jax.lax.optimization_barrier_p,
    primitives.name_p, jax.lax.psum_p, primitives.psum_invariant_p, primitives.reduce_scatter_p,
    jax.lax.all_gather_p, primitives.all_gather_invariant_p, jax.lax.all_to_all_p, jax.lax.ppermute_p,
    primitives.pvary_p})
"""The operations that carry the terms of sums into a leaf without making
new ones: sums combined (tied weights, accumulated microbatches, an update
added to a parameter, devices' partials) and moved, by a remat name
(`checkpoint_name`) or a collective too.
A product or quotient of a sum by a value that is none carries them too,
scaled element by element (`_applied`), as `s_i sum_k x_ik` sums the terms
`s_i x_ik`; a product of two sums ends them."""

SCALING = (jax.lax.mul_p, jax.lax.div_p, correctly_rounded_divide_p)
"""A product, and a quotient by a value that is no sum (JAX's and Dew's correctly rounded one), which scale
a sum's terms element by element (`_applied`)."""


def float32_values(jaxpr) -> int:
    """How many values of float32 `jaxpr`, and every jaxpr it holds, makes: none in a float64 program
    that computes in float64 throughout (`dew.nn.precision.at_least_fp32`)."""
    made = sum(any(getattr(getattr(var, "aval", None), "dtype", None) == jnp.float32
                   for var in eqn.outvars) for eqn in jaxpr.eqns)
    return made + sum(float32_values(inner) for eqn in jaxpr.eqns
                      for inner in jax.extend.core.jaxprs_in_params(eqn.params))


def float64_forward(module, variables, *args, **kwargs):
    """`module`'s own forward of `args` in float64, and each output's terms
    (`float64_terms`): the module computing in float64 over its variables
    widened, integer inputs as they are, `kwargs` given as they are. A
    reference's float64 truth written elsewhere (a fixture's) is sized by
    these terms, what Dew's own float64 sums add.

    Only `module`'s own `dtype` is widened: a module holding another (a
    multimodal wrapper's language model) that keeps its own dtype computes
    that part in it, so an output that is not float64 is refused, and such a
    model is passed widened whole (`ModelConfig.with_dtype`)."""
    from reference_error import widened

    with jax.enable_x64(new_val=True):
        wide = module.clone(dtype=jnp.float64)
        # One forward a module, so each call of it reads one trace (`traced`); keyword options its own.
        forward = (FORWARDS.setdefault(wide, wide.apply) if not kwargs
                   else lambda held, *given: wide.apply(held, *given, **kwargs))
        outputs, terms = float64_terms(forward, *widened((variables, *args)))
    narrow = sorted({str(leaf.dtype) for leaf in jax.tree.leaves(outputs)
                     if jnp.issubdtype(leaf.dtype, jnp.floating) and leaf.dtype != jnp.float64})
    if narrow:
        raise ValueError(f"{type(module).__name__} widened to float64 still gives {', '.join(narrow)} "
                         "outputs: a module it holds keeps its own dtype; widen the whole model")
    return outputs, terms


def float64_terms(fn, *args):
    """`fn(*args)` evaluated op by op, as float64 as `args` are, and the tree of
    each output leaf's terms, alike: the root sum of squares of the terms its
    final sums add (`SUMS`), at their operands' values, carried into the leaf
    (`CARRIES`). That is the size of the sum's partial sums, which its
    rounding scales with in any order: about the leaf's own size where the
    terms' signs are independent, far above it where they cancel by design
    (a softmax's shift invariance). A leaf made by any other operation is
    its own size, and an input carries no terms, so a parameter's terms
    after an update are the update's. Nothing propagates through the layers
    behind a sum, which would bound the whole computation.

    A program whose every step its shapes decide (no branch or loop its
    values pick, `static`) is evaluated by the same rules traced once into
    one compiled float64 program (`_compiled`), which each later call of `fn`
    on arguments of the same shapes runs; in any other, each run of such
    steps between its branches and loops is one (`_span`), the
    branches and loops walked as their values pick. The operations and their
    order are the walk's, compiled where the walk dispatched each alone."""
    tree, values, squares = _termed(fn, args, power=2)
    root = jnp.sqrt if _traced_values(jax.tree.leaves(args)) else np.sqrt
    return tree.unflatten(values), tree.unflatten([root(square) for square in squares])


def float64_magnitudes(fn, *args):
    """`fn(*args)` evaluated as `float64_terms` evaluates it, and the tree of each output leaf's terms'
    magnitudes, summed: the terms its sums add, each by its size and not its square, carried as the terms
    are, and through the sums behind a sum too: a sum's operand that sums made is taken at its own
    magnitudes, and a product of two sums at the product of theirs, so a contraction selected by a mask
    or nested in another is measured by what it multiplies and adds, however its formula is written. No
    partial sum reaching the leaf through sums, products, scalings and moves is larger, in any order: a
    leaf whose magnitudes lie inside a format's range overflows in no order of those sums, which the
    derivative contracts hold a value finite by (tests/contract_support.py). A value any other operation
    makes is its own size, and is added at it where a sum's terms meet it (a bias added to a sum, a
    carry's start)."""
    tree, values, sizes = _termed(fn, args, power=1)
    return tree.unflatten(values), tree.unflatten(sizes)


def float64_signed_sums(fn, *args):
    """`fn(*args)` evaluated as `float64_terms` evaluates it, and for each output leaf the sum of its
    positive terms and the size of the sum of its negative ones, the terms `float64_magnitudes` reads:
    every partial sum of those terms, in every order, lies between the one and the other's negative, and
    some order reaches each, so they are the largest a summation of them can pass through on its way to
    the value. The value is the positive terms less the negative and the magnitudes are the two together,
    so each is half the magnitudes with half the value added or taken away. A complex leaf's terms have no
    sign, and both are its magnitudes."""
    def halves(value, magnitudes):
        value, magnitudes = np.asarray(value), np.asarray(magnitudes)
        if np.iscomplexobj(value):
            return magnitudes, magnitudes
        return (magnitudes + value) / 2, (magnitudes - value) / 2

    values, magnitudes = float64_magnitudes(fn, *args)
    tree = jax.tree.structure(values)
    pairs = list(map(halves, jax.tree.leaves(values), jax.tree.leaves(magnitudes)))
    return (values, tree.unflatten([positive for positive, _ in pairs]),
            tree.unflatten([negative for _, negative in pairs]))


def _termed(fn, args, *, power: int):
    """`fn(*args)`'s outputs' tree, their values and each one's terms summed at `power` (`POWER`: their
    squares, or their magnitudes), the evaluation `float64_terms` describes. Under a trace (`_traced_values`:
    a site's terms program traced whole) its compiled evaluation inlined into the trace, its outputs the
    trace's own, nothing on the host, a branch its platform picks read for the platform of the store's
    device its prepare compiles for (`truths_ahead.CompiledTruths.device`), which the program lowers for. A
    trace outside a store's prepare, whose lowering's platform no trace states (a CPU's inputs under a GPU's
    default), and a program whose values steer it (`static` none), which no trace holds, are refused."""
    if _traced_values(jax.tree.leaves(args)):
        store = STORE.get()
        if store is None or not store.prepare:
            raise TypeError("float64_terms: a traced call reads the branches its platform picks for the "
                            "device a store's prepare compiles for, and none is preparing")
        # Traced afresh, its constants this trace's, never a cached trace's (`traced`'s).
        closed, shapes = jax.make_jaxpr(unrewritten(fn), return_shape=True)(*args)
        if not static(closed.jaxpr):
            raise TypeError("float64_terms: a program whose values steer it (a branch or loop they pick) is "
                            "walked as it runs, which no trace holds")
        token = POWER.set(power)
        try:
            with jax.default_device(store.device()), evaluated_on([]):
                values, energies = _compiled(closed.jaxpr)(closed.consts, jax.tree.leaves(args))
                return jax.tree.structure(shapes), values, [_or_size(energy, value) for energy, value
                                                            in zip(energies, values, strict=True)]
        finally:
            POWER.reset(token)
    ran = running()
    steered = None if ran is None else ran[0].walked(ran[1], f"terms {power}", args)
    if ran is not None and steered is None:
        # A run reading programs compiled ahead (`truths_ahead`): the truth's compiled run and the walk's,
        # its terms alone, in turn (`arm_store._held_walk`); a walk its values steer runs as it is
        # walked, on the GPU, named.
        store, label = ran
        values = jax.tree.leaves(store.called(label, "truth", args))
        energies = store.called(label, f"terms {power}", args)
        return store.tree(label, "truth", args), values, [
            np.asarray(_or_size(energy, value)) for energy, value in zip(energies, values, strict=True)]
    closed, tree = traced(fn, args)
    leaves, consts = on_float64_host((jax.tree.leaves(args), closed.consts))
    if ran is not None:
        # In a part's run, a walk its values steer (its prepare held none) compiles its spans on the GPU as
        # it walks, its float64 on the GPU's FP64 units (bible §8), named as its prepare declared it.
        from truths_ahead import AT_RUN

        ran[0].unprepared(f"{ran[1]}: its terms walk: {steered}", AT_RUN)
    token = POWER.set(power)
    try:
        with on_float64_device(), evaluated_on(leaves):
            if static(closed.jaxpr):
                values, energies = _compiled(closed.jaxpr)(consts, leaves)
            else:
                values, energies = evaluated(closed.jaxpr, consts, leaves, compiling=True)
            # A leaf of no energy (a literal output) sized where the walk ran.
            return tree, values, [np.asarray(_or_size(energy, value))
                                  for energy, value in zip(energies, values, strict=True)]
    finally:
        POWER.reset(token)


TRACES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
"""Each function's traced programs, by its arguments' structure, shapes, dtypes and weak types (`traced`),
held while the function lives."""

FORWARDS: dict = {}
"""Each module's float64 forward (`float64_forward`), one function a module, so its trace is reused."""

COMPILED: dict = {}
"""Each static program's compiled evaluation (`_compiled`), by its jaxpr and the platform it runs on."""

STATIC: dict = {}
"""Whether each jaxpr is static (`static`), by the jaxpr."""


def static(jaxpr) -> bool:
    """Whether `jaxpr`, its calls' and scans' bodies included, takes no branch or loop its values pick
    (`cond`, `while`), which the walk reads concretely (`_applied`), once a jaxpr (a scan is static to every
    walk, bound whole where traced, `_terms_scan`, `_bound_walk`). A branch its platform picks
    (`jax.lax.platform_dependent`'s) is static: the walk reads its platform where it runs (`_platform`)."""
    if jaxpr not in STATIC:
        picked = _platform_picked(jaxpr)
        STATIC[jaxpr] = all(_static_equation(equation, picked) for equation in jaxpr.eqns)
    return STATIC[jaxpr]


def _platform_picked(jaxpr) -> frozenset:
    """The variables of `jaxpr` its platform makes (`platform_index`), which a branch they pick reads."""
    return frozenset(var for equation in jaxpr.eqns if equation.primitive is primitives.platform_index_p
                     for var in equation.outvars)


PLACED: contextvars.ContextVar = contextvars.ContextVar("float64_terms_placed", default=None)
"""The device the evaluation in progress runs on (`evaluated_on`), None outside one."""

POWER: contextvars.ContextVar = contextvars.ContextVar("float64_terms_power", default=2)
"""The power of a term's size the evaluation in progress sums into an energy (`_termed`): 2, the terms'
squares (`float64_terms`), or 1, their magnitudes (`float64_magnitudes`). Every compiled evaluation's key
holds it."""


@contextlib.contextmanager
def evaluated_on(leaves):
    """The scope of an evaluation of `leaves` (every evaluator's entry: `float64_terms`,
    `rounding_bound.rounding_scale`): the device it runs on, as jit would run them, the first committed
    array's, else JAX's default device (its `jax_default_device`, else its first), so a branch its
    platform picks is that device's (`_platform`) and every compiled evaluation is keyed by it
    (`_target`): a float64 evaluation's within `arm_arithmetic.on_float64_device`, its host's on a TPU's
    process."""
    committed = next((leaf for leaf in leaves if isinstance(leaf, jax.Array) and leaf.committed), None)
    default = jax.config.jax_default_device
    if committed is not None:
        device = min(committed.devices(), key=lambda device: device.id)
    elif default is None or isinstance(default, str):
        device = jax.devices(default)[0]
    else:
        device = default
    token = PLACED.set(device)
    try:
        yield device
    finally:
        PLACED.reset(token)


def _placed():
    """The device the evaluation in progress runs on (`evaluated_on`); none outside one is refused, as no
    branch its platform picks can be read there."""
    device = PLACED.get()
    if device is None:
        raise RuntimeError("float64_terms evaluates within `evaluated_on`, which names the device it runs on")
    return device


def _platform(equation):
    """A `platform_index` equation's index on the device the evaluation runs on (`evaluated_on`): the branch
    its platforms name for the platform JAX lowers for there (`programs.lowering_platform`, CUDA's or
    ROCm's on a GPU), else its default (`programs.platform_branch`), read from its parameters as its
    lowering reads them, so the branch it picks is one static step even within a compiled run (`_compiled`,
    keyed by that device's platform), and a program compiled ahead for a GPU on a host with none picks the
    GPU's (`truths_ahead`)."""
    return np.asarray(programs.platform_branch(equation.params["platforms"],
                                               programs.lowering_platform(_placed())), np.int32)


def _target() -> str:
    """The platform the evaluation runs on (`evaluated_on`), which every compiled evaluation's key holds: a
    branch its platform picks is read into the program it compiles (`_platform`)."""
    return _placed().platform


def traced(fn, args):
    """`fn`'s program at `args` and its outputs' tree, traced once a function and arguments' structure,
    shapes, dtypes and weak types, as jit traces (its closure's arrays the program's constants), so its
    compiled runs (`_compiled`, `_span`) compile once."""
    leaves, structure = jax.tree.flatten(args)
    key = (structure, tuple((np.shape(leaf), jnp.result_type(leaf),
                             getattr(jax.typeof(leaf), "weak_type", False)) for leaf in leaves))
    traces = TRACES.setdefault(fn, {})
    if key not in traces:
        # A truth's program as written: traced with the algebra pass off (`arm_arithmetic.unrewritten`).
        closed, shapes = jax.make_jaxpr(unrewritten(fn), return_shape=True)(*args)
        traces[key] = (closed, jax.tree.structure(shapes))
    return traces[key]


def lowered(closed):
    """`closed`, a traced program, as lowering receives it. Where it holds calls of JAX's high-level
    primitives (hijax's: one of Dew's linear maps under forward mode, `dew.nn.kernels.generation.linear_map`;
    a checkpoint's recomputation), it is JAX's own lowering of the program
    (`jax.interpreters.partial_eval.lower_jaxpr2`), each such call written as the program it expands to,
    which a walk reads as any call's (`dew.nn.programs.called`), with the constants that program holds: the
    traced ones and those an expansion closes over. Any other program is itself."""
    return pe.lower_jaxpr2(closed.jaxpr) if closed.jaxpr.is_high else closed


def released() -> None:
    """Forget every trace and compiled evaluation the walk keeps, and the arrays they hold: where the
    suite releases its compiled programs (tests/conftest.py's `release`)."""
    for cache in (TRACES, FORWARDS, COMPILED, STATIC, LAYOUTS, RUNS, READING):
        cache.clear()


def _compiled(jaxpr):
    """The compiled evaluation (`evaluated`) of `jaxpr`, a static program, taking its constants and
    its inputs, compiled once a program (`traced`)."""
    key = (jaxpr, _target(), POWER.get())
    if key not in COMPILED:
        COMPILED[key] = jax.jit(lambda consts, given: evaluated(jaxpr, consts, given))
    return COMPILED[key]


class Rule(Protocol):
    """How a walk other than the terms' evaluates each primitive (`evaluated`'s `rule`): its outputs from
    its inputs (`primitive`), and the rule of a walk's equation, step or iteration `index` within it, an
    equation's primitive's name given for a path's text (`at`), so a rule that draws reads a draw of its own
    at every equation it meets (tests/rounding_bound.py's taps) and check 2 names each equation by its path
    (tests/certified.py), and the dtype its values hold each traced type in (`held`)."""

    def at(self, index: int, name: str | None = None) -> Rule: ...

    def primitive(self, equation, inputs: list) -> list: ...

    def held(self, dtype) -> np.dtype:
        """The dtype the walk's values of traced type `dtype` take under this rule (a float64 walk's
        floats float64): a traced loop's carries and a branch's outputs are cast to it, exactly."""
        ...

    pruned: bool
    """Whether the walk runs only what the outputs read (`_reading`), as an evaluation does; false where
    every equation is the rule's to hold, check 2's (tests/certified.py), whose contract is every equation
    executed."""


def walked(jaxpr, consts, args, rule: Rule, *, compiling: bool = False) -> list:
    """`jaxpr`'s outputs on `consts` and `args`, each primitive evaluated by `rule` (`Rule`) and the walk's
    structure the gate's one: a call's body (`dew.nn.programs.called`), a scan, a loop, a branch and a
    shard_map walked inside, only the equations its outputs read, a static run compiled where `compiling`
    (`evaluated`). Every program evaluator of the gate walks here: float64's replay
    (tests/float64_replay.py) and a tapped evaluation (tests/rounding_bound.py). Within `evaluated_on`,
    whose device a branch its platform picks reads."""
    return evaluated(jaxpr, consts, args, compiling=compiling, rule=rule)[0]


def evaluated(jaxpr, consts, args, given=None, compiling: bool = False, rule=None, used=None):
    """`jaxpr`'s outputs and their terms' energies, the sums of their squares (`float64_terms`; of their
    magnitudes under `float64_magnitudes`, `POWER`), from `args` and the energies `given` for them (none
    for a computation's own inputs). A value no sum made has none. `compiling` evaluates each run of
    static steps between a branch and a loop (`_runs`) as one compiled program (`_span`).

    `rule`, where given, evaluates every primitive in place of the terms
    arithmetic (`Rule`): the walk, its calls' bodies, scans, loops, branches
    and maps, is the same, each equation's rule its own (`Rule.at`), and no
    value carries an energy. A rule a compiled run takes is a pytree of JAX's,
    passed in as its values are (`_span`).

    Only the steps `jaxpr`'s `used` outputs read run (every output where None), each call, scan,
    branch and map told which of its own outputs are read (`_reading`), and an output not read is
    None: an initializer's forward pass, whose outputs its init drops, never runs."""
    from jax.extend.core import Literal

    wanted, last, runs = _layout(jaxpr)
    values, energies = {}, {}
    for var, value, energy in zip(jaxpr.constvars + jaxpr.invars, [*consts, *args],
                                  [None] * len(consts) + list(given or [None] * len(args)), strict=True):
        values[var], energies[var] = value, energy

    def read(var):
        # A literal at its traced type, as the program reads it: its Python value may be wider (a float64
        # scalar for a float32), which a call's body would take in as it is.
        return np.asarray(var.val, var.aval.dtype) if isinstance(var, Literal) else values.get(var)

    def energy(var):
        return None if isinstance(var, Literal) else energies.get(var)

    spans = runs if compiling else tuple((index, index + 1) for index in range(len(jaxpr.eqns)))
    pruned = rule is None or rule.pruned
    used = tuple([True] * len(jaxpr.outvars) if used is None or not pruned else used)
    readings = _reading(jaxpr, spans, used) if pruned else [None] * len(spans)
    for (start, stop), reading in zip(spans, readings, strict=True):
        if pruned and reading is None:
            continue
        run = jaxpr.eqns[start:stop]
        if stop - start > 1:
            # A static run of the walk's steps, compiled once (`_span`), the values it reads in.
            reads, written, freed, runner = _span(jaxpr, start, stop)
            outputs, made = runner([read(var) for var in reads], [energy(var) for var in reads], rule)
        else:
            (equation,) = run
            inputs = [read(var) for var in equation.invars]
            with equation.ctx.manager:
                outputs, made = _applied(equation, inputs, [energy(var) for var in equation.invars],
                                         any(var in wanted for var in equation.outvars), compiling,
                                         None if rule is None else rule.at(start, equation.primitive.name),
                                         reading)
            written = equation.outvars
            freed = [var for var in equation.invars
                     if not isinstance(var, Literal) and last.get(var) == start]
        for var in freed:
            values.pop(var, None)
            energies.pop(var, None)
        for var, value, made_energy in zip(written, outputs, made, strict=True):
            values[var], energies[var] = value, made_energy
    return ([read(var) if kept else None for var, kept in zip(jaxpr.outvars, used, strict=True)],
            [energy(var) for var in jaxpr.outvars])


READING: dict = {}
"""Each walked jaxpr's reads (`_reading`), by the jaxpr, its spans and the outputs read."""


def _reading(jaxpr, spans, used) -> list:
    """For each span of `jaxpr`, which of its outputs a later span or a `used` output reads, or None where it
    makes nothing read: a span runs whole once any of it is read (a compiled span reads all it reads), so a
    read span's every input is read."""
    from jax.extend.core import Var

    key = (jaxpr, spans, used)
    if key not in READING:
        read = {var for var, kept in zip(jaxpr.outvars, used, strict=True) if kept and isinstance(var, Var)}
        found = []
        for start, stop in reversed(spans):
            outvars = [var for equation in jaxpr.eqns[start:stop] for var in equation.outvars]
            if not any(var in read for var in outvars):
                found.append(None)
                continue
            found.append(tuple(var in read for var in outvars))
            read.update(var for equation in jaxpr.eqns[start:stop] for var in equation.invars
                        if isinstance(var, Var))
        READING[key] = found[::-1]
    return READING[key]


LAYOUTS: dict = {}
"""Each walked jaxpr's layout (`_layout`), by the jaxpr: a traced program's (`traced`) is the same object
every draw, its bodies' too."""


def _layout(jaxpr):
    """What the walk reads of `jaxpr`'s structure alone, once a jaxpr and measure (`POWER`): the variables
    whose terms reach an output (`_carrying`), the step that reads each variable last, after which it goes as
    an execution frees it (a chunked scan's float64 intermediates would otherwise all stay), and its spans
    (`_runs`)."""
    from jax.extend.core import Literal

    key = (jaxpr, POWER.get())
    if key not in LAYOUTS:
        last = {var: index for index, equation in enumerate(jaxpr.eqns)
                for var in equation.invars if not isinstance(var, Literal)}
        for var in jaxpr.outvars:
            if not isinstance(var, Literal):
                last.pop(var, None)
        LAYOUTS[key] = (_carrying(jaxpr), last, tuple(_runs(jaxpr)))
    return LAYOUTS[key]


def _static_equation(equation, picked: frozenset = frozenset()) -> bool:
    """Whether `equation` takes no branch or loop its values pick, its bodies' included (`static`): a
    branch whose index is among `picked`, its platform's (`_platform_picked`), is not one. A scan is: it is
    bound as a scan whose body the walk evaluates once (the terms walk's, `_terms_scan`, its steps alone only
    until its carries' terms keep their kinds; a rule's traced one, `_bound_walk`), so a run holding it
    compiles its body once, never a step at a time (a scanned model's float64 terms dispatched each layer's
    steps alone: 1,083 s on a GH200 at 21% busy, row 7096), nor every step unrolled into the program (a
    token scan's 64 steps and their adjoint's took 128 s of XLA's CPU compile, row 6004)."""
    primitive = equation.primitive
    if primitive is jax.lax.while_p or (primitive is jax.lax.cond_p and equation.invars[0] not in picked):
        return False
    return all(static(body) for body in programs.bodies(equation))


SPAN_STEPS = 1024
"""The most steps one compiled span holds (`_runs`), a call's body's steps counted among them (`steps`): a run
of static steps past it is compiled in parts, and a call whose body is past it walked as its own body's
spans, so no one compile holds a model's whole walk (row 9780: a Gemma 3n site's tapped walk, 6534 steps,
compiled whole peaked at 9.1 GiB; in parts of 1024, 4.5 GiB, in the same 176 s; row 10950: a jit-wrapped
reference is one step whose body is the program)."""


def steps(jaxpr) -> int:
    """`jaxpr`'s steps, every program's within it at any depth counted (`programs.equations`): what a compile
    of its walk holds."""
    return sum(1 for _ in programs.equations(jaxpr))


def _steps(equation) -> int:
    """`equation` and its bodies' steps (`steps`)."""
    return 1 + sum(steps(getattr(body, "jaxpr", body)) for body in programs.bodies(equation))


def _runs(jaxpr) -> list[tuple[int, int]]:
    """`jaxpr`'s steps as spans: each maximal run of static ones (`_static_equation`) one span, holding at
    most `SPAN_STEPS` steps, its calls' bodies' counted; each branch or loop a span of its own, and so each
    call of more steps than a span holds, which the walk then walks as its body's spans."""
    spans, start, held = [], 0, 0
    picked = _platform_picked(jaxpr)
    for index, equation in enumerate(jaxpr.eqns):
        size = _steps(equation)
        if not _static_equation(equation, picked) or size > SPAN_STEPS:
            if start < index:
                spans.append((start, index))
            spans.append((index, index + 1))
            start, held = index + 1, 0
        elif held + size > SPAN_STEPS:
            spans.append((start, index))
            start, held = index, size
        else:
            held += size
    if start < len(jaxpr.eqns):
        spans.append((start, len(jaxpr.eqns)))
    return spans


RUNS: dict = {}
"""Each static span's record (`_span`), by its program, its span and the platform it runs on."""


def _span(jaxpr, start: int, stop: int):
    """`jaxpr`'s run of static steps `start` to `stop`, read once a program (`traced`) and span: the
    variables it reads from before it and those it keeps for after it (an output, or a later step's
    input), those whose last reader it holds, which go after it, and its steps evaluated as the walk
    evaluates them (`_applied`), compiled, from the read variables' values and energies and the walk's
    rule (`evaluated`'s, None for none) to the kept ones'."""
    from jax.extend.core import Literal

    key = (jaxpr, start, stop, _target(), POWER.get())
    if key not in RUNS:
        wanted, last, _ = _layout(jaxpr)
        run = jaxpr.eqns[start:stop]
        # A platform's index the run reads is read again inside it (`_platform`), never passed in: the
        # branch it picks is then one static step of the compiled run.
        platforms = {var: equation for equation in jaxpr.eqns
                     if equation.primitive is primitives.platform_index_p for var in equation.outvars}
        written, reads = set(), []
        for equation in run:
            reads += [var for var in equation.invars
                      if not isinstance(var, Literal) and var not in written and var not in reads
                      and var not in platforms]
            written.update(equation.outvars)
        picked = {var for equation in run for var in equation.invars
                  if not isinstance(var, Literal) and var in platforms}
        kept = [var for equation in run for var in equation.outvars
                if last.get(var, len(jaxpr.eqns)) >= stop]
        freed = [var for var in reads if start <= last.get(var, -1) < stop]
        marks = [any(var in wanted for var in equation.outvars) for equation in run]

        def evaluate(read_values, read_energies, rule):
            values = dict(zip(reads, read_values, strict=True))
            energies = dict(zip(reads, read_energies, strict=True))
            for var in picked:
                values[var], energies[var] = _platform(platforms[var]), None
            for index, (equation, mark) in enumerate(zip(run, marks, strict=True), start=start):
                inputs = [var.val if isinstance(var, Literal) else values[var] for var in equation.invars]
                given = [None if isinstance(var, Literal) else energies.get(var) for var in equation.invars]
                with equation.ctx.manager:
                    at = None if rule is None else rule.at(index, equation.primitive.name)
                    outputs, made = _applied(equation, inputs, given, mark, rule=at)
                values.update(zip(equation.outvars, outputs, strict=True))
                energies.update(zip(equation.outvars, made, strict=True))
            return [values[var] for var in kept], [energies[var] for var in kept]

        RUNS[key] = (reads, kept, freed, jax.jit(evaluate))
    return RUNS[key]


def _carrying(jaxpr) -> set:
    """The variables whose terms reach an output of `jaxpr`."""
    from jax.extend.core import Var

    wanted = {var for var in jaxpr.outvars if isinstance(var, Var)}
    for equation in reversed(jaxpr.eqns):
        if any(var in wanted for var in equation.outvars):
            wanted.update(equation.invars[index] for index in _carried(equation)
                          if isinstance(equation.invars[index], Var))
    return wanted


def _real(var) -> bool:
    return jnp.issubdtype(var.aval.dtype, jnp.floating)


def _carried(equation) -> list[int]:
    """The operands whose terms `equation` carries to its outputs: real floats
    alone, as a complex value's terms have no order to sum in. A sum carries
    none of its operands' squares, and their magnitudes (`float64_magnitudes`)."""
    primitive, operands = equation.primitive, equation.invars
    if not all(_real(var) or not jnp.issubdtype(var.aval.dtype, jnp.inexact) for var in equation.outvars):
        return []
    floats = [index for index, var in enumerate(operands) if _real(var)]
    if (primitive in CARRIES or _body(equation) is not None or programs.control_flow(primitive)
            or primitive is primitives.shard_map_p):
        return floats
    if _moved(equation) or primitive in SCALING:
        return floats
    # A sum's operands that sums made carry their magnitudes into it; their squares end at it.
    return floats if primitive in SUMS and POWER.get() == 1 else []


def _moved(equation) -> bool:
    """Whether `equation`, a sum, adds nothing: one over unit axes, which moves its operand as a broadcast's
    transpose does and carries its terms as a move (`_carried`)."""
    return equation.primitive is jax.lax.reduce_sum_p and all(
        equation.invars[0].aval.shape[axis] == 1 for axis in equation.params["axes"])


def _body(equation):
    """The jaxpr and constants a call runs (`dew.nn.programs.called`: a jit, a remat, a custom rule's primal,
    a marked span, each evaluated op by op as the computation's own), or None. A shard_map's per-device
    program is evaluated too, on each device's blocks (`_mapped`); a body that runs elsewhere, a Pallas
    kernel, is one operation, its outputs their own size."""
    body = programs.called(equation)
    return None if body is None else (body, body.consts)


def _sized(value):
    """`value`'s size at the evaluation's power (`POWER`): its square, or its magnitude."""
    return jnp.square(jnp.abs(value)) if POWER.get() == 2 else jnp.abs(value)


def _measured(value):
    """A term's energy (`_sized`), none for a value that is no float."""
    return _sized(value) if jnp.issubdtype(jnp.result_type(value), jnp.inexact) else None


def _applied(equation, inputs, given, wanted: bool, compiling: bool = False, rule=None, used=None):
    """`equation`'s outputs from `inputs`, and their energies where `wanted`, a body's runs of static
    steps compiled where `compiling` (`evaluated`); a primitive's by `rule` where one is given; a body's
    `used` outputs alone (`evaluated`'s)."""
    primitive, params = equation.primitive, equation.params
    if primitive is primitives.platform_index_p:
        return [_platform(equation)], [None]
    body = _body(equation)
    walked = wanted or rule is not None
    if walked and body is not None:
        return evaluated(*body, inputs, given, compiling, rule, used)
    # A branch picks by its selector alone: one known (a platform's, `_platform`) walks its branch beside
    # traced operands, never every dead one (a platform's other branches, primitives no rule evaluates).
    picking = inputs[:1] if primitive is jax.lax.cond_p else inputs
    if rule is not None and programs.control_flow(primitive) and _traced_values(picking):
        # Traced, its values pick no branch and bound no Python loop: the primitive itself over walked bodies.
        outputs = _bound_walk(equation, inputs, rule, used)
        return outputs, [None] * len(outputs)
    static_scan = primitive is jax.lax.scan_p and all(map(static, programs.bodies(equation)))
    if walked and static_scan and rule is None:
        # The terms walk's scan of a static body, bound whole over the walk of it (`_terms_scan`), traced or
        # not; one whose body's branch or loop its values pick is stepped on concrete values (`_scanned`).
        return _terms_scan(equation, inputs, given, used)
    if walked and primitive is jax.lax.scan_p:
        return _scanned(equation, inputs, given, compiling, rule, used)
    if walked and primitive is jax.lax.while_p:
        return _looped(equation, inputs, given, compiling, rule)
    if walked and primitive is primitives.shard_map_p:
        return _mapped(equation, inputs, given, rule)
    if walked and primitive is jax.lax.cond_p:
        branch = programs.branches(equation)[int(inputs[0])]
        return evaluated(branch.jaxpr, branch.consts, inputs[1:], given[1:], compiling, rule, used)
    if rule is not None:
        outputs = rule.primitive(equation, inputs)
        return outputs, [None] * len(outputs)
    outputs = equation.primitive.bind(*inputs, **equation.primitive.get_bind_params(params))
    outputs = list(outputs) if equation.primitive.multiple_results else [outputs]
    if not wanted:
        return outputs, [None] * len(outputs)
    carried = _carried(equation)
    summing = primitive in SUMS and all(_real(var) for var in equation.outvars)
    if summing and (not carried or POWER.get() == 1) and not _moved(equation):
        # Each operand at its own size, or at the magnitudes the sums behind it carried (`_carried`).
        nested = POWER.get() == 1
        sizes = [energy if nested and energy is not None else _measured(value)
                 if _real(var) or jnp.issubdtype(var.aval.dtype, jnp.complexfloating) else value
                 for var, value, energy in zip(equation.invars, inputs, given, strict=True)]
        return outputs, [equation.primitive.bind(*sizes, **equation.primitive.get_bind_params(params))]
    if not carried:
        return outputs, [None] * len(outputs)
    if primitive in SCALING:
        summed = [index for index in carried if given[index] is not None]
        if len(summed) == 2 and primitive is jax.lax.mul_p and POWER.get() == 1:
            # A product of two sums is within the product of their magnitudes; its squares end here.
            return outputs, [given[0] * given[1]]
        if len(summed) != 1 or (primitive is not jax.lax.mul_p and summed != [0]):
            return outputs, [None] * len(outputs)
        (index,) = summed
        scale = _sized(inputs[1 - index])
        return outputs, [given[index] * scale if primitive is jax.lax.mul_p else given[index] / scale]
    if all(given[index] is None for index in carried):
        return outputs, [None] * len(outputs)
    # An operand no sum made adds no terms to the squares; a partial sum reaches its size too (`POWER`).
    untracked = _or_zero if POWER.get() == 2 else _or_own
    operands = [untracked(given[index], value) if index in carried else value
                for index, value in enumerate(inputs)]
    made = (jax.lax.add_p.bind(*operands) if primitive is jax.lax.sub_p else
            operands[0] if primitive is jax.lax.neg_p else
            primitive.bind(*operands, **primitive.get_bind_params(params)))
    made = list(made) if primitive.multiple_results else [made]
    return outputs, [energy if jnp.issubdtype(jnp.result_type(value), jnp.inexact) else None
                     for energy, value in zip(made, outputs, strict=True)]


def scan_groups(equation) -> tuple[int, int]:
    """A scan's `equation`'s constants and carries, its step's first operands, as its operand tree declares
    them (`ft_in`)."""
    groups = equation.params["ft_in"].update(range(len(equation.invars))).unpack()
    constants, carries, _ = (len(group) for group in groups)
    return constants, carries


def scanned_steps(equation, inputs: list[tuple], body, lanes: int) -> list[tuple]:
    """A scan (lax.scan's `equation`) over `inputs` a step at a time, as the gate's evaluators
    walk one (`float64_terms`, check 2's `certified`): each input a tuple of lanes, a value and what an
    evaluator carries beside it, sliced at each step lane by lane (a lane of None stays None), and
    `body(step, step_inputs)` the body's outputs for one step, carries first, tuples alike. The carries
    thread from step to step and each further output stacks over the steps in scan order, a lane where
    every step has one (else None); an empty scan's from its outvars' avals, its `lanes` - 1 others None."""
    jaxpr, params = programs.scan_body(equation).jaxpr, equation.params
    constants, carries = scan_groups(equation)
    consts, carry, xs = (inputs[:constants], inputs[constants:constants + carries],
                         inputs[constants + carries:])
    ys = []
    steps = range(params["length"])
    # Each lane's steps sliced once, a dispatch for them all (an array's iteration unstacks it), where a
    # step's slice of each lane was a gather of its own: on a GPU the walk's every step paid one a lane,
    # host-bound (ledger 6795).
    held = [tuple(None if lane is None else list(lane) for lane in x) for x in xs]
    for step in reversed(steps) if params["reverse"] else steps:
        sliced = [tuple(None if lane is None else lane[step] for lane in x) for x in held]
        outputs = body(step, [*consts, *carry, *sliced])
        carry = outputs[:carries]
        ys.append(outputs[carries:])
    if params["reverse"]:
        ys.reverse()
    if not ys:
        return [*carry, *((jnp.zeros((0, *var.aval.shape), var.aval.dtype), *(None,) * (lanes - 1))
                          for var in jaxpr.outvars[carries:])]
    return [*carry, *(tuple(None if any(entry is None for entry in lane) else _stacked(*lane)
                            for lane in zip(*column, strict=True))
                      for column in zip(*ys, strict=True))]


@jax.jit
def _stacked(*steps):
    """A scanned output's `steps` stacked in one compiled call, where `jnp.stack` dispatched each step's
    expansion on its own (a host-bound walk on a GPU, ledger 6795)."""
    return jnp.stack(steps)


def _traced_values(values) -> bool:
    """Whether any of `values` is a tracer: a walk inside a trace (a compiled span, a jit of the walk), a
    program calling `float64_terms` traced whole (a site's terms program its prepare compiles,
    `arm_store._held_program`)."""
    return any(isinstance(value, jax.core.Tracer) for value in values)


def _bound_walk(equation, inputs: list, rule, used=None) -> list:
    """A traced scan, while loop or branch of a rule's walk, bound as its own primitive (`jax.lax.scan`,
    `while_loop`, `switch`) whose bodies the walk evaluates, each step's or iteration's rule its own
    (`Rule.at`) as when it steps them concretely: a scan's at its step's index, a loop's at its iteration
    and its test at `-1 - iteration`, a branch's the walk's own. Of a branch's and a scan's outputs its
    `used` ones alone are walked, zeros of their types the rest, so each body keeps its signature."""
    primitive, params = equation.primitive, equation.params
    if primitive is jax.lax.cond_p:
        index, *operands = inputs
        taken = [lambda *values, branch=branch: _filled(branch.jaxpr, branch.consts, list(values), rule, used)
                 for branch in programs.branches(equation)]
        return list(jax.lax.switch(index, taken, *operands))
    if primitive is jax.lax.while_p:
        condition, body = programs.loop_programs(equation)
        tested, run = params["cond_nconsts"], params["body_nconsts"]
        condition_consts, body_consts = inputs[:tested], inputs[tested:tested + run]

        def test(state):
            iteration, carry = state
            return walked(condition.jaxpr, condition.consts, [*condition_consts, *carry],
                          rule.at(-1 - iteration))[0]

        def step(state):
            iteration, carry = state
            return iteration + 1, _filled(body.jaxpr, body.consts, [*body_consts, *carry], rule.at(iteration),
                                          None)

        carry = [_held(value, var, rule)
                 for value, var in zip(inputs[tested + run:], body.jaxpr.invars[run:], strict=True)]
        return list(jax.lax.while_loop(test, step, (0, carry))[1])
    body, length, reverse = programs.scan_body(equation), params["length"], params["reverse"]
    constants, carries = scan_groups(equation)
    consts, reading = inputs[:constants], _scan_reads(equation, used)

    def step(state, sliced):
        counter, carry = state
        index = length - 1 - counter if reverse else counter
        outputs = _filled(body.jaxpr, body.consts, [*consts, *carry, *sliced], rule.at(index), reading)
        return (counter + 1, outputs[:carries]), outputs[carries:]

    carry = [_held(value, var, rule) for value, var in
             zip(inputs[constants:constants + carries], body.jaxpr.invars[constants:constants + carries],
                 strict=True)]
    (_, carry), ys = jax.lax.scan(step, (0, carry), list(inputs[constants + carries:]), length=length,
                                  reverse=reverse)
    return [*carry, *ys]


def _scan_reads(equation, used, compiling: bool = False) -> tuple:
    """Which of a scan body's outputs its `used` outputs read (`evaluated`'s `used`, every one where None):
    a used carry or output, and every carry a read output reads back in through the next step, until none
    is added, over the spans its steps run (`evaluated`'s: each run of static steps whole where
    `compiling`, which reads every input of a run any of it reads, else each equation)."""
    from jax.extend.core import Var

    body = programs.scan_body(equation).jaxpr
    constants, carries = scan_groups(equation)
    reading = tuple([True] * len(body.outvars) if used is None else used)
    spans = (_layout(body)[2] if compiling
             else tuple((index, index + 1) for index in range(len(body.eqns))))
    while True:
        read = {body.outvars[index] for index, kept in enumerate(reading)
                if kept and isinstance(body.outvars[index], Var)}
        for (start, stop), found in zip(spans, _reading(body, spans, reading), strict=True):
            if found is not None:
                read.update(var for equation in body.eqns[start:stop] for var in equation.invars
                            if isinstance(var, Var))
        added = [index for index in range(carries)
                 if not reading[index] and body.invars[constants + index] in read]
        if not added:
            return reading
        reading = tuple(kept or index in added for index, kept in enumerate(reading))


def _filled(jaxpr, consts, values: list, rule, used) -> list:
    """`jaxpr`'s `used` outputs walked by `rule` (`evaluated`), the rest zeros, each in the dtype the rule
    holds its type in (`Rule.held`): a traced body's outputs keep one signature at every step and branch."""
    outputs = evaluated(jaxpr, consts, values, rule=rule, used=used)[0]
    return [_held(output, var, rule) for output, var in zip(outputs, jaxpr.outvars, strict=True)]


def _held(value, var, rule):
    """`value`, of `var`'s type (zeros of it where None), in the dtype `rule` holds that type in."""
    dtype = rule.held(var.aval.dtype)
    return jnp.zeros(var.aval.shape, dtype) if value is None else jnp.asarray(value).astype(dtype)


def _scanned(equation, inputs, given, compiling: bool = False, rule=None, used=None):
    """A scan's outputs and energies on concrete values (`float64_terms`), its body evaluated a step at a
    time, each step's rule its own (`Rule.at`), its `used` outputs alone (`_scan_reads`): a rule's walk's,
    and the terms walk's of a body whose branch or loop its values pick; the terms walk's of a static body
    is `_terms_scan`."""
    body = programs.scan_body(equation)
    reading = _scan_reads(equation, used, compiling)

    def step(index, pairs):
        outputs, made = evaluated(body.jaxpr, body.consts, [value for value, _ in pairs],
                                   [energy for _, energy in pairs], compiling,
                                   None if rule is None else rule.at(index), reading)
        return list(zip(outputs, made, strict=True))

    walked = scanned_steps(equation, list(zip(inputs, given, strict=True)), step, lanes=2)
    return [value for value, _ in walked], [energy for _, energy in walked]


def _kind(energy) -> tuple | None:
    """What a scan's carry must hold alike at every step of an energy (`_terms_scan`): its shape, dtype and
    weak type, or None for none."""
    if energy is None:
        return None
    typed = energy if isinstance(energy, jax.ShapeDtypeStruct) else jax.typeof(energy)
    return tuple(typed.shape), np.dtype(typed.dtype), bool(typed.weak_type)


def _terms_scan(equation, inputs, given, used=None):
    """A scan's outputs and energies (`float64_terms`) as the walk of its body a step at a time, each step's
    own, its `used` outputs alone (`_scan_reads`), bound as one `jax.lax.scan` wherever it can be: an
    energy is a value or none, and a scan's carry holds the same kinds (`_kind`) at every step. The kinds a
    step's carries leave with are a function of those they enter with, so the steps' kinds run from the
    first into a cycle (`jax.eval_shape` of a step): the steps before it run alone, the cycle's steps are
    one step of a scan over as many whole cycles as the length holds, and the steps left over run alone
    (one alone where a carry's first step makes its terms; none where its kinds never change). A stacked
    output's energy is where every step's is, as `scanned_steps` stacks one; an empty scan's outputs carry
    none. Each step is the stepped walk's evaluation, the scan's body compiled once."""
    body, params = programs.scan_body(equation), equation.params
    length, reverse = params["length"], params["reverse"]
    constants, carries = scan_groups(equation)
    reading = _scan_reads(equation, used)
    consts, carry, xs = (inputs[:constants], inputs[constants:constants + carries],
                         inputs[constants + carries:])
    const_given, carry_given, xs_given = (given[:constants], given[constants:constants + carries],
                                          given[constants + carries:])
    outvars, kept = body.jaxpr.outvars, reading[carries:]
    ys_vars = outvars[carries:]
    if length == 0:
        return ([*carry, *(jnp.zeros((0, *var.aval.shape), var.aval.dtype) if read else None
                           for var, read in zip(ys_vars, kept, strict=True))],
                [*carry_given, *(None for _ in ys_vars)])

    def walk(carry_values, carry_energies, x_values, x_energies):
        outputs, made = evaluated(body.jaxpr, body.consts, [*consts, *carry_values, *x_values],
                                   [*const_given, *carry_energies, *x_energies], used=reading)
        filled = [jnp.zeros(var.aval.shape, var.aval.dtype) if value is None else value
                  for value, var in zip(outputs, outvars, strict=True)]
        # Each carry at the type it entered with: a scan's carry keeps one at every step.
        values = [jnp.asarray(value, jnp.result_type(entered))
                  for value, entered in zip(filled[:carries], carry_values, strict=True)]
        return [*values, *filled[carries:]], list(made)

    # The steps' slices in the order the scan takes them (its last first where `reverse`).
    xs = [x[::-1] if reverse else x for x in xs]
    xs_given = [None if f is None else f[::-1] if reverse else f for f in xs_given]

    def at(step):
        return [x[step] for x in xs], [None if f is None else f[step] for f in xs_given]

    def following(kinds):
        energies = [None if kind is None else jax.ShapeDtypeStruct(kind[0], kind[1], weak_type=kind[2])
                    for kind in kinds]
        left = jax.eval_shape(lambda c, e, x, f: walk(c, e, x, f)[1][:carries], carry, energies, *at(0))
        return tuple(map(_kind, left))

    sequence, first = [tuple(map(_kind, carry_given))], {}
    while sequence[-1] not in first and len(sequence) <= length:
        first[sequence[-1]] = len(sequence) - 1
        sequence.append(following(sequence[-1]))
    cycle = sequence[-1] in first
    alone = first[sequence[-1]] if cycle else length
    period = len(sequence) - 1 - alone if cycle else 0
    whole = (length - alone) // period if period else 0
    parts = []

    def stepped(steps):
        nonlocal carry, carry_given
        for step in steps:
            outputs, made = walk(carry, carry_given, *at(step))
            carry, carry_given = outputs[:carries], made[:carries]
            parts.append(([y[None] for y in outputs[carries:]],
                          [None if energy is None else energy[None] for energy in made[carries:]]))

    stepped(range(min(alone, length)))
    if whole:
        span = whole * period

        def grouped(x):
            return None if x is None else x[alone:alone + span].reshape(whole, period, *x.shape[1:])

        def cycled(state, sliced):
            values, energies = state
            made_ys, made_energies = [], []
            for offset in range(period):
                outputs, made = walk(values, energies, [x[offset] for x in sliced[0]],
                                     [None if f is None else f[offset] for f in sliced[1]])
                values, energies = outputs[:carries], made[:carries]
                made_ys.append(outputs[carries:])
                made_energies.append(made[carries:])
            stacked = [jnp.stack(lane) for lane in zip(*made_ys, strict=True)]
            stacked_energies = [None if any(energy is None for energy in lane) else jnp.stack(lane)
                                for lane in zip(*made_energies, strict=True)]
            return (values, energies), (stacked, stacked_energies)

        (carry, carry_given), (ys, ys_given) = jax.lax.scan(
            cycled, (carry, carry_given), ([grouped(x) for x in xs], [grouped(f) for f in xs_given]),
            length=whole)
        parts.append(([y.reshape(span, *y.shape[2:]) for y in ys],
                      [None if energy is None else energy.reshape(span, *energy.shape[2:])
                       for energy in ys_given]))
    stepped(range(alone + whole * period, length))
    ys = [jnp.concatenate([values[lane] for values, _ in parts]) for lane in range(len(ys_vars))]
    ys_given = [None if any(energies[lane] is None for _, energies in parts)
                else jnp.concatenate([energies[lane] for _, energies in parts])
                for lane in range(len(ys_vars))]
    if reverse:
        ys = [y[::-1] for y in ys]
        ys_given = [None if energy is None else energy[::-1] for energy in ys_given]
    return ([*carry, *(y if read else None for y, read in zip(ys, kept, strict=True))],
            [*carry_given, *(energy if read else None for energy, read in zip(ys_given, kept, strict=True))])


def _looped(equation, inputs, given, compiling: bool = False, rule=None):
    """A while loop's carries and energies (`float64_terms`), its body evaluated a step at a time, each
    iteration's rule its own (`Rule.at`)."""
    (condition, body), params = programs.loop_programs(equation), equation.params
    tested, run = params["cond_nconsts"], params["body_nconsts"]
    condition_consts, body_consts, carry = inputs[:tested], inputs[tested:tested + run], inputs[tested + run:]
    body_terms, carry_terms = given[tested:tested + run], given[tested + run:]
    iteration = 0
    while bool(evaluated(condition.jaxpr, condition.consts, [*condition_consts, *carry],
                          compiling=compiling, rule=None if rule is None else rule.at(-1 - iteration))[0][0]):
        carry, carry_terms = evaluated(body.jaxpr, body.consts, [*body_consts, *carry],
                                        [*body_terms, *carry_terms], compiling,
                                        None if rule is None else rule.at(iteration))
        iteration += 1
    return carry, carry_terms


def _mapped(equation, inputs, given, rule=None):
    """A shard_map's outputs and energies (`float64_terms`), its program evaluated
    op by op on each device's blocks, as the map runs it: an expert layer's
    dispatch (`dew.nn.moe`), whose grouped matmul and its transpose sum."""
    (body,), params = programs.bodies(equation), equation.params
    specs, out_specs = tuple(params["in_specs"]), tuple(params["out_specs"])

    def program(values, energies):
        return tuple(map(tuple, evaluated(body.jaxpr, body.consts, values, energies, rule=rule)))

    return jax.shard_map(program, mesh=params["mesh"], in_specs=(specs, specs),
                         out_specs=(out_specs, out_specs), axis_names=params["newly_manual_axes"],
                         check_vma=params["check_vma"])(tuple(inputs), tuple(given))


def _or_zero(energy, value):
    return jnp.zeros_like(value) if energy is None else energy


def _or_own(energy, value):
    """A carried operand's magnitudes, its own size where no sum made it."""
    return _sized(value) if energy is None else energy


def _or_size(energy, value):
    """An output's energy: the one its sums carried, else its own size's (`_sized`)."""
    if not jnp.issubdtype(jnp.result_type(value), jnp.inexact):
        return jnp.zeros(np.shape(value)) if _traced_values([value]) else np.zeros(np.shape(value))
    return _measured(value) if energy is None else energy
