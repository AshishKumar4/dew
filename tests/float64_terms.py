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
from arm_arithmetic import unrewritten

from dew.nn import programs

SUMS = frozenset({"dot_general", "ragged_dot_general", "reduce_sum", "conv_general_dilated", "cumsum",
                  "scatter-add"})
"""The operations that sum terms: a leaf's terms are those its own sums add,
taken at their float64 values (`float64_terms`). A ragged dot, an expert
layer's grouped matmul, sums as a dot does within each group, and its
transpose, which the kernel's gradient runs, is one too."""

CARRIES = frozenset({"add", "add_any", "sub", "neg", "convert_element_type", "reshape", "transpose",
                     "broadcast_in_dim", "squeeze", "expand_dims", "copy", "copy_p", "reduce_precision",
                     "slice", "dynamic_slice", "concatenate", "pad", "select_n", "optimization_barrier",
                     "name", "psum", "psum_invariant", "reduce_scatter", "all_gather", "all_gather_invariant",
                     "all_to_all", "ppermute", "pvary"})
"""The operations that carry the terms of sums into a leaf without making
new ones: sums combined (tied weights, accumulated microbatches, an update
added to a parameter, devices' partials) and moved, by a remat name
(`checkpoint_name`) or a collective too.
A product or quotient of a sum by a value that is none carries them too,
scaled element by element (`_applied`), as `s_i sum_k x_ik` sums the terms
`s_i x_ik`; a product of two sums ends them."""


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
        # One forward a module, so each call of it reads one trace (`_traced`); keyword options its own.
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
    values pick, `_static`) is evaluated by the same rules traced once into
    one compiled float64 program (`_compiled`), which each later call of `fn`
    on arguments of the same shapes runs; in any other, each run of such
    steps between its branches and loops is one (`_span`), the
    branches and loops walked as their values pick. The operations and their
    order are the walk's, compiled where the walk dispatched each alone."""
    closed, tree = _traced(fn, args)
    leaves = jax.tree.leaves(args)
    with evaluated_on(leaves):
        if _static(closed.jaxpr):
            values, energies = _compiled(closed.jaxpr)(closed.consts, leaves)
        else:
            values, energies = _evaluated(closed.jaxpr, closed.consts, leaves, compiling=True)
    return tree.unflatten(values), tree.unflatten([np.sqrt(np.asarray(_or_size(energy, value)))
                                                   for energy, value in zip(energies, values, strict=True)])


TRACES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
"""Each function's traced programs, by its arguments' structure, shapes, dtypes and weak types (`_traced`),
held while the function lives."""

FORWARDS: dict = {}
"""Each module's float64 forward (`float64_forward`), one function a module, so its trace is reused."""

COMPILED: dict = {}
"""Each static program's compiled evaluation (`_compiled`), by its jaxpr and the platform it runs on."""

STATIC: dict = {}
"""Whether each jaxpr is static (`_static`), by the jaxpr."""


def _static(jaxpr) -> bool:
    """Whether `jaxpr`, its calls' and scans' bodies included, takes no branch or loop its values pick
    (`cond`, `while`), which the walk reads concretely (`_applied`), once a jaxpr. A branch its platform
    picks (`jax.lax.platform_dependent`'s) is static: the walk reads its platform where it runs
    (`_platform`)."""
    if jaxpr not in STATIC:
        picked = _platform_picked(jaxpr)
        STATIC[jaxpr] = all(_static_equation(equation, picked) for equation in jaxpr.eqns)
    return STATIC[jaxpr]


def _platform_picked(jaxpr) -> frozenset:
    """The variables of `jaxpr` its platform makes (`platform_index`), which a branch they pick reads."""
    return frozenset(var for equation in jaxpr.eqns if equation.primitive.name == "platform_index"
                     for var in equation.outvars)


PLACED: contextvars.ContextVar = contextvars.ContextVar("float64_terms_placed", default=None)
"""The device the evaluation in progress runs on (`evaluated_on`), None outside one."""


@contextlib.contextmanager
def evaluated_on(leaves):
    """The scope of an evaluation of `leaves` (every evaluator's entry: `float64_terms`,
    `rounding_bound.rounding_scale`): the device it runs on, as jit would run them, the first committed
    array's, else JAX's default device (its `jax_default_device`, else its first), so a branch its
    platform picks is that device's (`_platform`) and every compiled evaluation is keyed by it
    (`_target`)."""
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
    """A `platform_index` equation's index on the device the evaluation runs on (`evaluated_on`), read
    concretely, so the branch it picks is one static step even within a compiled run (`_compiled`, keyed
    by that device's platform)."""
    with jax.default_device(_placed()), jax.ensure_compile_time_eval():
        return np.asarray(equation.primitive.bind(**equation.params))


def _target() -> str:
    """The platform the evaluation runs on (`evaluated_on`), which every compiled evaluation's key holds: a
    branch its platform picks is read into the program it compiles (`_platform`)."""
    return _placed().platform


def _traced(fn, args):
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


def released() -> None:
    """Forget every trace and compiled evaluation the walk keeps, and the arrays they hold: where the
    suite releases its compiled programs (tests/conftest.py's `release`)."""
    for cache in (TRACES, FORWARDS, COMPILED, STATIC, LAYOUTS, RUNS, READING):
        cache.clear()


def _compiled(jaxpr):
    """The compiled evaluation (`_evaluated`) of `jaxpr`, a static program, taking its constants and
    its inputs, compiled once a program (`_traced`)."""
    key = (jaxpr, _target())
    if key not in COMPILED:
        COMPILED[key] = jax.jit(lambda consts, given: _evaluated(jaxpr, consts, given))
    return COMPILED[key]


class Rule(Protocol):
    """How a walk other than the terms' evaluates each primitive (`_evaluated`'s `rule`): its outputs from
    its inputs (`primitive`), and the rule of a walk's equation, step or iteration `index` within it
    (`at`), so a rule that draws reads a draw of its own at every equation it meets
    (tests/rounding_bound.py's taps), and the dtype its values hold each traced type in (`held`)."""

    def at(self, index: int) -> Rule: ...

    def primitive(self, equation, inputs: list) -> list: ...

    def held(self, dtype) -> np.dtype:
        """The dtype the walk's values of traced type `dtype` take under this rule (a float64 walk's
        floats float64): a traced loop's carries and a branch's outputs are cast to it, exactly."""
        ...


def walked(jaxpr, consts, args, rule: Rule, *, compiling: bool = False) -> list:
    """`jaxpr`'s outputs on `consts` and `args`, each primitive evaluated by `rule` (`Rule`) and the walk's
    structure the gate's one: a call's body (`dew.nn.programs.called`), a scan, a loop, a branch and a
    shard_map walked inside, only the equations its outputs read, a static run compiled where `compiling`
    (`_evaluated`). Every program evaluator of the gate walks here: float64's replay
    (tests/float64_replay.py), a reference under a summation tree (tests/reference_trees.py), a tapped
    evaluation (tests/rounding_bound.py). Within `evaluated_on`, whose device a branch its platform picks
    reads."""
    return _evaluated(jaxpr, consts, args, compiling=compiling, rule=rule)[0]


def _evaluated(jaxpr, consts, args, given=None, compiling: bool = False, rule=None, used=None):
    """`jaxpr`'s outputs and their terms' energies, the sums of their squares
    (`float64_terms`), from `args` and the energies `given` for them (none for
    a computation's own inputs). A value no sum made has none. `compiling`
    evaluates each run of static steps between a branch and a loop
    (`_runs`) as one compiled program (`_span`).

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
        return var.val if isinstance(var, Literal) else values.get(var)

    def energy(var):
        return None if isinstance(var, Literal) else energies.get(var)

    spans = runs if compiling else tuple((index, index + 1) for index in range(len(jaxpr.eqns)))
    used = tuple([True] * len(jaxpr.outvars) if used is None else used)
    for (start, stop), reading in zip(spans, _reading(jaxpr, spans, used), strict=True):
        if reading is None:
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
                                         None if rule is None else rule.at(start), reading)
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
"""Each walked jaxpr's layout (`_layout`), by the jaxpr: a traced program's (`_traced`) is the same object
every draw, its bodies' too."""


def _layout(jaxpr):
    """What the walk reads of `jaxpr`'s structure alone, once a jaxpr: the variables whose terms reach an
    output (`_carrying`), the step that reads each variable last, after which it goes as an execution
    frees it (a chunked scan's float64 intermediates would otherwise all stay), and its spans (`_runs`)."""
    from jax.extend.core import Literal

    if jaxpr not in LAYOUTS:
        last = {var: index for index, equation in enumerate(jaxpr.eqns)
                for var in equation.invars if not isinstance(var, Literal)}
        for var in jaxpr.outvars:
            if not isinstance(var, Literal):
                last.pop(var, None)
        LAYOUTS[jaxpr] = (_carrying(jaxpr), last, tuple(_runs(jaxpr)))
    return LAYOUTS[jaxpr]


def _static_equation(equation, picked: frozenset = frozenset()) -> bool:
    """Whether `equation` takes no branch or loop its values pick, its bodies' included (`_static`): a
    branch whose index is among `picked`, its platform's (`_platform_picked`), is not one."""
    name = equation.primitive.name
    if name == "while" or (name == "cond" and equation.invars[0] not in picked):
        return False
    for value in equation.params.values():
        for body in value if isinstance(value, tuple | list) else (value,):
            inner = getattr(body, "jaxpr", body)
            if hasattr(inner, "eqns") and not _static(inner):
                return False
    return True


def _runs(jaxpr) -> list[tuple[int, int]]:
    """`jaxpr`'s steps as spans: each maximal run of static ones (`_static_equation`) one span, each
    branch or loop a span of its own."""
    spans, start = [], 0
    picked = _platform_picked(jaxpr)
    for index, equation in enumerate(jaxpr.eqns):
        if not _static_equation(equation, picked):
            if start < index:
                spans.append((start, index))
            spans.append((index, index + 1))
            start = index + 1
    if start < len(jaxpr.eqns):
        spans.append((start, len(jaxpr.eqns)))
    return spans


RUNS: dict = {}
"""Each static span's record (`_span`), by its program, its span and the platform it runs on."""


def _span(jaxpr, start: int, stop: int):
    """`jaxpr`'s run of static steps `start` to `stop`, read once a program (`_traced`) and span: the
    variables it reads from before it and those it keeps for after it (an output, or a later step's
    input), those whose last reader it holds, which go after it, and its steps evaluated as the walk
    evaluates them (`_applied`), compiled, from the read variables' values and energies and the walk's
    rule (`_evaluated`'s, None for none) to the kept ones'."""
    from jax.extend.core import Literal

    key = (jaxpr, start, stop, _target())
    if key not in RUNS:
        wanted, last, _ = _layout(jaxpr)
        run = jaxpr.eqns[start:stop]
        # A platform's index the run reads is read again inside it (`_platform`), never passed in: the
        # branch it picks is then one static step of the compiled run.
        platforms = {var: equation for equation in jaxpr.eqns if equation.primitive.name == "platform_index"
                     for var in equation.outvars}
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
                    outputs, made = _applied(equation, inputs, given, mark,
                                             rule=None if rule is None else rule.at(index))
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
    alone, as a complex value's terms have no order to sum in."""
    name, operands = equation.primitive.name, equation.invars
    if not all(_real(var) or not jnp.issubdtype(var.aval.dtype, jnp.inexact) for var in equation.outvars):
        return []
    floats = [index for index, var in enumerate(operands) if _real(var)]
    if name in CARRIES or _body(equation) is not None or name in ("scan", "while", "cond", "shard_map"):
        return floats
    if name == "reduce_sum":
        # A sum over unit axes adds nothing: it moves its operand, as a broadcast's transpose does.
        return floats if all(operands[0].aval.shape[axis] == 1 for axis in equation.params["axes"]) else []
    if name in ("mul", "div"):
        return floats
    return []


def _body(equation):
    """The jaxpr and constants a call runs (`dew.nn.programs.CALLS`: a jit, a remat, a custom rule's primal, a
    marked span, each evaluated op by op as the computation's own), or None. A shard_map's per-device program
    is evaluated too, on each device's blocks (`_mapped`); a body that runs elsewhere, a Pallas kernel, is
    one operation, its outputs their own size."""
    body = programs.called(equation)
    return None if body is None else (body, body.consts)


def _squared(value):
    return jnp.square(jnp.abs(value)) if jnp.issubdtype(jnp.result_type(value), jnp.inexact) else None


def _applied(equation, inputs, given, wanted: bool, compiling: bool = False, rule=None, used=None):
    """`equation`'s outputs from `inputs`, and their energies where `wanted`, a body's runs of static
    steps compiled where `compiling` (`_evaluated`); a primitive's by `rule` where one is given; a body's
    `used` outputs alone (`_evaluated`'s)."""
    name, params = equation.primitive.name, equation.params
    if name == "platform_index":
        return [_platform(equation)], [None]
    body = _body(equation)
    walked = wanted or rule is not None
    if walked and body is not None:
        return _evaluated(*body, inputs, given, compiling, rule, used)
    if rule is not None and name in ("scan", "while", "cond") and _traced_values(inputs):
        # Traced, its values pick no branch and bound no Python loop: the primitive itself over walked bodies.
        outputs = _bound_walk(name, params, inputs, rule, used)
        return outputs, [None] * len(outputs)
    if walked and name == "scan":
        return _scanned(params, inputs, given, compiling, rule, used)
    if walked and name == "while":
        return _looped(params, inputs, given, compiling, rule)
    if walked and name == "shard_map":
        return _mapped(params, inputs, given, rule)
    if walked and name == "cond":
        branch = params["branches"][int(inputs[0])]
        return _evaluated(branch.jaxpr, branch.consts, inputs[1:], given[1:], compiling, rule, used)
    if rule is not None:
        outputs = rule.primitive(equation, inputs)
        return outputs, [None] * len(outputs)
    outputs = equation.primitive.bind(*inputs, **equation.primitive.get_bind_params(params))
    outputs = list(outputs) if equation.primitive.multiple_results else [outputs]
    if not wanted:
        return outputs, [None] * len(outputs)
    carried = _carried(equation)
    if name in SUMS and not carried and all(_real(var) for var in equation.outvars):
        squares = [_squared(value) if _real(var) or jnp.issubdtype(var.aval.dtype, jnp.complexfloating)
                   else value for var, value in zip(equation.invars, inputs, strict=True)]
        return outputs, [equation.primitive.bind(*squares, **equation.primitive.get_bind_params(params))]
    if not carried:
        return outputs, [None] * len(outputs)
    if name in ("mul", "div"):
        summed = [index for index in carried if given[index] is not None]
        if len(summed) != 1 or (name == "div" and summed != [0]):
            return outputs, [None] * len(outputs)
        (index,) = summed
        scale = jnp.square(jnp.abs(inputs[1 - index]))
        return outputs, [given[index] * scale if name == "mul" else given[index] / scale]
    if all(given[index] is None for index in carried):
        return outputs, [None] * len(outputs)
    operands = [_or_zero(given[index], value) if index in carried else value
                for index, value in enumerate(inputs)]
    primitive = jax.lax.add_p if name == "sub" else equation.primitive
    made = (primitive.bind(*operands) if name == "sub" else
            operands[0] if name == "neg" else
            primitive.bind(*operands, **primitive.get_bind_params(params)))
    made = list(made) if primitive.multiple_results else [made]
    return outputs, [energy if jnp.issubdtype(jnp.result_type(value), jnp.inexact) else None
                     for energy, value in zip(made, outputs, strict=True)]


def scan_groups(params, operands: int) -> tuple[int, int]:
    """A scan equation's (`params`, of `operands` operands) constants and carries, its step's first
    operands, as its operand tree declares them (`ft_in`)."""
    constants, carries, _ = (len(group) for group in params["ft_in"].update(range(operands)).unpack())
    return constants, carries


def scanned_steps(params, inputs: list[tuple], body, lanes: int) -> list[tuple]:
    """A scan (lax.scan's equation `params`) over `inputs` a step at a time, as the gate's evaluators
    walk one (`float64_terms`, check 2's `certified`): each input a tuple of lanes, a value and what an
    evaluator carries beside it, sliced at each step lane by lane (a lane of None stays None), and
    `body(step, step_inputs)` the body's outputs for one step, carries first, tuples alike. The carries
    thread from step to step and each further output stacks over the steps in scan order, a lane where
    every step has one (else None); an empty scan's from its outvars' avals, its `lanes` - 1 others None."""
    jaxpr = params["jaxpr"].jaxpr
    constants, carries = scan_groups(params, len(inputs))
    consts, carry, xs = (inputs[:constants], inputs[constants:constants + carries],
                         inputs[constants + carries:])
    ys = []
    steps = range(params["length"])
    for step in reversed(steps) if params["reverse"] else steps:
        sliced = [tuple(None if lane is None else lane[step] for lane in x) for x in xs]
        outputs = body(step, [*consts, *carry, *sliced])
        carry = outputs[:carries]
        ys.append(outputs[carries:])
    if params["reverse"]:
        ys.reverse()
    if not ys:
        return [*carry, *((jnp.zeros((0, *var.aval.shape), var.aval.dtype), *(None,) * (lanes - 1))
                          for var in jaxpr.outvars[carries:])]
    return [*carry, *(tuple(None if any(entry is None for entry in lane) else jnp.stack(lane)
                            for lane in zip(*column, strict=True))
                      for column in zip(*ys, strict=True))]


def _traced_values(values) -> bool:
    """Whether any of `values` is a tracer: a walk inside a trace (a compiled span, a jit of the walk)."""
    return any(isinstance(value, jax.core.Tracer) for value in values)


def _bound_walk(name: str, params, inputs: list, rule, used=None) -> list:
    """A traced scan, while loop or branch of a rule's walk, bound as its own primitive (`jax.lax.scan`,
    `while_loop`, `switch`) whose bodies the walk evaluates, each step's or iteration's rule its own
    (`Rule.at`) as when it steps them concretely: a scan's at its step's index, a loop's at its iteration
    and its test at `-1 - iteration`, a branch's the walk's own. Of a branch's and a scan's outputs its
    `used` ones alone are walked, zeros of their types the rest, so each body keeps its signature."""
    if name == "cond":
        index, *operands = inputs
        branches = [lambda *values, branch=branch: _filled(branch.jaxpr, branch.consts, list(values), rule,
                                                           used)
                    for branch in params["branches"]]
        return list(jax.lax.switch(index, branches, *operands))
    if name == "while":
        condition, body = params["cond_jaxpr"], params["body_jaxpr"]
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
    body, length, reverse = params["jaxpr"], params["length"], params["reverse"]
    constants, carries = scan_groups(params, len(inputs))
    consts, reading = inputs[:constants], _scan_reads(params, len(inputs), used)

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


def _scan_reads(params, operands: int, used) -> tuple:
    """Which of a scan body's outputs its `used` outputs read (`_evaluated`'s `used`, every one where None):
    a used carry or output, and every carry a read output reads back in through the next step, until none
    is added."""
    from jax.extend.core import Var

    body = params["jaxpr"].jaxpr
    constants, carries = scan_groups(params, operands)
    reading = tuple([True] * len(body.outvars) if used is None else used)
    spans = tuple((index, index + 1) for index in range(len(body.eqns)))
    while True:
        read = {body.outvars[index] for index, kept in enumerate(reading)
                if kept and isinstance(body.outvars[index], Var)}
        for (start, _), found in zip(spans, _reading(body, spans, reading), strict=True):
            if found is not None:
                read.update(var for var in body.eqns[start].invars if isinstance(var, Var))
        added = [index for index in range(carries)
                 if not reading[index] and body.invars[constants + index] in read]
        if not added:
            return reading
        reading = tuple(kept or index in added for index, kept in enumerate(reading))


def _filled(jaxpr, consts, values: list, rule, used) -> list:
    """`jaxpr`'s `used` outputs walked by `rule` (`_evaluated`), the rest zeros, each in the dtype the rule
    holds its type in (`Rule.held`): a traced body's outputs keep one signature at every step and branch."""
    outputs = _evaluated(jaxpr, consts, values, rule=rule, used=used)[0]
    return [_held(output, var, rule) for output, var in zip(outputs, jaxpr.outvars, strict=True)]


def _held(value, var, rule):
    """`value`, of `var`'s type (zeros of it where None), in the dtype `rule` holds that type in."""
    dtype = rule.held(var.aval.dtype)
    return jnp.zeros(var.aval.shape, dtype) if value is None else jnp.asarray(value).astype(dtype)


def _scanned(params, inputs, given, compiling: bool = False, rule=None, used=None):
    """A scan's outputs and energies (`float64_terms`), its body evaluated a step at a time, each step's
    rule its own (`Rule.at`), its `used` outputs alone (`_scan_reads`)."""
    body = params["jaxpr"]
    reading = _scan_reads(params, len(inputs), used)

    def step(index, pairs):
        outputs, made = _evaluated(body.jaxpr, body.consts, [value for value, _ in pairs],
                                   [energy for _, energy in pairs], compiling,
                                   None if rule is None else rule.at(index), reading)
        return list(zip(outputs, made, strict=True))

    walked = scanned_steps(params, list(zip(inputs, given, strict=True)), step, lanes=2)
    return [value for value, _ in walked], [energy for _, energy in walked]


def _looped(params, inputs, given, compiling: bool = False, rule=None):
    """A while loop's carries and energies (`float64_terms`), its body evaluated a step at a time, each
    iteration's rule its own (`Rule.at`)."""
    condition, body = params["cond_jaxpr"], params["body_jaxpr"]
    tested, run = params["cond_nconsts"], params["body_nconsts"]
    condition_consts, body_consts, carry = inputs[:tested], inputs[tested:tested + run], inputs[tested + run:]
    body_terms, carry_terms = given[tested:tested + run], given[tested + run:]
    iteration = 0
    while bool(_evaluated(condition.jaxpr, condition.consts, [*condition_consts, *carry],
                          compiling=compiling, rule=None if rule is None else rule.at(-1 - iteration))[0][0]):
        carry, carry_terms = _evaluated(body.jaxpr, body.consts, [*body_consts, *carry],
                                        [*body_terms, *carry_terms], compiling,
                                        None if rule is None else rule.at(iteration))
        iteration += 1
    return carry, carry_terms


def _mapped(params, inputs, given, rule=None):
    """A shard_map's outputs and energies (`float64_terms`), its program evaluated
    op by op on each device's blocks, as the map runs it: an expert layer's
    dispatch (`dew.nn.moe`), whose grouped matmul and its transpose sum."""
    body, specs, out_specs = params["jaxpr"], tuple(params["in_specs"]), tuple(params["out_specs"])

    def program(values, energies):
        return tuple(map(tuple, _evaluated(body.jaxpr, body.consts, values, energies, rule=rule)))

    return jax.shard_map(program, mesh=params["mesh"], in_specs=(specs, specs),
                         out_specs=(out_specs, out_specs), axis_names=params["newly_manual_axes"],
                         check_vma=params["check_vma"])(tuple(inputs), tuple(given))


def _or_zero(energy, value):
    return jnp.zeros_like(value) if energy is None else energy


def _or_size(energy, value):
    """An output's energy: the one its sums carried, else its own size's square."""
    if not jnp.issubdtype(jnp.result_type(value), jnp.inexact):
        return np.zeros(np.shape(value))
    return _squared(value) if energy is None else energy
