"""The float64 evaluation of a program the gate's checks read (tests/reference_error.py): a model's forward
in float64 (`float64_forward`), and a program evaluated op by op with each output's terms
(`float64_terms`), the size check 1 sizes a row that cancels by and the one-draw rule sizes a row by. Its
own module, a seam of the gate's: it reads a program's jaxpr and nothing of a check."""

import weakref

import jax
import jax.numpy as jnp
import numpy as np

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

CALLS = frozenset({"pjit", "jit", "closed_call", "core_call", "remat", "remat2", "checkpoint",
                   "custom_jvp_call", "custom_vjp_call", "custom_vjp_call_jaxpr"})
"""The calls whose bodies are evaluated op by op, as the computation's own:
a jit, a remat (`jax.checkpoint` binds 'remat2'), a custom rule's primal. A
shard_map's per-device program is too, on each device's blocks (`_mapped`).
A body that runs elsewhere, a Pallas kernel, is one operation, its outputs
their own size."""


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
        outputs, terms = float64_terms(lambda held, *given: wide.apply(held, *given, **kwargs),
                                       *widened((variables, *args)))
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
    steps between its branches and loops is one (`_compiled_run`), the
    branches and loops walked as their values pick. The operations and their
    order are the walk's, compiled where the walk dispatched each alone."""
    closed, tree = _traced(fn, args)
    leaves = jax.tree.leaves(args)
    if _static(closed.jaxpr):
        values, energies = _compiled(closed.jaxpr)(closed.consts, leaves)
    else:
        values, energies = _evaluated(closed.jaxpr, closed.consts, leaves, compiling=True)
    return tree.unflatten(values), tree.unflatten([np.sqrt(np.asarray(_or_size(energy, value)))
                                                   for energy, value in zip(energies, values, strict=True)])


TRACES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
"""Each function's traced programs, by its arguments' structure, shapes and dtypes (`_traced`), held
while the function lives."""

PROGRAMS: dict = {}
"""Each program's first traced jaxpr, by its text: the one every later trace of it evaluates (`_traced`)."""

COMPILED: dict = {}
"""Each static program's compiled evaluation (`_compiled`), by its jaxpr."""


def _static(jaxpr) -> bool:
    """Whether `jaxpr`, its calls' and scans' bodies included, takes no branch or loop its values pick
    (`cond`, `while`), which the walk reads concretely (`_applied`)."""
    for equation in jaxpr.eqns:
        if equation.primitive.name in ("cond", "while"):
            return False
        for value in equation.params.values():
            bodies = value if isinstance(value, tuple | list) else (value,)
            for body in bodies:
                inner = getattr(body, "jaxpr", body)
                if hasattr(inner, "eqns") and not _static(inner):
                    return False
    return True


def _traced(fn, args):
    """`fn`'s program at `args` and its outputs' tree, traced once a function and arguments' structure,
    shapes and dtypes, as jit traces (its closure's arrays the program's constants); the jaxpr the first
    trace of the same program text gave, so a function made afresh each call (`float64_forward`'s) reads
    one program, and its compiled runs (`_compiled`, `_compiled_run`) compile once."""
    from jax.extend.core import ClosedJaxpr

    leaves, structure = jax.tree.flatten(args)
    key = (structure, tuple((np.shape(leaf), jnp.result_type(leaf)) for leaf in leaves))
    traces = TRACES.setdefault(fn, {})
    if key not in traces:
        closed, shapes = jax.make_jaxpr(fn, return_shape=True)(*args)
        jaxpr = PROGRAMS.setdefault(str(closed.jaxpr), closed.jaxpr)
        traces[key] = (ClosedJaxpr(jaxpr, closed.consts), jax.tree.structure(shapes))
    return traces[key]


def _compiled(jaxpr):
    """The compiled evaluation (`_evaluated`) of `jaxpr`, a static program, taking its constants and
    its inputs, compiled once a program (`_traced`)."""
    if jaxpr not in COMPILED:
        COMPILED[jaxpr] = jax.jit(lambda consts, given: _evaluated(jaxpr, consts, given))
    return COMPILED[jaxpr]


def _evaluated(jaxpr, consts, args, given=None, compiling: bool = False):
    """`jaxpr`'s outputs and their terms' energies, the sums of their squares
    (`float64_terms`), from `args` and the energies `given` for them (none for
    a computation's own inputs). A value no sum made has none. `compiling`
    evaluates each run of static steps between a branch and a loop
    (`_runs`) as one compiled program (`_compiled_run`)."""
    from jax.extend.core import Literal

    wanted = _carrying(jaxpr)
    values, energies = {}, {}
    for var, value, energy in zip(jaxpr.constvars + jaxpr.invars, [*consts, *args],
                                  [None] * len(consts) + list(given or [None] * len(args)), strict=True):
        values[var], energies[var] = value, energy

    def read(var):
        return var.val if isinstance(var, Literal) else values[var]

    def energy(var):
        return None if isinstance(var, Literal) else energies.get(var)

    # Each value and energy goes once its last reader has run, as an execution
    # frees it: a chunked scan's float64 intermediates would otherwise all stay.
    last = {var: index for index, equation in enumerate(jaxpr.eqns)
            for var in equation.invars if not isinstance(var, Literal)}
    for var in jaxpr.outvars:
        if not isinstance(var, Literal):
            last.pop(var, None)
    spans = _runs(jaxpr) if compiling else [(index, index + 1) for index in range(len(jaxpr.eqns))]
    for start, stop in spans:
        run = jaxpr.eqns[start:stop]
        if stop - start > 1:
            # A static run of the walk's steps, compiled once (`_compiled_run`), the values it reads in.
            reads = _run_reads(run)
            kept = [var for var in _run_writes(run) if last.get(var, len(jaxpr.eqns)) >= stop]
            outputs, made = _compiled_run(jaxpr, start, stop, reads, kept, wanted)(
                [read(var) for var in reads], [energy(var) for var in reads])
            written = kept
        else:
            (equation,) = run
            inputs = [read(var) for var in equation.invars]
            with equation.ctx.manager:
                outputs, made = _applied(equation, inputs, [energy(var) for var in equation.invars],
                                         any(var in wanted for var in equation.outvars), compiling)
            written = equation.outvars
        for var in {var for equation in run for var in equation.invars if not isinstance(var, Literal)}:
            if start <= last.get(var, -1) < stop:
                values.pop(var, None)
                energies.pop(var, None)
        for var, value, made_energy in zip(written, outputs, made, strict=True):
            values[var], energies[var] = value, made_energy
    return [read(var) for var in jaxpr.outvars], [energy(var) for var in jaxpr.outvars]


def _static_equation(equation) -> bool:
    """Whether `equation` takes no branch or loop its values pick, its bodies' included (`_static`)."""
    if equation.primitive.name in ("cond", "while"):
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
    for index, equation in enumerate(jaxpr.eqns):
        if not _static_equation(equation):
            if start < index:
                spans.append((start, index))
            spans.append((index, index + 1))
            start = index + 1
    if start < len(jaxpr.eqns):
        spans.append((start, len(jaxpr.eqns)))
    return spans


def _run_writes(run) -> list:
    """The variables a run of steps defines, in order."""
    return [var for equation in run for var in equation.outvars]


def _run_reads(run) -> list:
    """The variables a run of steps reads from before it, in order of first read."""
    from jax.extend.core import Literal

    written, reads = set(), []
    for equation in run:
        for var in equation.invars:
            if not isinstance(var, Literal) and var not in written and var not in reads:
                reads.append(var)
        written.update(equation.outvars)
    return reads


RUNS: dict = {}
"""Each static run's compiled evaluation (`_compiled_run`), by its program, its span and the energies it
reads."""


def _compiled_run(jaxpr, start: int, stop: int, reads, kept, wanted):
    """`jaxpr`'s run of static steps `start` to `stop` evaluated as the walk evaluates them (`_applied`),
    compiled once a program (`_traced`) and span, reading `reads`' values and energies and returning
    `kept`'s."""
    from jax.extend.core import Literal

    run = jaxpr.eqns[start:stop]
    marks = tuple(any(var in wanted for var in equation.outvars) for equation in run)
    key = (jaxpr, start, stop, marks)
    if key not in RUNS:
        def evaluate(read_values, read_energies):
            values, energies = dict(zip(reads, read_values, strict=True)), dict(
                zip(reads, read_energies, strict=True))
            for equation, mark in zip(run, marks, strict=True):
                inputs = [var.val if isinstance(var, Literal) else values[var] for var in equation.invars]
                given = [None if isinstance(var, Literal) else energies.get(var) for var in equation.invars]
                with equation.ctx.manager:
                    outputs, made = _applied(equation, inputs, given, mark)
                values.update(zip(equation.outvars, outputs, strict=True))
                energies.update(zip(equation.outvars, made, strict=True))
            return [values[var] for var in kept], [energies[var] for var in kept]
        RUNS[key] = jax.jit(evaluate)
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
    """The jaxpr and constants a call runs (`CALLS`), or None."""
    if equation.primitive.name not in CALLS:
        return None
    for name in ("jaxpr", "call_jaxpr", "fun_jaxpr"):
        body = equation.params.get(name)
        if body is not None:
            return (body.jaxpr, body.consts) if hasattr(body, "consts") else (body, [])
    return None


def _squared(value):
    return jnp.square(jnp.abs(value)) if jnp.issubdtype(jnp.result_type(value), jnp.inexact) else None


def _applied(equation, inputs, given, wanted: bool, compiling: bool = False):
    """`equation`'s outputs from `inputs`, and their energies where `wanted`, a body's runs of static
    steps compiled where `compiling` (`_evaluated`)."""
    name, params = equation.primitive.name, equation.params
    body = _body(equation)
    if wanted and body is not None:
        return _evaluated(*body, inputs, given, compiling)
    if wanted and name == "scan":
        return _scanned(params, inputs, given, compiling)
    if wanted and name == "while":
        return _looped(params, inputs, given, compiling)
    if wanted and name == "shard_map":
        return _mapped(params, inputs, given)
    if wanted and name == "cond":
        branch = params["branches"][int(inputs[0])]
        return _evaluated(branch.jaxpr, branch.consts, inputs[1:], given[1:], compiling)
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


def _scanned(params, inputs, given, compiling: bool = False):
    """A scan's outputs and energies (`float64_terms`), its body evaluated a step at a time."""
    body = params["jaxpr"]

    def step(_, pairs):
        outputs, made = _evaluated(body.jaxpr, body.consts, [value for value, _ in pairs],
                                   [energy for _, energy in pairs], compiling)
        return list(zip(outputs, made, strict=True))

    walked = scanned_steps(params, list(zip(inputs, given, strict=True)), step, lanes=2)
    return [value for value, _ in walked], [energy for _, energy in walked]


def _looped(params, inputs, given, compiling: bool = False):
    """A while loop's carries and energies (`float64_terms`), its body evaluated a step at a time."""
    condition, body = params["cond_jaxpr"], params["body_jaxpr"]
    tested, run = params["cond_nconsts"], params["body_nconsts"]
    condition_consts, body_consts, carry = inputs[:tested], inputs[tested:tested + run], inputs[tested + run:]
    body_terms, carry_terms = given[tested:tested + run], given[tested + run:]
    while bool(_evaluated(condition.jaxpr, condition.consts, [*condition_consts, *carry],
                          compiling=compiling)[0][0]):
        carry, carry_terms = _evaluated(body.jaxpr, body.consts, [*body_consts, *carry],
                                        [*body_terms, *carry_terms], compiling)
    return carry, carry_terms


def _mapped(params, inputs, given):
    """A shard_map's outputs and energies (`float64_terms`), its program evaluated
    op by op on each device's blocks, as the map runs it: an expert layer's
    dispatch (`dew.nn.moe`), whose grouped matmul and its transpose sum."""
    body, specs, out_specs = params["jaxpr"], tuple(params["in_specs"]), tuple(params["out_specs"])

    def program(values, energies):
        return tuple(map(tuple, _evaluated(body.jaxpr, body.consts, values, energies)))

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
