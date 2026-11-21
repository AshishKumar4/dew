"""A check-1 site's arms compiled ahead into a store of programs (`truths_ahead.CompiledTruths`, a GPU tier's
prepare) and run from it (`_stored_arms`): each arm's program traced as its run traces it and compiled as the
gate compiles it (`arm_forms._compiled`), its references' twins as written (`arm_forms._written_twin`) held
apart."""

import dataclasses
import functools
import itertools
from collections.abc import Callable, Mapping

import jax
import numpy as np
from arm_arithmetic import as_written, products_held, summed_as_run, unrewritten
from arm_forms import _bounded, _declared, _given, _own_function, _refuse_numpy_arms, _RunToRun, _Windowed
from truths_ahead import AT_RUN, CompiledTruths, arm_role, named

from dew.parity_stages import staged_units


def armed(label: str, written: bool = False) -> str:
    """The label a site's arms are held under in a store of programs compiled ahead (`_compile_arms`), apart
    from its truths' (`reference_error._compile_site`): each set compiled in its own arithmetic, the arms'
    without x64, and its references' twins as written (`written`) apart from its arms compiled alike."""
    return f"{label}, its arms as written" if written else f"{label}, its arms"


def _written_references(reference_fn, also: Mapping) -> list[tuple[str | None, Callable]]:
    """The references a site's accuracy claim also holds Dew's arm against compiled as written
    (`arm_forms._bounded`, `arm_forms._written_twin`), each by its name (the site's own None), in its draws'
    order."""
    return [(name, fn) for name, fn in ((None, reference_fn), *also.items()) if _bounded(fn)]


def _written_role(name: str | None) -> str:
    """The role of reference `name`'s twin compiled as written in a site's store (`_written_references`)."""
    return "written" if name is None else f"written {name}"


def _arm_compile(compile: Callable, written: dict | None) -> Callable:
    """How the gate compiles a site's arms (`arm_forms._arms`), its results placed
    (`dew.training.ahead.compile_programs`' `compiled`): by `compile` (`arm_forms._compile_for`'s: as Dew's
    users compile them, `jax.jit`, beside an arm run as given, else without YNNPACK's fusions, `without_ynn`),
    with the `written` options where they are a twin's (`arm_forms._written_twin`), its products held
    (`arm_arithmetic.products_held`)."""
    def compiled(function, *, out_shardings):
        # An arm held whole (`_Whole`) is a jit of its own compiler options, lowered as itself: no other jit
        # may hold it.
        if isinstance(function, _Whole):
            return function
        return compile(function if written is None else products_held(function), out_shardings=out_shardings,
                       compiler_options=written)

    return compiled


def _compile_arms(store: CompiledTruths, arms: Mapping[str, tuple[Callable, bool]], draw, label: str, *,
                  seeds: int, orders, compile: Callable, sizing: range = range(0),
                  written: bool = False) -> None:
    """A site's `arms` (each a role's function and whether it is a reference, unrewritten) compiled ahead into
    `store` at the shapes of every draw of its stages and each of its `orders`, and of every seed of `sizing`
    where a qualification run sizes the site (`reference_error._qualified`, `reference_error._sizing_seeds`),
    each traced as its run traces it, for the store's devices (`CompiledTruths.tracing`, in the arithmetic
    arms run in, `summed_as_run`), and compiled as the gate compiles it (`_arm_compile`, by `compile`; as
    written where `written`, its references' twins, held apart, `armed`): its run calls each draw's executable
    (`_ahead_arm`), tracing, lowering and compiling none, its sizing too. An arm the gate runs as given, or
    one running Dew's own programs (`dew_programs`, a host loop), holds no program. A site its target's
    compile refuses (`report.CompiledOnDevice`: a convolution's algorithm, which its GPU picks) is written at
    run (`compile_site`'s `at_run`): its run compiles every arm on its GPU, as given (`_stored_arm`)."""
    # Each arm named as each process first finds it, before any call of it here (`arm_role`); one no
    # statement holds runs unprepared in its run.
    named_arms = {role: (named_role, fn, reference) for role, (fn, reference) in arms.items()
                  if fn is not None and _holds_program(fn)
                  and (named_role := arm_role(role, fn, label)) is not None}
    if not named_arms:
        return
    if store.target is None:
        # An arm computing in NumPy refused as a run refuses it, by a call on a draw; for a GPU's store the
        # prepare computes nothing on its host (bible §8), and its trace refuses an arm that cannot trace.
        _refuse_numpy_arms(draw, *(fn for _, fn, _ in named_arms.values()))
    inputs: dict[str, tuple] = {}
    for seed in itertools.chain(range(staged_units(seeds)), sizing):
        drawn = drawn_on_the_host(draw, seed)
        for ordered in (drawn,) if orders is None else orders(drawn):
            inputs.setdefault(named("arms", ordered), ordered)
    programs: dict = {}
    held: dict = {}
    # The arithmetic arms run in on the store's devices (`summed_as_run`, read where the store places its
    # calls: a GPU's own on a target's compile-only GPU, whatever the host the prepare runs on).
    with store.tracing(), summed_as_run():
        for role, fn, reference in named_arms.values():
            for drawn in inputs.values():
                _arm_program(programs, held, role, fn, drawn, reference=reference, written=written)
    if programs:
        from dew.nn.kernels.templates import report

        compiled = _arm_compile(compile, as_written(store.device().platform) if written else None)
        with summed_as_run():
            try:
                store.compile_site(armed(label, written), programs, held, compiled=compiled, arms=True)
            except report.CompiledOnDevice as refused:
                # Check 1 measures the arms as their GPU compiles them, a choice no compile without it
                # holds: every arm of the site compiles in its run, as the prepare states (`_stored_arm`).
                store.compile_site(armed(label, written), programs, held, compiled=compiled, arms=True,
                                   at_run=str(refused))


def _arm_program(programs: dict, held: dict, role: str, fn, inputs, *, reference: bool,
                 written: bool = False):
    """Into `programs` and `held`, arm `fn`'s program at a draw's `inputs` as the gate compiles it
    (`arm_forms._compiled`): within a run-to-run declaration and a window loop (at one window's inputs), a
    reference unrewritten but one of Dew's own paths (an arm that holds a program, `_holds_program`). Traced
    as its run traces it, and again where its compile lowers it (a launch a spill refused derived at its next
    config), its constants the first trace's, by place (`_hold`), and the routes that trace reported; held
    under `role`, its name with its fingerprint (`arm_role`). Its outputs' shapes and dtypes at `inputs`, its
    trace's (a window loop's at one window's). A twin compiled as written (`written`,
    `arm_forms._written_twin`) traces a `without_ynn` program's own function (`arm_forms._own_function`), as
    its run compiles it."""
    # The kernels' records read where an arm is prepared, never at load: a test of the gate imports no kernel
    # (the test cache keys a row on every module it loads).
    from dew.nn.kernels import generation, tuning

    if isinstance(fn, _RunToRun):
        return _arm_program(programs, held, role, fn.fn, inputs, reference=reference, written=written)
    if isinstance(fn, _Windowed):
        return _arm_program(programs, held, role, fn.fn, jax.tree.map(lambda leaf: leaf[0], inputs),
                            reference=reference, written=written)
    fn = _own_function(fn) if written else fn
    name = named(role, inputs)
    traced_fn = unrewritten(fn) if reference and not _declared(fn) else fn
    if getattr(fn, "without_ynn", False) and hasattr(fn, "trace"):
        # A jit of its own compiler options (`without_ynn`'s: a library arm's cuBLAS,
        # `generation_support.cublas_compiled`), which JAX lowers only as a top-level program: held whole
        # (`_Whole`), never traced inside another, as the gate runs it as it is (`arm_forms._compiled`), a
        # reference's trace and lowering unrewritten as its call is (`Unrewritten`'s).
        leaves = jax.tree.leaves(inputs)
        if list(inputs) != leaves:
            raise ValueError(f"{name}: an arm compiled with its own compiler options is held whole, over its "
                             "inputs as arrays, and these inputs are a tree of them")
        with generation.routes() as reported, tuning.seen() as launched:
            out = traced_fn.trace(*inputs).out_info
        held[name] = {"constants": [], "tree": jax.tree.structure(out), "routes": list(reported),
                      "launches": launched.held()}
        programs[name] = (_Whole(traced_fn, held[name]), [*map(jax.typeof, leaves)])
        return out
    with generation.routes() as reported, tuning.seen() as launched:
        closed, shapes = jax.make_jaxpr(traced_fn, return_shape=True)(*inputs)
    structure = jax.tree.structure(inputs)
    schema = [jax.typeof(constant) for constant in closed.consts]

    def again(flat):
        # Its lowering's own trace, whose constants are the program's held (a relowering's, past a spill,
        # derives its launches afresh: a halved chunk's sequence table), never the first trace's; refused
        # where they are other arguments than the program was compiled to take.
        with tuning.seen() as relaunched:
            retraced = jax.make_jaxpr(traced_fn)(*jax.tree.unflatten(structure, flat))
        if [jax.typeof(constant) for constant in retraced.consts] != schema:
            raise ValueError(f"{name}: relowered, its program holds other constants than its first trace's, "
                             "which its compile ahead takes as its arguments")
        # The launches of the lowering compiled, which its run reports (`CompiledTruths.ran`): a relowering
        # past a spill takes another config than the first trace's.
        held[name]["constants"] = [*map(np.asarray, retraced.consts)]
        held[name]["launches"] = relaunched.held()
        return retraced

    # The routes its trace reported and the launches it took, which its run, calling the program and tracing
    # none, reports at each call (`truths_ahead.CompiledTruths.ran`), until a lowering of it takes its own
    # (`again`).
    held_now = {"routes": list(reported), "launches": launched.held()}
    _hold(programs, held, name, closed, jax.tree.structure(shapes), inputs, again)
    held[name].update(held_now)
    return shapes


@dataclasses.dataclass(frozen=True)
class _Whole:
    """An arm's program held whole (`_arm_program`): `fn`, a jit of its own compiler options, which the
    store's compile lowers as itself (`_arm_compile`) over its inputs' leaves, each lowering's launches into
    its `record` (`lower`): its run reports the kept lowering's, as an arm's relowering does (`again`)."""

    fn: Callable
    record: dict = dataclasses.field(compare=False, hash=False)

    def __call__(self, *leaves):
        return self.fn(*leaves)

    def lower(self, *args, **options):
        from dew.nn.kernels import tuning

        with tuning.seen() as found:
            lowered = self.fn.lower(*args, **options)
        self.record["launches"] = found.held()
        return lowered


def _hold(programs: dict, held: dict, name: str, closed, tree, inputs, again: Callable) -> None:
    """Into `programs` and `held` under `name`, a traced program (`closed`, its outputs' `tree`) as a function
    of its constants and `inputs`' leaves, flat, its program `again(inputs' leaves)`'s where its compile
    traces it, and its constants and tree held beside it: no executable serializes an array its program
    closed over (`truths_ahead`)."""
    count = len(closed.consts)

    def run(*flat):
        return tree.unflatten(jax.core.eval_jaxpr(again(flat[count:]).jaxpr, flat[:count], *flat[count:]))

    programs[name] = (run, [*map(jax.typeof, closed.consts), *map(jax.typeof, jax.tree.leaves(inputs))])
    held[name] = {"constants": [*map(np.asarray, closed.consts)], "tree": tree}


def _stored_arms(store: CompiledTruths, label: str, dew_fn, reference_fn, also: Mapping, compiled: tuple):
    """A site's arms as compiled (`arm_forms._arms_with`'s `compiled`) whose every call is its draw's program
    its prepare compiled into `store` (`_ahead_arm`), each named as its prepare named it, before any call of
    it (`_compile_arms`); one no statement holds runs unprepared, named, and one that holds no program as it
    is: the run's verdict's and a qualification run's sizing's (`reference_error._qualified`) alike."""
    return (_stored_arm(store, label, dew_fn, compiled[0], "dew"),
            _stored_arm(store, label, reference_fn, compiled[1], "reference"),
            [_stored_arm(store, label, fn, arm, f"further {name}")
             for (name, fn), arm in zip(also.items(), compiled[2], strict=True)])


def _stored_arm(store: CompiledTruths, label: str, given_fn, arm, role: str, *, written: bool = False):
    """A site's arm `role` (`given_fn` as the site passed it, `arm` as compiled) whose every call is its
    draw's program its prepare compiled into `store` (`_ahead_arm`), named as its prepare named it; one no
    statement holds, or whose site's arms its prepare did not finish (its truths written, a failure before
    its arms': a flow's cell its prepare deferred, `gpu_flow.run_cells`), runs unprepared, named, and one
    that holds no program as it is (`_stored_arms`). One of a site its prepare wrote at run (`at_run`, before
    `holds`) runs as given, compiled on its GPU, named (`truths_ahead.AT_RUN`), nothing unprepared. A twin
    compiled as written (`written`) is held apart from the arms compiled alike (`armed`)."""
    if not _holds_program(given_fn):
        return arm
    named_role = arm_role(role, given_fn, label)
    if named_role is None:
        store.unprepared(f"{label}: its {role} arm has no statement of what it computes")
        return arm
    at_run = store.at_run(armed(label, written))
    if at_run is not None:
        # Compiled on its GPU in the run, as its prepare stated (`_compile_arms`).
        store.unprepared(f"{label}: its {role} arm: {at_run}", AT_RUN)
        return arm
    if not store.holds(armed(label, written)):
        store.unprepared(f"{label}: its {role} arm ran unprepared: its prepare wrote the site's truths and "
                         "compiled none of its arms")
        return arm
    return _ahead_arm(given_fn, arm, functools.partial(store.ran, armed(label, written)), named_role)


def _holds_program(fn) -> bool:
    """Whether arm `fn`, within a run-to-run declaration and a window loop, is one the store holds programs
    of (`_compile_arms`): none an arm the gate runs as given or one running Dew's own programs."""
    while isinstance(fn, (_RunToRun, _Windowed)):
        fn = fn.fn
    return not (_given(fn) or getattr(fn, "dew_programs", False))


def _ahead_arm(given, compiled, call: Callable, role: str):
    """A site's arm (`given`, as the site passed it) whose programs a store compiled ahead (`_compile_arms`),
    as the gate runs it (`compiled`, `arm_forms._compiled`'s): each plain program's call its draw's executable
    (`call(role, inputs)`, `truths_ahead.CompiledTruths.ran`), within a run-to-run declaration and a window
    loop (an arm that holds a program, `_holds_program`)."""
    if isinstance(given, _RunToRun):
        return _RunToRun(_ahead_arm(given.fn, compiled.fn, call, role), given.why)
    if isinstance(given, _Windowed):
        return _Windowed(_ahead_arm(given.fn, compiled.fn, call, role))
    return _Ahead(compiled, functools.partial(call, role))


class _Ahead:
    """A compiled arm (`compiled`) whose every call is its draw's executable compiled ahead (`run(inputs)`,
    `_ahead_arm`), its other attributes the arm's (its marks, its mode)."""

    def __init__(self, compiled, run: Callable):
        self.compiled, self.run = compiled, run

    def __call__(self, *inputs):
        return self.run(inputs)

    def __getattr__(self, name):
        return getattr(self.compiled, name)


def drawn_on_the_host(inputs_of: Callable, seed: int):
    """`inputs_of(seed)` drawn on the host's CPU, as a recording's inputs were, whatever device the test runs
    on: jax.random's draws are the same bits on every device but its normal's float transform is not (a
    recorded site's seed drawn on an H100 had another digest than on the CPU, the recording's, in 157 of the
    recurrence GPU flow's cells, draw-digest at 996fa29f1), so a draw on the test's device would not be the
    recorded inputs. The arrays then go to the device where the run reads them."""
    with jax.default_device(jax.devices("cpu")[0]):
        return inputs_of(seed)
