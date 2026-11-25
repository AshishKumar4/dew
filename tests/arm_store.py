"""A check-1 site's arms compiled ahead into a store of programs (`truths_ahead.CompiledTruths`, a GPU tier's
prepare) and run from it (`_stored_arms`): each arm's program traced as its run traces it and compiled as the
gate compiles it (`arm_forms._compiled`), its references' twins as written (`arm_forms._written_twin`) held
apart."""

import contextlib
import dataclasses
import functools
import itertools
import os
import warnings
from collections.abc import Callable, Iterable, Mapping

import jax
import numpy as np
from arm_arithmetic import as_written, drawn_on_the_host, products_held, summed_as_run, unrewritten, widened
from arm_forms import (
    _compile_for,
    _declared,
    _given,
    _holds_program,
    _own_function,
    _refuse_numpy_arms,
    _RunToRun,
    _Windowed,
    _written_references,
    _written_role,
)
from check1_wire import QUALIFY
from float64_terms import POWER, evaluated, evaluated_on, static, traced
from reference_error import _of_float64, _site, _SiteTerms, _sizing_seeds, _truth_program
from store_wire import COMPILED
from tier_wire import stop_past_cap
from truths_ahead import AT_RUN, CompiledTruths, arm_role, named

from dew.parity_stages import staged_units


def armed(label: str, written: bool = False) -> str:
    """The label a site's arms are held under in a store of programs compiled ahead (`_compile_arms`), apart
    from its truths' (`compile_site`): each set compiled in its own arithmetic, the arms'
    without x64, and its references' twins as written (`written`) apart from its arms compiled alike."""
    return f"{label}, its arms as written" if written else f"{label}, its arms"


def _arm_compile(compile: Callable, written: dict | None) -> Callable:
    """How the gate compiles a site's arms (`arm_forms._arms`), its results placed
    (`dew.training.ahead.compile_programs`' `compiled`): by `compile` (`arm_forms._compile_for`'s: as Dew's
    users compile them, `jax.jit`, beside an arm run as given, else without YNNPACK's fusions, `without_ynn`),
    with the `written` options where they are a twin's (`arm_forms._written_twin`), its products held
    (`arm_arithmetic.products_held`)."""
    def compiled(function, *, out_shardings):
        # An arm held whole (`Whole`) is a jit of its own compiler options, lowered as itself: no other jit
        # may hold it.
        if isinstance(function, Whole):
            return function
        return compile(function if written is None else products_held(function), out_shardings=out_shardings,
                       compiler_options=written)

    return compiled


_CONCRETE = (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError,
             jax.errors.TracerIntegerConversionError, jax.errors.TracerBoolConversionError)
"""What a draw raises traced abstractly (`jax.eval_shape`) where it reads a concrete value of what it draws:
its shapes decided on the host by those values, which only its real draw gives."""


class ConcreteDraws(UserWarning):
    """A prepared site whose draws read a concrete value of what they draw (`_CONCRETE`), each of them drawn
    in full to learn its shapes: its programs are right, its prepare's time spent on those draws."""


def _first_draws(draw, seeds: Iterable[int], label: str) -> list:
    """The real draw (`drawn_on_the_host`) of each of `seeds` whose shapes no earlier seed's draw had, in seed
    order: each seed's shapes read from its draw traced abstractly (`jax.eval_shape`: no array made), its real
    draw made for a new one alone, so site `label`'s programs are those of these draws, each its first seed's,
    as every draw's would be (row 12431: a Gemma vocabulary's 1,280 draws of a 64 MiB head, all for one
    program). The seeds drawn in full to learn their shapes are counted, a `ConcreteDraws` warning."""
    firsts, concrete, seen = [], 0, set()
    for seed in seeds:
        # A dry prepare past its cap stops before a draw (`tier_wire.stop_past_cap`).
        stop_past_cap()
        try:
            shapes = jax.eval_shape(functools.partial(drawn_on_the_host, draw, seed))
        except _CONCRETE:
            # A draw that reads a concrete value decides its shape on the host: its real draw keys it.
            concrete += 1
            firsts.append(drawn_on_the_host(draw, seed))
            continue
        leaves, tree = jax.tree.flatten(shapes)
        key = (str(tree), tuple((leaf.shape, str(leaf.dtype), leaf.weak_type) for leaf in leaves))
        if key not in seen:
            seen.add(key)
            firsts.append(drawn_on_the_host(draw, seed))
    if concrete:
        warnings.warn(f"{label}: {concrete} of its draws read a concrete value of what they draw, each drawn "
                      f"in full to learn its shapes", ConcreteDraws, stacklevel=2)
    return firsts


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
    staged = itertools.chain(range(staged_units(seeds)), sizing)
    for drawn in _first_draws(draw, staged, armed(label, written)):
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
        # (`Whole`), never traced inside another, as the gate runs it as it is (`arm_forms._compiled`), a
        # reference's trace and lowering unrewritten as its call is (`Unrewritten`'s).
        leaves = jax.tree.leaves(inputs)
        if list(inputs) != leaves:
            raise ValueError(f"{name}: an arm compiled with its own compiler options is held whole, over its "
                             "inputs as arrays, and these inputs are a tree of them")
        with generation.routes() as reported, tuning.seen() as launched:
            out = traced_fn.trace(*inputs).out_info
        held[name] = {"constants": [], "tree": jax.tree.structure(out), "routes": list(reported),
                      "launches": launched.held()}
        programs[name] = (Whole(traced_fn, held[name]), [*map(jax.typeof, leaves)])
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
class Whole:
    """An arm's program held whole (`_arm_program`): `fn`, a jit of its own compiler options, which the
    store's compile lowers as itself (`_arm_compile`) over its inputs' leaves, each lowering's launches into
    its `record` (`lower`): its run reports the kept lowering's, as an arm's relowering does (`again`)."""

    fn: Callable
    record: dict = dataclasses.field(compare=False, hash=False)

    def __call__(self, *leaves):
        return self.fn(*leaves)

    def trace(self, *args):
        """Its program traced as itself, a top-level jit's trace (what the store reads of its program before
        compiling it: `truths_ahead._calls_the_host`), never inside another trace, which its compiler
        options refuse (row 12062, #8613's receipt)."""
        return self.fn.trace(*args)

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


def _ahead(programs: dict, held: dict, label: str, truth, terms_fn, *, terms, wide) -> None:
    """Into `programs` and `held` (`truths_ahead.CompiledTruths.compile_site`), the programs a site's draw of
    widened inputs `wide` takes: its truth's (`truth`; a window loop's on one window, its every window's
    shapes), unless the gate runs it as given or replays it (`_given`); its own terms program's
    (`_SiteTerms`), or the float64 walk of its truth its rows may be sized by (`terms`, unless False: the walk
    a site's first draw decides on, `_cancelling_terms`), where the walk is static: one that takes a branch
    or loop its values pick is read as it runs, so its run walks it on its GPU where it needs it, named
    (`float64_terms._termed`). Each traced as written, its constants arguments held beside it."""
    if isinstance(truth, _Windowed):
        # Its truth window by window; its terms on the whole draw, as its run reads them: a site's own
        # program, or the walk of its window's function (`terms=True`, `_site`).
        window = jax.tree.map(lambda leaf: leaf[0], wide)
        _ahead(programs, held, label, truth.fn, None, terms=False, wide=window)
        if isinstance(terms_fn, _SiteTerms):
            _held_program(programs, held, "site terms", terms_fn.fn, wide)
        elif terms is True and not _given(truth):
            _held_walk(programs, held, truth.fn, wide)
        return
    if not _given(truth):
        _held_program(programs, held, "truth", truth, wide)
    if isinstance(terms_fn, _SiteTerms):
        _held_program(programs, held, "site terms", terms_fn.fn, wide)
    elif terms is not False and not _given(truth):
        _held_walk(programs, held, truth, wide)


def _held_walk(programs: dict, held: dict, fn, wide) -> None:
    """`fn`'s float64 terms walk at `wide` (`float64_terms`) as a program of its constants and inputs, flat,
    its terms alone, beside `fn`'s truth at `wide` (`_held_program`), which its values are: two programs a
    draw runs in turn, each within a GPU part's memory where the walk's values and terms together were not
    (a sinks flow's training step: 1.33 GiB and 1.71 GiB against 2.86 GiB, its worker's 2.84). Where the
    walk is static; held as one its run walks as it runs where its values steer it."""
    closed, _ = traced(fn, wide)
    name = named(f"terms {POWER.get()}", wide)
    if not static(closed.jaxpr):
        held[name] = {"walked": "its truth takes a branch or loop its values pick, which its terms' walk "
                                "reads as it runs"}
        return
    if named("truth", wide) not in programs:
        _held_program(programs, held, "truth", fn, wide)
    count = len(closed.consts)

    def walked(*flat):
        return evaluated(closed.jaxpr, flat[:count], list(flat[count:]))[1]

    programs[name] = (walked, [*map(jax.typeof, closed.consts), *map(jax.typeof, jax.tree.leaves(wide))])
    held[name] = {"constants": [*map(np.asarray, closed.consts)]}


def _held_program(programs: dict, held: dict, role: str, fn, wide) -> None:
    """`fn` at `wide` as a program of its constants and inputs, flat (`truths_ahead.CompiledTruths`), traced
    as written (`float64_terms.traced`)."""
    closed, tree = traced(fn, wide)
    _hold(programs, held, named(role, wide), closed, tree, wide, lambda flat: closed)


def _shaped_truth(truth, wide):
    """`truth` (a traced one, not one the gate runs as given) on widened inputs `wide` as `_of_float64` reads
    it, its leaves' dtypes and sizes: its shapes alone (a window loop's on one window), no float64
    computed."""
    if isinstance(truth, _Windowed):
        return jax.eval_shape(unrewritten(truth.fn), *jax.tree.map(lambda leaf: leaf[0], wide))
    return jax.eval_shape(unrewritten(truth), *wide)


def compile_site(store: CompiledTruths, dew_fn, reference_fn, draw, label: str, *, seeds: int, truth_fn,
                  terms, also, orders, plain: bool, library_fn=None) -> None:
    """`assert_as_exact`'s site prepared into `store` (`truths_ahead.compiled_truths`): every program its
    truths and terms take (`_ahead`), at each shape the draws of its stages take, compiled for the devices its
    run computes on, and its truth held to float64 on its first draw's shapes (`_of_float64`); and each of its
    arms' (`_compile_arms`), at a qualification run's sizing draws too where its verdict is the `plain` pair's
    (`_qualified`); no truth computed and no arm run. A truth the gate runs as given or replays (a
    recording) has no program."""
    with store.preparing(label, armed(label)):
        terms_fn, also = _site(dew_fn, reference_fn, label, seeds=seeds, truth_fn=truth_fn, library_fn=None,
                               library_row=None, library_leaves=(), also=also, terms=terms)
        truth = truth_fn if truth_fn is not None else reference_fn
        # One draw of each shapes' (`_first_draws`), widened to name its truth's program: widening is its
        # draw's shapes' alone, so the programs are the ones every draw widened would name, each the first
        # draw of its shapes', with no float64 copy of every staged draw made.
        shaped = {}
        for inputs in _first_draws(draw, range(staged_units(seeds)), label):
            shaped.setdefault(named("truth", widened(inputs)), inputs)

        # The algebra's switch and the kernels' report read where a site prepares, never at load: a test of
        # the gate imports no kernel (the test cache keys a row on every module it loads).
        from dew.algebra import passes
        from dew.nn.kernels.templates import report

        programs, held = {}, {}
        with (jax.enable_x64(new_val=True), passes(on=False), jax.default_device(store.device()),
              evaluated_on([])):
            for index, inputs in enumerate(shaped.values()):
                wide = widened(inputs)
                # A truth the gate runs as given has its value alone to read, held to float64 in its run on
                # its first draw (`_drawn`): the prepare computes none (bible section 8; on its default
                # device, a compile-only GPU, its eager operations raised "'Executable' object has no
                # attribute 'local_devices'", row 4881's class).
                if index == 0 and not _given(truth):
                    _of_float64(label, _truth_program(truth), inputs, _shaped_truth(truth, wide))
                _ahead(programs, held, label, truth, terms_fn, terms=terms, wide=wide)
            try:
                store.compile_site(label, programs, held)
            except report.CompiledOnDevice as refused:
                # A truth its target's GPU would leave a choice of to run time (a convolution's algorithm,
                # `report.at_run_time`), which no compile without that GPU holds: its run compiles its truths
                # on the GPU, their float64 on its FP64 units, as its arms (`_compile_arms`), named
                # (`truths_ahead.running`); the choice moves a float64 truth by float64's rounding alone
                # (row 8576).
                store.compile_site(label, programs, held, at_run=str(refused))
        # A qualification run sizes a plain site first (`_qualified`), on its sizing draws through its arms.
        sizing = _sizing_seeds(reference_fn) if os.environ.get(QUALIFY) and plain else range(0)
        compile = _compile_for(dew_fn, reference_fn)
        # A library arm (`library_fn`) one of them, run on Dew's own inputs as a further reference is (row
        # 8468).
        _compile_arms(store, {"dew": (dew_fn, False), "reference": (reference_fn, True),
                              **({} if library_fn is None else {"library": (library_fn, True)}),
                              **{f"further {name}": (fn, True) for name, fn in also.items()}},
                      draw, label, seeds=seeds, orders=orders, sizing=sizing, compile=compile)
        # Each reference its accuracy claim holds Dew's arm against as written, compiled so (`_written_twin`).
        twins = {_written_role(name): (fn, True) for name, fn in _written_references(reference_fn, also)}
        _compile_arms(store, twins, draw, label, seeds=seeds, orders=orders, compile=compile, sizing=sizing,
                      written=True)


def stored_call(fn: Callable, label: str, inputs, *, host: bool, unprepared: Callable):
    """`reference_error.ahead_arm`'s call of `fn` at `inputs` within a store (`store_wire.COMPILED`): in a
    prepare traced and compiled as a site's arms are and its zeros given, in a run its stored executable, one
    no statement holds or its prepare did not reach `unprepared(*inputs)`, recorded."""
    store = COMPILED.get()
    role = arm_role("dew", fn, label)
    if role is None:
        store.unprepared(f"{label}: its arm has no statement of what it computes")
        return unprepared(*inputs)
    # Its placement is its program's too: a host's arm and a device's of one function are two programs.
    role = f"{role} on the host" if host else role
    if store.prepare:
        if armed(label) in store.arms:
            return store.arm_again(armed(label), named(role, inputs))
        programs: dict = {}
        held: dict = {}
        # A host-placed arm traced, and retraced where its compile lowers it, on the host's CPU.
        with (store.preparing(armed(label)),
              jax.default_device(jax.devices("cpu")[0]) if host else contextlib.nullcontext()):
            with contextlib.nullcontext() if host else store.tracing():
                shapes = _arm_program(programs, held, role, fn, inputs, reference=False)
            store.compile_site(armed(label), programs, held, compiled=_arm_compile(jax.jit, None),
                               arms=True, host=host)
        name = named(role, inputs)
        store.arms[armed(label)] = (name, shapes, held[name]["launches"])
        return store.arm_again(armed(label), name)
    if not store.holds(armed(label)):
        store.unprepared(f"{label}: its prepare did not reach this arm")
        return unprepared(*inputs)
    return store.ran(armed(label), role, inputs)
