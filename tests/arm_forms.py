"""How the gate builds a check's arms: run as given (`as_given`) or compiled, as a pair of Dew paths without
YNNPACK's fusions (`without_ynn`, `_arms`) or as Dew's users compile them, a reference unrewritten
(`_compiled`); and how it runs an arm of one draw beyond one call: over a site's windows of stacked draws
(`windows`, `over_windows`) and declared to vary from run to run (`run_to_run`), each an adapter
`reference_error`'s check 1 reads through (`_layers`)."""

import contextlib
import contextvars
import dataclasses
import functools
import os
from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from arm_arithmetic import Unrewritten, Written, as_written, unrewritten
from check1_judge import SWITCHED
from check1_wire import CENSUS


def _stack_draws(trees):
    """A draw's leaves of each window stacked on a new leading axis. A JAX array or Python scalar stacks
    as JAX stacks it, its weak type kept, and a host array as NumPy stacks it, its dtype kept."""
    def stack(*leaves):
        if all(isinstance(leaf, jax.Array | bool | int | float | complex) for leaf in leaves):
            return jnp.stack([jnp.asarray(leaf) for leaf in leaves])
        return np.stack([np.asarray(leaf) for leaf in leaves])
    return jax.tree.map(stack, *trees)


def stacked_on_host(trees):
    """Outputs (an arm's of each window, a site's of each example) stacked on a new leading axis as host
    arrays of their own dtypes."""
    return jax.tree.map(lambda *leaves: np.stack([np.asarray(leaf) for leaf in leaves]), *trees)


def _window_count(arguments) -> int:
    """The windows an arm's stacked arguments hold, the leading axis of each."""
    leaves = jax.tree.leaves(arguments)
    count = len(leaves[0]) if leaves else 0
    if count < 1:
        raise ValueError(f"an arm over {count} windows")
    return count


def windows(draw, count: int):
    """A check-1 draw of `count` windows a seed. Seed `s` gives the site's draws of seeds `s * count` to
    `s * count + count - 1`, each leaf stacked on a new leading axis, and every argument is a window. A
    seed's evidence then pools `count` draws, as independent as the site's draws of different seeds
    are. A leaf that is one row in a draw, such as a parameter's gradient, becomes `count` rows. Its
    arms take the windows through `over_windows`."""
    if count < 1:
        raise ValueError(f"windows of {count} draws")

    def drawn(seed):
        return _stack_draws([draw(seed * count + index) for index in range(count)])
    return drawn


@dataclasses.dataclass(frozen=True)
class _Windowed:
    """An arm of one draw (`fn`) over `windows`' stacked arguments, window by window (`over_windows`):
    the gate compiles `fn` as it compiles any arm (`_arms`) and runs the windows through that program. A
    frozen record, stated by its arm (`truths_ahead.arm_role`)."""

    fn: Callable

    def __call__(self, *arguments):
        return stacked_on_host([self.fn(*jax.tree.map(lambda leaf, index=index: leaf[index], arguments))
                            for index in range(_window_count(arguments))])


@dataclasses.dataclass(frozen=True)
class _RunToRun:
    """An arm whose outputs vary in their last bits from run to run on the same inputs (`run_to_run`):
    the gate compiles `fn` as it compiles any arm (`_arms`) and judges each draw by the one run it makes. A
    frozen record, stated by its arm and its reason (`truths_ahead.arm_role`)."""

    fn: Callable
    why: str

    def __call__(self, *inputs, **options):
        return self.fn(*inputs, **options)

    def __getattr__(self, name):
        # Its arm's marks and mode; its own field, before it is set, and no dunder (a jit's `__wrapped__`,
        # which would state the record as the wrapper it is not, `dew._stated.stated`).
        if name == "fn" or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.fn, name)


def _layers(fn):
    """`fn` and every arm it adapts, outermost first: through a declaration (`_RunToRun`), a window loop
    (`_Windowed`) and a reference's unrewritten scope (`Unrewritten`)."""
    while True:
        yield fn
        if isinstance(fn, _RunToRun | _Windowed):
            fn = fn.fn
        elif isinstance(fn, Unrewritten):
            fn = fn.wrapped
        else:
            return


def _varies(fn) -> bool:
    """Whether `fn`, or an arm it adapts, is declared to vary from run to run (`run_to_run`)."""
    return any(isinstance(layer, _RunToRun) for layer in _layers(fn))


def run_to_run(fn, why: str):
    """`fn`, an arm declared to vary from run to run on a GPU on the same inputs, for `why`: an
    accumulation in arrival order (atomic adds), which a deterministic form would make 2-2.7 times as slow
    (row 4448). Check 1 pairs it with the other arm within each draw's one run, as it pairs every
    arm, and never holds it to a run of itself again (`_of_its_inputs`) but on a CPU, where its interpreted
    program is a function of its inputs: a draw's evidence is one run's, and the redraw and every further
    seed are runs of their own, so its spread across runs is the spread across draws check 1 weighs. A
    window loop's arm (`over_windows`) is declared inside its loop, which the gate reads outermost."""
    if isinstance(fn, _Windowed):
        return _Windowed(_RunToRun(fn.fn, why))
    return _RunToRun(fn, why)


def over_windows(fn):
    """`fn`, an arm of one draw, run over `windows`' stacked arguments window by window, each window's
    arguments sliced from the stack, with its outputs stacked on their leading axis as host arrays.

    Each window runs the program one draw runs, never a batched one: a batch axis changes the program
    XLA compiles (its fusions, its reductions' order, its dots' batching, a kernel's route), which XLA
    holds to no common bits with one draw's, so check 1 would judge a program users never run. The gate
    compiles `fn` as it compiles an arm of one draw (`_arms`), once a signature, and every window calls
    that program; a torch or NumPy reference passes `as_given`. The loop keeps `fn`, which a recorded
    truth's rounding count traces on one window (`indexed_recording`)."""
    return _Windowed(fn)


def _one_draws_program(fn, arguments):
    """`fn` and its `arguments` as one draw's program reads them: a window loop's (`over_windows`) own
    function on its first window's arguments, any other function on them all."""
    if not isinstance(fn, _Windowed):
        return fn, arguments
    return fn.fn, jax.tree.map(lambda leaf: leaf[0], arguments)


WITHOUT_YNN = {"xla_cpu_experimental_ynn_fusion_type": ""}
"""XLA:CPU's compiler options with every YNNPACK fusion off (dots, reductions, element-wise and layout
ops alike), so that a program runs on XLA's own code, compiled for the host's instruction set with no
dispatch at run time. YNNPACK picks its microkernels at run time by that instruction set, and both
the interpreted SSD kernel and XLA's scan fuse parts of themselves to it by default, which moved
check 1 between the two across PARITY by host; without the fusions the verdicts held on two hosts
of different instruction sets, though the bits still part with the host (docs/performance.md, "The
SSD kernel's host check 1 without YNNPACK"); and XLA:CPU's fusion choices are not Dew's (rows 4520, 4522).
The empty list, not the documented `-individual_dot`, because compiler options take no such spelling and
that one leaves a scan's reductions fused. The census (`CENSUS`) compiles without it (`switched_off`)."""


def switched_off() -> dict:
    """WITHOUT_YNN, or none under the census (`CENSUS`), read when the gate compiles."""
    return {} if os.environ.get(CENSUS) else WITHOUT_YNN


def without_ynn(fn, **jit_options):
    """`fn` jitted with WITHOUT_YNN on the CPU lane, where the context places a call
    (`generation.placed_platform`: a GPU tier part's prepare, a CPU process, places on its compile-only
    GPU, and its run on the GPU, so both build one arm, which their statements name alike), and plain
    `jax.jit` elsewhere: how the gate compiles both arms of a
    check 1 between two Dew paths (`_arms`), such as an interpreted kernel and the XLA program it
    replaces, so that the two run the same kind of code on any host. A program Dew's users run compiles
    with the fusions, as they do. On the CPU lane its compiler options refuse to be nested in another
    jit, differentiated through or batched, so the gate compiles an arm whole; a site passes its arms
    as functions and never calls this. A caller's own compiler options take precedence over
    WITHOUT_YNN's; `fn` already compiled so is returned as it is."""
    from dew.nn.kernels.generation import placed_platform

    if getattr(fn, "without_ynn", False):
        return fn
    if placed_platform() == "cpu":
        jit_options["compiler_options"] = {**switched_off(), **(jit_options.get("compiler_options") or {})}
    compiled = jax.jit(fn, **jit_options)
    compiled.without_ynn = True
    return compiled


def as_recorded[Arm: Callable](fn: Arm, pilots: range | None = None) -> Arm:
    """`fn`, marked an arm whose values are read from recordings (`indexed_recording`'s and a recorded
    sizing's), the mark the gate reads (`getattr(fn, "recorded", False)`): it holds its recorded seeds alone,
    so no paired sizing or orbit draws past them, and it is run as given, never rewritten; with `pilots`, the
    seeds past its verdict's it holds too, a sizing's (`getattr(fn, "pilots", None)`). Marked in place, `fn`
    itself returned, its kind kept (a function, a partial, an arm's object)."""
    vars(fn)["recorded"] = True
    if pilots is not None:
        vars(fn)["pilots"] = pilots
    return fn



HELD: contextvars.ContextVar[Callable | None] = contextvars.ContextVar("held", default=None)
"""The site's `held` (`reference_error.assert_as_exact`), called on every result of Dew's arm a draw of it
makes (`_drawn`), wherever check 1 draws it: a stage, a sizing, a map task, a prepared run."""


@contextlib.contextmanager
def holding(held: Callable | None):
    """`held` (`HELD`) for the site judged within, where given."""
    token = HELD.set(held)
    try:
        yield
    finally:
        HELD.reset(token)


class _RoundedTruth:
    """`ROUNDED`, a reference that is check 1's own float64 truth of each draw (its live or prepared run,
    `reference_error._draw_truth`) rounded once to float32 (`rounded_like`), formed from that truth by the
    gate (`reference_error._arm_ran`) and never run: an arm run as given, so Dew's arm compiles as its users
    run it, with the fusions (`_arms`), and holds no program a store compiles. The reference of draws whose
    truth float32 holds exactly or rounds once, judged by check 1's parity as any reference is: where the
    truth is a float32 the reference's error is zero and Dew must equal it; where it rounds, Dew is held
    within PARITY of that one rounding's error (not bit for bit: a tie rounded away from even has the
    reference's error). A second float64 run of the truth, computed under the arms' arithmetic preset (a
    CPU's float32 sums), was a second owner of check 1's truth and computed it otherwise (rows 9481, 9515,
    9532)."""

    as_given, recorded, rounded, programs = True, False, True, None

    def __repr__(self) -> str:
        return "ROUNDED"

    def __call__(self, *inputs, **options):
        raise TypeError("ROUNDED is check 1's own truth of a draw rounded once, which the gate forms from "
                        "that truth (`reference_error._arm_ran`); it is never run")


ROUNDED = _RoundedTruth()


def rounded_like(truth, made):
    """`truth`'s float64 leaves each rounded once to float32 (NumPy's cast: to nearest, ties to even), the
    dtype of `made`'s leaf there (Dew's arm's result), whose tree it is: refused for a leaf of another dtype,
    which no one rounding of a float64 reaches as a float32's does."""
    def once(true, leaf):
        if np.dtype(leaf.dtype) != np.float32:
            raise TypeError(f"ROUNDED rounds a float64 truth to float32 once; Dew's leaf is {leaf.dtype}")
        return np.asarray(true, np.float64).astype(np.float32)

    return jax.tree.map(once, truth, made)


def refuse_rounded(label: str, *references, truth_fn, followed=None, conditioning=None,
                   beside_dew: bool = True) -> None:
    """Refuse `ROUNDED` among `label`'s `references` (its reference and further ones) without the `truth_fn`
    it rounds (it has none of its own), beside `followed` roundings (each arm's own truth) or a weighed check
    (`conditioning`, its reference's own rounding), or with no arm of Dew's beside it (`beside_dew`: a sizing
    or null run of the reference against itself), where it would be the arm run; and refuse it adapted (any
    layer of a reference, `_layers`), which the gate cannot see through to form it."""
    rounded = [fn for fn in references if any(layer is ROUNDED for layer in _layers(fn))]
    if any(fn is not ROUNDED for fn in rounded):
        raise ValueError(f"{label}: ROUNDED is given bare, never adapted (a window loop, a run-to-run "
                         f"declaration): the gate forms it from the truth, and an adapter would call it")
    if rounded and (
            truth_fn is None or followed is not None or conditioning not in (None, False) or not beside_dew):
        raise ValueError(f"{label}: ROUNDED is the site's float64 truth rounded once, beside an arm of "
                         f"Dew's: it takes `truth_fn`, and no followed roundings, weighed check or run of "
                         f"itself")


def as_given(fn, programs: Callable | None = None):
    """`fn`, an arm the gate runs as it is given and never compiles: a torch or NumPy reference, or a Dew
    arm that runs whole programs of its own (a trainer's fit, a placement on a mesh and the program
    run there, a host loop). The other arm of its check then compiles as Dew's users run it, with the
    fusions (`_arms`), as the programs this one runs do. `programs`, where `fn` is a float64 truth held to
    a recorded one (`certifier`): the programs a draw's run calls, `programs(*inputs)`, each a callable
    and its arguments, which the certifier traces to count the run's roundings (`_draws_roundings`)."""
    @functools.wraps(fn)
    def given(*inputs, **options):
        return fn(*inputs, **options)
    given.as_given, given.programs = True, programs
    return given


def dew_programs(fn):
    """`fn`, a Dew arm that runs Dew's own compiled programs (a trainer's step, a task's forward, a
    server's decode), which the gate cannot compile but which take their XLA options from Dew's one
    seam (`dew.nn.kernels.options.compiler_options`). Beside another Dew arm, the gate runs it with
    that seam holding WITHOUT_YNN open (`compiled_with`), so the programs it compiles go without
    YNNPACK's fusions as the gate's own compile of its other arm does; beside an arm `as_given`, as it
    is, with the fusions. A program it compiled before its check, or compiles by a bare `jax.jit`
    outside the seam, keeps its own options."""
    @functools.wraps(fn)
    def programs(*inputs, **options):
        return fn(*inputs, **options)
    programs.dew_programs = True
    return programs


def dew_path(fn):
    """`fn`, a reference that is one of Dew's own paths: the same model run another way (rows packed against
    each row alone, a remat policy against none, another layout), so its site's claim is invariance under
    what differs, not the algebra's rewrite. The gate runs it as it runs Dew's arm, under the rewrite
    (`_compiled`), so one function at one shape is one program for both arms (a decision head's `decide`,
    packed and alone, compiled once, where the reference unrewritten compiled it twice); the rewrite's
    correctness is held by the sites whose references are written apart from Dew, which run unrewritten. A
    reference undeclared is written apart. Only a site whose claim is invariance may declare it, as a
    declaration elsewhere drops the reference's independence (docs/design/algebra.md §3). It keeps the marks
    of what it wraps (`as_given`), and an adapter's (`run_to_run`, `over_windows`) it declares within, so
    the adapter stays the gate's to read; a float64 truth the gate takes from the reference stays unrewritten
    (`_drawn`, `_conditioning`)."""
    if isinstance(fn, _RunToRun):
        return _RunToRun(dew_path(fn.fn), fn.why)
    if isinstance(fn, _Windowed):
        return _Windowed(dew_path(fn.fn))

    @functools.wraps(fn)
    def path(*inputs, **options):
        return fn(*inputs, **options)
    path.dew_path = True
    return path


def _declared(fn) -> bool:
    """Whether reference `fn`, or an arm it adapts (`_layers`), is one of Dew's own paths (`dew_path`)."""
    return any(getattr(layer, "dew_path", False) for layer in _layers(fn))


def _given(fn) -> bool:
    """Whether the gate runs `fn` as it is given (`as_given`, a recorded reference), a window loop's
    arm by its own (`over_windows`)."""
    if isinstance(fn, _Windowed):
        return _given(fn.fn)
    return bool(getattr(fn, "as_given", False) or getattr(fn, "recorded", False))


def _compiles(fn) -> bool:
    """Whether the gate compiles `fn` itself: a plain function, not an arm it runs as given, a program
    already compiled (a jit) or an arm of Dew's own programs (`dew_programs`)."""
    fn = fn.fn if isinstance(fn, _Windowed) else fn
    return not (_given(fn) or getattr(fn, "dew_programs", False) or hasattr(fn, "lower"))


def _refuse_numpy_arms(draw, *arms) -> None:
    """Refuse an arm the gate would compile that computes in NumPy on a draw of NumPy inputs: called once as
    given on them (a window's, for a window loop's arm), an arm returning NumPy arrays is a NumPy reference,
    compiled computing in JAX, in float32 where its draw is float64; it passes `as_given`. A draw holding no
    NumPy array runs none: the gate compiles each plain arm (`_compiled`), whose trace refuses a NumPy call on
    its inputs (a model arm run eagerly here took a minute of a check, ledger 3774)."""
    plain = [fn for fn in dict.fromkeys(arms) if _compiles(fn)]
    drawn = draw(0) if plain else ()
    if not any(isinstance(leaf, np.ndarray | np.generic) for leaf in jax.tree.leaves(drawn)):
        return
    for fn in plain:
        inputs = drawn
        if isinstance(fn, _Windowed):
            fn, inputs = fn.fn, jax.tree.map(lambda leaf: leaf[0], drawn)
        outputs = jax.tree.leaves(fn(*inputs))
        kinds = {type(leaf).__name__ for leaf in outputs if not isinstance(leaf, jax.Array)}
        if kinds:
            raise TypeError(f"{getattr(fn, '__qualname__', repr(fn))} returns {sorted(kinds)}: a NumPy arm, "
                            f"which the gate would compile in JAX; pass it as_given, or return its JAX "
                            f"values unconverted")


def _compiled(fn, compile, pairing=None, *, reference: bool = False, written: bool = False):
    """`fn` as the gate runs it: as given (`_given`), as it is where it is a `without_ynn` program already
    (another jit around it would nest its compiler options), or compiled by `compile`, a window loop's arm
    within it (`over_windows`), without YNNPACK's fusions, paired by `pairing` (`_Paired`); a `reference`
    unrewritten (`arm_arithmetic.unrewritten`), but one of Dew's own paths (`dew_path`), which runs under the
    rewrite as Dew's arm does. `written`, compiled as written (`arm_arithmetic.as_written`: none of XLA's
    rewrites, its products held from its fusions, the twin an accuracy claim bounds Dew's arm by,
    `_written_twin`), a `without_ynn`
    program from its own function; else with them, as Dew's arm is."""
    run = unrewritten if reference and not _declared(fn) else _as_is
    if isinstance(fn, _RunToRun):
        return _RunToRun(_compiled(fn.fn, compile, pairing, reference=reference, written=written), fn.why)
    fn = _own_function(fn) if written else fn
    if _given(fn) or getattr(fn, "without_ynn", False):
        return run(fn)
    # As written for the platform each call runs on (`arm_arithmetic.Written`), read where it runs: an arm
    # running Dew's own programs, whose products Dew's compile point compiles beyond the gate's hold, without
    # fusions too (`as_written`'s `held`).
    options = functools.partial(as_written, held=False) if written else dict
    if getattr(fn, "dew_programs", False):
        plain = _with_options(fn, options) if written else fn
        return run(plain) if compile is jax.jit else _Paired(
            pairing, run(_with_options(fn, lambda: {**_switched_off_here(), **options()})), run(plain))
    if isinstance(fn, _Windowed):
        return _Windowed(_compiled(fn.fn, compile, pairing, reference=reference, written=written))
    if written:
        return run(Written(fn)) if compile is jax.jit else _Paired(pairing, run(Written(fn, compile)),
                                                                   run(Written(fn)))
    return run(compile(fn)) if compile is jax.jit else _Paired(pairing, run(compile(fn)), run(jax.jit(fn)))


def _own_function(fn):
    """`fn`'s own function where it is a jit (a plain one, one of compiler options of its own: a
    `without_ynn` program, a library arm's cuBLAS), which the gate compiles as written (`_compiled`'s
    `written`), so no jit of options nests in another, which JAX refuses (options on a program's outermost
    jit alone); else `fn`. An arm takes arrays alone, so a jit's static arguments are none of its call."""
    return fn.__wrapped__ if hasattr(fn, "lower") and hasattr(fn, "__wrapped__") else fn


def _as_is(fn):
    """`fn` itself: a candidate arm, run as its users run it."""
    return fn


@dataclasses.dataclass(eq=False)
class _Pairing:
    """Whether a check's paired arms (`_Paired`) run with YNNPACK's fusions, decided once a site before it
    draws (`_ynn_decided`): `refused`, XLA:CPU's refusal of an arm without them where it gave one."""

    decided: bool = False
    refused: str | None = None

    @property
    def with_ynn(self) -> bool:
        return self.refused is not None


@dataclasses.dataclass(frozen=True)
class _Paired:
    """An arm of a check between two Dew paths (`_arms`): `without`, the arm without YNNPACK's fusions,
    unless its site's arms were decided otherwise, then `plain`, the arm as Dew's users compile it
    (`_ynn_decided`): where XLA:CPU cannot compile an arm of the check without the fusions and can with
    them, every arm of it takes them, so the two still run the same kind of code, here the kind whose
    microkernels the host picks."""

    pairing: _Pairing
    without: Callable
    plain: Callable
    ahead: dict = dataclasses.field(default_factory=dict, compare=False)
    """`without` compiled ahead on the site's first draw to decide its pairing (`_ynn_decided`), by its
    inputs' signature (`_signature`): the program the arm runs on draws of that signature, never compiled
    twice."""

    @property
    def without_ynn(self) -> bool:
        """Whether the arm runs without YNNPACK's fusions (`without_ynn`'s mark)."""
        return not self.pairing.with_ynn

    def __call__(self, *inputs, **options):
        if self.pairing.with_ynn:
            return self.plain(*inputs, **options)
        compiled = self.ahead.get(_signature(inputs)) if self.ahead and not options else None
        return (compiled or self.without)(*inputs, **options)

    @property
    def trace(self):
        """The trace of the arm the site runs (JAX's `jit(...).trace`), where it is a compiled program's:
        an arm of Dew's own programs has none."""
        return (self.plain if self.pairing.with_ynn else self.without).trace


def _signature(inputs) -> tuple:
    """A draw's inputs as a compiled program takes them: whether JAX's 64-bit types are on (a float of the
    truth's widened draw is one width or another by it), their tree, and each leaf's type, shape and dtype, a
    JAX array's weak type and sharding too (a host array or a Python number is placed as the call places
    it; a leaf of no dtype, which no compile takes, by its type alone)."""
    leaves, tree = jax.tree.flatten(inputs)

    def taken(leaf) -> tuple:
        if isinstance(leaf, jax.Array):
            return type(leaf), leaf.shape, leaf.dtype, leaf.weak_type, leaf.sharding
        return type(leaf), getattr(leaf, "shape", ()), getattr(leaf, "dtype", None)

    return jax.dtypes.canonicalize_dtype(np.float64), tree, tuple(taken(leaf) for leaf in leaves)


def _paired_arms(arms) -> list[_Paired]:
    """The paired arms among `arms`, through every arm each adapts (`_layers`)."""
    return [layer for arm in arms for layer in _layers(arm) if isinstance(layer, _Paired)]


@functools.cache
def _ynn_option_held() -> None:
    """Raise unless XLA:CPU compiles a trivial program with WITHOUT_YNN's options, once a process: where an
    XLA release refuses the options themselves, every paired site would otherwise move onto the fusions
    and pass (`_ynn_decided`)."""
    try:
        without_ynn(lambda x: x * 2)(jnp.zeros((2, 2), jnp.float32))
    except jax.errors.JaxRuntimeError as error:
        raise RuntimeError(f"XLA:CPU refuses WITHOUT_YNN's options {WITHOUT_YNN} on a trivial program: "
                           f"{str(error).splitlines()[0]}; no check may move its arms onto the fusions"
                           ) from error


def _ynn_decided(label: str, arms, inputs) -> None:
    """The mode of a site's paired arms (`_Paired`), decided once before its draws: each compiled ahead on
    its first draw's `inputs` without YNNPACK's fusions; where XLA:CPU refuses one so and compiles it with
    them (its algebraic simplifier forms a dot of no contracting dimension from a contraction of length
    one, which only YNNPACK's fusion runs: attention's XLA backward over a single key), every paired arm
    of the site takes the fusions, logged in its stages' lines (`_logged`), reported (`SWITCHED`) and named in
    its findings (`_with_mode`). One the fusions refuse too raises, and so does a process whose XLA refuses
    the options themselves (`_ynn_option_held`). An arm running Dew's own programs through Dew's
    seam (`dew_programs`) compiles at its call, without them, as does a window loop's arm, whose program
    takes each window's slices and not the draw (`over_windows`). An arm compiled ahead runs that program on
    draws of its signature (`_Paired.ahead`), compiled once. Off XLA:CPU, which alone has the fusions,
    nothing is compiled ahead: the arms run as they compile."""
    pairings = {arm.pairing for arm in _paired_arms(arms)}
    if not pairings or all(pairing.decided for pairing in pairings):
        return
    from dew.nn.kernels.generation import placed_platform

    if placed_platform() != "cpu":
        # YNNPACK's fusions are XLA:CPU's: elsewhere an arm compiles one way (`without_ynn`), and a compile
        # ahead to decide them would be its second.
        for pairing in pairings:
            pairing.decided = True
        return
    _ynn_option_held()
    refused = None
    for arm in arms:
        # Through a declaration alone: a window loop's arm takes each window's slices, which the draw is not.
        arm = arm.fn if isinstance(arm, _RunToRun) else arm
        if not isinstance(arm, _Paired) or not hasattr(arm.without, "lower"):
            continue
        try:
            arm.ahead[_signature(inputs)] = arm.without.lower(*inputs).compile()
        except jax.errors.JaxRuntimeError as error:
            arm.plain.lower(*inputs).compile()
            refused = refused or str(error).splitlines()[0]
    for pairing in pairings:
        pairing.decided, pairing.refused = True, refused
    if refused is not None:
        SWITCHED.append({"label": label, "refused": refused})


def _mode_of(arms) -> str | None:
    """XLA:CPU's refusal that moved a site's arms onto YNNPACK's fusions (`_ynn_decided`), or None."""
    return next((arm.pairing.refused for arm in _paired_arms(arms) if arm.pairing.refused), None)


def _switched_off_here() -> dict:
    """WITHOUT_YNN on the CPU lane, where the context places a call (`switched_off`,
    `generation.placed_platform`), else none."""
    from dew.nn.kernels.generation import placed_platform

    return switched_off() if placed_platform() == "cpu" else {}


def _with_options(fn, compiler_options: Callable[[], dict]):
    """`fn`, an arm that runs Dew's own programs (`dew_programs`), run with Dew's compile seam adding the
    compiler options `compiler_options()` gives at each call (where the call is placed) to every program it
    compiles (`compiled_with`)."""
    from dew.nn.kernels.options import compiled_with

    @functools.wraps(fn)
    def run(*inputs, **options):
        with compiled_with(compiler_options()):
            return fn(*inputs, **options)
    return run


def _arms(dew_fn, reference_fn):
    """The two arms of a check as the gate runs them (docs/design/testing.md, "the numerical contract"):
    where both are Dew paths, each without YNNPACK's fusions (`without_ynn`), whose microkernels the host's
    instruction set picks at run time, so the two run the same kind of code on any host: a plain function
    the gate compiles so, an arm that runs Dew's own programs (`dew_programs`) with them compiled so through
    Dew's seam; where one runs as it is given (`as_given`: torch, NumPy, a recording, a host loop over
    programs of its own), the other as Dew's users run it, with the fusions (`jax.jit`). A site passes plain
    functions; the gate compiles them, so no site can compile one arm otherwise than its other. The
    reference runs unrewritten (`_compiled`), so one function as both arms compiles twice, but one of Dew's
    own paths (`dew_path`), which runs under the rewrite as Dew's arm does. A program XLA:CPU
    cannot compile without the fusions turns both arms over to them, decided once a site (`_ynn_decided`)."""
    compile, pairing = _compile_for(dew_fn, reference_fn), _Pairing()
    _refuse_owned_policies(compile, dew_fn, reference_fn)
    return _compiled(dew_fn, compile, pairing), _compiled(reference_fn, compile, pairing, reference=True)


def _compile_for(dew_fn, reference_fn):
    """How the gate compiles a check's arms (`_arms`): each without YNNPACK's fusions (`without_ynn`) where
    it compiles both, as Dew's users run them (`jax.jit`) beside an arm run as given."""
    return jax.jit if _given(dew_fn) or _given(reference_fn) else without_ynn


def _bounded(fn) -> bool:
    """Whether an accuracy claim also holds Dew's arm against reference `fn` compiled as written
    (`_written_twin`): a reference the gate compiles, written apart from Dew; not one of Dew's own paths (an
    invariance claim's, `dew_path`, or one running Dew's own programs, `dew_programs`), nor one run as given
    or recorded."""
    return not (_given(fn) or _declared(fn)
                or any(getattr(layer, "dew_programs", False) for layer in _layers(fn)))


def _written_twin(dew_fn, reference_fn):
    """`reference_fn` compiled as written (`_compiled`'s `written`), unrewritten, beside `dew_fn` as `_arms`
    pairs the two: the reference an accuracy claim holds Dew's arm within PARITY to the power WRITTEN of
    (`dew.parity.WRITTEN`), so a defect a rewrite of XLA's would share between the two arms compiled alike
    fails (docs/design/testing.md, check 1); and, where a site names no truth, its truth on the draw widened
    (`reference_error._site_judged`), every truth compiled as written."""
    return _compiled(reference_fn, _compile_for(dew_fn, reference_fn), _Pairing(), reference=True,
                     written=True)


def _arms_with(dew_fn, reference_fn, further: list, label: str):
    """A site's arms and each further reference's, compiled as `_arms` pairs them, Dew's once beside all:
    refused where, beside a plain Dew arm, a further reference is not of the reference's kind (plain or
    as given), as no one compilation of Dew's pairs with both."""
    if not _given(dew_fn) and any(_given(fn) != _given(reference_fn) for fn in further):
        raise ValueError(f"{label}: beside a plain Dew arm each further reference is of the reference's "
                         f"kind, plain or as given, so one compilation of Dew's pairs with all")
    dew, reference = _arms(dew_fn, reference_fn)
    return dew, reference, [_arms(dew_fn, fn)[1] for fn in further]


def _refuse_owned_policies(compile, *arms) -> None:
    """Refuse an arm compiled with a policy the gate does not own (row 2287): a `without_ynn` program,
    which `_compiled` runs as it is, beside an arm run `as_given`, where the gate compiles Dew's arm as
    its users run it, with the fusions (`jax.jit`) — it would run without them, unseen. A plain
    `jax.jit` arm the gate's own compile nests, which applies the gate's options; a jit with compiler
    options of its own refuses that nesting itself (`compiler_options` only at top level). Pass the
    function plain and the gate compiles it as the pairing asks."""
    for fn in arms:
        inner = fn.fn if isinstance(fn, _Windowed) else fn
        if compile is jax.jit and getattr(inner, "without_ynn", False) and not _given(inner):
            raise TypeError(f"{getattr(inner, '__qualname__', repr(inner))} is compiled without YNNPACK's "
                            f"fusions, beside an arm run as given, where the gate compiles Dew's arm with "
                            f"them: pass it plain, and the gate compiles it as the pairing asks")


def _written_references(reference_fn, also: Mapping) -> list[tuple[str | None, Callable]]:
    """The references a site's accuracy claim also holds Dew's arm against compiled as written
    (`arm_forms._bounded`, `arm_forms._written_twin`), each by its name (the site's own None), in its draws'
    order."""
    return [(name, fn) for name, fn in ((None, reference_fn), *also.items()) if _bounded(fn)]


def _written_role(name: str | None) -> str:
    """The role of reference `name`'s twin compiled as written in a site's store (`_written_references`)."""
    return "written" if name is None else f"written {name}"


def _holds_program(fn) -> bool:
    """Whether arm `fn`, within a run-to-run declaration and a window loop, is one the store holds programs
    of (`_compile_arms`): none an arm the gate runs as given or one running Dew's own programs."""
    while isinstance(fn, (_RunToRun, _Windowed)):
        fn = fn.fn
    return not (_given(fn) or getattr(fn, "dew_programs", False))
