"""Dew's rounding error held to the reference's own, both measured from float64.

The gate the suite converts to holds every element over orders: a comparison drawn over
exact symmetries of its computation (`assert_leaves_as_exact_over_orders`), each element against the
worst admitted member of the reference's ensemble (`assert_elements_as_exact`), and a kernel held where
it rounds, as a graph of the values it rounds and holds (`assert_graph_as_exact`), all three in
tests/reference_ensemble.py. The one-draw rules below hold the comparisons that have not converted
yet.

A port that computes what its reference computes, at the same precision,
differs from it only by rounding: the two sum the same terms in other
orders and round at other points. The reference evaluated in float64
(u = 2^-53, nine orders below fp32) is the exact value to every digit an
fp32 or bf16 run resolves, so both runs are measured from it, and Dew's
root-mean-square error over all entries may be at most twice the
reference's:

    rms(dew - f64) <= 2 rms(reference - f64)

Why the root mean square and why twice. Independent roundings add in
variance, so the RMS error of an output is sqrt(sum of the variances of the
roundings that reach it), and over the hundreds to thousands of entries of
a fixture that estimate is stable to a few percent; the largest entry is an
extreme value and is not (the fp32 decoder fixtures put Dew's largest error
at 0.9 to 2.2 times the reference's while the RMS ratios sit at 1.06 to
1.43). A factor of 2 admits four times the reference's error variance, that
is Dew making up to three roundings of the reference's size for each one
the reference makes; a computation that differs rather than rounds (a
dropped term, a wrong scale, a format coarser than the reference's, a mask
one step off) moves the RMS error by more than that and fails.

Why not once, the reference's own distance plus a floor for summing in
another order. Measured over the suite (862 float64 comparisons in 85
files), Dew's distance over the reference's sits at a median of 1.00,
p25 0.94, p75 1.07 and p95 1.55: exactly as exact, the two
part by their own rounding draws, so a ratio above one is as likely as
below. At once, 95 comparisons in 20 files failed with no defect behind
them, and holding them needed a reduction-length floor of half the
reference's own error at each, a weaker bound than twice the reference.
At twice, one comparison failed, a deliberate one (a sinkless MiMo tower).

What the rule cannot see is a difference below that factor: an attention
softmax taken in bf16 instead of fp32 lands at 1.2 to 1.4 times the
reference's error, since the bf16 logits both runs share carry most of it.
Where Dew claims the reference's order of operations, it is held instead to
the reference's own bf16 output:

    rms(dew - reference) <= rms(reference - f64) / 4

Rounding at the same points, the two outputs part only where their fp32
accumulations, summed in different orders, fall on opposite sides of a bf16
rounding boundary: an entry flips with probability about sqrt(K) 2^-24 /
2^-8, K the reduction length, by one bf16 spacing, about twice the typical
rounding error, so the ratio is near 2 sqrt(sqrt(K) 2^-16), 0.04 at K 512.
One rounding at a point the reference does not round is an independent
error of the reference's own size and puts the ratio near one.

Where the truth is an oracle written apart from the reference (a recurrence
or an equation, not the reference run in float64), the rule only measures
rounding if the reference computes the oracle's function: a reference that
shares Dew's mistake would widen the bound to fit it. So the reference's own
arithmetic in float64 is held to the oracle first, within the float64
rounding of the computation itself (`assert_computes_the_oracle`).
"""

import dataclasses
import functools
import hashlib
import itertools
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from arm_arithmetic import Truth, Unrewritten, summed_as_run, unrewritten, widened
from check1_statistics import (
    CONFIDENCE,
    FACTOR,
    FEWEST_SEEDS,
    LOG_ZERO,
    LOWER_TAIL,
    NARROW_FACTORS,
    PARITY,
    PREVIOUS_TAILS,
    ROW_FACTOR,
    UPPER_TAIL,
    Finding,
    GateFailure,
    Inconclusive,
    ZeroRows,
    _interval,
    _over,
    _paired_rows,
    _pooled,
    _scaled_rms,
    _sized,
    _status,
    _typical_status,
    _unresolved_against_itself,
    _verdict,
    log_over,
    log_ratio,
    log_sum,
    pooled_zero_rows,
    root_mean,
    row_factor,
    shown,
)
from check1_wire import DRAWS, REDRAW, STAGES, staged_units
from conditioning import Conditioning, conditioned
from float64_terms import float64_terms
from map_judge import PoolRefused, ask_later, drawing, judging, pooled_units, record_units
from site_draws import _Draws
from truth_sketch import SKETCH_DEVIATIONS, sketch_rounding, truth_sketch

from dew._accuracy import distance, reduction_bound, within

PRECISION = "precision"
"""The entry of a reference's array file that records the precision its generator's matmuls took."""


def ieee_fixture(path) -> dict[str, np.ndarray]:
    """The arrays of a reference's array file, refused unless its generator
    recorded IEEE matmuls under `PRECISION`.

    The rule above takes the reference's error as its budget. A reference
    regenerated with TF32 matmuls, ten mantissa bits, sits farther from
    float64 and so widens the budget to fit a coarser Dew, and the float32
    arrays it stores look the same; only the generator's record tells."""
    with np.load(path) as stored:
        arrays = {name: stored[name] for name in stored.files}
    recorded = str(arrays.pop(PRECISION)) if PRECISION in arrays else "no precision"
    if recorded != "ieee":
        raise ValueError(f"{path} records {recorded!r} for its matmuls, not 'ieee'; regenerate it "
                         f"with IEEE matmuls")
    return arrays


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


class _Windowed:
    """An arm of one draw (`fn`) over `windows`' stacked arguments, window by window (`over_windows`):
    the gate compiles `fn` as it compiles any arm (`_arms`) and runs the windows through that program."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *arguments):
        return stacked_on_host([self.fn(*jax.tree.map(lambda leaf, index=index: leaf[index], arguments))
                            for index in range(_window_count(arguments))])


class _RunToRun:
    """An arm whose outputs vary in their last bits from run to run on the same inputs (`run_to_run`):
    the gate compiles `fn` as it compiles any arm (`_arms`) and judges each draw by the one run it makes."""

    def __init__(self, fn, why: str):
        self.fn, self.why = fn, why

    def __call__(self, *inputs, **options):
        return self.fn(*inputs, **options)

    def __getattr__(self, name):
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
    (row 4448, M-JS-5). Check 1 pairs it with the other arm within each draw's one run, as it pairs every
    arm, and never holds it to a run of itself again (`_of_its_inputs`) but on a CPU, where its interpreted
    program is a function of its inputs: a draw's evidence is one run's, and the redraw and every further
    seed are runs of their own, so its spread across runs is the spread across draws check 1 weighs."""
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


CENSUS = "DEW_GATE_CENSUS"
"""The environment variable whose being set runs check 1's arms with YNNPACK's fusions on (WITHOUT_YNN
empty), as CPU users compile them: the census the whole suite runs every fifth publish, each red it finds
filed and judged Dew's or XLA:CPU's (row 4471; docs/design/testing.md, "the YNNPACK switch")."""

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
    """`fn` jitted with WITHOUT_YNN on the CPU lane, the process's default backend
    (`jax.default_backend()`), and plain `jax.jit` elsewhere: how the gate compiles both arms of a
    check 1 between two Dew paths (`_arms`), such as an interpreted kernel and the XLA program it
    replaces, so that the two run the same kind of code on any host. A program Dew's users run compiles
    with the fusions, as they do. On the CPU lane its compiler options refuse to be nested in another
    jit, differentiated through or batched, so the gate compiles an arm whole; a site passes its arms
    as functions and never calls this. A caller's own compiler options take precedence over
    WITHOUT_YNN's; `fn` already compiled so is returned as it is."""
    if getattr(fn, "without_ynn", False):
        return fn
    if jax.default_backend() == "cpu":
        jit_options["compiler_options"] = {**switched_off(), **(jit_options.get("compiler_options") or {})}
    compiled = jax.jit(fn, **jit_options)
    compiled.without_ynn = True
    return compiled


def as_given(fn):
    """`fn`, an arm the gate runs as it is given and never compiles: a torch or NumPy reference, or a Dew
    arm that runs whole programs of its own (a trainer's fit, a placement on a mesh and the program
    run there, a host loop). The other arm of its check then compiles as Dew's users run it, with the
    fusions (`_arms`), as the programs this one runs do."""
    @functools.wraps(fn)
    def given(*inputs, **options):
        return fn(*inputs, **options)
    given.as_given = True
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


def _compiled(fn, compile, pairing=None, *, reference: bool = False):
    """`fn` as the gate runs it: as given (`_given`), as it is where it is a `without_ynn` program already
    (another jit around it would nest its compiler options), or compiled by `compile`, a window loop's arm
    within it (`over_windows`), without YNNPACK's fusions, paired by `pairing` (`_Paired`); a `reference`
    unrewritten (`arm_arithmetic.unrewritten`)."""
    run = unrewritten if reference else _as_is
    if isinstance(fn, _RunToRun):
        return _RunToRun(_compiled(fn.fn, compile, pairing, reference=reference), fn.why)
    if _given(fn) or getattr(fn, "without_ynn", False):
        return run(fn)
    if getattr(fn, "dew_programs", False):
        return run(fn) if compile is jax.jit else _Paired(pairing, run(_with_ynn_off(fn)), run(fn))
    if isinstance(fn, _Windowed):
        return _Windowed(_compiled(fn.fn, compile, pairing, reference=reference))
    return run(compile(fn)) if compile is jax.jit else _Paired(pairing, run(compile(fn)), run(jax.jit(fn)))


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

    @property
    def without_ynn(self) -> bool:
        """Whether the arm runs without YNNPACK's fusions (`without_ynn`'s mark)."""
        return not self.pairing.with_ynn

    def __call__(self, *inputs, **options):
        return (self.plain if self.pairing.with_ynn else self.without)(*inputs, **options)


def _paired_arms(arms) -> list[_Paired]:
    """The paired arms among `arms`, through every arm each adapts (`_layers`)."""
    return [layer for arm in arms for layer in _layers(arm) if isinstance(layer, _Paired)]


SWITCHED: list[dict] = []
"""The sites decided onto YNNPACK's fusions (`_ynn_decided`) since the last test's report took them: each its
label and XLA:CPU's refusal, which tests/conftest.py reports as the test's JUnit property `ynn_switched`."""


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
    takes each window's slices and not the draw (`over_windows`)."""
    pairings = {arm.pairing for arm in _paired_arms(arms)}
    if not pairings or all(pairing.decided for pairing in pairings):
        return
    _ynn_option_held()
    refused = None
    for arm in arms:
        # Through a declaration alone: a window loop's arm takes each window's slices, which the draw is not.
        arm = arm.fn if isinstance(arm, _RunToRun) else arm
        if not isinstance(arm, _Paired) or not hasattr(arm.without, "lower"):
            continue
        try:
            arm.without.lower(*inputs).compile()
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


def _with_mode(findings: list, arms) -> list:
    """`findings`, each naming its site's arms' YNNPACK mode where they were decided onto the fusions
    (`_ynn_decided`), so a failure says which code its arms ran."""
    refused = _mode_of(arms)
    if refused is None:
        return findings
    return [dataclasses.replace(finding, message=f"{finding.message} (both arms with YNNPACK's fusions: "
                                                 f"XLA:CPU refused one without, {refused!r})")
            for finding in findings]


def _truth_program(fn):
    """`fn`, a check's float64 truth, as the gate runs it, unrewritten (`arm_arithmetic.unrewritten`):
    compiled, once a signature, so the truth runs as one float64 program on the device's FP64 units rather
    than one eager dispatch an equation, every primitive at its own float64 semantics (`Truth`). A truth the
    gate runs as given (`as_given`: NumPy, torch, a recording, a host loop), one compiled already (a jit,
    `dew_programs`), or none, as it is; a window loop's by its own. A truth with host work inside that does
    not state it fails at its trace, loudly, and passes `as_given`."""
    if fn is None or _given(fn) or getattr(fn, "dew_programs", False) or hasattr(fn, "lower"):
        return unrewritten(fn)
    if isinstance(fn, _Windowed):
        return _Windowed(_truth_program(fn.fn))
    return Truth(fn)


def _with_ynn_off(fn):
    """`fn` run with Dew's compile seam holding WITHOUT_YNN open on the CPU lane (`dew_programs`)."""
    from dew.nn.kernels.options import compiled_with

    @functools.wraps(fn)
    def run(*inputs, **options):
        with compiled_with(switched_off() if jax.default_backend() == "cpu" else {}):
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
    reference runs unrewritten (`_compiled`), so one function as both arms compiles twice. A program XLA:CPU
    cannot compile without the fusions turns both arms over to them, decided once a site (`_ynn_decided`)."""
    compile, pairing = (jax.jit if _given(dew_fn) or _given(reference_fn) else without_ynn), _Pairing()
    _refuse_owned_policies(compile, dew_fn, reference_fn)
    return _compiled(dew_fn, compile, pairing), _compiled(reference_fn, compile, pairing, reference=True)


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


def with_gradients(fn, *, has_aux: bool = False):
    """`fn` as `run(*inputs, cotangent)`, which returns `fn(*inputs)` and the gradients of its inputs
    under the cotangent passed last: a differentiated call's leaves, as the gate compares them. Where
    `has_aux`, `fn` returns its differentiated output and values beside it that carry no gradient,
    which `run` returns after the gradients, from the same evaluation."""
    def run(*arguments):
        if has_aux:
            out, pull, aux = jax.vjp(fn, *arguments[:-1], has_aux=True)
            return out, *pull(arguments[-1]), aux
        out, pull = jax.vjp(fn, *arguments[:-1])
        return out, *pull(arguments[-1])

    return run


def _host_tensor(value):
    """A torch copy of a host array, of its own shape and dtype. NumPy holds bfloat16, the float8
    formats and the 4-bit integers as ml_dtypes' types: a float of those goes through float32, which
    holds it exactly, to torch's format of the same name, and an integer enters as int32, which holds
    its values. A copy, so autograd never writes an array JAX holds."""
    import torch

    value = np.array(value)
    if value.dtype.type.__module__.startswith("ml_dtypes"):
        if jnp.issubdtype(value.dtype, jnp.floating):
            return torch.as_tensor(value.astype(np.float32)).to(getattr(torch, value.dtype.name))
        return torch.as_tensor(value.astype(np.int32))
    return torch.as_tensor(value)


def _host_array(value):
    """A tensor as a host array. A format NumPy lacks, such as bfloat16, comes back as float32, which holds
    its values exactly."""
    import torch

    value = value.detach()
    narrow = value.is_floating_point() and value.dtype not in (torch.float16, torch.float32, torch.float64)
    return (value.float() if narrow else value).numpy()


def torch_with_gradients(fn):
    """`with_gradients` for a torch arm. `fn(*tensors)` becomes `run(*inputs, cotangent)`, which returns
    `fn`'s output and the gradients of its inputs under the cotangent passed last, as host arrays.
    - Each floating input enters as a tensor of its own dtype that requires a gradient; a module's
      parameters are inputs too, called through `torch.func.functional_call`.
    - The cotangent has the output's tree (an array, or a tuple or dict of them), as `jax.vjp` requires,
      each leaf cast to its output's dtype.
    - An input the output does not reach has a zero gradient, a non-floating one none, and an output no
      input reaches pulls nothing back.
    - A narrow float array (bfloat16, the float8 formats) goes in and comes out through the float32 that
      holds it.
    The gate runs the torch arm as it is given (`as_given`)."""
    def run(*arguments):
        import torch

        inputs, cotangent = jax.tree.map(_host_tensor, arguments[:-1]), arguments[-1]
        for leaf in jax.tree.leaves(inputs):
            if leaf.is_floating_point():
                leaf.requires_grad_()
        out = fn(*inputs)
        if jax.tree.structure(out) != jax.tree.structure(cotangent):
            raise ValueError(f"a cotangent of tree {jax.tree.structure(cotangent)} for an output of tree "
                             f"{jax.tree.structure(out)}")
        flat, tree = jax.tree.flatten(inputs)
        wanted = [index for index, leaf in enumerate(flat) if leaf.requires_grad]
        reached = [(leaf, c) for leaf, c in zip(jax.tree.leaves(out), jax.tree.leaves(cotangent), strict=True)
                   if leaf.requires_grad]
        pulled = [None] * len(wanted)
        if wanted and reached:
            pulled = torch.autograd.grad([leaf for leaf, _ in reached], [flat[index] for index in wanted],
                                         grad_outputs=[_host_tensor(c).to(leaf.dtype) for leaf, c in reached],
                                         allow_unused=True)
        gradients = [None] * len(flat)
        for index, grad in zip(wanted, pulled, strict=True):
            gradients[index] = _host_array(torch.zeros_like(flat[index]) if grad is None else grad)
        return jax.tree.map(_host_array, out), *jax.tree.unflatten(tree, gradients)

    return as_given(run)


def _floating(dtype) -> bool:
    return jnp.issubdtype(dtype, jnp.floating)


def equations(graph, *, platform: str | None = None):
    """Every equation of a jaxpr, its sub-jaxprs' (scan, cond, custom rules) in turn. With `platform`
    (a lowering platform's name: 'cpu', 'cuda', 'tpu'), a platform-dependent cond
    (`jax.lax.platform_dependent`) gives only the branch that platform runs, as lowering keeps it."""
    if hasattr(graph, "eqns"):
        for equation in graph.eqns:
            yield equation
            chosen = equation.params.get("branches_platforms") if platform is not None else None
            if chosen is None:
                yield from equations(equation.params, platform=platform)
                continue
            taken = next(index for index, platforms in enumerate(chosen)
                         if platforms is None or platform in platforms)
            yield from equations(equation.params["branches"][taken], platform=platform)
    elif hasattr(graph, "jaxpr"):
        yield from equations(graph.jaxpr, platform=platform)
    elif isinstance(graph, dict):
        for value in graph.values():
            yield from equations(value, platform=platform)
    elif isinstance(graph, (tuple, list)):
        for value in graph:
            yield from equations(value, platform=platform)


def dot_precisions(function, *arguments) -> list:
    """The precision of every dot_general `function` traces over `arguments` under JAX's own default
    precision (the test lane sets 'highest' for every product that states none), with the dtype its
    operands promote to and their shapes, sub-jaxprs included, as a GPU runs them: a CPU's float64
    sums (`dew.nn.precision.accumulated`) are tests/test_cpu_accumulation.py's."""
    with jax.default_matmul_precision(None):
        graph = jax.make_jaxpr(function)(*arguments)
    return [(equation.params["precision"],
             jnp.result_type(*(variable.aval.dtype for variable in equation.invars)),
             tuple(variable.aval.shape for variable in equation.invars))
            for equation in equations(graph, platform="cuda") if equation.primitive.name == "dot_general"]


def assert_fp32_reduction_bound(dew, reference, magnitudes, terms: int) -> None:
    """Two fp32 reductions differ by at most 2 gamma_n sum(abs(products))
    (`dew._accuracy.reduction_bound`): `magnitudes` holds the elementwise sum of
    absolute products and `terms` counts the products and any bias term in each
    reduction."""
    error = np.abs(np.asarray(dew, np.float64) - np.asarray(reference, np.float64))
    bound = reduction_bound(magnitudes, terms)
    assert within(error, bound), (float(error.max()), float(bound.max()))


RESOLUTION = 1024 * float(np.finfo(np.float64).eps)
"""The smallest row float64 resolves, over its leaf's size: a row below it is
float64's own rounding of the leaf (a packed row's padding at 4e-19 of a
leaf at 1e-3), a zero `rows_over` sizes by the reference's row. 1024
roundings is the longest chain the suite's oracles allow
(`assert_computes_the_oracle`); a row at a float32 resolution is ten orders
above it."""


def row_errors(value, truth, terms=None) -> tuple[np.ndarray, np.ndarray]:
    """Each row's float64 size and `value`'s RMS distance from float64 on it,
    a row every axis but the last. A row's size is the RMS of its float64
    values, or, where `terms` gives each entry's float64 terms
    (`float64_terms`), the larger of that and the RMS of what the row sums,
    so a row that cancels is sized by its terms' rounding and not by its
    near-zero result; a row below float64's `RESOLUTION` of its leaf has size
    zero. No other row's size enters: a small row beside large ones has its
    own. A value of rank one or less is one row, which `distance` already
    judges, and has none here."""
    truth = np.asarray(truth, np.float64)
    if truth.ndim < 2:
        return np.zeros(0), np.zeros(0)
    return _row_sizes(truth, terms)[1], _rows_rms(np.asarray(value, np.float64) - truth, truth.shape[-1])


def _rows_rms(values, width: int) -> np.ndarray:
    """Each row's RMS of `values`, a row `width` wide (`_scaled_rms`)."""
    return _scaled_rms(np.reshape(values, (-1, width)))


def _row_sizes(truth: np.ndarray, terms=None) -> tuple[np.ndarray, np.ndarray]:
    """A truth leaf's rows, of rank two or more: each row's own RMS, and its size as `row_errors` reads
    it, the RMS (or, with `terms`, the larger of it and its terms') where it is past RESOLUTION of the
    leaf, else zero."""
    own = _rows_rms(truth, truth.shape[-1])
    sized = own if terms is None else np.maximum(own, _rows_rms(terms, truth.shape[-1]))
    leaf = max(distance(truth, 0), 0.0 if terms is None else distance(terms, 0))
    return own, np.where(sized > RESOLUTION * leaf, sized, 0.0)

NO_ZERO_ROWS: ZeroRows = (np.zeros(0, np.int64), np.zeros(0), np.zeros(0))


@dataclasses.dataclass(frozen=True)
class RecordedLeaf:
    """A recorded reference's evidence for one leaf at one seed, all check 1 reads of it (`_drawn`): its
    mean squared error from its own float64 truth, its worst sized row over its size, its RMS error on
    each row the truth sizes zero (below RESOLUTION of its leaf) by the row's index, and whether it and
    its truth were finite; read from a leaf here (`leaf_summary`), also the truth's own RMS on each of
    those rows, which decides whether a row a live reference measures is zero (`_paired`)."""
    mse: float
    worst: float
    zero_rows: np.ndarray
    zero_errors: np.ndarray
    finite: bool
    dtype: str
    zero_sizes: np.ndarray | None = None
    worst_log: float | None = None
    share: float | None = None
    """The share of its worst sized row's squared error the row's largest element carries (`_Draws.factor`),
    zero where no row is sized; None where the leaf was recorded as rows alone (`rows_summary`)."""

    @property
    def log_worst(self) -> float:
        """The log of `worst`, exact where this leaf was read here (`leaf_summary`), which keeps its
        order past float64's range; a recording's, from the value it holds."""
        return self.worst_log if self.worst_log is not None else float(log_over(self.worst, 1.0))


def leaf_summary(reference, truth) -> RecordedLeaf:
    """`RecordedLeaf` of a leaf against its float64 truth: check 1's evidence of the reference's leaf, or
    of Dew's (`_drawn` reads both so, a recorded reference's as recorded), read as `_evidence` reads a
    draw's (`leaf_summaries`). An empty leaf is exact."""
    return leaf_summaries([reference], [truth])[0]


def leaf_summaries(references, truths) -> list[RecordedLeaf]:
    """Each reference leaf's `leaf_summary` against its truth, read at once (`_summaries`), each held to
    its truth's shape first: a recording's record (tools/recorded_references.py)."""
    for reference, truth in zip(references, truths, strict=True):
        if np.shape(reference) != np.shape(truth):
            raise AssertionError(f"a leaf {np.shape(reference)} against a truth {np.shape(truth)}")
    return _summaries(references, _LaidTruths.of(truths))


def _missed(mse, missed):
    """Each mean squared error where its leaf `missed` its truth somewhere: a leaf whose every error is below
    2^-537 squares to zero, so it takes the least positive float64 instead, and an arm that misses an exact
    reference is never read as exact (check 1's exact-reference rule, `_judgement`)."""
    return np.where((np.asarray(mse) == 0) & missed, np.nextafter(0.0, 1.0), mse)


def rows_summary(rows, truth, *, finite: bool, dtype: str, mse: float | None = None) -> RecordedLeaf:
    """`leaf_summary` of a leaf from each row's RMS distance from its float64 truth `rows` alone, where the
    reference ran apart and kept those (a recorder whose outputs are too large to keep, its rows its
    leaf's every axis but the last, in C order): its mean squared error the rows' mean square, every row
    one width, unless `mse` gives it; `finite` whether the reference's leaf was, and `dtype` its dtype.
    The truth's rows are read as `_summaries` reads them (`_LaidTruths`), and an empty leaf is exact,
    as `leaf_summary` reads one."""
    if not np.size(truth):
        return RecordedLeaf(0.0, 0.0, np.zeros(0, np.int64), np.zeros(0), finite=finite, dtype=dtype)
    held, rows = _LaidTruths.of([truth]), np.asarray(rows, np.float64)
    sized = held.sizes > 0
    logged = float(np.max(log_over(rows[sized], held.sizes[sized]))) if sized.any() else LOG_ZERO
    return RecordedLeaf(float(_missed(np.mean(np.square(rows)), np.any(rows != 0))) if mse is None else mse,
                        float(np.max(_over(rows[sized], held.sizes[sized]))) if sized.any() else 0.0,
                        np.flatnonzero(~sized), rows[~sized], finite=finite and bool(held.finite.all()),
                        dtype=dtype, zero_sizes=held.own[~sized], worst_log=logged)


@dataclasses.dataclass(frozen=True)
class _LaidTruths:
    """A draw's truth leaves read at once for both arms (`_summaries`). Every leaf that is not empty lies
    end to end with the others in C order, by row width (its last axis's, 0 for a leaf of rank under two)
    and then size: each run of one width and size is one (leaves, size) matrix, and each width's rows one
    (rows, width) matrix, so a draw takes a few passes where it took a pipeline a leaf (ledger 1848:
    190 µs a leaf at the gemma3n gradient's tiny leaves). A leaf in another layout is read as its
    C-ordered copy. Each row has its own RMS (`_scaled_rms`) and its size, the RMS where it is past
    RESOLUTION of its leaf, else zero (`row_errors`'). `truths` is every leaf in float64 in C order,
    `laid` the leaves laid in order, `runs` each run's (leaves, size, width), `values` the laid leaves,
    `own` and `sizes` their rows' own RMS and size, and `finite` whether each laid leaf's every value
    is."""

    truths: list
    laid: list[int]
    runs: list[tuple[int, int, int]]
    values: np.ndarray
    own: np.ndarray
    sizes: np.ndarray
    finite: np.ndarray

    @staticmethod
    def of(truths, terms=None) -> "_LaidTruths":
        """`truths` laid, each row sized by its own RMS or, given each leaf's float64 `terms`
        (`float64_terms`), by the larger of it and its terms' RMS, as `row_errors` sizes a row that
        cancels (`assert_as_exact`'s `terms`)."""
        truths = [np.asarray(truth, np.float64, order="C") for truth in truths]
        keys = sorted((truth.shape[-1] if truth.ndim >= 2 else 0, truth.size, at)
                      for at, truth in enumerate(truths) if truth.size)
        runs = [(len(list(run)), size, width)
                for (width, size), run in itertools.groupby(keys, lambda key: key[:2])]
        values = np.concatenate([np.zeros(0), *(truths[at].reshape(-1) for *_, at in keys)])
        blocks, resolved = _blocks(values, runs), [np.zeros(0)]
        for (_, size, width), block in zip(runs, blocks, strict=True):
            if width:
                # Each leaf's `distance(truth, 0)`, at each of its rows.
                resolved.append(np.repeat(RESOLUTION * np.sqrt(np.mean(np.square(block - 0.0), axis=1)),
                                          size // width))
        own = np.concatenate([np.zeros(0), *map(_scaled_rms, _rows(values, runs))])
        sized = own
        if terms is not None:
            # The terms laid as the truths are, leaf by leaf; a leaf no sum made is its own size.
            laid_terms = [np.abs(np.asarray(term if term is not None else truths[at], np.float64,
                                            order="C")).reshape(truths[at].shape)
                          for at, term in enumerate(terms)]
            termed = np.concatenate([np.zeros(0), *(laid_terms[at].reshape(-1) for *_, at in keys)])
            sized = np.maximum(own, np.concatenate([np.zeros(0), *map(_scaled_rms, _rows(termed, runs))]))
        finite = [np.isfinite(block).all(axis=1) for block in blocks]
        return _LaidTruths(truths, [at for *_, at in keys], runs, values, own,
                           np.where(sized > np.concatenate(resolved), sized, 0.0),
                           np.concatenate([np.zeros(0, bool), *finite]))


def _blocks(values: np.ndarray, runs) -> list[np.ndarray]:
    """`values` laid by `runs` (`_LaidTruths`), each run's leaves a (leaves, size) matrix."""
    ends = list(itertools.accumulate((leaves * size for leaves, size, _ in runs), initial=0))
    return [values[start:stop].reshape(leaves, size)
            for start, stop, (leaves, size, _) in zip(ends[:-1], ends[1:], runs, strict=True)]


def _rows(values: np.ndarray, runs) -> list[np.ndarray]:
    """`values` laid by `runs` (`_LaidTruths`), each width's rows, its runs' together, one (rows, width)
    matrix."""
    ends = list(itertools.accumulate((leaves * size for leaves, size, _ in runs), initial=0))
    spans = {}
    for start, stop, (_, _, width) in zip(ends[:-1], ends[1:], runs, strict=True):
        if width:
            spans[width] = (spans.get(width, (start,))[0], stop)
    return [values[start:stop].reshape(-1, width) for width, (start, stop) in spans.items()]


def _largest_share(rows: np.ndarray) -> np.ndarray:
    """Each row's largest element's share of the row's squared error, the row scaled by its largest first so
    no square overflows: one for a row one element holds alone, zero for a row of zeros."""
    rows = np.abs(np.asarray(rows, np.float64))
    largest = np.max(rows, axis=1, keepdims=True) if rows.shape[1] else np.zeros((len(rows), 1))
    scaled = np.square(rows / np.where(largest > 0, largest, 1.0))
    total = np.sum(scaled, axis=1)
    return np.where(total > 0, 1.0 / np.where(total > 0, total, 1.0), 0.0)


def _summaries(references, held: _LaidTruths) -> list[RecordedLeaf | None]:
    """Each reference leaf's evidence against its truth in `held` (`_LaidTruths.of`) as check 1 reads it
    (`RecordedLeaf`), every leaf read in C order with the others in a few passes; None for a reference of
    None. The caller holds each reference to its truth's shape (`leaf_summaries`, `_evidence`)."""
    names = {}

    def dtype(reference) -> str:
        kind = reference.dtype if hasattr(reference, "dtype") else np.asarray(reference).dtype
        if kind not in names:
            names[kind] = str(jnp.dtype(kind))
        return names[kind]

    # An empty leaf is exact; `held` lays every other.
    summaries, laid = [None] * len(references), set(held.laid)
    for at, reference in enumerate(references):
        if reference is not None and at not in laid:
            summaries[at] = RecordedLeaf(0.0, 0.0, np.zeros(0, np.int64), np.zeros(0), finite=True,
                                         dtype=dtype(reference))
    read = [references[at] is not None for at in held.laid]
    if not any(read):
        return summaries
    # A leaf of no reference lies as its truth, a zero error never read.
    values = np.concatenate([np.zeros(0), *(
        np.asarray(references[at], np.float64, order="C").reshape(-1) if ok else held.truths[at].reshape(-1)
        for ok, at in zip(read, held.laid, strict=True))])
    finite = np.concatenate([np.isfinite(block).all(axis=1) for block in _blocks(values, held.runs)])
    errors = values - held.values
    mse = np.concatenate([np.mean(block, axis=1) for block in _blocks(np.square(errors), held.runs)])
    mse = _missed(mse, np.concatenate([np.zeros(0, bool),
                                       *((block != 0).any(axis=1) for block in _blocks(errors, held.runs))]))
    rows = np.concatenate([np.zeros(0), *map(_scaled_rms, _rows(errors, held.runs))])
    shares = np.concatenate([np.zeros(0), *map(_largest_share, _rows(errors, held.runs))])
    # Each laid leaf's rows, its first among them, and those sized past RESOLUTION.
    counts = np.repeat([size // width if width else 0 for _, size, width in held.runs],
                       [leaves for leaves, _, _ in held.runs])
    starts = np.cumsum(counts) - counts
    sized = held.sizes > 0
    before = np.concatenate([[0], np.cumsum(sized)])
    taken = before[starts + counts] - before[starts]
    logs, over = log_over(rows[sized], held.sizes[sized]), _over(rows[sized], held.sizes[sized])
    # The worst sized row of each leaf that has one: the largest of its run of the sized rows.
    some = taken > 0
    worst_log, worst, share = np.full(len(counts), LOG_ZERO), np.zeros(len(counts)), np.zeros(len(counts))
    if some.any():
        firsts = (np.cumsum(taken) - taken)[some]
        worst_log[some] = np.maximum.reduceat(logs, firsts)
        worst[some] = np.maximum.reduceat(over, firsts)
        # The worst row's largest element's share of its squared error, each leaf's worst sized row's.
        sized_shares = shares[sized]
        share[some] = [sized_shares[first + int(np.argmax(logs[first:first + count]))]
                       for first, count in zip(firsts.tolist(), taken[some].tolist(), strict=True)]
    # The rows of float64 zero, each leaf's its span of them.
    unsized = np.flatnonzero(~sized)
    zero_rows = unsized - np.repeat(starts, counts - taken)
    zero_errors, zero_sizes = rows[unsized], held.own[unsized]
    ends = list(itertools.accumulate((counts - taken).tolist(), initial=0))
    mse, worst, worst_log, share, finite = mse.tolist(), worst.tolist(), worst_log.tolist(), share.tolist(), (
        finite & held.finite).tolist()
    for index, at in enumerate(held.laid):
        if read[index]:
            span = slice(ends[index], ends[index + 1])
            summaries[at] = RecordedLeaf(mse[index], worst[index], zero_rows[span], zero_errors[span],
                                         finite=finite[index], dtype=dtype(references[at]),
                                         zero_sizes=zero_sizes[span], worst_log=worst_log[index],
                                         share=share[index])
    return summaries


class UnpairedDraw(AssertionError):
    """A check's draw or arm that depends on more than its seed or its inputs (`_of_its_seed`,
    `_of_its_inputs`): randomness of its own, or state a call left, so its arms compare draws no seed
    pairs."""


def _same_bits(one, other) -> str | None:
    """The first leaf path at which `one` and `other` differ in structure, shape or bits, or None."""
    if jax.tree.structure(one) != jax.tree.structure(other):
        return "its structure"
    for (path, first), second in zip(jax.tree_util.tree_leaves_with_path(one), jax.tree.leaves(other),
                                     strict=True):
        if isinstance(first, RecordedLeaf) or not (hasattr(first, "shape")
                                                    or isinstance(first, (int, float, complex, bool))):
            # A recording's evidence (`RecordedLeaf`), read from its seed's rows: no values of its own.
            continue
        first, second = np.asarray(first), np.asarray(second)
        if first.shape != second.shape or first.dtype != second.dtype or first.tobytes() != second.tobytes():
            return jax.tree_util.keystr(path) or "its value"
    return None


def _of_its_seed(label: str, draw, seed, inputs) -> None:
    """Raise unless `draw` gives `inputs` again at `seed`: a draw of a randomness its seed does not fix (a
    global generator) gives each run of a site other inputs (ledger 2641)."""
    if (where := _same_bits(inputs, draw(seed))) is not None:
        raise UnpairedDraw(f"{label}: the draw at seed {seed} differs at {where} drawn again, so its seed "
                           f"does not fix it")


def _repeat_promised() -> bool:
    """Whether a run promises its arms a bitwise repeat (docs/design/testing.md, check 1): on a CPU, or in
    deterministic mode (`dew._devices.deterministic_ops_requested`). A GPU default that reduces with
    atomics (a backward's dQ summed in arrival order) promises none."""
    from dew._devices import deterministic_ops_requested

    return jax.default_backend() == "cpu" or deterministic_ops_requested()


def _of_its_inputs(label: str, name: str, fn, inputs, first) -> None:
    """Raise unless `fn`, `name` of a check's arms, gives `first` again on the same `inputs`: an arm that
    draws a noise of its own, or reads state its last call left, compares Dew with a reference on draws no
    seed pairs (ledger 2641). Asked only where a repeat is promised (`_repeat_promised`, ledger 2842), and
    of an arm declared to vary from run to run (`run_to_run`) only on a CPU; the CPU lane runs every
    check-1 site, so each arm is held to it there."""
    if not _repeat_promised() or (_varies(fn) and jax.default_backend() != "cpu"):
        return
    if (where := _same_bits(first, fn(*inputs))) is not None:
        raise UnpairedDraw(f"{label}: {name} differs at {where} run again on the same inputs, so its output "
                           f"depends on more than them")


def _over_orders(first: list[tuple], later: list[tuple]) -> list[tuple]:
    """An arm's evidence on a draw (`_evidence`) over two of its orders: each leaf's errors and rows of
    float64 zero the first order's, its worst rows the worse of the two, each worst row's share
    (`RecordedLeaf.share`) that of the order it is, its values finite at both."""
    return [(one[0], one[1], float(np.fmax(one[2], other[2])), float(np.fmax(one[3], other[3])), one[4],
             one[5] and other[5], tuple(other[6][arm] if other[2 + arm] > one[2 + arm] else one[6][arm]
                                        for arm in (0, 1)))
            for one, other in zip(first, later, strict=True)]


def _conditioning(reference_fn, truth_fn, draw, seeds, ran=None) -> Conditioning:
    """Check 1's conditioning of `reference_fn` (tests/conditioning.py): its rounding scale at a draw as its
    arm runs (`summed_as_run`, a window loop's by window), its heavy elements over `seeds` as `ran`, its
    compiled arm where given, runs it; the reference and its truth unrewritten, as their arms run."""
    from rounding_bound import rounding_scale
    scaled = reference_fn.fn if isinstance(reference_fn, _Windowed) else reference_fn

    def scale_of(inputs, seed):
        with summed_as_run():
            if not isinstance(reference_fn, _Windowed):
                return rounding_scale(scaled, *inputs, seed=seed)[1]
            each = [rounding_scale(scaled, *jax.tree.map(lambda leaf, at=at: leaf[at], inputs), seed=seed)[1]
                    for at in range(_window_count(inputs))]
            return jax.tree.map(lambda *scale: None if scale[0] is None else np.stack(scale), *each,
                                is_leaf=lambda leaf: leaf is None)

    return conditioned(unrewritten(ran or reference_fn), unrewritten(truth_fn), draw, seeds, scale_of)


def _truth_and_terms(truth_fn, terms_fn, inputs):
    """The float64 truth of `inputs` (`truth_fn` on them widened) and, with `terms_fn`, its terms (else
    None): a site's own terms program's (`_SiteTerms`), or the truth's own evaluated op by op
    (`float64_terms`)."""
    if terms_fn is None or isinstance(terms_fn, _SiteTerms):
        with jax.enable_x64(new_val=True):
            truth = truth_fn(*widened(inputs))
        if terms_fn is None:
            return truth, None
        with jax.enable_x64(new_val=True):
            return truth, terms_fn.of(truth, *widened(inputs))
    with jax.enable_x64(new_val=True):
        return terms_fn(*widened(inputs))


def _judged_leaves(dew, weighing) -> tuple[list[str], tuple[int, ...]]:
    """The names and row widths of the leaves check 1 judges of Dew's arm `dew`: its own, or with a
    `weighing` each floating leaf's classes, each compacted by its values a row (`Weighed.classed`)."""
    judged = dew if weighing is None else weighing.classed(jax.device_get(dew))
    return ([jax.tree_util.keystr(path) for path, _ in jax.tree_util.tree_leaves_with_path(judged)],
            tuple(int(np.shape(leaf)[-1]) if np.ndim(leaf) >= 2 else 0 for leaf in jax.tree.leaves(judged)))


def _drawn(dew_fn, reference_fn, draw, seeds, truth_fn=None, reference_seeds=None, orders=None,
           library_fn=None, library_leaves: Sequence[str] = (), also: Sequence = (), terms_fn=None,
           label: str = "", examples: bool = False, conditioning=None):
    """Every leaf's errors (`_Draws`) over `seeds`: both functions run on `draw(seed)`'s inputs, truth
    `truth_fn` on them (by default `reference_fn` on them widened to float64). With `reference_seeds`,
    the reference runs on those draws instead, each held to its own draw's truth (`null_runs`), and a
    row of float64 zero is paired only where the two truths are the same values (an orbit's draws).
    With `orders`, each arm also runs at every other order of its own draw (`orders(inputs)` past its
    first, the draw itself), held to that draw's one truth: each arm's worst row is its worst over the
    orders, measured rows of float64 zero included, and its values finite at every one; its errors and
    its rows of float64 zero are the draw's own order's. With `library_fn`, the library arm runs beside
    the reference on Dew's own inputs, held to Dew's truth, and the draws are judged against the larger
    of the two on `library_leaves` (`_Draws.against_larger`), Dew and the truth evaluated once. With
    `also`, further references run beside it on Dew's own inputs, each held to Dew's truth, Dew and the
    truth evaluated once for them all: the reference's draws, then each further reference's, a list. With
    `terms_fn`, the truth is that program's with each output's terms (`float64_terms`),
    every row sized by the larger of its own RMS and its terms' (`_LaidTruths.of`); with a site's own
    terms program (`_SiteTerms`), the truth is as without and its terms that program's, on the same
    widened inputs. With `conditioning` (`conditioning.Conditioning`), every arm's leaves are read as
    residuals in units of the reference's rounding at its truth's inputs (`Weighed.weighed`)."""
    if orders is not None and any(getattr(fn, "recorded", False) for fn in (reference_fn, *also)):
        raise ValueError("a recorded reference holds its recorded draws alone, never another order of them")
    arms = [reference_fn] + ([] if library_fn is None else [library_fn]) + list(also)
    mine, theirs, worst, zero_rows, finite, names, widths = [], [[] for _ in arms], [[] for _ in arms], \
        [[] for _ in arms], [[] for _ in arms], None, None
    # Each arm's per-example ratios of the draw being read, and of every draw read (`examples`).
    stacked: list = [None for _ in arms]
    stacked_rows: list = [None for _ in arms]
    per_example: list[list] = [[] for _ in arms]
    shares: list[list] = [[] for _ in arms]
    per_example_rows: list[list] = [[] for _ in arms]

    def truth_and_terms(inputs):
        return _truth_and_terms(truth_fn or reference_fn, terms_fn, inputs)

    def rows_at(at, seed):
        """Every arm's evidence on the draw at `seed`, the `at`th, its orders' worst rows the larger."""
        nonlocal names, widths
        inputs = draw(seed)
        if not at:
            _of_its_seed(label, draw, seed, inputs)
        # The site's arms' mode, decided on its first draw's inputs before any arm runs, no draw of its own,
        # on the programs the arms run (their scope, `summed_as_run`).
        with summed_as_run():
            _ynn_decided(label, [dew_fn, *arms], inputs)
        truth, terms = truth_and_terms(inputs)
        if reference_seeds is None:
            their_inputs, their_truth, their_terms = inputs, truth, terms
        else:
            their_inputs = draw(reference_seeds[at])
            their_truth, their_terms = truth_and_terms(their_inputs)
        # Each arm in units of the reference's rounding at its truth's inputs, the truth ones.
        weighed = None if conditioning is None else conditioning.at(
            ((truth, inputs, seed), (their_truth, their_inputs, seed if reference_seeds is None
                                     else reference_seeds[at])))
        if orders is None:
            ordered = [(inputs, their_inputs)]
        else:
            ordered = list(zip(orders(inputs), orders(their_inputs), strict=True))
            if len(ordered) < 2:
                raise ValueError(f"orders of {len(ordered)}; an orbit holds the draw's own and another")
        rows = [None for _ in arms]
        stacked[:] = [None for _ in arms]
        stacked_rows[:] = [None for _ in arms]
        for mine_at, theirs_at in ordered:
            # Each arm, and its replay on the same inputs, in the arithmetic arms run in (`summed_as_run`).
            with summed_as_run():
                dew = dew_fn(*mine_at)
                if not at and mine_at is ordered[0][0]:
                    _of_its_inputs(label, "Dew's arm", dew_fn, mine_at, dew)
            # The leaves judged: a weighed arm's split into its heavy and light classes (`Weighed.classed`).
            if names is None:
                names, widths = _judged_leaves(dew, None if weighed is None else weighed[id(truth)])
            # Dew's side read once for every reference (`_prepared`).
            prepared = _prepared(dew, truth, terms) if weighed is None else _prepared(
                *weighed[id(truth)].weighed(dew, "Dew's arm"), weighed[id(truth)].ones)
            for at_arm, arm in enumerate(arms):
                # The reference against itself on the same inputs is one evaluation (`seeds_for`'s
                # sizing); the library arm runs on Dew's own inputs.
                here, true_here, terms_here = ((theirs_at, their_truth, their_terms) if arm is reference_fn
                                               else (mine_at, truth, terms))
                with summed_as_run():
                    other = dew if dew_fn is arm and here is mine_at else arm(*here)
                    if not at and arm is reference_fn and arm is not dew_fn and here is ordered[0][1]:
                        _of_its_inputs(label, "the reference", arm, here, other)
                shared = true_here is truth
                if weighed is not None:
                    terms_here = weighed[id(true_here)].ones
                    other, true_here = weighed[id(true_here)].weighed(other, f"reference {at_arm}")
                if examples and mine_at is ordered[0][0]:
                    # The draw's own order's examples, paired where both arms hold the one truth.
                    held_here = jax.device_get(other)
                    stacked[at_arm], stacked_rows[at_arm] = (
                        (_example_ratios(prepared.leaves, held_here, prepared.truths),
                         _example_worst_rows(prepared.leaves, held_here, prepared.truths))
                        if shared else ([None] * len(prepared.leaves),) * 2)
                evidence = _evidence(prepared, other, true_here, shared=shared, their_terms=terms_here)
                rows[at_arm] = evidence if rows[at_arm] is None else _over_orders(rows[at_arm], evidence)
        return rows

    for at, seed in enumerate(seeds):
        rows = rows_at(at, seed)
        mine.append([entry[0] for entry in rows[0]])
        for at_arm, row in enumerate(rows):
            theirs[at_arm].append([entry[1] for entry in row])
            worst[at_arm].append([(entry[2], entry[3]) for entry in row])
            zero_rows[at_arm].append([entry[4] for entry in row])
            finite[at_arm].append([entry[5] for entry in row])
            shares[at_arm].append([entry[6] for entry in row])
            per_example[at_arm].append(stacked[at_arm])
            per_example_rows[at_arm].append(stacked_rows[at_arm])
    drawn = [_Draws(names, np.asarray(mine), np.asarray(theirs[at_arm]),
                    np.asarray(worst[at_arm]).reshape(len(mine), -1, 2), zero_rows[at_arm],
                    np.asarray(finite[at_arm]), widths, per_example[at_arm] if examples else None,
                    per_example_rows[at_arm] if examples else None, _shares(shares[at_arm], len(mine)))
             for at_arm in range(len(arms))]
    first = drawn[0] if library_fn is None else drawn[0].against_larger(drawn[1], library_leaves)
    return [first, *drawn[len(arms) - len(also):]] if also else first


def _shares(found: list[list], count: int) -> np.ndarray:
    """The worst rows' shares over a run's seeds (`_Draws.shares`, seed by leaf by arm), NaN where one is
    unknown (a recording's summary keeps no row's elements)."""
    return np.asarray([[[np.nan if share is None else share for share in pair] for pair in seed]
                       for seed in found], np.float64).reshape(count, -1, 2)


def _example_ratios(leaves, references, truths) -> list[np.ndarray | None]:
    """Each leaf's per-example log RMS ratios of Dew (`leaves`) against the reference (`references`, a
    tree of the truth's structure), a leaf's first axis its independent examples (`assert_as_exact`'s
    `examples`): half the log of each example's mean squared error from its float64 truth over the
    reference's, examples both arms hold exactly left out. None for a leaf of no axis, or where one arm is
    exact on an example and the other is not."""
    found: list[np.ndarray | None] = []
    for leaf, reference, truth in zip(leaves, jax.tree.leaves(references), truths, strict=True):
        truth = np.asarray(truth, np.float64)
        if truth.ndim == 0 or isinstance(reference, RecordedLeaf):
            found.append(None)
            continue
        axes = tuple(range(1, truth.ndim))
        mine = np.mean(np.square(np.asarray(leaf, np.float64) - truth), axis=axes)
        theirs = np.mean(np.square(np.asarray(reference, np.float64) - truth), axis=axes)
        if np.any((mine > 0) != (theirs > 0)):
            found.append(None)
            continue
        kept = mine > 0
        found.append(0.5 * np.log(mine[kept] / theirs[kept]))
    return found


def _example_worst_rows(leaves, references, truths) -> list[np.ndarray | None]:
    """Each leaf's per-example paired worst rows at a site that stacks examples (`assert_as_exact`'s
    `examples`): on each example, the log of Dew's worst row's RMS error from float64 over its size, less
    the reference's worst row's on the same example, its rows those of the leaf's last axis, sized by their
    truth's RMS, rows under RESOLUTION of their example's size left out. None for a leaf under rank two (no
    rows), or an example with no sized row."""
    found: list[np.ndarray | None] = []
    for leaf, reference, truth in zip(leaves, jax.tree.leaves(references), truths, strict=True):
        truth = np.asarray(truth, np.float64)
        if truth.ndim < 2 or isinstance(reference, RecordedLeaf):
            found.append(None)
            continue
        width = truth.shape[-1]
        rows = truth.reshape(truth.shape[0], -1, width)
        sizes = np.stack([_scaled_rms(example) for example in rows])
        sized = sizes > RESOLUTION * np.sqrt(np.mean(np.square(rows), axis=(1, 2)))[:, None]
        if not sized.any(axis=1).all():
            found.append(None)
            continue
        worst = []
        for arm in (leaf, reference):
            errors = (np.asarray(arm, np.float64) - truth).reshape(rows.shape)
            logs = log_over(np.stack([_scaled_rms(example) for example in errors]),
                            np.where(sized, sizes, 1.0))
            worst.append(np.max(np.where(sized, logs, LOG_ZERO), axis=1))
        found.append(worst[0] - worst[1])
    return found


def _leaf_paths(tree) -> list[str]:
    return [jax.tree_util.keystr(path) for path, _ in jax.tree_util.tree_leaves_with_path(tree)]


@dataclasses.dataclass(frozen=True)
class _Prepared:
    """Dew's side of a draw's evidence, read once for every reference (`_evidence`): its leaves and its
    truth's on the host, their paths, the truth's laid rows and Dew's summaries against them."""
    truth: object
    paths: list
    leaves: list
    truths: list
    held: object
    mine: list


def _term_leaves(terms, truth):
    """A truth's terms tree (`float64_terms`) as its leaves on the host, one a leaf of `truth`, a leaf no
    sum made None; or None. Read by the truth's own structure, so an output the truth holds as None (a
    gradient of an input given as None) takes no leaf of the terms either."""
    if terms is None:
        return None
    return jax.tree.structure(truth).flatten_up_to(jax.device_get(terms))


def _prepared(dew, truth, terms=None) -> _Prepared:
    """Dew's leaves and its truth's on the host in one transfer a tree, held alike (the same leaves at the
    same paths, each the shape of its truth), the truth's rows laid once and Dew's summaries read against
    them (`_summaries`)."""
    dew, truth = jax.device_get((dew, truth))
    paths = _leaf_paths(truth)
    if _leaf_paths(dew) != paths:
        raise AssertionError(f"dew's leaves are {_leaf_paths(dew)}, the truth's {paths}")
    leaves, truths = jax.tree.leaves(dew), jax.tree.leaves(truth)
    for path, leaf, true in zip(paths, leaves, truths, strict=True):
        if np.shape(leaf) != np.shape(true):
            raise AssertionError(f"{path}: dew's shape is {np.shape(leaf)}, the truth's {np.shape(true)}")
    held = _LaidTruths.of(truths, _term_leaves(terms, truth))
    return _Prepared(truth, paths, leaves, truths, held, _summaries(leaves, held))


def _evidence(prepared: _Prepared, reference, their_truth, *, shared: bool, their_terms=None) -> list[tuple]:
    """One draw's `_paired` evidence of every leaf of Dew (`prepared`, read once a draw) and `reference`
    against their truths: the reference's leaves (a recorded one's `RecordedLeaf`) and its truth held to
    Dew's truth's leaves, paths and shapes, so no leaf is paired with another's truth or compared through
    broadcasting; its truth the same tree as Dew's where `shared`."""
    reference = jax.device_get(reference)
    their_truth = prepared.truth if shared else jax.device_get(their_truth)
    paths, leaves, truths = prepared.paths, prepared.leaves, prepared.truths
    for name, tree in (("the reference", reference), ("the reference's truth", their_truth)):
        if _leaf_paths(tree) != paths:
            raise AssertionError(f"{name}'s leaves are {_leaf_paths(tree)}, the truth's {paths}")
    references, their_truths = jax.tree.leaves(reference), jax.tree.leaves(their_truth)
    recorded = [isinstance(their_leaf, RecordedLeaf) for their_leaf in references]
    for path, their_leaf, true, their_true, was in zip(paths, references, truths, their_truths, recorded,
                                                       strict=True):
        shapes = {"the reference's truth": np.shape(their_true)}
        if not was:
            shapes["the reference"] = np.shape(their_leaf)
        for name, shape in shapes.items():
            if shape != np.shape(true):
                raise AssertionError(f"{path}: {name}'s shape is {shape}, the truth's {np.shape(true)}")
    # The truths' rows are read once a draw, for both arms where the two truths are the same values.
    same = all(their_true is true for their_true, true in zip(their_truths, truths, strict=True))
    theirs = _summaries([None if was else leaf for leaf, was in zip(references, recorded, strict=True)],
                        prepared.held if same else _LaidTruths.of(their_truths,
                                                                   _term_leaves(their_terms, their_truth)))
    row = []
    for path, leaf, their_leaf, true, their_true, was, summary, their_summary in zip(
            paths, leaves, references, truths, their_truths, recorded, prepared.mine, theirs, strict=True):
        their = their_leaf if was else their_summary
        if np.size(leaf) == 0:
            # An empty leaf (a one-tap convolution's history) is exact by construction: held to the
            # reference's dtype (its shape is held above; the truth's dtype, float64, is its own), it
            # adds no error and resolves nothing.
            kept = str(jnp.dtype(leaf.dtype))
            assert kept == their.dtype, (path, kept, their.dtype)
            row.append((0.0, 0.0, LOG_ZERO, LOG_ZERO, NO_ZERO_ROWS, True, (0.0, 0.0)))
            continue
        # Rows pair where the two truths are the same values: the same draw, or an orbit's draws.
        paired = was or shared or np.array_equal(np.asarray(true), np.asarray(their_true))
        row.append(_paired(path, summary, their, paired=paired))
    return row


def _paired(name: str, mine: RecordedLeaf, theirs: RecordedLeaf, *, paired: bool = True) -> tuple:
    """One seed's evidence of a leaf for check 1 (`_Draws`), Dew's and the reference's alike from
    `leaf_summary`: their mean squared errors, their worst sized rows, the rows of float64 zero with
    both arms' errors on them (`ZeroRows`, pooled over the run's seeds by `pooled_zero_rows`), and
    whether every value was finite. A row the truth sizes zero (below RESOLUTION of its leaf)
    is one of float64 zero unless a live reference measures it: its error there below the row's own
    float64 RMS, so it carries the row better than zero would. The two are taken by one arithmetic
    (`_scaled_rms`), so a reference exactly zero on a row is exactly the row's size from it and never
    measures it (ledger 1831: two arithmetics broke that tie, and Dew's ordinary rounding on a residue
    row read 1e8 of the row). That is an engineering definition of a measured row, not a proof of
    float64's accuracy on it; a sum that cancels to float64's residue is never measured so, and a
    product far below its leaf, which float64 holds to its own precision, is.
    A measured row is judged at its own size as any row is, its error over its size entering both worst
    rows: against the reference's worst row over the leaf, which holds it with less power than the zero
    rows' rule held it against the reference's own error on it. Both arms are classified alike, from the
    one truth they share. A recorded reference keeps no row's own size, and the recording's sketch holds
    the truth at the leaf's scale, not a tiny row's, so its rows keep the zero rows' rule. A row of zero
    in Dew's truth the reference's lacks (a recorded truth recomputed otherwise) is refused. Not
    `paired`, the two held to truths of their own draws (`null_runs`), no row is paired."""
    if not paired:
        return (mine.mse, theirs.mse, mine.log_worst, theirs.log_worst, NO_ZERO_ROWS,
                mine.finite and theirs.finite, (mine.share, theirs.share))
    at_zero = dict(zip(theirs.zero_rows.tolist(), theirs.zero_errors.tolist(), strict=True))
    unrecorded = [at for at in mine.zero_rows.tolist() if at not in at_zero]
    if unrecorded:
        raise AssertionError(f"{name}: the truth holds rows of float64 zero the reference's did not, "
                             f"{unrecorded[:4]}")
    sizes = (dict(zip(theirs.zero_rows.tolist(), theirs.zero_sizes.tolist(), strict=True))
             if theirs.zero_sizes is not None else {})
    worst, their_worst, zero = mine.log_worst, theirs.log_worst, []
    for at, error in zip(mine.zero_rows.tolist(), mine.zero_errors.tolist(), strict=True):
        size = sizes.get(at, 0.0)
        if at_zero[at] < size:
            worst = max(worst, float(log_over(error, size)))
            their_worst = max(their_worst, float(log_over(at_zero[at], size)))
        elif error or at_zero[at]:
            # A row both arms hold exactly scales both pooled errors alike and passes on every seed: no
            # evidence, not carried (a window's untouched embedding rows, 283k a draw at 2469's chunk sites).
            zero.append((at, error, at_zero[at]))
    rows, errors, their_errors = (np.asarray(column, dtype) for column, dtype in zip(
        zip(*zero, strict=True) if zero else ((), (), ()), (np.int64, np.float64, np.float64), strict=True))
    # A measured row of float64 zero that became an arm's worst keeps no share of its own: unknown.
    shares = (mine.share if worst == mine.log_worst else None,
              theirs.share if their_worst == theirs.log_worst else None)
    return (mine.mse, theirs.mse, worst, their_worst, (rows, errors, their_errors),
            mine.finite and theirs.finite, shares)


def _judged(draws: _Draws, index: int, tails: tuple[float, float] = (LOWER_TAIL, UPPER_TAIL)
            ) -> tuple[float, float, float]:
    """A leaf's pooled ratio and its interval over the draws, its `tails` left out."""
    a, b = draws.mine[:, index], draws.theirs[:, index]
    return (_pooled(a, b), *_interval(a, b, tails=tails))


def _judgement(draws: _Draws, label: str, tails: tuple[float, float] = (LOWER_TAIL, UPPER_TAIL), *,
               contained: bool = True, typical: bool = False) -> tuple[list[Finding], list[Finding]]:
    """Check 1's findings over `draws` (`assert_as_exact`): each leaf over PARITY (`_status`), a row or
    a value it fails, and each leaf its draws leave inconclusive; with `typical` (a final verdict), a
    leaf its pooled ratio leaves inconclusive passed by its typical draw where that decides
    (`_typical_status`)."""
    over, unresolved = [], []
    for index, name in enumerate(draws.names):
        leaf, count = label + name, len(draws.mine)
        a, b = draws.mine[:, index], draws.theirs[:, index]
        if not draws.finite[:, index].all():
            nonfinite = int(np.sum(~draws.finite[:, index]))
            over.append(Finding(leaf, "not finite", nonfinite,
                                f"{leaf}: a value is not finite on {nonfinite} of {count} seeds"))
            continue
        if np.sum(b) == 0:
            # No ratio to judge, but rows over an orbit's other orders still are; Dew's own RMS error is
            # what a change can make worse.
            if np.sum(a) > 0:
                over.append(Finding(leaf, "exact reference", root_mean(a),
                                    f"{leaf}: the reference equals float64 on every seed and dew does not"))
        else:
            ratio, low, high = _judged(draws, index, tails)
            status = _status(ratio, low, high, contained=contained)
            # The ratio's log from its sums' logs, finite where a sum or the ratio itself would overflow.
            measure = (log_sum(a) - log_sum(b)) / 2
            found = None
            if status == "inconclusive" and contained and typical:
                # The pooled ratio's sums unresolved: its draws' paired ratios pass it where they qualify.
                found = _typical_status(a, b, tails, draws.of_examples(index), draws.scalar(index))
                status = found[0]
            if status == "inconclusive":
                unresolved.append(Finding(
                    leaf, "inconclusive", measure, f"{leaf}: inconclusive: the pooled ratio {ratio:.3f}'s "
                    f"interval over {count} seeds is [{low:.3f}, {high:.3f}], wider than {PARITY}"
                    + ("" if found is None else f", and the typical draw's {found[1]:.3f}'s "
                       f"[{found[2]:.3f}, {found[3]:.3f}]: {found[4]}")
                    + "; give the site larger inputs or more seeds (`seeds_for`)"))
            elif status == "over":
                over.append(Finding(leaf, "pooled", measure, f"{leaf}: dew's RMS error from float64 is "
                                    f"{ratio:.3f} times the reference's over {count} seeds, interval "
                                    f"[{low:.3f}, {high:.3f}] (allowed {PARITY})"))
        # The paired examples' test keeps its width's factor (its null is exchangeability, whatever a row's
        # error is); the maximum's takes a row one value rules at one value's (`_Draws.factor`).
        factor = row_factor(draws.widths[index])
        paired = draws.of_example_rows(index) if contained else None
        if paired is not None and (found_rows := _paired_rows(paired, factor)) < tails[0]:
            # A site that stacks examples: each example's worst row against the reference's on it.
            over.append(Finding(
                leaf, "row", float(np.quantile(paired, 1 - tails[0])),
                f"{leaf}: dew's worst row is past {factor} times the reference's on the same example on "
                f"{int(np.sum(paired > math.log(factor)))} of {len(paired)} examples, the reference's past "
                f"it on {int(np.sum(paired < -math.log(factor)))} (p {found_rows:.1e}; `_paired_rows`)"))
        my_row, their_row = float(np.max(draws.worst[:, index, 0])), float(np.max(draws.worst[:, index, 1]))
        factor = draws.factor(index)
        if paired is None and not my_row <= math.log(factor) + their_row:
            over.append(Finding(leaf, "row", my_row - their_row,
                                f"{leaf}: the worst row over {count} seeds is {shown(my_row)} from float64 "
                                f"over its size, the reference's {shown(their_row)}, allowed "
                                f"{shown(math.log(factor) + their_row)} ({factor} times it)"))
        zero = pooled_zero_rows([seed[index] for seed in draws.zero_rows])
        if zero:
            # The measure is the worst of them, the first the example.
            at, my_error, their_error = zero[0]
            worst = log_ratio([row[1] for row in zero], [row[2] for row in zero])
            over.append(Finding(leaf, "zero rows", worst,
                                f"{leaf}: {len(zero)} rows of float64 zero are further from it over "
                                f"{count} seeds than ROW_FACTOR times the reference's own, row {at} by "
                                f"{my_error:.3e} against {their_error:.3e} (`pooled_zero_rows`)"))
    return over, unresolved


def _staged(drawn, seeds: int, label: str, *,
            two_stage: bool = True) -> tuple[list[Finding], list[Finding], int]:
    """Check 1's two pre-declared stages over `drawn(start, stop)`, the draws of seeds start to stop: at
    `seeds`, a site whose every leaf passes (`_judgement`: its ratio, rows and values) passes; any leaf
    that does not draws the second stage, REDRAW times as many further seeds, and every leaf is judged
    again over all of them, that verdict final. Its findings, the leaves left inconclusive and the stage
    reached. With `two_stage` false, the procedure before it, for comparison: tails
    PREVIOUS_TAILS, the interval as it is, only a leaf whose ratio is over PARITY drawing again."""
    tails = (LOWER_TAIL, UPPER_TAIL) if two_stage else PREVIOUS_TAILS
    draws = drawn(0, seeds)
    if two_stage:
        further = any(_judgement(draws, label, tails))
    else:
        further = any(np.sum(draws.theirs[:, index]) > 0 and _judged(draws, index, tails)[0] > PARITY
                      for index in range(len(draws.names)))
    if further:
        draws = draws.joined(drawn(seeds, staged_units(seeds)))
    # The stage drawn by the pooled ratio alone; the final verdict's leaves it leaves inconclusive asked of
    # their typical draw (`_judgement`'s `typical`).
    return (*_judgement(draws, label, tails, contained=two_stage, typical=two_stage), 2 if further else 1)


STAGES_LOG = STAGES
"""The environment variable (`check1_wire.STAGES`) naming a file each check-1 judgement appends its stage, its
verdict and its leaves of narrow rows (NARROW_FACTORS) to, one JSON line (`_logged`): what a train reads every
site's stage-two rate from, every attempt counted, and which sites' local errors rest on check 2."""


def _logged(label: str, stage: int, verdict: str, narrow_leaves: list[str], ynn: str | None = None) -> None:
    import json

    path = os.environ.get(STAGES_LOG)
    if path:
        with open(path, "a", encoding="utf-8") as log:
            log.write(json.dumps({"site": os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" (", 1)[0],
                                  "label": label, "stage": stage, "verdict": verdict,
                                  "narrow_leaves": narrow_leaves, "ynn": ynn}) + "\n")


@dataclasses.dataclass(frozen=True)
class _SiteTerms:
    """A site's own program of its truth's terms (`assert_as_exact`'s `terms`): `fn` on a draw's inputs
    widened to float64, the truth's tree of each entry's terms as `float64_terms` sizes them."""

    fn: Callable

    def of(self, truth, *inputs):
        """The terms of `truth`, the draw's at `inputs`, held to its leaves, paths and shapes."""
        terms = jax.device_get(self.fn(*inputs))
        if _leaf_paths(terms) != _leaf_paths(truth):
            raise ValueError(f"the site's terms are {_leaf_paths(terms)}, the truth's {_leaf_paths(truth)}")
        for path, term, true in zip(_leaf_paths(truth), jax.tree.leaves(terms), jax.tree.leaves(truth),
                                    strict=True):
            if np.shape(term) != np.shape(true):
                raise ValueError(f"{path}: the site's terms' shape is {np.shape(term)}, the truth's "
                                 f"{np.shape(true)}")
        return terms


def _cancelling_terms(draw, truth, program=None):
    """The truth program a site's rows are sized by their terms with (`assert_as_exact`'s `terms`), where its
    truth holds a row of float64 zero (below RESOLUTION of its leaf) on its first draw, decided from the truth
    alone before any arm runs (bible §7, ledger 2851): a row whose truth is exactly zero or cancels is judged
    at the size of what it sums, as the one-draw rule judges it, not held to the reference's residue there (a
    fused attention's causal first row, whose one key makes its gradient exactly zero in float64, its
    rounding's residue Dew's and the reference's alike). None where no row is zero, or where the site
    states a truth no program's terms are traced from (run as given, recorded, a window loop's; ledger
    3198): every other truth traces, and a failure to is the site's. `program`, the truth as the draws
    run it (`_truth_program`), evaluates the first draw's truth, so its first call's run and trace serve
    the draws too, once a process (a map task's draws, DRAWS); a reference standing for it runs
    compiled, not eagerly (ledger 3774)."""
    if truth is None or isinstance(truth, _Windowed) or _given(truth) or getattr(truth, "recorded", False):
        return None
    # The truth on the site's first draw, and its terms only where it holds a zero row: only a zero row whose
    # terms are not zero too gains a size, so a site whose zero rows sum nothing (a query that keeps no key)
    # keeps the reference's residue rule without tracing its terms a draw.
    run = program or unrewritten(jax.jit(truth))
    # The draw as every seed's is drawn, outside float64, then widened (under x64 a draw's own defaults
    # would draw float64 values the arms never see).
    inputs = widened(draw(0))
    with jax.enable_x64(new_val=True):
        leaves = [*map(np.asarray, jax.tree.leaves(run(*inputs)))]
        if not np.any(_LaidTruths.of(leaves).sizes == 0):
            return None
        truths, terms = float64_terms(truth, *inputs)
    plain, termed = _LaidTruths.of(jax.tree.leaves(truths)), _LaidTruths.of(
        jax.tree.leaves(truths), _term_leaves(terms, truths))
    return truth if bool(np.any((plain.sizes == 0) & (termed.sizes > 0))) else None


def _site(dew_fn, reference_fn, label: str, *, seeds: int, truth_fn, library_fn, library_row, library_leaves,
          also, terms):
    """A check 1 site's arguments held to what it may ask (`assert_as_exact`): its term program, the
    traced truth (refused run as given or recorded), and its further references; refused below
    `FEWEST_SEEDS`, or with a library arm not named by its row and leaves."""
    terms_fn = None
    if callable(terms):
        terms_fn = _SiteTerms(unrewritten(terms))
    elif terms:
        terms_fn = truth_fn if truth_fn is not None else reference_fn
        terms_fn = terms_fn.fn if isinstance(terms_fn, _Windowed) else terms_fn
        if _given(terms_fn) or getattr(terms_fn, "recorded", False):
            raise ValueError(f"{label}: term sizing traces the truth program; a truth run as given or "
                             f"recorded has no terms")
    if seeds < FEWEST_SEEDS:
        raise Inconclusive([Finding(label, "inconclusive", FEWEST_SEEDS / max(seeds, 1), f"{label}: {seeds} "
                                    f"seeds cannot be judged; check 1 takes at least {FEWEST_SEEDS} "
                                    f"(`seeds_for`)")])
    if (library_fn is None) != (library_row is None) or (library_fn is None) != (not library_leaves):
        raise ValueError(f"{label}: a library arm is the owner's exception, named with its ledger row and "
                         f"the leaves it covers")
    return terms_fn, dict(also or {})


def _recorded(label: str, drawn_for, references: int, arms, upto: int | None = None,
              weighed: dict | None = None, offset: int = 0) -> bool:
    """A map task's draws (`record_units`): each of a site's `references` draws over its range, up to
    `upto` where given, with its arms' YNNPACK mode (`_mode_of`) and, weighed, its classes' digest
    (`Conditioning.digest`), the seconds its weighing took beside its draws'. False where the process is
    no map task."""
    def payload(start, stop):
        draws = [drawn_for(index)(start, stop) for index in range(references)]
        held = (weighed or {}).get("of")
        return {"draws": draws, "refused": _mode_of(arms), "heavy": None if held is None else held.digest()}
    return record_units(label, payload, upto=upto, offset=offset,
                        phases=lambda: {"weighing": (weighed or {}).get("seconds", 0.0)})


def _pooled_draws(label: str, references: int, arms):
    """A judge's draws (`pooled_units`): `drawn_for(index)` over its map tasks' records of this site,
    `drawn(start, stop)` the pooled draws of those seeds, refused past the seeds the records hold from seed
    0 on; the site's arms take the mode the map tasks ran them in, which must be one. None where the
    process is no judge."""
    pieces = pooled_units(label)
    if pieces is None:
        return None
    held = pieces[-1][1]
    modes = {payload["refused"] for _, _, payload in pieces}
    if len(modes) > 1:
        raise PoolRefused(f"{label}: the map tasks ran the arms in different modes: "
                          f"{sorted(map(str, modes))}", held)
    refused = modes.pop()
    if len({payload.get("heavy") for _, _, payload in pieces}) > 1:
        raise PoolRefused(f"{label}: the map tasks weighed by different classes (`Conditioning.digest`)",
                          held)
    pairings = {arm.pairing for arm in _paired_arms(arms)}
    for pairing in pairings:
        pairing.decided, pairing.refused = True, refused
    if pairings and refused is not None:
        SWITCHED.append({"label": label, "refused": refused})
    pooled = [functools.reduce(_Draws.joined, (payload["draws"][index] for _, _, payload in pieces))
              for index in range(references)]

    def drawn_for(index: int):
        def drawn(start, stop):
            if stop > held and start == held:
                # The second stage's seeds, drawn by later map tasks a judge asks for (`ask_later`).
                ask_later(label, held, stop)
            if stop > held:
                raise PoolRefused(f"{label}: its stage draws seeds {start} to {stop}; the map tasks drew 0 "
                                  f"to {held} ({DRAWS} covers 0 to (1 + REDRAW) times the declared seeds)",
                                  held)
            return pooled[index].between(start, stop)
        return drawn
    return drawn_for


def assert_as_exact(dew_fn, reference_fn, draw, label: str, *, seeds: int, truth_fn=None,
                    orders=None, library_fn=None, library_row: int | None = None,
                    library_leaves: Sequence[str] = (), also: Mapping[str, Callable] | None = None,
                    terms: bool | Callable | None = None, examples: bool = False,
                    conditioning: int | None = None) -> None:
    """Check 1 of the gate: on par. Each leaf's RMS error from float64,
    pooled over the site's declared seeds, within PARITY times the reference's, and its worst row over
    those seeds within ROW_FACTOR times the reference's worst (NARROW_FACTORS' factor at rows of one or
    two values); each row of float64 zero (below RESOLUTION of its leaf where a live reference does not
    measure it, `_paired`) no further from it than ROW_FACTOR times the reference on that row, pooled
    over the seeds on which the row is so classified (`pooled_zero_rows`: the root of the sum of
    squared errors, not seed by seed); every value finite.

    `draw(seed)` returns the inputs of one draw, the same for a seed on every run; `dew_fn(*inputs)`
    and `reference_fn(*inputs)` return the same tree, and truth is `reference_fn` on the inputs
    widened to float64, or `truth_fn` on them where the reference fixes a precision of its own (an
    output dtype it closes over). A draw may be an orbit's: the same values under an exact symmetry
    the seed picks. `seeds` is the site's declared count, sized by `seeds_for` so the reference
    compared with itself resolves PARITY, or, where the arms share their upstream rounding, so Dew
    paired with the reference does. The gate compiles the two functions (`_arms`), once for every seed:
    two Dew paths without YNNPACK's fusions, a Dew arm beside an arm run `as_given` with them.

    Where a computation has an orbit of summation orders, `orders(inputs)` lists a draw's inputs at K
    of them, the draw itself first: an exact symmetry of the result (a batch's rows in another order
    under a mean, a sum's terms in another), so one truth holds at every order. Both arms run at every
    one (`_drawn`), and each one's worst row is its worst over them, compared alike; the pooled ratio,
    the rows of float64 zero and the sizing stay the draw's own order's. A row that nearly cancels
    rounds worst at some orders and well at others, so one order's worst row is no measure of it, and
    the reference's luckiest order would fail an equally exact Dew; over the orbit Dew's worst is held
    within ROW_FACTOR of the reference's worst at any of the same orders, which is as strict as the
    reference is spread over them: a Dew worse than the reference at every order still fails, and one
    the reference is exact against at every order fails at any order it is not.

    A leaf passes when its pooled ratio (`_pooled`) is at most PARITY and its seeds resolve it: the
    ratio's bootstrap interval over them (tails LOWER_TAIL and UPPER_TAIL, widened to hold the ratio)
    lying wholly at or under PARITY, or no wider than PARITY. Check 1 is two pre-declared stages
    (`_staged`): where every leaf passes at the declared seeds the site passes; where any leaf does not
    (over PARITY, inconclusive, failing a row or a value), the site draws REDRAW times as many further
    seeds and every leaf is judged again over all of them, its ratio, its interval and its rows. Over
    PARITY there, it fails, a finding in the code or the reference arm; still inconclusive there, it is
    Inconclusive, never a pass, and its site is sized again (`seeds_for`). A pass certifies the ratio at
    most PARITY² and an over by its ratio above one, each at a nominal 5e-4 (CONFIDENCE). Every
    judgement's stage is logged (STAGES_LOG). A leaf the reference holds exactly on every seed, Dew
    must too.

    `library_fn`, the owner's exception under BIBLE §7 (recorded on ledger row `library_row`): where Dew's
    arm decomposes a product the reference computes whole and the library's own accuracy at the
    pieces' shapes is the residual, a library arm on the same inputs runs the arm's pieces through the
    library and combines them exactly (in float64), and Dew is held within PARITY of the larger of the
    reference's error and that arm's, seed by seed, on the leaves the exception covers
    (`library_leaves`, by their paths; `_Draws.against_larger`). Every other leaf, and every other rule,
    is as above; a site without it is judged against the reference alone.

    `also` names further references Dew is held to on the same draws, each judged by its own check 1
    as the reference is (its label `label`, against its name), Dew and the truth evaluated once a draw
    for them all (`_drawn`): one Dew arm against several references, the truth as above. Each compiles
    paired with Dew's arm (`_arms`), and a plain Dew arm,
    whose compilation its pairing decides, takes references of one kind, plain or `as_given`.

    `examples` declares that every leaf's first axis stacks independent examples, each its own draw of the
    site's geometry: a leaf its pooled ratio leaves inconclusive is then judged by its examples' paired
    ratios (`_typical_status`: each draw's statistic its examples' median, and their tail about it).

    `terms` sizes each row by the larger of its float64 truth's RMS and its terms' (owner, 10-29); by
    default (None) a site whose truth holds a row of float64 zero on its first draw is sized so
    (`_cancelling_terms`, ledger 2851), False never: the
    truth program, `truth_fn` or the reference's, evaluated op by op with each output's terms
    (`float64_terms`), as the one-draw rule sizes a row that cancels (`row_errors`). A row whose truth
    cancels, exactly or but for one term holding most of its weight (an attention row that keeps one key),
    is judged at the size of what it sums by the row rule, every arm alike, not by the reference's own
    distance there; a row whose terms are zero too (a query that keeps no key) has no size and stays held
    to the reference's distance, exactly zero where the reference is. Every other row, and every leaf's
    pooled ratio, as without. The truth program is traced, so it is JAX's: a truth run as given has no
    terms and is refused. The traced terms are those of each leaf's final sums alone, so a row whose
    cancellation sits behind its final sum has none there: an attention gradient's query that keeps one
    key, whose logit gradient P (dP - delta) is zero in float64, sums zero terms into dQ. Such a site
    passes its own terms program as `terms` instead: a function of the draw's inputs widened to float64
    returning the truth's tree of each entry's terms (`float64_terms`' sizes), carried through that
    cancellation (tests/attention_oracle.py's `attention_terms`), every arm sized by it alike; the truth
    is as without, so it may run as given.

    `conditioning`, the weighed check's own declared seeds (`seeds_for(..., conditioning=True)`), judges
    the site weighed where it stays undecided with its budget spent (tests/conditioning.py, ledger 1714),
    each verdict logged under the check that decided it, the weighed one's ", conditioned".

    This is an engineering gate, deterministic and reproducible, not a guarantee against every error
    law: "never wrong" is check 2's, each element within its operation's certified bound
    (tests/certified.py)."""
    if conditioning is not None:
        _deferred(dew_fn, reference_fn, draw, label, seeds=seeds, conditioning=conditioning,
                  truth_fn=truth_fn, orders=orders, library_fn=library_fn, library_row=library_row,
                  library_leaves=library_leaves, also=also, terms=terms, examples=examples)
        return
    # A map task draws no seed past the site's stages (`REDRAW`): a sharded test's cases each declare their
    # own, the shards cut at the most of them, and a recording holds no more.
    _site_judged(dew_fn, reference_fn, draw, label, seeds=seeds, truth_fn=truth_fn, orders=orders,
                 library_fn=library_fn, library_row=library_row, library_leaves=library_leaves, also=also,
                 terms=terms, examples=examples, weighing=None, upto=staged_units(seeds))


def _deferred(dew_fn, reference_fn, draw, label: str, *, seeds: int, conditioning: int, truth_fn, orders,
              library_fn, library_row, library_leaves, also, terms, examples: bool) -> None:
    """`assert_as_exact`'s site with `conditioning` (tests/conditioning.py): the unweighted check over its
    own `seeds`, whole, and where it stays undecided the weighed check over `conditioning` seeds, its heavy
    elements classed over the draws it takes, alike in every process that runs it. A sharded site's map
    tasks draw each check's units apart, the weighed check's past the unweighted one's in the plan's units
    (tools/armada/ci.py `draws_of`), and its judge pools the weighed check's where it defers."""
    plain, span = staged_units(seeds), staged_units(conditioning)

    def weighed():
        _site_judged(dew_fn, reference_fn, draw, f"{label}, conditioned", seeds=conditioning,
                     truth_fn=truth_fn, orders=orders, library_fn=library_fn, library_row=library_row,
                     library_leaves=library_leaves, also=also, terms=False, examples=examples,
                     weighing=functools.partial(_conditioning, draw=draw, seeds=range(span)), upto=span,
                     offset=plain)

    try:
        _site_judged(dew_fn, reference_fn, draw, label, seeds=seeds, truth_fn=truth_fn, orders=orders,
                     library_fn=library_fn, library_row=library_row, library_leaves=library_leaves,
                     also=also, terms=terms, examples=examples, weighing=None, upto=plain)
    except Inconclusive:
        weighed()
    else:
        if drawing():
            weighed()


def _site_judged(dew_fn, reference_fn, draw, label: str, *, seeds: int, truth_fn, orders, library_fn,
                 library_row, library_leaves, also, terms, examples: bool, weighing, upto: int | None = None,
                 offset: int = 0) -> None:
    """`assert_as_exact`'s judgment of one site under `label`, weighed where `weighing` is given (tests/
    conditioning.py: built of the site's compiled reference and truth where it draws, never in its judge),
    a map task recording its range's draws up to `upto` where given, its classes' digest beside them."""
    given = reference_fn
    terms_fn, also = _site(dew_fn, reference_fn, label, seeds=seeds, truth_fn=truth_fn, library_fn=library_fn,
                           library_row=library_row, library_leaves=library_leaves, also=also, terms=terms)
    truth_program = _truth_program(truth_fn)
    terms_fn = _site_terms(terms_fn, terms, draw, truth_fn if truth_fn is not None else reference_fn,
                           truth_program)
    narrow_leaves: list[str] = []
    compiled = _arms_with(dew_fn, reference_fn, list(also.values()), label)
    _refuse_numpy_arms(draw, dew_fn, reference_fn, *([] if library_fn is None else [library_fn]),
                       *also.values())
    dew_fn, reference_fn, further = compiled
    library_fn = None if library_fn is None else _arms(library_fn, library_fn)[1]
    truth_fn = truth_program
    # Each range's draws, every reference's, computed once for the stages of them all.
    ranges: dict[tuple[int, int], list[_Draws]] = {}
    # The site's weighing where it draws (`weighing`), built once, and the seconds that took.
    weighed: dict[str, object] = {}

    def conditioning_now():
        if weighing is not None and "of" not in weighed:
            began = time.perf_counter()
            weighed["of"] = weighing(given, truth_fn, ran=reference_fn)
            weighed["seconds"] = time.perf_counter() - began
        return weighed.get("of")

    def drawn_for(index: int):
        def drawn(start, stop):
            assert not judging(), f"{label}: a judge draws nothing of its own, its draws are its pool's"
            if (start, stop) not in ranges:
                draws = _drawn(dew_fn, reference_fn, draw, range(start, stop), truth_fn, orders=orders,
                               library_fn=library_fn, library_leaves=library_leaves, also=further,
                               terms_fn=terms_fn, label=label, examples=examples,
                               conditioning=conditioning_now())
                ranges[start, stop] = draws if further else [draws]
                narrow_leaves[:] = _narrow(ranges[start, stop][0])
            return ranges[start, stop][index]
        return drawn

    arms = [dew_fn, reference_fn, *([] if library_fn is None else [library_fn]), *further]
    # A map task of the site's draws, or their judge (DRAWS, POOL).
    if _recorded(label, drawn_for, 1 + len(further), arms, upto, weighed, offset):
        return
    pooled = _pooled_draws(label, 1 + len(further), arms)
    if pooled is not None:
        drawn_for = pooled
        narrow_leaves[:] = _narrow(drawn_for(0)(0, 1))
    _verdicts(drawn_for, seeds, [label, *(f"{label}, against {name}" for name in also)], narrow_leaves, arms)


def _site_terms(terms_fn, terms, draw, truth, truth_program):
    """The site's terms program: `terms_fn`, the site's own, or by default (`terms` None) one sized from
    `truth` where its first draw holds a row of float64 zero (`_cancelling_terms`); none where `terms` is
    False, or in a judge, whose draws are its map tasks' records, each row sized by the terms its task
    found (`map_judge.judging`): no truth runs there. A program traced for its terms is traced once a
    signature, its static runs compiled once for the site's every draw (`float64_terms`)."""
    if terms is not False and not judging() and terms_fn is None:
        program = truth_program if isinstance(truth_program, Truth) else None
        terms_fn = _cancelling_terms(draw, truth, program)
    return terms_fn if terms_fn is None or isinstance(terms_fn, _SiteTerms) else functools.partial(
        float64_terms, terms_fn)


def _narrow(draws: _Draws) -> list[str]:
    """The leaves of rows of one or two values (NARROW_FACTORS)."""
    return [name for name, width in zip(draws.names, draws.widths, strict=True) if width in NARROW_FACTORS]


def _verdicts(drawn_for, seeds: int, labels: list[str], narrow_leaves: list[str], arms) -> None:
    """Each reference's check 1 over its draws (`_staged`, `drawn_for(index)` the draws against reference
    `index`, judged under `labels[index]`), each logged (STAGES_LOG); a GateFailure where any leaf is
    over, else an Inconclusive where any is left unresolved."""
    overs, unresolveds = [], []
    for index, named in enumerate(labels):
        over, unresolved, stage = _staged(drawn_for(index), seeds, named)
        _logged(named, stage, _verdict(over, unresolved), narrow_leaves, _mode_of(arms))
        overs += over
        unresolveds += unresolved
    if overs:
        raise GateFailure(_with_mode(overs + unresolveds, arms))
    if unresolveds:
        raise Inconclusive(_with_mode(unresolveds, arms))


def _moved(draws: _Draws, stage: int) -> tuple[list[str], list[str], bool]:
    """The leaves a change moves beyond rounding (`assert_moves`), why each other one does not, and
    whether any of those is unresolved (its interval still reaching above PARITY). A stage spends half of
    LOWER_TAIL, shared among the leaves, so the two looks over every leaf spend no more than check 1's."""
    moved, kept, unresolved = [], [], False
    tail = LOWER_TAIL / 2 / max(len(draws.names), 1)
    for index, name in enumerate(draws.names):
        mine, theirs = draws.mine[:, index], draws.theirs[:, index]
        if not draws.finite[:, index].all():
            kept.append(f"{name}: a value not finite, which moves nothing a ratio can show")
            unresolved = True
        elif np.sum(theirs) == 0:
            # The original holds the float64 truth exactly: any move off it is past its rounding.
            (moved if np.sum(mine) > 0 else kept).append(
                f"{name}: {'moved off' if np.sum(mine) > 0 else 'kept'} the original's exact value")
        else:
            ratio = _pooled(mine, theirs)
            low, high = _interval(mine, theirs, tails=(tail, UPPER_TAIL))
            (moved if low > PARITY else kept).append(
                f"{name}: at stage {stage}, the changed RMS distance from the original's float64 run is "
                f"{ratio:.3f} times the original's own, its lower end {low:.3f}")
            unresolved |= low <= PARITY and max(high, ratio) > PARITY
    return moved, kept, unresolved


def assert_moves(original_fn, changed_fn, draw, label: str, *, seeds: int, truth_fn=None) -> None:
    """A change that must move the result beyond rounding (a sensitivity check): over check 1's draws
    (`assert_as_exact`), some leaf of `changed_fn` sits further from the original's float64 run than
    PARITY times `original_fn`'s own rounding, by the lower end of its pooled ratio's interval, so a
    change rounding could account for passes at a nominal 5e-4 at most over its leaves and both stages
    (`_moved`). The truth is `original_fn` on the inputs widened to float64, or `truth_fn`, which must be
    the original's own float64 semantics (an output dtype it closes over), never another computation's.
    A leaf the original holds exactly moves on any departure from it. Two pre-declared stages, as check
    1's: at `seeds`, then, where no leaf moved, REDRAW times as many further seeds; still unmoved, it
    fails, or is Inconclusive where a leaf's interval still reaches above PARITY. Where the change is
    structural and the alternative is bitwise identity (another seed, another mask), exact inequality
    says enough and this is not needed."""
    if seeds < FEWEST_SEEDS:
        raise Inconclusive([Finding(label, "inconclusive", FEWEST_SEEDS / max(seeds, 1), f"{label}: {seeds} "
                                    f"seeds cannot be judged; check 1 takes at least {FEWEST_SEEDS} "
                                    f"(`seeds_for`)")])
    _refuse_numpy_arms(draw, changed_fn, original_fn)
    changed_fn, original_fn = _arms(changed_fn, original_fn)
    truth_fn = _truth_program(truth_fn)
    draws = _drawn(changed_fn, original_fn, draw, range(seeds), truth_fn, label=label)
    moved, kept, unresolved = _moved(draws, 1)
    if not moved:
        further = range(seeds, staged_units(seeds))
        draws = draws.joined(_drawn(changed_fn, original_fn, draw, further, truth_fn, label=label))
        moved, kept, unresolved = _moved(draws, 2)
    if not moved:
        found = (f"{label}: the change moves no leaf beyond rounding over {len(draws.mine)} seeds:\n"
                 + "\n".join(kept))
        raise (Inconclusive([Finding(label, "inconclusive", 1.0, found)]) if unresolved
               else AssertionError(found))


SIZING_FROM = 1 << 20
"""The first seed a paired sizing draws (`seeds_for` with `dew_fn`): past every site's verdict seeds
and below every null run's (tools/gate_null_runs.py, from 2^16 runs of at least 80 seeds on), so the
count a site declares is fixed by draws its verdict never sees."""


MOST_SEEDS = 512
"""The most seeds a site's sizing declares (`seeds_for`, `seeds_resolving`): a site that needs more
takes larger inputs or fewer shared factors, not more seeds."""


def recorded_pilots(reference_fn, draw, count: int):
    """A view of `reference_fn`, a recorded reference (one `.recorded`), that holds `count` pilot draws
    from SIZING_FROM on, disjoint from every verdict's seeds, the reference and its truth recorded on the
    inputs Dew's arm takes live: the draws a paired sizing of it takes (`seeds_for`). The view is the site's
    own; `reference_fn` is left as it is. Every pilot is read now (`draw`'s inputs at each), so a
    recording that does not hold them all is refused here, not at the doubling that reaches one."""
    if not getattr(reference_fn, "recorded", False):
        raise ValueError("pilots belong to a recorded reference; a live one draws its own from SIZING_FROM")
    if count < 1:
        raise ValueError(f"a recording of {count} pilots")
    for seed in range(SIZING_FROM, SIZING_FROM + count):
        reference_fn(*draw(seed))

    @functools.wraps(reference_fn)
    def view(*inputs):
        return reference_fn(*inputs)
    view.recorded, view.pilots = True, range(SIZING_FROM, SIZING_FROM + count)
    return view


def seeds_for(reference_fn, draw, label: str, *, start: int = FEWEST_SEEDS, most: int = MOST_SEEDS,
              truth_fn=None, dew_fn=None, orders=None, examples: bool = False,
              conditioning: bool = False) -> int:
    """The seeds a site declares: the fewest, doubling from `start`, for which the pooled ratio's
    interval is no wider than PARITY in every leaf, on the count's own draws and again on as many more
    (`_confirmed`): a doubling stops at the first count whose draws resolve, and where a leaf's errors
    are heavy-tailed across draws those draws may hold none of the tail the verdict's will (ledger 1824:
    a site sized at 16 left Inconclusive at 160), so a second block of the count's size must resolve
    too. Of the reference compared with itself, its errors on
    two disjoint seed sets resampled apart (`_interval`'s control), by default; with `dew_fn`, paired:
    Dew's errors against the reference's on the same draws, the two resampled by the same seeds as the
    verdict resamples them (`_interval`), on draws from SIZING_FROM on that no verdict takes. A pair
    that shares its upstream rounding, the same inputs through the same upstream arithmetic, has errors
    whose sizes vary across draws together: their ratio resolves in fewer draws than each alone does, or
    where each alone, heavy-tailed across draws, never does, and the verdict judges that same paired
    ratio, so a paired count is the one its verdict needs (docs/design/testing.md states how it was
    validated). The width asked of it is PARITY
    either way. Paired sizing holds out its draws only where `draw` keys an independent stream by the
    whole seed: a draw that enumerates finitely or by a modulus (`(arange(n) + seed) % vocab`) repeats a
    verdict's inputs at SIZING_FROM and needs a pilot stream of its own first. A recorded reference
    (`indexed_recording`), which holds its recorded seeds alone, is refused, its site keeping its
    recorded count, unless its recording holds pilot draws of their own from SIZING_FROM on, the
    reference and its truth on the inputs Dew's arm takes live (`recorded_pilots`, or a summarized
    recording's pilots, `indexed_recording`): its paired sizing then takes those pilots alone,
    up to as many as are held. Fails past `most`, where the site needs larger inputs or fewer shared
    factors, not more seeds.

    Each draw is evaluated once however many doublings the sizing takes: a count's draws are its
    predecessor's and the seeds it adds, twice the count of them with the confirming block, the
    reference compared with itself takes its seeds 0 to four times the count (each block's first half
    against its second), and the reference on the same inputs runs once. `orders` sizes a site whose
    verdict sweeps them (`assert_as_exact`'s `orders`): each seed's evidence is its draw's over every
    order, as the verdict reads it. `examples` sizes a site that stacks them as its verdict judges it."""
    if dew_fn is not None and getattr(reference_fn, "recorded", False):
        pilots = getattr(reference_fn, "pilots", None)
        if pilots is None:
            raise ValueError(f"{label}: a recorded reference holds its recorded seeds alone, never a paired "
                             f"sizing's from SIZING_FROM; its site keeps its recorded count")
        most = min(most, len(pilots) // 2)
    paired = dew_fn is not None
    _refuse_numpy_arms(draw, *([dew_fn] if paired else []), reference_fn)
    # A weighed site (`assert_as_exact`'s `conditioning`) is sized as it is judged, weighed, its heavy
    # elements classed over its first stages' draws.
    weighing = (_conditioning(reference_fn, truth_fn, draw, range(staged_units(max(start, FEWEST_SEEDS))))
                if conditioning else None)
    dew_fn, reference_fn = _arms(dew_fn if paired else reference_fn, reference_fn)
    dew_fn = dew_fn if paired else None
    truth_fn = _truth_program(truth_fn)
    count, drawn, wide = max(start, FEWEST_SEEDS), None, []
    if count > most:
        raise AssertionError(f"{label}: sizing starts at {count} seeds, past the {most} it may take; "
                             f"give the site more pilots or start lower")
    first = 0 if dew_fn is None else SIZING_FROM
    while count <= most:
        held = 0 if drawn is None else len(drawn.theirs)
        block = 2 * count if dew_fn is None else count
        fresh = _drawn(dew_fn or reference_fn, reference_fn, draw, range(first + held, first + 2 * block),
                       truth_fn, orders=orders, label=label, examples=examples and dew_fn is not None,
                       conditioning=weighing)
        drawn = fresh if drawn is None else drawn.joined(fresh)
        wide = _confirmed(drawn.names, drawn.theirs, count, None if dew_fn is None else drawn.mine)
        if not wide:
            return count
        count *= 2
    if dew_fn is not None:
        # The leaves the pooled ratio does not size within `most`, its draws ruled by their largest (each arm
        # rounding apart, ledger 2469), sized by their typical draw on the draws already held, every other
        # leaf by its pooled ratio as before: the first count at which each is.
        stubborn, count = tuple(wide), max(start, FEWEST_SEEDS)
        while count <= most:
            wide = _confirmed(drawn.names, drawn.theirs, count, drawn.mine, typical=stubborn,
                              examples=drawn.examples, scalars=[drawn.scalar(index)
                                                                for index in range(len(drawn.names))])
            if not wide:
                return count
            count *= 2
    compared = "the reference compared with itself" if dew_fn is None else "Dew paired with the reference"
    raise AssertionError(f"{label}: {compared} does not resolve {PARITY} within {most} seeds at {wide[:4]}; "
                         f"give the site larger inputs or fewer shared factors")


def _leaf_examples(examples: list, start: int, count: int, index: int) -> list[np.ndarray] | None:
    """Leaf `index`'s per-example log ratios over seeds `start` to `start + count` of a sizing's draws
    (`_Draws.examples`), or None where a seed's are not paired."""
    found = [seed[index] for seed in examples[start:start + count]]
    return None if any(ratios is None for ratios in found) else found


def _confirmed(names, theirs: np.ndarray, count: int, mine: np.ndarray | None = None,
               typical: tuple = (), examples: list | None = None, scalars: Sequence[bool] = ()) -> list[str]:
    """The leaves `seeds_for` leaves unresolved at `count`: on the count's own block of draws or on the
    next block of as many (`seeds_for`'s confirmation), each judged as the sizing judges it: paired,
    `mine` Dew's squared errors against the reference's (`theirs`, seed by leaf) on the same draws, or
    the reference compared with itself (`_unresolved_against_itself`, each block twice the count); the
    leaves named in `typical` sized by their typical draw too (`_sized`), `scalars` saying which leaves hold
    one value an example."""
    wide: list[str] = []
    for start in (0, count) if mine is not None else (0, 2 * count):
        if mine is not None:
            held, ours = theirs[start:start + count], mine[start:start + count]
            unresolved = [name for index, name in enumerate(names) if np.sum(held[:, index]) > 0
                          and not _sized(ours[:, index], held[:, index], name in typical,
                                         None if examples is None else _leaf_examples(examples, start, count,
                                                                                      index),
                                         bool(scalars) and scalars[index])]
        else:
            unresolved = _unresolved_against_itself(theirs[start:start + 2 * count], names, count)
        wide += [name for name in unresolved if name not in wide]
    return wide


def seeds_resolving(errors: np.ndarray, names, label: str, *, start: int = FEWEST_SEEDS,
                    most: int = MOST_SEEDS) -> int:
    """`seeds_for`'s count for the reference compared with itself, from its errors already measured:
    `errors` holds its squared errors a seed (rows, from seed 0) and a leaf (columns), such as a
    recording's merged shards, and the count is the fewest, doubling from `start` to `most`, that
    resolves every leaf on its own block and the next (`_confirmed`), as `seeds_for` doubles. Fails past
    `most`, as `seeds_for` does, and where the rows held run out first: the recording then holds too
    few seeds."""
    count, wide = max(start, FEWEST_SEEDS), []
    while count <= most and 4 * count <= len(errors):
        wide = _confirmed(names, errors, count)
        if not wide:
            return count
        count *= 2
    if count > most:
        raise AssertionError(f"{label}: the reference compared with itself does not resolve {PARITY} within "
                             f"{most} seeds at {wide[:4]}; give the site larger inputs or fewer shared "
                             f"factors")
    raise AssertionError(f"{label}: the reference compared with itself does not resolve {PARITY} within the "
                         f"{len(errors)} seeds recorded, {len(errors) // 4} a half of each block")


def recording(path) -> dict[str, np.ndarray]:
    """A recording of a reference over seeds (`indexed_recording`'s `stored`), read whole: where a pin
    holds it, from the fetched store's local copy, checked against the pin; else from the file at
    `path`, as git holds one of at most 1 MiB (tools/fetched_fixtures.py)."""
    from tool_support import load

    with np.load(load("fetched_fixtures").local(path), allow_pickle=False) as stored:
        return dict(stored)


def null_runs(reference_fn, draw, label: str, *, seeds: int, runs: int, first: int = 0,
              truth_fn=None, two_stage: bool = True, dew_fn=None, orders=None,
              paired: bool = False, library_fn=None,
              library_leaves: Sequence[str] = (), conditioning: int | None = None
              ) -> list[tuple[str, int, list[str]]]:
    """Check 1 run `runs` times on an equally exact pair, the reference against itself on disjoint
    draws: each run's verdict, "pass", "over" or "inconclusive", the stage it reached and its findings
    (`_judgement`'s, each naming its leaf and rule), a run judged as `assert_as_exact` judges a site
    (`_staged`: its declared `seeds`, its second stage, every leaf's ratio, interval and rows), or as the
    procedure before it with `two_stage` false. Run r of a call takes draws no other run of any call
    with the same `first` grid takes: seeds from (first + r)·2·(1 + REDRAW)·seeds on, the first half its
    "Dew"'s and the second its reference's, so the runs are independent repetitions as far as `draw`
    makes its draws so. The null control of a site's operation class: its false-red rate, every
    nonpassing verdict counted, bounds the gate's own on that class. With `dew_fn` a fault of the
    reference (a dropped term, a wrong scale), the runs measure the procedure's power on that class,
    both procedures on the same draws. `paired` runs both arms on the same draws, as a site does, `dew_fn`
    then the other implementation (an equally exact one, or a fault); `orders` judges rows over each
    draw's orbit, as a site that declares one. The runs evaluate their arms as given: a class's
    arms compile themselves where they are programs, and a fault injects itself on the host.
    `library_fn` judges each run against the larger of the reference's error and that library arm's
    on `library_leaves`, as `assert_as_exact` does (paired runs: the library arm is measured on Dew's
    own draws). `conditioning`, the weighed check's declared seeds, judges every run as a weighed site is
    (`assert_as_exact`): unweighted first over `seeds`, and weighed over its own where that stays
    undecided, the verdict then ", conditioned"'s."""
    if paired and dew_fn is None:
        raise ValueError("a paired null run takes its other implementation, `dew_fn`")
    # The reference and the truth run their programs unrewritten, as a site's do (`_arms`, `_truth_program`).
    reference_fn = (_Windowed(unrewritten(reference_fn.fn)) if isinstance(reference_fn, _Windowed)
                    else unrewritten(reference_fn))
    truth_fn = unrewritten(truth_fn)
    verdicts = []
    # Each run's draws: both budgets whole, the unweighted check's and the weighed one's, and the
    # reference's own beside them where unpaired, so no two runs share a draw.
    span = staged_units(max(seeds, conditioning or 0))
    for run in range(first, first + runs):
        base = run * 2 * span

        def judged(weighing, count, base=base):
            def drawn(start, stop):
                return _drawn(dew_fn or reference_fn, reference_fn, draw, range(base + start, base + stop),
                              truth_fn, orders=orders,
                              reference_seeds=None if paired else range(base + span + start,
                                                                        base + span + stop),
                              library_fn=library_fn, library_leaves=library_leaves, conditioning=weighing)
            return _staged(drawn, count, label, two_stage=two_stage)

        over, unresolved, stage = judged(None, seeds)
        verdict = _verdict(over, unresolved)
        if conditioning is not None and verdict == "inconclusive":
            weighing = _conditioning(reference_fn, truth_fn, draw,
                                     range(base, base + staged_units(conditioning)))
            over, unresolved, stage = judged(weighing, conditioning)
            verdict = f"{_verdict(over, unresolved)}, conditioned"
        verdicts.append((verdict, stage, [str(finding) for finding in over + unresolved]))
    return verdicts


def null_rate_bound(reds: int, runs: int, confidence: float = CONFIDENCE) -> float:
    """The one-sided Clopper-Pearson upper bound on a false-red rate at `confidence` after `reds` red
    runs of `runs`: 1 - (1 - confidence)^(1/runs) at none, so 0 of 1,400 bounds it at 0.49%."""
    from scipy import stats

    return 1.0 if reds >= runs else float(stats.beta.ppf(confidence, reds + 1, runs - reds))


def declared_seeds(path) -> dict[str, int]:
    """Each site's declared seed count in a recording of a reference over seeds (`indexed_recording`):
    its `<site>.seeds`, written at record time by `seeds_for` on the recorded reference."""
    return {name.removesuffix(".seeds"): int(value) for name, value in recording(path).items()
            if name.endswith(".seeds")}


def drawn_on_the_host(inputs_of: Callable, seed: int):
    """`inputs_of(seed)` drawn on the host's CPU, as a recording's inputs were, whatever device the test runs
    on: jax.random's draws are the same bits on every device but its normal's float transform is not (a
    recorded site's seed drawn on an H100 had another digest than on the CPU, the recording's, in 157 of the
    recurrence GPU flow's cells, draw-digest at 996fa29f1), so a draw on the test's device would not be the
    recorded inputs. The arrays then go to the device where the run reads them."""
    with jax.default_device(jax.devices("cpu")[0]):
        return inputs_of(seed)


def input_digest(inputs) -> np.uint64:
    """A seed's inputs (a dict of NumPy arrays, a torch draw's converted) as 64 bits of the SHA-256 of
    each array's name, dtype by name, shape and bytes in C order: what a recording of outputs alone
    keeps, so that inputs drawn again are checked to be the recorded ones. The dtype goes by its name, as
    some narrow formats share a byte code (float8_e4m3fn's and float8_e4m3fnuz's), and the shape as the
    array's own, a scalar's included."""
    digest = hashlib.sha256()
    for name in sorted(inputs):
        value = np.asarray(inputs[name])
        digest.update(f"{name}:{value.dtype}:{value.shape};".encode())
        digest.update(value.tobytes())
    return np.frombuffer(digest.digest()[:8], np.uint64)[0]


LEAF_ENTRIES = ("reference", "truth", "mse", "worst", "finite", "sketch", "zero_rows", "zero_errors",
                "zero_offsets", "shape", "dtype")
"""The kinds of a recording's entries kept one a leaf, `<site>.<kind>.<leaf>` (a pilot's
`<site>.pilot.<kind>.<leaf>`): a full recording's outputs and truths, a summarized one's evidence and
sketches. Every other entry (inputs, digests, counts, provenance) is the whole recording's."""


def _of_leaves(stored, site: str, leaves) -> dict:
    """`stored` with the entries of `leaves` alone of those kept one a leaf (LEAF_ENTRIES), verdict and
    pilot alike, and every entry the whole recording shares; a leaf the recording lacks is refused."""
    wanted, found, kept = set(leaves), set(), {}
    for key, value in stored.items():
        kind, _, name = key.removeprefix(f"{site}.").removeprefix("pilot.").partition(".")
        if key.startswith(f"{site}.") and kind in LEAF_ENTRIES and name:
            if name not in wanted:
                continue
            found.add(name)
        kept[key] = value
    if wanted - found:
        raise ValueError(f"{site}: the recording holds no leaves {sorted(wanted - found)}")
    return kept


def indexed_recording(stored, site: str, seeds: int, *, inputs_of=None, truth_fn=None, leaves=None):
    """`assert_as_exact`'s draw, reference and truth for a reference recorded over seeds, where it cannot
    run in the test (a CUDA-only library, a fetched source): `stored` holds `<site>.reference.<name>` and
    `<site>.truth.<name>`, and `<site>.input.<name>` unless `inputs_of(seed)` draws a seed's inputs again
    (a seeded NumPy or torch draw the generator made them by), every array stacked over the seeds on its
    first axis, the truth in float64, and `<site>.redraw`, check 1's REDRAW it was recorded for. Inputs
    drawn again are checked against the recording's `<site>.input_digest` (`input_digest`), so a
    generator that changed since is named, not judged. A draw is `(index, inputs)`, the index an
    integer `widened` keeps; the reference and the truth read the recorded outputs at it, and Dew's
    function takes the same two arguments and ignores the index. The recording covers check 1's redraw,
    (1 + REDRAW) times the declared seeds, so a numerical failure stays a finding rather than running
    out of recorded seeds.

    A recording of the reference's evidence alone (`record_case(..., summary=True)`: `<site>.mse.<name>`,
    `<site>.worst.<name>`, `<site>.finite.<name>` and `<site>.sketch.<name>` over the seeds, the truth's
    shape `<site>.shape.<name>`, and its
    rows of float64 zero's indices and errors, `<site>.zero_rows.<name>` and `<site>.zero_errors.<name>`,
    each seed's span of them in `<site>.zero_offsets.<name>`) keeps no outputs: its reference arm gives
    check 1 that evidence (`RecordedLeaf`), and its truth is `truth_fn(inputs)`, Dew's own float64 path,
    held at every seed to the recorded truth's sketch (`truth_sketch`) within float64's rounding of the
    computation itself, the oracle rule's bound (`assert_computes_the_oracle`, its roundings counted on
    `truth_fn`'s program, `chain_roundings`): a truth recomputed wrong moves its sketch by far more, so
    check 1 never judges Dew against its own float64 path where that path is not the reference's, as
    far as a wrong truth's move is independent of the sketch's signs (SKETCH). A seed whose recomputed
    truth has a row of float64 zero the recording's had not is refused. Its reference and truth are
    host functions, called as check 1 calls them, not traced. Where it holds pilots (`<site>.pilot.`,
    `merge_shards`), seeds from SIZING_FROM on read them, and its reference carries them
    (`recorded_pilots`'s `.pilots`), so a paired sizing of it takes those draws alone (`seeds_for`).

    With `leaves`, the recording is read for those leaves alone (`_of_leaves`): a site whose leaves are
    judged in groups, each its own test, reads each group so, its Dew arm and truth giving that group's
    leaves. Check 1 judges every leaf on its own tails, so a group draws its second stage on its own
    leaves' account alone, which looks at a passing leaf no more often than the whole site would."""
    if leaves is not None:
        stored = _of_leaves(stored, site, leaves)
    if any(name.startswith(f"{site}.sketch.") for name in stored):
        return _summarized(stored, site, seeds, inputs_of, truth_fn)

    def tree(kind):
        prefix = f"{site}.{kind}."
        return {name.removeprefix(prefix): value for name, value in stored.items() if name.startswith(prefix)}

    inputs, reference, truth = tree("input"), tree("reference"), tree("truth")
    narrow = [name for name, value in truth.items()
              if jnp.issubdtype(value.dtype, jnp.floating) and value.dtype != np.float64]
    if narrow:
        raise AssertionError(f"{site}: the recorded truth {narrow} is not float64; record it in float64, as "
                             f"no later widening recovers what a narrower recording dropped")
    if inputs_of is None and not inputs:
        raise AssertionError(f"{site}: no inputs recorded and no `inputs_of` to draw them again")
    if int(stored.get(f"{site}.redraw", -1)) != REDRAW:
        raise AssertionError(f"{site}: the recording holds no `{site}.redraw` of {REDRAW}, which "
                             f"record_case writes and the fetched store's rule reads it by")
    digests = stored.get(f"{site}.input_digest")
    if inputs_of is not None and digests is None:
        raise AssertionError(f"{site}: `inputs_of` draws the inputs again and the recording keeps no "
                             f"digest to check them by; record the outputs alone (`store_inputs=False`)")
    recorded = [*reference.values(), *truth.values(),
                *(inputs.values() if inputs_of is None else [digests])]
    count = min(len(value) for value in recorded)
    if count < staged_units(seeds):
        raise AssertionError(f"{site}: {count} recorded seeds, and check 1 over {seeds} may draw "
                             f"{staged_units(seeds)}; record (1 + REDRAW) times the declared seeds")

    def draw(seed):
        if inputs_of is None:
            return np.int32(seed), {name: value[seed] for name, value in inputs.items()}
        drawn = drawn_on_the_host(inputs_of, seed)
        if input_digest(drawn) != digests[seed]:
            raise AssertionError(f"{site}: seed {seed}'s inputs drawn again are not the recorded ones (their "
                                 f"digest differs); the generator's draw changed since the recording")
        return np.int32(seed), drawn

    def lookup(values):
        looked = jax.jit(lambda index, inputs: {name: jnp.asarray(value)[index]
                                              for name, value in values.items()})

        def recorded(index, inputs):
            return looked(index, inputs)
        recorded.recorded = True
        return recorded

    return draw, lookup(reference), lookup(truth)


def _summarized(stored, site: str, seeds: int, inputs_of, truth_fn):
    """`indexed_recording` of a recording of the reference's evidence alone."""
    def tree(kind):
        prefix = f"{site}.{kind}."
        return {name.removeprefix(prefix): np.asarray(value) for name, value in stored.items()
                if name.startswith(prefix)}

    kinds = ("mse", "worst", "finite", "sketch", "zero_rows", "zero_errors", "zero_offsets")
    verdict, pilot = ({kind: tree(prefix + kind) for kind in kinds} for prefix in ("", "pilot."))
    mse, worst, finite, sketch, zero_offsets = (verdict[kind] for kind in
                                                ("mse", "worst", "finite", "sketch", "zero_offsets"))
    shapes, dtypes = tree("shape"), tree("dtype")
    if truth_fn is None or inputs_of is None:
        raise AssertionError(f"{site}: a recording of the reference's evidence alone takes `truth_fn`, "
                             f"Dew's float64 path, and `inputs_of`, its inputs drawn again")
    if int(stored.get(f"{site}.redraw", -1)) != REDRAW:
        raise AssertionError(f"{site}: the recording holds no `{site}.redraw` of {REDRAW}")
    digests = stored[f"{site}.input_digest"]
    pilots = len(stored.get(f"{site}.pilot.input_digest", ()))
    count = min([len(value) for part in (mse, worst, finite, sketch) for value in part.values()]
                + [len(value) - 1 for value in zero_offsets.values()])
    if min(count, len(digests)) < staged_units(seeds):
        raise AssertionError(f"{site}: {min(count, len(digests))} recorded seeds, and check 1 over {seeds} "
                             f"may draw {staged_units(seeds)}")
    roundings = {}

    def held(seed):
        """`seed`'s evidence, its input digests and its row in them: a verdict seed's, or a pilot's."""
        if seed < SIZING_FROM:
            return verdict, digests, seed
        if seed >= SIZING_FROM + pilots:
            raise AssertionError(f"{site}: seed {seed} is past the recording's {pilots} pilots from "
                                 f"SIZING_FROM")
        return pilot, stored[f"{site}.pilot.input_digest"], seed - SIZING_FROM

    def draw(seed):
        drawn = drawn_on_the_host(inputs_of, seed)
        _, recorded, row = held(seed)
        if input_digest(drawn) != recorded[row]:
            raise AssertionError(f"{site}: seed {seed}'s inputs drawn again are not the recorded ones (their "
                                 f"digest differs); the generator's draw changed since the recording")
        return np.int32(seed), drawn

    def reference(index, inputs):
        evidence, _, row = held(int(index))

        def leaf(name):
            start, stop = evidence["zero_offsets"][name][row], evidence["zero_offsets"][name][row + 1]
            return RecordedLeaf(float(evidence["mse"][name][row]), float(evidence["worst"][name][row]),
                                evidence["zero_rows"][name][start:stop],
                                evidence["zero_errors"][name][start:stop],
                                finite=bool(evidence["finite"][name][row]), dtype=str(dtypes[name]))

        return {name: leaf(name) for name in mse}

    reference.recorded = True
    if pilots:
        reference.pilots = range(SIZING_FROM, SIZING_FROM + pilots)

    # The truth recomputed, compiled as the gate compiles a truth; its roundings are counted on its trace.
    live = _truth_program(truth_fn)

    @as_given
    def truth(index, inputs):
        index = int(index)
        computed = live(inputs)
        if not roundings:
            # One draw's roundings, a window loop's its window's: each window is a draw of its own.
            program, arguments = _one_draws_program(truth_fn, (inputs,))
            roundings["count"] = chain_roundings(jax.make_jaxpr(program)(*arguments))
        eps = float(np.finfo(np.float64).eps)
        if set(computed) != set(sketch):
            raise AssertionError(f"{site}: `truth_fn` gives the leaves {sorted(computed)}, the recording "
                                 f"{sorted(sketch)}")
        for name, value in computed.items():
            value = np.asarray(value, np.float64)
            if value.shape != tuple(shapes[name].tolist()):
                raise AssertionError(f"{site}: the truth of {name!r} recomputed is {value.shape}, the "
                                     f"recorded reference's {tuple(shapes[name].tolist())}")
            evidence, _, row = held(index)
            recorded = evidence["sketch"][name][row]
            if not np.isfinite(value).all():
                raise AssertionError(f"{site}: seed {index}'s float64 truth of {name!r} recomputed is not "
                                     f"finite")
            # The leaf's norm, scaled by its largest magnitude first so no square overflows.
            largest = float(np.max(np.abs(value))) if value.size else 0.0
            norm = largest * float(np.linalg.norm(value / largest)) if largest > 0 else 0.0
            # The oracle rule's bound on the difference's norm, and the rounding of both sketches, the
            # recorded truth's magnitudes no more than this one's and the difference's (`sketch_rounding`).
            oracle = 2 * roundings["count"] * eps * norm
            magnitude = float(np.sum(np.abs(value))) * (1 + value.size * eps) + np.sqrt(value.size) * oracle
            bound = (SKETCH_DEVIATIONS * oracle + 2 * sketch_rounding(magnitude)) * (1 + 1e-6)
            apart = truth_sketch(value, index) - recorded
            # Its root mean square, scaled by its largest first so no square overflows.
            widest = float(np.max(np.abs(apart))) if apart.shape == recorded.shape else np.inf
            away = (widest * float(np.sqrt(np.mean(np.square(apart / widest)))) if 0 < widest < np.inf
                    else widest)
            if not (np.isfinite(bound) and away <= bound):
                raise AssertionError(
                    f"{site}: seed {index}'s float64 truth of {name!r} recomputed is not the recorded "
                    f"reference's: its sketch is {away:.3e} away (root mean square), past float64's "
                    f"rounding of the computation "
                    f"({roundings['count']} roundings); `truth_fn` does not compute what the reference's "
                    f"truth did")
        return computed

    return draw, reference, truth




FALSE_FAILURE = 1e-6
"""The chance a port exactly as exact as its reference fails the K-order
rule. The suite holds about 150 rules and CI runs it some tens of times a
day, so this is one false failure in a few hundred days."""

ORDERS = 52
"""The rounding orders the K-order rule takes on each side: the fewest K
for which a mean of K squared distances over another such mean exceeds
FACTOR^2 with probability at most FALSE_FAILURE when both are equally exact.
A squared RMS of Gaussian roundings is a weighted sum of one-degree
chi-squares, whose spread is widest when one direction carries all of it,
as a training step's can (Kimi Linear's updated logits spread like 3 to 5
degrees, where one draw against one exceeds the factor about 10% of the
time). At one degree the ratio is F(K, K), and F(52, 52) passes 4 with
probability 8e-7; any larger spread in degrees is more concentrated still,
so K holds for any computation (tests/test_bf16_reference.py derives it)."""


def assert_as_exact_over_orders(dew, reference, label: str) -> None:
    """Dew's RMS over ORDERS rounding orders within FACTOR times the
    reference's RMS over the same orders.

    `dew` and `reference` hold one distance from float64 per order (each a
    `distance`), where an order is an exact symmetry of the computation, a
    permutation of the residual stream (tests/residual_orders.py), so only
    the rounding moves."""
    dew, reference = np.asarray(dew, np.float64), np.asarray(reference, np.float64)
    assert dew.shape == reference.shape == (ORDERS,), f"{label}: {dew.shape} and {reference.shape} orders"
    mine, theirs = float(np.sqrt(np.mean(dew ** 2))), float(np.sqrt(np.mean(reference ** 2)))
    assert within(mine, FACTOR * theirs), (
        f"{label}: dew is {mine:.3e} from float64 (rms over {ORDERS} orders), the reference "
        f"{theirs:.3e} (allowed {FACTOR} times it)")


def assert_rows_as_exact_over_orders(dew, reference, label: str, sizes=None) -> None:
    """Each row held over ORDERS rounding orders, beside `assert_as_exact_over_orders`'s global RMS:
    `dew` and `reference` `[ORDERS, rows]` hold each order's distance of each row from float64, and
    `sizes` `[rows]` each row's float64 size (its terms', `row_errors`), one where none is given. Each
    row's RMS over the orders over its size, Dew's worst within ROW_FACTOR of the reference's worst,
    as `rows_over` holds one draw."""
    _assert_rows_over_orders(dew, np.asarray(reference, np.float64)[None], label, sizes)


def _assert_rows_over_orders(dew, members, label: str, sizes=None) -> None:
    """`assert_rows_as_exact_over_orders` against the worst of `members` `[members, ORDERS, rows]`
    at each row."""
    dew, members = np.asarray(dew, np.float64), np.asarray(members, np.float64)
    assert members.shape[1:] == dew.shape and dew.shape[0] == ORDERS, (
        f"{label}: {dew.shape} and {members.shape[1:]} rows over orders, the rule takes {ORDERS} orders")
    sizes = np.ones(dew.shape[1]) if sizes is None else np.asarray(sizes, np.float64)
    sized = sizes > 0
    if not sized.any():
        return  # rows of float64 zero, which the element rule holds (`assert_elements_as_exact`)
    mine = np.sqrt(np.mean(np.square(dew), axis=0))[sized] / sizes[sized]
    theirs = np.max(np.sqrt(np.mean(np.square(members), axis=1)), axis=0)[sized] / sizes[sized]
    worst, their_worst = float(np.max(mine)), float(np.max(theirs))
    assert within(worst, ROW_FACTOR * their_worst), (
        f"{label}: dew's worst row is {worst:.3e} from float64 (rms over {ORDERS} orders, over its size), "
        f"the reference's {their_worst:.3e} (allowed {ROW_FACTOR:g} times it)")


def assert_computes_the_oracle(reference_in_float64, truth, label: str, *, roundings: int) -> None:
    """The reference run in float64 within the float64 rounding of the
    computation itself of an independent float64 oracle: FACTOR times
    `roundings` (the longest chain of roundings an output goes through, its
    reductions' lengths summed along the way) float64 unit roundoffs of the
    truth's RMS scale. Two float64 runs of one function part by that much at
    most; any difference in the function moves the twin by many orders of
    magnitude more. The fp32 error is no measure of it: a near-exact fp32 path
    leaves the float64 runs a few ulps apart all the same."""
    eps = float(np.finfo(np.float64).eps)
    apart, scale = distance(reference_in_float64, truth), float(np.sqrt(np.mean(np.square(truth))))
    assert within(apart, FACTOR * roundings * eps * scale), (
        f"{label}: the reference in float64 is {apart:.3e} from the oracle (rms), more than float64 "
        f"rounding over {roundings} steps of its scale {scale:.3e}")


def assert_same_artifact(recomputed, committed, f64, name: str) -> None:
    """A committed fixture is an evaluation of its reference at its own precision (an artifact check, not
    a parity): its largest error from the float64 run `f64` at most twice `recomputed`'s, the reference
    run again here. Another torch build sums in another order, so the two need not agree bit for bit
    across machines; a fixture that is not this reference's output sits orders further."""
    fresh, stored = (float(np.max(np.abs(value - f64))) for value in (recomputed, committed))
    print(f"{name}: committed {stored:.3e} and recomputed {fresh:.3e} from float64, "
          f"{float(np.max(np.abs(recomputed - committed))):.3e} apart")
    # Written so a NaN or an infinity on either side fails it.
    if not (np.isfinite(stored) and np.isfinite(fresh) and within(stored, 2 * fresh)):
        raise AssertionError(f"{name}: the committed fixture is not this reference's output")


def chain_roundings(closed_jaxpr) -> int:
    """An upper bound on the roundings any output of a traced computation
    goes through, for `assert_computes_the_oracle`: each operation rounds
    once and each reduction once per term after its first (a contraction's
    terms, a convolution's window times its input features, a reduction's
    axes), summed over every operation of the jaxpr and of the jaxprs it
    calls, a scan's body once per step. A chain from an input to an output
    visits each operation at most once, so no chain makes more."""

    def length(equation) -> int:
        name, params = equation.primitive.name, equation.params
        shape = equation.invars[0].aval.shape if equation.invars else ()
        if name == "dot_general":
            (contracting, _), _ = params["dimension_numbers"]
            return int(np.prod([shape[axis] for axis in contracting], dtype=np.int64))
        if name == "conv_general_dilated":
            kernel = equation.invars[1].aval.shape
            spec = params["dimension_numbers"].rhs_spec
            return int(np.prod([kernel[axis] for axis in spec[1:]], dtype=np.int64))
        if name.startswith(("reduce_", "argmax", "argmin", "cum")) and "axes" in params:
            return int(np.prod([shape[axis] for axis in params["axes"]], dtype=np.int64))
        return 1

    def walk(jaxpr, times: int) -> int:
        total = 0
        for equation in jaxpr.eqns:
            total += times * length(equation)
            steps = equation.params.get("length", 1) if equation.primitive.name == "scan" else 1
            for value in equation.params.values():
                for inner in value if isinstance(value, (tuple, list)) else (value,):
                    body = getattr(inner, "jaxpr", inner)
                    if hasattr(body, "eqns"):
                        total += walk(body, times * steps)
        return total

    return walk(closed_jaxpr.jaxpr, 1)


def assert_rounds_where_the_reference_does(dew, reference, truth, label: str) -> None:
    """Dew within a quarter of the reference's RMS rounding error of the
    reference's own output."""
    apart, theirs = distance(dew, reference), distance(reference, truth)
    assert within(apart, theirs / 4), (
        f"{label}: dew is {apart:.3e} from the reference (rms), whose own error is {theirs:.3e} "
        f"(ratio {apart / theirs:.2f}, allowed 0.25)")
