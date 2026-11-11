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
import os
import time
from collections.abc import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from arm_arithmetic import FORMATTED, Truth, rounds_nothing, summed_as_run, unrewritten, widened
from arm_forms import (
    _layers,
    _one_draws_program,
    _RunToRun,
    _varies,
    _window_count,
    _Windowed,
    over_windows as over_windows,
    run_to_run as run_to_run,
    stacked_on_host as stacked_on_host,
    windows as windows,
)
from check1_wire import CENSUS, DRAWS, STAGES
from float64_terms import evaluated_on, float64_terms, lowered, scan_groups
from map_judge import PoolRefused, ask_later, drawing, judging, pooled_units, record_units
from truth_sketch import SKETCH_DEVIATIONS, sketch_rounding, truth_sketch

from dew._accuracy import distance, reduction_bound, within
from dew.nn import programs
from dew.parity import (
    CONFIDENCE,
    FACTOR,
    FEWEST_SEEDS,
    LOWER_TAIL,
    MOST_SEEDS,
    NARROW_FACTORS,
    PARITY,
    RECORDED_KINDS,
    ROW_FACTOR,
    SIZING_FROM,
    UPPER_TAIL,
    Conditioning,
    Finding,
    GateFailure,
    Inconclusive,
    RecordedLeaf,
    _below_one,
    _drawn_from,
    _Draws,
    _enough_seeds,
    _evidence,
    _example_ratios,
    _example_worst_rows,
    _interval,
    _judged,
    _judged_leaves,
    _LaidTruths,
    _leaf_paths,
    _pooled,
    _prepared,
    _sized_count,
    _staged,
    _term_leaves,
    _verdict,
    heavy_masks,
    recorded_kinds,
)
from dew.parity_stages import REDRAW, staged_units
from dew.training.compiled_program import lowering_platform

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

    @property
    def trace(self):
        """The trace of the arm the site runs (JAX's `jit(...).trace`), where it is a compiled program's:
        an arm of Dew's own programs has none."""
        return (self.plain if self.pairing.with_ynn else self.without).trace


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


def _platform_branch(equation, platform: str | None):
    """The branch of a platform's choice (`jax.lax.platform_dependent`) that a program lowered for
    `platform` keeps; None for any other equation, or for no platform."""
    chosen = equation.params.get("branches_platforms") if platform is not None else None
    return None if chosen is None else programs.branches(equation)[programs.platform_branch(chosen, platform)]


def equations(graph, *, platform: str | None = None):
    """Every equation of a jaxpr, its sub-jaxprs' (scan, cond, custom rules) in turn. With `platform`
    (a lowering platform's name: 'cpu', 'cuda', 'tpu'), a platform-dependent cond
    (`jax.lax.platform_dependent`) gives only the branch that platform runs, as lowering keeps it."""
    if hasattr(graph, "eqns"):
        for equation in graph.eqns:
            yield equation
            kept = _platform_branch(equation, platform)
            yield from equations(equation.params if kept is None else kept, platform=platform)
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
    # The types the program states, a float64 sum's traced under x64 (`dew.nn.precision.float64_sum`) among
    # them, read as they are.
    with jax.enable_x64(new_val=True):
        return [(equation.params["precision"],
                 jnp.result_type(*(variable.aval.dtype for variable in equation.invars)),
                 tuple(variable.aval.shape for variable in equation.invars))
                for equation in equations(graph, platform="cuda")
                if equation.primitive is jax.lax.dot_general_p]


def logits_operands(function, *arguments, width: int) -> set:
    """The dtypes every product `function` traces over `arguments` that contracts one axis `width` long (an
    attention's logits over its head width) multiplies in, nested programs included, as the program states
    them: the witness of a policy that widens the query and keys before their product (`wide_logits`), which
    no output shows on a CPU, whose compiler may keep a bf16 result wider than bf16 (rows 5429 and 5439).
    jax.nn's 16-bit logits are fp32 sums of 16-bit operands, which this tells apart. A float64 sum Dew's wide
    CPU arithmetic takes (`dew.nn.precision.float64_sum`) multiplies float64."""
    graph = jax.make_jaxpr(function)(*arguments)
    found = set()
    for equation in equations(graph):
        if equation.primitive is not jax.lax.dot_general_p:
            continue
        # Its contracted axes' lengths, which its two operands share.
        (contracted, _), _ = equation.params["dimension_numbers"]
        if [equation.invars[0].aval.shape[axis] for axis in contracted] == [width]:
            found.add(jnp.promote_types(*(variable.aval.dtype for variable in equation.invars)))
    return found


def assert_fp32_reduction_bound(dew, reference, magnitudes, terms: int) -> None:
    """Two fp32 reductions differ by at most 2 gamma_n sum(abs(products))
    (`dew._accuracy.reduction_bound`): `magnitudes` holds the elementwise sum of
    absolute products and `terms` counts the products and any bias term in each
    reduction."""
    error = np.abs(np.asarray(dew, np.float64) - np.asarray(reference, np.float64))
    bound = reduction_bound(magnitudes, terms)
    assert within(error, bound), (float(error.max()), float(bound.max()))


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
    if not _repeat_promised():
        return
    if _varies(fn) and jax.default_backend() != "cpu":
        why = next(layer.why for layer in _layers(fn) if isinstance(layer, _RunToRun))
        _logged(label, 0, f"{name} not repeated: declared to vary from run to run ({why})", [])
        return
    if (where := _same_bits(first, fn(*inputs))) is not None:
        raise UnpairedDraw(f"{label}: {name} differs at {where} run again on the same inputs, so its output "
                           f"depends on more than them")


class NarrowTruth(TypeError):
    """A check-1 site whose truth is not float64's (`_of_float64`): `leaf`, the path of the first leaf that
    is narrower than float64 or fed by a narrower value."""

    def __init__(self, leaf: str, message: str):
        self.leaf = leaf
        super().__init__(message)


def _below_float64(dtype) -> bool:
    """Whether `dtype` is a float's or a complex number's format narrower than float64's; a value of no
    dtype (None) is none."""
    return dtype is not None and jnp.issubdtype(dtype, jnp.inexact) and jnp.finfo(dtype).bits < 64


def _inexact(var) -> bool:
    """Whether `var` holds floats or complex numbers."""
    dtype = getattr(var.aval, "dtype", None)
    return dtype is not None and jnp.issubdtype(dtype, jnp.inexact)


def _read_through_floats(program, fed: Sequence[bool], lowering: str, found: dict) -> list[bool]:
    """Which of `program`'s inputs its `fed` outputs read through floats, and into `found` (its keys, in the
    order met) each result on the way that an operation rounds narrower than float64: what makes a float64
    truth's traced program a second fp32 run (`_of_float64`).

    A result is read through floats where a chain of float values runs from it to a fed float output. One
    read only by a comparison or as an integer (a dropout's float32 uniform against its rate, a code's
    index) is a decision of the program's, the same in every arm and in the truth, and no rounding of an
    output's value. The chain runs through the programs within an equation by what each is (`_within`): a
    call's, a branch's and a step's outputs are its equation's one for one, so only the fed ones are read
    there, a loop's carries to their fixed point.

    A narrow result is a format's own, and kept, where its operation rounds to a stated format (a cast,
    FORMATTED) or rounds nothing (`arm_arithmetic.rounds_nothing`: a move of entries, a selection, a
    negation): a truth that states a rounding (a quantized weight, a bf16 output) keeps it and what it
    selects of it, and one that computes in fp32 does not. A constant or a literal is its exact value
    whatever its format, as JAX promotes it where a float64 operation reads it: a parameter left float32
    under a float64 run is no rounding. A native kernel whose result an output reads is refused whatever
    its results' dtype: its body writes through references and runs in its own format, which this walk
    does not read (`programs.opaque`); a truth is the portable arithmetic."""
    jaxpr = getattr(program, "jaxpr", program)
    feeding = {var for var, read in zip(jaxpr.outvars, fed, strict=True)
               if read and programs.computed(var) and _inexact(var)}
    for equation in reversed(jaxpr.eqns):
        outs = [var in feeding for var in equation.outvars]
        if not any(outs):
            continue
        reads = _within(equation, outs, lowering, found)
        if reads is None:
            primitive = equation.primitive
            made = [var.aval for var, read in zip(equation.outvars, outs, strict=True)
                    if read and _below_float64(var.aval.dtype)]
            formats = (primitive is jax.lax.convert_element_type_p or primitive in FORMATTED
                       or rounds_nothing(primitive))
            where = equation.source_info.name_stack or "the truth itself"
            if programs.opaque(equation):
                found[f"a native kernel (`{primitive.name}`) in {where}, whose arithmetic is its own and is "
                      f"not read here"] = None
            elif made and not formats:
                found[f"`{primitive.name}` computed in {made[0].str_short()} in {where}"] = None
            # A barrier hands each operand on as the result at its place; any other operation's results
            # read its every operand.
            reads = outs if primitive is jax.lax.optimization_barrier_p else [True] * len(equation.invars)
        feeding.update(var for var, read in zip(equation.invars, reads, strict=True)
                       if read and programs.computed(var) and _inexact(var))
    return [var in feeding for var in jaxpr.invars]


def _within(equation, outs: list[bool], lowering: str, found: dict) -> list[bool] | None:
    """Which of `equation`'s operands its `outs` results read through floats, by the programs it runs within
    it (`_read_through_floats` of each); None for an operation that runs none. A call's program is its
    equation's, operand for operand and result for result (`programs.called`). A conditional's branches each
    take its operands after the index and give its results, of a platform's choice the one branch a
    program lowered for `lowering` keeps. A scan's step and a while loop's body carry results back into
    operands, so what they read is taken to its fixed point; a loop's predicate decides, and feeds no
    value. Any other equation that holds programs (a reduction's combining body) has them read whole."""
    def read(body, fed, into=found):
        return _read_through_floats(body, fed, lowering, into)

    def carried(body, fed, before: int, count: int):
        # A carry an output reads is an output the step before must feed: to the fixed point, then once more
        # for what is found on the way.
        while True:
            reads = read(body, fed, {})
            again = [old or new for old, new in zip(fed[:count], reads[before:before + count], strict=True)]
            if again == fed[:count]:
                return read(body, fed)
            fed = again + fed[count:]

    primitive, params = equation.primitive, equation.params
    if primitive is jax.lax.cond_p:
        kept = _platform_branch(equation, lowering)
        taken = programs.branches(equation) if kept is None else [kept]
        each = [read(branch, outs) for branch in taken]
        return [False, *(any(reads) for reads in zip(*each, strict=True))]
    if primitive is jax.lax.scan_p:
        return carried(programs.scan_body(equation), outs, *scan_groups(equation))
    if primitive is jax.lax.while_p:
        _, body = programs.loop_programs(equation)
        return [False] * params["cond_nconsts"] + carried(body, outs, params["body_nconsts"], len(outs))
    body = programs.called(equation)
    if body is not None:
        return read(body, outs)
    within = programs.bodies(equation)
    for body in within:
        read(body, [True] * len(getattr(body, "jaxpr", body).outvars))
    return [True] * len(equation.invars) if within else None


def _of_float64(label: str, truth_fn, inputs, truth) -> None:
    """Raise unless a site's truth is float64's: `truth`, its value on the site's first draw, holds no float
    leaf narrower than float64, and the program `truth_fn`'s own calls trace on those `inputs` widened
    computes no narrower result that a float output reads through floats (`_read_through_floats`, by
    JAX's primitives' facts), a stated rounding apart. A model of a fixed float32 dtype
    run on widened inputs computes in fp32, or rounds its answer to it: a second fp32 run, which equals
    the reference bit for bit and judges Dew against a truth no better than itself (a forked call's cells;
    row 5430). A leaf of no
    entries holds no value and is not read. A truth the gate runs as given (NumPy, torch, a recording) or
    that runs programs of its own traces none here, and is held by its leaves alone. The gate cannot tell a
    cast the arms share (a bf16 output's rounding) from one the truth alone takes (a round trip through
    float32 with no operation between): both are kept."""
    paths, leaves = _leaf_paths(truth), jax.tree.leaves(truth)
    held = [bool(np.size(leaf)) for leaf in leaves]
    for path, leaf, read in zip(paths, leaves, held, strict=True):
        # The leaf's own dtype: JAX's reading of a host float64 outside x64 would narrow it.
        dtype = np.asarray(leaf).dtype if not hasattr(leaf, "dtype") else leaf.dtype
        if read and _below_float64(dtype):
            raise NarrowTruth(path, f"{label}: the float64 truth's leaf {path or 'itself'} is {dtype}: a "
                              f"truth computes in float64 (a float64 clone of the model, `float64_held` "
                              f"variables)")
    windowed = any(isinstance(layer, _Windowed) for layer in _layers(truth_fn))
    trace = None if _given(truth_fn) else getattr(next(
        layer for layer in _layers(truth_fn) if not isinstance(layer, _RunToRun | _Windowed)), "trace", None)
    if trace is None:
        return
    # A window loop's truth is traced on a window, as its loop calls it.
    wide = widened(jax.tree.map(lambda leaf: leaf[0], inputs) if windowed else inputs)
    # The branch of a platform's choice that the truth's run keeps: on the device its evaluation runs on.
    with jax.enable_x64(new_val=True), evaluated_on(jax.tree.leaves(wide)) as device:
        program = lowered(trace(*wide).jaxpr).jaxpr
    lowering = lowering_platform(device)
    count = len(program.outvars)

    def narrow(fed: list[bool]) -> list[str]:
        """The first few narrow results the `fed` outputs read."""
        found: dict = {}
        _read_through_floats(program, fed, lowering, found)
        return list(found)[:3]

    if not narrow(held):
        return
    # The first leaf a narrow result feeds, and what feeds it: each output's own program, read on the refusal.
    at, found = next((index, "; ".join(found)) for index in range(count)
                     if held[index] and (found := narrow([index == other for other in range(count)])))
    raise NarrowTruth(paths[at], f"{label}: the float64 truth's leaf {paths[at] or 'itself'} is fed by "
                      f"{found}: no float64 truth, which computes in float64 (a float64 clone of the "
                      f"model; a stated rounding, a cast, is kept)")


def _over_orders(first: list[tuple], later: list[tuple]) -> list[tuple]:
    """An arm's evidence on a draw (`_evidence`) over two of its orders: each leaf's errors and rows of
    float64 zero the first order's, its worst rows the worse of the two, each worst row's share
    (`RecordedLeaf.share`) that of the order it is, its values finite at both."""
    return [(one[0], one[1], float(np.fmax(one[2], other[2])), float(np.fmax(one[3], other[3])), one[4],
             one[5] and other[5], tuple(other[6][arm] if other[2 + arm] > one[2 + arm] else one[6][arm]
                                        for arm in (0, 1)))
            for one, other in zip(first, later, strict=True)]


def _conditioning(reference_fn, truth_fn, draw, seeds, ran=None, scale=None) -> Conditioning:
    """Check 1's conditioning of `reference_fn` (dew.parity's `Conditioning`): its rounding scale at a draw as
    its arm runs (`summed_as_run`, a window loop's by window), or `scale(inputs, seed)` where its program is
    not one here (`assert_as_exact`'s `conditioning_scale`), its heavy elements over `seeds` as `ran`, its
    compiled arm where given, runs it; the reference and its truth unrewritten, as their arms run."""
    from rounding_bound import rounding_scale
    scaled = reference_fn.fn if isinstance(reference_fn, _Windowed) else reference_fn

    def scale_of(inputs, seed):
        if scale is not None:
            return scale(inputs, seed)
        with summed_as_run():
            if not isinstance(reference_fn, _Windowed):
                return rounding_scale(scaled, *inputs, seed=seed)[1]
            each = [rounding_scale(scaled, *jax.tree.map(lambda leaf, at=at: leaf[at], inputs), seed=seed)[1]
                    for at in range(_window_count(inputs))]
            return jax.tree.map(lambda *scale: None if scale[0] is None else np.stack(scale), *each,
                                is_leaf=lambda leaf: leaf is None)

    # Its heavy elements over `seeds`' draws, each run as its arm runs (`summed_as_run`) beside its float64
    # truth, `truth_fn`'s or its own, classed alike in every process that runs them.
    ran_fn, true_fn = unrewritten(ran or reference_fn), unrewritten(truth_fn)

    def ran_and_true(seed):
        inputs = draw(seed)
        with summed_as_run():
            ran_here = jax.device_get(ran_fn(*inputs))
        with jax.enable_x64(new_val=True):
            return ran_here, jax.device_get((true_fn or ran_fn)(*widened(inputs)))

    return Conditioning(scale_of, heavy_masks(map(ran_and_true, seeds)))


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
    widened inputs. With `conditioning` (dew.parity's `Conditioning`), every arm's leaves are read as
    residuals in units of the reference's rounding at its truth's inputs (`Weighed.weighed`)."""
    if orders is not None and any(getattr(fn, "recorded", False) for fn in (reference_fn, *also)):
        raise ValueError("a recorded reference holds its recorded draws alone, never another order of them")
    arms = [reference_fn] + ([] if library_fn is None else [library_fn]) + list(also)
    evidence: list[list] = [[] for _ in arms]
    names, widths = None, None
    # Each arm's per-example ratios of the draw being read, and of every draw read (`examples`).
    stacked: list = [None for _ in arms]
    stacked_rows: list = [None for _ in arms]
    per_example: list[list] = [[] for _ in arms]
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
        if not at:
            _of_float64(label, truth_fn or reference_fn, inputs, truth)
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
        for at_arm, row in enumerate(rows):
            evidence[at_arm].append(row)
            per_example[at_arm].append(stacked[at_arm])
            per_example_rows[at_arm].append(stacked_rows[at_arm])
    drawn = [_drawn_from(names, widths, evidence[at_arm], per_example[at_arm] if examples else None,
                        per_example_rows[at_arm] if examples else None) for at_arm in range(len(arms))]
    first = drawn[0] if library_fn is None else drawn[0].against_larger(drawn[1], library_leaves)
    return [first, *drawn[len(arms) - len(also):]] if also else first


STAGES_LOG = STAGES
"""The environment variable (`check1_wire.STAGES`) naming a file each check-1 judgement appends its stage, its
verdict, its leaves of narrow rows (NARROW_FACTORS) and whether its reference equalled the truth on every
first-stage seed (`exact`: a site that measures no rounding, or a truth that is its reference run again) to,
one JSON line (`_logged`): what a train reads every site's stage-two rate from, every attempt counted, and
which sites' local errors rest on check 2."""


def _logged(label: str, stage: int, verdict: str, narrow_leaves: list[str], ynn: str | None = None,
            exact: bool | None = None) -> None:
    import json

    path = os.environ.get(STAGES_LOG)
    if path:
        with open(path, "a", encoding="utf-8") as log:
            log.write(json.dumps({"site": os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" (", 1)[0],
                                  "label": label, "stage": stage, "verdict": verdict,
                                  "narrow_leaves": narrow_leaves, "ynn": ynn, "exact": exact}) + "\n")


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
    _enough_seeds(seeds, label)
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
                    conditioning: int | None = None, conditioning_scale: Callable | None = None) -> None:
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
    the site weighed where it stays undecided with its budget spent (dew.parity's conditioning, ledger 1714),
    each verdict logged under the check that decided it, the weighed one's ", conditioned". Its scale is
    the reference program's rounding; a reference whose program is not one here (a recorded one) gives its
    own (`conditioning_scale(inputs, seed)`), its upstream computation's, read from its float64 twin with
    every operation tapped (`_dew_torch.rounding.rounding_scale`) and recorded beside its outputs,
    never Dew's arm's, whose own forms would weigh its own amplified errors down.

    This is an engineering gate, deterministic and reproducible, not a guarantee against every error
    law: "never wrong" is check 2's, each element within its operation's certified bound
    (tests/certified.py)."""
    if conditioning is not None:
        _deferred(dew_fn, reference_fn, draw, label, seeds=seeds, conditioning=conditioning,
                  truth_fn=truth_fn, orders=orders, library_fn=library_fn, library_row=library_row,
                  library_leaves=library_leaves, also=also, terms=terms, examples=examples,
                  scale=conditioning_scale)
        return
    # A map task draws no seed past the site's stages (`REDRAW`): a sharded test's cases each declare their
    # own, the shards cut at the most of them, and a recording holds no more.
    _site_judged(dew_fn, reference_fn, draw, label, seeds=seeds, truth_fn=truth_fn, orders=orders,
                 library_fn=library_fn, library_row=library_row, library_leaves=library_leaves, also=also,
                 terms=terms, examples=examples, weighing=None, upto=staged_units(seeds))


def _deferred(dew_fn, reference_fn, draw, label: str, *, seeds: int, conditioning: int, truth_fn, orders,
              library_fn, library_row, library_leaves, also, terms, examples: bool, scale=None) -> None:
    """`assert_as_exact`'s site with `conditioning` (dew.parity's `Conditioning`): the unweighted check over
    its own `seeds`, whole, and where it stays undecided the weighed check over `conditioning` seeds, its
    heavy elements classed over the draws it takes, alike in every process that runs it. A sharded site's map
    tasks draw each check's units apart, the weighed check's past the unweighted one's in the plan's units
    (tools/armada/ci.py `draws_of`), and its judge pools the weighed check's where it defers."""
    plain, span = staged_units(seeds), staged_units(conditioning)

    def weighed():
        _site_judged(dew_fn, reference_fn, draw, f"{label}, conditioned", seeds=conditioning,
                     truth_fn=truth_fn, orders=orders, library_fn=library_fn, library_row=library_row,
                     library_leaves=library_leaves, also=also, terms=False, examples=examples,
                     weighing=functools.partial(_conditioning, draw=draw, seeds=range(span), scale=scale),
                     upto=span,
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
    """`assert_as_exact`'s judgment of one site under `label`, weighed where `weighing` is given (dew.parity's
    `Conditioning`: built of the site's compiled reference and truth where it draws, never in its judge), a
    map task recording its range's draws up to `upto` where given, its classes' digest beside them."""
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
    _verdicts(drawn_for, seeds, [label, *(f"{label}, against {name}" for name in also)], narrow_leaves, arms,
              library=library_fn is not None)


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


def _verdicts(drawn_for, seeds: int, labels: list[str], narrow_leaves: list[str], arms, *,
              library: bool = False) -> None:
    """Each reference's check 1 over its draws (`_staged`, `drawn_for(index)` the draws against reference
    `index`, judged under `labels[index]`), each logged (STAGES_LOG) with whether its reference equalled
    the truth on every first-stage seed, unknown (None) for the first where a `library` arm's error stands
    in for its reference's on some leaves; a GateFailure where any leaf is over, else an Inconclusive where
    any is left unresolved."""
    overs, unresolveds = [], []
    for index, named in enumerate(labels):
        over, unresolved, stage = _staged(drawn_for(index), seeds, named)
        exact = None if library and not index else not np.any(drawn_for(index)(0, seeds).theirs)
        _logged(named, stage, _verdict(over, unresolved), narrow_leaves, _mode_of(arms), exact)
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
    _enough_seeds(seeds, label)
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


BELOW: list[dict] = []
"""What each FP8 site this test judged measured (`assert_fp8_below`), the JUnit property `fp8_below`
(tests/conftest.py) of a test that passes as of one that fails: a leaf, the arm it is held below, Dew's
pooled ratio to that arm and its interval, Dew's ratio to the dequantized reference (the measured miss of
check 1's parity, which goes to the owner) and the seeds they were measured over."""


def _fp8_codes(operand) -> bool:
    """Whether `operand` holds an FP8 format's codes: floats of eight bits with a significand (E4M3, E5M2
    and their kin), which a block scale's E8M0 has none of, as JAX's dtypes read them."""
    dtype = jnp.result_type(operand)
    return jnp.issubdtype(dtype, jnp.floating) and jnp.finfo(dtype).bits == 8 and jnp.finfo(dtype).nmant > 0


def assert_fp8_below(dew_fn, named_fn, draw, label: str, *, seeds: int, named: str, codes, reference_fn,
                     truth_fn) -> None:
    """The contract of an FP8 form whose scales are a whole operand's, where no FP8 kernel reaches check
    1's parity (docs/design/testing.md, "FP8 products": a tensor core's running sum within a step is
    narrower than fp32, and the dequantized reference's fp32 sums of FP8 products are exact): `dew_fn`'s
    error from the float64 truth is below `named_fn`'s, the arm the spec names (`named`, cuBLASLt's FP8
    GEMM, which sums the whole depth narrow), on the same draws. A block-scaled form sums a block as
    cuBLASLt's GEMM of that block does and is held to parity with it, check 1's own (`assert_as_exact`
    with that arm as its reference), not this.

    It is check 1 against the named arm (`_staged`: its draws, its two pre-declared stages, its rows and
    values) with one rule changed, the pooled ratio's: a leaf holds where its ratio's interval lies wholly
    under one (`dew.parity._below_one`), where check 1 allows PARITY. A leaf the named arm holds exactly
    has nothing to be below: Dew off float64 there fails as under check 1, and Dew exact there fails too,
    as a form at parity holds check 1 itself. `truth_fn` is the float64 product of the operands as
    quantized, a row that cancels sized by its terms as check 1 sizes it (`_site_terms`), and
    `reference_fn` the dequantized product with fp32 sums, check 1's reference for an FP8 product: it runs
    beside the two on the same draws and Dew's ratio to it is recorded with the verdict's (`BELOW`), never
    judged.

    Only an FP8 product's site takes this contract: `codes(*inputs)` gives the product's multiplicands of a
    draw's inputs, and each must hold an FP8 format's codes (`_fp8_codes`: not a narrower format's, nor an
    FP8 scale beside them), else the site is refused and holds check 1."""
    _enough_seeds(seeds, label)
    multiplied = tuple(codes(*draw(0)))
    if not multiplied or not all(_fp8_codes(operand) for operand in multiplied):
        raise TypeError(f"{label}: the product's multiplicands are "
                        f"{[str(jnp.result_type(operand)) for operand in multiplied]}, not FP8 codes; an "
                        f"error below a named arm's is the FP8 products' contract alone "
                        f"(docs/design/testing.md), and this site holds check 1 (`assert_as_exact`)")
    _refuse_numpy_arms(draw, dew_fn, named_fn, reference_fn)
    dew_fn, named_fn, (reference_fn,) = _arms_with(dew_fn, named_fn, [reference_fn], label)
    truth_program = _truth_program(truth_fn)
    terms_fn = _site_terms(None, None, draw, truth_fn, truth_program)
    arms = [dew_fn, named_fn, reference_fn]
    # Each range's draws against the named arm and against the reference, Dew and the truth run once.
    ranges: dict[tuple[int, int], list[_Draws]] = {}

    def drawn(start, stop):
        ranges[start, stop] = _drawn(dew_fn, named_fn, draw, range(start, stop), truth_program,
                                     also=[reference_fn], terms_fn=terms_fn, label=label)
        return ranges[start, stop][0]

    held = f"{label}, against {named}"
    over, unresolved, stage = _staged(drawn, seeds, held, ratio_rule=_below_one)
    against, parity = (functools.reduce(_Draws.joined, each) for each in zip(*ranges.values(), strict=True))
    for index, name in enumerate(against.names):
        ratio, low, high = _judged(against, index)
        BELOW.append({"leaf": held + name, "named": named, "seeds": len(against.mine), "ratio": ratio,
                      "interval": [low, high],
                      "reference_ratio": _pooled(parity.mine[:, index], parity.theirs[:, index])})
    _logged(held, stage, _verdict(over, unresolved), _narrow(against), _mode_of(arms))
    if over:
        raise GateFailure(_with_mode(over + unresolved, arms))
    if unresolved:
        raise Inconclusive(_with_mode(unresolved, arms))


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
              conditioning: bool = False, conditioning_scale=None) -> int:
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
    weighing = (_conditioning(reference_fn, truth_fn, draw, range(staged_units(max(start, FEWEST_SEEDS))),
                              scale=conditioning_scale) if conditioning else None)
    dew_fn, reference_fn = _arms(dew_fn if paired else reference_fn, reference_fn)
    dew_fn = dew_fn if paired else None
    truth_fn = _truth_program(truth_fn)
    first = 0 if dew_fn is None else SIZING_FROM

    def drawn(start_at: int, stop: int):
        return _drawn(dew_fn or reference_fn, reference_fn, draw, range(first + start_at, first + stop),
                      truth_fn, orders=orders, label=label, examples=examples and dew_fn is not None,
                      conditioning=weighing)

    return _sized_count(drawn, label, start=start, most=most, paired=dew_fn is not None)


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


def _of_leaves(stored, site: str, leaves) -> dict:
    """`stored` with the entries of `leaves` alone of those kept one a leaf (`dew.parity.RECORDED_KINDS`),
    verdict and pilot alike, and every entry the whole recording shares; a leaf the recording lacks is
    refused."""
    wanted, found, kept = set(leaves), set(), {}
    for key, value in stored.items():
        kind, _, name = key.removeprefix(f"{site}.").removeprefix("pilot.").partition(".")
        if key.startswith(f"{site}.") and kind in RECORDED_KINDS and name:
            if name not in wanted:
                continue
            found.add(name)
        kept[key] = value
    if wanted - found:
        raise ValueError(f"{site}: the recording holds no leaves {sorted(wanted - found)}")
    return kept


def recorded_weighing(stored, site: str) -> dict:
    """The weighed check a recorded `site` declares (`assert_as_exact`'s `conditioning` and
    `conditioning_scale`): its declared seeds, and each draw's rounding scale as recorded beside the
    reference's outputs (`<site>.scale.<leaf>`, tools/recorded_references.py's `record_case(scale=...)`),
    read at the draw's index, the first of the indexed inputs (`indexed_recording`)."""
    scales = {name.removeprefix(f"{site}.scale."): value for name, value in stored.items()
              if name.startswith(f"{site}.scale.")}

    def scale(inputs, seed):
        return {name: value[int(inputs[0])] for name, value in scales.items()}

    return {"conditioning": int(stored[f"{site}.conditioning"]), "conditioning_scale": scale}


def indexed_recording(stored, site: str, seeds: int, *, inputs_of=None, truth_fn=None, leaves=None):
    """`assert_as_exact`'s draw, reference and truth for a reference recorded over seeds, where it cannot
    run in the test (a CUDA-only library, a fetched source): `stored` holds `<site>.reference.<name>` and
    `<site>.truth.<name>`, and `<site>.input.<name>` unless `inputs_of(seed)` draws a seed's inputs again
    (a seeded NumPy or torch draw the generator made them by), and `<site>.replayed.<name>`, the leaves
    the reference made that Dew replays (`record_case`), which a seed's inputs carry, every array stacked
    over the seeds on its first axis, the truth in float64, and `<site>.redraw`, check 1's REDRAW it was
    recorded for. Inputs drawn again are checked against the recording's `<site>.input_digest`
    (`input_digest`), so a generator that changed since is named, not judged. A draw is `(index, inputs)`,
    the index an integer `widened` keeps; the reference and the truth read the recorded outputs at it, and
    Dew's function takes the same two arguments and ignores the index. The recording covers check 1's
    redraw, (1 + REDRAW) times the declared seeds, so a numerical failure stays a finding rather than
    running out of recorded seeds.

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

    inputs, reference, truth, replayed = tree("input"), tree("reference"), tree("truth"), tree("replayed")
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
            drawn = {name: value[seed] for name, value in inputs.items()}
        else:
            drawn = drawn_on_the_host(inputs_of, seed)
            if input_digest(drawn) != digests[seed]:
                raise AssertionError(f"{site}: seed {seed}'s inputs drawn again are not the recorded ones "
                                     f"(their digest differs); the generator's draw changed since the "
                                     f"recording")
        return np.int32(seed), {**drawn, **{name: value[seed] for name, value in replayed.items()}}

    def lookup(values):
        # A draw's recorded leaves are read from the host at the draw's index, in the dtypes the caller's
        # context holds them in. A recording is never a program's constants, which exhaust a host's memory
        # where a weighed recording keeps thousands of draws whole. An arm that wraps this one and is traced
        # reads at run time (`jax.pure_callback`); JAX keeps a callback as long as its program, so the
        # gate's own read, its index in hand, goes through none.
        def read(at):
            return {name: value[int(at)] for name, value in values.items()}

        def recorded(index, inputs):
            if not isinstance(index, jax.core.Tracer):
                return {name: jnp.asarray(value) for name, value in read(index).items()}
            kept = {name: jax.ShapeDtypeStruct(value.shape[1:], jax.dtypes.canonicalize_dtype(value.dtype))
                    for name, value in values.items()}
            return jax.pure_callback(
                lambda at: {name: np.asarray(value, kept[name].dtype) for name, value in read(at).items()},
                kept, index, vmap_method="sequential")
        recorded.recorded = True
        return recorded

    return draw, lookup(reference), lookup(truth)


def _summarized(stored, site: str, seeds: int, inputs_of, truth_fn):
    """`indexed_recording` of a recording of the reference's evidence alone."""
    def tree(kind):
        prefix = f"{site}.{kind}."
        return {name.removeprefix(prefix): np.asarray(value) for name, value in stored.items()
                if name.startswith(prefix)}

    # A summary's evidence, each kind as the recording's layout holds it (`dew.parity.RECORDED_KINDS`).
    kinds = recorded_kinds("seed", "span", "offsets")
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
            examples = {field: evidence[field][name][row] for field in ("example_mse", "example_worst")
                        if name in evidence[field]}
            return RecordedLeaf(float(evidence["mse"][name][row]), float(evidence["worst"][name][row]),
                                evidence["zero_rows"][name][start:stop],
                                evidence["zero_errors"][name][start:stop],
                                finite=bool(evidence["finite"][name][row]), dtype=str(dtypes[name]),
                                **{field: None if np.isnan(value).all() else value
                                   for field, value in examples.items()})

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
        primitive, params = equation.primitive, equation.params
        shape = equation.invars[0].aval.shape if equation.invars else ()
        if primitive is jax.lax.dot_general_p:
            (contracting, _), _ = params["dimension_numbers"]
            return int(np.prod([shape[axis] for axis in contracting], dtype=np.int64))
        if primitive is jax.lax.conv_general_dilated_p:
            kernel = equation.invars[1].aval.shape
            spec = params["dimension_numbers"].rhs_spec
            return int(np.prod([kernel[axis] for axis in spec[1:]], dtype=np.int64))
        if programs.reduction(primitive):
            return int(np.prod([shape[axis] for axis in params["axes"]], dtype=np.int64))
        return 1

    def walk(jaxpr, times: int) -> int:
        total = 0
        for equation in jaxpr.eqns:
            total += times * length(equation)
            steps = equation.params["length"] if equation.primitive is jax.lax.scan_p else 1
            total += sum(walk(body, times * steps) for body in programs.bodies(equation))
        return total

    return walk(closed_jaxpr.jaxpr, 1)


def assert_rounds_where_the_reference_does(dew, reference, truth, label: str) -> None:
    """Dew within a quarter of the reference's RMS rounding error of the
    reference's own output."""
    apart, theirs = distance(dew, reference), distance(reference, truth)
    assert within(apart, theirs / 4), (
        f"{label}: dew is {apart:.3e} from the reference (rms), whose own error is {theirs:.3e} "
        f"(ratio {apart / theirs:.2f}, allowed 0.25)")
