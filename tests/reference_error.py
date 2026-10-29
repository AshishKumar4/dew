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
from collections.abc import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from dew.accuracy import distance, reduction_bound, within

FACTOR = 2.0

ROW_FACTOR = 4.0
"""How far Dew's worst row may sit from float64 against the reference's worst
row (`rows_over`; check 1's at rows of one or two values is NARROW_FACTORS').
An RMS over every entry averages a few wrong rows into a
large tensor, so each comparison also holds Dew's worst row to the
reference's. The worst of many rows is an extreme value, not an average,
and parts further by chance: over the suite's 3,660 comparisons whose rows
differ from float64 at all (99 files), Dew's worst row over the
reference's sat at a median of 0.21, p99 1.55 and at most 2.42 (Mixtral's
exported logits). A row that differs rather than rounds moves further: the
suite's deliberate failures, a GDN chunk's plain fp32 prefix and a MiMo tower
with its sinks zeroed, put their worst rows at 7.1 and 3,419 times the
reference's."""

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


def widened(tree):
    """`tree` as float64 host arrays, its integer leaves as they are: the
    inputs of a float64 truth."""
    return jax.tree.map(lambda leaf: np.asarray(
        leaf, np.float64 if _floating(leaf.dtype) else leaf.dtype), tree)


def _stack_draws(trees):
    """A draw's leaves of each window stacked on a new leading axis. A JAX array or Python scalar stacks
    as JAX stacks it, its weak type kept, and a host array as NumPy stacks it, its dtype kept."""
    def stack(*leaves):
        if all(isinstance(leaf, jax.Array | bool | int | float | complex) for leaf in leaves):
            return jnp.stack([jnp.asarray(leaf) for leaf in leaves])
        return np.stack([np.asarray(leaf) for leaf in leaves])
    return jax.tree.map(stack, *trees)


def _stack_host(trees):
    """An arm's outputs of each window stacked on a new leading axis as host arrays of their own dtypes."""
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
        return _stack_host([self.fn(*jax.tree.map(lambda leaf, index=index: leaf[index], arguments))
                            for index in range(_window_count(arguments))])


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
SSD kernel's host check 1 without YNNPACK"). The empty list, not the documented `-individual_dot`,
because compiler options take no such spelling and that one leaves a scan's reductions fused."""


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
        jit_options["compiler_options"] = {**WITHOUT_YNN, **(jit_options.get("compiler_options") or {})}
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
    """Refuse an arm the gate would compile that computes in NumPy: called once as it is given on a
    draw's inputs (a window's, for a window loop's arm), a JAX arm returns JAX arrays, and an arm that
    returns NumPy arrays is a NumPy reference, which compiled would compute in JAX, in float32 where
    its draw is float64, by other sums. It passes `as_given`; a Dew arm that only converts its
    outputs returns them unconverted instead."""
    plain = [fn for fn in dict.fromkeys(arms) if _compiles(fn)]
    if not plain:
        return
    drawn = draw(0)
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


def _compiled(fn, compile):
    """`fn` as the gate runs it: as given (`_given`), as it is where it is a `without_ynn` program
    already (another jit around it would nest its compiler options), or compiled by `compile`, a
    window loop's arm within it (`over_windows`)."""
    if _given(fn) or getattr(fn, "without_ynn", False):
        return fn
    if getattr(fn, "dew_programs", False):
        return fn if compile is jax.jit else _with_ynn_off(fn)
    if isinstance(fn, _Windowed):
        return _Windowed(_compiled(fn.fn, compile))
    return compile(fn)


class _Truth:
    """A check's float64 truth as the gate runs it (`_truth_program`): its first call runs `fn` as it is
    given and decides the rest, compiled where that call returned JAX values, as given where it
    returned host values (a NumPy truth). A truth that returns JAX values is traced at that first call,
    so one that does host work on its inputs (a widening or a conversion on the host) fails there, at
    its first seed, naming itself; it passes `as_given`."""

    def __init__(self, fn):
        self.fn, self.run = fn, None

    def __call__(self, *inputs):
        if self.run is not None:
            return self.run(*inputs)
        first = self.fn(*inputs)
        self.run = self.fn
        if all(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(first)):
            self.run = jax.jit(self.fn)
            self.run.trace(*inputs)
        return first


def _truth_program(fn):
    """`fn`, a check's float64 truth, as the gate runs it: compiled, once a signature, so the truth runs
    as one float64 program on the device's FP64 units rather than one eager dispatch an equation, every
    primitive at its own float64 semantics. A truth the gate runs as given (`as_given`: torch, a
    recording, a host loop), one compiled already (a jit, `dew_programs`), or none, as it is; a window
    loop's by its own; a NumPy truth (its first call returns host values) as given after that call. A
    JAX truth with host work inside fails at its trace, loudly, and passes `as_given`."""
    if fn is None or _given(fn) or getattr(fn, "dew_programs", False) or hasattr(fn, "lower"):
        return fn
    if isinstance(fn, _Windowed):
        return _Windowed(_truth_program(fn.fn))
    return _Truth(fn)


def _with_ynn_off(fn):
    """`fn` run with Dew's compile seam holding WITHOUT_YNN open on the CPU lane (`dew_programs`)."""
    from dew.nn.kernels.options import compiled_with

    @functools.wraps(fn)
    def run(*inputs, **options):
        with compiled_with(WITHOUT_YNN if jax.default_backend() == "cpu" else {}):
            return fn(*inputs, **options)
    return run


def _arms(dew_fn, reference_fn):
    """The two arms of a check as the gate runs them (docs/design/testing.md, "the numerical contract"):
    where both are Dew paths, each without YNNPACK's fusions (`without_ynn`), whose microkernels the
    host's instruction set picks at run time, so the two run the same kind of code on any host: a
    plain function the gate compiles so, an arm that runs Dew's own programs (`dew_programs`) with
    them compiled so through Dew's seam; where one runs as it is given (`as_given`: torch, NumPy, a
    recording, a host loop over programs of its own), the other as Dew's users run it, with the
    fusions (`jax.jit`). A site passes plain functions; the gate compiles them, so no site can compile
    one arm otherwise than its other. One function as both arms compiles once."""
    compile = jax.jit if _given(dew_fn) or _given(reference_fn) else without_ynn
    reference = _compiled(reference_fn, compile)
    return (reference if dew_fn is reference_fn else _compiled(dew_fn, compile)), reference


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


def assert_fp32_reduction_bound(dew, reference, magnitudes, terms: int) -> None:
    """Two fp32 reductions differ by at most 2 gamma_n sum(abs(products))
    (`dew.accuracy.reduction_bound`): `magnitudes` holds the elementwise sum of
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


def rows_over(dew, reference, truth, terms=None, floor: float = 0.0) -> list[tuple[str, float, str]]:
    """Where Dew's rows are further from float64 than ROW_FACTOR times the
    reference's (`row_errors`): its worst row over its own size against the
    reference's worst, at least `floor` (an order floor in the rows' units),
    and each row of size zero, a sum that cancels exactly before anything its
    terms see (a query that keeps one key), against the reference's distance
    on that same row, which is all the evidence such a row has: where the
    reference is exactly the float64 zero, Dew must be too. Each as `(rule,
    measure, message)`, a `Finding`'s rule, measure and message (RULES)."""
    sizes, mine = row_errors(dew, truth, terms)
    _, theirs = row_errors(reference, truth, terms)
    sized = sizes > 0
    over = []
    if sized.any():
        # Compared by their logs, so a row's error over its size orders where the quotient overflows.
        row = float(np.max(_log_over(mine[sized], sizes[sized])))
        their_row = float(np.max(_log_over(theirs[sized], sizes[sized])))
        allowed = max(their_row, float(_log_over(floor, 1.0)))
        if not row <= math.log(ROW_FACTOR) + allowed:
            over.append(("row", row - allowed, f"the worst row is {_shown(row)} from float64 over its size, "
                         f"the reference's {_shown(their_row)}, allowed "
                         f"{_shown(math.log(ROW_FACTOR) + allowed)}"))
    zero = ~sized & ~(mine <= ROW_FACTOR * theirs)
    if zero.any():
        at = int(np.flatnonzero(zero)[0])
        # The measure is the worst of them, the first the example.
        over.append(("zero rows", log_ratio(mine[zero], theirs[zero]),
                     f"{int(zero.sum())} rows of float64 zero are further from it than the reference's own, "
                     f"row {at} by {mine[at]:.3e} against {theirs[at]:.3e}"))
    return over


def assert_as_exact_as_the_reference(dew, reference, truth, label: str, *, terms=None) -> None:
    """Dew within FACTOR times the reference's RMS distance from float64, and
    its rows within ROW_FACTOR times the reference's (`rows_over`), each
    sized by its float64 `terms` where they are given (`float64_terms`)."""
    mine, theirs = distance(dew, truth), distance(reference, truth)
    if not (math.isfinite(mine) and math.isfinite(theirs)):
        raise GateFailure([Finding(label, "not finite", 1, f"{label}: a value of dew, the reference or "
                                   f"float64 is not finite (dew {mine:.3e} and the reference {theirs:.3e} "
                                   f"from float64)")])
    if not theirs > 0:
        # No measure: Dew's own distance from float64 is what a change can make worse, and none at all
        # is a site that cannot be judged (`no measure`), never a pass.
        if mine > 0:
            raise GateFailure([Finding(label, "exact reference", mine, f"{label}: the reference equals "
                                       f"float64 and dew is {mine:.3e} from it (rms)")])
        raise GateFailure([Finding(label, "no measure", 0.0, f"{label}: the reference and dew both equal "
                                   f"float64, so the site measures nothing")])
    found = []
    if not within(mine, FACTOR * theirs):
        found.append(("pooled", log_ratio(mine, theirs), f"dew is {mine:.3e} from float64 (rms), the "
                      f"reference {theirs:.3e} (ratio {mine / theirs:.2f}, allowed {FACTOR})"))
    # Every rule's findings at once, so a change that eases one rule and breaks another shows both.
    found += rows_over(dew, reference, truth, terms)
    if found:
        raise GateFailure([Finding(label, rule, measure, f"{label}: {message}")
                           for rule, measure, message in found],
                          f"{label}: " + "; ".join(message for _, _, message in found))



PARITY = 1.5
"""The gate's on-par factor: each leaf's RMS error from float64 within 1.5 times the
reference's, on the same inputs, with no floor of any kind. Check 1 (`assert_as_exact`) holds it now;
the one-draw rule's FACTOR joins it once every site that rule would then fail has moved to check 1."""

BOOTSTRAP = 2000
"""Resamples of a site's seeds for the pooled ratio's interval (`_interval`), drawn by one fixed key so
an assertion decides alike on every run."""
CONFIDENCE = 0.999
"""Check 1's stated confidence in its ratio's decisions: a pass certifies a leaf's RMS ratio no more
than PARITY², and an over by its ratio that it is above one, each but for a nominal 5e-4 (LOWER_TAIL,
UPPER_TAIL). A leaf failed by a row, a row of float64 zero or a value not finite is failed by that
rule, which certifies no ratio."""
LOWER_TAIL = 5e-4
"""The lower tail of a leaf's bootstrap interval, the over side's whole 5e-4, spent at stage 2 alone, where
an over is decided (`_staged`). Nominal: the rates hold as far as the percentile bootstrap's endpoint
coverage does, and 2,000 resamples interpolate an endpoint this far out."""
UPPER_TAIL = 2.5e-4
"""The upper tail at each of check 1's two stages: the pass side's 5e-4 spent equally over them, a pass
possible at either (nominal, as LOWER_TAIL)."""
FEWEST_SEEDS = 8
"""The fewest seeds check 1 judges by, an engineering minimum and no statistical threshold: over very few
seeds a bootstrap has so few distinct resamples that its interval can sit wholly to one side of PARITY by
the draws' luck, so fewer is Inconclusive whatever their ratio reads; above it the measured interval
decides. `seeds_for` never declares fewer."""
REDRAW = 4
"""A failing site's pooled ratio is drawn again over REDRAW times its seeds before it fails."""


@dataclasses.dataclass(frozen=True)
class Finding:
    """One finding of the gate as data, beside its message: the leaf it names (the site's label and the
    leaf's path), the rule it fails (RULES), and its measure, finite and larger worse: the log of Dew's
    error over the reference's for a ratio's rule (`log_ratio`, `log_sum`), the seeds a value is not
    finite on, Dew's own distance from float64 where the reference is exactly float64, and for too few
    seeds the fewest over the given. What a train matches a candidate's failures to its parent's by:
    the same leaf and rule, a measure no larger."""
    leaf: str
    rule: str
    measure: float
    message: str

    def __str__(self) -> str:
        return self.message


RULES = frozenset({"pooled", "row", "zero rows", "not finite", "exact reference", "no measure",
                   "inconclusive"})
"""The rules a finding names: the pooled RMS ratio over its factor, the worst row over its factor, a row
of float64 zero further from it than the reference's, a value not finite, a reference exactly float64
where Dew is not (measured by Dew's own distance from it), a site where both are (nothing to measure,
measured 0), and a leaf its seeds leave undecided."""


LOG_ZERO = -1075 * math.log(2)
"""What a finding's measure reads as the log of an error of exactly zero (`log_ratio`): below the log of
the least positive float64, 2^-1074, so a reference error of zero orders under every positive one and
a Dew error of zero under every positive one. An order among findings, never a bound any verdict
reads."""


def _logs(values) -> np.ndarray:
    """Each nonnegative value's natural log, exact for every positive float64, subnormals included, and
    LOG_ZERO for zero."""
    values = np.asarray(values, np.float64)
    with np.errstate(divide="ignore"):
        return np.where(values > 0, np.log(np.where(values > 0, values, 1.0)), LOG_ZERO)


LOG_LARGEST = math.log(np.finfo(np.float64).max)
"""The log of float64's largest value: a quotient of finite evidence past it is held there where it is
recorded or shown (`_over`), never where it is judged (`_log_over`)."""


def _log_over(numerators, denominators) -> np.ndarray:
    """The log of each nonnegative numerator over its positive denominator (`_logs`), exact and finite for
    every finite pair, so a quotient past float64's range (an error of 1e154 over a row of 1e-160) keeps
    its order against every other; LOG_ZERO for a zero numerator, and not a number for one that is not."""
    numerators = np.asarray(numerators, np.float64)
    logs = np.where(numerators > 0, _logs(numerators) - _logs(denominators), LOG_ZERO)
    return np.where(np.isnan(numerators), np.nan, logs)


def _over(numerators, denominators) -> np.ndarray:
    """Each quotient of `_log_over` as a value, held at float64's largest: a recorded reference's worst
    row (`RecordedLeaf.worst`) and a message's figure; zero over anything is zero."""
    quotient = np.exp(np.minimum(_log_over(numerators, denominators), LOG_LARGEST))
    return np.where(np.asarray(numerators, np.float64) > 0, quotient, 0.0)


def _shown(log: float) -> str:
    """A log quotient as a message shows it, held at float64's largest."""
    return f"{float(np.exp(np.minimum(log, LOG_LARGEST))):.3e}"


def _scaled_rms(rows) -> np.ndarray:
    """Each row's root mean square of finite values, the row scaled by its largest first, so no square
    overflows (row 1628): zero for a row of zeros."""
    rows = np.abs(np.asarray(rows, np.float64))
    largest = np.max(rows, axis=1, keepdims=True) if rows.shape[1] else np.zeros((len(rows), 1))
    scale = np.where(largest > 0, largest, 1.0)
    return scale[:, 0] * np.sqrt(np.mean(np.square(rows / scale), axis=1))


def log_sum(values) -> float:
    """The natural log of the sum of nonnegative values, scaled by their largest first, so no sum of
    finite values overflows on the way: LOG_ZERO for a sum of zeros."""
    values = np.asarray(values, np.float64)
    largest = float(np.max(values)) if values.size else 0.0
    if not largest > 0:
        return LOG_ZERO
    return math.log(largest) + math.log(float(np.sum(values / largest)))


def root_mean(values) -> float:
    """The root of the mean of nonnegative values (each seed's mean squared error), scaled by their
    largest first, so no sum of finite values overflows on the way: finite wherever they are."""
    values = np.asarray(values, np.float64)
    largest = float(np.max(values)) if values.size else 0.0
    if not largest > 0:
        return 0.0
    return math.sqrt(largest) * math.sqrt(float(np.mean(values / largest)))


def log_ratio(mine, theirs) -> float:
    """A ratio rule's measure: the natural log of Dew's error over the reference's, the largest over its
    elements, each log exact for every positive error (`_logs`). Finite at every finite error and
    growing with Dew's, where the ratio itself overflows past an error of 4 over a reference of zero."""
    return float(np.max(_logs(mine) - _logs(theirs)))


class GateFailure(AssertionError):
    """A site the gate fails, its findings as data (`Finding`), its message theirs, one a line, unless a
    rule words its findings together. A test's report carries them as the JUnit property
    `gate_findings` (tests/conftest.py)."""

    def __init__(self, findings, message: str | None = None):
        self.findings = tuple(findings)
        super().__init__("\n".join(map(str, self.findings)) if message is None else message)


class Inconclusive(GateFailure):
    """Check 1 could not decide a leaf: its seeds do not resolve PARITY. Never a pass; the site takes
    larger inputs or more seeds (`seeds_for`)."""


def _pooled(mine: np.ndarray, theirs: np.ndarray) -> float:
    """sqrt(sum a / sum b): the RMS ratio over every draw and element, each draw's mean squared error
    weighed alike, as the owner's estimand reads it."""
    return float(np.sqrt(np.sum(mine) / np.sum(theirs))) if np.sum(theirs) > 0 else (
        0.0 if np.sum(mine) == 0 else float("inf"))


def _interval(mine: np.ndarray, theirs: np.ndarray | None = None, control: np.ndarray | None = None,
              tails: tuple[float, float] = (LOWER_TAIL, UPPER_TAIL)) -> tuple[float, float]:
    """The pooled ratio's bootstrap interval over seeds, its lower and upper `tails` left out: Dew's and
    the reference's errors resampled by the same seeds, or, for `control`, the reference's errors on
    two disjoint seed sets resampled apart (the reference compared with itself, an equally exact
    pair)."""
    rng = np.random.default_rng(0)
    count = len(mine)
    picks = rng.integers(0, count, (BOOTSTRAP, count))
    if control is None:
        ratios = np.sqrt(mine[picks].sum(axis=1) / np.maximum(theirs[picks].sum(axis=1), 1e-300))
    else:
        other = rng.integers(0, len(control), (BOOTSTRAP, len(control)))
        ratios = np.sqrt(mine[picks].sum(axis=1) / np.maximum(control[other].sum(axis=1), 1e-300))
    return float(np.quantile(ratios, tails[0])), float(np.quantile(ratios, 1 - tails[1]))


def _resolves(low: float, high: float) -> bool:
    """Whether a pooled ratio's interval is no wider than PARITY, so it tells an equally exact
    computation from one PARITY off."""
    return high <= PARITY * max(low, 1e-300)


def _status(ratio: float, low: float, high: float, *, contained: bool = True) -> str:
    """A leaf's verdict by its pooled ratio and interval: "pass", "over" or "inconclusive". The interval
    is taken to hold the ratio (low and high widened to it, which no endpoint's coverage loses by), so a
    pass certifies the ratio's high end at most PARITY², whether the interval lies wholly at or under
    PARITY or is resolved (no wider than PARITY) with the ratio under it, and an over its low end above
    one. Straddling PARITY and wider than it, the leaf is inconclusive, as is a ratio or end that is not
    a number. Not `contained`, the earlier rule: the interval as it is, a leaf over PARITY
    by its ratio wherever the interval does not straddle it unresolved."""
    if not contained:
        if low <= PARITY < high and not _resolves(low, high):
            return "inconclusive"
        return "over" if ratio > PARITY else "pass"
    if not all(np.isfinite(value) for value in (ratio, low, high)):
        return "inconclusive"
    low, high = min(low, ratio), max(high, ratio)
    if high <= PARITY:
        return "pass"
    if low > PARITY:
        return "over"
    if _resolves(low, high):
        return "pass" if ratio <= PARITY else "over"
    return "inconclusive"


ZeroRows = tuple[np.ndarray, np.ndarray, np.ndarray]
"""One seed's rows of float64 zero of a leaf that check 1 judges by the reference's error on them
(`_paired`): their indices, Dew's RMS errors on them and the reference's."""

NO_ZERO_ROWS: ZeroRows = (np.zeros(0, np.int64), np.zeros(0), np.zeros(0))


@dataclasses.dataclass
class _Draws:
    """Each leaf's errors over a run of seeds: Dew's and the reference's mean squared error from
    float64 (seed by leaf), the log of each seed's worst sized row over its size (Dew's, the
    reference's; `_log_over`), each
    seed's rows of float64 zero with Dew's and the reference's errors on them (`ZeroRows`, seed by leaf;
    `pooled_zero_rows`), whether every value of the leaf was finite on every seed, and each leaf's row
    width (its last axis's length, 0 for a leaf of no rows; NARROW_FACTORS)."""
    names: list[str]
    mine: np.ndarray
    theirs: np.ndarray
    worst: np.ndarray
    zero_rows: list[list[ZeroRows]]
    finite: np.ndarray
    widths: tuple[int, ...]

    def joined(self, other: "_Draws") -> "_Draws":
        return _Draws(self.names, *(np.concatenate([getattr(self, field), getattr(other, field)])
                                    for field in ("mine", "theirs", "worst")),
                      self.zero_rows + other.zero_rows, np.concatenate([self.finite, other.finite]),
                      self.widths)

    def against_larger(self, library: "_Draws", leaves: Sequence[str]) -> "_Draws":
        """These draws judged against the larger of their reference's error and `library`'s, the same
        Dew on the same draws against the library arm (`assert_as_exact`'s `library_fn`), on the
        exception's `leaves` alone: seed by seed and leaf by leaf, the reference's evidence (its error,
        its worst row, its rows of float64 zero) is whichever of the two arms' error is the larger there,
        and a value must be finite in both. Every other leaf is the reference's alone."""
        if library.names != self.names or len(library.mine) != len(self.mine):
            raise AssertionError("the library arm's draws are not the reference's")
        unknown = sorted(set(leaves) - set(self.names))
        if unknown or not leaves:
            raise ValueError(f"a library arm names the leaves it covers, of {self.names}; it names "
                             f"{list(leaves)}")
        covered = np.isin(self.names, leaves)
        pick = (library.theirs > self.theirs) & covered
        worst = self.worst.copy()
        worst[..., 1] = np.where(pick, library.worst[..., 1], self.worst[..., 1])
        zero_rows = [[(theirs if chosen else mine)
                      for mine, theirs, chosen in zip(seed, other, picked, strict=True)]
                     for seed, other, picked in zip(self.zero_rows, library.zero_rows, pick, strict=True)]
        return _Draws(self.names, self.mine, np.where(pick, library.theirs, self.theirs), worst, zero_rows,
                      self.finite & (library.finite | ~covered), self.widths)


def pooled_zero_rows(evidence: list[ZeroRows]) -> list[tuple[int, float, float]]:
    """The rows of float64 zero Dew is further from than ROW_FACTOR times the reference over a run's
    seeds, as (row, Dew's RMS error, the reference's): each row's squared errors summed over the seeds
    it is zero on, Dew's within ROW_FACTOR² of the reference's. A row of float64 zero is all residue,
    float32's rounding of a sum float64 cancels, with no size to judge it at, so it is held to the
    reference's residue on the same row; one seed's residues are heavy-tailed, and their ratio, a
    ratio of two, failed equally exact pairs seed by seed where the pooled one does not, while one
    row held ten times further still fails. Where the reference is exactly the float64 zero on a row
    over every seed, Dew must be too.

    Calibrated on attention's packed bias gradients, whose key rows are zero (`python
    tools/gate_null_runs.py attention-key-bias --paired --seeds K --runs N --chunk C`, and `--fault
    zero-row-10x`): 0 of 15,000 runs failed at 8 seeds (a red rate at most 4.6e-4 at 99.9%) and 0 of
    3,000 at 40 (at most 2.3e-3), and the 10x row failed every run at both. Judged seed by seed, the
    same pairs failed on 6.6% of runs of 8 seeds and 42% of 64."""
    mine: dict[int, list[float]] = {}
    theirs: dict[int, list[float]] = {}
    for rows, my_errors, their_errors in evidence:
        for row, my_error, their_error in zip(rows.tolist(), my_errors.tolist(), their_errors.tolist(),
                                              strict=True):
            mine.setdefault(row, []).append(my_error)
            theirs.setdefault(row, []).append(their_error)
    # Each row's errors pooled as a scaled root mean square and compared by their logs, so errors near
    # float64's largest pool and order without a square or a product overflowing.
    pooled = [(row, float(_scaled_rms([mine[row]])[0]), float(_scaled_rms([theirs[row]])[0]))
              for row in sorted(mine)]
    return [(row, my_rms, their_rms) for row, my_rms, their_rms in pooled
            if not _logs(my_rms) <= math.log(ROW_FACTOR) + _logs(their_rms)]


SKETCH = 32
"""The random sign projections a recording keeps of each leaf of its reference's float64 truth
(`truth_sketch`): a move of the truth independent of the signs, a wrong row or index, moves their root
mean square by about its own norm (a single element's by exactly its size); a move built against the
signs, in the span their complement leaves, is not seen, which no fixed sketch can exclude."""
SKETCH_DEVIATIONS = 2.0
"""How many times the oracle rule's bound on a difference's norm the root mean square of its SKETCH sign
projections may reach. A truth recomputed within the oracle rule, its difference independent of the
signs, is refused with probability under 1e-11 (a Rademacher sum's square is no heavier-tailed than a
normal's: Chernoff's bound on a chi-square of SKETCH degrees past 2² times its mean). A single
element's move past twice the bound and the sketches' own rounding (`sketch_rounding`) is refused
always; a wider move past them in norm is refused but
for its chance of lying near the signs' complement, which is no guarantee: two entries swapped whose
sign words agree move no projection, one pair in 2^32 at a seed, and each seed draws its own signs, so
a wrong truth repeated over a site's seeds hides at all of them only by that chance to their power."""
SKETCH_BLOCK = 64
"""The entries `truth_sketch` sums in float64 in one block, each block's sum then summed exactly: its
rounding at most gamma(SKETCH_BLOCK) of the magnitudes', in whatever order a host takes a block."""


def _sign_bits(size: int, seed: int) -> np.ndarray:
    """SKETCH random signs for each of `size` entries at a recorded seed, one bit a sign, the same on every
    host and run, and drawn apart at every seed."""
    return np.random.default_rng([0x5E7C4, size, seed]).integers(0, 1 << SKETCH, size, dtype=np.uint64)


_BYTE_SIGNS = ((np.arange(256)[:, None] >> np.arange(8)) & 1) * 2.0 - 1
"""A (256, 8) float64 table whose [byte, k] entry is +1.0 where the byte's bit k is set and -1.0 where it
is not, least significant bit first."""


def truth_sketch(truth, seed: int) -> np.ndarray:
    """A float64 truth leaf's sketch `[SKETCH]` at a recorded seed: its entries summed under SKETCH random
    signs of that seed's own (`_sign_bits`), each sign's
    product exact, in blocks of SKETCH_BLOCK whose float64 sums are summed exactly (`math.fsum`), so a
    truth recomputed apart is held to the recorded one without the recording keeping it, every host
    rounding the sketch alike within `sketch_rounding` (`indexed_recording`)."""
    flat = np.asarray(truth, np.float64).reshape(-1)
    if not flat.size:
        return np.zeros(SKETCH)
    bits, partials = _sign_bits(flat.size, seed), []
    for start in range(0, flat.size, 1 << 16):
        chunk = flat[start:start + (1 << 16)]
        # Sign k of an entry is its word's bit k: its low four bytes, read little-endian whatever the host's
        # order, so the first byte holds bits 0 to 7.
        words = bits[start:start + chunk.size].astype("<u8").view(np.uint8).reshape(chunk.size, 8)[:, :4]
        products = np.zeros((chunk.size + -chunk.size % SKETCH_BLOCK, SKETCH))
        np.multiply(chunk[:, None], _BYTE_SIGNS[words].reshape(chunk.size, SKETCH), out=products[:chunk.size])
        partials.append(products.reshape(-1, SKETCH_BLOCK, SKETCH).sum(axis=1))
    return np.asarray([math.fsum(column) for column in np.concatenate(partials).T.tolist()])


def sketch_rounding(magnitude: float) -> float:
    """The most `truth_sketch` of a leaf whose magnitudes sum to `magnitude` may round by, on any host:
    gamma(SKETCH_BLOCK) of it within the blocks, and the exact sum's one rounding."""
    eps = float(np.finfo(np.float64).eps)
    return (SKETCH_BLOCK * eps / 2 / (1 - SKETCH_BLOCK * eps / 2) + eps / 2) * magnitude


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

    @property
    def log_worst(self) -> float:
        """The log of `worst`, exact where this leaf was read here (`leaf_summary`), which keeps its
        order past float64's range; a recording's, from the value it holds."""
        return self.worst_log if self.worst_log is not None else float(_log_over(self.worst, 1.0))


@dataclasses.dataclass(frozen=True)
class _TruthRows:
    """What `leaf_summary` reads of a truth leaf alone, where a caller reads it once for both arms
    (`_LaidTruths` reads a draw's at once): each row's own RMS (`_scaled_rms`), its size (the RMS where
    it is past RESOLUTION of the leaf, else zero: `row_errors`'s), and whether every value is finite."""

    own: np.ndarray
    sizes: np.ndarray
    finite: bool

    @staticmethod
    def of(truth: np.ndarray) -> "_TruthRows":
        own, sizes = _row_sizes(truth) if truth.ndim >= 2 else (np.zeros(0), np.zeros(0))
        return _TruthRows(own, sizes, bool(np.isfinite(truth).all()))


def leaf_summary(reference, truth, rows_of: _TruthRows | None = None) -> RecordedLeaf:
    """`RecordedLeaf` of a leaf against its float64 truth: check 1's evidence of the reference's leaf, or
    of Dew's (`_drawn` reads both so, a recorded reference's as recorded). An empty leaf is exact.
    `rows_of` is the truth's own rows (`_TruthRows.of`), where a caller has read them already."""
    dtype = str(jnp.dtype(getattr(reference, "dtype", np.asarray(reference).dtype)))
    reference, truth = np.asarray(reference, np.float64), np.asarray(truth, np.float64)
    if not reference.size:
        assert truth.shape == reference.shape, (f"an empty leaf {reference.shape} against a truth "
                                                f"{truth.shape}")
        return RecordedLeaf(0.0, 0.0, np.zeros(0, np.int64), np.zeros(0), finite=True, dtype=dtype)
    held = rows_of or _TruthRows.of(truth)
    # Each row's own size by the arithmetic its errors take (`_scaled_rms`), so an arm exactly zero on a
    # row is exactly its size from it: `row_errors`' sizes and errors, the truth's read once.
    sizes, own = held.sizes, held.own
    rows = _rows_rms(reference - truth, truth.shape[-1]) if truth.ndim >= 2 else np.zeros(0)
    sized = sizes > 0
    logged = float(np.max(_log_over(rows[sized], sizes[sized]))) if sized.any() else LOG_ZERO
    return RecordedLeaf(float(np.mean(np.square(reference - truth))),
                        float(np.max(_over(rows[sized], sizes[sized]))) if sized.any() else 0.0,
                        np.flatnonzero(~sized), rows[~sized],
                        finite=bool(np.isfinite(reference).all()) and held.finite, dtype=dtype,
                        zero_sizes=own[~sized], worst_log=logged)


@dataclasses.dataclass(frozen=True)
class _LaidTruths:
    """A draw's truth leaves read at once for both arms (`_summaries`), each as `_TruthRows` reads one.
    Every leaf that is not empty and is C-contiguous in float64 lies end to end with the others, by row
    width (its last axis's, 0 for a leaf of rank under two) and then size: each run of one width and size
    is one (leaves, size) matrix, its rows one (rows, width) matrix. Each reduction `leaf_summary` takes
    over a row or a leaf is then the same reduction of the same values, NumPy's pairwise sum of that row
    or leaf alone wherever it lies, and each log and exponential the same elementwise, so a draw takes a
    few passes where it took a pipeline a leaf (ledger 1848: 190 µs a leaf at the gemma3n gradient's
    tiny leaves). A leaf in another layout is read alone: NumPy sums an F-ordered row in another order.
    `truths` is every leaf in float64, `laid` the leaves laid in order, `runs` each run's (leaves, size,
    width), `values` the laid leaves, `own` and `sizes` their rows' own RMS and size, and `finite` each
    laid leaf's."""

    truths: list
    laid: list[int]
    runs: list[tuple[int, int, int]]
    values: np.ndarray
    own: np.ndarray
    sizes: np.ndarray
    finite: np.ndarray

    @staticmethod
    def of(truths) -> "_LaidTruths":
        truths = [np.asarray(truth, np.float64) for truth in truths]
        keys = sorted((truth.shape[-1] if truth.ndim >= 2 else 0, truth.size, at)
                      for at, truth in enumerate(truths) if truth.size and truth.flags.c_contiguous)
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
        finite = [np.isfinite(block).all(axis=1) for block in blocks]
        return _LaidTruths(truths, [at for *_, at in keys], runs, values, own,
                           np.where(own > np.concatenate(resolved), own, 0.0),
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


def _summaries(references, held: _LaidTruths) -> list[RecordedLeaf | None]:
    """Each leaf's `leaf_summary` against its truth in `held` (`_LaidTruths.of`), None for a reference of
    None. Every leaf `held` lays whose reference is C-contiguous in float64 too is read in a few passes
    over all of them, bit for bit the summary `leaf_summary` reads of it alone; every other leaf is read
    alone (`leaf_summary`), the order its arithmetic sums in following its layout."""
    summaries = [None] * len(references)
    arms = [None if references[at] is None else np.asarray(references[at], np.float64) for at in held.laid]
    for arm, at in zip(arms, held.laid, strict=True):
        if arm is not None and arm.shape != held.truths[at].shape:
            raise AssertionError(f"a leaf {arm.shape} against a truth {held.truths[at].shape}")
    read = [arm is not None and arm.flags.c_contiguous for arm in arms]
    if any(read):
        # An arm read alone or not at all lies as its truth, a zero error never read.
        values = np.concatenate([np.zeros(0), *((arm if ok else held.truths[at]).reshape(-1)
                                                for arm, ok, at in zip(arms, read, held.laid, strict=True))])
        finite = np.concatenate([np.isfinite(block).all(axis=1) for block in _blocks(values, held.runs)])
        errors = values - held.values
        mse = np.concatenate([np.mean(block, axis=1) for block in _blocks(np.square(errors), held.runs)])
        rows = np.concatenate([np.zeros(0), *map(_scaled_rms, _rows(errors, held.runs))])
        # Each laid leaf's rows, its first among them, and those sized past RESOLUTION.
        counts = np.repeat([size // width if width else 0 for _, size, width in held.runs],
                           [leaves for leaves, _, _ in held.runs])
        starts = np.cumsum(counts) - counts
        sized = held.sizes > 0
        before = np.concatenate([[0], np.cumsum(sized)])
        taken = before[starts + counts] - before[starts]
        logs, over = _log_over(rows[sized], held.sizes[sized]), _over(rows[sized], held.sizes[sized])
        # The worst sized row of each leaf that has one: the largest of its run of the sized rows.
        some = taken > 0
        worst_log, worst = np.full(len(counts), LOG_ZERO), np.zeros(len(counts))
        if some.any():
            firsts = (np.cumsum(taken) - taken)[some]
            worst_log[some] = np.maximum.reduceat(logs, firsts)
            worst[some] = np.maximum.reduceat(over, firsts)
        # The rows of float64 zero, each leaf's its span of them.
        unsized = np.flatnonzero(~sized)
        zero_rows = unsized - np.repeat(starts, counts - taken)
        zero_errors, zero_sizes = rows[unsized], held.own[unsized]
        ends = list(itertools.accumulate((counts - taken).tolist(), initial=0))
        mse, worst, worst_log, finite = mse.tolist(), worst.tolist(), worst_log.tolist(), (
            finite & held.finite).tolist()
        names = {}
        for index, at in enumerate(held.laid):
            if read[index]:
                reference, span = references[at], slice(ends[index], ends[index + 1])
                dtype = reference.dtype if hasattr(reference, "dtype") else np.asarray(reference).dtype
                if dtype not in names:
                    names[dtype] = str(jnp.dtype(dtype))
                summaries[at] = RecordedLeaf(mse[index], worst[index], zero_rows[span], zero_errors[span],
                                             finite=finite[index], dtype=names[dtype],
                                             zero_sizes=zero_sizes[span], worst_log=worst_log[index])
    for at, reference in enumerate(references):
        if reference is not None and summaries[at] is None:
            summaries[at] = leaf_summary(reference, held.truths[at])
    return summaries


def _drawn(dew_fn, reference_fn, draw, seeds, truth_fn=None, reference_seeds=None, orders=None,
           library_fn=None, library_leaves: Sequence[str] = (), also: Sequence = ()):
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
    truth evaluated once for them all: the reference's draws, then each further reference's, a list."""
    if orders is not None and getattr(reference_fn, "recorded", False):
        raise ValueError("a recorded reference holds its recorded draws alone, never another order of them")
    arms = [reference_fn] + ([] if library_fn is None else [library_fn]) + list(also)
    mine, theirs, worst, zero_rows, finite, names, widths = [], [[] for _ in arms], [[] for _ in arms], \
        [[] for _ in arms], [[] for _ in arms], None, None

    def truth_of(inputs):
        with jax.enable_x64(new_val=True):
            return (truth_fn or reference_fn)(*widened(inputs))

    for at, seed in enumerate(seeds):
        inputs = draw(seed)
        truth = truth_of(inputs)
        if reference_seeds is None:
            their_inputs, their_truth = inputs, truth
        else:
            their_inputs = draw(reference_seeds[at])
            their_truth = truth_of(their_inputs)
        if orders is None:
            ordered = [(inputs, their_inputs)]
        else:
            ordered = list(zip(orders(inputs), orders(their_inputs), strict=True))
            if len(ordered) < 2:
                raise ValueError(f"orders of {len(ordered)}; an orbit holds the draw's own and another")
        rows = [None for _ in arms]
        for mine_at, theirs_at in ordered:
            dew = dew_fn(*mine_at)
            names = names or [jax.tree_util.keystr(path)
                              for path, _ in jax.tree_util.tree_leaves_with_path(dew)]
            widths = widths or tuple(int(np.shape(leaf)[-1]) if np.ndim(leaf) >= 2 else 0
                                     for leaf in jax.tree.leaves(dew))
            for at_arm, arm in enumerate(arms):
                # The reference against itself on the same inputs is one evaluation (`seeds_for`'s
                # sizing); the library arm runs on Dew's own inputs.
                here, true_here = (theirs_at, their_truth) if arm is reference_fn else (mine_at, truth)
                other = dew if dew_fn is arm and here is mine_at else arm(*here)
                evidence = _evidence(dew, other, truth, true_here)
                rows[at_arm] = evidence if rows[at_arm] is None else [
                    (first[0], first[1], float(np.fmax(first[2], later[2])),
                     float(np.fmax(first[3], later[3])), first[4], first[5] and later[5])
                    for first, later in zip(rows[at_arm], evidence, strict=True)]
        mine.append([entry[0] for entry in rows[0]])
        for at_arm, row in enumerate(rows):
            theirs[at_arm].append([entry[1] for entry in row])
            worst[at_arm].append([(entry[2], entry[3]) for entry in row])
            zero_rows[at_arm].append([entry[4] for entry in row])
            finite[at_arm].append([entry[5] for entry in row])
    drawn = [_Draws(names, np.asarray(mine), np.asarray(theirs[at_arm]),
                    np.asarray(worst[at_arm]).reshape(len(mine), -1, 2), zero_rows[at_arm],
                    np.asarray(finite[at_arm]), widths) for at_arm in range(len(arms))]
    first = drawn[0] if library_fn is None else drawn[0].against_larger(drawn[1], library_leaves)
    return [first, *drawn[len(arms) - len(also):]] if also else first


def _leaf_paths(tree) -> list[str]:
    return [jax.tree_util.keystr(path) for path, _ in jax.tree_util.tree_leaves_with_path(tree)]


def _evidence(dew, reference, truth, their_truth) -> list[tuple]:
    """One draw's `_paired` evidence of every leaf of `dew` and `reference` against their truths. The
    trees are held alike first: Dew's, the reference's (a recorded one's leaves `RecordedLeaf`) and
    both truths the same leaves at the same paths, and each leaf the shape of its truth, so no leaf is
    paired with another's truth or compared through broadcasting. Every leaf's summary is read at once
    (`_summaries`)."""
    # Every leaf on the host in one transfer a tree, not one a leaf (`leaf_summary` reads them on the host);
    # one truth stays one tree, as the draw gave it.
    shared = their_truth is truth
    dew, reference, truth = jax.device_get((dew, reference, truth))
    their_truth = truth if shared else jax.device_get(their_truth)
    paths = _leaf_paths(truth)
    for name, tree in (("dew", dew), ("the reference", reference), ("the reference's truth", their_truth)):
        if _leaf_paths(tree) != paths:
            raise AssertionError(f"{name}'s leaves are {_leaf_paths(tree)}, the truth's {paths}")
    leaves, references, truths, their_truths = (jax.tree.leaves(tree)
                                                for tree in (dew, reference, truth, their_truth))
    recorded = [isinstance(their_leaf, RecordedLeaf) for their_leaf in references]
    for path, leaf, their_leaf, true, their_true, was in zip(paths, leaves, references, truths, their_truths,
                                                             recorded, strict=True):
        shapes = {"dew": np.shape(leaf), "the reference's truth": np.shape(their_true)}
        if not was:
            shapes["the reference"] = np.shape(their_leaf)
        for name, shape in shapes.items():
            if shape != np.shape(true):
                raise AssertionError(f"{path}: {name}'s shape is {shape}, the truth's {np.shape(true)}")
    # The truths' rows are read once a draw, for both arms where the two truths are the same values.
    held = _LaidTruths.of(truths)
    same = all(their_true is true for their_true, true in zip(their_truths, truths, strict=True))
    mine = _summaries(leaves, held)
    theirs = _summaries([None if was else leaf for leaf, was in zip(references, recorded, strict=True)],
                        held if same else _LaidTruths.of(their_truths))
    row = []
    for path, leaf, their_leaf, true, their_true, was, summary, their_summary in zip(
            paths, leaves, references, truths, their_truths, recorded, mine, theirs, strict=True):
        their = their_leaf if was else their_summary
        if np.size(leaf) == 0:
            # An empty leaf (a one-tap convolution's history) is exact by construction: held to the
            # reference's dtype (its shape is held above; the truth's dtype, float64, is its own), it
            # adds no error and resolves nothing.
            kept = str(jnp.dtype(leaf.dtype))
            assert kept == their.dtype, (path, kept, their.dtype)
            row.append((0.0, 0.0, LOG_ZERO, LOG_ZERO, NO_ZERO_ROWS, True))
            continue
        # Rows pair where the two truths are the same values: the same draw, or an orbit's draws.
        paired = (was or their_truth is truth
                  or np.array_equal(np.asarray(true), np.asarray(their_true)))
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
                mine.finite and theirs.finite)
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
            worst = max(worst, float(_log_over(error, size)))
            their_worst = max(their_worst, float(_log_over(at_zero[at], size)))
        else:
            zero.append((at, error, at_zero[at]))
    rows, errors, their_errors = (np.asarray(column, dtype) for column, dtype in zip(
        zip(*zero, strict=True) if zero else ((), (), ()), (np.int64, np.float64, np.float64), strict=True))
    return (mine.mse, theirs.mse, worst, their_worst, (rows, errors, their_errors),
            mine.finite and theirs.finite)


def _judged(draws: _Draws, index: int, tails: tuple[float, float] = (LOWER_TAIL, UPPER_TAIL)
            ) -> tuple[float, float, float]:
    """A leaf's pooled ratio and its interval over the draws, its `tails` left out."""
    a, b = draws.mine[:, index], draws.theirs[:, index]
    return (_pooled(a, b), *_interval(a, b, tails=tails))


def _judgement(draws: _Draws, label: str, tails: tuple[float, float] = (LOWER_TAIL, UPPER_TAIL), *,
               contained: bool = True) -> tuple[list[Finding], list[Finding]]:
    """Check 1's findings over `draws` (`assert_as_exact`): each leaf over PARITY (`_status`), a row or
    a value it fails, and each leaf its draws leave inconclusive."""
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
            if status == "inconclusive":
                unresolved.append(Finding(
                    leaf, "inconclusive", measure, f"{leaf}: inconclusive: the pooled ratio {ratio:.3f}'s "
                    f"interval over {count} seeds is [{low:.3f}, {high:.3f}], wider than {PARITY}; give the "
                    f"site larger inputs or more seeds (`seeds_for`)"))
            elif status == "over":
                over.append(Finding(leaf, "pooled", measure, f"{leaf}: dew's RMS error from float64 is "
                                    f"{ratio:.3f} times the reference's over {count} seeds, interval "
                                    f"[{low:.3f}, {high:.3f}] (allowed {PARITY})"))
        my_row, their_row = float(np.max(draws.worst[:, index, 0])), float(np.max(draws.worst[:, index, 1]))
        factor = NARROW_FACTORS.get(draws.widths[index], ROW_FACTOR)
        if not my_row <= math.log(factor) + their_row:
            over.append(Finding(leaf, "row", my_row - their_row,
                                f"{leaf}: the worst row over {count} seeds is {_shown(my_row)} from float64 "
                                f"over its size, the reference's {_shown(their_row)}, allowed "
                                f"{_shown(math.log(factor) + their_row)} ({factor} times it)"))
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


NARROW_FACTORS = {1: 90.0, 2: 11.0}
"""The worst-row rule's factor at rows of one or two values, in place of ROW_FACTOR. At either width
ROW_FACTOR fails an equally exact Dew far more often than LOWER_TAIL as sites run check 1 (1.5e-2 of
runs at one value a row, 1.5e-3 at two): a row of few values is sized by a few float64 values, which sit
near zero far more often than a wide row's RMS does, and the worst row over a site's rows and seeds is
the row whose truth is smallest. Each factor sits above the 1 - 1e-4 quantile of the factor that would
fail equally exact pairs at both stages, both arms on one truth (`tools/gate_null_runs.py --rows`, its
`row_controls`, through `_staged` itself), at every layout measured, 16 to 4,096 rows a draw and 8 to 512
declared seeds: that quantile is 40 to 53 at one value and 7 to 8 at two (10,000 to 600,000 runs a
layout), so 90 and 11 stand about 1.7 and 1.4 times above it, at measured rates near 2e-5 and at most
3.3e-5; 28,000 fresh runs at the 16-row and 256-row layouts bound them within LOWER_TAIL at 99.9% (at
most 4.7e-4). Rows of three values keep ROW_FACTOR at a measured rate of at most 2.9e-4 over 16 to 2,048
rows, a point rate, bounded within LOWER_TAIL at 256 rows and 8 seeds (80,000 runs) and not at 2,048. The
rate is a leaf's: a site of L narrow leaves may fail an equally exact Dew L times as often.

The power this gives up is local: a narrow row whose error is between ROW_FACTOR and these factors
times the reference's worst no longer fails by its row (the pooled ratio and the leaf's other rows still
judge it), and a narrow false alarm no longer draws the second stage for the site's other leaves, which
that stage exists to judge on real findings. Check 2 (tests/certified.py `assert_certified`) holds
each element within its own operation's certified bound where a site runs it, which resolves such an
error at a pointwise step or a short sum: a sum of n terms is held within γ_n Σ|terms|, about n·u
Σ|terms| for u the unit roundoff, of its exact value, so its reach in units of the sum's own last place grows
with n and with the sum's cancellation (for standard normal terms, the smallest error it flagged in a
review's measurement, the median over 200 sums, was 37 units in the sum's last place at 16 terms, 918 at
128 and 22,966 at 1,024): past a few dozen terms of that kind an error in this window passes both checks
wherever the pooled ratio does not see it."""


PREVIOUS_TAILS = (5e-4, 5e-4)
"""The interval's tails before check 1's two stages (`_staged`'s `two_stage=False`)."""


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
        draws = draws.joined(drawn(seeds, (1 + REDRAW) * seeds))
    return (*_judgement(draws, label, tails, contained=two_stage), 2 if further else 1)


STAGES_LOG = "DEW_CHECK1_STAGES"
"""The environment variable naming a file each check-1 judgement appends its stage, its verdict and its
leaves of narrow rows (NARROW_FACTORS) to, one JSON line (`_logged`): what a train reads every site's
stage-two rate from, every attempt counted, and which sites' local errors rest on check 2."""


def _logged(label: str, stage: int, verdict: str, narrow_leaves: list[str]) -> None:
    import json
    import os

    path = os.environ.get(STAGES_LOG)
    if path:
        with open(path, "a", encoding="utf-8") as log:
            log.write(json.dumps({"site": os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" (", 1)[0],
                                  "label": label, "stage": stage, "verdict": verdict,
                                  "narrow_leaves": narrow_leaves}) + "\n")


def assert_as_exact(dew_fn, reference_fn, draw, label: str, *, seeds: int, truth_fn=None,
                    orders=None, library_fn=None, library_row: int | None = None,
                    library_leaves: Sequence[str] = (), also: Mapping[str, Callable] | None = None) -> None:
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
    for them all (`_drawn`): one Dew arm against several references, the truth `truth_fn` (which a site
    with several references names). Each compiles paired with Dew's arm (`_arms`), and a plain Dew arm,
    whose compilation its pairing decides, takes references of one kind, plain or `as_given`.

    This is an engineering gate, deterministic and reproducible, not a guarantee against every error
    law: "never wrong" is check 2's, each element within its operation's certified bound
    (tests/certified.py)."""
    if seeds < FEWEST_SEEDS:
        raise Inconclusive([Finding(label, "inconclusive", FEWEST_SEEDS / max(seeds, 1), f"{label}: {seeds} "
                                    f"seeds cannot be judged; check 1 takes at least {FEWEST_SEEDS} "
                                    f"(`seeds_for`)")])
    narrow_leaves: list[str] = []
    if (library_fn is None) != (library_row is None) or (library_fn is None) != (not library_leaves):
        raise ValueError(f"{label}: a library arm is the owner's exception, named with its ledger row and "
                         f"the leaves it covers")
    also = dict(also or {})
    mixed = not _given(dew_fn) and any(_given(fn) != _given(reference_fn) for fn in also.values())
    if also and (truth_fn is None or mixed):
        raise ValueError(f"{label}: further references name the truth, and beside a plain Dew arm each is "
                         f"of the reference's kind, plain or as given, so one compilation of Dew's pairs "
                         f"with all")
    _refuse_numpy_arms(draw, dew_fn, reference_fn, *([] if library_fn is None else [library_fn]),
                       *also.values())
    further = [_arms(dew_fn, fn)[1] for fn in also.values()]
    dew_fn, reference_fn = _arms(dew_fn, reference_fn)
    library_fn = None if library_fn is None else _arms(library_fn, library_fn)[1]
    truth_fn = _truth_program(truth_fn)
    # Each range's draws, every reference's, computed once for the stages of them all.
    ranges: dict[tuple[int, int], list[_Draws]] = {}

    def drawn_for(index: int):
        def drawn(start, stop):
            if (start, stop) not in ranges:
                draws = _drawn(dew_fn, reference_fn, draw, range(start, stop), truth_fn, orders=orders,
                               library_fn=library_fn, library_leaves=library_leaves, also=further)
                ranges[start, stop] = draws if further else [draws]
                narrow_leaves[:] = [name for name, width in zip(ranges[start, stop][0].names,
                                                                ranges[start, stop][0].widths, strict=True)
                                    if width in NARROW_FACTORS]
            return ranges[start, stop][index]
        return drawn

    overs, unresolveds = [], []
    for index, named in enumerate([label, *(f"{label}, against {name}" for name in also)]):
        over, unresolved, stage = _staged(drawn_for(index), seeds, named)
        _logged(named, stage, _verdict(over, unresolved), narrow_leaves)
        overs += over
        unresolveds += unresolved
    if overs:
        raise GateFailure(overs + unresolveds)
    if unresolveds:
        raise Inconclusive(unresolveds)


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
    draws = _drawn(changed_fn, original_fn, draw, range(seeds), truth_fn)
    moved, kept, unresolved = _moved(draws, 1)
    if not moved:
        further = range(seeds, (1 + REDRAW) * seeds)
        draws = draws.joined(_drawn(changed_fn, original_fn, draw, further, truth_fn))
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
              truth_fn=None, dew_fn=None, orders=None) -> int:
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
    order, as the verdict reads it."""
    if dew_fn is not None and getattr(reference_fn, "recorded", False):
        pilots = getattr(reference_fn, "pilots", None)
        if pilots is None:
            raise ValueError(f"{label}: a recorded reference holds its recorded seeds alone, never a paired "
                             f"sizing's from SIZING_FROM; its site keeps its recorded count")
        most = min(most, len(pilots) // 2)
    paired = dew_fn is not None
    _refuse_numpy_arms(draw, *([dew_fn] if paired else []), reference_fn)
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
                       truth_fn, orders=orders)
        drawn = fresh if drawn is None else drawn.joined(fresh)
        wide = _confirmed(drawn.names, drawn.theirs, count, None if dew_fn is None else drawn.mine)
        if not wide:
            return count
        count *= 2
    compared = "the reference compared with itself" if dew_fn is None else "Dew paired with the reference"
    raise AssertionError(f"{label}: {compared} does not resolve {PARITY} within {most} seeds at {wide[:4]}; "
                         f"give the site larger inputs or fewer shared factors")


def _confirmed(names, theirs: np.ndarray, count: int, mine: np.ndarray | None = None) -> list[str]:
    """The leaves `seeds_for` leaves unresolved at `count`: on the count's own block of draws or on the
    next block of as many (`seeds_for`'s confirmation), each judged as the sizing judges it: paired,
    `mine` Dew's squared errors against the reference's (`theirs`, seed by leaf) on the same draws, or
    the reference compared with itself (`_unresolved_against_itself`, each block twice the count)."""
    wide: list[str] = []
    for start in (0, count) if mine is not None else (0, 2 * count):
        if mine is not None:
            held, ours = theirs[start:start + count], mine[start:start + count]
            unresolved = [name for index, name in enumerate(names) if np.sum(held[:, index]) > 0
                          and not _resolves(*_interval(ours[:, index], held[:, index]))]
        else:
            unresolved = _unresolved_against_itself(theirs[start:start + 2 * count], names, count)
        wide += [name for name in unresolved if name not in wide]
    return wide


def _unresolved_against_itself(errors: np.ndarray, names, count: int) -> list[str]:
    """The leaves the reference compared with itself does not resolve at `count` seeds: `errors`, its
    squared errors a seed (rows) and a leaf (columns), on seeds 0 to `count` against `count` to twice
    it (`_interval`'s control); a leaf with no error on either half is resolved."""
    halves = errors[:count], errors[count:2 * count]
    return [name for index, name in enumerate(names)
            if np.sum(halves[0][:, index]) > 0 and np.sum(halves[1][:, index]) > 0
            and not _resolves(*_interval(halves[0][:, index], control=halves[1][:, index]))]


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
              library_leaves: Sequence[str] = ()) -> list[tuple[str, int, list[str]]]:
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
    own draws)."""
    if paired and dew_fn is None:
        raise ValueError("a paired null run takes its other implementation, `dew_fn`")
    verdicts = []
    span = (1 + REDRAW) * seeds
    for run in range(first, first + runs):
        base = run * 2 * span

        def drawn(start, stop, base=base):
            return _drawn(dew_fn or reference_fn, reference_fn, draw, range(base + start, base + stop),
                          truth_fn, orders=orders,
                          reference_seeds=None if paired else range(base + span + start, base + span + stop),
                          library_fn=library_fn, library_leaves=library_leaves)

        over, unresolved, stage = _staged(drawn, seeds, label, two_stage=two_stage)
        verdicts.append((_verdict(over, unresolved), stage, [str(finding) for finding in over + unresolved]))
    return verdicts


def _verdict(over: list[Finding], unresolved: list[Finding]) -> str:
    """A judgement's verdict: "over" where any leaf failed, else "inconclusive" where any is unresolved,
    else "pass"."""
    return "over" if over else "inconclusive" if unresolved else "pass"


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
    if count < (1 + REDRAW) * seeds:
        raise AssertionError(f"{site}: {count} recorded seeds, and check 1 over {seeds} may draw "
                             f"{(1 + REDRAW) * seeds}; record (1 + REDRAW) times the declared seeds")

    def draw(seed):
        if inputs_of is None:
            return np.int32(seed), {name: value[seed] for name, value in inputs.items()}
        drawn = inputs_of(seed)
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
    if min(count, len(digests)) < (1 + REDRAW) * seeds:
        raise AssertionError(f"{site}: {min(count, len(digests))} recorded seeds, and check 1 over {seeds} "
                             f"may draw {(1 + REDRAW) * seeds}")
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
        drawn = inputs_of(seed)
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


def assert_no_less_exact(dew, reference, truth, label: str) -> None:
    """An unchanged computation has no larger float64 distance than its source."""
    mine, theirs = distance(dew, truth), distance(reference, truth)
    assert within(mine, theirs), (
        f"{label}: {mine:.3e} from float64 (rms), the unchanged source {theirs:.3e}")


def assert_leaves_as_exact(dew, reference, truth, *, reductions: int = 1, terms=None,
                           rows: bool = True) -> None:
    """Each leaf of the tree `dew` within FACTOR times the matching leaf of
    `reference`'s RMS distance from float64 (`truth`), the three trees alike.

    A port that sums `reductions` terms the reference sums in another order
    rounds by up to sqrt(reductions) float32 units of what it sums in RMS,
    so the reference's distance counts as at least that: one draw of
    rounding below it is luck, not a bar, and a leaf that lost a term still
    fails by orders of magnitude. What a leaf sums is its own float64 size,
    or, where `terms` gives the tree of each leaf's terms (`float64_terms`),
    the larger of that and their root sum of squares, the size of the
    partial sums a reordered sum rounds. A leaf whose sum cancels, as a key
    bias's gradient does under the softmax's shift invariance, is held to
    its terms' rounding, not to its own near-zero size; one whose terms'
    signs are independent keeps about its own. Each leaf's rows are held to
    the reference's too (`rows_over`), unless `rows` is false: after a
    second training step a row inherits the first step's rounding through
    the parameters it reads, which its own terms do not see, so such a
    comparison holds each tensor alone and its first step holds the rows."""
    unit = float(np.finfo(np.float32).eps)
    spread = np.sqrt(reductions)
    sizes = jax.tree.leaves(truth) if terms is None else jax.tree.leaves(terms)
    over = []
    leaves = zip(jax.tree_util.tree_leaves_with_path(dew), jax.tree.leaves(reference), jax.tree.leaves(truth),
                 sizes, strict=True)
    for (path, mine), theirs, true, summed in leaves:
        scale = max(distance(true, 0), distance(summed, 0))
        bar = FACTOR * max(distance(theirs, true), spread * unit * scale)
        if not within(distance(mine, true), bar):
            over.append(f"{jax.tree_util.keystr(path)}: {distance(mine, true):.3e} from float64 (rms), "
                        f"the reference {distance(theirs, true):.3e}, allowed {bar:.3e}")
        # Each row over its own terms where they are given, the order floor in those units.
        over += [f"{jax.tree_util.keystr(path)}: {message}"
                 for _, _, message in (rows_over(mine, theirs, true, None if terms is None else summed,
                                                 spread * unit) if rows else ())]
    assert not over, "\n".join(over)


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
    behind a sum, which would bound the whole computation."""
    closed = jax.make_jaxpr(fn)(*args)
    values, energies = _evaluated(closed.jaxpr, closed.consts, jax.tree.leaves(args))
    tree = jax.tree.structure(jax.eval_shape(fn, *args))
    return tree.unflatten(values), tree.unflatten([np.sqrt(np.asarray(_or_size(energy, value)))
                                                   for energy, value in zip(energies, values, strict=True)])


def _evaluated(jaxpr, consts, args, given=None):
    """`jaxpr`'s outputs and their terms' energies, the sums of their squares
    (`float64_terms`), from `args` and the energies `given` for them (none for
    a computation's own inputs). A value no sum made has none."""
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
    for index, equation in enumerate(jaxpr.eqns):
        inputs = [read(var) for var in equation.invars]
        with equation.ctx.manager:
            outputs, made = _applied(equation, inputs, [energy(var) for var in equation.invars],
                                     any(var in wanted for var in equation.outvars))
        for var in equation.invars:
            if not isinstance(var, Literal) and last.get(var) == index:
                values.pop(var, None)
                energies.pop(var, None)
        for var, value, made_energy in zip(equation.outvars, outputs, made, strict=True):
            values[var], energies[var] = value, made_energy
    return [read(var) for var in jaxpr.outvars], [energy(var) for var in jaxpr.outvars]


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


def _applied(equation, inputs, given, wanted: bool):
    """`equation`'s outputs from `inputs`, and their energies where `wanted`."""
    name, params = equation.primitive.name, equation.params
    body = _body(equation)
    if wanted and body is not None:
        return _evaluated(*body, inputs, given)
    if wanted and name == "scan":
        return _scanned(params, inputs, given)
    if wanted and name == "while":
        return _looped(params, inputs, given)
    if wanted and name == "shard_map":
        return _mapped(params, inputs, given)
    if wanted and name == "cond":
        branch = params["branches"][int(inputs[0])]
        return _evaluated(branch.jaxpr, branch.consts, inputs[1:], given[1:])
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


def _scanned(params, inputs, given):
    """A scan's outputs and energies (`float64_terms`), its body evaluated a step at a time."""
    body = params["jaxpr"]
    constants, carries, _ = (len(group) for group in params["ft_in"].update(range(len(inputs))).unpack())
    consts, carry, xs = (inputs[:constants], inputs[constants:constants + carries],
                         inputs[constants + carries:])
    held, carry_terms, xs_terms = (given[:constants], given[constants:constants + carries],
                                   given[constants + carries:])
    ys, ys_terms = [], []
    steps = range(params["length"])
    for step in reversed(steps) if params["reverse"] else steps:
        outputs, made = _evaluated(body.jaxpr, body.consts, [*consts, *carry, *(x[step] for x in xs)],
                                   [*held, *carry_terms, *(None if t is None else t[step] for t in xs_terms)])
        carry, carry_terms = outputs[:carries], made[:carries]
        ys.append(outputs[carries:])
        ys_terms.append(made[carries:])
    if params["reverse"]:
        ys.reverse()
        ys_terms.reverse()
    stacked = [jnp.stack(column) for column in zip(*ys, strict=True)] if ys else []
    stacked_terms = [None if any(t is None for t in column) else jnp.stack(column)
                     for column in zip(*ys_terms, strict=True)] if ys_terms else []
    return [*carry, *stacked], [*carry_terms, *stacked_terms]


def _looped(params, inputs, given):
    """A while loop's carries and energies (`float64_terms`), its body evaluated a step at a time."""
    condition, body = params["cond_jaxpr"], params["body_jaxpr"]
    tested, run = params["cond_nconsts"], params["body_nconsts"]
    condition_consts, body_consts, carry = inputs[:tested], inputs[tested:tested + run], inputs[tested + run:]
    body_terms, carry_terms = given[tested:tested + run], given[tested + run:]
    while bool(_evaluated(condition.jaxpr, condition.consts, [*condition_consts, *carry])[0][0]):
        carry, carry_terms = _evaluated(body.jaxpr, body.consts, [*body_consts, *carry],
                                        [*body_terms, *carry_terms])
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
