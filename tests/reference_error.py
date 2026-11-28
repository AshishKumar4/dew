"""Dew's rounding error held to the reference's own, both measured from float64.

The gate is check 1 over a site's sized seeds (`assert_as_exact`, docs/design/testing.md) beside check 2's
certified bounds (tests/certified.py). The one-draw rules below hold the comparisons that have not
converted yet.

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

import contextlib
import dataclasses
import functools
import hashlib
import json
import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple

import arm_arithmetic
import jax
import jax.numpy as jnp
import numpy as np
from arm_arithmetic import (
    FORMATTED,
    Truth,
    drawn_on_the_host,
    rounds_nothing,
    summed_as_run,
    unrewritten,
    widened,
)
from arm_forms import (
    HELD,
    _arms,
    _arms_with,
    _bounded,
    _declared,
    _given,
    _holds_program,
    _layers,
    _mode_of,
    _one_draws_program,
    _refuse_numpy_arms,
    _RunToRun,
    _varies,
    _window_count,
    _Windowed,
    _written_references,
    _written_role,
    _written_twin,
    _ynn_decided,
    as_given as as_given,
    as_recorded as as_recorded,
    dew_path as dew_path,
    dew_programs as dew_programs,
    holding,
    over_windows as over_windows,
    refuse_rounded,
    rounded_like,
    run_to_run as run_to_run,
    stacked_on_host as stacked_on_host,
    windows as windows,
    without_ynn as without_ynn,
)
from check1_judge import SWITCHED as SWITCHED, Verdict, _judged_pooled, _logged, _narrow, _verdicts
from check1_wire import QUALIFY
from float64_terms import evaluated_on, float64_terms, lowered, scan_groups
from map_judge import drawing, host_class, judging, record_units
from store_wire import COMPILED, judged, reading, running
from truth_sketch import SKETCH_DEVIATIONS, sketch_rounding, truth_sketch

from dew._accuracy import distance, reduction_bound, within
from dew.nn import programs
from dew.parity import (
    CONFIDENCE,
    FACTOR,
    FEWEST_SEEDS,
    LOWER_TAIL,
    MOST_SEEDS,
    NARROW_FACTORS as NARROW_FACTORS,
    PARITY,
    RECORDED_KINDS,
    ROW_FACTOR,
    SIZING_FROM,
    UPPER_TAIL,
    WRITTEN,
    Conditioning,
    Finding,
    GateFailure as GateFailure,
    Inconclusive,
    Paired,
    RecordedLeaf,
    _drawn_from,
    _Draws,
    _enough_seeds,
    _evidence,
    _exact,
    _example_ratios,
    _example_worst_rows,
    _interval,
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
    log_errors,
    recorded_kinds,
)
from dew.parity_stages import REDRAW, staged_units
from dew.telemetry.compiles import Compiles

PRECISION = "precision"
"""The entry of a reference's array file that records the precision its generator's matmuls took."""

_log = logging.getLogger(__name__)


def ieee_fixture(path) -> dict[str, np.ndarray]:
    """The arrays of a reference's array file, refused unless its generator
    recorded IEEE matmuls under `PRECISION`.

    The rule above takes the reference's error as its budget. A reference
    regenerated with TF32 matmuls, ten mantissa bits, sits farther from
    float64 and so widens the budget to fit a coarser Dew, and the float32
    arrays it stores look the same; only the generator's record tells."""
    arrays, recorded = _stored_precision(path)
    if recorded != "ieee":
        raise ValueError(f"{path} records {recorded!r} for its matmuls, not 'ieee'; regenerate it "
                         f"with IEEE matmuls")
    return arrays


def upstream_fixture(path) -> dict[str, np.ndarray]:
    """The arrays of a port's upstream recording, refused unless its generator recorded the upstream at its
    own default precision (`_dew_torch.precision.UPSTREAM_DEFAULT`): a port multiplies as its upstream does
    at the default precision (fla's Triton dots an fp32 product as one TF32), so its reference is the
    upstream's own arithmetic, like with like, never one a switch made IEEE (`ieee_fixture`'s, every other
    reference's)."""
    from _dew_torch.precision import UPSTREAM_DEFAULT

    arrays, recorded = _stored_precision(path)
    if not recorded.startswith(UPSTREAM_DEFAULT):
        raise ValueError(f"{path} records {recorded!r} for its matmuls, not its upstream at its own default "
                         "precision; regenerate it with no precision switch set")
    return arrays


def _stored_precision(path) -> tuple[dict[str, np.ndarray], str]:
    """A reference's array file's arrays and what its generator recorded of its matmuls (`PRECISION`)."""
    with np.load(path) as stored:
        arrays = {name: stored[name] for name in stored.files}
    return arrays, str(arrays.pop(PRECISION)) if PRECISION in arrays else "no precision"


class Rounded(NamedTuple):
    """An arm's result at a site of followed roundings (`assert_as_exact`'s `followed`): the leaves check 1
    judges (`judged`), every rounding's input among them, and the decisions its roundings made
    (`decisions`, each rounding's output as the arm computed it), which its own truth follows and the
    gate holds to the site's rounding of the judged inputs (`Followed.rounds`)."""

    judged: Any
    decisions: Any


class _RecordedRounded(Rounded):
    """A recording's arm at a site of followed roundings (`indexed_recording`): its decisions the
    reference's own, which its recorded truth followed (`_recorded_following`); no live arm's."""


class Followed(NamedTuple):
    """A quantized computation's truth (`assert_as_exact`'s `followed`, docs/design/testing.md): `truth`,
    the float64 computation that rounds where an arm rounded, of its decisions and the draw's inputs; and
    `rounds`, the site's roundings of an arm's judged leaves (the values its roundings rounded, judged by
    check 1 against the truth), which every live arm's decisions must equal in structure, shape, dtype
    and bits, so no arm's truth follows a decision its roundings did not make."""

    truth: Callable
    rounds: Callable


class UnroundedDecisions(AssertionError):
    """A live arm's decisions at a site of followed roundings that the site's roundings of its judged
    leaves do not make (`Followed.rounds`)."""


def _legal(rounds, result: Rounded, label: str) -> None:
    """`result`'s decisions held, in structure, shape, dtype and bits (`_same_bits`), to `rounds` of its
    judged leaves: one source of the values it rounded, which check 1 judges."""
    made = jax.device_get(rounds(result.judged))
    where = _same_bits(made, jax.device_get(result.decisions))
    if where is not None:
        raise UnroundedDecisions(f"{label}: the decisions at {where} are not the site's rounding of the "
                                 f"values the arm rounded")


def _followed_truth(prepared, followed: Followed, result, inputs):
    """An arm's own truth at a site of followed roundings: the site's truth, as the gate runs a truth
    (`_truth_program`, `prepared`), of the arm's decisions (`Rounded`) and the draw's `inputs` widened to
    float64, the float64 computation that rounds where that arm rounded, each of a live arm's decisions
    first held to the site's rounding of what it rounded (`_legal`); a recording's arm's are the
    reference's own (`_RecordedRounded`)."""
    if not isinstance(result, Rounded):
        raise AssertionError("an arm at a site of followed roundings returns its judged leaves and its "
                             "roundings' decisions (`Rounded`)")
    if not isinstance(result, _RecordedRounded):
        _legal(followed.rounds, result, "a followed arm")
    with jax.enable_x64(new_val=True):
        return prepared(jax.device_get(result.decisions), *widened(inputs))


def _truth_read(truth_fn, followed, result):
    """The truth `_of_float64` reads a site's program of: `truth_fn`, or at a site of followed roundings
    (`result` Dew's `Rounded`) `followed` bound to Dew's decisions, as the gate runs a truth
    (`_truth_program`), as given where `followed` runs on the host (a recording's, `indexed_recording`);
    unrewritten either way."""
    if result is None:
        return unrewritten(truth_fn)
    bound = functools.partial(followed.truth, jax.device_get(result.decisions))
    return _truth_program(as_given(bound) if _given(followed.truth) else bound)


def _own_truth(prepared, followed, result, inputs, truth, arm):
    """`arm`'s result and its truth on `inputs`: at a site of followed roundings (`prepared`, its truth as
    the gate runs it) its judged leaves and the truth of its own decisions (`_followed_truth`), each live
    arm's a `Rounded` (a reference or a further arm returning bare leaves would be held to Dew's truth with
    no decision of its own: spec alignment, 8999); a recording's its recorded decisions where it holds them,
    else its leaves held to Dew's; at any other site the two as given."""
    if isinstance(result, Rounded):
        return result.judged, _followed_truth(prepared, followed, result, inputs)
    if prepared is not None and not getattr(arm, "recorded", False):
        raise AssertionError("a live arm at a site of followed roundings returns its judged leaves and its "
                             "roundings' decisions (`Rounded`)")
    return result, truth


def _truth_program(fn, device=None):
    """`fn`, a check's float64 truth, as the gate runs it, unrewritten (`arm_arithmetic.unrewritten`):
    compiled, once a signature, so the truth runs as one float64 program on the device's FP64 units rather
    than one eager dispatch an equation, every primitive at its own float64 semantics (`Truth`), with XLA's
    passes, whose rewrites move a float64 value within its own rounding (a reference's twin as written is a
    float32 arm's bound, `_written_twin`; a whole model's float64 truth compiled as written took 30 times its
    run, test_edm2's 1755 s against 57). A truth the gate runs as given (`as_given`: NumPy, torch, a
    recording, a host loop), one of Dew's own programs (`dew_programs`), or none, as it is (`_holds_program`);
    one compiled already (a jit) as it is too, but on a named `device` (a GPU run's sizing's, the host's:
    `_qualified`), where it is that device's program as a plain one is, and one placed already stays where it
    is; a window loop's by its own. A truth with host work inside that does not state it fails at its trace,
    loudly, and passes `as_given`."""
    if isinstance(fn, Truth):
        return fn if device is None or fn.device is not None else Truth(fn.fn, device)
    if isinstance(fn, _Windowed):
        return _Windowed(_truth_program(fn.fn, device))
    if fn is None or not _holds_program(fn) or (device is None and hasattr(fn, "lower")):
        return unrewritten(fn)
    return Truth(fn, device)


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


def equations(graph, *, platform: str | None = None, skip=None, prune=None):
    """Every equation of a jaxpr, its sub-jaxprs' (scan, cond, custom rules) in turn. With `platform`
    (a lowering platform's name: 'cpu', 'cuda', 'tpu'), a platform-dependent cond
    (`jax.lax.platform_dependent`) gives only the branch that platform runs, as lowering keeps it. With
    `skip`, an equation it holds and every equation within it are passed over; with `prune`, an equation it
    holds is given and the equations within it passed over (a kernel's launch, its outputs the memory the
    program reads and writes, never its body's)."""
    if hasattr(graph, "eqns"):
        for equation in graph.eqns:
            if skip is not None and skip(equation):
                continue
            yield equation
            if prune is not None and prune(equation):
                continue
            kept = _platform_branch(equation, platform)
            yield from equations(equation.params if kept is None else kept, platform=platform, skip=skip,
                                 prune=prune)
    elif hasattr(graph, "jaxpr"):
        yield from equations(graph.jaxpr, platform=platform, skip=skip, prune=prune)
    elif isinstance(graph, dict):
        for value in graph.values():
            yield from equations(value, platform=platform, skip=skip, prune=prune)
    elif isinstance(graph, (tuple, list)):
        for value in graph:
            yield from equations(value, platform=platform, skip=skip, prune=prune)


def _ulp(magnitude: np.ndarray, dtype) -> np.ndarray:
    """The spacing of `dtype`'s values at each `magnitude` (a float64 array): its binade's, or its least
    subnormal's below its normals (`jnp.finfo`, which knows bfloat16)."""
    info = jnp.finfo(dtype)
    exponent = np.frexp(np.maximum(magnitude, float(info.tiny)))[1]
    return np.maximum(np.ldexp(1.0, exponent - 1 - info.nmant), float(info.smallest_subnormal))

class EvaluatedDraw(NamedTuple):
    """One seed's inputs and host leaves (`assert_as_exact(..., returns=True)`): Dew's leaves with paths,
    the primary reference's leaves and the float64 truth's leaves."""

    inputs: tuple
    dew: list[tuple[tuple, np.ndarray]]
    reference: list[np.ndarray]
    truth: list[np.ndarray]


def _record_draw(captured: dict[int, EvaluatedDraw] | None, seed: int, limit: int,
                 inputs, dew, other, truth) -> None:
    """Store one draw's host leaves in `captured` when `seed < limit`, for explicit downstream checks."""
    if captured is None or seed >= limit or seed in captured:
        return
    captured[seed] = EvaluatedDraw(
        inputs,
        [(path, np.asarray(leaf)) for path, leaf in jax.tree_util.tree_leaves_with_path(jax.device_get(dew))],
        [np.asarray(leaf) for leaf in jax.tree.leaves(jax.device_get(other))],
        [np.asarray(leaf) for leaf in jax.tree.leaves(jax.device_get(truth))],
    )


def _grouped(mine: np.ndarray, theirs: np.ndarray, exact: np.ndarray,
             held: np.ndarray | None = None):
    """Finite entries of 1D `[N]` or 2D `[R, N]` row arrays sorted by `(row, mine)` with group `starts`."""
    mine, theirs, exact = (np.atleast_2d(x) for x in (mine, theirs, exact))
    rows, cols = mine.shape
    mask = np.isfinite(mine) & np.isfinite(theirs) & np.isfinite(exact)
    if held is not None:
        held = np.atleast_2d(held)
        mask &= np.isfinite(held)
    flat = mask.ravel()
    if not flat.any():
        return None
    row = np.repeat(np.arange(rows, dtype=np.intp), cols)[flat]
    m, t, e = mine.ravel()[flat], theirs.ravel()[flat], exact.ravel()[flat]
    order = np.lexsort((m, row))
    row, m, t, e = row[order], m[order], t[order], e[order]
    starts = np.flatnonzero(np.concatenate([[True], (row[1:] != row[:-1]) | (m[1:] != m[:-1])]))
    h = None if held is None else held.ravel()[flat][order]
    return starts, m.size, t, e, h


def _has_candidate_tie(mine: np.ndarray, theirs: np.ndarray, exact: np.ndarray, dtype) -> bool:
    """Whether any row in `[R, N]` holds a group equal in `mine` and apart in `theirs` that can fail
    `_merged`: for pairs, truths spread past the plain bound; for 3+ equal entries, also past the minimum
    subset bound in case `rounded_fn` yields a non-finite entry on part of the group."""
    grouped = _grouped(mine, theirs, exact)
    if grouped is None:
        return False
    starts, size, kept, true, _ = grouped
    spread_kept = np.maximum.reduceat(kept, starts) - np.minimum.reduceat(kept, starts)
    if not np.any(spread_kept > 0):
        return False
    spread_true = np.maximum.reduceat(true, starts) - np.minimum.reduceat(true, starts)
    err = np.abs(kept - true)
    abs_true = np.abs(true)
    plain = (spread_true > 2 * np.maximum.reduceat(err, starts)
             + _ulp(np.maximum.reduceat(abs_true, starts), dtype))
    counts = np.diff(np.append(starts, size))
    subset = (counts >= 3) & (spread_true > 2 * np.minimum.reduceat(err, starts)
                              + _ulp(np.minimum.reduceat(abs_true, starts), dtype))
    return bool(np.any((spread_kept > 0) & (plain | subset)))


def assert_merges_no_ties(dew_fn, reference_fn, truth_fn, draw, label: str, *, seeds: int,
                          along: int = 1, rounded_fn=None,
                          evaluated: Mapping[int, EvaluatedDraw] | None = None) -> None:
    """A kernel's ties (docs/design/testing.md, fast and accurate forms: the accurate form's claim) over
    `seeds` draws, among a leaf's entries along `along` (an attention leaf's positions, `[B, S, H, D]`'s
    second axis), every other coordinate held: no group of them `dew_fn` returns equal that the reference
    keeps apart, its values (`reference_fn`'s) not all equal and their float64 truths (`truth_fn` on the
    inputs widened) spread past twice the reference's own error over the group (its largest distance from
    the truth: two entries each that far off may meet) and one ulp more, the ulp of the group's largest
    magnitude in the leaf's dtype (each entry's rounding to its dtype moves it at most half its own ulp);
    and, where the site states the roundings any computation in its format makes (`rounded_fn`: the float64
    truth with those intermediates each correctly rounded to the format, an attention's logits), whose
    rounded truths spread past one ulp too. Entries of other rows (another batch, head or channel) are no
    pair the claim holds: equal by chance in a leaf of many, they are no tie its computation made (row
    6302's collisions across batches). A pair a correctly rounded computation merges is no pair the
    reference keeps apart but by its own error: keys an ulp apart whose float64 logits round to one fp32
    logit, which Dew's correctly rounded logits share and the reference's, 0.55 ulps off, split, their dV
    6 ulps apart at the truth (row 6302). When `evaluated` comes from `assert_as_exact(..., returns=True)`,
    its per-seed leaves are read directly without re-running the arms; `rounded_fn` is jitted and evaluated
    lazily only on seeds where some leaf has a candidate group."""
    truth_of = None
    rounded_of = None
    for seed in range(seeds):
        if evaluated is not None and seed in evaluated:
            inputs, found, theirs_all, truth = evaluated[seed]
        else:
            inputs = draw(seed)
            if truth_of is None:
                truth_of = jax.jit(truth_fn)
            with jax.enable_x64(new_val=True):
                truth = jax.tree.leaves(truth_of(*widened(inputs)))
            found = jax.tree_util.tree_leaves_with_path(dew_fn(*inputs))
            theirs_all = jax.tree.leaves(reference_fn(*inputs))
        laid = [(path, np.asarray(mine).dtype,
                 _rows_along(np.asarray(mine, np.float64), along),
                 _rows_along(np.asarray(theirs, np.float64), along),
                 _rows_along(np.asarray(exact, np.float64), along))
                for (path, mine), theirs, exact in zip(found, theirs_all, truth, strict=True)]
        if rounded_fn is not None:
            if not any(_has_candidate_tie(mine_r, theirs_r, exact_r, dtype)
                       for _, dtype, mine_r, theirs_r, exact_r in laid):
                continue
            if rounded_of is None:
                rounded_of = jax.jit(rounded_fn)
            with jax.enable_x64(new_val=True):
                rounded_leaves = jax.tree.leaves(rounded_of(*widened(inputs)))
            held_rows = [_rows_along(np.asarray(x, np.float64), along) for x in rounded_leaves]
        else:
            held_rows = [exact_r for _, _, _, _, exact_r in laid]
        for (path, dtype, mine_r, theirs_r, exact_r), held_r in zip(laid, held_rows, strict=True):
            count, widest = _merged(mine_r, theirs_r, exact_r, held_r, dtype)
            assert not count, (f"{label}{jax.tree_util.keystr(path)} seed {seed}: {count} groups of equal "
                               f"values the reference keeps apart, the widest {widest:.0f} ulps")


def _rows_along(x: np.ndarray, along: int) -> np.ndarray:
    """`x`'s entries as rows along axis `along`, every other coordinate a row of its own (`[rows, along]`); a
    leaf of no such axis one row."""
    if x.ndim <= along:
        return x.reshape(1, -1)
    return np.moveaxis(x, along, -1).reshape(-1, x.shape[along])


def _merged(mine: np.ndarray, theirs: np.ndarray, exact: np.ndarray, held: np.ndarray,
            dtype) -> tuple[int, float]:
    """The ties Dew merges across rows `[R, N]` (or a 1D row `[N]`) that the reference keeps apart
    (`assert_merges_no_ties`), `held` its truths with the format's own roundings (`rounded_fn`'s, else the
    truths): their count and the widest's truths' spread in ulps."""
    grouped = _grouped(mine, theirs, exact, held)
    if grouped is None:
        return 0, 0.0
    starts, _, kept, true, rounded = grouped
    assert rounded is not None

    def spread(x):
        return np.maximum.reduceat(x, starts) - np.minimum.reduceat(x, starts)

    ulp = _ulp(np.maximum.reduceat(np.abs(true), starts), dtype)
    merged = ((spread(kept) > 0) & (spread(true) > 2 * np.maximum.reduceat(np.abs(kept - true), starts) + ulp)
              & (spread(rounded) > ulp))
    return int(merged.sum()), float(np.max(spread(true)[merged] / ulp[merged], initial=0.0))

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
    CPU arithmetic takes (`dew.nn.precision.float64_sum`) multiplies float64. A selection's scoring, which
    ranks keys rather than weighing them (`dew.nn.sparse_selection.SELECTING`), is no logits' product."""
    graph = jax.make_jaxpr(function)(*arguments)
    found = set()
    for equation in equations(graph, skip=_selecting):
        if equation.primitive is not jax.lax.dot_general_p:
            continue
        # Its contracted axes' lengths, which its two operands share.
        (contracted, _), _ = equation.params["dimension_numbers"]
        if [equation.invars[0].aval.shape[axis] for axis in contracted] == [width]:
            found.add(jnp.promote_types(*(variable.aval.dtype for variable in equation.invars)))
    return found


def _selecting(equation) -> bool:
    """Whether `equation` runs within a selection's scoring scope (`dew.nn.sparse_selection.SELECTING`)."""
    from dew.nn.sparse_selection import SELECTING

    return any(getattr(scope, "name", None) == SELECTING for scope in equation.source_info.name_stack.stack)


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


def _arm_name(arm, reference_fn, library_fn, at_arm: int) -> str:
    """How a refusal names one of a site's arms beside Dew's: the reference, the library arm, or a further
    reference by its place among the arms."""
    return ("the reference" if arm is reference_fn else "the library arm" if arm is library_fn
            else f"reference {at_arm}")


def _arm_ran(label: str, arm, dew_fn, dew, here, mine_at, true_here, name: str | None):
    """An arm's result on `here` beside Dew's (`dew`, on `mine_at`): Dew's own where the arm is Dew's on its
    inputs, check 1's own truth of `here` rounded once (`true_here`, `rounded_like`) where it is `ROUNDED`,
    else the arm run, and with a `name` (the draw's first, its own order) held to be a function of its inputs
    (`_of_its_inputs`), as every arm whose result enters the verdict is: the reference, the library arm and
    each further reference, each run again."""
    if getattr(arm, "rounded", False):
        return rounded_like(true_here, dew)
    other = dew if dew_fn is arm and here is mine_at else arm(*here)
    if name is not None and arm is not dew_fn:
        _of_its_inputs(label, name, arm, here, other)
    return other


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


def _paired_most(*arms, most: int = MOST_SEEDS, paired: bool = True) -> int | None:
    """The most seeds a sizing of a site's `arms` takes (`seeds_for`'s, its draws from SIZING_FROM on,
    `parity._sized_count`: twice paired, four times a reference compared with itself): `most`, or the
    fewest any recorded arm's pilots hold a share of; None where an arm is a recording without pilots (a
    judged recording, Dew's arm, among them), which no sizing takes (`_qualified`), its count its
    recording's. The one reading of the pair's pilot budget: a qualification's (`_qualified`), a sizing's
    (`seeds_for`) and a prepare's (`_sizing_seeds`). An arm of None (an unpaired sizing's Dew) is none."""
    held = most
    for fn in arms:
        if fn is None or not getattr(fn, "recorded", False):
            continue
        pilots = getattr(fn, "pilots", None)
        if pilots is None:
            return None
        held = min(held, len(pilots) // (2 if paired else 4))
    return held


def _sizing_seeds(dew_fn, reference_fn) -> range:
    """The seeds a qualification run's paired sizing of a site may draw (`_qualified`, `seeds_for`), none
    where it takes none (`_paired_most` of the pair)."""
    most = _paired_most(dew_fn, reference_fn)
    return range(0) if most is None else range(SIZING_FROM, SIZING_FROM + 2 * most)


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
    with (jax.enable_x64(new_val=True), arm_arithmetic.on_float64_device(),
          evaluated_on(jax.tree.leaves(wide)) as device):
        program = lowered(trace(*wide).jaxpr).jaxpr
    lowering = programs.lowering_platform(device)
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


def _over_orders(first: list[Paired], later: list[Paired]) -> list[Paired]:
    """An arm's evidence on a draw (`_evidence`) over two of its orders: each leaf's errors and rows of
    float64 zero the first order's, its worst rows the worse of the two, each worst row's shares
    (`Paired.shares`) those of the order it is, its values finite at both."""
    def over(one: Paired, other: Paired) -> Paired:
        later_worse = np.asarray([other.worst > one.worst, other.their_worst > one.their_worst])
        return one._replace(worst=float(np.fmax(one.worst, other.worst)),
                            their_worst=float(np.fmax(one.their_worst, other.their_worst)),
                            finite=one.finite and other.finite,
                            shares=np.where(later_worse, other.shares, one.shares))
    return [over(one, other) for one, other in zip(first, later, strict=True)]


def _conditioning(reference_fn, truth_fn, draw, seeds, ran=None, scale=None) -> Conditioning:
    """Check 1's conditioning of `reference_fn` (dew.parity's `Conditioning`): its rounding scale at a draw as
    its arm runs (`summed_as_run`, a window loop's by window), or `scale(inputs, seed)` where its program is
    not one here (`assert_as_exact`'s `conditioning_scale`), its heavy elements over `seeds` as `ran`, its
    compiled arm where given, runs it; the reference and its truth unrewritten, as their arms run. A
    recording that keeps no outputs gives its conditioning whole (`recorded_weighing`: the scale, the heavy
    elements and the reference's weighed evidence, each read when it was made), and nothing here is run."""
    from rounding_bound import rounding_scale
    if isinstance(scale, Conditioning):
        return scale
    scaled = reference_fn.fn if isinstance(reference_fn, _Windowed) else reference_fn

    def scale_of(inputs, seed):
        # A site's own scale too, in the arithmetic its arms run in (a Gemma 3n site's, stated without it,
        # weighed float64 CPU sums its arms never run: row 9780's debugger).
        with summed_as_run():
            if scale is not None:
                return scale(inputs, seed)
            if not isinstance(reference_fn, _Windowed):
                return rounding_scale(scaled, *inputs, seed=seed)[1]
            each = [rounding_scale(scaled, *jax.tree.map(lambda leaf, at=at: leaf[at], inputs), seed=seed)[1]
                    for at in range(_window_count(inputs))]
            return jax.tree.map(lambda *scale: None if scale[0] is None else np.stack(scale), *each,
                                is_leaf=lambda leaf: leaf is None)

    # Its heavy elements over `seeds`' draws, each run as its arm runs (`summed_as_run`) beside its float64
    # truth, `truth_fn`'s or its own, classed alike in every process that runs them.
    # The reference as its arm runs (under the rewrite where `dew_path`), its truth unrewritten.
    ran_fn = ran or reference_fn
    ran_fn, true_fn = ran_fn if _declared(reference_fn) else unrewritten(ran_fn), unrewritten(truth_fn)

    def ran_and_true(seed):
        inputs = draw(seed)
        with summed_as_run():
            ran_here = jax.device_get(ran_fn(*inputs))
        with jax.enable_x64(new_val=True), arm_arithmetic.on_float64_device():
            return ran_here, jax.device_get((true_fn or unrewritten(ran_fn))(*widened(inputs)))

    return Conditioning(scale_of, heavy_masks(map(ran_and_true, seeds)))


def _truth_and_terms(truth_fn, terms_fn, inputs):
    """The float64 truth of `inputs` (`truth_fn` on them widened, unrewritten, as every truth the gate runs:
    one taken from a reference of Dew's own path, `dew_path`, which runs under the rewrite, included) and,
    with `terms_fn`, its terms (else None): a site's own terms program's (`_SiteTerms`), or the truth's own
    evaluated op by op (`float64_terms`)."""
    truth_fn = unrewritten(truth_fn)
    if terms_fn is None or isinstance(terms_fn, _SiteTerms):
        with jax.enable_x64(new_val=True), arm_arithmetic.on_float64_device():
            truth = _truth_of(truth_fn, widened(inputs))
        if terms_fn is None:
            return truth, None
        with jax.enable_x64(new_val=True), arm_arithmetic.on_float64_device():
            return truth, terms_fn.of(truth, *widened(inputs))
    with jax.enable_x64(new_val=True), arm_arithmetic.on_float64_device():
        return terms_fn(*widened(inputs))


def _truth_of(truth_fn, wide):
    """`truth_fn`, a site's truth as the gate runs it, on a draw's widened inputs `wide`: by the program a
    store compiled ahead where a run reads one (`truths_ahead.running`), unless the gate runs the truth as
    given."""
    ran = running()
    if ran is None or _given(truth_fn):
        return truth_fn(*wide)
    if isinstance(truth_fn, _Windowed):
        # A window loop's truth by its window's program, window by window, as the loop runs it.
        return stacked_on_host([
            _truth_of(truth_fn.fn, jax.tree.map(lambda leaf, index=index: leaf[index], wide))
            for index in range(_window_count(wide))])
    store, label = ran
    return store.called(label, "truth", wide)


def _draw_truth(truth_and_terms, prepared, followed, dew_fn, inputs):
    """A draw's truth and its terms, and where the site follows roundings (`prepared`, its `followed` as
    the gate runs a truth) Dew's result (`Rounded`, else None): run first, its truth its own decisions'
    (`_followed_truth`), with no terms. Each reference's truth is then its own decisions' too, a recorded
    one's the recording's (`indexed_recording`), paired with Dew's rows only where the two are the same
    values."""
    if prepared is None:
        return (*truth_and_terms(inputs), None)
    with summed_as_run():
        ran = dew_fn(*inputs)
    return _followed_truth(prepared, followed, ran, inputs), None, ran


def _first_timed(timed: dict, rows_at: Callable, seed):
    """The first draw's rows (`rows_at(0, seed)`), its seconds and the backend compiles JAX reports within
    it written to `timed` (a jit compiles at its first call): what the process pays once, whatever it goes
    on to draw."""
    began = time.perf_counter()
    with Compiles(None) as compiles:
        rows = rows_at(0, seed)
    timed.update(first=time.perf_counter() - began, compiled=sum(compiles.seconds))
    return rows


def _ordered(orders, inputs, their_inputs) -> list[tuple]:
    """A draw's orders (`_drawn`'s `orders`), each with the reference's draw at that order: the draw's own
    alone where none are asked; an orbit holds the draw's own and another at least."""
    if orders is None:
        return [(inputs, their_inputs)]
    ordered = list(zip(orders(inputs), orders(their_inputs), strict=True))
    if len(ordered) < 2:
        raise ValueError(f"orders of {len(ordered)}; an orbit holds the draw's own and another")
    return ordered


def _drawn(dew_fn, reference_fn, draw, seeds, truth_fn=None, reference_seeds=None, orders=None,
           library_fn=None, library_leaves: Sequence[str] = (), also: Sequence = (), terms_fn=None,
           label: str = "", examples: bool = False, conditioning=None, timed: dict | None = None,
           followed=None, library_also: int | None = None, *, repeats: bool,
           captured: dict[int, EvaluatedDraw] | None = None, capture_seeds: int = 0) -> list[_Draws]:
    """Every leaf's errors (`_Draws`) over `seeds`, a list (the reference's, each further one's after): both
    functions run on `draw(seed)`'s inputs, truth `truth_fn` (else `reference_fn`) on them widened,
    unrewritten. With `reference_seeds`, the reference runs on those draws instead, each held to its own
    draw's truth (`null_runs`), and a row of float64 zero is paired only where the two truths are the same
    values (an orbit's draws). With `orders`, each arm also runs at every other order of its own draw
    (`orders(inputs)` past its first, the draw itself), held to that draw's one truth: each arm's worst row is
    its worst over the orders, measured rows of float64 zero included, and its values finite at every one; its
    errors and its rows of float64 zero are the draw's own order's. With `library_fn`, the library arm runs
    beside the reference on Dew's own inputs, held to Dew's truth, and the draws are judged against the larger
    of the two on `library_leaves` (`_Draws.against_larger`), Dew and the truth evaluated once. With `also`,
    further references run beside it on Dew's own inputs, each held to Dew's truth, Dew and the truth
    evaluated once, each one's draws after the reference's, the one at `library_also` (the reference's own as
    written, `_written_twin`) judged as the reference is. With `terms_fn`, the truth is that program's with
    each output's terms (`float64_terms`), every row sized by the larger of its own RMS and its terms'
    (`_LaidTruths.of`); with a site's own terms program (`_SiteTerms`), the truth is as without and its terms
    that program's, on the same widened inputs. With `conditioning` (dew.parity's `Conditioning`), every arm's
    leaves are read as residuals in units of the reference's rounding at its truth's inputs
    (`Weighed.weighed`). With `followed`, each arm is held to the truth of its own decisions (`_draw_truth`,
    `_followed_truth`). Its first draw is held to be its seed's and each arm a function of its inputs, run
    again on them, where `repeats`, which each driver states: its site's first draw, once a site, not a map
    task's or a later stage's past it (rows 7603, 7604, 10880); each null run's first draw is its own run's,
    a block every such run checks (ledger 2641). When `captured` is given, records draws `< capture_seeds`."""
    if orders is not None and any(getattr(fn, "recorded", False) for fn in (reference_fn, *also)):
        raise ValueError("a recorded reference holds its recorded draws alone, never another order of them")
    arms = [reference_fn] + ([] if library_fn is None else [library_fn]) + list(also)
    names, widths = None, None
    prepared_followed = None if followed is None else _truth_program(followed.truth)
    stacked, stacked_rows = ([None for _ in arms] for _ in range(2))
    evidence, per_example, per_example_rows = ([[] for _ in arms] for _ in range(3))
    truth_and_terms = functools.partial(_truth_and_terms, truth_fn or reference_fn, terms_fn)
    def rows_at(at, seed):
        """Every arm's evidence on the draw at `seed`, the `at`th, its orders' worst rows the larger."""
        nonlocal names, widths
        inputs = draw(seed)
        site_first = repeats and not at
        if site_first:
            _of_its_seed(label, draw, seed, inputs)
        # The site's arms' mode, decided on its first draw's inputs before any arm runs, no draw of its own,
        # on the programs the arms run (their scope, `summed_as_run`).
        with summed_as_run():
            _ynn_decided(label, [dew_fn, *arms], inputs)
        truth, terms, ran = _draw_truth(truth_and_terms, prepared_followed, followed, dew_fn, inputs)
        # Else held (`arm_store.compile_site`).
        if site_first and (running() is None or _given(truth_fn or reference_fn)):
            _of_float64(label, _truth_read(truth_fn or reference_fn, followed, ran), inputs, truth)
        if reference_seeds is None:
            their_inputs, their_truth, their_terms = inputs, truth, terms
        else:
            their_inputs = draw(reference_seeds[at])
            their_truth, their_terms = (None, None) if followed else truth_and_terms(their_inputs)
        # Each arm in units of the reference's rounding at its truth's inputs, the truth ones.
        weighed = None if conditioning is None else conditioning.at(
            ((truth, inputs, seed), (their_truth, their_inputs, seed if reference_seeds is None
                                     else reference_seeds[at])))
        ordered = _ordered(orders, inputs, their_inputs)
        rows = [None for _ in arms]
        stacked[:] = [None for _ in arms]
        stacked_rows[:] = [None for _ in arms]
        for mine_at, theirs_at in ordered:
            # Each arm, and its replay on the same inputs, in the arithmetic arms run in (`summed_as_run`).
            with summed_as_run():
                dew = dew_fn(*mine_at) if ran is None else ran
                if site_first and mine_at is ordered[0][0]:
                    _of_its_inputs(label, "Dew's arm", dew_fn, mine_at, dew)
            dew = dew if ran is None else ran.judged
            if (held := HELD.get()) is not None:
                held(dew, mine_at)
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
                    name = (_arm_name(arm, reference_fn, library_fn, at_arm)
                            if site_first and mine_at is ordered[0][0] else None)
                    other = _arm_ran(label, arm, dew_fn, dew, here, mine_at, true_here, name)
                other, true_here = _own_truth(prepared_followed, followed, other, here, true_here, arm)
                if at_arm == 0 and mine_at is ordered[0][0]:
                    _record_draw(captured, seed, capture_seeds, inputs, dew, other, true_here)
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
                evidence = _evidence(prepared, other, true_here, shared=shared, their_terms=terms_here,
                                     own=followed is not None)
                rows[at_arm] = evidence if rows[at_arm] is None else _over_orders(rows[at_arm], evidence)
        return rows

    for at, seed in enumerate(seeds):
        rows = rows_at(at, seed) if timed is None or at else _first_timed(timed, rows_at, seed)
        for at_arm, row in enumerate(rows):
            evidence[at_arm].append(row)
            per_example[at_arm].append(stacked[at_arm])
            per_example_rows[at_arm].append(stacked_rows[at_arm])
    drawn = [_drawn_from(names, widths, evidence[at_arm], per_example[at_arm] if examples else None,
                        per_example_rows[at_arm] if examples else None) for at_arm in range(len(arms))]
    first = drawn[0] if library_fn is None else drawn[0].against_larger(drawn[1], library_leaves)
    return [first, *(draws if at != library_also else draws.against_larger(drawn[1], library_leaves)
                     for at, draws in enumerate(drawn[len(arms) - len(also):]))]


@dataclasses.dataclass(frozen=True)
class _SiteTerms:
    """A site's own program of its truth's terms (`assert_as_exact`'s `terms`): `fn` on a draw's inputs
    widened to float64, the truth's tree of each entry's terms as `float64_terms` sizes them."""

    fn: Callable

    def of(self, truth, *inputs):
        """The terms of `truth`, the draw's at `inputs`, held to its leaves, paths and shapes: by the program
        a store compiled ahead where a run reads one (`truths_ahead.running`)."""
        ran = running()
        terms = jax.device_get(self.fn(*inputs) if ran is None
                               else ran[0].called(ran[1], "site terms", inputs))
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
    run it (`_truth_program`: a Truth, or a truth compiled already, a jit or Dew's own programs), evaluates
    the first draw's truth, so its first call's run and compiles serve the draws too, once a process (a
    map task's draws, DRAWS): a truth compiled apart here compiled the float64 truth twice a site
    (pass_weights' 10 s). None, a reference standing for it, runs compiled (`_truth_program`), not eagerly
    (ledger 3774)."""
    if truth is None or isinstance(truth, _Windowed) or _given(truth) or getattr(truth, "recorded", False):
        return None
    # The truth on the site's first draw, and its terms only where it holds a zero row: only a zero row whose
    # terms are not zero too gains a size, so a site whose zero rows sum nothing (a query that keeps no key)
    # keeps the reference's residue rule without tracing its terms a draw.
    run = program or _truth_program(truth)
    # The draw as every seed's is drawn, outside float64, then widened (under x64 a draw's own defaults
    # would draw float64 values the arms never see).
    inputs = widened(draw(0))
    with jax.enable_x64(new_val=True), arm_arithmetic.on_float64_device():
        leaves = [*map(np.asarray, jax.tree.leaves(_truth_of(run, inputs)))]
        zero_leaves = [index for index, leaf in enumerate(leaves)
                       if np.any(_LaidTruths.of([leaf]).sizes == 0)]
        if not zero_leaves:
            return None
        # Only these leaves can change the decision below: a terms program of every output compiled each
        # leaf's sums too (the SSD mesh contract's data mesh 205 s, 78 s on its zero leaf alone; rows 5770,
        # 5961). A run reading programs compiled ahead reads every leaf's, its prepare's one walk.
        def selected(*args):
            outputs = jax.tree.leaves(truth(*args))
            return [outputs[index] for index in zero_leaves]

        if running() is None:
            truths, terms = float64_terms(selected, *inputs)
        else:
            whole, every = float64_terms(truth, *inputs)
            truths, terms = ([jax.tree.leaves(tree)[index] for index in zero_leaves]
                             for tree in (whole, every))
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


def _recorded(label: str, drawn_for, verdict: "Verdict", arms, upto: int | None = None,
              weighed: dict | None = None, offset: int = 0, first: dict | None = None) -> bool:
    """A map task's draws (`record_units`): each of a site's `references` draws over its range, up to
    `upto` where given, with its host's class (`map_judge.host_class`), its arms' YNNPACK mode (`_mode_of`)
    and, weighed, its classes' digest
    (`Conditioning.digest`), the seconds its weighing took beside its draws', and those of its first draw
    with the backend compiles JAX reported within it (`first`, `compiled`, `_drawn`'s): a site's arms
    compile at their first call, so a task's first draw holds its compiles, a cost it pays once whatever
    it draws, which the plan must not read as each draw's (tools/armada/costs.py's `timed`; a site cut to
    two draws a task paid its compiles 128 times, row 5492). The test's manifest notes the site with its
    `verdict`, which its judge reads (`judged_from_records`). False where the process is no map task."""
    def payload(start, stop):
        draws = [drawn_for(index)(start, stop) for index in range(len(verdict.labels))]
        held = (weighed or {}).get("of")
        return {"draws": draws, "refused": _mode_of(arms), "heavy": None if held is None else held.digest(),
                "host": host_class()}
    return record_units(label, payload, upto=upto, offset=offset, verdict=verdict,
                        phases=lambda: {"weighing": (weighed or {}).get("seconds", 0.0), **(first or {})})

def _prepare_site(dew_fn, reference_fn, draw, label: str, *, seeds: int, truth_fn, orders, library_fn,
                  also, terms, conditioning, followed, returns: bool) -> tuple[bool, bool]:
    """Validate `assert_as_exact` options and handle `store.prepare`; return `(plain, done)`."""
    # A plain site has no library arm, further reference, custom terms program, or weighed check beside Dew
    # and its reference (`arm_store.compile_site`, row 9044).
    plain = not (library_fn or also or callable(terms) or conditioning)
    refuse_rounded(label, reference_fn, *(also or {}).values(), truth_fn=truth_fn, followed=followed,
                   conditioning=conditioning)
    if returns and (conditioning is not None or followed is not None
                    or getattr(reference_fn, "recorded", False)):
        raise ValueError(f"{label}: returns=True is unsupported with conditioning, followed, or recorded "
                         "references")
    store = COMPILED.get()
    if store is not None and conditioning is not None:
        raise ValueError(f"{label}: a weighed site reads its reference's own rounding at its truth's inputs, "
                         "which no program compiled ahead holds")
    if store is not None and store.prepare:
        import tier_wire
        from arm_store import compile_site

        try:
            tier_wire.stop_past_cap()
            compile_site(store, dew_fn, reference_fn, draw, label, seeds=seeds, truth_fn=truth_fn,
                         terms=terms, also=also, orders=orders, plain=plain, library_fn=library_fn)
        except tier_wire.PastCap:
            # Past its prepare's cap (tests/tier_wire.py's `PREPARE_BY`), before the site or between its
            # draws: named unprepared by its label (a flow's cell may hold many, as test_norm_swiglu's
            # configs), its run compiling it on its GPU; the sites before it kept.
            from truths_ahead import reasoned

            test = os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" (", 1)[0]
            reasoned(store.directory / tier_wire.UNPREPARED, test, tier_wire.CAPPED,
                     tier_wire.capped("the site"), label)
        return plain, True
    return plain, False


def assert_as_exact(dew_fn, reference_fn, draw, label: str, *, seeds: int, truth_fn=None,
                    orders=None, library_fn=None, library_row: int | None = None,
                    library_leaves: Sequence[str] = (), also: Mapping[str, Callable] | None = None,
                    terms: bool | Callable | None = None, examples: bool = False,
                    conditioning: int | None = None, conditioning_scale: Callable | None = None,
                    followed: Callable | None = None, held: Callable | None = None,
                    returns: bool = False) -> dict[int, EvaluatedDraw] | None:
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
    two Dew paths without YNNPACK's fusions, a Dew arm beside an arm run `as_given` with them. A reference
    written apart from Dew, an accuracy claim's, is compiled as written too (`_written_twin`), Dew's arm held
    within PARITY to the power WRITTEN of it, each of check 1's factors so raised; one of Dew's own paths
    (`dew_path`, `dew_programs`) is not.

    `orders(inputs)` lists a draw's inputs at K orders of an orbit of summation orders, the draw's own first
    (`_judged_site`). `ROUNDED` as the reference (`arm_forms.ROUNDED`, with `truth_fn`) is check 1's own
    float64 truth of each draw rounded once, no evaluation of its own, judged by parity as any reference is:
    the reference of draws whose truth float32 holds exactly or rounds once (rows 9481, 9515, 9532).
    `held(outputs, inputs)`, where given, is called on every result of Dew's arm a draw of the site's makes
    (its stages, a sizing's and a map task's draws, a prepared run's), its AssertionError the site's failure.

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
    never Dew's arm's, whose own forms would weigh its own amplified errors down. `followed` (a
    `Followed`) is a quantized computation's truth, each arm's own (`_followed_site`).

    This is an engineering gate, deterministic and reproducible, not a guarantee against every error
    law: "never wrong" is check 2's, each element within its operation's certified bound
    (tests/certified.py)."""
    terms = terms if followed is None else _followed_site(label, truth_fn, orders, library_fn, terms,
                                                          conditioning, examples)
    plain, prepared = _prepare_site(dew_fn, reference_fn, draw, label, seeds=seeds, truth_fn=truth_fn,
                                    orders=orders, library_fn=library_fn, also=also, terms=terms,
                                    conditioning=conditioning, followed=followed, returns=returns)
    if prepared:
        return None
    captured: dict[int, EvaluatedDraw] | None = {} if returns and not (drawing() or judging()) else None
    with holding(held):
        if os.environ.get(QUALIFY) and not (drawing() or judging()):
            _qualified(dew_fn, reference_fn, draw, label, seeds=seeds, truth_fn=truth_fn, orders=orders,
                       examples=examples, plain=plain, followed=followed)
        with judged(label):
            _judged_site(dew_fn, reference_fn, draw, label, seeds=seeds, truth_fn=truth_fn, orders=orders,
                         library_fn=library_fn, library_row=library_row, library_leaves=library_leaves,
                         also=also, terms=terms, examples=examples, conditioning=conditioning,
                         conditioning_scale=conditioning_scale, followed=followed, captured=captured)
    return captured


def _judged_site(dew_fn, reference_fn, draw, label: str, *, seeds: int, truth_fn, orders, library_fn,
                 library_row, library_leaves, also, terms, examples, conditioning, conditioning_scale,
                 followed, captured: dict[int, EvaluatedDraw] | None = None) -> None:
    """`assert_as_exact`'s site judged, deferred to the weighed check where it states `conditioning`: its
    truths computed by the programs a store compiled ahead where one is read (`truths_ahead.running`).

    Where a computation has an orbit of summation orders, `orders(inputs)` lists a draw's inputs at K
    of them, the draw itself first: an exact symmetry of the result (a batch's rows in another order
    under a mean, a sum's terms in another), so one truth holds at every order. Both arms run at every
    one (`_drawn`), and each one's worst row is its worst over them, compared alike; the pooled ratio,
    the rows of float64 zero and the sizing stay the draw's own order's. A row that nearly cancels
    rounds worst at some orders and well at others, so one order's worst row is no measure of it, and
    the reference's luckiest order would fail an equally exact Dew; over the orbit Dew's worst is held
    within ROW_FACTOR of the reference's worst at any of the same orders, which is as strict as the
    reference is spread over them: a Dew worse than the reference at every order still fails, and one
    the reference is exact against at every order fails at any order it is not."""
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
                 terms=terms, examples=examples, weighing=None, upto=staged_units(seeds), followed=followed,
                 captured=captured)

def _followed_site(label: str, truth_fn, orders, library_fn, terms, conditioning, examples) -> bool:
    """The terms of a site of followed roundings, none (`assert_as_exact`'s `followed`, testing.md): each
    arm, further references too, returns its leaves, every rounding's input among them, and its roundings'
    decisions (`Rounded`), and is held to the float64 computation that rounds where it rounded. A decision
    is no arm's error, but each rounding's input is a leaf held to the float64 value it rounds, so an
    arithmetic error before a rounding is charged where the decision it led to is followed. Refused with
    anything that reads one truth for every arm (a `truth_fn`, `orders`, a library arm, `terms`,
    `conditioning`, `examples`) or a prepared truth."""
    if (truth_fn is not None or orders is not None or library_fn is not None or terms
            or conditioning is not None or examples):
        raise ValueError(f"{label}: a site of followed roundings holds each arm to the truth of its own "
                         f"decisions, with none of truth_fn, orders, a library arm, terms, conditioning or "
                         f"examples")
    if COMPILED.get() is not None:
        raise ValueError(f"{label}: a followed truth is the arm's own, which no program compiled ahead holds")
    return False


def _deferred(dew_fn, reference_fn, draw, label: str, *, seeds: int, conditioning: int, truth_fn, orders,
              library_fn, library_row, library_leaves, also, terms, examples: bool, scale=None) -> None:
    """`assert_as_exact`'s site with `conditioning` (dew.parity's `Conditioning`): the unweighted check over
    its own `seeds`, whole, and where it stays undecided the weighed check over `conditioning` seeds, its
    heavy elements classed over the draws it takes, alike in every process that runs it. A sharded site's map
    tasks draw each check's units apart, the weighed check's past the unweighted one's in the plan's units
    (tools/armada/ci.py `draws_of`), and its judge pools the weighed check's where it defers."""
    plain, span = staged_units(seeds), staged_units(conditioning)

    def weighed():
        try:
            _site_judged(dew_fn, reference_fn, draw, f"{label}, conditioned", seeds=conditioning,
                         truth_fn=truth_fn, orders=orders, library_fn=library_fn, library_row=library_row,
                         library_leaves=library_leaves, also=also, terms=False, examples=examples,
                         weighing=functools.partial(_conditioning, draw=draw, seeds=range(span), scale=scale),
                         upto=span, offset=plain)
        finally:
            # Its weighing's programs, a test's every conditioned site's, held past it alone (row 9780: a
            # task weighing 20 sites of Gemma 3n's loop kept each site's and was killed at 10 GiB).
            from rounding_bound import released_scales

            released_scales()

    try:
        _site_judged(dew_fn, reference_fn, draw, label, seeds=seeds, truth_fn=truth_fn, orders=orders,
                     library_fn=library_fn, library_row=library_row, library_leaves=library_leaves,
                     also=also, terms=terms, examples=examples, weighing=None, upto=plain,
                     defers_to=f"{label}, conditioned")
    except Inconclusive:
        weighed()
    else:
        if drawing():
            weighed()


class StaleSeeds(AssertionError):
    """A site whose declared seeds are fewer than its own sizing takes where it runs (`_qualified`): the count
    was sized on other arithmetic, an arm's rounding having changed since (8eff72712a6 moved one site's from 8
    to 256 and two others' down), or on another host level."""


def _qualified(dew_fn, reference_fn, draw, label: str, *, seeds: int, truth_fn, orders, examples: bool,
               plain: bool, followed=None) -> None:
    """A qualification run's sizing of one site (`QUALIFY`) before its verdict: the count `seeds_for` sizes
    here, Dew paired with its reference as the verdict pairs them, on sizing seeds no verdict draws, as every
    site's count is sized (without its terms: `seeds_for` reads none); above the site's declared `seeds` it
    raises StaleSeeds. Each site appends a line to the run's file (`QUALIFY`'s value): its test, its label,
    its declared count and the count sized, or why it is not sized: either arm is a recording without
    pilots (a judged recording is Dew's arm), whose count is its recording's, or its verdict is not the
    `plain` pair's (a library arm, further references, terms of its own, a weighed check), which this sizing
    does not judge. A site whose draw
    cannot draw a sizing seed (one that indexes a fixed set of draws) fails with its draw's own error: it has
    no pilot stream, so its count cannot be sized again (`seeds_for`'s contract). A site judged under a
    store of programs compiled ahead (`truths_ahead.compiled_truths`: a GPU flow's) sizes in its run, its
    sizing draws' truths computed here from its own `truth_fn`, on the host as a sizing's are; none sizes in
    a prepare, a map task or its judge, its arms the programs its prepare compiled at its sizing draws'
    shapes (`_routed_arms`), never compiled in the run."""

    record = {"site": os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" (", 1)[0], "label": label,
              "declared": seeds}
    recorded = _paired_most(dew_fn, reference_fn) is None
    if not plain or recorded:
        record["unsized"] = "its verdict is not the plain pair's" if not plain else "a recording's count"
    else:
        # Its sizing draws' truths its own, never the store's programs', computed on the host under one (a GPU
        # run's): the GPU's memory is the arms'.
        store = COMPILED.get()
        arms = None
        if store is not None:
            # Its sizing draws' truths the host's: a truth function's, else its reference's widened, the
            # original reference's, never the stored arm's, which holds no float64 program.
            truth_fn = _truth_program(truth_fn if truth_fn is not None else reference_fn,
                                      jax.local_devices(backend="cpu")[0])
            if store.holds(label):
                from arm_store import _routed_arms

                # Its arms the programs its prepare compiled at the sizing draws' shapes too
                # (`_compile_arms`'s `sizing`), never compiled here.
                arms = _routed_arms(store, label, dew_fn, reference_fn, {},
                                    _arms_with(dew_fn, reference_fn, [], label))[:2]
            else:
                # A site its prepare did not reach sizes and runs unprepared, recorded before its sizing
                # compiles (a sizing that fails reaches no verdict to record it, `truths_ahead.running`).
                store.unprepared(f"{label}: its prepare did not reach this site")
        # An accuracy claim's bound sized too (`arm_forms._bounded`), Dew paired with the reference as written
        # (`_written_twin`; a store's, its prepare's twin at the sizing draws' shapes), the larger declared.
        pairs = {False: arms}
        if _bounded(reference_fn):
            pairs[True] = _written_pair(store, label, dew_fn, reference_fn, draw, arms)
            pairs[False] = arms or pairs[True][2]
        token = COMPILED.set(None)
        try:
            sized = {written: seeds_for(reference_fn, draw, label, truth_fn=truth_fn, dew_fn=dew_fn,
                                        orders=orders, examples=examples, followed=followed,
                                        arms=pair if pair is None else pair[:2], written=written)
                     for written, pair in pairs.items()}
        finally:
            COMPILED.reset(token)
        record["sized"], record["sized as written"] = max(sized.values()), sized.get(True)
    with open(os.environ[QUALIFY], "a", encoding="utf-8") as log:
        log.write(json.dumps(record) + "\n")
    if (sized := record.get("sized", 0)) > seeds:
        raise StaleSeeds(f"{label}: this site sizes {sized} seeds here and declares {seeds}: its count was "
                         f"sized on other arithmetic or another host level; declare what it sizes now")


def _written_pair(store, label: str, dew_fn, reference_fn, draw, arms):
    """A qualification run's pair of Dew and its reference as written (`_qualified`), and beside it the pair
    compiled alike: a store's prepared arms and twin where `arms` are its (`_routed_arm`), else both compiled
    here, their CPU mode decided once over all three arms (`_ynn_decided`) on a sizing draw, as the verdict
    decides it over all of them, so neither sizing takes another arithmetic than its verdict's.
    `(dew, twin, (dew, reference))`."""
    twin = _written_twin(dew_fn, reference_fn)
    if arms is not None:
        from arm_store import _routed_arm

        return arms[0], _routed_arm(store, label, reference_fn, twin, _written_role(None), written=True), arms
    _refuse_numpy_arms(draw, dew_fn, reference_fn)
    from arm_store import _routed_arm, _routed_arms

    # Each its sizing's own compile, a library arm held to its call (`arm_store._routed_arm`, row 14096).
    compiled = _arms_with(dew_fn, reference_fn, [], label)
    alike = _routed_arms(None, label, dew_fn, reference_fn, {}, compiled)[:2]
    twin = _routed_arm(None, label, reference_fn, twin, _written_role(None), written=True)
    with summed_as_run():
        _ynn_decided(label, [*alike, twin], draw(SIZING_FROM))
    return alike[0], twin, alike


def _site_judged(dew_fn, reference_fn, draw, label: str, *, seeds: int, truth_fn, orders, library_fn,
                 library_row, library_leaves, also, terms, examples: bool, weighing, upto: int | None = None,
                 offset: int = 0, followed=None, defers_to: str | None = None,
                 captured: dict[int, EvaluatedDraw] | None = None) -> None:
    """`assert_as_exact`'s judgment of one site under `label`, weighed where `weighing` is given (dew.parity's
    `Conditioning`: built of the site's compiled reference and truth where it draws, never in its judge), a
    map task recording its range's draws up to `upto` where given, its classes' digest beside them, its
    judge judging its map tasks' pooled draws (`_judged_pooled`), no arm or truth built. Its verdict
    (`Verdict`) names the weighed check it `defers_to` where it stays undecided (`_deferred`)."""
    given = reference_fn
    terms_fn, also = _site(dew_fn, reference_fn, label, seeds=seeds, truth_fn=truth_fn, library_fn=library_fn,
                           library_row=library_row, library_leaves=library_leaves, also=also, terms=terms)
    # Each reference written apart from Dew, compiled as written too (`_written_twin`), after every reference
    # compiled alike, its factors to WRITTEN; the site's own twin judged as it is beside a library arm.
    written = _written_references(reference_fn, also)
    labels = (label, *(f"{label}, against {name}" for name in also),
              *(f"{label}, against {name or 'its reference'} as written" for name, _ in written))
    powers = (1,) * (1 + len(also)) + (WRITTEN,) * len(written)
    verdict = Verdict(seeds, labels, powers, library_fn is not None, defers_to)
    library_also = len(also) if library_fn is not None and written and written[0][0] is None else None
    if judging():
        _judged_pooled(label, verdict)
        return
    truth_program = _truth_program(truth_fn)
    terms_fn = _site_terms(terms_fn, terms, draw, truth_fn if truth_fn is not None else reference_fn,
                           truth_program)
    narrow_leaves: list[str] = []
    compiled = _arms_with(dew_fn, reference_fn, list(also.values()), label)
    twins = [_written_twin(dew_fn, reference_fn if name is None else fn) for name, fn in written]
    # Its truth, where it names none its reference's arm as compiled here, resolved before the store's
    # programs take the arms' place: a reference arm read from the store is its own program at its own
    # inputs alone, which a truth its run computes (its truths at run, `truths_ahead.running`) never is.
    truth_program = truth_program if truth_program is not None else compiled[1]
    # The arms read from the store wherever its prepare reached the site, its truths' at run or not; else each
    # its run's own compile, a library arm held to its call on it (`arm_store._routed_arm`, row 14096).
    store = None if (ahead := reading()) is None else ahead[0]
    if store is None:
        _refuse_numpy_arms(draw, dew_fn, reference_fn, *([] if library_fn is None else [library_fn]),
                           *also.values())
    from arm_store import _routed_arm, _routed_arms

    # Each arm's draws run by the programs its prepare compiled (`_compile_arms`), whose trace refused an arm
    # computing in NumPy; every role routed once.
    dew_fn, reference_fn, further = _routed_arms(store, label, dew_fn, reference_fn, also, compiled)
    twins = [_routed_arm(store, label, fn, twin, _written_role(name), written=True)
             for (name, fn), twin in zip(written, twins, strict=True)]
    further = [*further, *twins]
    # A library arm's draws run as the other arms' are (row 8468).
    library_fn = (None if library_fn is None
                  else _routed_arm(store, label, library_fn, _arms(library_fn, library_fn)[1], "library"))
    truth_fn = truth_program
    # Each range's draws, every reference's, computed once for the stages of them all.
    ranges: dict[tuple[int, int], list[_Draws]] = {}
    # The site's weighing where it draws (`weighing`), built once, and the seconds that took.
    weighed: dict[str, object] = {}
    # The seconds of the first draw this process takes of the site and of its compiles (`_drawn`'s `timed`).
    first: dict[str, float] = {}

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
                ranges[start, stop] = _drawn(dew_fn, reference_fn, draw, range(start, stop), truth_fn,
                                             orders=orders, library_fn=library_fn,
                                             library_leaves=library_leaves, also=further, terms_fn=terms_fn,
                                             label=label, examples=examples,
                                             conditioning=conditioning_now(), timed=first, followed=followed,
                                             library_also=library_also, repeats=start == 0,
                                             captured=captured, capture_seeds=seeds)
                narrow_leaves[:] = _narrow(ranges[start, stop][0])
            return ranges[start, stop][index]
        return drawn

    arms = [dew_fn, reference_fn, *([] if library_fn is None else [library_fn]), *further]
    # A map task of the site's draws, or their judge (DRAWS, POOL).
    if _recorded(label, drawn_for, verdict, arms, upto, weighed, offset, first):
        return
    _verdicts(drawn_for, verdict, narrow_leaves, lambda: _mode_of(arms))


def _site_terms(terms_fn, terms, draw, truth, truth_program):
    """The site's terms program: `terms_fn`, the site's own, or by default (`terms` None) one sized from
    `truth` where its first draw holds a row of float64 zero (`_cancelling_terms`); none where `terms` is
    False, or in a judge, whose draws are its map tasks' records, each row sized by the terms its task
    found (`map_judge.judging`): no truth runs there. A program traced for its terms is traced once a
    signature, its static runs compiled once for the site's every draw (`float64_terms`)."""
    if terms is not False and not judging() and terms_fn is None:
        terms_fn = _cancelling_terms(draw, truth, truth_program)
    return terms_fn if terms_fn is None or isinstance(terms_fn, _SiteTerms) else functools.partial(
        float64_terms, terms_fn)


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
        elif _exact(theirs):
            # The original holds the float64 truth exactly: any move off it is past its rounding.
            off = not _exact(mine)
            (moved if off else kept).append(
                f"{name}: {'moved off' if off else 'kept'} the original's exact value")
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
    [draws] = _drawn(changed_fn, original_fn, draw, range(seeds), truth_fn, label=label, repeats=True)
    moved, kept, unresolved = _moved(draws, 1)
    if not moved:
        further = range(seeds, staged_units(seeds))
        draws = draws.joined(_drawn(changed_fn, original_fn, draw, further, truth_fn, label=label,
                                    repeats=False)[0])
        moved, kept, unresolved = _moved(draws, 2)
    if not moved:
        found = (f"{label}: the change moves no leaf beyond rounding over {len(draws.mine)} seeds:\n"
                 + "\n".join(kept))
        raise (Inconclusive([Finding(label, "inconclusive", 1.0, found)]) if unresolved
               else AssertionError(found))


def seeds_for(reference_fn, draw, label: str, *, start: int = FEWEST_SEEDS, most: int = MOST_SEEDS,
              truth_fn=None, dew_fn=None, orders=None, examples: bool = False,
              conditioning: bool = False, conditioning_scale=None, followed=None, arms: tuple | None = None,
              written: bool = False, progress=None) -> int:
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
    validated). The width asked of it is PARITY either way, and either mode draws from SIZING_FROM on, no
    verdict's inputs. A sizing holds out its draws only where `draw` keys an independent stream by the
    whole seed: a draw that enumerates finitely or by a modulus (`(arange(n) + seed) % vocab`) repeats a
    verdict's inputs at SIZING_FROM and needs a pilot stream of its own first. A recorded reference
    (`indexed_recording`), which holds its recorded seeds alone, is refused, its site keeping its
    recorded count, unless its recording holds pilot draws of their own from SIZING_FROM on, the
    reference and its truth on the inputs Dew's arm takes live (`recordings.recorded_pilots`, or a summarized
    recording's pilots, `indexed_recording`): its sizing then takes those pilots alone, up to as many as
    are held. Fails past `most`, where the site needs larger inputs or fewer shared factors, not more seeds.
    With `written`, Dew paired with the reference compiled as written (`_written_twin`): the count an
    accuracy claim's bound asks where the pair's count leaves it inconclusive, which the site then declares.

    Each draw is evaluated once however many doublings the sizing takes: a count's draws are its
    predecessor's and the seeds it adds, twice the count of them with the confirming block, the
    reference compared with itself takes its seeds 0 to four times the count (each block's first half
    against its second), and the reference on the same inputs runs once. `orders` sizes a site whose
    verdict sweeps them (`assert_as_exact`'s `orders`): each seed's evidence is its draw's over every
    order, as the verdict reads it. `examples` sizes a site that stacks them as its verdict judges it.
    `arms`, where given, are the pair as compiled (Dew's and the reference's, a store's, `_routed_arms`; with
    `written` the reference's twin as written), taken in place of compiling them here, an arm computing in
    NumPy refused where they were made. `progress`, where given, is told each count before its draws are
    drawn (`parity.Sized`)."""
    if COMPILED.get() is not None:
        raise ValueError(f"{label}: a sizing draws seeds past a prepared site's, whose truths' programs no "
                         "prepare compiled")
    paired = dew_fn is not None
    refuse_rounded(label, reference_fn, truth_fn=truth_fn, followed=followed, conditioning=conditioning,
                   beside_dew=paired)
    held = _paired_most(dew_fn, reference_fn, most=most, paired=paired)
    if held is None:
        raise ValueError(f"{label}: a recorded arm without pilots holds its recorded seeds alone, never a "
                         f"sizing's from SIZING_FROM; its site keeps its recorded count")
    most = held
    if arms is None:
        _refuse_numpy_arms(draw, *([dew_fn] if paired else []), reference_fn)
    # Dew paired with the reference compiled as written where `written` (`_written_twin`), else alike; the
    # pair as given (`arms`, a store's: its twin where `written`) in place of compiling one here.
    twin = _written_twin(dew_fn, reference_fn) if written and paired and arms is None else None
    # A weighed site (`assert_as_exact`'s `conditioning`) is sized as it is judged, weighed, its heavy
    # elements classed over its first stages' draws.
    weighing = (_conditioning(reference_fn, truth_fn, draw, range(staged_units(max(start, FEWEST_SEEDS))),
                              scale=conditioning_scale) if conditioning else None)
    if arms is None:
        from arm_store import _routed_arm

        # The pair its sizing compiles, a library arm held to its call on the executable it runs, as the
        # verdict's are (`arm_store._routed_arm`, row 14096).
        given = dew_fn if paired else reference_fn
        compiled = _arms(given, reference_fn)
        arms = (_routed_arm(None, label, given, compiled[0], "dew"),
                _routed_arm(None, label, reference_fn, compiled[1], "reference"))
        twin = twin and _routed_arm(None, label, reference_fn, twin, _written_role(None), written=True)
    dew_fn, reference_fn = arms
    reference_fn = twin or reference_fn
    dew_fn = dew_fn if paired else None
    truth_fn = _truth_program(truth_fn)
    # Both modes from SIZING_FROM, the draws no verdict takes: the reference compared with itself from seed 0
    # chose its count by the verdict's own inputs, its confirming blocks overlapping at least the whole first
    # stage (spec alignment, 9005).
    first = SIZING_FROM

    def drawn(start_at: int, stop: int):
        return _drawn(dew_fn or reference_fn, reference_fn, draw, range(first + start_at, first + stop),
                      truth_fn, orders=orders, label=label, examples=examples and dew_fn is not None,
                      conditioning=weighing, followed=followed, repeats=start_at == 0)[0]

    return _sized_count(drawn, label, start=start, most=most, paired=dew_fn is not None, progress=progress)


def recording(path) -> dict[str, np.ndarray]:
    """A recording of a reference over seeds (`indexed_recording`'s `stored`), read whole: where a pin
    holds it, from the fetched store's local copy, checked against the pin; else from the file at
    `path`, as git holds one of at most 1 MiB (tools/fetched_fixtures.py). Read, decompressed and
    checked once a process (`_read_recording`), each call its own copy of the arrays: a site reads its
    recording at each of its calls (`declared_seeds`, `indexed_recording`), the fla template's 41 MB 24
    times in one map task (16.4 s)."""
    return {name: value.copy() for name, value in _read_recording(str(path)).items()}


def recorded_names(path) -> frozenset[str]:
    """The arrays a recording (`recording`) holds, by name: read once a process, none copied."""
    return frozenset(_read_recording(str(path)))


@functools.cache
def _read_recording(path: str) -> dict[str, np.ndarray]:
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
    refuse_rounded(label, reference_fn, truth_fn=truth_fn, conditioning=conditioning,
                   beside_dew=dew_fn is not None)
    reference_fn, truth_fn = unrewritten(reference_fn), unrewritten(truth_fn)
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
                              library_fn=library_fn, library_leaves=library_leaves, conditioning=weighing,
                              repeats=start == 0)[0]
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


def ahead_arm(fn: Callable, label: str, *, host: bool = False, launched: bool = False) -> Callable:
    """`fn` (a function of arrays: a test's computation with no check-1 site, a bitwise check's kernel run on
    the GPU) as an arm with no site, through the arms' own path: within a GPU tier part's prepare (`COMPILED`,
    a store that prepares) its call is traced and compiled at its inputs' shapes as a site's arms are
    (`_arm_program` in the store's `tracing`, `_arm_compile`, held under `armed(label)` by Dew's statement of
    it, `arm_role`) and gives zeros of its outputs' shapes, nothing computing there (a test's later failure
    in its prepare, past its stored program, judges nothing); in its run its call is that executable
    (`CompiledTruths.ran`), tracing, lowering and compiling none; outside a part, `fn` itself. With `host`,
    its call is placed on the host's CPU, its program traced and compiled for it and run there (a GPU host's
    CPU-placed computation, row 8376's). One no statement holds, or one its prepare did not reach, runs
    unprepared, recorded and named (`unprepared`, as `running` records a site). One program a label: a second
    call of it in one prepare at the same program (its role and its inputs' shapes, `named`: two draws of one
    shape, one forward read by two backwards) gives that program's zeros and launches, compiling nothing
    again (`CompiledTruths.arm_again`), and one at another program is refused, as a second site of one label
    is (`compile_site`). Its inputs are host arrays, NumPy's: a prepare's default device computes nothing,
    not even their placement (bible §8). With
    `launched`, a call returns its outputs and the kernel launches the program it ran took (`tuning.Seen`,
    held): in a prepare its trace's, in its run its record's (`CompiledTruths.ran` notes them), outside a part
    those of its one lowering, traced and run once a call, so a witness judges the launches of the program
    it ran and no trace of its own (row 9550)."""
    def unprepared(*inputs):
        cpu = jax.devices("cpu")[0] if host else None
        with contextlib.nullcontext() if cpu is None else jax.default_device(cpu):
            placed = inputs if cpu is None else jax.device_put(inputs, cpu)
            # With `launched`, the launches of the program run: one lowering, traced where they are seen.
            lowered = fn if hasattr(fn, "lower") else jax.jit(fn)
            run = lowered.lower(*placed).compile() if launched else fn
            return run(*placed)

    def called(*inputs):
        if COMPILED.get() is None:
            return unprepared(*inputs)
        from arm_store import stored_call

        return stored_call(fn, label, inputs, host=host, unprepared=unprepared)

    if not launched:
        return called

    def with_launches(*inputs):
        from dew.nn.kernels import tuning

        with tuning.seen() as found:
            outputs = called(*inputs)
        return outputs, found.held()

    return with_launches


def ran_here(fn: Callable, label: str) -> list[Callable]:
    """`fn` each way this run takes a law's computation with no check-1 site: jitted as an arm (`ahead_arm`,
    under `label`), which a GPU tier part's prepare compiles ahead and its run runs from that program, and op
    by op on the CPU lane alone, whose eager dispatch no prepare compiles ahead (a GPU's tier takes the
    jitted way; row 10169's cells ran unprepared for it). A caller runs every way before it judges any: a
    prepare's arm gives zeros, and a check failing there leaves the arms after it uncompiled."""
    import tier_wire

    def eager(*args):
        with jax.disable_jit():
            return fn(*args)
    cpu_lane = jax.default_backend() == "cpu" and not tier_wire.preparing()
    return [ahead_arm(jax.jit(fn), label), *([eager] if cpu_lane else [])]


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
    read at the draw's index, the first of the indexed inputs (`indexed_recording`). A recording that keeps
    no outputs (`record_case(scale=..., summary=True)`) holds the reference's side of the weighed check as
    its recorder read it, and gives the conditioning whole: the scale, each leaf's heavy elements
    (`<site>.heavy.<leaf>`) and the reference's weighed evidence a draw (`<site>.weighed`, a recording of
    evidence whose leaves are the classes', `dew.parity.Conditioning.recorded_evidence`)."""
    scales = {name.removeprefix(f"{site}.scale."): value for name, value in stored.items()
              if name.startswith(f"{site}.scale.")}

    def scale(inputs, seed):
        return {name: value[int(inputs[0])] for name, value in scales.items()}

    declared = int(stored[f"{site}.conditioning"])
    if f"{site}.weighed.input_digest" not in stored:
        return {"conditioning": declared, "conditioning_scale": scale}
    # The masks in the leaves' order, a truth's tree's (`heavy_masks`), None for a leaf of no heavy element.
    heavy = [stored.get(f"{site}.heavy.{name}") for name in sorted(scales)]
    evidence = _recorded_evidence(stored, f"{site}.weighed")
    dtypes = _recorded_tree(stored, f"{site}.weighed", "dtype")

    def recorded_of(inputs, seed):
        return {path: _recorded_leaf(evidence, str(dtypes[path]), path, int(inputs[0]))
                for path in evidence["mse"]}

    return {"conditioning": declared, "conditioning_scale": Conditioning(scale, heavy, recorded_of)}


class UnrecordedTruth(AssertionError):
    """A float64 path a summarized recording's truth is recomputed by (`indexed_recording`'s `truth_fn`) that
    does not compute the recorded truth at a seed: its sketch past float64's rounding of the computation."""


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
    truth has a row of float64 zero the recording's had not is refused. A recording of a followed site
    (`record_case(..., followed=True)`) holds the reference's decisions too, `<site>.decisions.<name>`,
    each verdict seed's: its `truth_fn(decisions, inputs)` follows the decisions it is given, and the
    third value returned is the truth of the site's `Followed`, which holds Dew's float64 path,
    following the recorded decisions, to the recorded truth at each seed before it follows an arm's. Its
    reference and truth are host functions, called as check 1 calls them, not traced. Where it holds
    pilots (`<site>.pilot.`: a summary's `merge_shards`, a whole recording's `<site>.pilot.input.<name>`,
    `.reference.` and `.truth.`, its generator's draws from SIZING_FROM on), seeds from SIZING_FROM on read
    them, and its reference carries them (`recordings.recorded_pilots`'s `.pilots`), so a sizing of it takes
    those draws alone (`seeds_for`).

    With `leaves`, the recording is read for those leaves alone (`_of_leaves`): a site whose leaves are
    judged in groups, each its own test, reads each group so, its Dew arm and truth giving that group's
    leaves. Check 1 judges every leaf on its own tails, so a group draws its second stage on its own
    leaves' account alone, which looks at a passing leaf no more often than the whole site would."""
    if leaves is not None:
        stored = _of_leaves(stored, site, leaves)
    if any(name.startswith(f"{site}.sketch.") for name in stored):
        return _summarized(stored, site, seeds, inputs_of, truth_fn)

    from recordings import whole_recording

    return whole_recording(stored, site, seeds, inputs_of)


def _recorded_tree(stored, site: str, kind: str) -> dict[str, np.ndarray]:
    """A recording's entries of `kind` (`<site>.<kind>.<leaf>`), by leaf."""
    prefix = f"{site}.{kind}."
    return {name.removeprefix(prefix): np.asarray(value) for name, value in stored.items()
            if name.startswith(prefix)}


def _recorded_evidence(stored, site: str) -> dict[str, dict[str, np.ndarray]]:
    """A summary's evidence under `site`, each kind as the recording's layout holds it
    (`dew.parity.RECORDED_KINDS`), by leaf: a site's own, its pilots' (`<site>.pilot`), or its weighed
    check's (`<site>.weighed`, the classes' leaves)."""
    return {kind: _recorded_tree(stored, site, kind) for kind in recorded_kinds("seed", "span", "offsets")}


def _recorded_leaf(evidence, dtype: str, name: str, row: int) -> RecordedLeaf:
    """The leaf `name`'s recorded evidence at a seed's `row` of `evidence` (`_recorded_evidence`)."""
    start, stop = evidence["zero_offsets"][name][row], evidence["zero_offsets"][name][row + 1]
    examples = {field: evidence[field][name][row] for field in ("example_mse", "example_worst")
                if name in evidence[field]}
    # A recording made before its worst rows' shares were kept holds none: the gate then reads the row by
    # its width's factor (`_Draws.factor`).
    share = float(evidence["share"][name][row]) if name in evidence["share"] else None
    return RecordedLeaf(float(log_errors(evidence["mse"][name][row])), float(evidence["worst"][name][row]),
                        evidence["zero_rows"][name][start:stop], evidence["zero_errors"][name][start:stop],
                        finite=bool(evidence["finite"][name][row]), dtype=dtype, share=share,
                        **{field: None if np.isnan(value).all() else value
                           for field, value in examples.items()})


def _recorded_decisions(site: str, decided: dict[str, np.ndarray], pilots: int):
    """A followed recording's decisions (`<site>.decisions.<name>`, `decided`) at a verdict seed: none past
    them, so a followed recording holds no pilots."""
    if decided and pilots:
        raise AssertionError(f"{site}: a followed recording holds no pilots, its decisions its verdict "
                             f"seeds'")

    def at(index: int) -> dict[str, np.ndarray]:
        if index >= SIZING_FROM:
            raise AssertionError(f"{site}: a followed recording holds its verdict seeds' decisions alone")
        return {name: value[index] for name, value in decided.items()}
    return at


def _recorded_following(decisions_at, certified, live):
    """A followed recording's truth (a `Followed`'s `truth`): Dew's float64 path (`live`) following
    an arm's decisions, the path held first, at the seed, to the recorded truth by following the reference's
    own (`certified`), which it is where the two arms decided alike."""
    @as_given
    def followed(decisions, index, inputs):
        recorded = decisions_at(int(index))
        held_truth = certified(int(index), recorded, inputs)
        if decisions.keys() == recorded.keys() and all(
                np.array_equal(np.asarray(decisions[name]), recorded[name]) for name in recorded):
            return held_truth
        return live(decisions, inputs)
    return followed


def _draws_roundings(truth_fn, arguments, counted: dict) -> int:
    """The roundings one draw of the float64 truth `truth_fn` chains on `arguments` (`chain_roundings`): its
    program's, a window loop's its window's (each window a draw of its own), counted once (`counted`); a truth
    run as given that states the programs it runs (`as_given`'s `programs`: a host loop of compiled rows or
    steps), the sum over those this draw runs, each traced here in float64 once a program and signature, as
    a draw's programs may vary with it. The sum bounds every leaf's chain, which runs through some of those
    programs in turn: looser than one program's count by at most their number, a draw's rows or steps, so
    float64's bound on a leaf stays near 1e-13 of its norm where float32's rounding is 6e-8."""
    program, traced = _one_draws_program(truth_fn, arguments)
    stated = getattr(program, "programs", None)
    if stated is None:
        if "draw" not in counted:
            counted["draw"] = chain_roundings(jax.make_jaxpr(program)(*traced))
        return counted["draw"]

    def summed(window) -> int:
        total = 0
        for run, held in stated(*window):
            signature = (run, jax.tree.structure(held),
                         tuple((np.shape(leaf), np.result_type(leaf)) for leaf in jax.tree.leaves(held)))
            if signature not in counted:
                counted[signature] = chain_roundings(jax.make_jaxpr(run)(*held))
            total += counted[signature]
        return total

    # A window loop's every window, each a draw of its own whose programs may differ: the most of them.
    count = len(jax.tree.leaves(arguments)[0]) if isinstance(truth_fn, _Windowed) else 0
    each = ([jax.tree.map(lambda leaf, at=at: leaf[at], arguments) for at in range(count)]
            if count else [traced])
    with jax.enable_x64(new_val=True):
        return max(map(summed, each))


def certifier(site: str, truth_fn, shapes: Mapping[str, np.ndarray], recorded_sketch):
    """`truth_fn`, Dew's float64 path, as a recorded truth holds it: its program as the gate compiles a
    truth (`live`), and `certified(index, *arguments)`, its leaves on `arguments` (a draw's inputs, a
    followed site's recorded decisions before them), each held to the recorded truth's sketch at draw
    `index` (`recorded_sketch(index, name)`, of the leaves `shapes` names) within float64's rounding of the
    computation, its roundings counted on its trace (`_draws_roundings`); past it, `UnrecordedTruth`: a
    summarized recording's (`indexed_recording`), and a truths recording's (tests/recorded_truths.py)."""
    live, counted = _truth_program(truth_fn), {}

    def certified(index: int, *arguments):
        computed = live(*arguments)
        count = _draws_roundings(truth_fn, arguments, counted)
        eps = float(np.finfo(np.float64).eps)
        if set(computed) != set(shapes):
            raise AssertionError(f"{site}: `truth_fn` gives the leaves {sorted(computed)}, the recording "
                                 f"{sorted(shapes)}")
        for name, value in computed.items():
            value = np.asarray(value, np.float64)
            if value.shape != tuple(shapes[name].tolist()):
                raise AssertionError(f"{site}: the truth of {name!r} recomputed is {value.shape}, the "
                                     f"recorded reference's {tuple(shapes[name].tolist())}")
            recorded = recorded_sketch(index, name)
            if not np.isfinite(value).all():
                raise AssertionError(f"{site}: seed {index}'s float64 truth of {name!r} recomputed is not "
                                     f"finite")
            # The leaf's norm, scaled by its largest magnitude first so no square overflows.
            largest = float(np.max(np.abs(value))) if value.size else 0.0
            norm = largest * float(np.linalg.norm(value / largest)) if largest > 0 else 0.0
            # The oracle rule's bound on the difference's norm, and the rounding of both sketches, the
            # recorded truth's magnitudes no more than this one's and the difference's (`sketch_rounding`).
            oracle = 2 * count * eps * norm
            magnitude = float(np.sum(np.abs(value))) * (1 + value.size * eps) + np.sqrt(value.size) * oracle
            bound = (SKETCH_DEVIATIONS * oracle + 2 * sketch_rounding(magnitude)) * (1 + 1e-6)
            apart = truth_sketch(value, index) - recorded
            # Its root mean square, scaled by its largest first so no square overflows.
            widest = float(np.max(np.abs(apart))) if apart.shape == recorded.shape else np.inf
            away = (widest * float(np.sqrt(np.mean(np.square(apart / widest)))) if 0 < widest < np.inf
                    else widest)
            if not (np.isfinite(bound) and away <= bound):
                raise UnrecordedTruth(
                    f"{site}: seed {index}'s float64 truth of {name!r} recomputed is not the recorded "
                    f"reference's: its sketch is {away:.3e} away (root mean square), past float64's "
                    f"rounding of the computation "
                    f"({count} roundings); `truth_fn` does not compute what the reference's "
                    f"truth did")
        return computed

    return live, certified


def _summarized(stored, site: str, seeds: int, inputs_of, truth_fn):
    """`indexed_recording` of a recording of the reference's evidence alone."""
    def tree(kind):
        return _recorded_tree(stored, site, kind)

    verdict, pilot = (_recorded_evidence(stored, under) for under in (site, f"{site}.pilot"))
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

    # A followed site's recording holds the reference's decisions (`<site>.decisions.<name>`), each verdict
    # seed's, which its truth followed: the recorded reference is an arm of them (`Rounded`), its truth Dew's
    # float64 path following them, held to the recorded truth (`certified`), and Dew's arm's the same path
    # following its own (`assert_as_exact`'s `followed`).
    decided = tree("decisions")
    decisions_at = _recorded_decisions(site, decided, pilots)

    def reference(index, inputs):
        evidence, _, row = held(int(index))
        leaves = {name: _recorded_leaf(evidence, str(dtypes[name]), name, row) for name in mse}
        return _RecordedRounded(leaves, decisions_at(int(index))) if decided else leaves

    as_recorded(reference, range(SIZING_FROM, SIZING_FROM + pilots) if pilots else None)

    def recorded_sketch(index: int, name: str):
        evidence, _, row = held(index)
        return evidence["sketch"][name][row]

    # The truth recomputed, compiled as the gate compiles a truth, held to the recorded sketch.
    live, certified = certifier(site, truth_fn, shapes, recorded_sketch)

    if decided:
        return draw, reference, _recorded_following(decisions_at, certified, live)

    @as_given
    def truth(index, inputs):
        return certified(int(index), inputs)

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
        return  # every row is float64 zero: this rule sizes a row by its float64 value and holds none
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
