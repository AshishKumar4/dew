"""Error-aware extraction (docs/design/algebra.md §1.5), at dev time: a function's equal forms the e-graph
holds under every rule set, the stable ones among them (`rules.admitted(dev=True)`), proposed as candidates
(the form as written, the cheapest by time, the one the stable proposal prices cheapest), and each one's sums
compensated (the error-free derivation, §6.9: `printed.source`'s), each admitted by the accuracy gate its
caller holds (`admits`, check 1 over its draws, the one owner of the policy), and the cheapest admitted
chosen. Error is a candidate's, never an e-class fact: the forms of one class are equal in the reals and
differ in error, which depends on every operand form a candidate reads."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence

import jax
from egglog import DagCostModel, Vec

from dew.algebra import cost, extract, forms, printed, program, rules, terms
from dew.nn import programs
from dew.nn.kernels import generation
from dew.nn.kernels.templates import product


@dataclasses.dataclass(frozen=True)
class Candidate:
    """A form of the function: how it was proposed, its price in seconds (`cost.model`), and its function
    over the operands, returning the function's outputs in their structure."""

    name: str
    seconds: float
    function: Callable
    compensated: bool = False
    """Whether its sums are compensated (`printed.source`'s error-free derivation): proposed after every
    uncompensated form, each at its own form's price, its compensation in spans protected from the pass that
    no price reads (`compensated`)."""
    form: Callable | None = None
    """The uncompensated form it compensates; None for an uncompensated candidate, its own `function`."""
    program: str | None = None
    """The printed text `function` runs (`printed.reprinted`, a compensated one's `printed.source`), which
    tells one proposed program from another; None where no plain call writes it."""


@dataclasses.dataclass(frozen=True)
class Choice:
    """The candidate chosen and every candidate proposed."""

    chosen: Candidate
    candidates: tuple[Candidate, ...]


VARIANTS = 4
"""The forms of each root the enumeration proposes (`EGraph.extract_multiple`), each beside the other roots'
cheapest, with the cheapest and the most stable extractions."""


def choose(function: Callable, operands: Sequence[jax.ShapeDtypeStruct], *, accurate: bool,
           saturation: program.Saturation, admits: Callable[[Callable, Callable], bool]) -> Choice:
    """`function`'s form chosen for operands of `operands`' shapes and dtypes: its candidates proposed in
    order of price (the form as written, the cheapest and the most stable extractions, and each root's
    forms beside the others' cheapest, `VARIANTS`, each distinct one priced as a whole, `_priced`), then each
    one's sums compensated (`compensated`), each program proposed once (`_distinct`), and the first the gate
    admits chosen (`admits(candidate, reference)`, whether `candidate` is as exact as `reference` from the
    float64 truth; docs/design/algebra.md §1.5, §6.5): under default numerics against
    the form as written, so one is always admitted; under the accurate numerics (`precision="accurate"`)
    against the form as written and every other candidate as exact as it, so a choice is never less
    accurate than any admissible form proposed, and a form that is not (one past the float's range) blocks
    none."""
    closed, shaped = jax.make_jaxpr(function, return_shape=True)(*operands)
    region = terms.region(closed)
    graph, bundle, roots = program.graphed(region)
    facts = cost.target()
    timed = cost.model(region, cost.fused(facts))
    _, written_seconds = extract.cheapest(graph, bundle, timed)
    program.saturate(graph, region, rules.admitted(accurate=False, dev=True), saturation, facts_settled=True)
    tree = jax.tree.structure(shaped)
    cheapest, _ = program.extracted(graph, bundle, timed)
    proposed = {"cheapest": cheapest,
                "stable": program.extracted(graph, bundle, cost.stable(region, cost.fused(facts)))[0]}
    for index, variants in enumerate(graph.extract_multiple(list(roots), VARIANTS)):
        for number, form in enumerate(variants):
            proposed[f"root {index}'s form {number}"] = [*cheapest[:index], form, *cheapest[index + 1:]]
    written = _reprinted("as written", written_seconds.value, function, operands, tree)
    candidates, seen = [written], set()
    for name, chosen in proposed.items():
        if (key := tuple(map(terms.key, chosen))) in seen:
            continue
        seen.add(key)
        valued = _valued(region, closed.consts, dict(zip(map(terms.key, roots), chosen, strict=True)), tree)
        candidates.append(_reprinted(name, _priced(region, chosen, timed), valued, operands, tree))
    candidates = _distinct(candidates)
    candidates = _distinct([*candidates, *(found for candidate in candidates
                                           if (found := compensated(candidate, operands, tree)) is not None)])
    candidates.sort(key=lambda candidate: (candidate.compensated, candidate.seconds))
    if not accurate:
        for candidate in candidates:
            if candidate is written or admits(candidate.function, written.function):
                return Choice(candidate, tuple(candidates))
    # Under the accurate numerics, every candidate as exact as the form as written is a reference, each
    # admitted against it once: one that is not (a regrouping past the float's range where the form as
    # written is finite) raises no bar and is never chosen.
    references = [written, *(candidate for candidate in candidates if candidate is not written
                             and admits(candidate.function, written.function))]
    admissible = {id(reference) for reference in references}
    for candidate in candidates:
        if id(candidate) in admissible and all(
                admits(candidate.function, reference.function) for reference in references
                if reference is not candidate and (candidate is written or reference is not written)):
            return Choice(candidate, tuple(candidates))
    raise ValueError("no candidate is as exact as every other: the gate admits none")


def compensated(candidate: Candidate, operands: Sequence[jax.ShapeDtypeStruct], tree) -> Candidate | None:
    """The dev-time error-free derivation (`printed.source`, §6.9): hold the additions, subtractions and
    the products they read as pairs, and compensate sum reductions. Their residuals are protected
    operations, not equalities over the reals, in which they would be extracted as zero. As with any
    stable candidate, the caller's gate checks the actual requested range, including overflow and
    underflow. Proposed after uncompensated forms at its written form's price (`Candidate.compensated`).
    Return None when there is no transform to apply, or the candidate cannot be printed."""
    closed = jax.make_jaxpr(candidate.function)(*operands)
    if printed.refused(closed, compensated=True) is not None:
        return None
    text, used = printed.source(closed, "compensated", compensated=True)
    if not used:
        return None
    flat = printed.loaded(text, "compensated")
    return Candidate(f"{candidate.name}, compensated", candidate.seconds,
                     lambda *values: jax.tree.unflatten(tree, list(flat(*values))), compensated=True,
                     form=candidate.function, program=text)


def _reprinted(name: str, seconds: float, function: Callable, operands: Sequence[jax.ShapeDtypeStruct],
               tree) -> Candidate:
    """A candidate measured as its printed text computes it (`printed.reprinted`), what a generated module
    of it computes, that text its program; one no plain call writes (`printed.Unprintable`) as it is."""
    found = printed.reprinted(function, operands, tree)
    if found is None:
        return Candidate(name, seconds, function)
    measured, text = found
    return Candidate(name, seconds, measured, program=text)


def _distinct(candidates: Sequence[Candidate]) -> list[Candidate]:
    """`candidates` in their order, each program once: a later one whose printed program an earlier one's
    is (another extraction printing as the form as written does) is the same measurement, and the gate,
    pairwise, would compare a program with itself (§1.5). One no plain call writes is kept; its extraction
    is told apart by its terms alone."""
    kept, programs = [], set()
    for candidate in candidates:
        if candidate.program is not None:
            if candidate.program in programs:
                continue
            programs.add(candidate.program)
        kept.append(candidate)
    return kept


def at_least(asked: jax.lax.DotAlgorithmPreset, dtype) -> Callable[[jax.lax.DotAlgorithmPreset], bool]:
    """A site's gate for `product_precision` stated as an algorithm: it admits a product of `dtype` operands
    in a preset at least as exact as `asked` in them, by what each states of its arithmetic
    (`dew.nn.programs.dot_arithmetic`): as many significand digits of each operand
    (`DotArithmetic.digits`), as wide an accumulation, an imprecise one only where `asked`'s is, and parts
    of as wide a range as `asked`'s hold of the operands (`DotArithmetic.largest`: a part whose format's
    largest is beneath rounds an operand near the dtype's largest to an infinity, row 11878: BF16x6 and x9
    of fp32 operands near float32's largest overflow where IEEE fp32's products are finite). So a
    port states the algorithm its upstream runs and takes the cheapest its backend runs that keeps as much,
    not a precision picked by hand (row 5473). `asked` names an algorithm: the default preset, which names
    none, is refused."""
    operand = jax.numpy.dtype(dtype)
    floor = programs.dot_arithmetic(asked, operand, operand)
    if floor is None:
        raise ValueError(f"{asked} names no dot algorithm for a product to be at least as exact as")

    reach = min(*floor.largest, float(jax.numpy.finfo(operand).max))

    def admits(preset: jax.lax.DotAlgorithmPreset) -> bool:
        arithmetic = programs.dot_arithmetic(preset, operand, operand)
        return (arithmetic is not None and arithmetic.digits >= floor.digits
                and arithmetic.accumulation_bits >= floor.accumulation_bits
                and (floor.imprecise or not arithmetic.imprecise) and min(arithmetic.largest) >= reach)
    return admits


def product_precision(rows: int, depth: int, columns: int, dtype, *,
                      admits: Callable[[jax.lax.DotAlgorithmPreset], bool],
                      stated: Iterable[jax.lax.DotAlgorithmPreset]
                      | Mapping[jax.lax.DotAlgorithmPreset, jax.lax.PrecisionLike] | None = None,
                      ) -> jax.lax.DotAlgorithmPreset:
    """The dot algorithm a `[rows, depth] @ [depth, columns]` product of `dtype` operands runs in for the
    accuracy its site asks: of the algorithms JAX runs operands of that dtype in as they are stored
    (`dew.nn.programs.dot_algorithms`, JAX's own conversion; no list of them here), first those its lowering
    runs as named (`_lowered`: `stated`, its own table, Pallas's Triton lowering's, or a mapping of each to
    the precision it runs it as, Mosaic TPU's; XLA's where not given: every one, on a TPU target XLA:TPU's
    alone, `product.TPU_ALGORITHMS`), an algorithm it refuses never taken (row 11738); then the cheapest of
    those on the process's target (`cost.product_work` under XLA's schedule, each at the precision its
    lowering runs it as, one whose work no rate prices after every priced one, row 10845) that `admits` holds:
    the
    site's gate run on its own family at that algorithm (as exact as the accuracy asked, from the float64
    truth), or `at_least` the algorithm its upstream runs. So a site states an accuracy and takes whatever
    product meets it at the best rate: three TF32 passes where one is not enough and IEEE fp32 costs more,
    one where it is (row 5473; docs/design/algebra.md §1.5). None admitted is refused: the site asked more
    than any stated algorithm gives, and no cheaper one is taken in its place."""
    operand = jax.numpy.dtype(dtype)
    target = cost.target()
    runs_as = _lowered(programs.dot_algorithms(operand), stated, target)
    schedule = cost.fused(target)

    def seconds(preset: jax.lax.DotAlgorithmPreset) -> tuple[float, ...]:
        # Work no rate prices last, as `cost.cheaper` weighs it: no seconds price it, so none makes its
        # algorithm cheaper.
        work = cost.product_work(rows, depth, columns, dtype=operand, precision=runs_as[preset])
        return sum(schedule.unpriced(work)), *cost.Seconds(work, schedule).key

    for preset in sorted(runs_as, key=seconds):
        if admits(preset):
            return preset
    raise ValueError(f"no dot algorithm JAX states for {operand} operands that its lowering runs meets the "
                     f"accuracy asked (of {[preset.name for preset in runs_as]})")


def _lowered(found: Sequence[jax.lax.DotAlgorithmPreset],
             stated: Iterable[jax.lax.DotAlgorithmPreset]
             | Mapping[jax.lax.DotAlgorithmPreset, jax.lax.PrecisionLike] | None,
             target) -> dict[jax.lax.DotAlgorithmPreset, jax.lax.PrecisionLike]:
    """Each of `found` a product's lowering runs as named, by what it runs it as, which prices it: `stated`'s
    table (an algorithm, run as named) or mapping (each to the precision it runs it as: Mosaic TPU's, which
    takes no algorithm itself); else XLA's own, every algorithm, but on a TPU target XLA:TPU's alone
    (`product.TPU_ALGORITHMS`), which refuses every other at its compile."""
    if stated is None:
        runs = product.TPU_ALGORITHMS if generation.tpu(target) else found
        return {preset: preset for preset in found if preset in runs}
    named = stated if isinstance(stated, Mapping) else {preset: preset for preset in stated}
    return {preset: named[preset] for preset in found if preset in named}


def _priced(region: terms.Region, chosen: list[terms.Term], model: DagCostModel[cost.Seconds]) -> float:
    """The price of `chosen` together under `model`, each shared value once: extracted from an unsaturated
    graph of `region`'s facts holding them, where no rule has made another form of theirs."""
    graph, _, _ = program.graphed(region)
    alone = terms.Term.bundle(Vec[terms.Term](*chosen))
    graph.register(alone)
    graph.run(rules.FACTS.saturate())
    _, price = extract.cheapest(graph, alone, model)
    return price.value


def _valued(region: terms.Region[terms.Term], consts, chosen: dict[Hashable, terms.Term], tree) -> Callable:
    """The function `region`'s program computes with each root in its chosen form (`chosen`, by the root's
    key, `terms.key`), over its operands: each output its root's form, a literal output its literal, in the
    program's output structure `tree`, an equation outside the language evaluated where a form reads its
    leaf, over its own inputs' chosen forms (the roots too), so the candidate computes the forms it was
    priced as."""
    def valued(*operands):
        value = forms.evaluator(region, [*consts, *operands], {}, chosen=chosen)
        outputs = [literal if term is None else value(chosen.get(terms.key(term), term))
                   for term, literal in zip(region.outputs, region.literal_outputs, strict=True)]
        return jax.tree.unflatten(tree, outputs)
    return valued
