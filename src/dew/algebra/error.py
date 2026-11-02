"""Error-aware extraction (docs/design/algebra.md §1.5), at dev time: a function's equal forms the e-graph
holds under every rule set, the stable ones among them (`rules.admitted(dev=True)`), proposed as candidates
(the form as written, the cheapest by time, the one the stable proposal prices cheapest), each admitted by the
accuracy gate its caller holds (`admits`, check 1 over its draws, the one owner of the policy), and the
cheapest admitted chosen. Error is a candidate's, never an e-class fact: the forms of one class are equal in
the reals and differ in error, which depends on every operand form a candidate reads."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Hashable, Sequence

import jax
from egglog import DagCostModel, Vec

from dew.algebra import cost, forms, program, rules, terms


@dataclasses.dataclass(frozen=True)
class Candidate:
    """A form of the function: how it was proposed, its price in seconds (`cost.model`), and its function
    over the operands, returning the function's outputs in their structure."""

    name: str
    seconds: float
    function: Callable


@dataclasses.dataclass(frozen=True)
class Choice:
    """The candidate chosen and every candidate proposed."""

    chosen: Candidate
    candidates: tuple[Candidate, ...]


def choose(function: Callable, operands: Sequence[jax.ShapeDtypeStruct], *, accurate: bool,
           saturation: program.Saturation, admits: Callable[[Callable, Callable], bool]) -> Choice:
    """`function`'s form chosen for operands of `operands`' shapes and dtypes: the cheapest candidate the gate
    admits (`admits(candidate, reference)`, whether `candidate` is as exact as `reference` from the float64
    truth): under default numerics against the form as written; under the accurate numerics
    (`precision="accurate"`) against every other candidate, so a choice is never less accurate than any form
    proposed, the form as written among them."""
    closed, shaped = jax.make_jaxpr(function, return_shape=True)(*operands)
    region = terms.region(closed)
    graph, bundle, roots = program.graphed(region)
    facts = cost.target()
    timed = cost.model(region, cost.fused(facts))
    _, written_seconds = graph.extract(bundle, include_cost=True, cost_model=timed, extractor="greedy-dag")
    program.saturate(graph, region, rules.admitted(accurate=False, dev=True), saturation)
    tree = jax.tree.structure(shaped)
    candidates = [Candidate("as written", written_seconds.value, function)]
    for name, model in (("cheapest", timed), ("stable", cost.stable(region, cost.fused(facts)))):
        chosen, _ = program.extracted(graph, bundle, model)
        candidates.append(Candidate(name, _priced(region, chosen, timed),
                                    _valued(region, closed.consts, dict(zip(map(terms.key, roots), chosen,
                                                                            strict=True)), tree)))
    references = candidates if accurate else candidates[:1]
    admitted = [candidate for candidate in candidates
                if all(candidate is reference or admits(candidate.function, reference.function)
                       for reference in references)]
    return Choice(min(admitted, key=lambda candidate: candidate.seconds), tuple(candidates))


def _priced(region: terms.Region, chosen: list[terms.Term], model: DagCostModel[cost.Seconds]) -> float:
    """The price of `chosen` together under `model`, each shared value once: extracted from an unsaturated
    graph of `region`'s facts holding them, where no rule has made another form of theirs."""
    graph, _, _ = program.graphed(region)
    alone = terms.Term.bundle(Vec[terms.Term](*chosen))
    graph.register(alone)
    graph.run(rules.FACTS.saturate())
    _, price = graph.extract(alone, include_cost=True, cost_model=model, extractor="greedy-dag")
    return price.value


def _valued(region: terms.Region, consts, chosen: dict[Hashable, terms.Term], tree) -> Callable:
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
