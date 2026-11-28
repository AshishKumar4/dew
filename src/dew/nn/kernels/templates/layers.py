"""A whole decoder layer's fusion plan (row 9038; docs/design/algebra.md, Fusion): every product region of
a layer's program, each and each of its fewer-staged variants (`dew.algebra.fusion.fewer`) weighed by the
one price the compile pass reads (`fusion.analyzed`: the template launch with its stages against XLA's
program of them, its product's launch routed as its selector routes it): the dev-time report the spec
states, a plan of the layer's launches and their prices, made on request and never checked in (its prices
are the cost model's of the day, its prose a reader's).

Two views. `today`: each body as the pass weighs it, a region of its own (a call's, a custom derivative's,
a marked span's), so its choices (`Weighed.passed`: the first variant that pays, most stages first) are
what `fusion.fused_program` takes. `search`: the layer flattened to one body of its math (`flattened`), where
a region may run across a boundary the pass never crosses, each stated with what crossing it needs
(`Crossing`: a kernel owning a derivative or a span's kind, or replacing a platform's launch), and the
variant that saves the most (`Weighed.best`) whatever it crosses; a variant across a boundary never pays.
A fused epilogue's price cites M-AL-7, which decides whether the template's epilogue is realizable at all.
No kernel is emitted: one launch for the layer, its generated megakernel, is a later step of the row, after
its launch-control measurement (row 6107).

`python -m dew.nn.kernels.templates.layers [OUT]` plans the layers `CASES` names and prints them, or
writes them to `OUT`."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.extend import core as jcore

from dew.nn import programs


class Case(NamedTuple):
    """A layer planned: a one-layer dense `CausalTransformer` (its canonical block: an RMSNorm, attention, a
    residual, an RMSNorm, the gated SiLU MLP and a residual) decoding `rows` sequences a token, its program
    lowered as on the CUDA `generation`."""

    name: str
    generation: str
    emb_features: int
    num_heads: int
    mlp_features: int
    rows: int
    vocab_size: int = 1024
    max_seq_len: int = 256


CASES = (Case("dense 1024x4096, 16 sequences, sm90", "sm90", 1024, 8, 4096, 16),
         Case("dense 1024x4096, 16 sequences, sm80", "sm80", 1024, 8, 4096, 16))
"""The layers the report plans (`main`)."""


class Crossing(NamedTuple):
    """A boundary of a program the flattened layer (`flattened`) runs across, which the pass never does (it
    weighs each body apart, `dew.algebra.program._within`): its `kind`, and what a kernel fusing across it
    needs (`needs`), so no price across it reads as one the pass would take."""

    kind: str
    needs: str


CALL = Crossing("call", "none: a nested program's body, the pass's region of its own")
CHECKPOINT = Crossing("checkpoint", "the kernel holds the checkpoint's recomputation as its rule states it")
DERIVATIVE = Crossing("custom derivative", "the kernel owns the derivative its rule states")
BATCHING = Crossing("batching rule", "the kernel owns the batching its rule states")
PLATFORM = Crossing("platform choice", "it replaces the launch the platform's branch makes, priced as that "
                                       "launch, not as the portable branch's math")


def _marked(kind) -> Crossing:
    """A marked span's crossing (`dew.nn.derivatives.marked`): the kernel owns its kind."""
    return Crossing(f"marked span: {type(kind).__name__}", f"the kernel owns the {type(kind).__name__} span")


class Flat(NamedTuple):
    """A program as one body of its math (`flattened`): its equations, and of each the boundaries it was
    found within (`within`: each a crossing's frame, by the frame's number, outermost first; `crossings`:
    each frame's crossing)."""

    body: jcore.Jaxpr
    within: tuple[tuple[int, ...], ...]
    crossings: dict[int, Crossing]

    def crossed(self, places: Sequence[int]) -> tuple[Crossing, ...]:
        """The crossings a region of `places` runs across: each frame some of its places are within and some
        are not, and a platform choice any of them is within (`PLATFORM`: its portable branch, which the
        GPU's launch may replace whole, however the region lies in it), once, outermost first."""
        frames = [set(self.within[place]) for place in places]
        crossed = (set.union(*frames) - set.intersection(*frames)) | {
            frame for frame in set.union(*frames) if self.crossings[frame] == PLATFORM}
        return tuple(dict.fromkeys(self.crossings[frame] for frame in sorted(crossed)))


def flattened(jaxpr: jcore.Jaxpr) -> Flat:
    """`jaxpr` as one body of its math: each call run in its caller (`programs.called`: a jit's, a
    checkpoint's, a custom derivative's primal, a marked span but a protected one, a numerics contract), its
    every value a fresh variable of its own (`programs.inlined`: one helper called twice two producers, not
    one), and each platform choice its default branch (a `cond` whose `branches_platforms` name one), each
    equation marked with the boundaries it was within (`Flat.within`). A native kernel's body
    (`programs.opaque`) and a protected span stay leaves."""
    equations: list[jcore.JaxprEqn] = []
    within: list[tuple[int, ...]] = []
    crossings: dict[int, Crossing] = {}
    # Each opened call's results by what made them, one map for every level: inlined, every value is its own.
    renamed: dict[jcore.Var, jcore.Var | jcore.Literal] = {}

    def value(atom):
        while isinstance(atom, jcore.Var) and atom in renamed:
            atom = renamed[atom]
        return atom

    def walk(body: Sequence[jcore.JaxprEqn], frames: tuple[int, ...]) -> None:
        for equation in body:
            equation = equation.replace(invars=[value(atom) for atom in equation.invars])
            opened = _opened(equation)
            if opened is None:
                equations.append(equation)
                within.append(frames)
                continue
            inner, consts, operands, crossing = opened
            frame = len(crossings)
            crossings[frame] = crossing
            made, returned = programs.inlined(inner, operands, consts)
            walk(made, (*frames, frame))
            renamed.update(zip(equation.outvars, returned, strict=True))

    walk(jaxpr.eqns, ())
    outvars = [value(atom) for atom in jaxpr.outvars]
    return Flat(jaxpr.replace(eqns=equations, outvars=outvars), tuple(within), crossings)


def _opened(equation: jcore.JaxprEqn) -> tuple[jcore.Jaxpr, tuple, list, Crossing] | None:
    """The body `flattened` runs `equation` as, its constants, the operands it reads and the boundary it
    crosses: a call's program (`programs.called`) or a platform choice's default branch, its index dropped;
    None for any other, a protected span's or a native kernel's among them."""
    from dew.nn import derivatives

    platforms = equation.params.get("branches_platforms") if equation.primitive is jax.lax.cond_p else None
    if platforms is not None and None in platforms:
        branch = equation.params["branches"][platforms.index(None)]
        return branch.jaxpr, tuple(branch.consts), list(equation.invars[1:]), PLATFORM
    if programs.opaque(equation) or (body := programs.called(equation)) is None:
        return None
    primitive = equation.primitive
    if primitive is derivatives.marked_call_p:
        kind = equation.params["kind"]
        if isinstance(kind, derivatives.Protected):
            return None
        crossing = _marked(kind)
    else:
        crossing = {"remat2": CHECKPOINT, "checkpoint": CHECKPOINT, "custom_jvp_call": DERIVATIVE,
                    "custom_vjp_call": DERIVATIVE, "custom_vmap_call": BATCHING}.get(primitive.name, CALL)
    # Its constants its own (`programs.called`: a Jaxpr carries them on JAX 0.12).
    return body, tuple(body.consts), list(equation.invars), crossing


class Segment(NamedTuple):
    """A product region weighed (`fusion.analyzed`): the places in its body it computes (its prologue's
    replayed among them), which stages it holds, its product's launch as its selector routes it (`kernel`:
    the template's, or XLA's where its cell refuses it, `refusal` its code), whether the pass takes it
    (`takes`: its own decision, `fusion.decided`), whether fused is the cheaper
    (`cheaper`: `fused_gemm.Analysis.cheaper`), its seconds fused and unfused (None where it has no stage to
    weigh or the target states no rate), the boundaries it runs across (`crossings`, the flattened layer's:
    none in a body the pass weighs) and the measurement its price awaits (`cites`: a fused epilogue's
    realizability, M-AL-7)."""

    places: tuple[int, ...]
    stages: tuple[str, ...]
    kernel: str | None
    refusal: str | None
    refused: str | None
    """Why its analysis stopped before any route (`fused_gemm.Analysis.refused`: a forced stage refused);
    None where it reached them."""
    takes: bool
    cheaper: bool
    fused: float | None
    unfused: float | None
    crossings: tuple[Crossing, ...] = ()
    cites: tuple[str, ...] = ()

    @property
    def pays(self) -> bool:
        """Whether the pass would take it (`takes`, the pass's own decision, `fusion.decided`), across no
        boundary the pass does not cross."""
        return self.takes and not self.crossings

    @property
    def saved(self) -> float:
        """The seconds it saves fused, where it pays (none where it does not, or is not priced)."""
        if not self.pays or self.fused is None or self.unfused is None:
            return 0.0
        return self.unfused - self.fused


class Weighed(NamedTuple):
    """One product's variants weighed (`Segment`s, most stages first), in its body (`body`: its path of
    equations from the step, the pass's region; `flattened`'s: the layer whole), the pass's choice (`passed`:
    the variant the pass selects, `fusion.taken`, where it crosses no boundary) and what saves the most,
    where it would cross a boundary (`best`)."""

    body: str
    product: int
    variants: tuple[Segment, ...]
    chosen: int | None = None
    """The place among `variants` of the one the pass selects (`fusion.taken`); None where it takes none."""
    stops: tuple[str, ...] = ()
    """Where the region's growth stopped: each equation outside it reading its result (a row-relabelling
    `reshape` before any epilogue, as a decode's projection's output is), by its primitive; a structural
    stop, which no price decides."""

    @property
    def passed(self) -> Segment | None:
        """The variant the pass selects (`fusion.taken`), where it crosses no boundary the pass does not:
        today's, the pass's choice in its body; the search's, the pass's choice were it to see the flattened
        layer as one body, None where that choice crosses a boundary, whatever variant of fewer stages
        crossing none it would take."""
        found = None if self.chosen is None else self.variants[self.chosen]
        return found if found is not None and not found.crossings else None

    @property
    def best(self) -> Segment | None:
        """The variant that saves the most priced, whatever it crosses (its `crossings` stating what that
        needs), but a platform choice: its math priced as the portable branch's XLA program, not as the
        launch it replaces (`PLATFORM`), it is unpriced; None where none is priced cheaper."""
        saved = [(segment.unfused - segment.fused, place) for place, segment in enumerate(self.variants)
                 if segment.takes and segment.cheaper and segment.fused is not None
                 and segment.unfused is not None and PLATFORM not in segment.crossings]
        return self.variants[max(saved, key=lambda entry: entry[0])[1]] if saved else None


class Plan(NamedTuple):
    """A layer's plan (`plan`): each product of each body as the pass weighs it today (`today`), each product
    of the layer flattened (`search`, crossings stated), and the norm and attention calls recognition writes
    over the flattened layer (`recognized`, each its family and places)."""

    today: tuple[Weighed, ...]
    search: tuple[Weighed, ...]
    recognized: tuple[tuple[str, tuple[int, ...]], ...]


def plan(jaxpr: jcore.Jaxpr) -> Plan:
    """`jaxpr`'s plan: each body the pass weighs (`program.fused_bodies`, in the context it weighs it in)
    weighed as it weighs it (`_weighed_body`), and the layer flattened (`flattened`) weighed whole, each
    variant stating the boundaries it crosses; recognition's calls over the flattened layer."""
    from dew.algebra import fusion, program, recognize
    from dew.nn.kernels import generation

    # The bodies the pass weighs on the platform the step lowers for: a platform choice's branch it runs.
    today = tuple(found for each in program.fused_bodies(jaxpr, lambda path, body: _weighed_body(
        body, "step" + "".join(f"/{name}@{place}.{number}" for name, place, number in path)),
        platform=generation.lowering_platform()) for found in each)
    flat = flattened(jaxpr)
    search = _weighed_body(flat.body, "flattened", flat)
    index = fusion.Index(flat.body)
    families = (("attention or recurrence", (*recognize.ATTENTION, *recognize.RECURRENCES)),
                ("norm", recognize.NORMS))
    recognized = tuple((family, tuple(written.places)) for family, kinds in families
                       for written in recognize.recognized(flat.body, index, kinds))
    return Plan(today, search, recognized)


def _weighed_body(body: jcore.Jaxpr, path: str, flat: Flat | None = None) -> tuple[Weighed, ...]:
    """Each product region of `body` the pass weighs (`fusion.eligible`: recognized attention and recurrence
    calls set aside) and its variants of fewer stages (`fusion.fewer`, in the pass's order), each weighed as
    the pass decides it over the body's one `fusion.Index` (`_weighed`), the crossings of each its flattened
    layer's (`flat`, of every place it replays, a shared prologue's among them; none in a body the pass
    weighs), the variant the pass selects (`fusion.taken`) and where its growth stopped (`_stops`)."""
    from dew.algebra import fusion

    index = fusion.Index(body)
    found = []
    for region in fusion.eligible(index):
        variants = (region, *fusion.fewer(index, region))
        product = next(place for place in region.places if fusion.product_form(body.eqns[place]) is not None)
        taken = fusion.taken(index, region)
        chosen = next((at for at, variant in enumerate(variants) if variant == taken), None)
        # The pass's selection is one of the variants it walks (`fusion.fewer`'s), both its `_fewer`.
        assert taken is None or chosen is not None, (taken, variants)
        found.append(Weighed(path, product, tuple(
            _weighed(index, variant, () if flat is None else flat.crossed(variant.replayed))
            for variant in variants), chosen, _stops(body, index, region)))
    return tuple(found)


def _stops(body: jcore.Jaxpr, index, region) -> tuple[str, ...]:
    """The primitives of the equations outside `region` that read a value it makes: where its growth
    stopped (`Weighed.stops`)."""
    places = set(region.places)
    made = [var for place in places for var in body.eqns[place].outvars if isinstance(var, jcore.Var)]
    outside = {reader for var in made for reader in index.readers.get(var, ()) if reader not in places}
    return tuple(sorted({body.eqns[reader].primitive.name for reader in outside}))


def _weighed(index, region, crossings: tuple[Crossing, ...]) -> Segment:
    """`region` as the pass decides it (`fusion.decided`, the pass's own rule): whether it takes it, and its
    product's route as its selector takes it, from the analysis the decision was priced by
    (`RegionDecision.analysis`)."""
    from dew.algebra import fusion

    decision = fusion.decided(index, region)
    takes, analysis = decision.taken, decision.analysis
    # A region the pass decides is one it priced (`RegionDecision.analysis`, none for `fewer`'s walk alone).
    assert analysis is not None, region
    # Its stages as its equations hold them, whatever its analysis reached (one a refused stage stopped
    # states no routes, `fused_gemm.Analysis.refused`).
    stages = (*(("prologue",) if region.prologue else ()), "product",
              *(("epilogue",) if _epilogue(index.jaxpr, region) else ()))
    launch = analysis.launch
    return Segment(tuple(region.replayed), stages, None if launch is None else launch.kernel,
                   None if launch is None or launch.refusal is None else launch.refusal.code,
                   analysis.refused, takes, analysis.cheaper, analysis.fused_seconds,
                   analysis.unfused_seconds, crossings, ("M-AL-7",) if "epilogue" in stages else ())


def _epilogue(body: jcore.Jaxpr, region) -> bool:
    """Whether `region` holds an equation reading its product's value: an epilogue's first (its captured
    operands' layout equations read none of it)."""
    from dew.algebra import fusion

    (product,) = (place for place in region.places if fusion.product_form(body.eqns[place]) is not None)
    made = set(body.eqns[product].outvars)
    return any(place > product and any(isinstance(atom, jcore.Var) and atom in made
                                       for atom in body.eqns[place].invars) for place in region.places)


def traced(case: Case) -> jcore.Jaxpr:
    """`case`'s decode step, a token a sequence through its own cache, traced (`jax.make_jaxpr`)."""
    from flax import linen as nn

    from dew.nn.backbones import CausalTransformer

    model = CausalTransformer(vocab_size=case.vocab_size, emb_features=case.emb_features, num_layers=1,
                              num_heads=case.num_heads, mlp_features=case.mlp_features,
                              max_seq_len=case.max_seq_len)
    ids = jnp.zeros((case.rows, 1), jnp.int32)
    variables = jax.eval_shape(lambda: nn.unbox(model.init(jax.random.key(0), ids)))
    cache = jax.eval_shape(lambda: nn.unbox(model.init(jax.random.key(0), ids, decode=True))["cache"])
    return jax.make_jaxpr(lambda variables, cache: model.apply(
        {**variables, "cache": cache}, ids, decode=True, mutable=["cache"]))(variables, cache).jaxpr


def written(case: Case) -> dict:
    """`case`'s plan as JSON (`report`), on the case's generation (`generation.assumed`) at JAX's default
    matmul precision: each product of each body as the pass weighs it, and of the flattened layer, its
    variants by stage, each with its launch's route, prices, crossings and citations, the pass's choice and
    what saves the most; and the calls recognition writes."""
    from dew.nn.kernels import generation
    from dew.nn.precision import configured_precision

    # At JAX's default matmul precision, as a model's step runs, whatever the process states: a step traced
    # at `highest` takes its products' portable forms, no template's (the test suite's environment planned
    # no product region, FS-9038-ENV2).
    with generation.assumed(case.generation), configured_precision("default", exact=True):
        planned = plan(traced(case))

    def segment(found: Segment | None) -> dict | None:
        if found is None:
            return None
        return {"stages": list(found.stages), "kernel": found.kernel, "refusal": found.refusal,
                "refused": found.refused, "takes": found.takes, "cheaper": found.cheaper,
                "fused": found.fused, "unfused": found.unfused,
                "crossings": [{"kind": crossing.kind, "needs": crossing.needs}
                              for crossing in found.crossings],
                "cites": list(found.cites)}

    def weighed(found: Weighed) -> dict:
        return {"body": found.body, "product": found.product, "stops": list(found.stops),
                "chosen": found.chosen,
                "variants": [segment(variant) for variant in found.variants],
                "passed": segment(found.passed), "best": segment(found.best)}

    return {"case": case._asdict(), "today": [weighed(found) for found in planned.today],
            "search": [weighed(found) for found in planned.search],
            "recognized": [{"family": family, "equations": len(places)}
                           for family, places in planned.recognized]}


def report(cases: Sequence[Case] = CASES) -> str:
    """Every case's plan (`written`), as the report prints it."""
    return json.dumps({case.name: written(case) for case in cases}, indent=1, sort_keys=True) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Plan every case and print the report, or write it to the path given."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", maxsplit=1)[0])
    parser.add_argument("out", nargs="?", type=Path, help="where to write the report (printed where none)")
    arguments = parser.parse_args(argv)
    text = report()
    if arguments.out is None:
        sys.stdout.write(text)
    else:
        arguments.out.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
