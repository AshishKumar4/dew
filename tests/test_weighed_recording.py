"""A weighed site's recording kept as evidence alone (tools/recorded_references.py's
`record_case(scale=..., summary=True)`, rows 5381 and 5382): no output of the reference or its truth, but
each draw's rounding scale, which weighs Dew's own arm in the test, and the reference's side of the weighed
check as its recorder read it with the outputs in hand (each leaf's heavy elements and the reference's
weighed evidence a draw). Check 1 judges the site from it as it does from a recording of the outputs whole.

The operation is tests/test_weighed_check.py's: one element a row pushed through 2^k and back, whose
rounding rules the unweighted check and leaves it undecided, beside products each rounded once."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from recorded_runs import assert_as_exact_as_recorded
from reference_error import GateFailure, Inconclusive
from rounding_bound import rounding_scale
from test_weighed_check import WEIGHED_SEEDS, biased, draw, reference, reformulated
from tool_support import load

from dew import parity

NAMES = ("a", "b", "big")


def inputs_of(seed: int) -> dict:
    """A seed's operands by name, as a recording draws them."""
    return dict(zip(NAMES, draw(seed), strict=True))


LAYOUTS = {"rows": lambda made: made, "one row": lambda made: made[0]}
"""The site's leaf as the operation makes it, its rows, or its first row alone, a vector (one row, testing.md:
a leaf's rows run along its last axis)."""


def leafed(fn, layout: str):
    """`fn`'s output as a site's one leaf, laid as `layout` (`LAYOUTS`)."""
    return lambda *operands: {"y": LAYOUTS[layout](fn(*operands))}


def at(narrow, layout: str):
    """A recorded site's `dew(dtype, inputs)` whose float32 arithmetic is `narrow`'s and whose float64 path
    is the reference's own, the truth a summary holds it to: the amplified element is not exact in float64
    either (a small value beside 2^24 passes its 53 bits), so another formulation's float64 run is not the
    recorded truth within float64's own rounding, and a summary would refuse it."""
    def dew(dtype, inputs):
        fn = narrow if jnp.dtype(dtype) == jnp.float32 else reference
        return leafed(fn, layout)(*(jnp.asarray(inputs[name], dtype) for name in NAMES))
    return dew


LEAFED = {layout: leafed(reference, layout) for layout in LAYOUTS}
"""The reference as the site's leaf, one function a layout: its rounding scale's program is compiled once for
it (`rounding_bound.SCALES`, keyed by the function), where a function made anew each draw compiled each draw's
(7,168 of them, 267 of the run's 302 s)."""

RAN = {layout: jax.jit(LEAFED[layout]) for layout in LAYOUTS}
"""The reference compiled, once a precision and a layout: a recording asks it of thousands of draws."""


def evaluated(layout: str):
    """The reference's float32 outputs beside their float64 truth, as a recording's worker answers."""
    def evaluate(inputs):
        operands = [inputs[name] for name in NAMES]
        with jax.enable_x64(new_val=False):
            ran = jax.tree.map(np.asarray, RAN[layout](*operands))
        with jax.enable_x64(new_val=True):
            return ran, jax.tree.map(np.asarray,
                                     RAN[layout](*(np.asarray(operand, np.float64) for operand in operands)))
    return evaluate


def scaled(layout: str):
    """The reference program's rounding scale at a draw (tests/rounding_bound.py), its recorder's own, each
    seed's once: the two recordings of a layout are of the same draws (`inputs_of(seed)`), so the second
    reads the first's."""
    held = {}

    def scale(inputs, seed):
        if seed not in held:
            held[seed] = rounding_scale(LEAFED[layout], *(inputs[name] for name in NAMES), seed=seed)[1]
        return held[seed]
    return scale


@pytest.fixture(scope="module", params=list(LAYOUTS))
def recordings(request, tmp_path_factory):
    """The site, laid as the run's layout (`LAYOUTS`), recorded twice over the same draws, its outputs
    whole and its evidence alone, each saved as a fixture's file, with its declared seeds."""
    layout = request.param
    recorder, directory = load("recorded_references"), tmp_path_factory.mktemp("weighed")
    dew = jax.jit(lambda inputs: at(reformulated, layout)(jnp.float32, inputs))
    saved, scale = {"layout": layout}, scaled(layout)
    saved["scale"] = scale
    with jax.enable_x64(new_val=True):
        for name, options in (("whole", {}), ("summary", {"summary": True})):
            recorded = recorder.record_case("site", inputs_of, evaluated(layout), dew=dew, scale=scale,
                                            store_inputs=False, **options)
            np.savez(directory / f"{name}.npz", **recorded)
            saved[name] = (directory / f"{name}.npz", recorded)
    return saved


def judged(recordings, name: str, fn, label: str) -> None:
    """Check 1 of `fn` from the recording `name`, weighed where undecided. Over `WEIGHED_SEEDS`, at which
    the unweighted check leaves this operation undecided (tests/test_weighed_check.py), so the weighed
    check is the one that decides, over the recording's own declared seeds of it; at the recording's
    declared unweighted seeds, the cap, the amplified element's pooled ratio resolves by itself."""
    path, _ = recordings[name]
    assert_as_exact_as_recorded(path, "site", WEIGHED_SEEDS, inputs_of, at(fn, recordings["layout"]), label,
                                weighed=True)


def test_a_weighed_site_is_judged_from_its_evidence_alone_as_from_its_outputs(recordings):
    """The reference's own math reformulated in float32 passes the weighed check from either recording; and
    an arm whose float32 products are eight units of their rounding off fails it decided from either, at
    the same leaves by the same rules: its light class, where the unweighted check's amplified element hid
    it. The summary holds no output and no truth of the reference's. Whether the leaf is rows or one row (a
    vector, whose summary's row is read from its recorded error and sized as Dew's arm's is: train 917's
    F1)."""
    found = {}
    for name in ("whole", "summary"):
        judged(recordings, name, reformulated, f"the reformulation, from the {name} recording")
        with pytest.raises(GateFailure) as raised:
            judged(recordings, name, biased, f"a biased product, from the {name} recording")
        assert not isinstance(raised.value, Inconclusive)
        found[name] = {(finding.leaf.rpartition(", conditioned")[2], finding.rule)
                       for finding in raised.value.findings}
    assert found["summary"] == found["whole"]
    assert any("light" in leaf for leaf, _ in found["summary"])
    kept = {key.removeprefix("site.").split(".")[0] for key in recordings["summary"][1]}
    assert not kept & {"reference", "truth"} and {"scale", "heavy", "weighed", "conditioning"} <= kept


def test_a_recorded_arm_with_no_weighed_evidence_is_refused_by_name():
    """An arm that is a recording's evidence holds no element to weigh: where its conditioning brings no
    weighed evidence of the draw (`Conditioning.recorded_of`), the weighing refuses it, never judging the
    unweighted summary in a weighed arm's place."""
    truth = {"y": np.zeros((1, 4))}
    conditioning = parity.Conditioning(lambda inputs, seed: {"y": np.ones((1, 4))})
    summary = parity.leaf_summaries([np.ones((1, 4), np.float32)], [truth["y"]])
    with pytest.raises(AssertionError):
        conditioning.of(truth, (), 0).weighed({"y": summary[0]}, "the recorded reference")


def test_a_recording_declaring_its_unweighted_count_is_the_one_its_ladder_makes(recordings):
    """A weighed site whose unweighted sizing never resolves (this one's lands at the cap), recorded
    declaring that count (`record_case`'s `unweighted`): the summary its paired ladder makes, entry for entry
    (multimodal-qwen35's GPU recording climbed the ladder again every slice); a count past the cap refused."""
    layout = recordings["layout"]
    recorder = load("recorded_references")
    dew = jax.jit(lambda inputs: at(reformulated, layout)(jnp.float32, inputs))
    _, laddered = recordings["summary"]
    options = {"dew": dew, "scale": recordings["scale"], "store_inputs": False, "summary": True}
    with jax.enable_x64(new_val=True):
        declared = recorder.record_case("site", inputs_of, evaluated(layout), unweighted=parity.MOST_SEEDS,
                                        **options)
        with pytest.raises(ValueError):
            recorder.record_case("site", inputs_of, evaluated(layout), unweighted=2 * parity.MOST_SEEDS,
                                 **options)
    assert declared.keys() == laddered.keys()
    for name in laddered:
        np.testing.assert_array_equal(declared[name], laddered[name], err_msg=name)

