"""A site's function held by its reference's own float64 truth, recorded alone at a few draws
(tools/recorded_references.py `record_truths`): where no count of draws resolves its float32 errors, the
two float64 paths still compute one function, leaf by leaf (docs/design/testing.md)."""

from reference_error import _recorded_tree, certifier


def assert_truths_as_recorded(stored, site: str, truth_fn) -> None:
    """Dew's float64 path (`truth_fn(inputs)`, a followed site's `truth_fn(decisions, inputs)` following
    the reference's recorded decisions) the reference's own float64 truth, per leaf, at every draw a truths
    recording holds (tools/recorded_references.py `record_truths`), on the draw's recorded inputs: each leaf
    within float64's rounding of the computation (`reference_error.certifier`). No statistics, no count to
    size: both paths compute one function, so a leaf past that rounding is another function (a missing
    detach, another straight-through rule, another reduction of the loss), which no float32 noise can
    hide."""
    count = int(stored[f"{site}.truths"])
    shapes, sketches = (_recorded_tree(stored, site, kind) for kind in ("shape", "sketch"))
    inputs, decided = (_recorded_tree(stored, site, kind) for kind in ("input", "decisions"))
    _, certified = certifier(site, truth_fn, shapes, lambda index, name: sketches[name][index])
    for index in range(count):
        drawn = {name: value[index] for name, value in inputs.items()}
        followed = ({name: value[index] for name, value in decided.items()},) if decided else ()
        certified(index, *followed, drawn)
