"""Check 1's recorded references (`reference_error.indexed_recording`, `recordings.whole_recording`,
tools/recorded_references.py's recorder): a recording's draws, sizing and pilots, read back as its live
reference reads them."""


import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import (
    REDRAW,
    GateFailure,
    Inconclusive,
    as_given,
    as_recorded,
    assert_as_exact,
    declared_seeds,
    indexed_recording,
    seeds_for,
    widened,
)
from test_reference_error import (
    _bits,
    _draw,
    _lost_term,
    _product,
    _reordered,
    _tiny_row,
    rounded_apart,
    rounding_apart,
)

from dew.parity import MOST_SEEDS
from dew.parity_stages import staged_units

pytestmark = pytest.mark.usefixtures("unqualified")
"""The gate's own tests: their sites are its subjects, declared at the counts they test it at."""


def test_a_whole_recording_with_pilots_is_sized_against_itself_on_its_pilots_alone():
    """A whole recording (its reference's outputs kept, `indexed_recording`) whose generator recorded pilots
    from SIZING_FROM on (`<site>.pilot.`, tools/block_override_reference.py's): its reference sized against
    itself (`seeds_for`) draws its pilots alone, none of its verdict seeds, where a whole recording read no
    pilots and such a sizing, which draws no verdict seed now, was refused (row 9005's head review)."""
    from reference_error import indexed_recording, seeds_for

    from dew.parity import SIZING_FROM
    from dew.parity_stages import REDRAW, staged_units

    def recorded(seeds):
        drawn = [_draw(seed) for seed in seeds]
        return ({"x": np.stack([x for x, _ in drawn]), "w": np.stack([w for _, w in drawn])},
                np.stack([np.asarray(x) @ np.asarray(w) for x, w in drawn]),
                np.stack([x.astype(np.float64) @ w.astype(np.float64) for x, w in drawn]))

    stored = {"site.redraw": np.asarray(REDRAW)}
    for under, seeds in (("site", range(staged_units(8))),
                         ("site.pilot", range(SIZING_FROM, SIZING_FROM + 32))):
        inputs, reference, truth = recorded(seeds)
        stored.update({f"{under}.input.{name}": value for name, value in inputs.items()})
        stored.update({f"{under}.reference.y": reference, f"{under}.truth.y": truth})
    draw, reference, truth = indexed_recording(stored, "site", 8)
    asked = []

    def counted(seed):
        asked.append(seed)
        return draw(seed)

    seeds_for(reference, counted, "a whole recording's pilots", truth_fn=truth, most=8)
    assert min(asked) >= SIZING_FROM


@pytest.mark.parametrize("missing", ["a leaf", "a row", "a shape"])
def test_a_whole_recordings_pilots_that_leave_out_a_verdict_leaf_or_row_are_refused(missing):
    """A whole recording's pilots hold every leaf of its verdict seeds, one row a pilot in each, each of the
    verdict's row shape: pilots of one leaf where the verdict holds two would size the count on that leaf
    alone, an array short of a row is no pilot, and a row of another shape another workload's (head
    review)."""
    from dew.parity import SIZING_FROM

    stored = {"site.redraw": np.asarray(REDRAW)}
    for under, count in (("site", staged_units(8)), ("site.pilot", 32)):
        seeds = range(count) if under == "site" else range(SIZING_FROM, SIZING_FROM + count)
        drawn = [_draw(seed) for seed in seeds]
        x, w = np.stack([x for x, _ in drawn]), np.stack([w for _, w in drawn])
        stored.update({f"{under}.input.x": x, f"{under}.input.w": w})
        for kind, dtype in (("reference", np.float32), ("truth", np.float64)):
            stored[f"{under}.{kind}.y"] = np.einsum("sij,sjk->sik", x, w).astype(dtype)
            if under == "site" or missing != "a leaf":
                stored[f"{under}.{kind}.z"] = stored[f"{under}.{kind}.y"]
    if missing == "a row":
        stored["site.pilot.truth.z"] = stored["site.pilot.truth.z"][:-1]
    if missing == "a shape":
        for kind in ("reference", "truth"):
            stored[f"site.pilot.{kind}.y"] = stored[f"site.pilot.{kind}.y"][:, :1]
    with pytest.raises(AssertionError):
        indexed_recording(stored, "site", 8)


def _tiny_row_recording(live):
    """A recording's draw and evaluation of `live` (`_tiny_row`'s reference): each seed's inputs by name,
    and the reference's float32 outputs beside their float64 truth."""
    def draw(seed):
        x, w = _draw(seed)
        return {"x": x, "w": w}

    def evaluate(inputs):
        with jax.enable_x64(new_val=False):
            reference = jax.tree.map(np.asarray, live(inputs["x"], inputs["w"]))
        truth = jax.tree.map(np.asarray, live(*(np.asarray(inputs[n], np.float64) for n in ("x", "w"))))
        return reference, truth
    return draw, evaluate


def test_a_summarized_recording_keeps_the_zero_rows_rule_for_its_small_rows():
    """A recording that keeps its reference's evidence and its truth's sketch alone (`record_case`'s
    summary) keeps no row's own size, and its sketch holds the truth at its leaf's scale, not a tiny
    row's: its rows below RESOLUTION keep the zero rows' rule, so Dew four float32 units off on the row
    the reference holds exactly fails there, where the live reference judges the row at its own size."""
    from tool_support import load

    recorder = load("recorded_references")
    dew, live = _tiny_row(4)
    draw, evaluate = _tiny_row_recording(live)

    def as_inputs(fn):
        return lambda inputs: fn(inputs["x"], inputs["w"])

    with jax.enable_x64(new_val=True):
        recorded = recorder.record_case("site", draw, evaluate, summary=True)
    seeds = int(recorded["site.seeds"])
    assert_as_exact(dew, live, _draw, "live", seeds=seeds)
    drawn, reference, truth = indexed_recording(recorded, "site", seeds, inputs_of=draw,
                                                truth_fn=as_inputs(live))
    with pytest.raises(AssertionError) as raised:
        assert_as_exact(lambda index, inputs: as_inputs(dew)(inputs), reference, drawn, "summarized",
                        seeds=seeds, truth_fn=truth)
    assert raised.type is GateFailure


def test_a_recording_keeps_the_larger_count_sized_on_another_backend():
    """A reference recorded on a CPU keeps the larger count a GPU's held-out paired sizing declared,
    including every draw its second stage may read."""
    from tool_support import load

    recorder = load("recorded_references")
    draw, evaluate = _tiny_row_recording(_product)

    def product(inputs):
        return _product(inputs["x"], inputs["w"])

    with jax.enable_x64(new_val=True):
        recorded = recorder.record_case("site", draw, evaluate, summary=True, start=256)
    seeds = int(recorded["site.seeds"])
    assert seeds >= 256
    assert len(recorded["site.input_digest"]) == staged_units(seeds)
    drawn, reference, truth = indexed_recording(recorded, "site", seeds, inputs_of=draw,
                                                truth_fn=product)
    assert_as_exact(lambda index, inputs: product(inputs), reference, drawn, "larger backend",
                    seeds=seeds, truth_fn=truth)


def test_a_recording_judged_is_held_to_dews_own_run_of_the_same_weights(tmp_path):
    """A recording of another library running what Dew exports (`record_case`'s and
    `assert_as_exact_as_recorded`'s `judged`) is the arm check 1 judges, Dew's own run its reference, its
    seeds sized with the two in those places: a library that sums nearer float64 than Dew's run is on par,
    where Dew's run judged against it is not, and an export read one part in ten thousand off is over."""
    from recorded_runs import assert_as_exact_as_recorded
    from tool_support import load

    def draw(seed):
        x, w = _draw(seed)
        return {"x": x, "w": w}

    def read(inputs):
        # The other library's run of the export: the product summed in float64 and rounded once.
        truth = np.asarray(inputs["x"], np.float64) @ np.asarray(inputs["w"], np.float64)
        return {"y": truth.astype(np.float32)}, {"y": truth}

    def dew(dtype, inputs):
        return {"y": np.asarray(inputs["x"], dtype) @ np.asarray(inputs["w"], dtype)}

    def held(name, label, arrays, **roles):
        np.savez(tmp_path / name, **arrays)
        assert_as_exact_as_recorded(tmp_path / name, "site", int(arrays["site.seeds"]), draw, dew, label,
                                    host=True, **roles)

    def recorded(**roles):
        return load("recorded_references").record_case(
            "site", draw, read, store_inputs=False, dew=lambda inputs: dew(np.float32, inputs), **roles)

    judged = recorded(judged=True)
    held("read.npz", "the export read", judged, judged=True)
    with pytest.raises(GateFailure):
        held("reference.npz", "Dew's run judged", recorded())
    with pytest.raises(GateFailure):
        held("off.npz", "an export read off", {**judged, "site.reference.y": judged[
            "site.reference.y"] * np.float32(1 + 1e-4)}, judged=True)


def test_a_recording_made_in_shards_is_the_recording_one_run_makes():
    """A summarized recording made in shards of seeds (`record_shard`), each drawn as one run draws it,
    and joined (`merge_shards`) is the recording `record_case`'s summary makes of the same reference:
    the same declared count, evidence and sketches, whatever order the shards come in. Shards that
    leave a seed out are refused."""
    from tool_support import load

    recorder = load("recorded_references")
    _, live = _tiny_row(4)
    draw, evaluate = _tiny_row_recording(live)

    with jax.enable_x64(new_val=True):
        whole = recorder.record_case("site", draw, evaluate, summary=True)
        kept = len(whole["site.input_digest"])
        cut = kept // 3
        shards = [recorder.record_shard("site", draw, evaluate, seeds)
                  for seeds in (range(cut, 2 * kept), range(cut))]
        merged = recorder.merge_shards("site", shards)
        assert merged.keys() == whole.keys()
        for key, value in whole.items():
            np.testing.assert_array_equal(merged[key], value, err_msg=key)
        with pytest.raises(ValueError):
            recorder.merge_shards("site", [shards[0], recorder.record_shard("site", draw, evaluate,
                                                                            range(cut - 1))])


def _apart_recording():
    """A recording's draw and evaluation of `rounded_apart`'s arms, by name: each seed's value, its
    heavy-tailed scale and the arms' sizes (`rounding_apart`), the reference arm 1's outputs beside their
    float64 truth, and Dew's run arm 0's, the two rounding apart on draws whose errors are heavy-tailed alike
    in both."""
    def draw(seed):
        x, scale, sizes = rounding_apart(seed)
        return {"x": x, "scale": np.asarray(scale), "sizes": sizes}

    def arm(index):
        rounded = rounded_apart(index)
        return lambda inputs: {"y": rounded(inputs["x"], inputs["scale"], inputs["sizes"])}

    def evaluate(inputs):
        return arm(1)(inputs), {"y": np.asarray(inputs["x"], np.float64)}
    return draw, evaluate, arm(0)


def test_a_recording_made_in_shards_sizes_paired_on_pilots_holding_dews_errors():
    """A summarized recording of arms whose errors are heavy-tailed alike in both (`_apart_recording`), made
    in shards whose pilots hold Dew's errors on their draws (`record_shard`'s `dew`), declares the count
    `record_case` declares sized paired on the same draws (its `dew`), and keeps none of Dew's errors, where
    the same shards without Dew's errors leave the count unresolved; pilots that hold them on some shards
    and not others are refused (row 6835)."""
    from tool_support import load

    from dew.parity import SIZING_FROM

    recorder = load("recorded_references")
    draw, evaluate, dew = _apart_recording()
    with jax.enable_x64(new_val=True):
        whole = recorder.record_case("site", draw, evaluate, summary=True, dew=dew)
        kept = len(whole["site.input_digest"])
        pilots = [range(start, start + MOST_SEEDS) for start in (SIZING_FROM, SIZING_FROM + MOST_SEEDS)]
        verdict = recorder.record_shard("site", draw, evaluate, range(kept))
        paired = [recorder.record_shard("site", draw, evaluate, seeds, dew=dew) for seeds in pilots]
        merged = recorder.merge_shards("site", [verdict, *paired])
        for key, value in whole.items():
            np.testing.assert_array_equal(merged[key], value, err_msg=key)
        assert not any(".dew_mse." in key for key in merged)
        with pytest.raises(Inconclusive):
            recorder.merge_shards("site", [verdict, *(recorder.record_shard("site", draw, evaluate, seeds)
                                                      for seeds in pilots)])
        with pytest.raises(ValueError):
            recorder.merge_shards("site", [verdict, paired[0],
                                           recorder.record_shard("site", draw, evaluate, pilots[1])])


def test_a_recording_kept_as_row_distances_judges_as_the_live_reference_does():
    """A reference recorded as each row's RMS distance from its float64 truth alone (`record_distances`,
    for a recorder whose outputs over the seeds are too large to keep): check 1 over that recording
    passes Dew's same sum in another order and fails Dew's sum with a term lost, as it does against the
    live reference."""
    from tool_support import load

    recorder = load("recorded_references")

    def draw(seed):
        x, w = _draw(seed)
        return {"x": x, "w": w}

    def as_inputs(fn):
        return lambda inputs: fn(inputs["x"], inputs["w"])

    with jax.enable_x64(new_val=True):
        def truth_of(seed):
            return jax.tree.map(np.asarray, _product(*(np.asarray(part, np.float64) for part in _draw(seed))))

        def distances(seed):
            found = np.asarray(_product(*_draw(seed))["y"], np.float64)
            return np.sqrt(np.mean(np.square(found - truth_of(seed)["y"]), axis=1))

        kept = {"y": np.stack([distances(seed) for seed in range(160)])}
        recorded = recorder.record_distances("site", draw, kept, truth_of, {"y": "float32"})
    seeds = int(recorded["site.seeds"])
    assert_as_exact(_reordered, _product, _draw, "live", seeds=seeds)
    drawn, reference, truth = indexed_recording(recorded, "site", seeds, inputs_of=draw,
                                                truth_fn=as_inputs(_product))
    assert_as_exact(lambda index, inputs: as_inputs(_reordered)(inputs), reference, drawn, "reordered",
                    seeds=seeds, truth_fn=truth)
    with pytest.raises(AssertionError) as raised:
        assert_as_exact(lambda index, inputs: as_inputs(_lost_term)(inputs), reference, drawn, "lost term",
                        seeds=seeds, truth_fn=truth)
    assert raised.type is GateFailure


def test_a_recording_with_pilot_shards_sizes_paired_on_them_as_the_live_draws_do():
    """A summarized recording made in shards with a run of pilots from SIZING_FROM (`merge_shards`) keeps
    them apart from its verdict's seeds, and its reference carries them: a paired sizing of it takes those
    pilots alone and declares what the live reference's paired sizing declares on the same draws, its
    lookups never past them. Pilots that leave a seed out of their run, or whose leaves are not the
    verdict's shapes, are refused, and so are a seed past them and a pilot drawn again otherwise."""
    from tool_support import load

    from dew.parity import SIZING_FROM

    recorder = load("recorded_references")
    draw, evaluate = _tiny_row_recording(_product)

    def product(inputs):
        return _product(inputs["x"], inputs["w"])

    # The product summed in the other order: Dew's own rounding, apart from the reference's on each draw.
    reordered = jax.jit(lambda x, w: {"y": jnp.dot(x[:, ::-1], w[::-1], precision="highest")})
    live = seeds_for(_product, _draw, "live", dew_fn=reordered)
    pilots = range(SIZING_FROM, SIZING_FROM + 2 * live)

    with jax.enable_x64(new_val=True):
        whole = recorder.record_case("site", draw, evaluate, summary=True)
        kept = len(whole["site.input_digest"])
        cut = SIZING_FROM + live // 2
        shards = [recorder.record_shard("site", draw, evaluate, seeds)
                  for seeds in (range(cut, pilots.stop), range(kept), range(SIZING_FROM, cut))]
        merged = recorder.merge_shards("site", shards)
        for key, value in whole.items():
            np.testing.assert_array_equal(merged[key], value, err_msg=key)
        assert len(merged["site.pilot.input_digest"]) == len(pilots)
        with pytest.raises(ValueError):
            recorder.merge_shards("site", shards[:2])
        # One pilot shard's leaf apart from the others', and every pilot's apart from the verdict's.
        for at in ((0,), (0, 2)):
            narrow = [dict(shard, **{"site.shape.y": np.asarray([1, 1], np.int64)}) if index in at else shard
                      for index, shard in enumerate(shards)]
            with pytest.raises(ValueError):
                recorder.merge_shards("site", narrow)
        # Pilots of another leaf than the verdict's.
        renamed = [{key.removesuffix(".y") + ".z" if key.endswith(".y") else key: value
                    for key, value in shard.items()} for shard in shards[::2]]
        with pytest.raises(ValueError):
            recorder.merge_shards("site", [shards[1], *renamed])
        straddling = recorder.record_shard("site", draw, evaluate, range(SIZING_FROM - 1, SIZING_FROM + 1))
        with pytest.raises(ValueError):
            recorder.merge_shards("site", [*shards, straddling])
        # Pilots joined onto a recording made whole before them are the pilots merged with its shards.
        joined = recorder.with_pilots("site", whole, shards[::2])
        assert joined.keys() == merged.keys()
        for key, value in merged.items():
            np.testing.assert_array_equal(joined[key], value, err_msg=key)
        with pytest.raises(ValueError):
            recorder.with_pilots("site", joined, shards[::2])

        seeds = int(merged["site.seeds"])
        drawn, reference, truth = indexed_recording(merged, "site", seeds, inputs_of=draw, truth_fn=product)
    assert reference.pilots == pilots
    asked = []

    def recorded(index, inputs):
        asked.append(int(index))
        return reference(index, inputs)
    as_recorded(recorded, reference.pilots)

    def dew(index, inputs):
        return reordered(inputs["x"], inputs["w"])

    assert seeds_for(recorded, drawn, "recorded", dew_fn=dew, truth_fn=truth) == live
    assert min(asked) >= SIZING_FROM and max(asked) < pilots.stop
    with pytest.raises(AssertionError):
        drawn(pilots.stop)
    with pytest.raises(AssertionError):
        indexed_recording({**merged, "site.pilot.input_digest": merged["site.pilot.input_digest"][::-1]},
                          "site", seeds, inputs_of=draw, truth_fn=product)[0](SIZING_FROM)


def test_a_recording_read_for_some_of_its_leaves_judges_them_alone():
    """A summarized recording read for some of its leaves (`indexed_recording(..., leaves=)`) holds those
    leaves' evidence and sketches, its pilots' too, and every entry the recording shares: its reference
    gives those leaves alone, at a verdict seed and a pilot's, and check 1 judges them over it; a leaf the
    recording lacks is refused."""
    from tool_support import load

    from dew.parity import SIZING_FROM

    recorder = load("recorded_references")

    @jax.jit
    def both(x, w):
        y = jnp.dot(x, w, precision="highest")
        return {"y": y, "z": 2 * y}

    def one(inputs):
        return {"y": both(inputs["x"], inputs["w"])["y"]}

    draw, evaluate = _tiny_row_recording(both)
    with jax.enable_x64(new_val=True):
        whole = recorder.record_case("site", draw, evaluate, summary=True)
        recorded = recorder.with_pilots("site", whole, [recorder.record_shard(
            "site", draw, evaluate, range(SIZING_FROM, SIZING_FROM + 16))])
        seeds = int(recorded["site.seeds"])
        drawn, reference, truth = indexed_recording(recorded, "site", seeds, inputs_of=draw, truth_fn=one,
                                                    leaves=("y",))
        for seed in (0, SIZING_FROM):
            assert set(reference(*drawn(seed))) == {"y"}
        with pytest.raises(ValueError):
            indexed_recording(recorded, "site", seeds, inputs_of=draw, truth_fn=one, leaves=("y", "w"))
    assert_as_exact(lambda index, inputs: one(inputs), reference, drawn, "leaves", seeds=seeds,
                    truth_fn=truth)


def test_a_recorded_truth_over_windows_counts_one_window_and_checks_every_one():
    """A summarized recording of windows read back through `indexed_recording` with a truth over windows
    (`over_windows`): its rounding count traces one window, as one draw's program, so the truth is read
    where a loop that stacks its windows on the host cannot be traced whole, and every window's truth is
    checked against the recording, a change confined to the last window refused."""
    from reference_error import _one_draws_program, chain_roundings, over_windows, windows
    from tool_support import load

    recorder = load("recorded_references")

    def product(inputs):
        return {"y": jnp.dot(inputs["x"], inputs["w"], precision="highest")}

    draw = windows(lambda seed: dict(zip(("x", "w"), _draw(seed), strict=True)), 4)

    def evaluate(inputs):
        with jax.enable_x64(new_val=False):
            reference = over_windows(product)(inputs)
        return reference, over_windows(product)(widened(inputs))

    one = dict(zip(("x", "w"), _draw(0), strict=True))
    program, arguments = _one_draws_program(over_windows(product), (draw(0),))
    assert program is product
    counted = chain_roundings(jax.make_jaxpr(program)(*arguments))
    assert counted == chain_roundings(jax.make_jaxpr(product)(one))
    last = draw(0)["x"][3, 0, 0]

    def moved_last(inputs):
        # The last window's truth moved by a millionth, every other window's as it is.
        return {"y": product(inputs)["y"] * jnp.where(inputs["x"][0, 0] == last, 1 + 1e-6, 1.0)}

    with jax.enable_x64(new_val=True):
        recorded = recorder.record_case("site", draw, evaluate, summary=True)
        seeds = int(recorded["site.seeds"])
        for truth_fn, holds in ((product, True), (moved_last, False)):
            _, _, truth = indexed_recording(recorded, "site", seeds, inputs_of=draw,
                                            truth_fn=over_windows(truth_fn))
            if holds:
                truth(0, widened(draw(0)))
            else:
                with pytest.raises(AssertionError):
                    truth(0, widened(draw(0)))


def _recorded_pairs(dew, reference, count: int, missing=()):
    """`dew` and `reference` recorded at `count` pilot seeds from SIZING_FROM on but the `missing` ones,
    each looked up by its seed, with a truth of zero, and the seeds the reference was asked for."""
    from dew.parity import SIZING_FROM

    pilots = range(SIZING_FROM, SIZING_FROM + count)
    mine = {seed: dew(seed) for seed in pilots}
    theirs = {seed: reference(seed) for seed in pilots if seed not in missing}
    asked = []

    def recorded(seed):
        asked.append(int(seed))
        return theirs[int(seed)]

    @as_given
    def recorded_dew(seed):
        return mine[int(seed)]

    def truth(seed):
        return {"y": np.zeros(64, np.float64)}

    return as_recorded(recorded), recorded_dew, truth, asked


def test_a_recorded_reference_with_pilots_sizes_paired_on_them_alone():
    """A recorded reference sizes paired only on pilot draws it holds (`recorded_pilots`, a view of its
    own), from SIZING_FROM on, never past them: an error law that never resolves exhausts the 24 pilots
    held at its last doubling (16), and no lookup reaches past them. One without pilots is refused, a
    view leaves its reference as it was, a live reference cannot hold pilots, a declaration of pilots
    the recording lacks is refused at once, and a start past the pilots is refused, not unbound."""
    from recordings import recorded_pilots

    from dew.parity import SIZING_FROM

    def draw(seed):
        return (np.int64(seed),)

    def straddling(seed):
        return {"y": np.full(64, 2.0 if seed % 2 == 0 else 0.1, np.float32)}

    def unit(seed):
        return {"y": np.ones(64, np.float32)}

    reference, dew, truth, asked = _recorded_pairs(straddling, unit, 24)
    with pytest.raises(ValueError):
        seeds_for(reference, draw, "no pilots", dew_fn=dew, truth_fn=truth)
    held = recorded_pilots(reference, draw, 24)
    assert not hasattr(reference, "pilots") and held.pilots == range(SIZING_FROM, SIZING_FROM + 24)
    with pytest.raises(AssertionError):
        seeds_for(held, draw, "never resolving", dew_fn=dew, truth_fn=truth)
    assert max(asked) < SIZING_FROM + 24
    with pytest.raises(KeyError):
        recorded_pilots(reference, draw, 25)
    holed, _, _, _ = _recorded_pairs(unit, unit, 24, missing={SIZING_FROM + 5})
    with pytest.raises(KeyError):
        recorded_pilots(holed, draw, 24)
    # Arms that agree exactly resolve at once, at the ceiling itself: half the pilots, the other half
    # its confirming block.
    same, same_dew, zero, _ = _recorded_pairs(unit, unit, 48)
    assert seeds_for(recorded_pilots(same, draw, 48), draw, "at the ceiling", start=24, dew_fn=same_dew,
                     truth_fn=zero) == 24
    with pytest.raises(AssertionError):
        seeds_for(held, draw, "start past the pilots", start=32, dew_fn=dew, truth_fn=truth)
    with pytest.raises(ValueError):
        recorded_pilots(_product, draw, 8)


def test_a_recorded_reference_is_drawn_by_its_index_and_covers_the_redraw(tmp_path):
    """A reference recorded over (1 + REDRAW)·K seeds: each draw reads its seed's inputs, the reference
    and truth read their recorded outputs at it, and check 1 judges Dew against them; a recording too
    short for the redraw is refused before anything is judged."""
    rng = np.random.default_rng(9)
    count, seeds = 40, 8
    x = rng.normal(size=(count, 32)).astype(np.float32)
    # x·x - 1, rounded twice at fp32 alike by NumPy and XLA (XLA's tanh is about 1.6 times libm's error
    # on a CPU, so a transcendental would compare the two libraries, not the recording).
    truth = x.astype(np.float64) ** 2 - 1
    path = tmp_path / "recorded.npz"
    np.savez(path, **{"site.input.x": x, "site.reference.y": x * x - np.float32(1),
                      "site.truth.y": truth, "site.seeds": np.int64(seeds),
                      "site.redraw": np.int64(REDRAW)})
    assert declared_seeds(path) == {"site": seeds}
    with np.load(path) as stored:
        draw, reference, truth_fn = indexed_recording(dict(stored), "site", seeds)
        squared = jax.jit(lambda index, inputs: {"y": inputs["x"] * inputs["x"] - 1})
        assert_as_exact(squared, reference, draw, "recorded", seeds=seeds, truth_fn=truth_fn)
        with pytest.raises(AssertionError):
            indexed_recording(dict(stored), "site", 16)
        for narrow in (np.float32, jnp.bfloat16):
            with pytest.raises(AssertionError):
                indexed_recording({**stored, "site.truth.y": truth.astype(narrow)}, "site", seeds)


def _squared_case(first: int):
    """A recording test's draw, 64 normal fp32 values a seed from `first` on, and its evaluation: the
    reference's `x x - 1` in fp32 beside its float64 truth."""
    def draw(seed):
        return {"x": np.random.default_rng(first + seed).normal(size=(64,)).astype(np.float32)}

    def evaluate(inputs):
        x = inputs["x"]
        return {"y": x * x - np.float32(1)}, {"y": x.astype(np.float64) ** 2 - 1}

    return draw, evaluate


def test_a_recording_declares_the_count_its_recorded_reference_resolves(tmp_path):
    """tools/recorded_references.py records a reference over the seeds its own recording resolves, and
    their redraw; check 1 then judges Dew against the recording by its index."""
    from tool_support import load

    recorder = load("recorded_references")

    draw, evaluate = _squared_case(100)

    def squared(inputs):
        return {"y": inputs["x"] * inputs["x"] - 1}

    recorded = recorder.record_case("site", draw, evaluate)
    seeds = int(recorded["site.seeds"])
    assert seeds >= 8 and len(recorded["site.input.x"]) == 5 * seeds
    path = tmp_path / "recorded.npz"
    np.savez(path, **recorded)
    assert declared_seeds(path) == {"site": seeds}
    with np.load(path) as stored:
        draw_recorded, reference, truth = indexed_recording(dict(stored), "site", seeds)
        assert_as_exact(jax.jit(lambda index, inputs: squared(inputs)), reference, draw_recorded, "recorded",
                        seeds=seeds, truth_fn=truth)
        with pytest.raises(AssertionError):
            assert_as_exact(lambda index, inputs: {"y": jnp.asarray(inputs["x"]) ** 2}, reference,
                            draw_recorded, "recorded with its subtraction missing", seeds=seeds,
                            truth_fn=truth)


def test_a_paired_recording_holds_its_sizing_draws_one_at_a_time():
    """A paired recording (`record_case`'s `dew`) reads each of its sizing's pilot draws as it draws it and
    never again, so its arrays do not pile up across the sizing: while one is evaluated, at most the one
    before it is still held."""
    import weakref

    from reference_error import SIZING_FROM
    from tool_support import load

    recorder = load("recorded_references")
    draw, evaluate = _squared_case(300)
    drawn, pilots, alive = [], [], []

    def drawing(seed):
        drawn.append(seed)
        return draw(seed)

    def evaluating(inputs):
        reference, truth = evaluate(inputs)
        if drawn[-1] >= SIZING_FROM:
            alive.append(sum(held() is not None for held in pilots))
            pilots.append(weakref.ref(reference["y"]))
        return reference, truth

    recorder.record_case("site", drawing, evaluating, summary=True,
                         dew=jax.jit(lambda inputs: {"y": inputs["x"] * inputs["x"] - 1}))
    assert len(pilots) > 2 and max(alive) <= 1


def test_a_recording_of_outputs_alone_draws_its_inputs_again(tmp_path):
    """A generator's seeded draw, called again by the test (`inputs_of`), stands in for recorded inputs:
    the recording holds the outputs alone and is judged as one that holds both. A draw that changed since
    the recording is refused by its digest, as is `inputs_of` on a recording with no digest."""
    from tool_support import load

    recorder = load("recorded_references")

    draw, evaluate = _squared_case(300)

    def squared(inputs):
        return {"y": inputs["x"] * inputs["x"] - 1}

    recorded = recorder.record_case("site", draw, evaluate, store_inputs=False)
    assert not any(name.startswith("site.input.") for name in recorded)
    seeds = int(recorded["site.seeds"])
    with pytest.raises(AssertionError):
        indexed_recording(recorded, "site", seeds)
    drawn, reference, truth = indexed_recording(recorded, "site", seeds, inputs_of=draw)
    assert_as_exact(jax.jit(lambda index, inputs: squared(inputs)), reference, drawn, "drawn again",
                    seeds=seeds, truth_fn=truth)
    changed, _, _ = indexed_recording(recorded, "site", seeds, inputs_of=lambda seed: draw(seed + 1))
    with pytest.raises(AssertionError):
        changed(0)
    undigested = {name: value for name, value in recorded.items() if name != "site.input_digest"}
    with pytest.raises(AssertionError):
        indexed_recording(undigested, "site", seeds, inputs_of=draw)


def test_a_summarized_recording_reads_back_as_its_reference_reads_live():
    """A summarized recording's evidence (`record_case`'s summary), read back as check 1 reads it
    (`indexed_recording`'s reference), is the evidence of the same reference outputs read live
    (`leaf_summary`), bit for bit in every field the recording keeps, its rows of float64 zero included,
    its squared error kept as the value of the log read live and read back as that value's log."""
    from tool_support import load

    from dew.parity import leaf_summary

    recorder = load("recorded_references")
    _, live = _tiny_row(4)
    draw, evaluate = _tiny_row_recording(live)

    def truth_fn(inputs):
        return live(inputs["x"], inputs["w"])

    with jax.enable_x64(new_val=True):
        recorded = recorder.record_case("site", draw, evaluate, summary=True)
        _, reference, _ = indexed_recording(recorded, "site", int(recorded["site.seeds"]), inputs_of=draw,
                                            truth_fn=truth_fn)
        zero_rows = 0
        for seed in range(len(recorded["site.input_digest"])):
            outputs, truth = evaluate(draw(seed))
            for name, kept in reference(np.int32(seed), draw(seed)).items():
                read = leaf_summary(outputs[name], truth[name])
                fields = ("worst", "share", "zero_rows", "zero_errors", "finite", "dtype")
                assert _bits([getattr(kept, field) for field in fields]) == _bits(
                    [getattr(read, field) for field in fields]), (seed, name)
                # The squared error the recording keeps as its value, and reads back as its log.
                stored = recorded[f"site.mse.{name}"][seed]
                assert _bits([stored, kept.log_mse]) == _bits([np.exp(read.log_mse), np.log(stored)])
                zero_rows += len(kept.zero_rows)
    assert zero_rows
