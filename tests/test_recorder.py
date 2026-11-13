"""tools/recorded_references.py's recorder (`record_case`): what it holds while it records and how a
recording goes on from what an interrupted run kept."""

import jax
import numpy as np
import pytest
from test_reference_error import _tiny_row, _tiny_row_recording

from dew.parity_stages import staged_units


def test_a_summarized_recording_holds_one_draws_outputs_at_a_time():
    """A summarized recording (`record_case`'s summary) keeps of each draw its evidence alone, read as the
    draw is made: when a seed is evaluated, of the seeds before it no truth is held but the last one's,
    which its reader may still name, and each seed is evaluated once. Every draw's inputs, reference and
    truth were held until the count was confirmed and its redraw drawn, a recording's whole outputs where
    its file keeps kilobytes."""
    import gc
    import weakref

    from tool_support import load

    recorder = load("recorded_references")
    _, live = _tiny_row(4)
    draw, evaluate = _tiny_row_recording(live)
    truths, held = [], []

    def watched(inputs):
        gc.collect()
        held.append(sum(truth() is not None for truth in truths))
        reference, truth = evaluate(inputs)
        truths.extend(weakref.ref(leaf) for leaf in jax.tree.leaves(truth))
        return reference, truth

    with jax.enable_x64(new_val=True):
        recorded = recorder.record_case("site", draw, watched, summary=True)
    leaves = len(truths) // len(held)
    assert len(held) == len(recorded["site.input_digest"]) == staged_units(int(recorded["site.seeds"]))
    assert max(held) <= leaves, held


def test_a_recording_resumed_from_what_an_interrupted_run_kept_makes_only_the_rest(tmp_path):
    """A recording that keeps each seed's evaluation as it is made (`record_case`'s `kept_in`) is resumed
    from an interrupted run's (`resumed_from`): of the seeds found there one is evaluated again, which
    proves the records this reference's, and no other; every record is kept on for a next resumption; and
    the recording made is the uninterrupted run's, array for array. A kept record of other inputs is
    refused, and so are another reference's records of the same inputs: another recording's."""
    import shutil

    from tool_support import load

    recorder = load("recorded_references")
    _, live = _tiny_row(4)
    draw, evaluate = _tiny_row_recording(live)
    evaluated = []

    def counted(inputs):
        evaluated.append(1)
        return evaluate(inputs)

    whole_kept, interrupted, resumed_kept = (tmp_path / name for name in ("whole", "interrupted", "resumed"))
    with jax.enable_x64(new_val=True):
        whole = recorder.record_case("a site", draw, counted, summary=True, kept_in=whole_kept)
        records = sorted(whole_kept.iterdir())
        assert len(records) == len(evaluated)
        interrupted.mkdir()
        for path in records[::2]:
            shutil.copy(path, interrupted / path.name)
        resumed = recorder.record_case("a site", draw, counted, summary=True, kept_in=resumed_kept,
                                       resumed_from=interrupted)
    assert len(evaluated) == len(records) + len(records[1::2]) + 1
    assert sorted(path.name for path in resumed_kept.iterdir()) == [path.name for path in records]
    assert resumed.keys() == whole.keys()
    for key, value in whole.items():
        np.testing.assert_array_equal(resumed[key], value, err_msg=key)

    def shifted(seed):
        return {name: value + 1 for name, value in draw(seed).items()}

    def another(inputs):
        reference, truth = evaluate(inputs)
        return {name: value * 2 for name, value in reference.items()}, truth

    with jax.enable_x64(new_val=True):
        for other_draw, other_reference in ((shifted, counted), (draw, another)):
            with pytest.raises(ValueError):
                recorder.record_case("a site", other_draw, other_reference, summary=True,
                                     resumed_from=whole_kept)
        # Another site's records, under another label, are none of this one's: nothing is found, all is made.
        before = len(evaluated)
        recorder.record_case("another site", draw, counted, summary=True, resumed_from=whole_kept)
    assert len(evaluated) - before == len(records)
