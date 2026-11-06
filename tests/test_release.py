"""The release contract's record (`dew.config.Release`): the one file every export's writer states what its
format cannot beside its weights, and every loader reads back."""

import json

import pytest

from dew.config import Release
from dew.config.release import RELEASE_FILE


def test_a_release_writes_the_fields_it_states_and_reads_them_back(tmp_path):
    """A release holds a task declaration, a decision layout and calibration and a window, each a record
    or a count; what it states is written as JSON beside the export's config and read back whole, and
    what it leaves out is no key of the file."""
    stated = Release(task={"settings": {"seq_len": 8}, "dtypes": {"compute": "float32"}},
                     layout={"class": "dew.decision.layout:JointLayout", "fields": {"max_len": 2048}},
                     max_length=2048, calibration={"temperatures": {}})
    stated.write(tmp_path)
    assert Release.load(tmp_path) == stated
    Release(max_length=4096).write(tmp_path)
    assert json.loads((tmp_path / RELEASE_FILE).read_text()) == {"max_length": 4096}
    assert Release.load(tmp_path) == Release(max_length=4096)


def test_a_release_that_states_nothing_writes_no_file_and_replaces_an_earlier_exports(tmp_path):
    """An export without a run's declaration leaves none of an earlier export of its directory behind, and a
    directory no run wrote reads as a source with none (None), not as a release that states nothing."""
    assert Release.load(tmp_path) is None
    Release(max_length=512).write(tmp_path)
    Release().write(tmp_path)
    assert not (tmp_path / RELEASE_FILE).exists()
    assert Release.load(tmp_path) is None


@pytest.mark.parametrize("record", [{"layouts": {}}, {"task": 3}, {"max_length": 0}, {"max_length": True},
                                    {"layout": []}])
def test_a_release_record_of_another_shape_is_refused_naming_what_it_holds(record):
    """A record that states a field a release does not hold, or a field of another shape, is refused
    where it is read, naming the fields a release holds, rather than read as one that states less."""
    with pytest.raises(ValueError):
        Release.read(record)
