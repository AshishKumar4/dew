"""The release contract's record (`dew.config.Release`): the one file every export's writer states what its
format cannot beside its weights, and every loader reads back."""

import json
from pathlib import Path

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


def test_a_release_declares_one_temperature_over_every_type_in_its_files_or_its_record(tmp_path):
    """A release's global temperature is its `Temperatures`' default over every type and bucket it names
    none for: Laya's file stating one number for `temperature` answers as the same release stating it for
    each of choice, score and noul, and a release record's calibration stating one over Laya's file's answers
    as the file stating it for each; released again, the one temperature answers as it did (row 4484)."""
    from shutil import copytree

    from dew.decision import Decide, Question
    from dew.decision.calibration import Calibration, Temperatures
    from dew.registry import to_record

    laya = Path(__file__).parent / "fixtures" / "laya"
    tiny, request = laya / "tiny", json.loads((laya / "cases.json").read_text())["many_options"]
    asked = {"intent": Question.from_wire(request["questions"]["intent"])}

    def released(name: str, temperature, record=None) -> Decide:
        copytree(tiny, tmp_path / name, symlinks=True)
        config = json.loads((tiny / "rl_agent_config.json").read_text())
        stated = {**config, "temperature": temperature, "temperature_by_options": {}}
        (tmp_path / name / "rl_agent_config.json").write_text(json.dumps(stated))
        if record is not None:
            Release(calibration=to_record(record, Calibration)).write(tmp_path / name)
        return Decide.from_pretrained(tmp_path / name, attention_impl="xla")

    one, each = released("one", 1.7), released("each", [1.7, 1.7, 1.7])
    assert {one.calibration.temperatures.of(kind, 4) for kind in ("choice", "score", "noul")} == {1.7}
    assert (one(request["state"], asked)["intent"].probabilities.tolist()
            == each(request["state"], asked)["intent"].probabilities.tolist())
    # A release record's one temperature over Laya's file's three answers as the file stating 0.8 for each.
    over = released("record", [1.3, 0.9, 1.7], Calibration(Temperatures(default=0.8)))
    eight = released("eight", [0.8, 0.8, 0.8])
    assert (over(request["state"], asked)["intent"].probabilities.tolist()
            == eight(request["state"], asked)["intent"].probabilities.tolist())
    # Released again, the one temperature is written as each type's, which reads back to the same answers.
    one.save_pretrained(tmp_path / "saved")
    again = Decide.from_pretrained(tmp_path / "saved", attention_impl="xla")
    assert (again(request["state"], asked)["intent"].probabilities.tolist()
            == one(request["state"], asked)["intent"].probabilities.tolist())
