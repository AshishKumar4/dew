"""Saved run records load as the code that wrote them meant them.

The fixtures are dewml/hybrid-dit-176m's run.json as published at two
commits: 3f11a17 predates EDM's `regime` and dfa94d6 carries it; both
predate `audio` and record versions.
"""

import json
from pathlib import Path

import pytest

from dew.config import migrations
from dew.objectives.diffusion import DiffusionRunConfig

RUNS = Path(__file__).resolve().parent / "fixtures" / "runs"


def record(name: str) -> dict:
    return json.loads((RUNS / name / "run.json").read_text())


def test_a_run_from_before_the_regime_trains_on_the_sigmas_it_recorded():
    """The record states EDM2's P_mean -0.4 and P_std 1.0, which the code
    that wrote it drew from; they override the regime the run now fills."""
    run = DiffusionRunConfig.load(str(RUNS / "hybrid-dit-176m-3f11a17"))
    assert run.audio is None
    assert (run.preset.P_mean, run.preset.P_std) == (-0.4, 1.0)
    assert run.preset.lognormal() == (-0.4, 1.0)


def test_a_run_with_a_regime_keeps_it():
    run = DiffusionRunConfig.load(str(RUNS / "hybrid-dit-176m-dfa94d6"))
    assert run.audio is None
    assert run.preset.regime == "latent" and run.preset.lognormal() == (-0.4, 1.0)


def test_a_migrated_run_writes_the_current_version_and_reads_back_equal():
    run = DiffusionRunConfig.from_dict(record("hybrid-dit-176m-dfa94d6"))
    written = json.loads(json.dumps(run.to_dict()))
    assert written["version"] == migrations.VERSION
    assert DiffusionRunConfig.from_dict(written) == run


def test_a_current_record_missing_a_field_is_refused():
    """Migration speaks for older versions only: a record at today's version
    that lacks a field is as incomplete as it was before versions."""
    written = DiffusionRunConfig.from_dict(record("hybrid-dit-176m-dfa94d6")).to_dict()
    del written["audio"]
    with pytest.raises(ValueError, match=r"missing fields \['audio'\]"):
        DiffusionRunConfig.from_dict(written)


@pytest.mark.parametrize("version", [migrations.VERSION + 1, -1, "1"])
def test_a_record_of_an_unknown_version_is_refused(version):
    with pytest.raises(ValueError, match="version"):
        DiffusionRunConfig.from_dict({**record("hybrid-dit-176m-dfa94d6"), "version": version})
