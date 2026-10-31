"""A run that starts from a saved Dew run (`--pretrained <run directory>`, ledger 2713): its source is the
release `dew export` writes of the run's checkpoint (`dew.interop.saved_sources`), at the step its
selection pins, which its record names."""

import jax
import numpy as np
import pytest
from run_kinds import decision, trained

from dew.decision.config import DecisionRunConfig
from dew.inference.tasks import SavedRun
from dew.interop.saved_sources import export_run, released_run


@pytest.mark.mesh(devices=8)
def test_a_decision_run_starts_from_a_saved_decision_run_as_from_its_export(tmp_path, monkeypatch):
    """A decision run named on the command line by a saved decision run (`--pretrained <run>`,
    `--source.step latest`) starts from the weights its export does (`dew export`):
    one step from each, alike in every other option, lands every leaf bit for bit alike, and its record
    names the run it started from and the step it read. A step beside a source that is no run is
    refused."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    first = decision(tmp_path, drawn=False)
    trained(first, tmp_path, "first", trainer={"checkpoint_final": True})
    source = tmp_path / "runs" / "first"
    exported = tmp_path / "exported"
    export_run(SavedRun.pinned(str(source)), exported)

    def started(pretrained, name: str, *flags: str):
        run = DecisionRunConfig.cli(["--pretrained", str(pretrained), *flags], default=first)
        return trained(run, tmp_path, name)[0]

    from_run = started(source, "from_run", "--source.step", "latest")
    from_export = started(exported, "from_export")
    for leaf, alike in zip(jax.tree.leaves(jax.device_get(from_run.variables)),
                           jax.tree.leaves(jax.device_get(from_export.variables)), strict=True):
        np.testing.assert_array_equal(np.asarray(leaf), np.asarray(alike), strict=True)
    recorded = DecisionRunConfig.load(str(tmp_path / "runs" / "from_run"))
    assert recorded.pretrained == str(source)
    assert recorded.source.step == SavedRun.pinned(str(source)).step
    assert released_run(source, step="latest") == released_run(source)
    with pytest.raises(ValueError):
        released_run(exported, step="latest")
