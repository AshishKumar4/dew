"""Runs saved by earlier mains, read at this one (row 2282).

The corpus (tests/fixtures/saved_runs, written by tools/saved_runs_reference.py) holds, for each main that
recorded it, a saved run of each run kind: its run.json and its last checkpoint, a recording that is
committed or pinned in the fetched store (`reference_error.recording`). This main reads each as a user's
process would: the record rebuilds its run (`RunConfig.load`), the checkpoint's inference declaration pins
(`SavedRun.pinned`) where its objective declares one, the kind's saved task loads it (`from_run`) and
answers one call, and the run resumes a step past its last. Its data is the kind's own, written here, since
a record names its data where the run that wrote it read them."""

import dataclasses
import io
import json
import tarfile
from pathlib import Path

import pytest
from reference_error import recording
from test_run_options import KINDS

from dew.checkpoints import Checkpoints
from dew.config import RunConfig
from dew.inference.tasks import SavedRun, TextGeneration

CORPUS = Path(__file__).with_name("fixtures") / "saved_runs"
SAVED = [(recorded.name, kind) for recorded in sorted(CORPUS.iterdir()) if recorded.is_dir()
         for kind in json.loads((recorded / "source.json").read_text())["kinds"]]
"""Each saved run: the main that recorded it, by its abbreviated commit, and its kind."""


def called(task) -> None:
    """One call of a loaded task, the one its kind's runs make: a language model continues one prompt by
    one token, through the tokenizer its run recorded; every other kind's task is answered by loading."""
    if isinstance(task, TextGeneration):
        assert len(task.decode(task("ab", max_new_tokens=1, key=0))) == 1


@pytest.mark.parametrize(("recorded", "kind"), SAVED, ids=[f"{recorded}-{kind}" for recorded, kind in SAVED])
def test_a_run_an_earlier_main_saved_loads_answers_and_resumes(tmp_path, recorded, kind):
    """A kind's run saved at an earlier main rebuilds from its record, pins its last checkpoint, loads as
    its saved task and answers a call, and resumes from that checkpoint to a step past it."""
    runs = tmp_path / "runs"
    stored = recording(f"tests/fixtures/saved_runs/{recorded}/{kind}.npz")["run"]
    with tarfile.open(fileobj=io.BytesIO(stored.tobytes()), mode="r:xz") as packed:
        packed.extractall(runs, filter="data")
    run = runs / kind
    with Checkpoints(str(run)) as checkpoints:
        last = checkpoints.latest
        declared = checkpoints.artifact(last) is not None
    # A run whose objective declares no inference record (a JEPA encoder's) loads by its resume alone.
    if declared:
        saved = SavedRun.pinned(str(run))
        assert saved.step == last
        called(saved.objective.saved_task.from_run(str(run)))
    recorded_run = RunConfig.load(str(run))
    here = KINDS[kind](tmp_path, drawn=False)
    sources = {name: getattr(here, name) for name in ("data", "validation", "pretrained")
               if hasattr(here, name)}
    trainer = dataclasses.replace(recorded_run.trainer, checkpoint_dir=str(runs), steps=last + 1,
                                  compilation_cache_dir=None, multi_host=False)
    state = dataclasses.replace(recorded_run, trainer=trainer, **sources).prepare().train(kind)
    assert int(state.step) == last + 1
