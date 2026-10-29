"""Runs saved by earlier mains, read at this one (row 2282).

The corpus (tests/fixtures/saved_runs, written by tools/saved_runs_reference.py) holds, for each main that
recorded it, a saved run of each run kind: its run.json and its checkpoints. This main reads each as a user's
process would: the record rebuilds its run (`RunConfig.load`), the checkpoint's inference declaration pins
(`SavedRun.pinned`), the kind's saved task loads it (`from_run`) and answers one call, and the run resumes a
step past its last. Its data is the kind's own, written here, since a record names its data where the run
that wrote it read them."""

import dataclasses
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
from test_run_options import KINDS

from dew.config import RunConfig
from dew.inference.tasks import SavedRun, TextGeneration

CORPUS = Path(__file__).with_name("fixtures") / "saved_runs"
SAVED = [(recorded.name, kind) for recorded in sorted(CORPUS.iterdir()) if recorded.is_dir()
         for kind in json.loads((recorded / "source.json").read_text())["kinds"]]
"""Each saved run: the main that recorded it, by its abbreviated commit, and its kind."""


def called(task) -> None:
    """One call of a loaded task, the one its kind's runs make: a language model continues a prompt of
    token ids by a token, which it draws in its vocabulary; every other kind's task is answered by
    loading."""
    if isinstance(task, TextGeneration):
        drawn = task([[1, 2, 3]], max_new_tokens=1, key=0)
        assert np.asarray(drawn.lengths).tolist() == [1]


@pytest.mark.parametrize(("recorded", "kind"), SAVED, ids=[f"{recorded}-{kind}" for recorded, kind in SAVED])
def test_a_run_an_earlier_main_saved_loads_answers_and_resumes(tmp_path, recorded, kind):
    """A kind's run saved at an earlier main rebuilds from its record, pins its last checkpoint, loads as
    its saved task and answers a call, and resumes from that checkpoint to a step past it."""
    runs = tmp_path / "runs"
    with tarfile.open(CORPUS / recorded / f"{kind}.tar.xz") as packed:
        packed.extractall(runs, filter="data")
    run = runs / kind
    saved = SavedRun.pinned(str(run))
    if saved.objective.saved_task is not None:
        called(saved.objective.saved_task.from_run(str(run)))
    recorded_run = RunConfig.load(str(run))
    here = KINDS[kind](tmp_path, drawn=False)
    sources = {name: getattr(here, name) for name in ("data", "validation", "pretrained")
               if hasattr(here, name)}
    trainer = dataclasses.replace(recorded_run.trainer, checkpoint_dir=str(runs), steps=saved.step + 1,
                                  compilation_cache_dir=None, multi_host=False)
    state = dataclasses.replace(recorded_run, trainer=trainer, **sources).prepare().train(kind)
    assert int(state.step) == saved.step + 1
