"""Runs saved by earlier mains, read at this one (row 2282).

The corpus (tests/fixtures/saved_runs, written by tools/saved_runs_reference.py) holds, for each main that
recorded it, a saved run of each run kind (`saved_run_kinds.SAVED_KINDS`): its run.json and its last
checkpoint, a recording that is committed or pinned in the fetched store (`reference_error.recording`).
This main reads each as a user's process would: the record rebuilds its run (`RunConfig.load`), the
checkpoint's inference declaration pins (`SavedRun.pinned`) where its objective declares a saved task,
which loads it (`from_run`) and answers one request, and the run resumes from its checkpoint for one step
more. Its data is the kind's own, written here, since a record names its data where the run that wrote it
read them."""

import dataclasses
import io
import json
import re
import tarfile
from pathlib import Path

import pytest
from reference_error import recording
from saved_run_kinds import SAVED_KINDS

from dew.checkpoints import Checkpoints
from dew.config import RunConfig
from dew.decision import Choice
from dew.decision.task import Decide
from dew.inference.images import TextToImage
from dew.inference.tasks import SavedRun, TextGeneration
from dew.registry import objectives

CORPUS = Path(__file__).with_name("fixtures") / "saved_runs"
SAVED = [(recorded.name, kind) for recorded in sorted(CORPUS.iterdir()) if recorded.is_dir()
         for kind in json.loads((recorded / "source.json").read_text())["kinds"]]
"""Each saved run: the main that recorded it, by its abbreviated commit, and its kind."""

IDS = [f"{recorded}-{kind}" for recorded, kind in SAVED]

RESUMES_WITHIN_ITS_REVISION = "decision"
"""The kind whose chain reads per-share, whose plan resumes only under the Dew revision that saved it
(`dew.data.plan.Plan.check`): its resume fails here, marked so, until ledger 2382's fix resumes it."""


ABSOLUTE = re.compile(rb"/(?:home|Users|root|tmp|mnt|private|var)/")
"""An absolute place of the machine that wrote a file: a home, a scratch or a mounted directory."""


def archived(recorded: str, kind: str) -> tarfile.TarFile:
    """The archive of `kind`'s run the main `recorded` saved (`tools/saved_runs_reference.py`)."""
    stored = recording(f"tests/fixtures/saved_runs/{recorded}/{kind}.npz")["run"]
    return tarfile.open(fileobj=io.BytesIO(stored.tobytes()), mode="r:xz")


@pytest.mark.parametrize(("recorded", "kind"), SAVED, ids=IDS)
def test_a_saved_run_names_no_owner_time_or_place_of_the_machine_that_wrote_it(recorded, kind):
    """Each archive of the corpus names nothing of its writer's machine, which the fetched store
    publishes: its members hold no owner, user or group, and one time, and no file of it names an
    absolute directory, a scratch directory, a checkout or a home (`saved_runs_reference.anonymous`)."""
    with archived(recorded, kind) as packed:
        members = packed.getmembers()
        assert {(member.uid, member.gid, member.uname, member.gname, member.mtime) for member in members} == {
            (0, 0, "", "", 0)}
        for member in members:
            if member.isfile():
                found = packed.extractfile(member)
                assert found is not None and not ABSOLUTE.search(found.read()), member.name


def answered(task) -> None:
    """One request of a loaded task, as its kind's users make it, and its answer: a language model
    continues a prompt by one token through the tokenizer its run recorded, a decision answers a
    question about a state, and a text-to-image run draws an image in one step."""
    if isinstance(task, TextGeneration):
        assert len(task.decode(task("ab", max_new_tokens=1, key=0))) == 1
    elif isinstance(task, Decide):
        question = {"team": Choice("which team", ["billing", "technical", "sales"])}
        ((answers, _),) = task.batch([({"text": "the site is down"}, question)])
        assert answers["team"].choice in ("billing", "technical", "sales")
    elif isinstance(task, TextToImage):
        assert len(task(["a cat"], steps=1, key=0).pil()) == 1
    else:
        raise AssertionError(f"no request is written for a saved {type(task).__name__}")


def extracted(tmp_path: Path, recorded: str, kind: str) -> Path:
    """The run directory of `kind`'s saved run, unpacked under `tmp_path`."""
    with archived(recorded, kind) as packed:
        packed.extractall(tmp_path / "runs", filter="data")
    return tmp_path / "runs" / kind


@pytest.mark.parametrize(("recorded", "kind"), SAVED, ids=IDS)
def test_a_run_an_earlier_main_saved_loads_and_answers(tmp_path, recorded, kind):
    """A kind's run saved at an earlier main rebuilds from its record, and, where its objective declares
    a saved task, pins its last checkpoint and loads as that task, which answers a request; an objective
    that declares none (a JEPA encoder's) records no inference declaration."""
    run = extracted(tmp_path, recorded, kind)
    recorded_run = RunConfig.load(str(run))
    task = objectives[recorded_run.objective.name].saved_task
    with Checkpoints(str(run)) as checkpoints:
        last = checkpoints.latest
        declared = checkpoints.artifact(last) is not None
    assert declared == (task is not None)
    if task is not None:
        saved = SavedRun.pinned(str(run))
        assert saved.step == last
        answered(task.from_run(str(run)))


@pytest.mark.parametrize(("recorded", "kind"), [
    pytest.param(recorded, kind, marks=pytest.mark.xfail(kind == RESUMES_WITHIN_ITS_REVISION, strict=True,
                                                         reason="its plan resumes within its revision, 2382"))
    for recorded, kind in SAVED], ids=IDS)
def test_a_run_an_earlier_main_saved_resumes_from_its_checkpoint(tmp_path, monkeypatch, recorded, kind):
    """A kind's run saved at an earlier main resumes from its last checkpoint for one step more: the fit
    starts at that step, from that checkpoint, and trains the one step past it."""
    run = extracted(tmp_path, recorded, kind)
    recorded_run = RunConfig.load(str(run))
    with Checkpoints(str(run)) as checkpoints:
        last = checkpoints.latest
    here = SAVED_KINDS[kind](tmp_path, drawn=False)
    # A source the repo holds is recorded by its path in the repo (a decision run's backbone), read from
    # the checkout's root as the run that wrote it read it.
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    trainer = dataclasses.replace(recorded_run.trainer, checkpoint_dir=str(tmp_path / "runs"), steps=last + 1,
                                  resume="require", compilation_cache_dir=None, multi_host=False)
    sources = {"data": here.data, "validation": here.validation}
    dataclasses.replace(recorded_run, trainer=trainer, **sources).prepare().train(kind)
    records = [json.loads(line) for line in (run / "tracking" / "records.jsonl").read_text().splitlines()]
    started = [row["value"] for row in records if row["type"] == "FitStarted"][-1]
    assert (started["start_step"], started["resumed_from"] is not None) == (last, True)
    scalars = (run / "tracking" / "scalars.jsonl").read_text().splitlines()
    stepped = [json.loads(line)["step"] for line in scalars]
    assert max(stepped) == last + 1
