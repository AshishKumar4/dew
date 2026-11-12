"""Runs saved by earlier mains, read at this one (row 2282).

The corpus (tests/fixtures/saved_runs, written by tools/saved_runs_reference.py) holds, for each main that
recorded it, a saved run of each run kind (`saved_run_kinds.SAVED_KINDS`): its run.json and its last
checkpoint, a recording that is committed or pinned in the fetched store (`reference_error.recording`), and
the inputs its runs read that the repo has since changed (`INPUTS`), read as they were (`checkout_of`).
This main reads each as a user's process would: the record rebuilds its run (`RunConfig.load`), the
checkpoint's inference declaration pins (`SavedRun.pinned`) where its objective declares a saved task,
which loads it (`from_run`) and answers one request, and the run resumes from its checkpoint for one step
more. Its data is the kind's own, written here, since a record names its data where the run that wrote it
read them. What a run serialized, recorded beside it where its main recorded it (`evidence`: its
parameters' and compute dtypes, its loss scale, the examples it consumed and its step's rung), is what
this main reads back: the record's compute, and the resumed run's parameters, scale, clock and rung."""

import dataclasses
import io
import json
import re
import tarfile
from collections.abc import Callable
from pathlib import Path

import jax
import pytest
from process_support import assert_exits_cleanly
from reference_error import recording
from saved_run_kinds import SAVED_KINDS, answered, trained_finite

from dew.checkpoints import ONE_ITEM, Checkpoints
from dew.config import RunConfig
from dew.data.plan import PLAN_VERSIONS
from dew.inference.tasks import SavedRun
from dew.interop.release import computed_in
from dew.registry import objectives
from dew.training import LocalTracker

CORPUS = Path(__file__).with_name("fixtures") / "saved_runs"
SAVED = [(recorded.name, kind) for recorded in sorted(CORPUS.iterdir()) if recorded.is_dir()
         for kind in json.loads((recorded / "source.json").read_text())["kinds"]]
"""Each saved run: the main that recorded it, by its abbreviated commit, and its kind."""

IDS = [f"{recorded}-{kind}" for recorded, kind in SAVED]

ABSOLUTE = re.compile(rb"/(?:home|Users|root|tmp|mnt|private|var)/")
"""An absolute place of the machine that wrote a file: a home, a scratch or a mounted directory."""


def computes_in(task) -> str:
    """The dtype a loaded task's model states it computes in (`dew.nn.protocols.ComputesIn`), by name, or
    where it states none its parameters' (`dew.interop.release.computed_in`)."""
    return computed_in(None, task.model, task.variables)


def evidence(recorded: str, kind: str) -> dict | None:
    """What `kind`'s run the main `recorded` saved serialized (`saved_runs_reference.evidence`), None
    where that main's generation records none: one recorded before it stated an evidence version
    (`source.json`'s `evidence`). A generation that states one holds every kind's."""
    source = json.loads((CORPUS / recorded / "source.json").read_text())
    if "evidence" not in source:
        return None
    return json.loads((CORPUS / recorded / f"{kind}.evidence.json").read_text())


def resumed_by_its_format(run: Path, resume: Callable[[], object]) -> bool:
    """Whether `run`'s last checkpoint is of a format this Dew resumes, the current one or the one before it,
    `resume` run on it then; else, a step that holds its whole state as one item (`dew.checkpoints.ONE_ITEM`)
    as an older Dew wrote it, or a data position of a plan version this Dew reads none of
    (`dew.data.plan.PLAN_VERSIONS`), `resume` refused, naming it."""
    with Checkpoints(str(run)) as checkpoints:
        whole = (Path(checkpoints.path(checkpoints.latest)) / ONE_ITEM).is_dir()
    if whole or RunConfig.load(str(run)).data_plan_version not in PLAN_VERSIONS:
        with pytest.raises(ValueError):
            resume()
        return False
    resume()
    return True


def archived(recorded: str, kind: str) -> tarfile.TarFile:
    """The archive of `kind`'s run the main `recorded` saved (`tools/saved_runs_reference.py`)."""
    stored = recording(f"tests/fixtures/saved_runs/{recorded}/{kind}.npz")["run"]
    return tarfile.open(fileobj=io.BytesIO(stored.tobytes()), mode="r:xz")


def test_every_task_a_registered_objective_saves_its_runs_as_is_a_recorded_kinds(tmp_path):
    """Every kind the corpus's writer saves (`SAVED_KINDS`) has a run a main recorded (`SAVED`), and every
    task a registered objective saves its runs as (`dew.registry.objectives`' `saved_task`) is the task of
    one of them, so a run of it a main saved loads back and answers
    (`test_a_run_an_earlier_main_saved_loads_and_answers`); a masked-diffusion run's task had no kind, and
    a kind the writer saves with no recording would pass a gate that read the writer's list (train 669's
    review, row 4674)."""
    def built(kind: str):
        # Each kind's run built in a directory of its own, as the corpus's writer builds it.
        (tmp_path / kind).mkdir()
        return SAVED_KINDS[kind](tmp_path / kind, drawn=False)

    recorded = {kind for _, kind in SAVED}
    assert set(SAVED_KINDS) <= recorded, sorted(set(SAVED_KINDS) - recorded)
    declared = {objectives[name].saved_task for name in objectives} - {None}
    saved = {objectives[built(kind).objective.name].saved_task for kind in SAVED_KINDS}
    assert declared <= saved, sorted(task.__name__ for task in declared - saved)


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


INPUTS = "inputs"
"""A generation's directory of the inputs its runs read that the repo has since changed, each at its path
in the checkout (the Gemma 3 fixture's tokenizer, which the repair of its mask token's row changed after
8a7909b0f068 trained over it): what was saved with a run, as it read them, is part of its record."""


def checkout_of(tmp_path: Path, recorded: str) -> Path:
    """The checkout as the main `recorded` read its runs' sources: this one, where its generation keeps no
    inputs of its own (`INPUTS`); else a mirror of it under `tmp_path`, every entry the checkout's own but
    those inputs, each its generation's copy. A record names a source by its path in the repo, which a
    run reads from the checkout's root."""
    root = Path(__file__).resolve().parents[1]
    kept = CORPUS / recorded / INPUTS
    if not kept.is_dir():
        return root
    mirror = tmp_path / "checkout"
    if mirror.is_dir():
        return mirror

    def opened(directory: Path, original: Path) -> None:
        # A directory of the mirror whose every entry is the checkout's, to replace one at a time.
        directory.mkdir()
        for entry in original.iterdir():
            (directory / entry.name).symlink_to(entry)

    opened(mirror, root)
    for given in sorted(path for path in kept.rglob("*") if path.is_file()):
        place = given.relative_to(kept)
        here, there = mirror, root
        for part in place.parts[:-1]:
            here, there = here / part, there / part
            if here.is_symlink():
                here.unlink()
                opened(here, there)
        (here / place.name).unlink(missing_ok=True)
        (here / place.name).write_bytes(given.read_bytes())
    return mirror


def extracted(tmp_path: Path, recorded: str, kind: str) -> Path:
    """The run directory of `kind`'s saved run, unpacked under `tmp_path`."""
    with archived(recorded, kind) as packed:
        packed.extractall(tmp_path / "runs", filter="data")
    return tmp_path / "runs" / kind


@pytest.mark.parametrize(("recorded", "kind"), SAVED, ids=IDS)
def test_a_run_an_earlier_main_saved_trained_to_finite_losses(tmp_path, recorded, kind):
    """Each run of the corpus logged finite losses and gradient norms at every step it logged them
    (`saved_run_kinds.trained_finite`): a run trained to NaN is no reference, and every later main's reads
    of it would compare NaN to NaN."""
    trained_finite(extracted(tmp_path, recorded, kind))


def test_a_run_that_logged_a_loss_not_finite_is_refused_as_a_reference(tmp_path):
    """`trained_finite` refuses a run whose logged loss or gradient norm is NaN or infinite, at any step,
    and one that logged none, which no reference can be; a run of finite ones reads back its steps'."""
    for logged in ({1: {"train/loss": 1.4}, 2: {"train/loss": float("nan")}},
                   {1: {"train/loss": 1.4, "train/grad_norm": float("inf")}}, {1: {"train/accuracy": 0.5}}):
        run = tmp_path / f"run-{len(list(tmp_path.iterdir()))}"
        with LocalTracker(run / "tracking") as tracker:
            for step, scalars in logged.items():
                tracker.log(scalars, step)
        with pytest.raises(ValueError):
            trained_finite(run)
    with LocalTracker(tmp_path / "finite" / "tracking") as tracker:
        tracker.log({"train/loss": 1.4, "train/grad_norm": 0.45, "train/accuracy": 0.3}, 1)
    assert trained_finite(tmp_path / "finite") == {1: {"train/loss": 1.4, "train/grad_norm": 0.45}}


@pytest.mark.parametrize(("recorded", "kind"), SAVED, ids=IDS)
def test_a_run_an_earlier_main_saved_loads_and_answers(tmp_path, monkeypatch, recorded, kind):
    """A kind's run saved at an earlier main rebuilds from its record, in the compute it recorded, and,
    where its objective declared a saved task then and the run recorded that declaration, pins its last
    checkpoint and loads as that task, which answers a request. A run whose objective declared none at its
    main recorded none (the corpus's JEPA runs, saved before a JEPA encoder had a release: row 4313,
    `dew.inference.models.SavedModel`), so pinning it is refused; resuming it saves a step that declares
    one (`test_a_run_an_earlier_main_saved_resumes_from_its_checkpoint`)."""
    run = extracted(tmp_path, recorded, kind)
    monkeypatch.chdir(checkout_of(tmp_path, recorded))
    recorded_run = RunConfig.load(str(run))
    seen = evidence(recorded, kind)
    task = objectives[recorded_run.objective.name].saved_task
    with Checkpoints(str(run)) as checkpoints:
        last = checkpoints.latest
        declared = checkpoints.artifact(last) is not None
    if seen is not None:
        assert recorded_run.compute_dtype() == seen["compute"]
    if declared:
        saved = SavedRun.pinned(str(run))
        assert saved.step == last
        loaded = task.from_run(str(run))
        # The task loaded computes in the dtype its run did, as its model and numerics state it.
        assert computes_in(loaded) == recorded_run.compute_dtype()
        answered(loaded)
    else:
        with pytest.raises(ValueError):
            SavedRun.pinned(str(run))


@pytest.mark.parametrize(("recorded", "kind"), SAVED, ids=IDS)
def test_a_run_an_earlier_main_saved_exports_and_loads_back_answering_as_its_run(tmp_path, recorded, kind):
    """A kind's run saved at an earlier main, exported by `dew export` where its checkpoint declared its task,
    loads back through `dew.pipeline` as the task its run loads (`from_run`) and answers one request as that
    task does, leaf for leaf (MF01). A run that declared none (the corpus's JEPA runs, saved before a JEPA
    encoder had a release) is refused by the export, as `SavedRun.pinned` refuses it."""
    from option_witness import answered_alike, answers

    from dew._cli.main import main
    from dew.inference.loading import pipeline

    run = extracted(tmp_path, recorded, kind)
    task = objectives[RunConfig.load(str(run)).objective.name].saved_task
    with Checkpoints(str(run)) as checkpoints:
        declared = checkpoints.artifact(checkpoints.latest) is not None
    release = tmp_path / "release"
    if not declared:
        with pytest.raises(ValueError):
            main(["export", str(run), str(release)])
        return
    assert main(["export", str(run), str(release)]) == 0
    assert answered_alike(answers(pipeline(str(release))), answers(task.from_run(str(run))))

@pytest.mark.parametrize(("recorded", "kind"), SAVED, ids=IDS)
def test_a_run_an_earlier_main_saved_resumes_from_its_checkpoint(tmp_path, monkeypatch, recorded, kind):
    """A kind's run saved at an earlier main resumes from its last checkpoint where this Dew reads its formats
    (its checkpoint's and its data plan's, `resumed_by_its_format`), fitting the one step past it, and is
    refused before it trains a step where it reads none of one of them."""
    run = extracted(tmp_path, recorded, kind)
    recorded_run = RunConfig.load(str(run))
    with Checkpoints(str(run)) as checkpoints:
        last = checkpoints.latest
    here = SAVED_KINDS[kind](tmp_path, drawn=False)
    # A source the repo holds is recorded by its path in the repo (a decision run's backbone), read from
    # the checkout's root as the run that wrote it read it.
    monkeypatch.chdir(checkout_of(tmp_path, recorded))
    # A run counted in examples reads on to its bound; one counted in steps, one step more.
    steps = {} if recorded_run.trainer.examples is not None else {"steps": last + 1}
    trainer = dataclasses.replace(recorded_run.trainer, checkpoint_dir=str(tmp_path / "runs"), stop=None,
                                  resume="require", compilation_cache_dir=None, multi_host=False, **steps)
    sources = {"data": here.data, "validation": here.validation}
    if not resumed_by_its_format(run, lambda: dataclasses.replace(
            recorded_run, trainer=trainer, **sources).prepare().train(kind)):
        stepped = [step for step, _ in LocalTracker(run / "tracking").scalars()]
        assert max(stepped) <= last
        return
    records = [json.loads(line) for line in (run / "tracking" / "records.jsonl").read_text().splitlines()]
    started = [row["value"] for row in records if row["type"] == "FitStarted"][-1]
    assert (started["start_step"], started["resumed_from"] is not None) == (last, True)


def rebuilt(tmp_path: Path, monkeypatch, recorded: str, kind: str, **stated) -> tuple[Path, int, object]:
    """`kind`'s saved run of `recorded` unpacked, its last step, and the run that made it built again, as
    `dew train` builds it from its Python or its flags (`SAVED_KINDS`), not read back from its record,
    `stated` set on it, to resume that step and train one more: a source the repo holds named by its
    path in the repo, as the corpus records it, read from the checkout's root."""
    run = extracted(tmp_path, recorded, kind)
    recorded_run = RunConfig.load(str(run))
    with Checkpoints(str(run)) as checkpoints:
        last = checkpoints.latest
    monkeypatch.chdir(checkout_of(tmp_path, recorded))
    clocked = recorded_run.trainer.examples is not None
    trainer = dataclasses.replace(recorded_run.trainer, checkpoint_dir=str(tmp_path / "runs"), stop=None,
                                  **({} if clocked else {"steps": last + 1}), resume="require",
                                  compilation_cache_dir=None, multi_host=False)
    here = SAVED_KINDS[kind](tmp_path, drawn=False)
    sourced = {"pretrained": recorded_run.pretrained} if hasattr(recorded_run, "pretrained") else {}
    return run, last, dataclasses.replace(here, trainer=trainer, **sourced, **stated)


@pytest.mark.parametrize(("recorded", "kind"), SAVED, ids=IDS)
def test_a_run_an_earlier_main_saved_resumes_when_its_command_runs_again(tmp_path, monkeypatch, recorded,
                                                                         kind):
    """A kind's run saved at an earlier main resumes from its last checkpoint when the run that made it is
    built again from its command (`rebuilt`): every field it leaves for its build to resolve takes the
    saved run's value (`RunConfig.BUILT`: a decision run's option text, its rows' writing, row 3848, and
    its source's storage), so the resume compares alike and fits the one step past its checkpoint."""
    run, last, built = rebuilt(tmp_path, monkeypatch, recorded, kind)
    if not resumed_by_its_format(run, lambda: built.prepare().train(kind)):
        return
    records = [json.loads(line) for line in (run / "tracking" / "records.jsonl").read_text().splitlines()]
    started = [row["value"] for row in records if row["type"] == "FitStarted"][-1]
    assert (started["start_step"], started["resumed_from"] is not None) == (last, True)


@pytest.mark.parametrize("given", ["chain", "loader"])
@pytest.mark.parametrize("named", [None, "another"])
@pytest.mark.parametrize(("recorded", "kind"), SAVED, ids=IDS)
def test_a_run_an_earlier_main_saved_resumes_through_its_python_train(tmp_path, monkeypatch, recorded, kind,
                                                                      named, given):
    """A kind's run saved at an earlier main resumes from its last checkpoint as a Python experiment trains
    it: its objective built, then handed to `RunConfig.train` with its data, the chain it states uncompiled
    as the examples hand it (`run.data`), or the loader its build compiled (`given`), the run naming no data
    plan version, as one written in Python does. The train resolves it itself, the saved run's (`resumed`,
    which `readied` asks of a chain; a loader reads by the version it was compiled at), so the resume
    compares alike and fits the one step past its checkpoint, and the record it writes names the version
    the saved run states (train 589's review: a run trained so read by this Dew's version and recorded
    none, so a saved run's resume was refused; train 627's: the chain's path was not the one proven). The
    run it resumes is the one `train` is given the name of, whatever its trainer names, none or another
    (`named`)."""
    run, last, built = rebuilt(tmp_path, monkeypatch, recorded, kind)
    stated = RunConfig.load(str(run)).data_plan_version
    if stated not in PLAN_VERSIONS:
        # Its data, compiled at the version it states, is refused where its run is prepared.
        with pytest.raises(ValueError):
            built.prepare()
        return
    prepared = built.prepare()
    python = dataclasses.replace(prepared.run, data_plan_version=None,
                                 trainer=dataclasses.replace(prepared.run.trainer, name=named))
    data, validation = ((built.chain(), built.validation) if given == "chain"
                        else (prepared.dataset, prepared.validation))
    if not resumed_by_its_format(run, lambda: python.train(
            prepared.objective, data, name=kind, metrics=prepared.metrics, rollout=prepared.rollout,
            summary=prepared.summary, validation=validation, after=prepared.after)):
        return
    records = [json.loads(line) for line in (run / "tracking" / "records.jsonl").read_text().splitlines()]
    started = [row["value"] for row in records if row["type"] == "FitStarted"][-1]
    assert (started["start_step"], started["resumed_from"] is not None) == (last, True)
    assert RunConfig.load(str(run)).data_plan_version == stated


@pytest.mark.parametrize(("landed", "version"), [(9, 9), (None, 9)])
def test_a_run_saved_while_its_version_was_data_plan_loads_and_resumes_by_it(tmp_path, monkeypatch, landed,
                                                                             version):
    """A run a Dew saved while a run's data plan version was its field `data_plan` (main 1af304d0, before
    the field was `data_plan_version`, train 589's review) records the version it compiled at, or None
    where it was trained in Python, which that Dew compiled at this Dew's, 9: it loads by that version and
    resumes when its command runs again (row 4107)."""
    recorded, kind = next((recorded, kind) for recorded, kind in SAVED if kind == "lm")
    run, last, built = rebuilt(tmp_path, monkeypatch, recorded, kind)
    record = json.loads((run / "run.json").read_text())
    fields = {name: value for name, value in record["fields"].items() if name != "data_plan_version"}
    (run / "run.json").write_text(json.dumps({**record, "fields": {**fields, "data_plan": landed}}))
    assert RunConfig.load(str(run)).data_plan_version == version
    if not resumed_by_its_format(run, lambda: built.prepare().train(kind)):
        return
    records = [json.loads(line) for line in (run / "tracking" / "records.jsonl").read_text().splitlines()]
    started = [row["value"] for row in records if row["type"] == "FitStarted"][-1]
    assert (started["start_step"], started["resumed_from"] is not None) == (last, True)


def test_reading_a_saved_run_opens_no_device_before_its_process_is_prepared(tmp_path):
    """`dew train run.json` reads the run before it prepares its process (`RunConfig.run`'s
    `prepare_process`, which joins a pool of processes before any device is opened), so reading a saved
    decision run's record opens none, whatever its record leaves to its checkpoints: the writing a run saved
    before row 3867 left to its layout is read where the run resumes (`resumed`, `check_resumed`), after
    the process is prepared, in a fresh interpreter (train 589's review)."""
    # A saved decision run's record as main before row 3867 wrote it, its writing left to its layout.
    run = extracted(tmp_path, *next((recorded, kind) for recorded, kind in SAVED if kind == "decision"))
    record = json.loads((run / "run.json").read_text())
    (run / "run.json").write_text(json.dumps({**record, "fields": {**record["fields"], "option_text": None}}))
    assert_exits_cleanly(f"""
import socket

import jax
from dew.config import RunConfig

RunConfig.load({str(run)!r})
# A pool of one process joined after the read, as `prepare_process` joins one: refused once a device is open.
with socket.socket() as free:
    free.bind(("localhost", 0))
    port = free.getsockname()[1]
jax.distributed.initialize(f"localhost:{{port}}", num_processes=1, process_id=0)
jax.distributed.shutdown()
""")


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("kind", ["decision", "decision-lettered"])
def test_a_decision_run_this_dew_saves_resumes_when_its_command_runs_again_or_from_its_record(tmp_path,
                                                                                              kind):
    """A decision run saved by this Dew, its option text left to its layout (a release's named, a fresh
    lettered head's described), records the writing its build resolved, as a pinned source's step is, so
    a later Dew's default writes none of its rows otherwise (row 3867); it resumes when its command runs
    again and when read back from its record. A record a Dew before that wrote holds None, which reading
    it leaves (no checkpoint is read before its process is prepared, train 589's review) and its resume
    reads off its checkpoint's layout (`resumed`, `dew.config.earlier_option_writing`), and it resumes
    either way."""
    from run_kinds import trained

    from dew.decision.config import DecisionRunConfig

    first = SAVED_KINDS[kind](tmp_path, drawn=False)
    assert isinstance(first, DecisionRunConfig) and first.option_text is None
    trained(first, tmp_path, kind, trainer={"name": kind, "checkpoint_final": True})
    directory = tmp_path / "runs" / kind
    written = {"decision": "named", "decision-lettered": "described"}[kind]
    saved = json.loads((directory / "run.json").read_text())
    assert saved["fields"]["option_text"] == written
    unresolved = json.dumps({**saved, "fields": {**saved["fields"], "option_text": None}})
    for steps in (2, 3):
        (directory / "run.json").write_text(unresolved)
        loaded = DecisionRunConfig.load(str(directory))
        assert loaded.option_text is None and loaded.resumed().option_text == written
        run = first if steps == 2 else loaded
        trained(run, tmp_path, kind, trainer={"name": kind, "steps": steps, "checkpoint_final": True})
        assert str(steps) in {path.name for path in directory.iterdir()}
        assert json.loads((directory / "run.json").read_text())["fields"]["option_text"] == written


@pytest.mark.parametrize(("recorded", "kind"),
                         [(recorded, kind) for recorded, kind in SAVED if kind == "diffusion"],
                         ids=lambda part: str(part))
def test_a_diffusion_runs_release_stores_its_encoders_parameters_in_the_dtype_a_load_asks(tmp_path, recorded,
                                                                                            kind):
    """A diffusion run's native release loaded with a storage dtype stores every parameter its run's load does
    in it, its condition encoders' among them, and the rest as written: leaf for leaf the dtypes of the run
    loaded so (review of the run release)."""
    from dew._cli.main import main
    from dew.inference.loading import pipeline

    run = extracted(tmp_path, recorded, kind)
    assert main(["export", str(run), str(tmp_path / "release")]) == 0
    loaded, released = (pipeline(str(path), param_dtype="bfloat16") for path in (run, tmp_path / "release"))
    assert (jax.tree.map(lambda leaf: leaf.dtype, released.variables)
            == jax.tree.map(lambda leaf: leaf.dtype, loaded.variables))
