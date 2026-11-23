"""Every option of a run kind's config reaches the same run by each path that sets it (MF01): set in Python
(`dataclasses.replace`), at its dotted path (`RunConfig.override`, as a run file or a sweep sets it), read
back from the run's JSON record (`RunConfig.read`) and from the run.json it saves (`RunConfig.save`,
`RunConfig.load`), each value an alternative its field's type declares (`option_witness.alternatives`): each
path holds the value as Python's does and writes the record the others write, and a value Python refuses,
the path refuses as well. The fields every run holds are swept on a language model's run, each
kind's own on its own."""

import dataclasses
import json
import typing

import pytest
from option_witness import alternatives, flag, held_at, replaced
from run_kinds import KINDS, saved_record, scalars, trained_by_cli, trained_record

from dew import registry
from dew.config import RunConfig, TrainerConfig
from dew.training import Layout, MeshSpec

SUPPRESSED = ("data", "validation")
"""The fields a run's record writes as its chains (`dew.data.records`), which a path sets one setting of a
recorded chain's call at a time; their paths are the data layer's (tests/test_config_cli.py)."""


def options(config, prefix: str = "") -> typing.Iterator[tuple[str, object, object]]:
    """Each option `config` holds, by its dotted path: a field of one of Dew's dataclasses it holds read
    field by field, any other field with each alternative its annotation declares about its value."""
    hints = typing.get_type_hints(type(config), include_extras=True)
    for field in dataclasses.fields(config):
        if not field.init or (not prefix and field.name in SUPPRESSED):
            continue
        value, hint = getattr(config, field.name), hints[field.name]
        if dataclasses.is_dataclass(value) and type(value).__module__.startswith("dew."):
            yield from options(value, f"{prefix}{field.name}.")
            continue
        for alternative in alternatives(hint, value):
            if alternative != value:
                yield f"{prefix}{field.name}", hint, alternative


def cases() -> list[tuple[str, str, int]]:
    """Each kind's options as (kind, path, index of its alternative): a field every run holds on a language
    model's run alone, each kind's own on its own."""
    shared = {field.name for field in dataclasses.fields(RunConfig)}
    found = []
    for kind in sorted(registry.runs):
        seen: dict[str, int] = {}
        for path, _, _ in options(registry.runs[kind]()):
            if kind != "lm" and path.split(".")[0] in shared:
                continue
            found.append((kind, path, seen.get(path, 0)))
            seen[path] = seen.get(path, 0) + 1
    return found


def option(kind: str, path: str, index: int) -> tuple[RunConfig, object, object]:
    """The run of `kind` from its defaults, the annotation of its option `path` and that option's
    alternative `index`."""
    base = registry.runs[kind]()
    held = [(hint, value) for name, hint, value in options(base) if name == path]
    return base, *held[index]


@pytest.mark.parametrize(("kind", "path", "index"), cases(), ids=lambda part: str(part))
def test_each_path_that_sets_an_option_sets_the_same_run(kind, path, index, tmp_path):
    """The option set in Python, at its path, and read back from the run's record and its run.json, is the
    value set, and each writes one record; refused in Python, it is refused at its path."""
    base, hint, value = option(kind, path, index)
    # The value as a record holds it, which a path sets (`RunConfig.override`).
    given = {path: registry.to_record(value, hint)}
    try:
        python = replaced(base, path, value)
    except (TypeError, ValueError) as refused:
        with pytest.raises(type(refused)):
            base.override(given)
        return
    record = json.loads(json.dumps(python.record()))
    python.save(str(tmp_path / "saved"))
    paths = {"path": base.override(given), "record": RunConfig.read(record),
             "run.json": RunConfig.load(str(tmp_path / "saved"))}
    # The value as the run holds it: a field may read a value as its own form (a tuple of one id, a rate
    # its default resolves).
    held = held_at(python, path)
    for name, run in paths.items():
        assert type(run) is type(python), name
        assert held_at(run, path) == held, name
        assert run.record() == record, name



SET = {"optim.lr": 0.0123, "data_seed": 3, "trainer.key": 5}
"""Options a run trained both ways is set with, each beside its default."""


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("kind", list(KINDS))
def test_a_run_with_options_set_trains_alike_from_python_its_command_and_its_record_and_resumes_as_it_ran(
        kind, tmp_path):
    """A kind's run with options set (`SET`, and its trainer's beside them on one mesh and layout) trains two
    steps three ways: from Python (`prepare().train`); from its record saved and read back
    (`RunConfig.load`); and by `dew train` over its run.json with the trainer on the command line, stopped at
    its first step's checkpoint and resumed for the second by `dew train` over the run's own record. Every
    way logs the same two losses, and the run.json each writes reads back as one run, but for the name each
    trains under (row 4511: prepare, train, the command and a saved, reloaded and resumed run agree)."""
    run = KINDS[kind](tmp_path, drawn=False).override(SET)
    hints = typing.get_type_hints(TrainerConfig)
    given = {"mesh": MeshSpec(fsdp=2), "layout": Layout(min_shard=64), "log_every": 1}
    trainer = dataclasses.replace(run.trainer, checkpoint_dir=str(tmp_path / "runs"),
                                  steps=2, eval_every=None, checkpoint_every=None,
                                  multi_host=False, compilation_cache_dir=None, **given)
    python = dataclasses.replace(run, trainer=trainer)
    python.prepare().train("python")
    RunConfig.load(str(saved_record(python, tmp_path, "saved"))).prepare().train("reloaded")
    stopped = trained_by_cli(run, tmp_path, "cli", {f"trainer.{name}": flag(value, hints[name])
                                                    for name, value in given.items()})
    trained_record(stopped, {"trainer.steps": "2"})
    runs = tmp_path / "runs"
    losses = {name: [scalars(runs / name)[step]["train/loss"] for step in (1, 2)]
              for name in ("python", "reloaded", "cli")}
    assert losses["reloaded"] == losses["cli"] == losses["python"], losses
    records = {name: RunConfig.load(str(runs / name)) for name in losses}
    for name, record in records.items():
        named = dataclasses.replace(records["python"], trainer=dataclasses.replace(
            records["python"].trainer, name=record.trainer.name))
        assert named.record() == record.record(), name
