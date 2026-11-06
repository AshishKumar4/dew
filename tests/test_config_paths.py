"""Every option of a run kind's config reaches the same run by each path that sets it (MF01): set in Python
(`dataclasses.replace`), on the command line (`dew train --set path=value`, `RunConfig.assigned`), read back
from the run's JSON record (`RunConfig.read`) and from the run.json it saves (`RunConfig.save`,
`RunConfig.load`), each value an alternative its field's type declares (`option_witness.alternatives`): each
path holds the value as Python's does and writes the record the others write, and a value Python refuses,
the command line refuses as well. The fields every run holds are swept on a language model's run, each
kind's own on its own."""

import dataclasses
import json
import typing

import pytest
from option_witness import alternatives, flag, held_at, replaced
from run_kinds import KINDS, scalars, trained_by_cli

from dew import registry
from dew.config import RunConfig, TrainerConfig
from dew.training import Layout, MeshSpec

SUPPRESSED = ("data", "validation")
"""The fields a run's record writes as its chains (`dew.data.records`), which `--set` sets one setting of a
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
    """The option set in Python, on the command line, and read back from the run's record and its
    run.json, is the value set, and each writes one record; refused in Python, it is refused on the
    command line."""
    base, hint, value = option(kind, path, index)
    given = f"{path}={flag(value, hint)}"
    try:
        python = replaced(base, path, value)
    except (TypeError, ValueError) as refused:
        with pytest.raises(type(refused)):
            base.assigned([given])
        return
    record = json.loads(json.dumps(python.record()))
    python.save(str(tmp_path / "saved"))
    paths = {"command line": base.assigned([given]), "record": RunConfig.read(record),
             "run.json": RunConfig.load(str(tmp_path / "saved"))}
    # The value as the run holds it: a field may read a value as its own form (a tuple of one id, a rate
    # its default resolves).
    held = held_at(python, path)
    for name, run in paths.items():
        assert type(run) is type(python), name
        assert held_at(run, path) == held, name
        assert run.record() == record, name



SET = ("optim.learning_rate=0.0123", "data_seed=3", "trainer.key=5")
"""Options a run trained both ways is set with, each beside its default."""


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("kind", list(KINDS))
def test_a_run_trained_from_python_trains_as_dew_train_trains_it(kind, tmp_path):
    """A kind's run with options set, trained one step from Python (`prepare().train`) and by `dew train`
    over its run.json with the same trainer on the command line, on one mesh and layout: both steps log
    one loss, and the run.json each writes reads back as one run, but for the name each trains under."""
    run = KINDS[kind](tmp_path, drawn=False).assigned(list(SET))
    hints = typing.get_type_hints(TrainerConfig)
    given = {"mesh": MeshSpec(fsdp=2), "layout": Layout(min_shard=64), "log_every": 1,
             "checkpoint_final": False}
    trainer = dataclasses.replace(run.trainer, checkpoint_dir=str(tmp_path / "runs"), steps=1,
                                  eval_every=None, checkpoint_every=None, multi_host=False,
                                  compilation_cache_dir=None, **given)
    dataclasses.replace(run, trainer=trainer).prepare().train("python")
    trained_by_cli(run, tmp_path, "cli", {f"trainer.{name}": flag(value, hints[name])
                                          for name, value in given.items()})
    runs = tmp_path / "runs"
    assert scalars(runs / "python")[1]["train/loss"] == scalars(runs / "cli")[1]["train/loss"]
    python, cli = (RunConfig.load(str(runs / name)) for name in ("python", "cli"))
    renamed = dataclasses.replace(python, trainer=dataclasses.replace(python.trainer, name=cli.trainer.name))
    assert renamed.record() == cli.record()
