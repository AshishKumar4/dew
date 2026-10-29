"""dew: train runs, run programs on accelerator clusters, prepare data and act on run directories.

dew tpu creates, sets up and reaches Cloud TPUs; `dew tpu --help` lists its commands.
"""

# Nothing here imports an array library at import time, so `dew --help`
# answers without loading JAX. A command that needs one imports it inside
# `run_command`, where the work is.

from __future__ import annotations

import dataclasses
import json
import sys
import types
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import tyro

from dew.cli import tpu
from dew.cli.gcloud import emit
from dew.cli.launch import Launch

if TYPE_CHECKING:
    from dew.config import RunConfig

CONFIG = tpu.CONFIG
Positional = tyro.conf.Positional


@dataclasses.dataclass(frozen=True)
class Export:
    """Write a trained run to the published layout its family reads back."""

    run: Positional[str]
    """The run directory: `run.json` beside its checkpoints."""
    destination: Positional[str]
    """Where the export lands; created if it is not there."""
    ema: bool | None = None
    """Read the run's averaged weights: unset where it kept them, True
    always, False never."""
    step: int | str | None = None
    """Which checkpoint to read: a step, `best` for the one the run ranked best, or `best:<metric>` for
    the best by one of its rankings (`Checkpoints.resolve`); unset takes the latest."""
    trust: tuple[str, ...] = ()
    """Packages outside Dew whose modules the run's record may import."""

    def run_command(self) -> int:
        from dew.inference.tasks import SavedRun
        from dew.interop import Pretrained
        from dew.objectives.base import ReleasedTask

        # A task that publishes its own released layout (a decision run's) writes it; any other
        # run writes its model in the source layout its family reads, both at the step pinned here.
        run = SavedRun.pinned(self.run, self.step, self.trust)
        task = run.objective.saved_task
        if task is not None and issubclass(task, ReleasedTask):
            task.from_saved(run, ema=self.ema).save_pretrained(self.destination)
        else:
            Pretrained.from_saved(run, ema=self.ema).save(self.destination)
        emit(f"exported {self.run} to {self.destination}")
        return 0


def user_module(path: Path) -> types.ModuleType:
    """The user's Python file at `path`, that file and no other, imported as the module its name names
    (`dew.registry.module_from_file`), with its directory first on the path, as `python -m` runs one
    beside its siblings, so it imports them."""
    from dew.registry import module_from_file

    directory = str(path.resolve().parent)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    return module_from_file(path, path.stem)


def chain_of(named: str, branch: Literal["train", "validation"], trust: tuple[str, ...] = ()):
    """The chain `named` names as a `dew.data.compile` pipeline: a Python file's `data`, or with
    `branch="validation"` its `validation` (each a chain, or a function returning one); a chain's
    record (a `.json` file `dew.data.records` reads); or a store's URL (`dew.data.sources.source`).
    Raises ValueError for a file that holds no such chain, or a store asked for its validation."""
    from dew.data import records
    from dew.data.pipeline import Dataset
    from dew.data.sources import source

    path = Path(named)
    if path.suffix == ".py":
        module = user_module(path)
        held = getattr(module, "data" if branch == "train" else "validation", None)
        chain = held() if callable(held) and not isinstance(held, Dataset) else held
    elif path.suffix == ".json":
        chain = records.decode(json.loads(path.read_text()), trust=trust)
    elif branch == "validation":
        raise ValueError(f"{named} is a store, which holds no validation chain; name a Python file's")
    else:
        chain = source(named)
    if not isinstance(chain, Dataset):
        name = "data" if branch == "train" else "validation"
        raise ValueError(f"{named} holds no {name} chain, but {chain!r}")
    return chain._chain()


@dataclasses.dataclass(frozen=True)
class DataExport:
    """Export a chain's deterministic records to committed storage."""

    data: Positional[str]
    """A Python file whose `data` (or `validation`, `--branch`) is a chain, a chain's `.json` record, or a
    store's URL (`chain_of`)."""
    destination: Positional[str]
    format: Literal["parquet", "array_record"] | None = None
    """Parquet for records, ArrayRecord for token stores unless explicitly named."""
    rows_per_shard: int = 100_000
    branch: Literal["train", "validation"] = "train"
    read: Literal["memory", "indexed", "streamed"] = "streamed"
    provision: bool = False
    tokenizer: str | None = None
    eos: bool = False

    def run_command(self) -> int:
        from dew.data.compile import iterate
        from dew.data.sinks import export

        format = self.format or ("array_record" if self.tokenizer is not None else "parquet")
        if self.tokenizer is not None and format != "array_record":
            raise ValueError("token exports must use --format array_record")
        pipeline = chain_of(self.data, self.branch)
        if self.tokenizer is not None:
            from dew.data.text import export_tokens

            manifest = export_tokens(pipeline, self.destination, self.tokenizer, eos=self.eos,
                                     rows_per_shard=self.rows_per_shard, read=self.read,
                                     provision=self.provision)
        else:
            manifest = export(iterate(pipeline, read=self.read, provision=self.provision), self.destination,
                              format=format, rows_per_shard=self.rows_per_shard)
        emit(json.dumps(manifest, sort_keys=True))
        return 0


@dataclasses.dataclass(frozen=True)
class DataProvision:
    """Prepare the named source for indexed reads."""

    source: Positional[str]

    def run_command(self) -> int:
        from dew.data.sources import source

        prepared = source(self.source, provision=True)
        emit(f"prepared {prepared.describe()}")
        return 0


@dataclasses.dataclass(frozen=True)
class DataPublish:
    """Fill and publish a chain's deterministic prefix to the artifact cache."""

    data: Positional[str]
    """A Python file's chain, a chain's `.json` record, or a store's URL, as `DataExport` reads it."""
    destination: Positional[str]
    branch: Literal["train", "validation"] = "train"
    read: Literal["memory", "indexed", "streamed"] = "streamed"
    provision: bool = False

    def run_command(self) -> int:
        from dew.data.cache import Cache
        from dew.data.compile import publish

        pipeline = chain_of(self.data, self.branch)
        generation = publish(pipeline, Cache(self.destination), read=self.read, provision=self.provision)
        emit(f"published {generation.url}")
        return 0


def run_named(named: str, trust: tuple[str, ...]) -> RunConfig:
    """The run `named` names: a Python file's `run` (a run config or a function returning one), under
    the file's own name where it names none, or a `run.json`. What trains is what the record reads
    back as, so its run.json trains the same run; a chain the record cannot describe trains as Python
    built it, and its run.json says why it can't be rebuilt (`RunConfig.unrecorded`)."""
    from dew.config import RunConfig

    path = Path(named)
    if path.suffix == ".json":
        return RunConfig.load(str(path), trust=trust)
    # The file imports as the module its name names, so a record names what it defines by that
    # module, which the run trusts: `dew train run.json --trust <name>`.
    built = user_module(path).run
    built = built() if callable(built) else built
    if not isinstance(built, RunConfig):
        raise TypeError(f"{named}'s run is {built!r}, not a run config")
    if built.trainer.name is None:
        built = dataclasses.replace(built, trainer=dataclasses.replace(built.trainer, name=path.stem))
    return built if built.unrecorded() else RunConfig.read(built.record(), trust=(*trust, path.stem))


@dataclasses.dataclass(frozen=True)
class Train:
    """Train a run: the one a Python file builds, or the one a run's `run.json` records."""

    run: Positional[str]
    """A Python file whose `run` is a run config or a function returning one, or a `run.json`."""
    set: Annotated[tuple[str, ...], tyro.conf.UseAppendAction] = ()
    """`path=value`, once a field: the run's field at the dotted path, read as its
    flag reads it (`--set trainer.steps=2000 --set model.num_layers=12`)."""
    trust: tuple[str, ...] = ()
    """Packages outside Dew whose modules a `run.json` may import."""

    def run_command(self) -> int:
        run_named(self.run, self.trust).assigned(self.set).run()
        return 0


@dataclasses.dataclass(frozen=True)
class Profile:
    """Profile a run: train it afresh for `warmup` steps and a window of `steps` more, traced, with no
    checkpoint or validation, and print where the window's steps spent their time (`RunConfig.profile`),
    its report also in `report.json` beside XProf's own reports."""

    run: Positional[str]
    """A Python file whose `run` is a run config or a function returning one, or a `run.json`."""
    warmup: int = 20
    """The steps before the window, the step's compile among them, rounded up to whole windows."""
    steps: int = 10
    """The steps the window traces."""
    out: str | None = None
    """Where the profile goes; unset writes profiles/<the run's name>/<the time it starts>."""
    set: Annotated[tuple[str, ...], tyro.conf.UseAppendAction] = ()
    """`path=value`, once a field, as `dew train` reads it."""
    trust: tuple[str, ...] = ()
    """Packages outside Dew whose modules a `run.json` may import."""
    host_tracer_level: int | None = None
    """The capture's host events: 1 the user's annotations, 2 those and Dew's and JAX's, 3 every host event;
    unset, `ProfileWindow`'s."""
    python_tracer_level: int | None = None
    """1 traces every Python call, which slows Python-heavy host work several times over; unset,
    `ProfileWindow`'s (none)."""
    activity_cap: int | None = None
    """The CUDA activity events a GPU capture keeps, CUPTI's cap, past which the capture is cut short; -1
    keeps every one (XLA's no limit); unset, `ProfileWindow`'s."""

    def run_command(self) -> int:
        from dew.training.profiling import ProfileWindow

        given = {"host_tracer_level": self.host_tracer_level, "python_tracer_level": self.python_tracer_level,
                 "activity_cap": self.activity_cap}
        capture = ProfileWindow("", self.steps,
                                **{name: value for name, value in given.items() if value is not None})
        report = run_named(self.run, self.trust).assigned(self.set).profile(self.out, warmup=self.warmup,
                                                                            steps=self.steps, capture=capture)
        if report is not None:
            emit(report.text())
        return 0


COMMANDS = {"export": Export, "launch": Launch, "profile": Profile, "train": Train}
DATA_COMMANDS = {"export": DataExport, "provision": DataProvision, "publish": DataPublish}


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["tpu"]:
        return tpu.main(args[1:])
    if args[:1] == ["data"]:
        command = tyro.extras.subcommand_cli_from_dict(
            DATA_COMMANDS, args=args[1:], prog="dew data", config=CONFIG)
        return command.run_command()
    command = tyro.extras.subcommand_cli_from_dict(
        COMMANDS, args=args, prog="dew", description=__doc__, config=CONFIG)
    return command.run_command()


if __name__ == "__main__":
    raise SystemExit(main())
