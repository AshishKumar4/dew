"""dew: train runs, run programs on accelerator clusters, prepare data and act on run directories.

dew tpu creates, sets up and reaches Cloud TPUs; `dew tpu --help` lists its commands.
"""

# Nothing here imports an array library at import time, so `dew --help`
# answers without loading JAX. A command that needs one imports it inside
# `run_command`, where the work is.

from __future__ import annotations

import dataclasses
import importlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal

import tyro

from dew.cli import tpu
from dew.cli.gcloud import emit
from dew.cli.launch import Launch

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
    step: int | None = None
    """Which checkpoint to read; unset takes the latest."""
    trust: tuple[str, ...] = ()
    """Packages outside Dew whose modules the run's record may import."""

    def run_command(self) -> int:
        from dew.inference.tasks import run_record
        from dew.interop import Pretrained
        from dew.objectives.base import ReleasedTask
        from dew.records import text
        from dew.registry import objectives

        # A task that publishes its own released layout (a decision run's) writes it; any other
        # run writes its model in the source layout its family reads.
        record, _ = run_record(self.run, self.step, self.trust)
        task = objectives[text(record['objective'], 'objective')].saved_task
        if task is not None and issubclass(task, ReleasedTask):
            task.from_run(self.run, ema=self.ema, step=self.step).save_pretrained(self.destination)
        else:
            loaded = Pretrained.from_run(self.run, ema=self.ema, step=self.step, trust=self.trust)
            loaded.save(self.destination)
        emit(f"exported {self.run} to {self.destination}")
        return 0


@dataclasses.dataclass(frozen=True)
class DataExport:
    """Export a recipe's deterministic records to committed storage."""

    recipe: Positional[str]
    destination: Positional[str]
    fields: str = "{}"
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
        from dew.records import record
        from dew.registry import datasets

        format = self.format or ("array_record" if self.tokenizer is not None else "parquet")
        if self.tokenizer is not None and format != "array_record":
            raise ValueError("token exports must use --format array_record")
        recipe = datasets.build(self.recipe, record(json.loads(self.fields), "recipe fields"))
        train, held, _ = recipe.pipelines()
        pipeline = train if self.branch == "train" else held
        if pipeline is None:
            raise ValueError(f"{self.recipe} has no validation source")
        metadata = None
        if self.tokenizer is not None:
            from dew.data.steps.text import Tokenize
            from dew.data.text import tokenizer_for

            vocabulary = tokenizer_for(self.tokenizer)
            pipeline = pipeline.map(Tokenize(self.tokenizer, eos=self.eos))
            metadata = {"token_store": True, "tokenizer": self.tokenizer,
                        "vocab_size": vocabulary.vocab_size, "dtype": "int32",
                        "eos_id": vocabulary.eos_id if self.eos else None}
        manifest = export(iterate(pipeline, read=self.read, provision=self.provision),
                          self.destination, format=format, rows_per_shard=self.rows_per_shard,
                          metadata=metadata, length_field="text" if self.tokenizer is not None else None)
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
    """Fill and publish a deterministic recipe prefix to the artifact cache."""

    recipe: Positional[str]
    destination: Positional[str]
    fields: str = "{}"
    branch: Literal["train", "validation"] = "train"
    read: Literal["memory", "indexed", "streamed"] = "streamed"
    provision: bool = False

    def run_command(self) -> int:
        from dew.data.cache import Cache
        from dew.data.compile import publish
        from dew.records import record
        from dew.registry import datasets

        recipe = datasets.build(self.recipe, record(json.loads(self.fields), "recipe fields"))
        train, held, _ = recipe.pipelines()
        pipeline = train if self.branch == "train" else held
        if pipeline is None:
            raise ValueError(f"{self.recipe} has no validation source")
        generation = publish(pipeline, Cache(self.destination), read=self.read, provision=self.provision)
        emit(f"published {generation.url}")
        return 0


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
        from dew.config import RunConfig

        path = Path(self.run)
        if path.suffix == ".json":
            run = RunConfig.load(str(path), trust=self.trust)
        else:
            # The file imports as the module its name names, as `python -m`
            # imports one beside it, so a record names what it defines by that
            # module, which the run trusts: `dew train run.json --trust <name>`.
            sys.path.insert(0, str(path.resolve().parent))
            built = importlib.import_module(path.stem).run
            built = built() if callable(built) else built
            if not isinstance(built, RunConfig):
                raise TypeError(f"{self.run}'s run is {built!r}, not a run config")
            if built.trainer.name is None:
                built = dataclasses.replace(built, trainer=dataclasses.replace(built.trainer, name=path.stem))
            # What trains is what the record reads back as, so its run.json trains the same run.
            run = RunConfig.read(built.record(), trust=(*self.trust, path.stem))
        run.assigned(self.set).run()
        return 0


COMMANDS = {"export": Export, "launch": Launch, "train": Train}
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
