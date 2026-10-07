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
from typing import Annotated

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
        from dew.interop import Pretrained

        Pretrained.from_run(self.run, ema=self.ema, step=self.step, trust=self.trust).save(self.destination)
        emit(f"exported {self.run} to {self.destination}")
        return 0


@dataclasses.dataclass(frozen=True)
class Tokenize:
    """Tokenize a text corpus into the train.bin, val.bin and meta.json a token dataset reads."""

    input: str
    """A text file, or a directory read as every *.txt inside it (recursive)."""
    out: str
    """The directory the token files are written into; created if it is not there."""
    tokenizer: str = "byte"
    """'byte' for utf-8 bytes, else a Hugging Face tokenizer name."""
    val_fraction: float = 0.01
    """The fraction of the token stream held out, from its head, as validation."""
    pack: bool = False
    """End every document (input file) with the tokenizer's eos id, so
    TokenWindows(pack=True) can cut the stream back into documents."""

    def run_command(self) -> int:
        from dew.data import TokenCorpus

        if self.tokenizer != "byte":
            # Every chunk is longer than the model's context, which is the
            # point of a corpus; without this the tokenizer warns about it.
            from transformers.utils import logging as hf_logging

            hf_logging.set_verbosity_error()
        corpus = TokenCorpus.write(self.input, self.out, tokenizer=self.tokenizer,
                                   val_fraction=self.val_fraction, pack=self.pack)
        out = Path(self.out)
        emit(f"wrote {corpus.train_tokens} tokens to {out / 'train.bin'} and "
             f"{corpus.val_tokens} to {out / 'val.bin'}")
        emit(f"{out / 'meta.json'}: {json.dumps(dataclasses.asdict(corpus))}")
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


COMMANDS = {"export": Export, "launch": Launch, "tokenize": Tokenize, "train": Train}


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["tpu"]:
        return tpu.main(args[1:])
    command = tyro.extras.subcommand_cli_from_dict(
        COMMANDS, args=args, prog="dew", description=__doc__, config=CONFIG)
    return command.run_command()


if __name__ == "__main__":
    raise SystemExit(main())
