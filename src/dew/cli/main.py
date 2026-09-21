"""dew: the commands that act on a run directory.

The second console script beside `dew-tpu`, and it keeps the same rule this
package states: nothing here imports an array library at import time, so
`dew --help` answers without loading JAX. A command that needs one imports
it inside `run`, where the work is.
"""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Sequence

import tyro

from dew.cli.gcloud import emit

CONFIG = (
    tyro.conf.FlagCreatePairsOff,
    tyro.conf.PositionalMetavarFromFieldName,
)
Positional = tyro.conf.Positional


@dataclasses.dataclass(frozen=True)
class Export:
    """Write a trained run to the published layout its family reads back."""

    run: Positional[str]
    """The run directory: `run.json` beside its checkpoints."""
    destination: Positional[str]
    """Where the export lands; created if it is not there."""
    ema: bool = True
    """Read the run's averaged weights, where it kept them."""
    step: int | None = None
    """Which checkpoint to read; unset takes the latest."""

    def run_command(self) -> int:
        from dew.interop.export import export_run

        export_run(self.run, self.destination, ema=self.ema, step=self.step)
        emit(f"exported {self.run} to {self.destination}")
        return 0


COMMANDS = {"export": Export}


def main(argv: Sequence[str] | None = None) -> int:
    command = tyro.extras.subcommand_cli_from_dict(
        COMMANDS, args=list(sys.argv[1:] if argv is None else argv), prog="dew",
        description=__doc__, config=CONFIG)
    return command.run_command()


if __name__ == "__main__":
    raise SystemExit(main())
