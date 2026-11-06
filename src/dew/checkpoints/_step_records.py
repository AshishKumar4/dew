"""What a checkpoint's step records beside its state: the table of every process's data position
(`gather_positions`, `read_position`) and the step's record of its EMA deltas, control state, rung,
inference artifact and boxed axes (`step_record`), which a persistent and a local save share."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import TYPE_CHECKING

import jax
import numpy as np

from dew import _position
from dew._coordination import from_every_process
from dew._records import JSON
from dew.mesh import LogicalAxes

if TYPE_CHECKING:
    from dew.data.dataset import DataPartition


def step_record(deltas: Mapping[str, list[str]], control: dict | None, rung: JSON, artifact: JSON,
                boxed: Mapping[tuple[str, ...], LogicalAxes] | None) -> dict:
    """What a persistent and a local save both record of a step: the leaves of each average it stores as
    their differences from their weights, by the average's name, its control state, its rung, its inference
    artifact and its boxed axes."""
    return {"deltas": dict(deltas), "control": copy.deepcopy(control or {}), "rung": rung,
            "artifact": artifact, "boxed": boxed_record(boxed)}


def boxed_record(boxed: Mapping[tuple[str, ...], LogicalAxes] | None) -> list[dict[str, list[str | None]]]:
    """The boxed axes as a step's metadata holds them, one path and its names each."""
    return [{"path": list(path), "names": list(names)} for path, names in (boxed or {}).items()]


def counted_processes(count: int) -> str:
    return f"{count} process" + ("es" if count != 1 else "")


def gather_positions(saved: bytes, share: DataPartition) -> dict:
    """Gather every process's iterator position into the checkpoint's `position` table.

    'rows' is a uint8 [process_count, longest] array with one row per
    process, 'lengths' the unpadded length of each, and 'shares' the
    `[index, count]` of the data share each process read. A process reports
    the position it holds, and Orbax writes a host array from process 0
    alone, so the rows are gathered onto every process before a save. The
    rows differ in length, so their lengths are stored with them.

    There is one row per process whatever kind the position is. A global
    position is the same bytes on every process, and gathering it lets
    `read_position` check that the rows agree before another partition reads it.
    """
    told = from_every_process([saved.hex(), share.index, share.count])
    every = [bytes.fromhex(hexed) for hexed, _, _ in told]
    # The integer width a jax array of int64 takes, as the table's always had.
    integer = jax.dtypes.canonicalize_dtype(np.int64)
    lengths = np.asarray([len(entry) for entry in every], integer)
    rows = np.zeros((len(every), int(lengths.max())), np.uint8)
    for row, entry in zip(rows, every, strict=True):
        row[:len(entry)] = np.frombuffer(entry, np.uint8)
    return {'rows': rows, 'lengths': lengths,
            'shares': np.asarray([[index, count] for _, index, count in told], integer)}


def _row(table: dict, index: int) -> bytes:
    """Return one row of a `gather_positions` table, without its padding."""
    row = np.asarray(table['rows'][index], np.uint8)
    return row[:int(table['lengths'][index])].tobytes()


def read_position(table: dict, where: str, share: DataPartition) -> bytes:
    """Read the position the reader of `share` resumes from, out of a saved table.

    A global position is a record count over an order that is the same at
    any partition, so every row is a valid position for every reader;
    `dew._position` declares that property and `read_position` relies on it.
    A share's own offset resumes only readers of that same share, whichever
    processes they are, and any other reader is refused with the shares named.
    """
    written = len(table['lengths'])
    # Row order, not set order: bytes hash differently per interpreter, and
    # which of the refusals below a broken checkpoint gets is a diagnostic.
    rows = [_row(table, index) for index in range(written)]
    if _position.translates(rows[0]):
        if any(row != rows[0] for row in rows[1:]):
            raise ValueError(
                f"The checkpoint at {where} holds a global data position that "
                f"differs between the {counted_processes(written)} that wrote it. A global "
                f"position is one place in one order, so those processes read "
                f"different orders and no single one of their positions is this "
                f"run's.")
        return rows[0]
    held = [(int(index), int(count)) for index, count in np.asarray(table['shares'])]
    mine = [row for row, written_share in zip(rows, held, strict=True)
            if written_share == (share.index, share.count)]
    if not mine:
        raise ValueError(
            f"The checkpoint at {where} holds data positions for the shares "
            f"{sorted(set(held))} (index, count), and this reader reads share "
            f"{share.index} of {share.count}. A position is where one share of "
            f"the data stopped and cannot be translated to another share, so "
            f"resume it on a mesh whose processes read those shares: {_position.PER_SHARE}.")
    if any(row != mine[0] for row in mine[1:]):
        raise ValueError(
            f"The checkpoint at {where} holds positions for share {share.index} "
            f"of {share.count} that differ between the processes that read it; "
            f"a share's readers read the same records, so no one of them is "
            f"where the share stopped.")
    return mine[0]
