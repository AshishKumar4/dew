"""Row datasets read from a parquet file or from JSON records.

`dew.data.prompts` and `dew.data.preferences` are the same shape: one file of
rows, or the same rows written out as JSON for a test, and a training stream
over whichever of the two a spec names. What they have in common is here, so
each spec is its own row validation and nothing else.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence

from dew.records import JSON

from .dataset import Dataset, DatasetSpec, Records, train_stream, validation_pass
from .tokens import bounded


def parquet_names(path: str, what: str) -> list[str]:
    """The column names of the parquet file at `path`.

    `what` names the reader in the refusal a host without pyarrow earns,
    since the library is an extra rather than a dependency.
    """
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise ImportError(
            f"reading {what} parquet needs pyarrow: pip install pyarrow") from exc
    return [field.name for field in parquet.read_schema(path)]


def parquet_rows(path: str, fields: Sequence[str],
                 names: Sequence[str]) -> list[Mapping[str, object]]:
    """`path`'s rows over the `fields` it has, one dict per row.

    `names` is the file's own columns, as `parquet_names` read them, so a
    field the file lacks is dropped rather than refused here.
    """
    import pyarrow.parquet as parquet

    table = parquet.read_table(path, columns=[name for name in fields if name in names])
    return table.to_pylist()


def json_records(records: Sequence[str]) -> list[JSON]:
    """Each string of `records` parsed as one JSON value, refused by index.

    A row's shape is the reader's to check; each source names the fields it
    wants and refuses a row that is not an object with them.
    """
    rows: list[JSON] = []
    for index, record in enumerate(records):
        try:
            rows.append(json.loads(record))
        except json.JSONDecodeError as exc:
            raise ValueError(f"record {index} is not JSON: {exc}") from exc
    return rows


def row_dataset(spec: DatasetSpec, *, batch: int, path: str | None,
                records: Sequence[str], val_path: str | None, val_batches: int | None,
                from_parquet: Callable[[str], Records],
                from_records: Callable[[Sequence[str]], Records]) -> Dataset:
    """The training stream and validation pass of a spec that reads one file.

    `path` names a parquet file and `records` holds JSON rows; exactly one of
    the two is set, and `from_parquet` or `from_records` turns it into the
    source. `val_path` is a second parquet file read as one pass, bounded to
    `val_batches` batches.
    """
    if (path is None) == (not records):
        raise ValueError(
            f"{type(spec).__name__} reads one source: --data.path names a parquet file, "
            "or records holds JSON rows")
    source = from_parquet(path) if path is not None else from_records(records)
    validation = None
    if val_path is not None:
        validation = bounded(validation_pass(
            from_parquet(val_path), [], batch=batch, seed=spec.seed,
            loading=spec.loading), val_batches)
    return Dataset(
        train=train_stream(source, [], batch=batch, seed=spec.seed, loading=spec.loading),
        val=validation,
        records=len(source),
        batch=batch,
    )
