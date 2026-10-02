"""The read-only builder lookup Dew uses, typed like `builder_from_directory`.

TFDS 4.9.10's logging decorator makes Pyright infer a zero-argument
`as_data_source` on the builder this returns; core/read_only_builder.py
returns the same builder `builder_from_directory` does.
"""

from os import PathLike

from tensorflow_datasets import _PreparedBuilder

def builder_from_files(
    builder_name: str, *, data_dir: str | PathLike[str] | None = None,
    config: str | None = None, version: str | None = None,
) -> _PreparedBuilder[object]: ...
