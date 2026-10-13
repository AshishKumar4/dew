"""URL resolution and binary-file signatures used by Dew's storage boundary.

The library's untyped mode branches infer text for binary reads. These
overloads describe its documented rb/wb and text modes separately.
"""

from . import core as core
from .spec import AbstractFileSystem

def filesystem(protocol: str, **storage_options: object) -> AbstractFileSystem: ...
def get_filesystem_class(protocol: str) -> type[AbstractFileSystem]: ...
def register_implementation(name: str, cls: type[AbstractFileSystem], clobber: bool = ..., errtxt: str | None = ...) -> None: ...
