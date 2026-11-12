from collections.abc import Mapping

from .spec import AbstractFileSystem

def url_to_fs(url: str, **kwargs: object) -> tuple[AbstractFileSystem, str]: ...

def _un_chain(path: str, kwargs: Mapping[str, object]) -> list[tuple[str, str, dict[str, object]]]:
    """Each link of a chained URL as fsspec deals it: its path, its protocol and the options it takes (a
    protocol's own, the unscoped ones its first link's), the innermost first."""
