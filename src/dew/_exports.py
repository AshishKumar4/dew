"""The lookup of a lazily exporting package: each name its `_EXPORTS` declares (a name to the module that
defines it), imported from that module when first used, so importing the package imports none of them.
tools/gen_exports.py writes the package's TYPE_CHECKING imports and `__all__` from the same mapping."""

from collections.abc import Callable
from importlib import import_module


def lookup(package: str, exports: dict[str, str]) -> Callable[[str], type | Callable]:
    """The module `__getattr__` (PEP 562) of `package`, which reads each of `exports` from its module."""

    def __getattr__(name: str) -> type | Callable:
        module = exports.get(name)
        if module is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        return getattr(import_module(module), name)

    return __getattr__
