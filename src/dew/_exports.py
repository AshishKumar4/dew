"""The lookup of a lazily exporting package: each name its `_EXPORTS` declares (a name to the module that
defines it), imported from that module when first used, so importing the package imports none of them.
tools/gen_exports.py writes the package's TYPE_CHECKING imports and `__all__` from the same mapping."""

from collections.abc import Callable
from importlib import import_module
from types import ModuleType


def lookup(package: str, exports: dict[str, str]) -> Callable[[str], type | Callable | ModuleType]:
    """The module `__getattr__` (PEP 562) of `package`, which reads each of `exports` from its module, and
    each of the package's modules that defines one, imported when it is first read as an attribute, as an
    eager package's imports of them bound it (`dew.sampling.text` after `import dew.sampling`)."""
    modules = {module.rpartition(".")[2]: module for module in exports.values()
               if module.rpartition(".")[0] == package}

    def __getattr__(name: str) -> type | Callable | ModuleType:
        module = exports.get(name)
        if module is not None:
            return getattr(import_module(module), name)
        if name in modules:
            return import_module(modules[name])
        raise AttributeError(f"module {package!r} has no attribute {name!r}")

    return __getattr__
