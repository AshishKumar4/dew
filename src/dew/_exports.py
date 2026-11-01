"""The lookup of a lazily exporting package: each name its `_EXPORTS` declares (a name to the module that
defines it, or to the package's own module of that name, which it exports as the module), imported from
that module when first used, so importing the package imports none of them. tools/gen_exports.py writes
the package's TYPE_CHECKING imports and `__all__` from the same mapping."""

from collections.abc import Callable
from importlib import import_module
from types import ModuleType


def lookup(package: str, exports: dict[str, str]
           ) -> tuple[Callable[[str], type | Callable | ModuleType], Callable[[], list[str]]]:
    """The module `__getattr__` and `__dir__` (PEP 562) of `package`: the first reads each of `exports` from
    its module, and each of the package's modules that defines one, imported when it is first read as an
    attribute, as an eager package's imports of them bound it (`dew.sampling.text` after
    `import dew.sampling`); the second lists every name it exports, the `__all__` tools/gen_exports.py
    writes from them, though it has imported none, which an editor and an interpreter complete from."""
    modules = {module.rpartition(".")[2]: module for module in exports.values()
               if module.rpartition(".")[0] == package}

    def __getattr__(name: str) -> type | Callable | ModuleType:
        module = exports.get(name)
        if module is not None:
            imported = import_module(module)
            return imported if module == f"{package}.{name}" else getattr(imported, name)
        if name in modules:
            return import_module(modules[name])
        raise AttributeError(f"module {package!r} has no attribute {name!r}")

    def __dir__() -> list[str]:
        return list(exports)

    return __getattr__, __dir__
