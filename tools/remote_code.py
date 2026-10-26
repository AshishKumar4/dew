"""A transformers modeling module rewritten as a repository's own remote code: its classes under another
model's prefix and its attributes renamed, every import absolute, the code otherwise the module's.

Tier 2's tests (tests/test_tier2_parts.py) load such a repository under `trust_remote_code=True`: a
decoder whose code is a registered family's under names no family reads, which Dew assembles from that
family's parts by its code (`dew.interop.parts`).
"""
import inspect
import re
from collections.abc import Mapping
from types import ModuleType


def renamed_module(module: ModuleType, prefix: tuple[str, str], renames: Mapping[str, str],
                   configuration: tuple[str, str]) -> str:
    """`module`'s source with each class's `prefix[0]` model prefix as `prefix[1]`, each name of
    `renames` as its value, its relative imports absolute and its configuration import
    (`configuration`: the module's own configuration class's import line, and the repository's)
    replaced."""
    source = inspect.getsource(module)
    package = module.__name__.rpartition(".")[0].rpartition(".")[0].rpartition(".")[0]
    source = re.sub(r"from \.\.\.(\S*) import",
                    lambda match: f"from {package}{'.' if match[1] else ''}{match[1]} import", source)
    source = source.replace(*configuration)
    source = re.sub(rf"\b{prefix[0]}(?=[A-Z])", prefix[1], source)
    for name, renamed in renames.items():
        source = re.sub(rf"\b{name}\b", renamed, source)
    return source
