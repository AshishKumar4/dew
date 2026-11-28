"""Dew's own XLA flag lexer (src/dew/_xla_flags.py) for a tool, loaded by its path from this checkout: a tool
runs in another tree's environment too (a perf gate's base), whose Dew may state no lexer of its own, and this
checkout's is the one its tool reads flags by."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "_dew_xla_flags", Path(__file__).resolve().parents[1] / "src" / "dew" / "_xla_flags.py")
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("src/dew/_xla_flags.py of this checkout cannot be loaded")
_LEXER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_LEXER)

spans, value, flag = _LEXER.spans, _LEXER.value, _LEXER.flag
