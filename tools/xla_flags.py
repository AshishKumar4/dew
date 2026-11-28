"""Dew's own XLA flag lexer (src/dew/_xla_flags.py) for a tool, loaded by its path from this checkout
(`reference_source.loaded`, the tools' one file loader): a tool runs in another tree's environment too (a
perf gate's base), whose Dew may state no lexer of its own, and this checkout's is the one its tool reads
flags by."""

from pathlib import Path

from reference_source import loaded

_LEXER = loaded("_dew_xla_flags", Path(__file__).resolve().parents[1] / "src" / "dew" / "_xla_flags.py")

spans, value, flag = _LEXER.spans, _LEXER.value, _LEXER.flag
