"""The prime search Dew's Engram layout reads (`dew.nn.engram`).

sympy ships no py.typed, and its unbounded search loop reads to a checker
as one that can fall through, so `nextprime` would return `int | None`.
It returns the `ith` prime above `n` (sympy 1.14.0, ntheory/generate.py).
"""

def nextprime(n: int, ith: int = 1) -> int: ...
