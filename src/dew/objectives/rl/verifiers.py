"""Verifiable rewards: score a completion by running it or by checking its answer.

Both are `Reward` callables, `(data_source, completion, ground_truth,
extra_info) -> float`, so they plug into `SampledRollout` and `AsyncRollout`
unchanged. `CodeReward` extracts the program from the completion and runs it
against test cases in a `SandboxFleet`. `MathReward` reads the final answer
and compares it with the reference as an exact rational number.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from fractions import Fraction

from .fleet import Program, SandboxFleet, outputs_match

_FENCE = re.compile(r"```([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.DOTALL)


def code_block(completion: str, language: str = "python") -> str | None:
    """The last fenced block tagged `language`, else the last untagged one, else None."""
    blocks = [(tag.lower(), body) for tag, body in _FENCE.findall(completion)]
    for wanted in (language, ""):
        tagged = [body for tag, body in blocks if tag == wanted]
        if tagged:
            return tagged[-1]
    return None


def _cases(ground_truth: str) -> list[tuple[str, str]]:
    """Read `[{"stdin": ..., "stdout": ...}, ...]` test cases from the reward column."""
    cases = json.loads(ground_truth)
    if not isinstance(cases, list) or not cases:
        raise ValueError("a code reward's ground truth is a nonempty JSON list of test cases")
    read: list[tuple[str, str]] = []
    for case in cases:
        if (not isinstance(case, dict) or not isinstance(case.get("stdout"), str)
                or not isinstance(case.get("stdin", ""), str)):
            raise ValueError("each test case holds string stdout and optional string stdin")
        read.append((case.get("stdin", ""), case["stdout"]))
    return read


@dataclass(frozen=True)
class CodeReward:
    """Run the completion's program on every test case; reward the fraction that pass.

    The ground truth is a JSON list of `{"stdin", "stdout"}` cases. A case
    passes when the program completes and its stdout matches, line by line
    up to trailing whitespace. A completion without a code block scores
    zero, as do timeouts, crashes and oversized output. `all_or_nothing`
    scores one only when every case passes. `interpreter` is the argv the
    program file is appended to; the default is this Python, isolated from
    the environment, site-packages and user site.
    """

    fleet: SandboxFleet
    interpreter: tuple[str, ...] = (sys.executable, "-I", "-S")
    language: str = "python"
    filename: str = "main.py"
    all_or_nothing: bool = False

    def __call__(self, data_source: str, completion: str, ground_truth: str, extra_info: str) -> float:
        cases = _cases(ground_truth)
        source = code_block(completion, self.language)
        if source is None:
            return 0.0
        outcomes = self.fleet.run(Program({self.filename: source}, (*self.interpreter, self.filename), stdin)
                                  for stdin, _ in cases)
        passed = sum(outputs_match(outcome, expected) for outcome, (_, expected) in zip(outcomes, cases, strict=True))
        if self.all_or_nothing:
            return float(passed == len(cases))
        return passed / len(cases)


_BOXED = re.compile(r"\\boxed\{")
_GROUPED = r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?"
"""A number written with thousands-group commas, the only commas read as part of one."""
_NUMBER = re.compile(rf"{_GROUPED}|-?\d+(?:\.\d+)?(?:/\d+)?|-?\.\d+")


def _boxed(text: str) -> str | None:
    """The contents of the last `\\boxed{...}`, braces balanced."""
    starts = [match.end() for match in _BOXED.finditer(text)]
    if not starts:
        return None
    depth, start = 1, starts[-1]
    for index in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[index], 0)
        if depth == 0:
            return text[start:index]
    return None


def _rational(text: str) -> Fraction | None:
    """Read a decimal, integer or `a/b` answer, allowing thousands separators and `\\frac{a}{b}`.

    Any other comma or inner space leaves the text unreadable as one number,
    so a list such as `1,2` or `2 3` never collapses into `12` or `23`.
    """
    cleaned = text.replace("$", "").strip()
    if re.fullmatch(_GROUPED, cleaned):
        cleaned = cleaned.replace(",", "")
    fraction = re.fullmatch(r"-?\\[dt]?frac\{(-?\d+)\}\{(-?\d+)\}", cleaned)
    if fraction is not None:
        numerator, denominator = int(fraction[1]), int(fraction[2])
        sign = -1 if cleaned.startswith("-") else 1
        return None if denominator == 0 else sign * Fraction(numerator, denominator)
    try:
        return Fraction(cleaned)
    except (ValueError, ZeroDivisionError):
        return None


@dataclass(frozen=True)
class MathReward:
    """Score one when the final answer equals the reference as a rational number.

    The answer is the last `\\boxed{}` in the completion; with
    `require_boxed=False` a completion without one falls back to its last
    number. Integers, decimals, `a/b` and `\\frac{a}{b}` compare exactly, so
    `0.5`, `1/2` and `\\frac{1}{2}` agree. A reference that is not a number
    compares as trimmed text.
    """

    require_boxed: bool = True

    def __call__(self, data_source: str, completion: str, ground_truth: str, extra_info: str) -> float:
        answer = _boxed(completion)
        if answer is None and not self.require_boxed:
            numbers = _NUMBER.findall(completion)
            answer = numbers[-1] if numbers else None
        if answer is None:
            return 0.0
        expected = _rational(ground_truth)
        if expected is None:
            return float(answer.strip() == ground_truth.strip())
        return float(_rational(answer) == expected)
