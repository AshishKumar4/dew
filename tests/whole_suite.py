"""The cases a group's 120 s target sends to the whole suite's run alone (`whole_suite`), by a test
module's table of them: {test: {case: seconds}}, each case keyed as its arguments (a tuple of several,
or its one value) or, for a `pytest.param`, its id too, and timed on armada, the run that timed it named
where the table is (bible §7, ledgers 2279 and 2327). Every other case of the test runs each train."""

import pytest

PARAMETER_SET = type(pytest.param())
"""What `pytest.param` returns, by pytest's public constructor."""


def timed(table: dict, test: str, cases: list) -> list:
    """`cases` (a test's arguments, a tuple of several or one value each, or a `pytest.param`) with those
    `table[test]` names marked to the whole suite's run, a `pytest.param` keeping its id and marks."""
    slow = table.get(test, {})

    def named(values: tuple, ident: str | None = None) -> bool:
        return values in slow or (len(values) == 1 and values[0] in slow) or ident in slow

    def one(case):
        if isinstance(case, PARAMETER_SET):
            if named(tuple(case.values), case.id):
                return pytest.param(*case.values, id=case.id, marks=(*case.marks, pytest.mark.whole_suite))
            return case
        values = case if isinstance(case, tuple) else (case,)
        return pytest.param(*values, marks=pytest.mark.whole_suite) if named(values) else case

    return [one(case) for case in cases]
