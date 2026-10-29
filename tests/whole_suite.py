"""The cases a group's 120 s target sends to the whole suite's run alone (`whole_suite`), by a test
module's table of them: {test: {case: seconds}}, each case keyed as its arguments (a tuple of several,
or its one value) and timed on armada, the run that timed it named where the table is (bible §7, ledger
2279). Every other case of the test runs each train."""

import pytest

PARAMETER_SET = type(pytest.param())
"""What `pytest.param` returns, by pytest's public constructor."""


def timed(table: dict, test: str, cases: list) -> list:
    """`cases` (a test's arguments, a tuple of several or one value each) with those `table[test]` names
    marked to the whole suite's run; a case already a `pytest.param` as it is."""
    slow = table.get(test, {})

    def one(case):
        if isinstance(case, PARAMETER_SET):
            return case
        values = case if isinstance(case, tuple) else (case,)
        if values in slow or (len(values) == 1 and values[0] in slow):
            return pytest.param(*values, marks=pytest.mark.whole_suite)
        return case

    return [one(case) for case in cases]
