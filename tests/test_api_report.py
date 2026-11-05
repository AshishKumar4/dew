"""Dew's public surface as committed (docs/api/report.txtar, tools/api_report.py; ledger 3037)."""
import importlib

import pytest
from tool_support import load


def test_the_api_report_reads_every_section_and_names_no_address():
    """Each section holds lines, and no value is shown by its memory address, which would differ by run."""
    report = load("api_report").report()
    sections = [line[3:-3] for line in report.splitlines() if line.startswith("-- ") and line.endswith(" --")]
    assert sections == ["exports", "config", "cli", "records"]
    assert " at 0x" not in report


@pytest.mark.parametrize("module, name", [("dew.optim", "OptimConfig"), ("dew.lora", "LoRA"),
                                          ("dew.optim", "ParamGroup")])
def test_a_dataclass_a_config_field_names_is_listed_with_its_fields_wherever_it_is_defined(module, name):
    """A run config's field typed by another module's dataclass lists that dataclass's fields and defaults,
    so changing one moves a line: `optim` names OptimConfig outright, `lora` names LoRA in an `Annotated`
    optional, and OptimConfig's `param_groups` names ParamGroup in a tuple."""
    report = load("api_report")
    expected = [f"  {line}" for line in report.fields(getattr(importlib.import_module(module), name))]
    lines = report.config()
    start = lines.index(f"{module}.{name}")
    assert lines[start + 1:start + 1 + len(expected)] == expected


def test_a_field_the_constructor_takes_by_keyword_or_not_at_all_is_marked():
    """A config field moved to keyword-only, or out of the constructor, breaks a caller's positional or
    keyword argument, so its line says so."""
    import dataclasses

    @dataclasses.dataclass
    class Fields:
        plain: int
        hidden: int = dataclasses.field(default=0, init=False)
        named: str = dataclasses.field(default="a", kw_only=True)

    assert load("api_report").fields(Fields) == [
        "plain: int", "hidden: int = 0  [no constructor argument]", "named: str = 'a'  [keyword-only]"]


def test_a_property_shows_its_getters_type_and_whether_it_is_set():
    """A property's type and a setter it gains or loses move its line."""
    class Held:
        @property
        def read(self) -> int:
            return 0

        @property
        def written(self) -> str:
            return ""

        @written.setter
        def written(self, value: str) -> None:
            pass

    assert load("api_report").members(Held) == [
        "  .read: property -> int", "  .written: property -> str [set]"]


def test_a_commands_arguments_are_as_it_declares_them():
    """A positional argument shows as positional, not as a flag, and a repeated one as repeated, each with
    its field's type, aliases resolved, its default and the spelling tyro is told of."""
    import dataclasses
    from typing import Annotated

    import tyro

    Count = Annotated[int | None, tyro.conf.arg(metavar="N", aliases=("-n",))]

    @dataclasses.dataclass
    class Command:
        run: tyro.conf.Positional[str]
        count: Count = None
        set: Annotated[tuple[str, ...], tyro.conf.UseAppendAction] = ()

    assert load("api_report").arguments(Command) == [
        "  run: str  [Positional]", "  count: int | None = None  [aliases=('-n',)]  [metavar='N']",
        "  set: tuple[str, ...] = ()  [UseAppendAction]"]


def test_a_signature_is_written_alike_on_every_python():
    """A union, `Optional` or `|`, and a generic over one print alike on 3.12 and 3.14 (`annotation`), in a
    signature as in a field: the report is one file every declared Python reads the same."""
    import typing

    optional = typing.Optional  # the spelling 3.12 prints a union with None in

    def old(first, second=None):
        return None

    old.__annotations__ = {"first": optional[int], "second": optional[dict[str, optional[float]]],
                           "return": optional[list[int]]}

    def new(first: int | None, second: dict[str, float | None] | None = None) -> list[int] | None:
        return None

    report = load("api_report")
    assert report.signature(old) == report.signature(new) == (
        "(first: int | None, second: dict[str, float | None] | None = None) -> list[int] | None")


def test_a_lambda_default_factory_is_shown_by_its_source_and_never_run():
    """A field whose default a lambda builds shows that lambda's source, which says what it builds, where a
    name would hide it; the lambda is not run (one may read the checkout)."""
    import dataclasses

    ran = []

    @dataclasses.dataclass
    class Built:
        sizes: list = dataclasses.field(default_factory=lambda: ran.append(1) or [2, 3])

    assert load("api_report").fields(Built) == ["sizes: list = (lambda: ran.append(1) or [2, 3])()"]
    assert not ran
