"""Dew's public surface as committed (docs/api/report.txtar, tools/api_report.py; ledger 3037)."""
from pathlib import Path

from tool_support import load

ROOT = Path(__file__).resolve().parents[1]


def test_the_committed_api_report_is_the_public_surface():
    """The committed report is what tools/api_report.py reads from Dew now: an API change regenerates it in
    the same commit (`python tools/api_report.py --write`), so the change shows as a moved line of its
    section, which thresh's published-language seams tap."""
    report = load("api_report")
    committed = (ROOT / "docs" / "api" / "report.txtar").read_text()
    assert report.report() == committed, (
        "the public surface moved: run `python tools/api_report.py --write` and commit docs/api/report.txtar")


def test_the_api_report_reads_every_section_and_names_no_address():
    """Each section holds lines, and no value is shown by its memory address, which would differ by run."""
    report = load("api_report").report()
    sections = [line[3:-3] for line in report.splitlines() if line.startswith("-- ") and line.endswith(" --")]
    assert sections == ["exports", "config", "cli", "records"]
    assert " at 0x" not in report


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
