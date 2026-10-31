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
