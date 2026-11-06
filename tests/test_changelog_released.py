"""tools/changelog_released.py, the release tooling's check that a released version keeps the notes it
shipped and `## Unreleased` comes first, on fixture changelogs."""

import json

from tool_support import load

released = load("changelog_released")

CHANGELOG = """# Changelog

## Unreleased

- a new entry

## 0.2.0

- shipped in 0.2.0

## 0.1.0

- shipped in 0.1.0
"""


def releases(text: str) -> dict[str, str]:
    """Every version of `text` recorded as `text` holds its notes."""
    recorded: dict[str, str] = {}
    for version in released.sections(text):
        if version != "Unreleased":
            recorded = released.recorded(text, recorded, version)
    return recorded


def test_a_changelog_whose_released_notes_are_as_shipped_holds():
    assert released.findings(CHANGELOG, releases(CHANGELOG)) == []


def test_an_entry_merged_under_a_released_header_changes_its_notes():
    """Row 3050: a head queued before a release lands its entry under the new version's header."""
    merged = CHANGELOG.replace("- shipped in 0.2.0", "- a late entry\n- shipped in 0.2.0")
    assert released.findings(merged, releases(CHANGELOG)) == [
        released.Finding(released.Fault.CHANGED, "0.2.0")]


def test_unreleased_comes_first_and_every_section_is_recorded_both_ways():
    reordered = CHANGELOG.replace("## Unreleased\n\n- a new entry\n\n", "") + "\n## Unreleased\n\n- late\n"
    assert released.Finding(released.Fault.UNRELEASED_NOT_FIRST, "0.2.0") in released.findings(
        reordered, releases(CHANGELOG))
    unrecorded = {version: notes for version, notes in releases(CHANGELOG).items() if version != "0.1.0"}
    assert released.findings(CHANGELOG, unrecorded) == [released.Finding(released.Fault.UNRECORDED, "0.1.0")]
    ghost = {**releases(CHANGELOG), "0.0.9": "0" * 64}
    assert released.findings(CHANGELOG, ghost) == [released.Finding(released.Fault.MISSING, "0.0.9")]


def test_a_release_records_its_notes_and_the_gate_reads_the_checkout(tmp_path):
    (tmp_path / "tools").mkdir()
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG)
    (tmp_path / "tools" / "changelog_releases.json").write_text(json.dumps(
        {"0.1.0": releases(CHANGELOG)["0.1.0"]}))
    assert released.main(["check"], tmp_path) == 1
    assert released.main(["record", "0.2.0"], tmp_path) == 0
    assert released.main(["check"], tmp_path) == 0
