"""A released version's notes are the ones it shipped: once `## X.Y.Z` is cut, its section's text is frozen,
its sha256 recorded by the release in tools/changelog_releases.json. The changelog's merge joins both sides'
new lines (.gitattributes, tools/changelog_merge.py), and the land queue merges each head onto main as the
first parent, so a head queued before a release that adds its entry at the head of `## Unreleased` lands
below the new version's header instead (row 3050: two entries filed under 0.2.0 at train 397); it fails
here, and belongs under `## Unreleased`."""

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def sections(text: str) -> dict[str, str]:
    """Each `## <name>` section of a changelog's `text`, by name: its lines between its header and the
    next."""
    found: dict[str, list[str]] = {}
    name = None
    for line in text.splitlines():
        if line.startswith("## "):
            name = line[3:].strip()
            found[name] = []
        elif name is not None:
            found[name].append(line)
    return {name: "\n".join(lines).strip("\n") for name, lines in found.items()}


def test_each_released_version_keeps_the_notes_it_shipped_and_unreleased_comes_first():
    written = sections((REPO / "CHANGELOG.md").read_text())
    released = json.loads((REPO / "tools/changelog_releases.json").read_text())
    assert next(iter(written)) == "Unreleased"
    assert {version for version in written if version != "Unreleased"} == set(released)
    changed = [version for version, digest in released.items()
               if hashlib.sha256(written[version].encode()).hexdigest() != digest]
    assert not changed, (f"released notes changed under {', '.join(changed)}: "
                         "move a new entry under ## Unreleased")
