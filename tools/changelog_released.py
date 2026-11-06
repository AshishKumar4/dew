"""A released version's notes are the ones it shipped: once `## X.Y.Z` is cut, its section's text is frozen,
its sha256 recorded by the release in tools/changelog_releases.json. The changelog's merge joins both sides'
new lines (.gitattributes, tools/changelog_merge.py), and the land queue merges each head onto main as the
first parent, so a head queued before a release that adds its entry at the head of `## Unreleased` lands
below the new version's header instead (row 3050: two entries filed under 0.2.0 at train 397).

    python tools/changelog_released.py check            # tools/check_gates' `changelog`
    python tools/changelog_released.py record VERSION   # a release records its notes when it is cut
"""

import enum
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

CHANGELOG = Path("CHANGELOG.md")
RELEASES = Path("tools/changelog_releases.json")


class Fault(enum.Enum):
    UNRELEASED_NOT_FIRST = "the first section is not ## Unreleased"
    UNRECORDED = "a released section the releases do not record"
    MISSING = "a recorded release the changelog has no section for"
    CHANGED = "a released section's notes changed since it shipped"


class Finding(NamedTuple):
    fault: Fault
    section: str


def sections(text: str) -> dict[str, str]:
    """Each `## <name>` section of a changelog's `text`, by name, in order: its lines between its header
    and the next."""
    found: dict[str, list[str]] = {}
    name = None
    for line in text.splitlines():
        if line.startswith("## "):
            name = line[3:].strip()
            found[name] = []
        elif name is not None:
            found[name].append(line)
    return {name: "\n".join(lines).strip("\n") for name, lines in found.items()}


def digest(notes: str) -> str:
    return hashlib.sha256(notes.encode()).hexdigest()


def findings(changelog: str, released: Mapping[str, str]) -> list[Finding]:
    """What `changelog` breaks of the releases `released` records (version to its notes' digest): its first
    section is `## Unreleased`, and each released version's notes are the ones it shipped."""
    written = sections(changelog)
    found = [] if next(iter(written), None) == "Unreleased" else [
        Finding(Fault.UNRELEASED_NOT_FIRST, next(iter(written), ""))]
    versions = [name for name in written if name != "Unreleased"]
    found += [Finding(Fault.UNRECORDED, name) for name in versions if name not in released]
    found += [Finding(Fault.MISSING, name) for name in released if name not in written]
    found += [Finding(Fault.CHANGED, name) for name in versions
              if name in released and digest(written[name]) != released[name]]
    return found


def recorded(changelog: str, released: Mapping[str, str], version: str) -> dict[str, str]:
    """`released` with `version`'s notes in `changelog` recorded, as the release that cuts it writes them."""
    written = sections(changelog)
    if version not in written:
        raise ValueError(f"the changelog has no ## {version} section to record")
    return {version: digest(written[version]), **released}


def main(argv: Sequence[str] | None = None, root: Path = Path(".")) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    changelog = (root / CHANGELOG).read_text()
    released = json.loads((root / RELEASES).read_text())
    if arguments[:1] == ["record"] and len(arguments) == 2:
        (root / RELEASES).write_text(json.dumps(recorded(changelog, released, arguments[1]), indent=1) + "\n")
        return 0
    if arguments != ["check"]:
        print(__doc__, file=sys.stderr)
        return 2
    found = findings(changelog, released)
    for finding in found:
        print(f"## {finding.section}: {finding.fault.value}; a new entry goes under ## Unreleased")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
