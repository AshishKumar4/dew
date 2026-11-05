"""Git's merge driver for CHANGELOG.md (.gitattributes: `CHANGELOG.md merge=changelog`).

Lanes add entries at the head of `## Unreleased` side by side, so two merged branches often insert lines at
the same place: a 3-way merge calls that a conflict though neither side removed anything. The driver joins
such a hunk, ours then theirs, as git's union merge would, an entry both sides carry once (row 3431). A hunk
where either side deleted or rewrote a line the base had stays a conflict: union kept the other side's copy
there, so a corrected entry came back beside its correction and a head's deletion vanished in its train's
merge (rows 3284, 3297). Everything else is git's own 3-way merge (`git merge-file --diff3`).

Git runs it once a clone names it (the land queue does the same on its merges):

    git config merge.changelog.driver \\
        'python3 "$(git rev-parse --show-toplevel)/tools/changelog_merge.py" %O %A %B %L'

A clone that does not merges CHANGELOG.md as any text file: the same hunks conflict, and none is joined.
"""

import subprocess
import sys
from pathlib import Path


def entries(lines: list[bytes]) -> list[list[bytes]]:
    """`lines` as the changelog's entries: each an item (`- ` at its first column) and the continuation lines
    under it, and every other line a block of its own."""
    blocks: list[list[bytes]] = []
    for line in lines:
        if blocks and blocks[-1][0].startswith(b"- ") and line[:1] == b" ":
            blocks[-1].append(line)
        else:
            blocks.append([line])
    return blocks


def unshared(ours: list[bytes], theirs: list[bytes]) -> list[bytes]:
    """`theirs`, a side's new lines, without the entries `ours` holds too (`entries`): an entry both sides
    carry, main having taken a train's and a head carrying the same as a commit of its own, is joined once,
    wherever each side holds it (row 3431)."""
    held = {tuple(block) for block in entries(ours) if block[0].startswith(b"- ")}
    return [line for block in entries(theirs) if tuple(block) not in held for line in block]


def joined(base: Path, ours: Path, theirs: Path, size: int) -> tuple[bytes, int]:
    """The merge of `ours` and `theirs` from `base`, each conflict hunk whose base side is empty joined (ours,
    then theirs' entries ours lacks, `unshared`), and the number of hunks left in conflict, marked as git
    marks them (`size` its markers' width)."""
    merged = subprocess.run(["git", "merge-file", "--stdout", "--diff3", f"--marker-size={size}",
                             "-L", "ours", "-L", "base", "-L", "theirs", str(ours), str(base), str(theirs)],
                            capture_output=True)
    # Its exit status is the number of conflicts, at most 127; a negative status (above 127 here) is an error.
    if merged.returncode > 127:
        raise SystemExit(f"git merge-file failed on {ours}: {merged.stderr.decode().strip()}")
    opened, based, split, closed = (b"<" * size, b"|" * size, b"=" * size, b">" * size)
    text: list[bytes] = []
    hunk: dict[str, list[bytes]] = {}
    side = None
    left = 0
    for line in merged.stdout.splitlines(keepends=True):
        bare = line.rstrip(b"\r\n")
        if side is None and bare.startswith(opened + b" "):
            hunk, side = {"open": [line], "ours": [], "base": [], "theirs": []}, "ours"
        elif side == "ours" and bare.startswith(based + b" "):
            hunk["based"], side = [line], "base"
        elif side == "base" and bare == split:
            hunk["split"], side = [line], "theirs"
        elif side == "theirs" and bare.startswith(closed + b" "):
            if hunk["base"]:
                text += [*hunk["open"], *hunk["ours"], *hunk["based"], *hunk["base"], *hunk["split"],
                         *hunk["theirs"], line]
                left += 1
            else:
                text += hunk["ours"] + unshared(hunk["ours"], hunk["theirs"])
            side = None
        elif side is None:
            text.append(line)
        else:
            hunk[side].append(line)
    return b"".join(text), left


def main() -> int:
    base, ours, theirs, size = sys.argv[1:5]
    text, left = joined(Path(base), Path(ours), Path(theirs), int(size))
    Path(ours).write_bytes(text)
    return 1 if left else 0


if __name__ == "__main__":
    sys.exit(main())
