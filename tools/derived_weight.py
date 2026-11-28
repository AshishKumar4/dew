"""What a package carries of its template derivations (tools/derived_catalogue.py's entries): the one rule of
their weight, and the one staging of a release's, by Python alone, as the package's build environment holds
no Dew.

    python tools/derived_weight.py stage INTO DERIVED GENERATION...

Each generation's entries (DERIVED/<generation>, tools/ci/derived.sh's) staged into INTO, their union, one
name one content whichever generation derived it, weighed once: its {"generations", "entries", "mib",
"limit_mib"} on stdout; exit 1 where a generation holds none, two hold one name differently, or the union is
past the limit.
"""

import filecmp
import json
import shutil
import sys
from pathlib import Path

LIMIT_MIB = 16.0
"""The most the entries may weigh, in MiB: a package carries them to every install."""


def weight(entries: list[Path]) -> int:
    """What `entries` weigh, in bytes: each read once."""
    return sum(entry.stat().st_size for entry in entries)


def within(bytes_: int, limit_mib: float = LIMIT_MIB) -> int:
    """`bytes_` (a set of entries' `weight`), refused past `limit_mib`."""
    if bytes_ > limit_mib * 2**20:
        raise SystemExit(f"the entries weigh {bytes_ / 2**20:.2f} MiB, past {limit_mib} MiB")
    return bytes_


def staged(into: Path, derived: Path, generations: list[str], limit_mib: float = LIMIT_MIB) -> dict:
    """Each of `generations`' entries in `derived` (one directory each) staged into `into`, their union: an
    entry named alike in two generations is one entry, refused where their contents differ (an entry is named
    by its key, so one name is one content); a generation holding none is refused; the union is then weighed
    once against `limit_mib`, so generations each within it are not past it together. What it staged, by
    generation and whole."""
    into.mkdir(parents=True, exist_ok=True)
    counts = {}
    for generation in generations:
        entries = sorted((derived / generation).glob("*.json"))
        if not entries:
            raise SystemExit(f"{derived} holds no derivation of {generation}")
        for entry in entries:
            target = into / entry.name
            if not target.exists():
                shutil.copyfile(entry, target)
            elif not filecmp.cmp(entry, target, shallow=False):
                raise SystemExit(f"derivation {entry.name} differs between generations")
        counts[generation] = len(entries)
    union = sorted(into.glob("*.json"))
    weighs = within(weight(union), limit_mib)
    return {"generations": counts, "entries": len(union), "mib": round(weighs / 2**20, 3),
            "limit_mib": limit_mib}


if __name__ == "__main__":
    if sys.argv[1:2] != ["stage"] or len(sys.argv) < 5:
        raise SystemExit(__doc__)
    print(json.dumps(staged(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:])), flush=True)
