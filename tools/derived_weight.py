"""What a package may carry of its template derivations (tools/derived_catalogue.py's entries): the one rule,
read of one generation's derivations as the catalogue writes them and of the union a release stages
(tools/ci/package.sh), by Python alone, as the package's build environment holds no Dew.

    python tools/derived_weight.py DIRECTORY   the directory's entries weighed, refused past the limit
"""

import json
import sys
from pathlib import Path

LIMIT_MIB = 16.0
"""The most the entries may weigh, in MiB: a package carries them to every install."""


def weighed(entries: list[Path], limit_mib: float = LIMIT_MIB) -> int:
    """What `entries` weigh, in bytes, refused past `limit_mib`."""
    weight = sum(entry.stat().st_size for entry in entries)
    if weight > limit_mib * 2**20:
        raise SystemExit(f"the entries weigh {weight / 2**20:.2f} MiB, past {limit_mib} MiB")
    return weight


if __name__ == "__main__":
    found = sorted(Path(sys.argv[1]).glob("*.json"))
    print(json.dumps({"entries": len(found), "mib": round(weighed(found) / 2**20, 3)}))
