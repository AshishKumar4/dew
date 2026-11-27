"""Writes src/dew/telemetry/xla_protos.binpb: XLA's xla/xla_data.proto, xla/service/metrics.proto and
xla/service/hlo.proto and their imports at an openxla/xla commit, one serialized FileDescriptorSet, the schema
dew.telemetry.hlo reads compiled modules and buffer assignments through; and that commit, as
`dew.telemetry.hlo.XLA_COMMIT`.

    python -m tools.xla_protos <xla commit>

The commit is the one the pinned jaxlib was built from. Needs grpcio-tools (its protoc) and the network (the
protos are fetched from openxla/xla at that commit).
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
import urllib.request
from pathlib import Path

PROTOS = ("xla/xla_data.proto", "xla/service/metrics.proto", "xla/service/hlo.proto")
TELEMETRY = Path(__file__).resolve().parents[1] / "src" / "dew" / "telemetry"


def described(root: Path) -> bytes:
    """`PROTOS` under `root` and their imports, one serialized FileDescriptorSet (protoc's)."""
    import grpc_tools
    from grpc_tools import protoc

    out = root / "set.pb"
    include = Path(grpc_tools.__file__).parent / "_proto"
    if protoc.main(["protoc", f"-I{root}", f"-I{include}", "--include_imports", f"--descriptor_set_out={out}",
                    *PROTOS]):
        sys.exit("protoc failed")
    return out.read_bytes()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("commit", help="the openxla/xla commit the pinned jaxlib was built from")
    commit = parser.parse_args().commit
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        sys.exit(f"{commit!r} is no full commit")
    with tempfile.TemporaryDirectory() as work:
        root = Path(work)
        for path in PROTOS:
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            url = f"https://raw.githubusercontent.com/openxla/xla/{commit}/{path}"
            with urllib.request.urlopen(url) as source:
                (root / path).write_bytes(source.read())
        (TELEMETRY / "xla_protos.binpb").write_bytes(described(root))
    reader = TELEMETRY / "hlo.py"
    reader.write_text(re.sub(r'^XLA_COMMIT = "[0-9a-f]{40}"$', f'XLA_COMMIT = "{commit}"', reader.read_text(),
                             count=1, flags=re.M))


if __name__ == "__main__":
    main()
