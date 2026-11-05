"""A receipt's paths as it records them: relative to the checkout inside it, by name outside it, so no
receipt names the directories of the host that ran it (bible §2; rows 2785, 2864, 2872)."""

from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
"""The checkout a tool runs from."""


def named(path: str | Path, parts: int = 1) -> str:
    """`path` as a receipt records it: a remote URI (a bucket's) as given; a local path, or a `file://`
    URI's, relative to the checkout inside it, outside it by its last `parts` parts (a TFDS directory's
    dataset and version, 2), its root never among them."""
    scheme, remote, rest = str(path).partition("://")
    if remote and scheme != "file":
        return str(path)
    local = Path(rest) if remote else Path(path)
    whole = local.resolve()
    if whole.is_relative_to(ROOT):
        return str(whole.relative_to(ROOT))
    return "/".join([part for part in local.parts if part != local.anchor][-parts:])


def recorded(config: object, paths: tuple[str, ...] = ()) -> dict:
    """`config`'s fields (a dataclass's) as a receipt records them: each path, and each field `paths` names
    that holds one as a string, `named`."""
    return {name: named(value) if isinstance(value, Path) or (name in paths and isinstance(value, str))
            else value for name, value in asdict(config).items()}
