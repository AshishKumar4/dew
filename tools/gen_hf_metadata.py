"""Generate `dew.interop.config_records.HF_METADATA`, the fields every Hugging Face config serializes that
describe no computation, from the pinned transformers' base config: `PretrainedConfig().to_dict()`'s fields,
and `torch_dtype`, the name transformers 4 wrote `dtype` under, which its configs still read. Dew states the
literal so that translating a config imports no transformers (row 3895); transformers alone decides it, and a
pin that changes its base config's fields changes it here.

    python -m tools.gen_hf_metadata [--check]

ruff wraps the literal, so the written file is the one the ruff gate passes."""

import argparse
import ast
import sys
from pathlib import Path

from tools.gen_exports import ruffed

REPO = Path(__file__).resolve().parents[1]
RECORDS = REPO / "src" / "dew" / "interop" / "config_records.py"
WRITTEN_BY = "# written from transformers' base config (tools/gen_hf_metadata.py)"


def metadata() -> list[str]:
    """The fields transformers' base config writes, and `torch_dtype`, sorted."""
    from transformers import PretrainedConfig

    return sorted({*PretrainedConfig().to_dict(), "torch_dtype"})


def written(path: Path = RECORDS) -> str:
    """`path` with its `HF_METADATA` assignment written from `metadata`."""
    text = path.read_text()
    node = next(node for node in ast.parse(text).body if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == "HF_METADATA" for target in node.targets))
    lines = text.splitlines()
    start = node.lineno - (2 if lines[node.lineno - 2] == WRITTEN_BY else 1)
    lines[start:node.end_lineno] = [WRITTEN_BY, "HF_METADATA = frozenset({",
                                    *(f'    "{name}",' for name in metadata()), "})"]
    return ruffed("\n".join(lines) + "\n", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HF_METADATA from transformers' base config.")
    parser.add_argument("--check", action="store_true", help="Check the written literal without writing it.")
    args = parser.parse_args()
    text = written()
    if text != RECORDS.read_text():
        if args.check:
            sys.exit("gen_hf_metadata: run python -m tools.gen_hf_metadata to rewrite "
                     f"{RECORDS.relative_to(REPO)}")
        RECORDS.write_text(text)
    print(f"gen_hf_metadata: {len(metadata())} fields")


if __name__ == "__main__":
    main()
