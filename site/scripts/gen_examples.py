"""Append the example scripts, described by their own docstrings, to the examples page.

sync-docs writes the page from docs/examples.md; this adds one section per
file in `examples/`. The build fails when a script is missing from GROUPS or
GROUPS names a script that is gone.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAGE = REPO / "site/src/content/docs/examples.md"
SOURCE = "https://github.com/AshishKumar4/dew/blob/main/"

GROUPS = [
    ("Train one model", ["readme_demo", "train_lm", "train_flowers", "train_diffusion", "train_jepa"]),
    ("Run a whole job", ["train_flowers_tpu", "sft_gemma4", "sft_diffusion_gemma", "train_rlvr",
                         "train_harbor", "evaluate_and_serve"]),
]

COMMAND = re.compile(r"^(python|JAX_PLATFORMS=|CUDA_VISIBLE_DEVICES=|uv |dew |XLA_FLAGS=)")


def markdown(docstring: str) -> str:
    """A docstring as Markdown: command paragraphs become shell blocks."""
    out = []
    for paragraph in re.split(r"\n\s*\n", docstring.strip()):
        lines = paragraph.split("\n")
        indented = all(line.startswith("    ") or not line.strip() for line in lines)
        commands = all(COMMAND.match(line.strip()) or line.startswith((" ", "\t")) for line in lines if line.strip())
        if indented or (commands and COMMAND.match(lines[0].strip())):
            body = "\n".join(line[4:] if line.startswith("    ") else line for line in lines)
            # A command wrapped with backslashes keeps its line breaks; runs of spaces
            # inside one line were a continuation that lost its backslash.
            body = re.sub(r"(\S) {4,}(--)", r"\1 \\\n    \2", body)
            out.append(f"```bash\n{body.strip()}\n```")
        else:
            out.append(" ".join(line.strip() for line in lines))
    return "\n\n".join(out)


def main() -> None:
    scripts = {path.stem: path for path in sorted((REPO / "examples").glob("*.py"))}
    listed = [name for _, names in GROUPS for name in names]
    problems = [f"examples/{name}.py is not in scripts/gen_examples.py" for name in scripts if name not in listed]
    problems += [f"scripts/gen_examples.py lists examples/{name}.py, which does not exist" for name in listed if name not in scripts]
    if problems:
        print("gen_examples: the example list and examples/ disagree:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        raise SystemExit(1)
    if not PAGE.exists():
        raise SystemExit("gen_examples: run sync-docs first; it writes the page these sections extend")

    sections = []
    for label, names in GROUPS:
        sections += [f"## {label}", ""]
        for name in names:
            docstring = ast.get_docstring(ast.parse(scripts[name].read_text()))
            if not docstring:
                raise SystemExit(f"gen_examples: examples/{name}.py has no module docstring to describe it")
            summary, _, rest = docstring.partition("\n")
            sections += [f"### {name}.py", "", summary.strip(), ""]
            if rest.strip():
                sections += [markdown(rest), ""]
            sections += [f"[Source on GitHub]({SOURCE}examples/{name}.py)", ""]
    PAGE.write_text(PAGE.read_text().rstrip() + "\n\n" + "\n".join(sections).rstrip() + "\n")
    print(f"gen_examples: {len(scripts)} scripts")


if __name__ == "__main__":
    main()
