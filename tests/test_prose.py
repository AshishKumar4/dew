"""Docstrings, comments and docs are written like a developer wrote them.

CONTRIBUTING.md bans the constructions machine-written text falls into. This
test reads the tree for the ones a regex can catch and fails on the first
match, with the line, so the fix lands where the text is. The research notes
and the design records are quoted material and stay out.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

FILES = sorted(
    [*ROOT.glob("src/dew/**/*.py"), *ROOT.glob("tests/*.py"), *ROOT.glob("recipes/**/*.py"),
     *ROOT.glob("examples/*.py"), *ROOT.glob("tools/*.py"), *ROOT.glob("docs/*.md"),
     *ROOT.glob("docs/concepts/*.md"), ROOT / "README.md", ROOT / "CONTRIBUTING.md"])

# Each row is a construction and the reason it goes. The pattern runs over
# prose only: docstrings, comments and markdown, never code.
BANNED = (
    (r"\bwhich is (what|how|why|where)\b", "a justification tail; state the fact"),
    (r"\b(used to|no longer|previously)\b", "narrates a past version; describe the code as it is"),
    (r"\bnot yet\b", "narrates a future version"),
    (r"\bin silence\b|\bsilently\b", "argues against an imagined bug; say what happens"),
    (r"\bexactly (as|what|where)\b", "emphasis; drop it"),
    (r"\bthe one (place|channel|owner|home|dispatch|thing|way|function|step|seam|test|construction|boundary)\b",
     "sells the design; name the function"),
    (r"\bnothing else\b|\bnothing more\b", "negative listing"),
    (r"\bhonest(ly)?\b|\bthe truth\b", "fake candour"),
    (r"\brather than\b|\binstead of\b(?= [a-z]+ing)", "binary contrast; state what the code does"),
    (r"(^|[^`])\bnot (just|only|merely) ", "binary contrast"),
    (r"\bby name\b", "a refusal is a ValueError with a message; say which"),
    (r"[—–]", "no dashes"),
    (r"\b(robust|seamless|leverage|utilize|delve|comprehensive|cutting-edge|elevate|harness|streamline|empower|paramount|intricate|transformative)\b",
     "words that sell"),
    (r", so (a|an|the|no|every|nothing|nobody|one) [^.\n]{0,60}\b(cannot|can never|never)\b", "argues with a reviewer"),
    (r"\bthat is (all|the whole|the entire)\b", "fake-profound"),
)

STRING = re.compile(r'("""|\'\'\')(.*?)\1', re.S)
COMMENT = re.compile(r"(?m)^\s*#(?!!).*$|(?<=\S)  #.*$")


def prose_of(path: Path) -> list[tuple[int, str]]:
    """Every prose line of `path` with its line number: the whole file for
    markdown, docstrings and comments for Python."""
    text = path.read_text(errors="replace")
    if path.suffix == ".md":
        lines = []
        fenced = False
        for number, line in enumerate(text.splitlines(), 1):
            if line.startswith("```"):
                fenced = not fenced
                continue
            # CONTRIBUTING's Writing section quotes the constructions it bans.
            if not fenced and not (path.name == "CONTRIBUTING.md" and line.startswith("- **")):
                lines.append((number, line))
        return lines
    spans = [m.span(2) for m in STRING.finditer(text)]
    spans += [m.span() for m in COMMENT.finditer(text)]
    starts = [text.count("\n", 0, a) + 1 for a, _ in spans]
    lines = []
    for (a, b), start in zip(spans, starts):
        for offset, line in enumerate(text[a:b].splitlines()):
            lines.append((start + offset, line))
    return lines


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_the_prose_reads_like_a_developer_wrote_it(path):
    found = []
    for number, line in prose_of(path):
        for pattern, reason in BANNED:
            if re.search(pattern, line, re.I):
                found.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()[:110]}  [{reason}]")
                break
    assert not found, f"{len(found)} banned construction(s):\n" + "\n".join(found[:40])
