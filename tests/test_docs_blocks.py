"""Every Python block in docs/ runs, page by page, unless it says why not.

A page's blocks run in order in one fresh interpreter on CPU, in an empty
working directory, so a block may use what the blocks above it built, and a
block whose call no longer matches the code fails here. A block that cannot
run in CI says why just above it: `<!-- not run: needs a GPU -->`. The design
records and research notes (docs/design, docs/research) sketch code rather
than instruct, and are not run.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
RECORDS = ("design", "research")
BLOCK = re.compile(r"^(?:<!-- not run: (?P<reason>.+?) -->\n)?```python\n(?P<code>.*?)^```", re.S | re.M)


def blocks(page: Path) -> list[tuple[int, str, str | None]]:
    """Each block's first line, its code and the reason it is not run, if it is not."""
    text = page.read_text()
    return [(text.count("\n", 0, found.start("code")) + 1, found["code"], found["reason"])
            for found in BLOCK.finditer(text)]


PAGES = [page.relative_to(DOCS).as_posix() for page in sorted(DOCS.rglob("*.md"))
         if page.relative_to(DOCS).parts[0] not in RECORDS and blocks(page)]


@pytest.mark.mesh(devices=0)
@pytest.mark.parametrize("page", PAGES)
def test_a_page_s_python_blocks_run_in_order(page, tmp_path):
    path = DOCS / page
    # Each block is compiled at its own lines of the page, so a traceback
    # names the page and the line in it.
    script = "\n".join(
        f"exec(compile({chr(10) * (line - 1) + code!r}, {str(path)!r}, 'exec'), scope)"
        for line, code, reason in blocks(path) if reason is None)
    # One CPU device, as a reader's machine has: the suite's device count
    # (XLA_FLAGS) would replicate each block's model across eight.
    env = {**{name: value for name, value in os.environ.items() if name != "XLA_FLAGS"},
           "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(ROOT / "src"), "WANDB_MODE": "disabled"}
    done = subprocess.run([sys.executable, "-c", f"scope = {{'__name__': '__main__'}}\n{script}"],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=1500)
    assert done.returncode == 0, f"{page}:\n{done.stdout[-2000:]}\n{done.stderr[-4000:]}"
