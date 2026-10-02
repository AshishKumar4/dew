"""Run the landing page's script on a terminal and record what it showed.

The landing page shows src/data/hero.py next to the terminal it ran on, kept
in src/data/capture.json, so the output must come from running exactly that
file. Run this after changing it, in an environment with Dew and pyte
installed, and commit the JSON it writes:

    python site/scripts/capture_snippets.py --where "a workstation"

The script's stdout is a pseudo-terminal of `--columns` columns, so
`Trainer.fit` draws its live display there as it would for a person; its
stderr (JAX's and Python's warnings) is kept apart. Every byte written to the
terminal goes to `--cast`, an asciinema v2 recording, and capture.json keeps
the screen as it was when the script exited, cell by cell with its colours.

`--where` names the machine for the caption. Dew's commit comes from the
imported checkout (with clean source) or pip's record of a git install.
`--script` and `--output` also record the other small landing examples through
this same path; arguments after `--` go to that script.
"""

from __future__ import annotations

import argparse
import codecs
import fcntl
import hashlib
import json
import os
import platform
import pty
import select
import struct
import subprocess
import sys
import termios
import time
from importlib.metadata import distribution, version
from importlib.util import find_spec
from pathlib import Path

import pyte

DATA = Path(__file__).resolve().parents[1] / "src/data"
# Enough rows that nothing the script prints scrolls off the screen.
ROWS = 200


def installed_commit() -> str:
    directory = Path(find_spec("dew").origin).parents[2]
    if (directory / ".git").exists():
        changes = subprocess.check_output(["git", "-C", str(directory), "status", "--porcelain", "--", "src/dew"], text=True)
        if changes:
            raise SystemExit("commit Dew's source changes before recording")
        return subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    record = distribution("dewml").read_text("direct_url.json")
    if record is None:
        raise SystemExit("dewml was not installed from git, so the page cannot name the commit it ran")
    return json.loads(record)["vcs_info"]["commit_id"]


def run_on_terminal(script: Path, columns: int, arguments: list[str] | None = None) -> tuple[int, float, list[tuple[float, str]], str]:
    """Run `script` with stdout on a pseudo-terminal: its exit code, its
    seconds, what it wrote to the terminal with the time of each write, and
    its stderr."""
    primary, secondary = pty.openpty()
    fcntl.ioctl(secondary, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, columns, 0, 0))
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TERM": "xterm-256color", "COLUMNS": str(columns),
           "LINES": str(ROWS)}
    env.pop("NO_COLOR", None)
    started = time.monotonic()
    child = subprocess.Popen([sys.executable, str(script), *(arguments or [])], stdin=subprocess.DEVNULL, stdout=secondary,
                             stderr=subprocess.PIPE, env=env, cwd=script.parent)
    os.close(secondary)
    writes: list[tuple[float, str]] = []
    decoder = codecs.getincrementaldecoder("utf-8")()
    stderr = b""
    open_fds = {primary, child.stderr.fileno()}
    while open_fds:
        ready, _, _ = select.select(list(open_fds), [], [])
        for fd in ready:
            try:
                chunk = os.read(fd, 65536)
            except OSError:  # The terminal's other end closed.
                chunk = b""
            if not chunk:
                open_fds.discard(fd)
            elif fd == primary:
                writes.append((time.monotonic() - started, decoder.decode(chunk)))
            else:
                stderr += chunk
    code = child.wait()
    os.close(primary)
    return code, time.monotonic() - started, writes, stderr.decode(errors="replace")


class DimScreen(pyte.Screen):
    """A pyte screen that also keeps faint text (SGR 2), which pyte drops.

    pyte's cells have no field for it, so a faint cell's colour is kept as
    "dim:<colour>" and `screen_cells` splits it out again."""

    dim = False

    def select_graphic_rendition(self, *attrs: int) -> None:
        codes = list(attrs) or [0]
        index = 0
        while index < len(codes):
            code = codes[index]
            if code in (38, 48):  # A colour's own parameters: 5;n or 2;r;g;b.
                index += 3 if codes[index + 1:index + 2] == [5] else 5
                continue
            if code in (0, 22):
                self.dim = False
            elif code == 2:
                self.dim = True
            index += 1
        super().select_graphic_rendition(*attrs)
        fg = self.cursor.attrs.fg.removeprefix("dim:")
        self.cursor.attrs = self.cursor.attrs._replace(fg=f"dim:{fg}" if self.dim else fg)


def screen_cells(text: str, columns: int) -> list[list[dict]]:
    """The screen `text` leaves on a terminal of `columns` columns, as rows of
    runs of text sharing one style, without the empty rows at the end."""
    screen = DimScreen(columns, ROWS)
    stream = pyte.Stream(screen)
    # A terminal moves to the next line's start on \n only through the pty's
    # output translation, which already turned the child's \n into \r\n.
    stream.feed(text)
    return terminal_cells(screen, columns)


def terminal_cells(screen: DimScreen, columns: int) -> list[list[dict]]:
    """Read the current terminal as styled runs, without trailing empty rows."""
    rows = []
    for y in range(screen.lines):
        line = screen.buffer[y]
        runs: list[dict] = []
        for x in range(columns):
            char = line[x]
            fg = char.fg.removeprefix("dim:")
            style = {key: value for key, value in (("fg", fg), ("bold", char.bold), ("dim", fg != char.fg))
                     if value not in ("default", False)}
            if runs and runs[-1]["style"] == style:
                runs[-1]["text"] += char.data
            else:
                runs.append({"text": char.data, "style": style})
        while runs and not runs[-1]["text"].rstrip():
            runs.pop()
        if runs:
            runs[-1]["text"] = runs[-1]["text"].rstrip()
        rows.append([{"text": run["text"], **run["style"]} for run in runs])
    while rows and not rows[-1]:
        rows.pop()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--where", required=True, help='the machine, for the caption: "a workstation"')
    # Wide enough for the display's header line to name the whole mesh.
    parser.add_argument("--columns", type=int, default=96)
    parser.add_argument("--cast", type=Path, default=Path("hero.cast"), help="where to write the recording")
    parser.add_argument("--script", type=Path, default=DATA / "hero.py")
    parser.add_argument("--output", type=Path, default=DATA / "capture.json")
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="script arguments after --")
    options = parser.parse_args()

    script = options.script.resolve()
    arguments = options.arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    code, seconds, writes, stderr = run_on_terminal(script, options.columns, arguments)
    if code != 0:
        raise SystemExit(f"{script.name} exited {code}:\n{stderr}")
    header = {"version": 2, "width": options.columns, "height": ROWS, "timestamp": int(time.time()),
              "env": {"TERM": "xterm-256color"}}
    options.cast.write_text("\n".join([json.dumps(header)] + [json.dumps([round(t, 6), "o", text]) for t, text in writes]) + "\n")
    capture = {
        "about": f"The screen {script.name} left on a terminal, written by site/scripts/capture_snippets.py.",
        "arguments": arguments,
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "meta": {"where": options.where, "python": platform.python_version(),
                 "jax": version("jax"), "flax": version("flax"), "optax": version("optax"),
                 "dew": installed_commit(), "date": time.strftime("%Y-%m-%d")},
        "hero": {"returncode": code, "seconds": round(seconds, 1), "columns": options.columns,
                 "screen": screen_cells("".join(text for _, text in writes), options.columns)},
    }
    options.output.write_text(json.dumps(capture, indent="\t") + "\n")
    print(f"wrote {options.output} and {options.cast}")  # noqa: T201


if __name__ == "__main__":
    main()
