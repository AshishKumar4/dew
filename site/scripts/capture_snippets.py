"""Run the landing page's script and record what it prints.

The landing page shows src/data/hero.py next to the output in
src/data/capture.json, so the output must come from running exactly that
file. Run this after changing it, in an environment with Dew installed, and
commit the JSON it writes:

    python site/scripts/capture_snippets.py --where "a workstation"

The script runs on JAX's default backend; `--where` names the machine for the
caption, and the device comes from JAX. The page reads the commit of the
installed Dew from pip's record of a git install, so install Dew from GitHub
or a local clone with `pip install "dewml @ git+..."`, not an editable
install.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from importlib.metadata import distribution, version
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "src/data"
SNIPPETS = ("hero",)


def installed_commit() -> str:
    record = distribution("dewml").read_text("direct_url.json")
    if record is None:
        raise SystemExit("dewml was not installed from git, so the page cannot name the commit it ran")
    return json.loads(record)["vcs_info"]["commit_id"]


def device() -> str:
    """The kind of device JAX runs the snippets on, asked in a process of its own."""
    return subprocess.run([sys.executable, "-c", "import jax; print(jax.devices()[0].device_kind)"],
                          capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--where", required=True, help='the machine, for the caption: "a workstation"')
    options = parser.parse_args()

    snippets = {}
    for name in SNIPPETS:
        started = time.monotonic()
        done = subprocess.run([sys.executable, str(DATA / f"{name}.py")], capture_output=True, text=True,
                              env={**os.environ, "PYTHONUNBUFFERED": "1"})
        if done.returncode != 0:
            raise SystemExit(f"{name}.py exited {done.returncode}:\n{done.stderr}")
        snippets[name] = {"returncode": 0, "seconds": round(time.monotonic() - started, 1), "stdout": done.stdout}
    capture = {
        "about": "What site/src/data/hero.py printed, written by site/scripts/capture_snippets.py.",
        "meta": {"where": options.where, "device": device(), "python": platform.python_version(),
                 "jax": version("jax"), "flax": version("flax"), "optax": version("optax"),
                 "dew": installed_commit(), "date": time.strftime("%Y-%m-%d")},
        "snippets": snippets,
    }
    (DATA / "capture.json").write_text(json.dumps(capture, indent="\t") + "\n")
    print(f"wrote {DATA / 'capture.json'}")


if __name__ == "__main__":
    main()
