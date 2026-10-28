"""tools/lock_environment.sh's resolution, with uv's resolver standing in: what it asks uv for each lock."""

import functools
import http.server
import os
import shutil
import subprocess
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UV = """#!/bin/sh
# uv's stand-in: each call's arguments, one line each, and the lock it would write, written empty.
echo "$@" >> "$CALLS"
while [ $# -gt 0 ]; do [ "$1" = -o ] && : > "$2"; shift; done
"""


def test_every_package_dews_index_serves_resolves_undated(tmp_path, monkeypatch):
    """Dew's package index (constraints.txt's extra index) serves its own builds of packages (JAX's,
    Qwix's), whose static pages give no upload times, so every package its root page lists resolves with no
    date, however many the index serves: a lock dated past one of them finds none of its versions
    (2026-10-28: qwix joined the index, and every lock check failed on "qwix==0.1.8 has no publish time")."""
    site = tmp_path / "site" / "simple"
    site.mkdir(parents=True)
    (site / "index.html").write_text('<!DOCTYPE html>\n<html><body><h1>Simple index</h1>\n'
                                     '<a href="jax/">jax</a><br>\n<a href="qwix/">qwix</a><br>\n'
                                     '<a href="newfork/">newfork</a><br>\n</body></html>\n')
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path / "site"))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        checkout = tmp_path / "dew"
        (checkout / "tools").mkdir(parents=True)
        (checkout / "locks").mkdir()
        shutil.copy(ROOT / "tools/lock_environment.sh", checkout / "tools/lock_environment.sh")
        for name in ("pyproject.toml", ".python-version"):
            shutil.copy(ROOT / name, checkout / name)
        index = f"http://127.0.0.1:{server.server_address[1]}/simple"
        (checkout / "constraints.txt").write_text(
            (ROOT / "constraints.txt").read_text().replace("https://pypi.dewml.dev/simple", index))
        (tmp_path / "bin").mkdir()
        (tmp_path / "bin" / "uv").write_text(UV)
        (tmp_path / "bin" / "uv").chmod(0o755)
        monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}:{os.environ['PATH']}")
        monkeypatch.setenv("CALLS", str(tmp_path / "calls"))
        done = subprocess.run(["sh", "tools/lock_environment.sh"], cwd=checkout, capture_output=True,
                              text=True, timeout=120)
        assert done.returncode == 0, done.stderr
    finally:
        server.shutdown()
    calls = (tmp_path / "calls").read_text().splitlines()
    assert calls and all(f"--exclude-newer-package {name}=false" in call
                         for call in calls for name in ("jax", "qwix", "newfork")), calls
