"""tools/lock_environment.sh's resolution, with uv's resolver standing in: what it asks uv for each lock."""

import functools
import http.server
import os
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
UV = """#!/bin/sh
# uv's stand-in: each call's arguments, one line each; the lock it writes keeps the pins its output holds
# already, as uv prefers them, else pins for the machine it resolves for, which $SPLIT_MACHINE makes differ by
# machine; and a resolution for the Python $NO_WHEEL names fails, as one no wheel of a pinned plugin serves.
echo "$@" >> "$CALLS"
python= machine=universal output=
while [ $# -gt 0 ]; do
  case $1 in
    --python-version) python=$2 ;;
    --python-platform) machine=$2 ;;
    -o) output=$2 ;;
  esac
  shift
done
[ "$python" != "${NO_WHEEL:-}" ] || { echo "no wheel matches the Python ABI tag cp${python#3.}" >&2; exit 1; }
[ -s "$output" ] && [ -z "${SPLIT_MACHINE:-}" ] || echo "plugin==1 ; $machine" > "$output"
"""


def locked(tmp_path: Path, monkeypatch, **environment: str) -> tuple[subprocess.CompletedProcess, list[str]]:
    """tools/lock_environment.sh run in a copy of the checkout with uv's stand-in (`UV`, under `environment`),
    Dew's package index served locally listing jax, qwix, a new fork and jax-cuda12-plugin: its run and its
    calls to uv."""
    site = tmp_path / "site" / "simple"
    site.mkdir(parents=True)
    (site / "index.html").write_text('<!DOCTYPE html>\n<html><body><h1>Simple index</h1>\n'
                                     '<a href="jax/">jax</a><br>\n<a href="qwix/">qwix</a><br>\n'
                                     '<a href="newfork/">newfork</a><br>\n'
                                     '<a href="jax-cuda12-plugin/">jax-cuda12-plugin</a><br>\n'
                                     '</body></html>\n')
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
        for name, value in environment.items():
            monkeypatch.setenv(name, value)
        done = subprocess.run(["sh", "tools/lock_environment.sh"], cwd=checkout, capture_output=True,
                              text=True, timeout=120)
    finally:
        server.shutdown()
    return done, (tmp_path / "calls").read_text().splitlines()


def test_every_package_dews_index_serves_resolves_undated(tmp_path, monkeypatch):
    """Dew's package index (constraints.txt's extra index) serves its own builds of packages (JAX's,
    Qwix's), whose static pages give no upload times, so every package its root page lists resolves with no
    date, however many the index serves, a dated list naming it or not: a lock dated past one of them finds
    none of its versions (2026-10-28: qwix joined the index, and every lock check failed on "qwix==0.1.8 has
    no publish time"; 2026-10-31: the CUDA plugins did, both undated and dated, and every version was
    filtered)."""
    done, calls = locked(tmp_path, monkeypatch)
    assert done.returncode == 0, done.stderr
    assert calls and all(f"--exclude-newer-package {name}=false" in call
                         for call in calls for name in ("jax", "qwix", "newfork", "jax-cuda12-plugin")), calls
    # A package the index serves takes no date of a dated list either: both flags at once filtered every
    # version of the CUDA plugin the index took (2026-10-31, ledger 3116: every lane's lock check failed).
    assert not any("--exclude-newer-package jax-cuda12-plugin=20" in call for call in calls), calls
    assert all("--exclude-newer-package jax-cuda13-plugin=20" in call for call in calls), calls


def test_a_gpu_extra_that_resolves_otherwise_for_aarch64_fails_the_lock(tmp_path, monkeypatch):
    """Each GPU lock serves aarch64 too only where it pins the same there: one whose aarch64 resolution pins
    otherwise fails, naming the lock (train 433's review: a stand-in writing every lock empty never could)."""
    done, _ = locked(tmp_path, monkeypatch, SPLIT_MACHINE="1")
    assert done.returncode == 1
    assert "locks/cuda12.txt resolves otherwise for aarch64" in done.stderr


# 3.12 is the locks' own Python, whose lock fails first, with uv's own word.
@pytest.mark.parametrize("python", ["3.13", "3.14"])
def test_a_gpu_extra_that_does_not_install_on_a_python_pyproject_admits_fails_the_lock(
        tmp_path, monkeypatch, python):
    """Each GPU extra resolves on every Python from requires-python's floor to the newest classifier, on
    each machine, so a CUDA plugin built for fewer Pythons fails the lock at its pin (row 3298: post42's
    plugins came for 3.12 and 3.14 alone, and dewml[cuda12] refused 3.13 while every lock, for 3.12, held)."""
    done, calls = locked(tmp_path, monkeypatch, NO_WHEEL=python)
    assert done.returncode == 1
    assert f"does not install on Python {python}" in done.stderr
    done, calls = locked(tmp_path / "every", monkeypatch, NO_WHEEL="3.11")
    assert done.returncode == 0, done.stderr
    asked = {(extra, minor, machine) for call in calls for extra in ("cuda12", "cuda13")
             for minor in ("3.12", "3.13", "3.14") for machine in ("x86_64", "aarch64")
             if f"--extra {extra} " in call and f"--python-version {minor} " in call
             and f"--python-platform {machine}-manylinux_2_35" in call}
    assert len(asked) == 12
