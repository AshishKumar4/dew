"""Every module of dew imports as the first thing a fresh interpreter does.

An import cycle only shows when a module is the first of its cycle to load:
`import dew.nn.multimodal` in a fresh process met `dew.nn.backbones`,
whose diffusion transformers imported `dew.diffusion`, whose masked-token
process imported `dew.nn.multimodal` again, half initialized. Imports from
inside the suite never hit it, since `dew` is already loaded.
"""
import os
import pkgutil
import subprocess
import sys
from pathlib import Path

import dew

SOURCE = Path(dew.__file__).parent.parent


def test_every_module_imports_first_in_a_fresh_interpreter():
    names = sorted(module.name for module in pkgutil.walk_packages(dew.__path__, "dew."))
    script = """
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

def imported(name):
    run = subprocess.run([sys.executable, "-c", f"import {name}"], capture_output=True, text=True)
    return f"{name}: {run.stderr.strip().splitlines()[-1]}" if run.returncode else None

with ThreadPoolExecutor(os.cpu_count() or 4) as pool:
    print("\\n".join(line for line in pool.map(imported, sys.argv[1:]) if line))
"""
    run = subprocess.run([sys.executable, "-c", script, *names], capture_output=True, text=True,
                         env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(SOURCE)}, timeout=1800)
    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.strip() == "", run.stdout
