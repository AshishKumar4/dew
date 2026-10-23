"""Run isolated single-draw probes in separate processes; print sensitive leaves only."""
import os
import subprocess
import sys


for mode in sys.argv[2:]:
    result = subprocess.run([sys.executable, "tests/probe_aligned_nosysquid.py", "wide:" + mode,
                             sys.argv[1]], env=os.environ, text=True, capture_output=True, check=True)
    print("\n".join(line for line in result.stdout.splitlines()
                    if any(name in line for name in (".log_decay", ".gate", ".log_a", ".bias"))), flush=True)
