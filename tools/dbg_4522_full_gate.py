"""Run the unchanged window-gradient test, recording or judging its two CPU forms."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

NODE = "tests/test_chunk_adaptation.py::test_each_windows_gradient_from_its_level_is_the_literal_loops"
judge = sys.argv[1] == "--judge"
outcomes = {}
modes = ("highest",) if os.environ.get("POLICY_ONLY") else ("baseline", "token")
if os.environ.get("USER_ONLY"):
    modes = ("user",)
for mode in modes:
    environment = {**os.environ, "PYTHONPATH": "src:tests", "DEW_DBG_TOKEN": str(int(mode == "token")),
                   "DEW_DBG_POLICY_HIGHEST": str(int(mode == "highest")),
                   "DEW_DBG_USER_DEFAULT": str(int(mode == "user"))}
    with tempfile.TemporaryDirectory(prefix="dbg-4522-") as temporary:
        if judge:
            pool = Path(temporary)
            for path in Path(sys.argv[2]).glob(f"**/{mode}/draws/*.pkl"):
                shutil.copyfile(path, pool / path.name)
            print("POOL", mode, len(list(pool.glob("*.pkl"))), flush=True)
            environment["DEW_CHECK1_POOL"] = str(pool)
        else:
            first = (int(sys.argv[1]) - 1) * 64
            environment["DEW_CHECK1_DRAWS"] = f"{first}:{min(first + 64, 2560)}"
            environment["DEW_CHECK1_RECORDS"] = f"/armada/task/artifacts/{mode}/draws"
        result = subprocess.run([sys.executable, "-m", "pytest", "-p", "tools.dbg_4522_plugin",
                                 "-q", "--tb=short", "-o", "addopts=", NODE], env=environment)
        outcomes[mode] = result.returncode
        print("MODE", mode, "EXIT", result.returncode, flush=True)
print("RESULT", json.dumps(outcomes), flush=True)
expected = ({"highest": 0} if os.environ.get("POLICY_ONLY") else
            {"baseline": 1, "token": 0} if judge else {"baseline": 0, "token": 0})
if os.environ.get("USER_ONLY"):
    expected = {"user": 0}
raise SystemExit(0 if outcomes == expected else 1)
