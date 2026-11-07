"""Each landing pool cell and the hero, run as the live kernel's training contexts run them: in a fresh
interpreter that installs guest_limits' "train" profile (no process creation) first. Prints each cell's outcome
and, for a failure, its exception and last frames."""
import json, subprocess, sys, pathlib, traceback
sys.path.insert(0, "site/snippets")
import cells as page

out = pathlib.Path(sys.argv[1])
programs = {name: page.cell(name) for name in page.CELLS["pool"]}
programs["hero"] = page.training_example(pathlib.Path("site/src/data/hero.py").read_text())
runner = (
    "import sys, traceback\n"
    "sys.path.insert(0, 'site/live/container')\n"
    "import guest_limits\n"
    "guest_limits.install('train')\n"
    "code = open(sys.argv[1]).read()\n"
    "try:\n"
    "    exec(compile(code, 'cell', 'exec'), {'__name__': '__main__'})\n"
    "except BaseException:\n"
    "    traceback.print_exc()\n"
    "    sys.exit(3)\n"
)
results = {}
for name, code in programs.items():
    path = out / f"{name}.py"
    path.write_text(code)
    done = subprocess.run([sys.executable, "-c", runner, str(path)], capture_output=True, text=True, timeout=1200)
    results[name] = {"exit": done.returncode, "tail": done.stderr[-2500:]}
    print(name, done.returncode, flush=True)
(out / "results.json").write_text(json.dumps(results, indent=1))
