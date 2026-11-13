"""A rehearsal of a recipe and its real run share their step: the run's length reaches its programs as
operands alone (`dew.optim_schedules.RateState.horizons`), so the real run, in a fresh process, reads every
program the rehearsal compiled from the persistent cache and compiles none."""

import json
import os
import subprocess
import sys
from pathlib import Path

RUN = r'''
import dataclasses, json, sys
from pathlib import Path
sys.path.insert(0, "tests")
from dew.training.runtime import prepare_process
prepare_process(multi_host=False, compilation_cache_dir=sys.argv[2])
from dew.telemetry.instrumentation import compilation_cache_counts, count_compilation_cache
from dew.optim import Cosine, OptimConfig
from run_kinds import lm
count_compilation_cache()
root, steps = Path(sys.argv[1]), int(sys.argv[3])
run = lm(root, drawn=False)
trainer = dataclasses.replace(run.trainer, checkpoint_dir=str(root / "runs"), steps=steps, log_every=1,
                              eval_every=None, checkpoint_every=None, checkpoint_final=False,
                              compilation_cache_dir=sys.argv[2], multi_host=False)
optim = OptimConfig(optimizer="adamw", learning_rate=Cosine(peak_value=1e-3, warmup_steps="10%"))
dataclasses.replace(run, trainer=trainer, optim=optim).prepare().train("run")
print(json.dumps(compilation_cache_counts()))
'''


def trained(root: Path, cache: Path, steps: int) -> dict[str, float]:
    """The persistent cache's hits and misses of a process that trains the recipe for `steps` updates."""
    done = subprocess.run([sys.executable, "-c", RUN, str(root), str(cache), str(steps)], capture_output=True,
                          text=True, timeout=900, env={**os.environ, "JAX_PLATFORMS": "cpu"})
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(done.stdout.splitlines()[-1])


def test_a_real_run_compiles_nothing_its_shorter_rehearsal_compiled(tmp_path):
    """A rehearsal of 4 updates, whose 10% warmup is none, then the real run of 40, whose warmup is 4, each
    a process and a directory of its own over one persistent cache: the real run reads its every program
    from the cache, its start and its step among them, and misses none."""
    rehearsal = trained(tmp_path / "rehearsal", tmp_path / "cache", 4)
    assert rehearsal.get("train/cache_misses", 0) >= 2
    real = trained(tmp_path / "real", tmp_path / "cache", 40)
    assert real.get("train/cache_misses", 0) == 0 and real.get("train/cache_hits", 0) >= 2, real
