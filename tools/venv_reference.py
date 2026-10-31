"""Records a reference implementation run in its own environment over drawn weights and inputs, for a
test of Dew against it: each seed's draw (the test's `draw`) run by the site's worker in float32 and
float64, the recording sized paired against Dew's float32 run (the test's `dew`) and kept as its evidence
and its truth's sketch (`record_case(..., summary=True)`). An oracle site records the reference's float64
run alone, each of its declared seeds' outputs whole (`reference_runs.Oracle`), for a test that holds
Dew's float64 run to it (`assert_computes_the_oracle`).

    python tools/venv_reference.py SITE PYTHON [SOURCE]

PYTHON is the interpreter of a virtual environment holding the site's package; SOURCE a checkout put on
the worker's path where the package is a repository rather than a distribution.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))

SITES = {"conv": {"test": "test_dreamerv3_conv", "worker": "dreamerv3_worker.py",
                  "reference": "danijar/dreamerv3 e01491fad6434b2245a3b8ca201dd7faedcc458c",
                  "license": "MIT"},
         **{site: {"test": "test_flight", "oracle": site.upper(), "worker": "flydreamer_worker.py",
                   "reference": "AshishKumar4/FlyDreamer 043c9ef382048278ccf26849e54814ef5510a737",
                   "license": "MIT"} for site in ("course", "race")}}
"""Each site's test module (its `FIXTURE`, `draw`, `dew` and `OUTPUTS`, or the `oracle` it names), worker
and reference."""


def main() -> None:
    from recorded_references import provenance, save

    site, python = sys.argv[1], sys.argv[2]
    known = SITES[site]
    test = importlib.import_module(known["test"])
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    environment |= {"JAX_PLATFORMS": "cpu", **({"PYTHONPATH": sys.argv[3]} if len(sys.argv) > 3 else {})}
    worker = subprocess.Popen([python, str(ROOT / "tools" / known["worker"]), site], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, text=True, env=environment)
    scratch = Path(tempfile.mkdtemp())

    def evaluate(inputs):
        given, out = scratch / "in.npz", scratch / "out.npz"
        np.savez(given, **{f"case.{key}": value for key, value in inputs.items()})
        worker.stdin.write(f"{given} {out}\n")
        worker.stdin.flush()
        if worker.stdout.readline().strip() != "done":
            raise RuntimeError(f"{site}'s worker stopped")
        with np.load(out) as answered:
            return {key.removeprefix("case."): answered[key] for key in answered.files}

    if "oracle" in known:
        oracle = getattr(test, known["oracle"])
        fixture, arrays = oracle.fixture, {
            f"{seed}.{leaf}": value for seed in range(oracle.seeds)
            for leaf, value in oracle.outputs_of(evaluate(oracle.draw(seed))).items()}
    else:
        fixture, arrays = test.FIXTURE, checked(site, test, evaluate)
    worker.stdin.close()
    worker.wait()
    written = save(fixture, {**arrays, "provenance": provenance(
        known["reference"], "", "cpu",
        f"python tools/venv_reference.py {site} PYTHON{' SOURCE' if len(sys.argv) > 3 else ''}")},
        reference=known["reference"], license=known["license"])
    print(f"{written}: {sorted(arrays)}")


def checked(site: str, test, evaluate) -> dict[str, np.ndarray]:
    """A check-1 site's recording, sized paired against Dew's float32 run (`record_case`)."""
    import jax
    import jax.numpy as jnp
    from recorded_references import record_case

    def run(inputs):
        answered = evaluate(inputs)
        return tuple({leaf: answered[f"{kind}.{leaf}"] for leaf in test.OUTPUTS}
                     for kind in ("reference", "truth"))

    dew = jax.jit(lambda inputs: test.dew(jnp.float32, inputs))
    return record_case(site, test.draw, run, store_inputs=False, summary=True, dew=dew)


if __name__ == "__main__":
    main()
