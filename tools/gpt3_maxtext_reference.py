"""Records MaxText's GPT-3 (maxtext 0.2.5) over drawn weights and tokens for tests/test_maxtext_gpt3.py:
each seed's draw (the test's `draw`) run by tools/maxtext_gpt3_worker.py in an environment holding
maxtext, in float32 and float64, the recording sized paired against Dew's float32 run and kept as its
evidence and its truth's sketch (`record_case(..., summary=True)`).

    python tools/gpt3_maxtext_reference.py MAXTEXT_PYTHON

MAXTEXT_PYTHON is the interpreter of a virtual environment holding `maxtext==0.2.5` and its JAX.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))


def main() -> None:
    import jax
    import test_maxtext_gpt3 as site
    from recorded_references import provenance, record_case, save

    python = sys.argv[1]
    version = subprocess.run([python, "-c", "import importlib.metadata as m; print(m.version('maxtext'))"],
                             capture_output=True, text=True, check=True).stdout.strip()
    if version != "0.2.5":
        raise SystemExit(f"the recording is of maxtext 0.2.5; {python} holds {version}")
    worker = subprocess.Popen([python, str(ROOT / "tools" / "maxtext_gpt3_worker.py"), "--serve"],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                              env={**{key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
                                   "JAX_PLATFORMS": "cpu"})
    scratch = Path(tempfile.mkdtemp())

    def evaluate(inputs):
        given, out = scratch / "in.npz", scratch / "out.npz"
        np.savez(given, **{f"case.{key}": value for key, value in inputs.items()})
        worker.stdin.write(f"{given} {out}\n")
        worker.stdin.flush()
        if worker.stdout.readline().strip() != "done":
            raise RuntimeError("MaxText's worker stopped")
        with np.load(out) as answered:
            return {"logits": answered["case.reference"]}, {"logits": answered["case.truth"]}

    model = site.decoder()
    dew = jax.jit(lambda inputs: site.logits(model, inputs))
    arrays = record_case("gpt3", site.draw, evaluate, store_inputs=False, summary=True, dew=dew)
    worker.stdin.close()
    worker.wait()
    written = save(site.FIXTURE, {**arrays, "provenance": provenance(
        "maxtext", version, "cpu", f"python tools/gpt3_maxtext_reference.py {python}")},
        reference=f"AI-Hypercomputer/maxtext {version}", license="Apache-2.0")
    print(f"{written}: {sorted(arrays)}")


if __name__ == "__main__":
    main()
