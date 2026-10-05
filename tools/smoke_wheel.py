#!/usr/bin/env python3
"""Install a built wheel into a new, empty virtual environment and use it.

    python tools/smoke_wheel.py dist/dewml-*.whl

The wheel is installed from PyPI's index alone, so a requirement the index
cannot serve fails here. Then, from outside the checkout so nothing of it
is importable:

- every module the wheel installed compiles (`compileall`), so a syntax
  error anywhere in the package fails, imported or not;
- the console entry point the wheel declares, `dew --help`, runs;
- a tiny language model trains two steps through `Trainer.fit` on token
  files written here, and its parameters stay finite.

Exits nonzero on the first failure. The release workflow runs this on the
artifact it then uploads; CI's package job runs it on every push.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

TRAIN = """
import json, sys
from pathlib import Path

import jax, jax.numpy as jnp, numpy as np, optax

import dew
from dew import Trainer
from dew.data import Loading, TokenWindows
from dew.nn.backbones import CausalTransformer
from dew.objectives.lm import LMObjective

assert "site-packages" in dew.__file__, dew.__file__
tokens = Path(sys.argv[1])
tokens.mkdir()
for split, repeats in (("train", 64), ("val", 16)):
    np.tile(np.array([1, 2, 3, 4], dtype=np.uint8), repeats).tofile(tokens / f"{split}.bin")
(tokens / "meta.json").write_text(json.dumps({
    "tokenizer": "symbols", "vocab_size": 8, "dtype": "uint8", "train_tokens": 256, "val_tokens": 64,
    "eos_id": None}))
data = TokenWindows(path=str(tokens), seq_len=8, val_batches=1,
                    loading=Loading(workers=0, threads=1, read_buffer=2)).load(batch=4)
model = CausalTransformer(vocab_size=8, emb_features=16, num_layers=1, num_heads=2, mlp_features=32,
                          max_seq_len=16, dtype=jnp.float32, attention_impl="xla")
state = Trainer(LMObjective(model, seq_len=8, ema_decay=None), optax.adam(1e-2),
                key=jax.random.key(0)).fit(data, steps=2, log_every=1)
assert int(state.step) == 2, int(state.step)
assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(state.variables["params"]))
print("trained", int(state.step), "steps")
"""


def run(*command: str | Path, cwd: Path) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run([str(part) for part in command], cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wheel", type=Path)
    wheel = parser.parse_args().wheel.resolve()
    with tempfile.TemporaryDirectory(prefix="dew-wheel-") as scratch:
        root = Path(scratch)
        venv.create(root / "env", with_pip=True)
        python = root / "env" / "bin" / "python"
        run(python, "-m", "pip", "install", "--quiet", "--index-url", "https://pypi.org/simple", wheel,
            cwd=root)
        # find_spec locates the package without importing it, so a module
        # that does not compile fails at compileall, named.
        locate = ("import importlib.util, pathlib; "
                  "print(pathlib.Path(importlib.util.find_spec('dew').origin).parent)")
        package = subprocess.run([str(python), "-c", locate], cwd=root, check=True, capture_output=True,
                                 text=True).stdout.strip()
        run(python, "-m", "compileall", "-q", package, cwd=root)
        run(root / "env" / "bin" / "dew", "--help", cwd=root)
        run(python, "-c", TRAIN, root / "tokens", cwd=root)
    print(f"{wheel.name}: installs, compiles, runs `dew --help` and trains")


if __name__ == "__main__":
    sys.exit(main())
