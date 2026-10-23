#!/usr/bin/env python3
"""Install a built wheel into a new, empty virtual environment and use it.

    python tools/smoke_wheel.py dist/dewml-*.whl

The wheel is installed from PyPI's index and Dew's package index, which
serves Dew's JAX build alone (`INDEX`), as a user installs it, so a
requirement neither can serve fails here. Then, from outside the checkout so
nothing of it is importable:

- every module the wheel installed compiles (`compileall`), so a syntax
  error anywhere in the package fails, imported or not;
- the console entry point the wheel declares, `dew --help`, runs;
- the installed dewml, jax and flax report their versions, and flax.nnx imports;
- a two-layer grouped-query decoder's jitted forward, on its default attention route, gives finite logits
  of its shape;
- a tiny language model trains two steps through `Trainer.fit` on token
  stores exported here, and its parameters stay finite.

Exits nonzero on the first failure. The release workflow runs this on the
artifact it then uploads; CI's package job runs it on every push.
"""

from __future__ import annotations

import argparse
import runpy
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

INDEX = runpy.run_path(str(Path(__file__).resolve().parents[1] / "src/dew/builds.py"))[
    "PACKAGE_INDEX"]
"""Dew's package index, the one place Dew's JAX build is served (dew.builds.PACKAGE_INDEX)."""

TRAIN = """
import sys

import jax, jax.numpy as jnp, numpy as np, optax

import dew
from dew import Trainer
from dew.data import Loading, Memory, presets
from dew.data.compile import compile
from dew.data.sinks import export
from dew.data.sources import TokenFiles
from dew.nn.backbones import CausalTransformer
from dew.objectives.lm import LMObjective

assert "site-packages" in dew.__file__, dew.__file__
import importlib.metadata
import flax.nnx  # imports on jax 0.11.2 from flax 0.12.10 on
print("dewml", importlib.metadata.version("dewml"), "jax", jax.__version__, "flax",
      importlib.metadata.version("flax"))
# A grouped-query decoder's compiled forward, on its default attention route.
gqa = CausalTransformer(vocab_size=64, emb_features=32, num_layers=2, num_heads=4, num_kv_heads=2,
                        mlp_features=64, max_seq_len=16)
ids = jnp.arange(16, dtype=jnp.int32).reshape(2, 8)
logits = jax.jit(gqa.apply)(gqa.init(jax.random.key(0), ids), ids)
assert logits.shape == (2, 8, 64) and bool(jnp.all(jnp.isfinite(logits))), logits.shape
# Four symbols repeated, a token store to train on and one to validate on, as an export commits them.
stores = {split: f"{sys.argv[1]}-{split}" for split in ("train", "val")}
for split, repeats in (("train", 64), ("val", 16)):
    export(Memory([{"text": np.tile(np.arange(1, 5, dtype=np.int32), repeats)}]), stores[split],
           format="array_record", length_field="text",
           metadata={"token_store": True, "tokenizer": "symbols", "vocab_size": 8, "dtype": "int32"})
train = TokenFiles(stores["train"]).shuffle(0).repeat().pipe(presets.token_windows, seq_len=8, batch=4)
held = TokenFiles(stores["val"]).pipe(presets.token_windows, seq_len=8, batch=4, records=4)
data = compile(train, validation=held, read="memory", loading=Loading(workers=0, threads=1, read_buffer=2))
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
        # tools/pip.txt's pip first: older pips trust the roots of eMudhra's compromised CAs (2026-10-20).
        run(python, "-m", "pip", "install", "--quiet", "-r", Path(__file__).with_name("pip.txt"), cwd=root)
        run(python, "-m", "pip", "install", "--quiet", "--index-url", "https://pypi.org/simple",
            "--extra-index-url", INDEX, wheel, cwd=root)
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
