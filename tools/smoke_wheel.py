#!/usr/bin/env python3
"""Install a built wheel into a new, empty virtual environment and use it, at both ends of its requirements.

    python tools/smoke_wheel.py dist/dewml-*.whl

The wheel is installed twice, each time into an empty environment, from PyPI's index and Dew's package index,
which serves Dew's JAX build, dewml's releases and the forks Dew carries (`INDEX`), so a requirement neither
can serve fails here: as a user installs it (pip, every requirement at its newest), and at the lower bound
each of dewml's requirements declares (uv's `lowest-direct` resolution), so a floor older than the code it
runs fails here too (row 862: 11 requirements had none, and optax's and huggingface_hub's predated an API
Dew calls). Then, in each, from outside the checkout so nothing of it is importable:

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

from packaging.requirements import Requirement

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_distribution

INDEX = runpy.run_path(str(Path(__file__).resolve().parents[1] / "src/dew/_builds.py"))[
    "PACKAGE_INDEX"]
"""Dew's package index, the one place Dew's JAX build is served (dew._builds.PACKAGE_INDEX)."""

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


def requirements(wheel: Path) -> list[str]:
    """The requirements `wheel` installs with no extra on this machine (its Requires-Dist, as
    tools/check_distribution.py reads it, each marker evaluated with no extra named)."""
    return [line for line in check_distribution.requirements(wheel)
            if (marker := Requirement(line).marker) is None or marker.evaluate({"extra": ""})]


def smoke(wheel: Path, lowest: bool) -> None:
    """`wheel` installed into an empty environment, every requirement at its newest or, `lowest`, at the lower
    bound dewml declares, and used (the module's doc)."""
    with tempfile.TemporaryDirectory(prefix="dew-wheel-") as scratch:
        root = Path(scratch)
        venv.create(root / "env", with_pip=True)
        python = root / "env" / "bin" / "python"
        # tools/pip.txt's pip first: older pips trust the roots of eMudhra's compromised CAs (2026-10-20).
        run(python, "-m", "pip", "install", "--quiet", "-r", Path(__file__).with_name("pip.txt"), cwd=root)
        indexes = ("--index-url", "https://pypi.org/simple", "--extra-index-url", INDEX)
        if lowest:
            # uv's lowest-direct lowers a project's own requirements, which a built wheel's are not to it:
            # they are named here as direct ones, read from the wheel's metadata, and the wheel goes in after.
            (root / "requirements.txt").write_text("\n".join(requirements(wheel)) + "\n")
            run("uv", "pip", "install", "--quiet", "--python", python, "--resolution", "lowest-direct",
                *indexes, "-r", root / "requirements.txt", cwd=root)
            run("uv", "pip", "install", "--quiet", "--python", python, "--no-deps", wheel, cwd=root)
            # What the lower bounds resolved to, for the log.
            run("uv", "pip", "freeze", "--python", python, cwd=root)
        else:
            run(python, "-m", "pip", "install", "--quiet", *indexes, wheel, cwd=root)
        # find_spec locates the package without importing it, so a module
        # that does not compile fails at compileall, named.
        locate = ("import importlib.util, pathlib; "
                  "print(pathlib.Path(importlib.util.find_spec('dew').origin).parent)")
        package = subprocess.run([str(python), "-c", locate], cwd=root, check=True, capture_output=True,
                                 text=True).stdout.strip()
        run(python, "-m", "compileall", "-q", package, cwd=root)
        run(root / "env" / "bin" / "dew", "--help", cwd=root)
        run(python, "-c", TRAIN, root / "tokens", cwd=root)
    print(f"{wheel.name}: installs at its requirements' {'lower bounds' if lowest else 'newest'}, compiles, "
          "runs `dew --help` and trains")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wheel", type=Path)
    wheel = parser.parse_args().wheel.resolve()
    for lowest in (False, True):
        smoke(wheel, lowest)


if __name__ == "__main__":
    sys.exit(main())
