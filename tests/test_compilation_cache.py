"""A process Dew readies for training keeps JAX's persistent compilation cache under the user's
cache directory, unless JAX's own switch turns it off (`dew.cache.persist_compilations`)."""

import json
import os
import pathlib
import subprocess
import sys

import pytest

RUN = """
import json
import sys

sys.path.insert(0, "tests")


def run():
    if sys.argv[1] == "prepare":
        from dew.training import prepare_process
        prepare_process(multi_host=False)
    import jax
    from affine_training import fit_one_step
    fit_one_step()
    return jax.config.jax_compilation_cache_dir


print(json.dumps({"directory": run()}))
"""


@pytest.mark.parametrize("enabled", [True, False], ids=["default", "turned-off"])
@pytest.mark.parametrize("entry", ["prepare", "fit"])
def test_a_run_caches_its_compilations_under_the_users_cache_directory(tmp_path, entry, enabled):
    """A run readied by `prepare_process` with no directory of its own, or by a `Trainer.fit` no
    `prepare_process` came before, writes its compiled programs under $XDG_CACHE_HOME/dew/xla, in its
    Python minor's directory, so its restart reads them; JAX_ENABLE_COMPILATION_CACHE=false leaves it
    without the cache."""
    env = {key: value for key, value in os.environ.items()
           if key not in ("JAX_COMPILATION_CACHE_DIR", "JAX_ENABLE_COMPILATION_CACHE")}
    env.update(JAX_PLATFORMS="cpu", XDG_CACHE_HOME=str(tmp_path))
    if not enabled:
        env["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    root = pathlib.Path(__file__).resolve().parent.parent
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(root / "src"), os.environ.get("PYTHONPATH"))))
    done = subprocess.run([sys.executable, "-c", RUN, entry], capture_output=True, text=True, env=env,
                          timeout=300, cwd=root)
    assert done.returncode == 0, done.stderr[-2000:]
    directory = json.loads(done.stdout.strip().splitlines()[-1])["directory"]
    minor = f"python{sys.version_info.major}.{sys.version_info.minor}"
    expected = tmp_path / "dew" / "xla" / minor
    if enabled:
        assert directory == str(expected)
        assert any(path.is_file() and "autotune" not in path.parts for path in expected.rglob("*"))
    else:
        assert not directory
        assert not (tmp_path / "dew" / "xla").exists()
