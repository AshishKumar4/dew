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


def test_a_tests_child_process_compiles_into_the_suites_cache():
    """A process a test starts (a checker run as a script, a CLI) compiles into the cache the suite's
    process keeps, under the same settings, so its programs reach the shared bundles and its next run is
    warm: tools/check_native_diffusion.py's xl case compiled cold each run, 139.6 s, and warm 43.3 s
    (2026-10-29: row 2323). Under DEW_TEST_NO_CACHE neither caches."""
    import jax

    names = ("jax_enable_compilation_cache", "jax_compilation_cache_dir",
             "jax_persistent_cache_min_compile_time_secs", "jax_persistent_cache_min_entry_size_bytes")
    read = f"import jax, json; print(json.dumps([getattr(jax.config, name) for name in {list(names)!r}]))"
    child = json.loads(subprocess.run([sys.executable, "-c", read], capture_output=True, text=True,
                                      check=True).stdout)
    assert child == [getattr(jax.config, name) for name in names]


LATE = """
import json
import sys

from dew.cache import enable_compilation_cache

import jax

# The cache on, every program kept, whatever the suite's own cache policy hands this process.
jax.config.update("jax_enable_compilation_cache", True)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
enable_compilation_cache(sys.argv[1])
jax.jit(lambda x: x * 3)(1).block_until_ready()
enable_compilation_cache(sys.argv[2])
jax.jit(lambda x: x - 5)(1).block_until_ready()
print(json.dumps({"directory": jax.config.jax_compilation_cache_dir}))
"""


def test_a_cache_directory_named_after_the_first_compile_holds_the_programs_compiled_after(tmp_path):
    """JAX settles its persistent cache at the process's first compile; a directory a run names after it
    (`prepare_process` in a process that compiled, as a second run in a notebook does) holds the
    programs compiled after, not the first directory (row 2836)."""
    from process_support import fresh

    first, later = tmp_path / "first", tmp_path / "later"
    held = fresh(LATE, str(first), str(later), cache_dir=first, flags="")
    assert held["directory"] == str(later)

    def programs(directory):
        return {path.name for path in directory.glob("*-cache")}

    assert len(programs(first)) == 1 and len(programs(later)) == 1



PICKS = """
import json
import os
import sys

from dew.cache import enable_compilation_cache, kernel_picks_path

enable_compilation_cache(sys.argv[1])
named = [flag for flag in os.environ["XLA_FLAGS"].split()
         if flag.startswith("--xla_gpu_per_fusion_autotune_cache_dir=")]
print(json.dumps({"xla": named[0].partition("=")[2], "kernels": kernel_picks_path("NVIDIA H100")}))
"""


def installed_plugin(site: pathlib.Path, build: str, version: str) -> None:
    """The CUDA `build`'s PJRT plugin at `version` as an install leaves it under `site`: its `jax_plugins`
    package, which `dew.devices.cuda_builds` finds the build by (JAX calls its `initialize`), and its
    distribution's metadata."""
    package = site / "jax_plugins" / f"xla_{build}"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("def initialize():\n    pass\n")
    info = site / f"jax_{build}_pjrt-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: jax-{build}-pjrt\nVersion: {version}\n")


def test_a_changed_pjrt_plugin_build_reads_no_earlier_pick(tmp_path):
    """XLA's picks and Dew's kernel picks are measured by the PJRT plugin's compile (XLA, Triton and Mosaic
    GPU live there), which post42 to post45 rebuild under one jaxlib (row 3560): a process names the directory
    of its jaxlib and its plugin's build, so a pick measured under post45 is read by post45 again and by no
    other build, in a process that finds the plugin as an install leaves it."""
    cache = tmp_path / "cache"
    base = {key: value for key, value in os.environ.items() if key != "XLA_FLAGS"}
    root = pathlib.Path(__file__).resolve().parent.parent
    sites, named = {}, {}
    for process, version in (("first", "0.12.0.1.post45"), ("other build", "0.12.0.1.post46"),
                             ("again", "0.12.0.1.post45")):
        site = sites[process] = tmp_path / process.replace(" ", "-")
        installed_plugin(site, "cuda12", version)
        env = {**base, "JAX_PLATFORMS": "cpu", "PYTHONPATH": os.pathsep.join(
            filter(None, (str(site), str(root / "src"), os.environ.get("PYTHONPATH"))))}
        done = subprocess.run([sys.executable, "-c", PICKS, str(cache)], capture_output=True, text=True,
                              env=env, timeout=300, cwd=root)
        assert done.returncode == 0, done.stderr[-2000:]
        named[process] = json.loads(done.stdout.strip().splitlines()[-1])
        if process == "first":
            pathlib.Path(named[process]["xla"], "pick").write_text("measured under post45")
            pathlib.Path(named[process]["kernels"]).parent.mkdir(parents=True)
            pathlib.Path(named[process]["kernels"]).write_text("{}")
    assert named["again"] == named["first"]
    assert pathlib.Path(named["again"]["xla"], "pick").read_text() == "measured under post45"
    other = named["other build"]
    for kind in ("xla", "kernels"):
        assert not pathlib.Path(other[kind]).is_relative_to(named["first"]["xla"]), kind
        assert not pathlib.Path(named["first"][kind]).is_relative_to(other["xla"]), kind
    assert not pathlib.Path(other["xla"], "pick").exists()
    assert not pathlib.Path(other["kernels"]).exists()
