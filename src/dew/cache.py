"""Where Dew keeps what it caches on disk, and JAX's persistent compilation cache.

Kept apart from the telemetry and the loaders that read it, so the
interop, config and inference modules take their cache paths without
importing the FLOP accounting.
"""

import importlib.metadata
import os
import sys

import jax


def dew_cache_dir() -> str:
    """Dew's cache directory: `$XDG_CACHE_HOME/dew`, else ~/.cache/dew."""
    return os.path.expanduser(
        os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.join("~", ".cache"), "dew")
    )


def default_compilation_cache_dir() -> str:
    """Where compiled executables go unless a run names somewhere else.

    The directory JAX is configured with (`jax_compilation_cache_dir`, which
    JAX_COMPILATION_CACHE_DIR sets) when there is one, so a machine keeps one
    cache for every entry point. Otherwise Python minors have separate
    defaults: jax 0.11.2 compresses with Python 3.14's stdlib zstd but names
    the codec "zlib" in the key, which says "zstandard" only for the
    zstandard package (`jax._src.compilation_cache.get_cache_key`), so an
    older interpreter sharing the directory would read those bytes with the
    wrong codec. Explicit paths passed to enable_compilation_cache remain
    unchanged.
    """
    if jax.config.jax_compilation_cache_dir:
        return jax.config.jax_compilation_cache_dir
    return os.path.join(dew_cache_dir(), 'xla', f"python{sys.version_info.major}.{sys.version_info.minor}")


def enable_compilation_cache(path: str):
    """Persist compiled executables so restarts skip XLA compilation, and the
    GPU autotuner's picks beside them.

    The dominant cost of a restart-heavy TPU workflow, where every run otherwise
    recompiles the same step function from scratch.

    XLA's GPU compile measures each fusion's candidate kernels on the device.
    The picks go to `<path>/autotune/jaxlib-<version>`, one file a fusion keyed
    by the fusion and the GPU model, for this XLA alone, so a later compile of
    the same fusions takes them without measuring. A process beside other work
    on its GPU would measure unfairly, so it reads them and adds none
    (`gpu-run -n 0.5` sets XLA's AUTOTUNE_CACHE_MODE_READ). Neither the
    directory nor the mode is in JAX's cache key, so a run and the process
    that compiled ahead of it (`TrainerConfig.compile_only`) share
    executables. XLA reads its flags at the first compile; a directory the
    run names is kept.
    """
    os.makedirs(path, exist_ok=True)
    jax.config.update('jax_compilation_cache_dir', path)
    # Defaults skip small/fast compilations; a training step is neither, and
    # caching everything keeps startup predictable.
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.0)
    flags = os.environ.get('XLA_FLAGS', '')
    if '--xla_gpu_per_fusion_autotune_cache_dir' not in flags:
        picks = os.path.join(path, 'autotune', f"jaxlib-{importlib.metadata.version('jaxlib')}")
        os.makedirs(picks, exist_ok=True)
        os.environ['XLA_FLAGS'] = f"{flags} --xla_gpu_per_fusion_autotune_cache_dir={picks}".strip()


def persist_compilations() -> None:
    """Point XLA at the on-disk executable cache, unless a directory is set.

    A loaded task compiles for seconds the first time its shapes are seen
    (a minute for a text-to-image sample on an A100), and a serving process
    restarts. Training turns the same cache on in `prepare_process`; a task
    turns it on where it is loaded, `dew.pipeline` or a saved run's record
    (`dew.inference.tasks.run_record`). Reading the setting is what makes it
    idempotent and what leaves a trainer's own directory, or a caller's, alone.
    """
    if jax.config.jax_compilation_cache_dir:
        return
    enable_compilation_cache(default_compilation_cache_dir())
