"""Where Dew keeps what it caches on disk, and JAX's persistent compilation cache.

Kept apart from the telemetry and the loaders that read it, so the
interop, config and inference modules take their cache paths without
importing the FLOP accounting.
"""

import dataclasses
import functools
import importlib.metadata
import os
import sys
from collections.abc import Callable
from pathlib import Path

import jax

from dew.devices import deterministic_ops_requested, xla_flag


def dew_cache_dir() -> str:
    """Dew's cache directory: `$XDG_CACHE_HOME/dew`, else ~/.cache/dew."""
    return os.path.expanduser(
        os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.join("~", ".cache"), "dew")
    )


def dew_config_dir() -> Path:
    """Dew's configuration directory, which every configuration file of Dew's is read from:
    `$DEW_CONFIG_DIR`, else `$XDG_CONFIG_HOME/dew`, else ~/.config/dew."""
    override = os.environ.get("DEW_CONFIG_DIR")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CONFIG_HOME")
    return (Path(xdg) if xdg else Path.home() / ".config") / "dew"


@dataclasses.dataclass(frozen=True)
class HubAnswer[Answer]:
    """An answer of the Hub's (`hub_answer`): `value`, and whether it was `recorded`, read back
    from an earlier answer with the Hub offline."""

    value: Answer
    recorded: bool


def hub_answer[Answer](question: tuple[str, ...], ask: Callable[[], Answer]) -> HubAnswer[Answer]:
    """The Hub's answer to `question`: `ask`'s, recorded where answers are kept (DEW_HUB_ANSWERS
    when set, else `hub/` in Dew's cache), or with the Hub offline, where `ask` raises
    OfflineModeIsEnabled as every Hub call does under HF_HUB_OFFLINE, the answer last recorded
    (`recorded_hub_answer`). What an online run pinned (a revision's commit, a commit's files) so
    resolves the same offline; a question never answered online raises the offline error."""
    import json

    from huggingface_hub.errors import OfflineModeIsEnabled

    from dew.files import write_atomically

    try:
        value = ask()
    except OfflineModeIsEnabled:
        recorded = recorded_hub_answer(question)
        if recorded is None:
            raise
        return recorded
    record = _hub_record(question)
    os.makedirs(os.path.dirname(record), exist_ok=True)
    write_atomically(record, json.dumps({"question": list(question), "answer": value}))
    return HubAnswer(value, recorded=False)


def recorded_hub_answer(question: tuple[str, ...]) -> HubAnswer | None:
    """The answer `hub_answer` last recorded to `question`, or None where it recorded none."""
    import json

    record = _hub_record(question)
    if not os.path.isfile(record):
        return None
    with open(record) as recorded:
        held = json.load(recorded)
    return HubAnswer(held["answer"], recorded=True) if held["question"] == list(question) else None


def _hub_record(question: tuple[str, ...]) -> str:
    """Where `question`'s answer is recorded: a file named by the hash of its parts, which no
    revision's text can steer outside the answers' directory; the record holds the question
    beside the answer."""
    import hashlib
    import json

    answers = os.environ.get("DEW_HUB_ANSWERS") or os.path.join(dew_cache_dir(), "hub")
    return os.path.join(answers, hashlib.sha256(json.dumps(list(question)).encode()).hexdigest() + ".json")


def default_compilation_cache_dir() -> str:
    """Where compiled executables go unless a run names somewhere else.

    The directory JAX is configured with (`jax_compilation_cache_dir`, which
    JAX_COMPILATION_CACHE_DIR sets) when there is one, so a machine keeps one
    cache for every entry point. Otherwise Python minors have separate
    defaults: jax 0.11.2 compresses with Python 3.14's stdlib zstd but names
    the codec "zlib" in the key, which says "zstandard" only for the
    zstandard package (`jax._src.compilation_cache.get_cache_key`), so an
    older interpreter sharing the directory would read those bytes with the
    wrong codec. Hosts of every CPU class share it: Dew's JAX build keys a
    CPU program by the host's CPU (0.12.0.1.post13), which XLA:CPU compiles
    for, so no host reads a program that does not link there. Explicit paths
    passed to enable_compilation_cache remain unchanged.
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
    directory nor the mode is in JAX's cache key, so a run and a process
    that compiled its programs ahead of it share executables. XLA reads its
    flags at the first compile; a directory the run names is kept.

    A run under `--xla_gpu_deterministic_ops` keeps no picks (`keeps_picks`):
    jax 0.11.2's XLA measures nothing there and gives each instruction the
    first configuration that compiles, but would still take a stored pick
    over that one, so such a run names an empty directory.
    """
    os.makedirs(path, exist_ok=True)
    jax.config.update('jax_compilation_cache_dir', path)
    # The autotuner's picks follow the policy above, through XLA_FLAGS; JAX's own placement of them
    # in the cache directory would name one directory for every jaxlib and write picks from a
    # process this policy keeps to reading (`jax._src.compiler.get_compile_options`).
    jax.config.update('jax_persistent_cache_enable_xla_caches', 'none')
    # Defaults skip small/fast compilations; a training step is neither, and
    # caching everything keeps startup predictable.
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.0)
    flags = os.environ.get('XLA_FLAGS', '')
    if '--xla_gpu_per_fusion_autotune_cache_dir' not in flags:
        picks = autotune_dir(path) if keeps_picks() else ''
        if picks:
            os.makedirs(picks, exist_ok=True)
        os.environ['XLA_FLAGS'] = f"{flags} --xla_gpu_per_fusion_autotune_cache_dir={picks}".strip()


def autotune_dir(path: str) -> str:
    """Where the GPU picks of the executables cached under `path` go, for this
    jaxlib alone: XLA's per fusion, and Dew's kernels' beside them
    (`kernel_picks_path`). jaxlib pins XLA, Triton and Mosaic, which each
    pick's speed depends on."""
    return os.path.join(path, 'autotune', f"jaxlib-{importlib.metadata.version('jaxlib')}")


def kernel_picks_path(device_kind: str) -> str:
    """The file of Dew's kernel configs measured on GPUs of `device_kind`
    (`dew.nn.kernels.tuning`), beside XLA's picks in the cache this process
    compiles to."""
    name = device_kind.lower().replace(' ', '_').replace('/', '_')
    return os.path.join(autotune_dir(default_compilation_cache_dir()), 'kernels', f"{name}.json")


KERNEL_TUNES = jax.make_user_context(0)
"""How many times this process's kernel picks changed (`dew.nn.kernels.tuning.tune`):
a context JAX keys its trace and compile caches on, so a program lowered
after picks change traces again. Made at `import dew`, as JAX makes a user
context only beside no other JAX call."""

KERNEL_LAUNCHES = jax.make_user_context(default_value=False)
"""Whether a traced kernel launch carries tokamax's description of itself
(`dew.nn.kernels.tuning.launches`), keyed on as `KERNEL_TUNES` is. Only a
lowering made to read its launches out of carries it, never a step's: jaxlib
0.11.2 writes a compiled program holding it to the persistent cache and cannot
read it back ("Invalid metadata payload id")."""

KERNEL_SEEN = jax.make_user_context(default_value=None)
"""The collection a lowering records its kernel launches into as it traces them
(`dew.nn.kernels.tuning.seen`): JAX keys its trace caches on its identity, so a
trace cached before traces, and records, again. The program is the same, and so
is its key in the persistent cache, which reads no user context."""

KERNEL_ROUTES = jax.make_user_context(())
"""The collections each open `dew.nn.kernels.generation.routes` block records its dispatches' routes
into: JAX keys its trace caches on their identities, as on `KERNEL_SEEN`'s, so a trace cached before
traces, and reports its routes, again."""

KERNEL_PINS = jax.make_user_context(())
"""The `(kernel, config)` pairs `dew.nn.kernels.tuning.pinned` holds in this
thread, keyed on as `KERNEL_TUNES` is."""

KERNEL_ASSUMED = jax.make_user_context(default_value=None)
"""The device `dew.nn.kernels.generation.assumed` stands in for in this thread, its generation and its
facts where given them, keyed on as `KERNEL_TUNES` is, so a program traced for one device's launches
traces again for another's."""

KERNEL_CHOICES = jax.make_user_context(())
"""The `(choice, kernel)` pairs `dew.nn.kernels.generation.chosen` holds in
this thread, keyed on as `KERNEL_TUNES` is, so a program traced before a
choice traces again within it."""

QUANTIZED_TRACE = jax.make_user_context(default_value=None)
"""What a quantized model's methods trace under, its provider's kind and spec
(`dew.quantization`), keyed on as `KERNEL_TUNES` is. Qwix intercepts
`jax.lax.dot_general` by swapping its code while a wrapped method runs, and
`jnp.matmul`, `jnp.dot` and the other contractions that reach it are jitted,
so a trace of one taken under the swap holds its quantized product, which JAX's
cache would hand a later plain call of the same shapes, and a plain trace a
quantized call. In this context their traces are kept apart."""


INTERCEPTED = jax.make_user_context(default_value=False)
"""Whether Dew installed a layer interceptor in this trace: adapters, input QDQ and Qwix retain
their individual projection calls. This is independent of Qwix's arithmetic trace key above."""


def keeps_picks() -> bool:
    """Whether this process reads and stores GPU picks, XLA's and the
    kernels' alike: not under `--xla_gpu_deterministic_ops`, whose kernels
    must stay the same between a run and its resumption while other runs
    measure into the cache. There XLA takes the first configuration that
    compiles (`enable_compilation_cache`) and Dew's kernels their heuristic
    ones (`dew.nn.kernels.tuning`), the same in every process."""
    return not deterministic_ops_requested()


def measures_picks() -> bool:
    """Whether this process may measure GPU picks and store them, XLA's and
    the kernels' alike: where it keeps picks at all (`keeps_picks`), and
    not beside other work on its GPU, where `gpu-run -n 0.5` asks XLA to
    read its picks and write none (AUTOTUNE_CACHE_MODE_READ), since a
    measurement there is unfair."""
    mode = xla_flag('xla_gpu_experimental_autotune_cache_mode')
    return keeps_picks() and mode != 'AUTOTUNE_CACHE_MODE_READ'


def persist_compilations() -> None:
    """Point XLA at the on-disk executable cache, unless a directory is set or JAX's cache is
    turned off.

    A run compiles its step for seconds to minutes, a loaded task the first time its shapes are
    seen (a minute for a text-to-image sample on an A100), and every restart compiles them again
    without it. A process Dew readies for training turns it on (`dew.training.runtime.
    ready_process`, from `prepare_process` or the first `fit`), and a task where it is loaded,
    `dew.pipeline` or a saved run's record (`dew.inference.tasks.SavedRun.pinned`). JAX's own switch,
    JAX_ENABLE_COMPILATION_CACHE=false (`jax_enable_compilation_cache`), turns it off. JAX writes
    entries from process 0 alone, so a pool takes the directory only where every process reads it
    as one (`_shared_by_pool`): caches apart would find a step on one process and compile it on
    another, which waits for ever for its peers. Each process keys a step by its own topology, so a
    CPU pool's other processes, whose topologies are their own devices, compile what process 0
    loads (docs/guides/multi-node.md). Reading the settings is what makes it idempotent and what
    leaves a trainer's own directory, or a caller's, alone.

    JAX settles whether a process uses the cache at its first compile: one that compiled with the
    cache turned off keeps it off, and one that compiled before any directory was set takes this
    one. XLA reads the autotuner's flags (`enable_compilation_cache`) as the backend opens, so a
    process whose backend is open keeps its executables here and measures its picks again.
    """
    if jax.config.jax_compilation_cache_dir or not jax.config.jax_enable_compilation_cache:
        return
    from dew._coordination import pool_size

    directory = default_compilation_cache_dir()
    if pool_size() > 1 and not _shared_by_pool(directory):
        return
    enable_compilation_cache(directory)


@functools.cache
def _shared_by_pool(directory: str) -> bool:
    """Whether every process of the pool reads `directory` as one (`seen_by_every_process`)."""
    from dew._coordination import seen_by_every_process

    return all(seen_by_every_process(directory))


def hub_commit(repo: str, revision: str | None, *, repo_type: str = "model") -> HubAnswer[str]:
    """The commit `revision` of the Hub's `repo` names now, its default branch's where None, as a run
    records it. Online the Hub's answer (`hub_answer`), written to the Hub cache's ref of the revision as
    huggingface_hub writes it on a download, so another process sharing the cache resolves it alike;
    offline the cache's ref, whichever tool wrote it, else the answer recorded online, both `recorded`."""
    from huggingface_hub import HfApi, constants
    from huggingface_hub.errors import OfflineModeIsEnabled
    from huggingface_hub.file_download import repo_folder_name

    from dew.files import write_atomically

    storage = Path(constants.HF_HUB_CACHE) / repo_folder_name(repo_id=repo, repo_type=repo_type)
    branch = revision or constants.DEFAULT_REVISION
    # The keys earlier runs recorded: a model's default branch as main, a dataset's as HEAD.
    question = (("commit", repo, branch) if repo_type == "model"
                else (f"{repo_type} commit", repo, revision or "HEAD"))

    def ask() -> str:
        info = (HfApi().model_info if repo_type == "model" else HfApi().dataset_info)(repo, revision=revision)
        if info.sha is None:
            raise ValueError(f"HF {repo!r} did not resolve {branch} to a commit")
        return info.sha

    ref = storage / "refs" / branch
    try:
        answer = hub_answer(question, ask)
    except OfflineModeIsEnabled:
        if not ref.is_file():
            raise
        return HubAnswer(ref.read_text().strip(), recorded=True)
    if answer.recorded and ref.is_file():
        # Offline the cache's ref stands over an answer recorded before another tool moved it.
        return HubAnswer(ref.read_text().strip(), recorded=True)
    if not answer.recorded and branch != answer.value and (
            not ref.is_file() or ref.read_text().strip() != answer.value):
        ref.parent.mkdir(parents=True, exist_ok=True)
        write_atomically(ref, answer.value)
    return answer
