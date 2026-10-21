"""The timing log the choice layer learns from (design §7).

A measured config of a launch is one row (`observation`): the launch and call by their encodings, the
config, how it was measured, its samples, its lowering and compile times, its outcome, and the analytic
estimate where the launch has one. Where the environment variable `DEW_TIMINGS` names a directory
(`directory`), `record` appends rows to `timings.jsonl` there; a caller builds rows only then, so with it
unset or empty nothing is built, imported or written. The tuner's measurement loop is that caller
(`tuning.observed`); logging only observes: it never tunes, never runs another config and never changes
a pick or an error.

A row carries three kinds of field, so a ranker trained on them can only read the first:
- predictors, which a trace has: the call, the config and the estimate (features join them);
- labels, which a measurement gives: the samples, the lowering and compile times, the outcome;
- identity and provenance: the run, the observation, the versions and the device.
"""

import os
import uuid

ENV = "DEW_TIMINGS"
"""The environment variable naming the directory timings are appended to."""

SCHEMA = 1
"""The rows' format, raised when a field changes meaning."""

IDENTITY = 0
"""How a row identifies its launch's hooks: 0, by tuning's module-level names (`tuning.encoded`), until
the traced-program fingerprint (design §4) replaces it."""

LOCK_SECONDS = 60.0
"""How long an append waits for another process's lock on the file before raising: a boundary wait."""

MAX_TEXT = 16384
"""The longest launch, call, config or method text a row keeps as written; a longer one is kept as its
SHA-256 and length."""

MAX_SAMPLES = 64
"""The most samples a row keeps; `samples_total` counts all of them."""

RUN = uuid.uuid4().hex
"""This process's run, which every row it writes carries."""

CUPTI_EVICTED = {'name': 'cupti_evicted', 'units': 'ms',
                 'aggregation': "each run's sum of the launch's own kernels"}
"""`tuning`'s measurement: CUPTI's device time of the launch's kernels, each run after its operands are
evicted from cache, the eviction's own kernels left out."""


def directory() -> str | None:
    """The directory `DEW_TIMINGS` names, None where it is unset or empty."""
    return os.environ.get(ENV) or None


def observation(launch: str, call: str, config: str, *, device, method: dict, status: str,
                samples_ms: tuple = (), lower_ms: float | None = None, compile_ms: float | None = None,
                estimate_s: float | None = None, features: dict | None = None,
                error: BaseException | None = None) -> dict:
    """One measured config as a row: `launch` the op's name, `call` and `config` their encodings,
    `device` the one it ran on, `method` how it was timed, `status` its outcome ('timed',
    'resource_exhausted', 'failed' or 'skipped'), `features` the launch's (`templates.features`'s row,
    None where it states none), and an error by its class's name alone. Texts and samples are bounded
    (`MAX_TEXT`, `MAX_SAMPLES`)."""
    from importlib.metadata import version

    import jax

    return {
        'schema': SCHEMA, 'identity': IDENTITY, 'run': RUN, 'observation': uuid.uuid4().hex,
        'launch': _bounded(launch), 'call': _bounded(call), 'config': _bounded(config),
        'estimate_s': estimate_s, 'features': features,
        'method': {key: _bounded(str(value)) for key, value in method.items()},
        'status': status, 'error': None if error is None else type(error).__name__,
        'samples_ms': [float(sample) for sample in samples_ms[:MAX_SAMPLES]],
        'samples_total': len(samples_ms), 'lower_ms': lower_ms, 'compile_ms': compile_ms,
        'jax': jax.__version__, 'jaxlib': version('jaxlib'), 'platform': device.platform,
        'device_kind': device.device_kind,
    }


def _bounded(text: str) -> str:
    """`text` as written, or past `MAX_TEXT` its SHA-256 and length."""
    if len(text) <= MAX_TEXT:
        return text
    import hashlib

    return f"sha256:{hashlib.sha256(text.encode()).hexdigest()}:{len(text)}"


def record(rows: list[dict], *, wait: float = LOCK_SECONDS) -> None:
    """Append `rows` to `timings.jsonl` in `DEW_TIMINGS`'s directory, created where missing, each row
    one line, all of them under one lock, waited for `wait` seconds, so a concurrent writer's lines
    never interleave with them. A failure to write raises: logging was asked for."""
    where = directory()
    if where is None or not rows:
        return
    import json
    from pathlib import Path

    import filelock

    path = Path(where)
    path.mkdir(parents=True, exist_ok=True)
    lines = ''.join(json.dumps(row, sort_keys=True, allow_nan=False) + '\n' for row in rows)
    with filelock.FileLock(path / 'timings.jsonl.lock', timeout=wait), \
            (path / 'timings.jsonl').open('a', encoding='utf-8') as file:
        file.write(lines)
