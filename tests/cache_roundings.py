"""Check 1's arms of a computation through a quantized key-value cache (`KVCache.quantized`), and the float64
truth that reads what an arm's cache held (`assert_as_exact`'s `followed`, docs/design/testing.md).

A quantized cache rounds each token's key (in the cache's rotation, `KVCache.key_rotation`) and value, a
vector a key head, to codes and a float32 scale (`dew.nn.kv_cache.quantize`) as its store writes them
(`KVStore.write`, a call's rows; `KVStore.write_tokens`, a serving step's tokens), and every query of the row
reads them back dequantized. Inside `traced`, each such rounding a trace writes reports itself as its
program runs (`io_callback`): the store's layer (its module's path), its payload (`cached_key`,
`cached_value`), each token's row and slot, the values it rounded and the codes and scales it decided. The
reports reach the innermost `collected` on the host, whichever program runs them: a server's steps are
traced once and run for every draw. With `decided`, each rounding takes its codes and scales from it at its
tokens' slots instead, so a float64 run reads the keys and values an arm's cache held and computes every
other value itself."""

import contextlib
import functools
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import io_callback
from quantized_basis import quantized_codes

from dew.nn import kv_cache
from dew.nn.kv_cache import KVStore

_COLLECTING: list["Written"] = []
"""The `collected` blocks open on the host, innermost last, which a running rounding reports to."""

CAPTURED = jax.make_user_context(default_value=False)
"""Whether a trace's cache roundings report themselves (`traced`), which JAX's trace caches key on as on Dew's
own trace contexts (`dew._cache`): a program a server compiles under the capture is never run by an ordinary
server of the same model, nor an ordinary one's under the capture."""


class Written:
    """The cache roundings a run's programs reported inside `collected`: every write of each token, by its
    store's layer and payload (`<module path>/<payload>`) and its row and slot."""

    def __init__(self):
        self.tokens: dict[str, dict[tuple[int, int], list]] = {}

    def met(self, name: str, rows_held: int, rows, positions, values, codes, scales) -> None:
        """A rounding's tokens as its write placed them: each kept where its slot is not -1 and its row is
        one of the `rows_held` its cache holds, which its store's write drops otherwise."""
        rows, positions = np.broadcast_to(rows, positions.shape).reshape(-1), positions.reshape(-1)
        values, codes = (np.reshape(held, (positions.size, *held.shape[-2:])) for held in (values, codes))
        scales = np.reshape(scales, (positions.size, scales.shape[-1]))
        writes = self.tokens.setdefault(name, {})
        for index in np.flatnonzero((positions >= 0) & (rows < rows_held)):
            writes.setdefault((int(rows[index]), int(positions[index])), []).append(
                (values[index], codes[index], scales[index]))

    def row(self) -> int:
        """The one row a run of one request wrote."""
        rows = {row for writes in self.tokens.values() for row, _ in writes}
        if len(rows) != 1:
            raise AssertionError(f"rows {sorted(rows)} were written: a run of one request writes one row")
        return rows.pop()


def held(rounds: Sequence[Written], length: int) -> tuple[dict, dict]:
    """The last of `rounds`' request's first `length` slots of every layer's payload, each one write: its own
    round's in its row (`Written.row`), or, a slot its round read from a shared prefix's pages (a prefix
    cache's, `Server.from_task`'s `prefix_cache`), the earliest earlier round's in that round's row, which
    wrote the pages the prefix cache kept. The values its roundings rounded, and their decisions, codes and
    scales, by name."""
    order = [(written, written.row()) for written in (rounds[-1], *rounds[:-1])]
    rounded, decided = {}, {}
    for name in rounds[-1].tokens:
        taken = []
        for slot in range(length):
            written = next((written.tokens[name][row, slot] for written, row in order
                            if (row, slot) in written.tokens.get(name, {})), [])
            if len(written) != 1:
                raise AssertionError(f"{name}: slot {slot} of the request's row written {len(written)} times")
            taken.append(written[0])
        values, codes, scales = (np.stack(part) for part in zip(*taken, strict=True))
        rounded[name], decided[name] = values, (codes, scales)
    return rounded, decided


def _reported(name: str, rows_held: int, *arrays) -> None:
    if not _COLLECTING:
        raise AssertionError(f"{name}: a cache rounding ran outside `collected`")
    _COLLECTING[-1].met(name, rows_held, *arrays)


@contextlib.contextmanager
def collected():
    """The cache roundings the programs traced under `traced` report while they run inside (`Written`),
    every report in before it closes."""
    written = Written()
    _COLLECTING.append(written)
    try:
        yield written
    finally:
        jax.effects_barrier()
        _COLLECTING.pop()


@contextlib.contextmanager
def traced(decided: dict | None = None):
    """Each cache rounding a store writes while tracing inside reports itself to `collected` as its program
    runs; with `decided` (one row's decisions by name, `held`'s), each takes its codes and scales
    from it at its tokens' slots, so a run of one row reads them. A program compiled inside, and so a
    server constructed inside, runs inside: its traces are keyed apart (`CAPTURED`)."""
    writing: list[tuple[KVStore, jax.Array, jax.Array, list[str]]] = []
    quantize, write, write_tokens = kv_cache.quantize, KVStore.write, KVStore.write_tokens

    @contextlib.contextmanager
    def placed(store: KVStore, rows: jax.Array, positions: jax.Array):
        writing.append((store, rows, positions, [payload for payload, _ in store.payloads]))
        try:
            yield
        finally:
            writing.pop()

    def written(store: KVStore, key, value, positions) -> None:
        rows = jnp.broadcast_to(jnp.arange(positions.shape[0])[:, None], positions.shape)
        with placed(store, rows, positions):
            write(store, key, value, positions)

    def tokens_written(store: KVStore, key, value, rows, positions, groups=None) -> None:
        with placed(store, rows, positions):
            write_tokens(store, key, value, rows, positions, groups)

    def rounded(values, dtype):
        # The store's payloads round in its order (`KVStore.payloads`): its keys', then its values'.
        store, rows, positions, payloads = writing[-1]
        name = "/".join((*store.module.path, payloads.pop(0)))
        if decided is None:
            codes, scales = quantize(values, dtype)
        else:
            codes, scales = (jnp.take(held, positions, axis=0) for held in decided[name])
        io_callback(functools.partial(_reported, name, store.rows), None, rows, positions, values, codes,
                    scales)
        return codes, scales

    with pytest.MonkeyPatch.context() as patch, CAPTURED(new_value=True):
        patch.setattr(kv_cache, "quantize", rounded)
        patch.setattr(KVStore, "write", written)
        patch.setattr(KVStore, "write_tokens", tokens_written)
        yield


def rounds(quantized: str):
    """A served quantized cache's `Followed.rounds`: each value its judged leaves `rounded/<name>` hold,
    rounded as the `quantized` format rounds it (`quantized_basis.quantized_codes`), which the decisions an
    arm's cache made (`<name>`, codes and scales) must equal."""
    def rounded(judged: dict) -> dict:
        return {name.removeprefix("rounded/"): quantized_codes(values, quantized)
                for name, values in judged.items() if name.startswith("rounded/")}
    return rounded
