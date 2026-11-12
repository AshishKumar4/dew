"""One process of a real pool writing a model whose weights span the pool through a run's export
(`dew.interop.saved_sources.export_run`).

Driven by tests/test_saved_sources.py. Every process places one small decoder over a mesh across the
pool, extends each of its layers' attention with DEX (`dew.nn.dex`), whose export folds each output
projection as it is written, and exports it as a saved run's release, the run's restore
(`Pretrained.from_saved`) standing in for the placed decoder and the release cut into small shards,
process 0 publishing and the other reading every tensor as it does (`publishes`). Each records the
files it wrote and, exporting a deeper decoder (`DEEP`), how far its allocated bytes rise during the
export and while it holds that decoder's whole tree home (`allocated_growth`), and the decoder's size;
outside the export's scope the write is refused. It also writes the same weights unplaced, as one
process does. Every mode records what it saw into --out and asserts nothing.

dew is imported inside the modes, not at module scope, because a JAX backend opened before
jax.distributed.initialize() would pin the process to its own devices.
"""

from __future__ import annotations

import contextlib
import functools
import gc
import hashlib
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

from flax import linen as nn
from process_support import serve_mode, worker_parser

FIELDS = {"vocab_size": 512, "emb_features": 64, "num_layers": 8, "num_heads": 4, "num_kv_heads": 2,
          "mlp_features": 128, "max_seq_len": 8}
"""A decoder of about 1.3 MB in float32, its largest tensor (the embedding) 128 KiB."""

DEEP = {**FIELDS, "num_layers": 32, "emb_features": 128, "mlp_features": 512}
"""A decoder of about 31 MB in float32 whose largest tensors (the embedding, a feed-forward kernel) are
256 KiB each, so a host holding one tensor home at a time holds a hundredth of it."""

TINY_SHARD = 4
"""Far below the production threshold, so the pool's mesh shards every kernel."""

SHARD = "64KB"
"""The release's shard size, so it is written in a score of shards."""

RELEASE_SHARD = "1MB"
"""A saved run's release's shard size (`released`), so its 17 MB are written in shards each a fraction of
it: the writer holds one shard at a time, and at the 5 GB default the release is one shard, which held
the whole release on each host while tracemalloc read 0.25 MB."""


def digests(directory: Path) -> dict[str, str]:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.glob("*.safetensors"))}


def tensors(directory: Path) -> dict[str, tuple[str, list[int], str]]:
    """Each tensor a directory's weight files hold, by name: its dtype, shape and bytes' digest."""
    from safetensors.numpy import load_file

    return {name: (str(array.dtype), list(array.shape), hashlib.sha256(array.tobytes()).hexdigest())
            for path in sorted(directory.glob("*.safetensors")) for name, array in load_file(path).items()}


SAMPLED = 0.005
"""Seconds between the counts `allocated_growth` takes inside its block."""


@contextlib.contextmanager
def allocated_growth(into: list[int]) -> Iterator[None]:
    """How far the bytes this process holds allocated (`dew.telemetry.benchmark.allocated_memory`) rise
    above where they start within the block, appended to `into` however it ends: the most of its counts
    every `SAMPLED` seconds and as it ends (`resident_peak` over that counter), after a collection. What a
    block holds home past a sample's interval is counted: a whole tree read home before it is written,
    which the export's bound is about."""
    from dew.telemetry.benchmark import allocated_memory, resident_peak

    gc.collect()
    peak = None
    try:
        with resident_peak(SAMPLED, counter=allocated_memory) as peak:
            yield
    finally:
        if peak is not None:
            into.append(peak.rise("Allocated"))


def exported(args) -> dict:
    """The decoder exported placed across the pool, and written unplaced and outside the scope; and the
    deeper decoder's export measured, and its whole tree held home."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental import multihost_utils

    from dew.inference.banks import HeldBanks
    from dew.interop.pretrained import Pretrained, PretrainedDecoder
    from dew.interop.releases import ReleasePlan
    from dew.interop.saved_sources import export_run
    from dew.nn.dex import DifferentialExtension
    from dew.registry import ModelConfig, models, to_record
    from dew.training import Layout, MeshSpec

    ReleasePlan.write = functools.partialmethod(ReleasePlan.write, max_shard_size=SHARD)

    def built(fields):
        """The decoder of `fields`, its variables, those placed across the pool, every layer's attention
        extended by DEX, and the placed decoder's bundle."""
        model = models.build("causal_transformer", **fields, scan_layers=args.scan)
        variables = nn.unbox(model.init(jax.random.key(0), jnp.ones((1, 4), jnp.int32)))
        placed = HeldBanks(variables).place(model, mesh=MeshSpec(fsdp=args.fsdp_size),
                                            layout=Layout(min_shard=TINY_SHARD, tolerance=1.0))
        extension = DifferentialExtension(tuple((layer, (0, 2)) for layer in range(fields["num_layers"])),
                                          anneal_steps=10)
        return model, variables, placed, extension, PretrainedDecoder.from_model(model, placed).with_dex(
            extension, key=0)

    model, variables, placed, extension, bundle = built(FIELDS)
    leaves = jax.tree.leaves(placed)
    root = Path(args.run_dir)
    rank = jax.process_index()
    try:
        bundle.save(root / f"outside{rank}")
        refused = False
    except RuntimeError:
        refused = True
    # A run of a causal decoder, as its record states its model, which an export reads its layout by.
    run = SimpleNamespace(objective=SimpleNamespace(saved_task=None),
                          record={"model": to_record(ModelConfig.from_model(model), ModelConfig)})

    def restoring(exported) -> None:
        """`export_run`'s model from here on: `exported`, which the run's restore stands in for."""
        Pretrained.from_saved = classmethod(lambda cls, run, **options: exported)

    def export(name: str) -> None:
        export_run(run, root / name, publishes=rank == 0)

    destination = root / f"release{rank}"
    restoring(bundle)
    export(destination.name)
    # The deeper decoder exported once before it is measured, a copy built apart so nothing the measured
    # one holds is left from it: the measured export finds its programs compiled and its modules imported.
    # The copy is released, its restore replaced, before the measure starts, not freed within it.
    restoring(built(DEEP)[4])
    export(f"warm{rank}")
    _, _, deep, _, deep_bundle = built(DEEP)
    restoring(deep_bundle)
    growth: list[int] = []
    with allocated_growth(growth):
        export(f"deep{rank}")
    # The same measure over the deeper decoder's whole tree held home, every process taking part as the
    # export's reads do: it sees what a host holding the model holds.
    with allocated_growth(growth):
        held = [np.array(leaf) if leaf.is_fully_addressable
                else np.asarray(multihost_utils.process_allgather(leaf, tiled=True))
                for leaf in jax.tree.leaves(deep)]
    del held
    alone = root / f"alone{rank}"
    PretrainedDecoder.from_model(model, variables).with_dex(extension, key=0).save(alone)
    written, own = tensors(destination), tensors(alone)
    return {"processes": jax.process_count(), "rank": rank,
            "differing": {name: [written.get(name), own.get(name)] for name in sorted({*written, *own})
                          if written.get(name) != own.get(name)} if rank == 0 else {},
            "spanning": sum(not leaf.is_fully_addressable for leaf in leaves),
            "refused_outside": refused, "release": digests(destination), "alone": digests(alone),
            "small_files": sorted(path.name for path in destination.iterdir()
                                  if not path.name.endswith((".safetensors", ".safetensors.index.json"))),
            "exported": growth[0], "held": growth[1],
            "whole": sum(leaf.nbytes for leaf in jax.tree.leaves(deep))}


def scoped_growth(modules) -> list[dict]:
    """Each `reading_home` scope the export opens (`dew.interop.saved_sources`), as it closes: how far the
    host's allocated bytes rose inside it (`allocated_growth`) and whether its writer wrote weights."""
    from dew import _coordination

    scopes: list[dict] = []
    opened = modules.reading_home

    @contextlib.contextmanager
    def measured(phase, **options):
        growth: list[int] = []
        try:
            with allocated_growth(growth), opened(phase, **options):
                writes = _coordination.writes_weights()
                yield
        finally:
            scopes.append({"growth": growth[0] if growth else None, "writes": writes})

    modules.reading_home = measured
    return scopes


def released(args) -> dict:
    """A saved run's release read through `released_run` over a cache the pool shares, which writes it
    once, and its export through `dew export` to a destination the pool shares; the release written in
    shards (`RELEASE_SHARD`), after two exports of the run outside the cache, so the measured one finds its
    programs compiled, its modules imported and their executions' native storage taken (XLA CPU's buffers,
    which the first run of a compiled program allocates and the measure would otherwise count), and the
    same measure over the release held home."""
    import jax
    from safetensors.numpy import load_file

    from dew._cli.main import main
    from dew.interop import saved_sources
    from dew.interop.releases import ReleasePlan

    ReleasePlan.write = functools.partialmethod(ReleasePlan.write, max_shard_size=RELEASE_SHARD)
    rank = jax.process_index()
    scopes = scoped_growth(saved_sources)
    # The first export compiles; the second warms the already-compiled execution's native storage.
    # Each restore and export ends before the measured restore, with neither warm tree held into it.
    for attempt in range(2):
        warm = Path(args.run_dir).parent / f"warm{attempt}-{rank}"
        saved_sources.export_run(saved_sources.pinned_run(args.run_dir), warm, publishes=rank == 0)
    scopes.clear()
    release = saved_sources.released_run(args.run_dir)
    read = list(scopes)
    scopes.clear()
    held: list[int] = []
    with allocated_growth(held):
        tensors = [load_file(path) for path in sorted(Path(release).glob("*.safetensors"))]
    del tensors
    destination = Path(args.run_dir).parent / "exported"
    code = main(["export", args.run_dir, str(destination)])
    return {"processes": jax.process_count(), "rank": rank, "release": str(release),
            "export_code": code,
            "release_files": digests(Path(release)), "read": read, "held": held[0], "exported": list(scopes),
            "export_files": digests(destination)}


MODES = {"exported": exported, "released": released}


def main(argv=None) -> None:
    parser = worker_parser(MODES, __doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--scan", action="store_true", help="the decoder scans its runs of like layers")
    serve_mode(MODES, parser.parse_args(argv))


if __name__ == "__main__":
    main()
