"""One process of a real pool writing a model whose weights span the pool through a run's export
(`dew.interop.saved_sources.export_run`).

Driven by tests/test_saved_sources.py. Every process places one small decoder over a mesh across the
pool, extends each of its layers' attention with DEX (`dew.nn.dex`), whose export folds each output
projection as it is written, and exports it as a saved run's release, the run's restore
(`Pretrained.from_saved`) standing in for the placed decoder and the release cut into small shards,
process 0 publishing and the other reading every tensor as it does (`publishes`). Each records the
files it wrote, the peak of its host allocations during the export (`tracemalloc`, which counts
NumPy's) and the model's size, and the same of the decoder at twice each width (`WIDER`), each export
measured after one of its own, so both peaks hold the same fixed cost; outside the export's scope the
write is refused. It also writes the same weights unplaced, as one process does. Every mode records
what it saw into --out and asserts nothing.

dew is imported inside the modes, not at module scope, because a JAX backend opened before
jax.distributed.initialize() would pin the process to its own devices.
"""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path
from types import SimpleNamespace

from flax import linen as nn
from process_support import serve_mode, worker_parser

FIELDS = {"vocab_size": 512, "emb_features": 64, "num_layers": 8, "num_heads": 4, "num_kv_heads": 2,
          "mlp_features": 128, "max_seq_len": 8}
"""A decoder of about 1.3 MB in float32, its largest tensor (the embedding) 128 KiB."""

WIDER = {**FIELDS, "vocab_size": 1024, "emb_features": 128, "mlp_features": 256}
"""The decoder at twice each width, about 5.2 MB: the same tensors by name, each 2 to 4 times as large,
so its export holds the same names and Python objects as `FIELDS`' and only its arrays grow."""

TINY_SHARD = 4
"""Far below the production threshold, so the pool's mesh shards every kernel."""

SHARD = "64KB"
"""The release's shard size, so it is written in a score of shards."""


def digests(directory: Path) -> dict[str, str]:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.glob("*.safetensors"))}


def tensors(directory: Path) -> dict[str, tuple[str, list[int], str]]:
    """Each tensor a directory's weight files hold, by name: its dtype, shape and bytes' digest."""
    from safetensors.numpy import load_file

    return {name: (str(array.dtype), list(array.shape), hashlib.sha256(array.tobytes()).hexdigest())
            for path in sorted(directory.glob("*.safetensors")) for name, array in load_file(path).items()}


def exported(args) -> dict:
    """The decoder exported placed across the pool, and written unplaced and outside the scope."""
    import tracemalloc

    import jax
    import jax.numpy as jnp

    from dew.inference.banks import HeldBanks
    from dew.interop.pretrained import Pretrained, PretrainedDecoder
    from dew.interop.saved_sources import export_run
    from dew.nn.dex import DifferentialExtension
    from dew.registry import models
    from dew.training import Layout, MeshSpec

    PretrainedDecoder.save = functools.partialmethod(PretrainedDecoder.save, max_shard_size=SHARD)
    extension = DifferentialExtension(tuple((layer, (0, 2)) for layer in range(FIELDS["num_layers"])),
                                      anneal_steps=10)

    def built(fields):
        """The decoder of `fields`, its variables, the variables placed across the pool, and its bundle."""
        model = models.build("causal_transformer", **fields)
        variables = nn.unbox(model.init(jax.random.key(0), jnp.ones((1, 4), jnp.int32)))
        placed = HeldBanks(variables).place(model, mesh=MeshSpec(fsdp=args.fsdp_size),
                                            layout=Layout(min_shard=TINY_SHARD, tolerance=1.0))
        bundle = PretrainedDecoder.from_model(model, placed).with_dex(extension, key=0)
        return model, variables, placed, bundle

    model, variables, placed, bundle = built(FIELDS)
    *_, wide, wider = built(WIDER)
    root = Path(args.run_dir)
    rank = jax.process_index()
    try:
        bundle.save(root / f"outside{rank}")
        refused = False
    except RuntimeError:
        refused = True
    run = SimpleNamespace(objective=SimpleNamespace(saved_task=None))

    def peak_of(exported, name: str) -> int:
        """The peak of the host's allocations while the pool exports `exported` to `name`."""
        Pretrained.from_saved = classmethod(lambda cls, run, **options: exported)
        tracemalloc.start()
        export_run(run, root / name, publishes=rank == 0)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return peak

    # Each width exported once before it is measured, a decoder built apart so nothing the measured bundle
    # holds is left from it: both measured exports find their programs compiled and their modules
    # imported, and what one holds beyond the other is what the wider arrays add.
    for index, fields in enumerate((FIELDS, WIDER)):
        peak_of(built(fields)[3], f"warm{index}_{rank}")
    destination = root / f"release{rank}"
    peak, wide_peak = peak_of(bundle, f"release{rank}"), peak_of(wider, f"wide{rank}")
    leaves, wide_leaves = jax.tree.leaves(placed), jax.tree.leaves(wide)
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
            "peak": peak, "whole": sum(leaf.nbytes for leaf in leaves),
            "largest_leaf": max(leaf.nbytes for leaf in leaves), "wide_peak": wide_peak,
            "wide_whole": sum(leaf.nbytes for leaf in wide_leaves),
            "wide_largest_leaf": max(leaf.nbytes for leaf in wide_leaves)}


def scoped_peaks(modules) -> list[dict]:
    """Each `reading_home` scope the export opens (`dew.interop.saved_sources`), as it closes: the peak
    of the host's allocations inside it (`tracemalloc`) and whether its writer wrote weights."""
    import contextlib
    import tracemalloc

    from dew import _coordination

    scopes: list[dict] = []
    opened = modules.reading_home

    @contextlib.contextmanager
    def measured(phase, **options):
        tracemalloc.start()
        try:
            with opened(phase, **options):
                writes = _coordination.writes_weights()
                yield
        finally:
            scopes.append({"peak": tracemalloc.get_traced_memory()[1], "writes": writes})
            tracemalloc.stop()

    modules.reading_home = measured
    return scopes


def released(args) -> dict:
    """A saved run's release read through `released_run` over a cache the pool shares, which writes it
    once, and its export through `dew export` to a destination the pool shares."""
    import jax

    from dew.cli.main import main
    from dew.interop import saved_sources

    scopes = scoped_peaks(saved_sources)
    release = saved_sources.released_run(args.run_dir)
    read = list(scopes)
    scopes.clear()
    destination = Path(args.run_dir).parent / "exported"
    code = main(["export", args.run_dir, str(destination)])
    return {"processes": jax.process_count(), "rank": jax.process_index(), "release": str(release),
            "export_code": code,
            "release_files": digests(Path(release)), "read": read, "exported": list(scopes),
            "export_files": digests(destination)}


MODES = {"exported": exported, "released": released}


def main(argv=None) -> None:
    parser = worker_parser(MODES, __doc__)
    parser.add_argument("--run-dir", required=True)
    serve_mode(MODES, parser.parse_args(argv))


if __name__ == "__main__":
    main()
