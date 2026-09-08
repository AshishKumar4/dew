"""One process of a real pool reading a host-resident parameter bank.

Driven by tests/test_host_placement.py. The pool places one tiny scanned
decoder twice, resident and with its layer banks in pinned host memory, on
the same sharded mesh, and records the logits, the greedy continuation, the
shard each process actually holds and a checkpoint round trip. Every mode
records what it saw into --out and asserts nothing, so the invariants stay in
the test that reads the files back.

dew is imported inside the modes, not at module scope, because a JAX backend
opened before jax.distributed.initialize() would pin the process to its own
devices.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

VOCAB = 32
SEQ_LEN = 8
PROMPT = 5
NEW_TOKENS = 3
FIELDS = dict(vocab_size=VOCAB, emb_features=16, num_layers=4, num_heads=4,
              num_kv_heads=2, mlp_features=32, max_seq_len=SEQ_LEN)
# Far below the production threshold, so a two-process mesh shards anything.
TINY_SHARD = 4


def layouts(tensor: int):
    """The resident and host-resident layouts of one topology.

    The production rules, with the feed-forward width redirected onto the
    tensor axis where the mesh has one, so a run's banks are split over both
    parameter axes and the memory kind is the only thing the two layouts
    disagree about.
    """
    from dew.training import Layout
    from dew.training.distributed import DEFAULT_RULES

    rules = DEFAULT_RULES if tensor == 1 else (
        ("mlp", "tensor"),) + tuple(rule for rule in DEFAULT_RULES if rule[0] != "mlp")
    return (Layout(rules=rules, min_shard=TINY_SHARD, tolerance=1.0),
            Layout(rules=rules, min_shard=TINY_SHARD, tolerance=1.0,
                   host_parameters=("params/layers_*",)))


def local(value):
    """This process's own shards of a global array, in shard order.

    Two placements of one model on one topology are compared shard for
    shard, which needs no collective and says which process disagreed.
    """
    import numpy as np
    return np.concatenate(
        [np.asarray(shard.data).ravel() for shard in value.addressable_shards])


def report(tree) -> dict:
    """Where each leaf of a store sits: its memory kind, its global shape and
    the shard this process holds of it."""
    import jax
    leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {jax.tree_util.keystr(path): {
        "kind": str(leaf.sharding.memory_kind),
        "spec": str(leaf.sharding.spec),
        "shape": list(leaf.shape),
        "shard": list(leaf.sharding.shard_shape(leaf.shape)),
        "addressable": len(leaf.addressable_shards),
    } for path, leaf in leaves}


def banked(args) -> dict:
    """Resident and host-resident banks of one model on one sharded mesh."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from dew.inference.banks import HeldBanks, host_banked
    from dew.registry import models
    from dew.sampling.text import Sampling, generate
    from dew.training import MeshSpec

    mesh = MeshSpec(fsdp=args.fsdp_size, tensor=args.tensor_size)
    resident_layout, host_layout = layouts(args.tensor_size)
    plain = models.build("causal_transformer", **FIELDS)
    scanned = models.build("causal_transformer", **FIELDS, scan_layers=True,
                           bank_layers=args.bank_layers)
    tokens = jnp.asarray(
        np.random.default_rng(0).integers(1, VOCAB, size=(jax.device_count(), PROMPT)),
        jnp.int32)
    variables = plain.init(jax.random.key(0), tokens)

    resident = host_banked(scanned, HeldBanks(variables), mesh=mesh, layout=resident_layout)
    on_host = host_banked(scanned, HeldBanks(variables), mesh=mesh, layout=host_layout)
    resident_logits = local(scanned.apply(resident, tokens))
    host_logits = local(scanned.apply(on_host, tokens))
    sampling = Sampling(temperature=0.0)
    resident_tokens = generate(scanned, resident, tokens, NEW_TOKENS, seed=0, sampling=sampling)
    host_tokens = generate(scanned, on_host, tokens, NEW_TOKENS, seed=0, sampling=sampling)

    record = {
        "processes": jax.process_count(),
        "process": jax.process_index(),
        "devices": jax.device_count(),
        "local_devices": jax.local_device_count(),
        "banks": sorted(on_host["params"]),
        "resident_placement": report(resident),
        "host_placement": report(on_host),
        "logits_equal": bool(np.array_equal(resident_logits, host_logits)),
        "logits_difference": float(np.max(np.abs(resident_logits - host_logits))),
        "tokens_equal": bool(np.array_equal(local(resident_tokens.tokens),
                                            local(host_tokens.tokens))),
        "host_tokens": local(host_tokens.tokens).tolist(),
    }
    if args.run_dir:
        record.update(_checkpointed(args, scanned, plain, tokens, mesh,
                                   resident_layout, host_layout, host_logits))
    return record


def _checkpointed(args, scanned, plain, tokens, mesh, resident_layout, host_layout,
                  host_logits) -> dict:
    """One step of a real run, saved by the pool and read back bank by bank."""
    import jax
    import numpy as np
    import optax
    from dew.checkpoints import Checkpoints
    from dew.inference.banks import CheckpointBanks, host_banked
    from dew.objectives.lm import LMObjective
    from dew.training import Trainer

    directory = str(Path(args.run_dir))
    checkpoints = Checkpoints(directory, keep=1)
    trainer = Trainer(LMObjective(plain, SEQ_LEN - 1, head_chunks=1), optax.sgd(0.1),
                      key=jax.random.PRNGKey(0), mesh=mesh, layout=resident_layout,
                      checkpoints=checkpoints)
    state, _, _ = trainer.place()
    batch = {"text": np.tile(np.arange(1, SEQ_LEN + 1, dtype=np.int32)[None],
                             (jax.device_count(), 1))}
    from dew.training.distributed import shard_batch
    placed = shard_batch(trainer.device_mesh, batch)
    state = trainer.compile(state, placed)(state, placed)[0]
    checkpoints.save(int(state.step), state, None, metrics={"loss": 1.0})
    checkpoints.wait()

    resident = host_banked(scanned, CheckpointBanks(directory), mesh=mesh,
                           layout=resident_layout)
    on_host = host_banked(scanned, CheckpointBanks(directory), mesh=mesh, layout=host_layout)
    from_resident = local(scanned.apply(resident, tokens))
    from_host = local(scanned.apply(on_host, tokens))
    return {
        "restored_equal": bool(np.array_equal(from_resident, from_host)),
        "restored_placement": report(on_host),
        "restored_is_trained": not bool(np.array_equal(from_host, host_logits)),
        "restored_difference": float(np.max(np.abs(from_resident - from_host))),
        "restored_logits": from_host[:4].tolist(),
    }


MODES = {"banked": banked}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=sorted(MODES))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--coordinator", help="host:port of the jax.distributed service")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--process-id", type=int, default=0)
    parser.add_argument("--fsdp-size", type=int, default=1)
    parser.add_argument("--tensor-size", type=int, default=1)
    parser.add_argument("--bank-layers", type=int)
    parser.add_argument("--run-dir")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.coordinator:
        os.environ.update({
            "OMPI_MCA_orte_hnp_uri": f"0.0;tcp://{args.coordinator}",
            "OMPI_COMM_WORLD_SIZE": str(args.processes),
            "OMPI_COMM_WORLD_RANK": str(args.process_id),
            "OMPI_COMM_WORLD_LOCAL_RANK": str(args.process_id),
            "JAX_COORDINATOR_ADDRESS": args.coordinator,
        })
    from dew.training.runtime import prepare_process

    prepare_process(multi_host=bool(args.coordinator))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(MODES[args.mode](args)))


if __name__ == "__main__":
    main()
