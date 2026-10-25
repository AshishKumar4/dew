"""One process of a pool training a hybrid decoder split over the sequence.

Driven by tests/test_multiprocess.py's test_a_pool_trains_a_hybrid_split_over_the_sequence. Every mode
records what it saw into --out and asserts nothing, so the invariants stay in the test that reads the
files back.
"""

from __future__ import annotations

import numpy as np
from multiprocess_worker import BATCH, MIXED_RATE, TINY, VOCAB, mixed_validity_step
from process_support import serve_mode, worker_parser


def mode_sequence_hybrid(args) -> dict:
    """One training step of a small hybrid decoder, Mamba-2 and attention layers, split two ways over
    the sequence across the pool, one device a process, so each half of every sequence lies on another
    process. No producer pads, so the pool fills validity all true on every process
    (`filled_validity`), which Mamba-2 under a sequence axis once refused. Each process reads the
    share its mesh gives it (`DataPartition.of`), and reports the loss, its largest update and the
    processes along the mesh's sequence axis."""
    import jax
    import jax.numpy as jnp
    import optax

    from dew.data.dataset import DataPartition
    from dew.nn.inputs import ModelInputs
    from dew.objectives.lm import LMObjective
    from dew.registry import models
    from dew.training import Layout, MeshSpec, Trainer

    length = 16
    mamba = {"mixer": {"class": "mamba2", "fields": {"num_heads": 4, "head_dim": 16, "state_size": 8,
                                                     "n_groups": 1, "chunk_size": 8}}}
    model = models.build("causal_transformer", vocab_size=VOCAB, emb_features=32, num_layers=4, num_heads=4,
                         num_kv_heads=2, mlp_features=64, max_seq_len=length,
                         layer_types=("mamba", "attention") * 2, kinds={"mamba": mamba, "attention": {}})
    trainer = Trainer(LMObjective(model, length), optax.sgd(MIXED_RATE), key=jax.random.key(0),
                      mesh=MeshSpec(fsdp=args.fsdp_size, sequence=2), layout=Layout(min_shard=TINY),
                      checkpoints=None, tracker=None)
    tokens = np.random.default_rng(0).integers(0, VOCAB, size=(BATCH, length + 1)).astype(np.int32)
    share = DataPartition.of(trainer.device_mesh)
    mine = tokens[share.index::share.count]
    trained = mixed_validity_step(trainer, {"text": ModelInputs(jnp.asarray(mine))})
    mesh = trainer.device_mesh
    along = np.moveaxis(np.asarray(mesh.devices), mesh.axis_names.index("sequence"), -1)
    return {"loss": trained["loss"], "placed_fields": trained["placed_fields"],
            "update_scale": max(float(np.max(np.abs(leaf))) for leaf in trained["update"].values()),
            "sequence_processes": [sorted({device.process_index for device in group})
                                   for group in along.reshape(-1, along.shape[-1])]}


MODES = {"sequence_hybrid": mode_sequence_hybrid}


def main(argv=None) -> None:
    serve_mode(MODES, worker_parser(MODES, __doc__).parse_args(argv))


if __name__ == "__main__":
    main()
