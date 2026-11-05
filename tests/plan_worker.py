"""A small decoder's step planned on each process of a pool whose devices leave each its own room: what
the plan search compiles and plans on each (`dew.training.planning.agreed`)."""

import numpy as np
from process_support import serve_mode, worker_parser


def agreement(args) -> dict:
    """A small decoder's step planned over every process's device, each device leaving its own room
    (`Device.memory_stats`): the room its derivation says the step needs at four microbatches, read off
    a plan made there at a whole limit's room, and then the splits the search compiles and the plan it
    makes where process 0's device leaves that room to the byte and each other process's a byte less per
    rank."""
    import jax
    import optax

    from dew.distributed.batches import shard_batch
    from dew.objectives.lm import LMObjective
    from dew.registry import models
    from dew.training import Layout, MeshSpec, Trainer

    limit, rows, length = 2 ** 40, 16, 256

    def leaving(room: int) -> None:
        jax.Device.memory_stats = lambda device: {"bytes_limit": limit, "bytes_in_use": limit - room}

    def planned(split: int | None) -> dict:
        model = models.build("causal_transformer", vocab_size=64, emb_features=32, num_layers=2,
                             num_heads=2, mlp_features=64, max_seq_len=length)
        trainer = Trainer(LMObjective(model, length), optax.sgd(1e-2), key=jax.random.key(0),
                          mesh=MeshSpec(fsdp=args.processes), layout=Layout(min_shard=256), split=split)
        state, _, _ = trainer.place()
        share = rows // args.processes
        tokens = ((np.arange(rows * (length + 1)).reshape(rows, length + 1) * 7) % 64).astype(np.int32)
        trainer.compile(state, shard_batch(trainer.device_mesh, {
            "text": tokens[args.process_id * share:(args.process_id + 1) * share]}))
        return trainer.plan

    leaving(limit)
    (judged,) = [entry for entry in planned(4)["derived"] if entry["split"] == 4]
    need = limit - judged["headroom"]
    leaving(need - args.process_id)
    plan = planned(None)
    return {"need": need, "asked": [probe["split"] for probe in plan["probes"]], "split": plan["split"],
            "derived": {str(entry["split"]): entry["headroom"] for entry in plan["derived"]}}


if __name__ == "__main__":
    serve_mode({"agreement": agreement}, worker_parser({"agreement": agreement}, __doc__).parse_args())
