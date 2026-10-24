"""One member of a `dew launch` pool training through a RolloutScheduler.

tests/test_distribution.py starts it through the launcher, so the process
joins exactly as a multi-node run does (`prepare_process`). Each process
schedules the rollouts of the task rows its share of the batch holds, from a
source that answers every sample of a task with a completion the task fixes;
with `--vary`, the answer also depends on the process, as independent draws
from engines do. With `--records`, the model is a mixture and every call
records its routing and its sampler's support, and the second half of the
tasks truncate, so the share that holds them packs no record at all. The
pool fits one update on `--mesh`, and process 0 writes the rows each process
handed the step and the parameters the pool ended with. The same script on
one process is the reference.
"""

import argparse
import itertools
import json
from concurrent.futures import Future
from pathlib import Path

import numpy as np

TINY_SHARD = 256
TASKS = 8
PACKED = ("input_ids", "response_mask", "old_log_probs", "behavior_log_probs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mesh", required=True, help="MeshSpec fields as JSON")
    parser.add_argument("--vary", action="store_true", help="each process's draws depend on the process")
    parser.add_argument("--records", action="store_true",
                        help="calls record routing and supports, and the second half of the tasks truncate")
    args = parser.parse_args()

    from dew.training.runtime import prepare_process

    prepare_process()

    import jax
    import jax.numpy as jnp
    import optax
    from jax.experimental import multihost_utils

    from dew.data.dataset import DataPartition, Loader
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.backbones.decoder_block import Mixture
    from dew.objectives.rl import GRPOObjective
    from dew.objectives.rl.scheduler import Asynchronous, RolloutScheduler
    from dew.objectives.rl.sessions import Call, Session, Status
    from dew.training import Layout, MeshSpec, Trainer

    process = jax.process_index()
    mixture = Mixture(experts=4, top_k=2, expert_features=16) if args.records else None
    model = CausalTransformer(vocab_size=13, emb_features=16, num_layers=1, num_heads=2, head_dim=8,
                              mlp_features=32, max_seq_len=12, dtype="float32", attention_impl="xla",
                              mixture=mixture)
    params = model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))

    class Scripted:
        def submit(self, task, samples, *, version, key):
            futures = []
            for sample in range(samples):
                draw = int(task.id) * 10 + sample + (process if args.vary else 0)
                sampled = (3 + draw % 4, 4 + sample)
                recorded = {"routed_experts": np.asarray([[[draw % 4, (draw + 1) % 4]]] * 3, np.uint8),
                            "support": tuple((token, 11) for token in sampled)} if args.records else {}
                call = Call((1 + int(task.id) % 7, 2 + int(task.id) % 5), sampled,
                            (-0.5 - 0.01 * sample, -0.25), "stop", version, **recorded)
                status = (Status.TRUNCATED if args.records and int(task.id) > TASKS // 2
                          else Status.COMPLETED)
                future = Future()
                future.set_result(Session(str(task.id), "", 0, 0, (call,), status, float(sample), {}, ""))
                futures.append(future)
            return futures

        def cancel(self, futures):
            for future in futures:
                future.cancel()

    class Publisher:
        version = 0

        def load(self, variables, version):
            self.version = version

    def share(partition):
        rows = partition.rows(TASKS)
        mine = np.arange(1, TASKS + 1, dtype=np.int32)[partition.index * rows:(partition.index + 1) * rows]
        return itertools.repeat({"task_id": mine})

    trained = []

    def rollout(state, batch, key):
        packed = scheduler(state, batch, key)
        trained.append(packed)
        return packed

    objective = GRPOObjective(model, seq_len=7, variables=params, behavior_importance=2.0)
    trainer = Trainer(
        objective,
        optax.sgd(0.1),
        key=jax.random.key(5),
        mesh=MeshSpec(**json.loads(args.mesh)),
        layout=Layout(min_shard=TINY_SHARD),
        rollout=rollout,
    )
    partition = DataPartition.of(trainer.device_mesh)
    # Two samples a task, two chains of four ids a row of eight.
    scheduler = RolloutScheduler(
        Asynchronous(Scripted(), Publisher(), objective, width=8, rows=partition.rows(TASKS), max_lag=1,
                     ahead=1),
        groups=2, support_capacity=16 if args.records else None)
    data = scheduler.tasks(Loader(train=share, val=None, records=TASKS, batch=TASKS))
    final = trainer.fit(data, steps=1, log_every=1, checkpoint_every=None)
    rows = multihost_utils.process_allgather({name: trained[0][name] for name in PACKED})
    leaves = {jax.tree_util.keystr(path): np.asarray(multihost_utils.process_allgather(leaf, tiled=True))
              for path, leaf in jax.tree_util.tree_flatten_with_path(final.variables["params"])[0]}
    if process == 0:
        np.savez(args.out.with_suffix(".npz"), **leaves)
        args.out.write_text(json.dumps({
            "processes": jax.process_count(), "step": int(final.step),
            "partition": {"count": partition.count, "readers": partition.readers},
            "rows": {name: np.asarray(rows[name]).tolist() for name in PACKED}}))


if __name__ == "__main__":
    main()
