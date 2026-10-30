"""Workers for a pool's run on the examples clock (`dew.training._fit.pooled_read`): each process a rank
of a run bounded in examples whose shares run out at steps apart.

dew is imported inside the modes, as `multiprocess_worker`'s are, so the pool starts before JAX does.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from multiprocess_worker import BATCH, dump_params
from process_support import serve_mode, worker_parser

RECORDS = 10
"""The records each run reads, ids 1 to RECORDS, which its bound reaches past in a second pass."""

FEATURES = 3
SHARD_ROWS = 3
"""The rows of each physical shard a streamed read deals its shares, the last one short."""


class Questioned:
    """An objective's expansion of each record into `id % 3 + 1` rows, one a question, as a decision
    layout of a question a row makes of an example (`dew.decision.rows.QuestionRows`), its identity
    stable, so a resumed run knows it for the step it saved."""

    identity = "questioned@1"

    def flat_map(self, element):
        return tuple({"id": element["id"], "question": np.int32(question)}
                     for question in range(int(element["id"]) % 3 + 1))


def examples_clock(args) -> dict:
    """A run bounded to `--examples` of RECORDS records, read as `--deal` deals them to the pool's shares:
    `order`, one global order; `records`, each share its own, an objective making `id % 3 + 1` rows of
    each; `shards`, physical shards of SHARD_ROWS rows and a last short one, each share its own, the
    objective making rows likewise. Its mesh puts `--tensor` processes on each share; `--pause` pauses it
    there (`Pause`) and resumes it; `--cooldown` branches it to a cooldown from the step before the first
    share's stream ends to the step the last one's does, across the steps they end apart at. Reports the
    step it ends at, the examples its state consumed, the examples each log read, each id's examples its
    steps trained on (each step's real rows by id over the rows each example makes, the objective's aux,
    which the pool's first process logs), each id's examples the pool's shares read apart (every share's
    stream of the run's data read on this process, `dealt`), the batches each share's stream holds and
    the cooldown's validation loss, its parameters beside the report."""
    import jax
    import jax.numpy as jnp

    from dew.batch import VALID_ROWS
    from dew.data import Memory
    from dew.data.compile import fitted
    from dew.data.dataset import Closeable, DataPartition
    from dew.data.pipeline import ObjectiveData, Pipeline
    from dew.objectives.base import Aux, Objective
    from dew.optim import Linear, OptimConfig
    from dew.training import Checkpoints, Cooldowns, MeshSpec, Pause, Trainer

    expands = args.deal != "order"

    class Pulled(Objective):
        """A weight pulled toward each real row's id, counting each id's examples among the rows it trains."""

        def fresh_variables(self, key, held):
            return {"params": {"w": jnp.zeros(FEATURES)}}

        def loss(self, variables, batch, step):
            ids = batch["id"]
            x = jnp.repeat(ids.astype(jnp.float32)[:, None], FEATURES, axis=1)
            loss = self.row_mean(jnp.square(variables["params"]["w"] - x), batch)
            real = jnp.asarray(batch[VALID_ROWS], bool) if VALID_ROWS in batch else jnp.ones(ids.shape, bool)
            rows = ids % 3 + 1 if expands else jnp.ones_like(ids)
            return loss, Aux({f"id_{record}": jnp.sum(jnp.where(real & (ids == record), 1.0 / rows, 0.0))
                              for record in range(1, RECORDS + 1)})

        def data(self):
            return ObjectiveData(train=(Questioned(),)) if expands else ObjectiveData()

    class Logged:
        def __init__(self):
            self.scalars: list[dict] = []

        def log(self, scalars, step):
            self.scalars.append(dict(scalars))

        def artifact(self, value, step):
            pass

    ids = np.arange(1, RECORDS + 1, dtype=np.int32)
    source = Memory({"id": ids}, shard_rows=SHARD_ROWS)
    read = "streamed" if args.deal == "shards" else "memory"
    shuffled = {"seed": 3, "window": RECORDS} if read == "streamed" else {"seed": 3}
    chain = Pipeline(source).shuffle(**shuffled).read(read).batch(BATCH)
    held = Pipeline(Memory({"id": ids[:BATCH]})).read("memory").batch(BATCH)
    tracker = Logged()
    optim = OptimConfig(optimizer="optax:sgd", learning_rate=Linear(peak_value=1e-2, end_value=0.0))
    checkpoints = Checkpoints(str(args.out.parent / "run")) if args.pause else None
    trainer = Trainer(Pulled(), optim, key=jax.random.key(0), mesh=MeshSpec(tensor=args.tensor),
                      tracker=tracker, checkpoints=checkpoints)
    # The examples each share's stream deals it and the batches it reads, read as the run's loader reads
    # them, apart from the run.
    loader, _ = fitted(chain, None, objective=Pulled().data(), examples=args.examples)
    pool = DataPartition.of(trainer.device_mesh)
    dealt, batches = dict.fromkeys(range(1, RECORDS + 1), 0.0), []
    for index in range(pool.count):
        stream = loader.train(dataclasses.replace(pool, index=index))
        batches.append(0)
        try:
            for batch in stream:
                batches[-1] += 1
                real = np.asarray(batch.get(VALID_ROWS, np.ones(len(batch["id"]), bool)), bool)
                for record in np.asarray(batch["id"])[real].tolist():
                    dealt[record] += 1 / (record % 3 + 1 if expands else 1)
        finally:
            if isinstance(stream, Closeable):
                stream.close()
    options = {"examples": args.examples, "log_every": 1}
    if args.cooldown:
        # A branch from the step before the first share's stream ends to the step the last one's does.
        at, length = min(batches) - 1, max(batches) - min(batches) + 1
        options.update(validation=held, cooldowns=Cooldowns(at=(at,), length=length))
    if args.pause:
        trainer.fit(chain, stop=Pause(args.pause), **options)
    state = trainer.fit(chain, **options)
    dump_params(args.out.with_suffix(".npz"), state.variables)
    trained = [scalars for scalars in tracker.scalars if "train/loss" in scalars]
    counts = {record: sum(scalars[f"train/id_{record}"] for scalars in trained)
              for record in range(1, RECORDS + 1)}
    cooled = [scalars["cooldown_val/loss"] for scalars in tracker.scalars if "cooldown_val/loss" in scalars]
    # A run whose plan counts its steps ahead keeps no clock (`TrainState.consumed`).
    consumed = None if state.consumed is None else int(state.consumed)
    return {"step": int(state.step), "consumed": consumed, "counts": counts, "dealt": dealt,
            "batches": batches, "cooled": cooled,
            "logged": [scalars.get("train/examples_consumed") for scalars in trained]}


MODES = {"examples_clock": examples_clock}


def main(argv=None) -> None:
    parser = worker_parser(MODES, __doc__)
    parser.add_argument("--deal", choices=("order", "records", "shards"), default="order")
    parser.add_argument("--examples", type=int, default=RECORDS)
    parser.add_argument("--tensor", type=int, default=1)
    parser.add_argument("--pause", type=float, default=0.0)
    parser.add_argument("--cooldown", action="store_true")
    serve_mode(MODES, parser.parse_args(argv))


if __name__ == "__main__":
    main()
