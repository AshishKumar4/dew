"""Workers for a pool's run on the examples clock (`dew.training._fit.pooled_read`): each process a rank
of a run whose shares run out at steps apart.

dew is imported inside the modes, as `multiprocess_worker`'s are, so the pool starts before JAX does.
"""

from __future__ import annotations

import numpy as np
from multiprocess_worker import BATCH, LossRecorder
from process_support import serve_mode, worker_parser


def examples_clock(args) -> dict:
    """A run of `--records` source examples on the examples clock, each of the first half's even ids
    expanded to three rows and every other example to one, so the pool's shares hold other counts of
    rows and run out at steps apart: the step each process ends at, the examples its clock and its
    state consumed, the examples its logs read at each step, and its learning rate's last value."""
    import jax
    from affine_run import FEATURES

    from dew.data import Memory
    from dew.data.pipeline import ObjectiveData, Pipeline
    from dew.objectives.base import Aux, Objective
    from dew.optim import Linear, OptimConfig
    from dew.training import MeshSpec, Trainer

    half = args.records // 2

    class Expanded:
        def flat_map(self, element):
            copies = 3 if int(element["id"]) < half and int(element["id"]) % 2 == 0 else 1
            return tuple({"x": element["x"]} for _ in range(copies))

    class Pulled(Objective):
        def fresh_variables(self, key, held):
            return {"params": {"w": jax.numpy.zeros(FEATURES)}}

        def loss(self, variables, batch, step):
            return self.row_mean(jax.numpy.square(variables["params"]["w"] - batch["x"]), batch), Aux({})

        def data(self):
            return ObjectiveData(train=(Expanded(),))

    class Logged(LossRecorder):
        def __init__(self):
            super().__init__()
            self.scalars: list[dict] = []

        def log(self, scalars, step):
            super().log(scalars, step)
            if "train/loss" in scalars:
                self.scalars.append(dict(scalars))

    ids = np.arange(args.records, dtype=np.int32)
    x = np.repeat(ids[:, None].astype(np.float32), FEATURES, axis=1)
    # A global order, which the pool's processes read disjoint shares of.
    chain = Pipeline(Memory({"x": x, "id": ids})).shuffle(0).read("memory").batch(BATCH)
    from dew.batch import EXAMPLES_CONSUMED
    from dew.data.compile import fitted
    from dew.data.dataset import DataPartition

    # The examples this process's share credits and the rows it holds, read apart before the run.
    loader, _ = fitted(chain, None, objective=Pulled().data(), examples=args.records)
    credited = rows = 0
    for batch in loader.train(DataPartition.of(MeshSpec().build())):
        credited += int(np.sum(batch[EXAMPLES_CONSUMED]))
        rows += len(batch[EXAMPLES_CONSUMED])
    tracker = Logged()
    optim = OptimConfig(optimizer="optax:sgd", learning_rate=Linear(peak_value=1e-2, end_value=0.0))
    trainer = Trainer(Pulled(), optim, key=jax.random.key(0), mesh=MeshSpec(), tracker=tracker)
    state = trainer.fit(chain, examples=args.records, log_every=1)
    return {"step": int(state.step), "consumed": int(state.consumed), "credited": credited, "rows": rows,
            "logged": [scalars["train/examples_consumed"] for scalars in tracker.scalars]}


MODES = {"examples_clock": examples_clock}


def main(argv=None) -> None:
    parser = worker_parser(MODES, __doc__)
    parser.add_argument("--records", type=int, default=24)
    serve_mode(MODES, parser.parse_args(argv))


if __name__ == "__main__":
    main()
