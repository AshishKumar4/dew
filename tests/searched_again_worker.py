"""The workers of a pool whose plan, made before its state was placed, one process's placed devices hold the
room to search again and the other's do not (tests/test_multiprocess.py).

dew is imported inside the mode, as `multiprocess_worker`'s are, so the pool starts before JAX does.
"""

from __future__ import annotations

import numpy as np
from process_support import serve_mode, worker_parser


def searched_again(args) -> dict:
    """A fresh fit whose plan, made before its state was placed, one process's placed devices hold the
    room to search again and the other's do not (process 0 the room for the batch whole once placed,
    process 1 short of it by more than the state it places): its needs measured first, with room past
    all, then the fit's one step, each process reporting the plan it stepped and whether it searched
    again. Every process searches again or none does, the search's coordination rounds paired across
    them."""
    import logging

    import jax
    from test_plan import BATCH, LIMIT, ROWS, lm, trained

    from dew.data.dataset import Loader
    from dew.training import MeshSpec

    def stated(room: int) -> None:
        jax.Device.memory_stats = lambda device: {"bytes_limit": LIMIT, "bytes_in_use": LIMIT - room}

    def pooled(trainer):
        trainer.device_mesh = MeshSpec().build()
        return trainer

    def need(trainer) -> int:
        stated(LIMIT)
        trained(trainer, steps=0)
        return LIMIT - trainer.plan["probes"][0]["headroom"]

    whole, halves = need(pooled(lm(split=1))), need(pooled(lm(split=2)))
    state, _, _ = pooled(lm()).place()
    held = sum(leaf.addressable_shards[0].data.nbytes for leaf in jax.tree.leaves(state)
               if isinstance(leaf, jax.Array))
    delta = (whole - halves - held) // 2
    stated([whole + held, whole - delta][jax.process_index()])
    said: list[str] = []
    heard = logging.Handler()
    heard.emit = lambda record: said.append(record.getMessage())
    logging.getLogger("dew.training.compiler").addHandler(heard)
    rows = ROWS // jax.process_count()
    local = np.asarray(BATCH["text"])[jax.process_index() * rows:(jax.process_index() + 1) * rows]
    trainer = pooled(lm())
    trainer.fit(Loader(lambda partition: iter(lambda: {"text": local}, None), val=None, batch=ROWS,
                       records=ROWS), steps=1)
    return {"whole": whole, "halves": halves, "held": held, "delta": delta,
            "plan": [trainer.plan["split"], trainer.plan["remat"]],
            "searched_again": any("searched again there" in message for message in said)}


MODES = {"searched_again": searched_again}


def main(argv=None) -> None:
    serve_mode(MODES, worker_parser(MODES, __doc__).parse_args(argv))


if __name__ == "__main__":
    main()
