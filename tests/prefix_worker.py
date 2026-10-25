"""One process of a pool serving a prefix run, driven by tests/test_multiprocess.py's
test_a_pool_serves_a_prefix_run_as_one_process_and_refuses_a_short_prompt_together. It records what it
saw into --out and asserts nothing, so the invariants stay in the test that reads the file back.
"""

from __future__ import annotations

import numpy as np
from multiprocess_worker import TINY, continuation_prompts
from process_support import serve_mode, worker_parser


def mode_prefix_generation(args) -> dict:
    """A prefix run's task serving a request each process holds a share of, as one global array.

    The blocks each prompt reads are made of each process's own rows inside the request's agreed check,
    so a pool answers the rows one process answers, and a prompt too short for the prefix on one rank alone
    is refused by both ranks, neither waiting on a peer that left.
    """
    import jax
    import jax.numpy as jnp
    from block_support import model as block_model

    from dew.coordination import PeerFailure
    from dew.inference import TextGeneration
    from dew.nn.inputs import ModelInputs, mesh_of
    from dew.nn.mixers import AttentionMixer
    from dew.training import Layout, MeshSpec
    from dew.training.distributed import shard_batch

    rank, processes = jax.process_index(), jax.process_count()
    model = block_model(1, full_attention={"mixer": AttentionMixer(bidirectional_blocks=True)})
    params = model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    mesh = MeshSpec(fsdp=args.fsdp_size).build(jax.devices())
    placed = jax.device_put(params, Layout(min_shard=TINY).shardings(mesh, params))
    task = TextGeneration(model, placed, prefix=3)
    prompts, lengths = continuation_prompts()
    mask = np.arange(prompts.shape[1])[None, :] >= prompts.shape[1] - lengths[:, None]

    def share(chosen):
        rows = len(chosen) // processes
        return chosen[rank * rows:(rank + 1) * rows]

    # Rows 0-2 and 4 hold at least three real tokens: served as one global array, each process placing its
    # share. Rows 3 and 5 hold fewer, both in rank 1's half of the six, which each process hands in as its
    # own rows.
    served = share(np.asarray([0, 1, 2, 4]))
    placed_rows = shard_batch(mesh_of(placed), {"ids": prompts[served], "mask": mask[served]})
    result = task(ModelInputs(placed_rows["ids"], {"attention_mask": placed_rows["mask"]}), 3, key=7).host()
    own = share(np.arange(len(prompts)))
    try:
        task(ModelInputs(jnp.asarray(prompts[own]), {"attention_mask": jnp.asarray(mask[own])}), 3, key=7)
    # One process refuses the request itself; a pool's ranks each raise the refusal its agreed check names.
    except (ValueError, PeerFailure):
        refused = True
    else:
        refused = False
    return {"tokens": result.tokens.tolist(), "lengths": result.lengths.tolist(), "refused": refused}


MODES = {"prefix_generation": mode_prefix_generation}


if __name__ == "__main__":
    serve_mode(MODES, worker_parser(MODES, __doc__).parse_args())
