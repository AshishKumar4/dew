"""One process of a pool serving text through one `Server` over a mesh that spans the pool, driven by
tests/test_serving.py's test_a_server_over_a_pool_draws_what_each_request_draws_alone. It records what it
drew into --out and asserts nothing, so the invariants stay in the test that reads the file back.
"""

from __future__ import annotations

from process_support import serve_mode, worker_parser

PROMPTS = ("12", "5", "1234567", "98", "321", "4444", "70", "123")
"""Every process's prompts: each process serves its share, a request of every width among them."""
BUDGETS = (5, 9, 3, 12, 7, 4, 6, 2)


def mode_pooled_server(args) -> dict:
    """The process's share of `PROMPTS`, served through one server over the pool's mesh, each draw keyed by
    its prompt's place in the list; the pool's processes step it together (`Server.run`)."""
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    from model_support import serving_task

    from dew.inference.loading import place
    from dew.inference.serving import Server
    from dew.interop.decoder_parts import decoder_axes
    from dew.nn.sharding import boxed
    from dew.training import Layout, MeshSpec

    rank, processes = jax.process_index(), jax.process_count()
    bound = serving_task()
    bound = bound.bind(nn.unbox(bound.model.init(jax.random.key(3), jnp.ones((1, 2), jnp.int32))))
    if args.fsdp_size > 1:
        with boxed(decoder_axes(bound.model)):
            bound = bound.bind(place(bound.variables, MeshSpec(fsdp=args.fsdp_size),
                                     Layout(min_shard=1, tolerance=1.0)))
    server = Server.from_task(bound, slots=4 * args.fsdp_size, capacity=128, admission=2 * args.fsdp_size)
    share = len(PROMPTS) // processes
    mine = range(rank * share, (rank + 1) * share)
    tickets = [server.submit(PROMPTS[index], BUDGETS[index], key=index) for index in mine]
    server.run()
    drawn = [ticket.result().host() for ticket in tickets]
    return {"tokens": [generation.tokens[0].tolist() for generation in drawn],
            "lengths": [int(generation.lengths[0]) for generation in drawn]}


MODES = {"pooled_server": mode_pooled_server}


if __name__ == "__main__":
    serve_mode(MODES, worker_parser(MODES, __doc__).parse_args())
