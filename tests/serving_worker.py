"""One process of a pool serving text through one `Server` over a mesh that spans the pool, driven by
tests/test_serving.py's test_a_server_over_a_pool_draws_what_each_request_draws_alone. It records what it
drew into --out and asserts nothing, so the invariants stay in the test that reads the file back.
"""

from __future__ import annotations

from process_support import serve_mode, worker_parser

PROMPTS = ("12", "5", "1234567", "98", "321", "4444", "70", "123412341234")
"""The pool's prompts, a request of every width among them."""
BUDGETS = (5, 9, 3, 2, 7, 4, 6, 12)


def mode_pooled_server(args) -> dict:
    """`PROMPTS` served through one server over the pool's mesh, each draw keyed by its prompt's place in
    the list, the pool stepping it together (`Server.run`), in two rounds: the last process's share
    alone, while the others submit nothing, then the rest, the last process's longest; one process serves
    every prompt in both rounds."""
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    from model_support import serving_task

    from dew.inference import Engine
    from dew.inference.loading import place
    from dew.inference.serving import Server
    from dew.interop.decoder_parts import decoder_axes
    from dew.nn.kv_cache import KVCache
    from dew.nn.sharding import boxed
    from dew.training import Layout, MeshSpec

    rank, processes = jax.process_index(), jax.process_count()
    bound = serving_task()
    bound = bound.bind(nn.unbox(bound.model.init(jax.random.key(3), jnp.ones((1, 2), jnp.int32))))
    if args.fsdp_size > 1:
        with boxed(decoder_axes(bound.model)):
            bound = bound.bind(place(bound.variables, MeshSpec(fsdp=args.fsdp_size),
                                     Layout(min_shard=1, tolerance=1.0)))
    paged = {"kv_cache": KVCache(page_size=4, pages=128), "prefix_cache": True} if args.paged else {}
    server = Server.from_task(bound, slots=4 * args.fsdp_size, capacity=128, admission=2 * args.fsdp_size,
                              **paged)
    last = range(len(PROMPTS) - 3, len(PROMPTS))
    rounds = [list(last), [index for index in range(len(PROMPTS)) if index not in last]]
    drawn = []
    for held in rounds:
        # The last process takes the first round alone, and the others share the second.
        mine = held if processes == 1 else (held if rank == processes - 1 else []) if held == rounds[0] else (
            held[rank::processes - 1] if rank < processes - 1 else [])
        tickets = [(index, server.submit(PROMPTS[index], BUDGETS[index], key=index)) for index in mine]
        server.run()
        drawn += [(index, ticket.result().host()) for index, ticket in tickets]
    try:
        Engine(server)
        engine = "takes it"
    except ValueError:
        engine = "refuses it"
    return {"engine": engine, "drawn": {str(index): {"tokens": generation.tokens[0].tolist(),
                                                     "length": int(generation.lengths[0])}
                                        for index, generation in drawn}}


MODES = {"pooled_server": mode_pooled_server}


if __name__ == "__main__":
    parser = worker_parser(MODES, __doc__)
    parser.add_argument("--paged", action="store_true")
    serve_mode(MODES, parser.parse_args())
