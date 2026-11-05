"""One process of a pool serving text through one `Server` over a mesh that spans the pool, driven by
tests/test_serving.py's test_a_server_over_a_pool_draws_what_each_request_draws_alone. It records what it
drew into --out and asserts nothing, so the invariants stay in the test that reads the file back.
"""

from __future__ import annotations

from process_support import serve_mode, worker_parser

PROMPTS = ("12", "5", "1234567", "98", "321", "4444", "70", "123412341234", "12341234", "1234123412")
"""The pool's prompts, a request of every width among them, the last three sharing pages of their prefix."""
BUDGETS = (5, 9, 3, 2, 7, 4, 6, 12, 5, 3)


def mode_pooled_server(args) -> dict:
    """`PROMPTS` served through one server over the pool's mesh, each draw keyed by its prompt's place in
    the list, the pool stepping it together (`Server.run`), in two rounds: three requests on the last
    process alone, while the others submit nothing; then every process's share at once, the last
    process's widest and sharing a prefix with its first round's. With `paged` the model is a hybrid
    stack, whose rows keep recurrent state beside their pages, over a cache that shares prompt pages and
    the states at them. One process serves every prompt in both rounds."""
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    from model_support import serving_task
    from test_serving import hybrid_task

    from dew.inference import Engine
    from dew.inference.loading import place
    from dew.inference.serving import Server
    from dew.interop.decoder_parts import decoder_axes
    from dew.nn.kv_cache import KVCache
    from dew.nn.sharding import boxed
    from dew.training import Layout, MeshSpec

    rank, processes = jax.process_index(), jax.process_count()
    if args.paged:
        bound = hybrid_task()
    else:
        bound = serving_task()
        bound = bound.bind(nn.unbox(bound.model.init(jax.random.key(3), jnp.ones((1, 2), jnp.int32))))
    if args.fsdp_size > 1:
        with boxed(decoder_axes(bound.model)):
            bound = bound.bind(place(bound.variables, MeshSpec(fsdp=args.fsdp_size),
                                     Layout(min_shard=1, tolerance=1.0)))
    paged = {"kv_cache": KVCache(page_size=4, pages=128), "prefix_cache": True} if args.paged else {}
    server = Server.from_task(bound, slots=4 * args.fsdp_size, capacity=128, admission=2 * args.fsdp_size,
                              **paged)
    first = [8, 1, 4]
    second = [index for index in range(len(PROMPTS)) if index not in first]
    # The last process's share of the second round: its widest prompt, sharing the first round's prefix.
    last = [7, 9]
    rest = [index for index in second if index not in last]
    if processes == 1:
        rounds = [first, second]
    elif rank == processes - 1:
        rounds = [first, last]
    else:
        rounds = [[], rest[rank::processes - 1]]
    drawn = []
    for mine in rounds:
        tickets = [(index, server.submit(PROMPTS[index], BUDGETS[index], key=index)) for index in mine]
        server.run()
        drawn += [(index, ticket.result().host()) for index, ticket in tickets]
    try:
        Engine(server)
        engine = "takes it"
    except ValueError:
        engine = "refuses it"
    return {"engine": engine, "shared": server.prefix_hits,
            "drawn": {str(index): {"tokens": generation.tokens[0].tolist(),
                                   "length": int(generation.lengths[0])}
                      for index, generation in drawn}}


MODES = {"pooled_server": mode_pooled_server}


if __name__ == "__main__":
    parser = worker_parser(MODES, __doc__)
    parser.add_argument("--paged", action="store_true")
    serve_mode(MODES, parser.parse_args())
