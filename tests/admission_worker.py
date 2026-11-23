"""One rank of a pool admitting a program's communicators and load (`dew.communicators.admitted_together`)."""
from process_support import serve_mode, worker_parser


def admission(args) -> dict:
    """A program over both processes' devices and one over this process's alone: whether each spans every
    process (`spans_every_process`); the error each process ends with where process 1 refuses the program
    over both, process 0 admitting it, and where process 1 refuses the one over its own devices alone."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec

    from dew import communicators
    from dew._coordination import PeerFailure
    from dew.distributed.placement import MeshSpec

    mesh = MeshSpec(fsdp=jax.device_count()).build()
    shape = jax.ShapeDtypeStruct((4 * jax.device_count(),), jnp.float32,
                                 sharding=NamedSharding(mesh, PartitionSpec("fsdp")))
    pooled = jax.jit(lambda values: values * 2).trace(shape).lower().compile()
    alone = jax.ShapeDtypeStruct((4,), jnp.float32)
    local = jax.jit(lambda values: values + 1).trace(alone).lower().compile()

    def admitting():
        if args.process_id == 1:
            raise communicators.CommunicatorRoom("refused on process 1", {0: 2}, {0: 1})

    found: dict = {"spans": [communicators.spans_every_process(pooled),
                             communicators.spans_every_process(local)]}
    for name, executable in (("pooled", pooled), ("local", local)):
        try:
            communicators.admitted_together(executable, admitting)
            found[name] = ["admitted"]
        except PeerFailure as error:
            found[name] = ["PeerFailure", error.phase, error.rank]
        except communicators.CommunicatorRoom as room:
            found[name] = ["CommunicatorRoom", str(room).split(":")[0]]
    return found


if __name__ == "__main__":
    modes = {"admission": admission}
    serve_mode(modes, worker_parser(modes, __doc__).parse_args())
