"""One rank of a pool judging a step's fit (`dew.training.memory.step_fits`) on allocators each process
reads of its own devices."""
from process_support import serve_mode, worker_parser


def step_fits(args) -> dict:
    """A small step compiled over a mesh of both processes' devices, its fit (`memory.step_fits`) on
    allocators each process reads of its own devices (`Device.memory_stats`): the pool's answer and this
    process's headroom where process 0 has room and process 1 has none, where process 1 reads none, where
    process 1's compile was refused (no executable), and the error each ends with where process 1's read
    fails, process 0 waiting on nothing."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec

    from dew._coordination import PeerFailure
    from dew.distributed.placement import MeshSpec
    from dew.training.memory import step_fits

    mesh = MeshSpec(fsdp=jax.device_count()).build()
    shape = jax.ShapeDtypeStruct((4 * jax.device_count(),), jnp.float32,
                                 sharding=NamedSharding(mesh, PartitionSpec("fsdp")))
    step = jax.jit(lambda values: values * 2).trace(shape).lower().compile()

    def stated(free):
        # An allocator of 1 GiB, `free` bytes of it one free block, none to grow; or one stating nothing.
        return lambda device: None if free is None else {
            "bytes_limit": 2 ** 30, "bytes_in_use": 2 ** 30 - free, "pool_bytes": 2 ** 30,
            "largest_free_block_bytes": free}

    found = {}
    for name, free, executable in (("roomy", 2 ** 29, step), ("tight", [2 ** 29, 0][args.process_id], step),
                                   ("unread", [2 ** 29, None][args.process_id], step),
                                   ("refused", 2 ** 29, [step, None][args.process_id])):
        jax.Device.memory_stats = stated(free)
        fits, headroom = step_fits(executable, mesh)
        found[name] = [fits, headroom is not None and headroom >= 0, headroom is None]

    def failing(device):
        raise RuntimeError("the allocator's statistics could not be read")
    jax.Device.memory_stats = stated(2 ** 29) if args.process_id == 0 else failing
    try:
        step_fits(step, mesh)
    except PeerFailure as error:
        found["failed"] = ["PeerFailure", error.phase, error.rank]
    except RuntimeError as error:
        found["failed"] = ["RuntimeError", str(error)]
    return found


if __name__ == "__main__":
    modes = {"step_fits": step_fits}
    serve_mode(modes, worker_parser(modes, __doc__).parse_args())
