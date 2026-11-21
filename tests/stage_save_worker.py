"""A pool's ranks for a pipeline whose stages are each one process's devices (`test_multiprocess`'s
stage-save cell, row 10780), run as `multiprocess_worker`'s modes are."""

import sys

import numpy as np
from process_support import serve_mode, worker_parser


def mode_stage_save(args) -> dict:
    """A two-stage pipeline whose stages are each one process's devices alone (its devices interleaved
    so the stage axis, the mesh's last, pairs one of each process's): its finite check over leaves placed
    on each stage alone; a step checkpointed on the local tier (`--local-dir`) alone, resumed from it to a
    step the persistent tier (`--run-dir`) holds too, and resumed from that one step further (row 10780:
    each rank read each stage's finite flag, those outside a stage holding none of it, and Orbax refused
    each stage's rows as host-local arrays)."""
    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from dew.checkpoints import Checkpoints
    from dew.training import MeshSpec
    from dew.training.transaction import finite_flags

    # One leaf on each process's devices alone, the second holding a NaN.
    flags = []
    for rank in range(jax.process_count()):
        devices = [device for device in jax.devices() if device.process_index == rank]
        sharding = NamedSharding(Mesh(devices, ("x",)), P())
        value = np.full((4,), np.nan if rank == 1 else 1.0, np.float32)
        flags.append(jax.make_array_from_single_device_arrays(
            value.shape, sharding, [jax.device_put(value, device) for device in devices
                                    if device.process_index == jax.process_index()], dtype=value.dtype))
    read = finite_flags(flags)

    from test_stage_local import ADAM, decoder, stream, trainer

    paired = [device for pair in zip(*(sorted((device for device in jax.devices()
                                                if device.process_index == rank), key=lambda d: d.id)
                                               for rank in range(jax.process_count())), strict=True)
              for device in pair]
    mesh = MeshSpec(fsdp=2, stage=2)
    persistent = Checkpoints(str(args.run_dir))

    def run(steps: int, *, local: bool = True, resume: str = "require") -> tuple[int, int | None]:
        kept = (Checkpoints(str(args.run_dir), local_directory=str(args.local_dir), local_every=1) if local
                else Checkpoints(str(args.run_dir)))
        # A frozen scalar a layer, which the store holds, each stage's rows on its own devices.
        staged = trainer(decoder(layer_scalar="frozen"), ADAM, mesh, len(paired), checkpoints=kept)
        staged.device_mesh = mesh.build(paired)
        stepped = int(staged.fit(stream(), steps=steps, checkpoint_every=2, log_every=1, resume=resume,
                                 checkpoint_final=False).step)
        return stepped, persistent.latest

    # Step 1 on the local tier alone (the persistent tier holds none), resumed from it to step 2, which the
    # persistent tier holds; then that resumed from the persistent tier alone to step 3, which it does not.
    steps = [run(1, resume="never"), run(2), run(3, local=False)]
    return {"process_index": jax.process_index(), "finite": read, "steps": steps}


MODES = {"stage_save": mode_stage_save}


if __name__ == "__main__":
    parser = worker_parser(MODES, __doc__)
    parser.add_argument("--run-dir", help="the run's persistent checkpoints")
    parser.add_argument("--local-dir", help="every process's local checkpoint directory")
    serve_mode(MODES, parser.parse_args(sys.argv[1:]))
