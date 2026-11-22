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
    staged_values, refused = _handled(mesh.build(paired), args.run_dir)
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
    return {"process_index": jax.process_index(), "finite": read, "steps": steps, "handled": staged_values,
            "refused": refused, "stray_refused": _stray(mesh.build(paired), args.run_dir),
            "lost": _lost(paired, args.run_dir, args.local_dir)}


def _lost(paired, run_dir, local_dir) -> list[int]:
    """A pipeline whose stages each copy their rows over a replicas axis of one device of each process, each
    reading its share of a step's rows, its step 1 saved on the local tier alone (`save_local`), process
    1's local copy then lost, as a replaced host's is, and the run resumed through its fit from that step to
    the next: the steps each fit stepped to, the lost copy recovered from process 0's replica of every
    stage's rows (rows 10813, 10814; row 10897: they were proven through their primitives alone)."""
    import functools
    import shutil
    from pathlib import Path

    import jax
    from affine_run import Replaying
    from jax.experimental import multihost_utils
    from test_stage_local import ADAM, BATCH, SEQ, VOCAB, decoder, trainer
    from token_support import token_rows

    from dew.checkpoints import Checkpoints
    from dew.data.dataset import Loader
    from dew.training import MeshSpec

    replicated = MeshSpec(replicas=2, stage=2)

    def shares() -> Loader:
        # Each process its share of a step's rows, as a pool over a replicas axis reads them.
        def batch_at(partition, step: int) -> dict:
            rows = token_rows(VOCAB, BATCH, SEQ + 1, seed=step)
            return {name: value[partition.index::partition.count] for name, value in rows.items()}
        return Loader(train=lambda partition: Replaying(functools.partial(batch_at, partition)), val=None,
                      records=None, batch=BATCH)

    def run(name: str, steps: int, resume: str):
        kept = Checkpoints(str(Path(run_dir) / name), local_directory=str(Path(local_dir) / name),
                           local_every=1)
        staged = trainer(decoder(layer_scalar="frozen"), ADAM, replicated, len(paired), checkpoints=kept)
        staged.device_mesh = replicated.build(paired)
        return staged.fit(shares(), steps=steps, checkpoint_every=2, log_every=1, resume=resume,
                          checkpoint_final=False)

    def own(state) -> list:
        # This process's shards of the weights and the optimizer's state, as the step left them.
        return [np.asarray(shard.data) for leaf in jax.tree.leaves((state.variables, state.opt_state))
                for shard in leaf.addressable_shards]

    first = int(run("lost", 1, "never").step)
    if jax.process_index() == 1:
        shutil.rmtree(Path(local_dir) / "lost" / "process1")
    multihost_utils.sync_global_devices("local copy lost")
    resumed = run("lost", 2, "require")
    # The run no copy of which was lost, two steps of the same rows from the same start: the recovered
    # run's state equal to it, its weights, its optimizer's state and the rows its data read on to.
    whole = run("whole", 2, "never")
    alike = all(np.array_equal(left, right) for left, right in zip(own(resumed), own(whole), strict=True))
    return [first, int(resumed.step), Checkpoints(str(Path(run_dir) / "lost")).latest, alike]


def _handled(mesh, directory) -> tuple[dict, bool]:
    """A scalar on one stage's devices and a vector on the other's, written by Dew's handler on the
    persistent tier and read back: each process's view of what it holds of them, which a process outside
    a stage holds nothing of; and whether a pmap's output, on each process's own devices under one name,
    is refused (row 10780)."""
    import jax
    import numpy as np
    import orbax.checkpoint as ocp
    from jax.sharding import NamedSharding, PartitionSpec as P

    from dew.checkpoints import _handler
    from dew.checkpoints.submesh import save_args
    from dew.training.stages import stage_mesh

    placed = {"scalar": NamedSharding(stage_mesh(mesh, 1), P()),
              "vector": NamedSharding(stage_mesh(mesh, 0), P())}
    values = {"scalar": np.float32(3.5), "vector": np.arange(4, dtype=np.float32)}
    arrays = {name: jax.make_array_from_callback(
        np.shape(values[name]), placed[name], lambda index, name=name: np.asarray(values[name])[index],
        dtype=np.float32) for name in values}
    with ocp.Checkpointer(_handler()) as checkpointer:
        # Saved as a pipeline's checkpoint saves its rows (`StageLocal.saved`, `HeldApart`).
        checkpointer.save(f"{directory}/handled", args=ocp.args.PyTreeSave(arrays, save_args=save_args(
            arrays, held_apart=True)))
        restored = checkpointer.restore(f"{directory}/handled", restore_args={
            "scalar": ocp.ArrayRestoreArgs(sharding=placed["scalar"], dtype=np.float32, global_shape=()),
            "vector": ocp.ArrayRestoreArgs(sharding=placed["vector"])})
    held = {name: [np.asarray(shard.data).tolist() for shard in restored[name].addressable_shards][:1]
            for name in restored}
    local = jax.pmap(lambda x: x * (jax.process_index() + 1))(np.ones((jax.local_device_count(), 2)))
    try:
        with ocp.Checkpointer(_handler()) as checkpointer:
            checkpointer.save(f"{directory}/pmapped", {"local": local})
        refused = False
    except ValueError:
        refused = True
    return {name: {"shape": list(restored[name].shape), "held": held[name]} for name in held}, refused


def _stray(mesh, directory) -> bool:
    """Whether a value each process puts on its own stage's devices under one name, after a pipeline's
    checkpoints wrote rows on those very devices, is refused: no checkpoint made it (row 10780)."""
    import jax
    import numpy as np
    import orbax.checkpoint as ocp
    from jax.sharding import NamedSharding, PartitionSpec as P

    from dew.checkpoints import _handler
    from dew.training.stages import stage_mesh

    own = NamedSharding(stage_mesh(mesh, jax.process_index()), P())
    value = np.full((2,), jax.process_index(), np.float32)
    stray = jax.make_array_from_callback((2,), own, lambda index: value[index], dtype=np.float32)
    try:
        with ocp.Checkpointer(_handler()) as checkpointer:
            checkpointer.save(f"{directory}/stray", {"x": stray})
    except ValueError:
        return True
    return False


MODES = {"stage_save": mode_stage_save}


if __name__ == "__main__":
    parser = worker_parser(MODES, __doc__)
    parser.add_argument("--run-dir", help="the run's persistent checkpoints")
    parser.add_argument("--local-dir", help="every process's local checkpoint directory")
    serve_mode(MODES, parser.parse_args(sys.argv[1:]))
