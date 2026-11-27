"""A pool's ranks writing and reading a train state's packed matrices (`dew.checkpoints._packed`, row 13742:
a run's frozen quantized collection, each matrix its checkpoint record), driven by tests/test_multiprocess.py
as `multiprocess_worker`'s modes are: each mode records what it read into --out. dew is imported inside
the modes, so the pool joins before any JAX backend opens."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
from process_support import refusal, serve_mode, worker_parser


def _packed(sharding) -> dict:
    """A frozen quantized collection of the two packed formats, every leaf placed at `sharding`: a
    `[256, 256]` bfloat16 kernel's window (`quantization.bf16_windows`, encoded where it lies) and an NVFP4
    matrix of the same extents (`formats.NvFp4.of`), each from a fixed seed."""
    import jax
    import jax.numpy as jnp

    from dew import quantization
    from dew.nn.kernels import formats

    rng = np.random.default_rng(13742)
    kernel = (rng.normal(size=(256, 256)) * 0.02).astype(jnp.bfloat16)
    placed = jax.make_array_from_callback(kernel.shape, sharding, lambda index: kernel[index],
                                          dtype=kernel.dtype)
    window = quantization.bf16_windows({"params": {"kernel": placed}}, sites=[""])[quantization.QUANTIZATION][
        "kernel"]
    codes = rng.integers(-6, 7, size=(256, 256)).astype(np.float32) / 2
    matrix = formats.NvFp4.of(jnp.asarray(codes), jnp.full((16, 256), 0.5, jnp.float8_e4m3fn),
                              jnp.float32(2.0 ** -6))
    return {"window": window, "nvfp4": jax.device_put(matrix, sharding)}


def _foreign(packed: dict) -> dict:
    """`packed` with its window written under another encoder's form (its digest's first bit flipped)."""
    window = packed["window"]
    flipped = bytes([window.form[0] ^ 1, *window.form[1:]])
    return {**packed, "window": dataclasses.replace(window, form=flipped)}


def _template(state):
    """`state`'s shapes, dtypes and placements, a restore's template (its packed matrices' forms their
    own)."""
    import jax

    return jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=leaf.sharding),
                        state)


def _read(written: dict, restored: dict) -> dict:
    """Whether each packed matrix `restored` holds is `written`'s: its type and form, each leaf's shape and
    dtype, and the bytes of every shard this process holds of it (a process holding none of a matrix reads
    none of it)."""
    import jax

    def own(tree) -> list[tuple]:
        return [(leaf.shape, leaf.dtype, [np.asarray(held.data).tobytes()
                                          for held in leaf.addressable_shards])
                for leaf in jax.tree.leaves(tree)]

    return {name: type(restored[name]) is type(matrix) and restored[name].form == matrix.form
            and own(matrix) == own(restored[name]) for name, matrix in written.items()}


def local_packed(args, *, lost: bool = False) -> dict:
    """A pool's train state holding a frozen quantized collection beside its own (its packed matrices
    placed whole on the run's mesh, so each process holds a replica of every shard), saved on the local tier
    alone (`Checkpoints.save_local`) and restored into its template: each packed matrix as written on every
    process. Then a step whose window is under another encoder's form, refused as every process reads it
    (`process_support.refusal`). Where `lost`, process 1's local copy of each step is gone before its
    restore, as a replaced host's is: its records, the foreign digest too, come from process 0's
    replica."""
    import shutil

    import jax
    from jax.experimental import multihost_utils
    from jax.sharding import NamedSharding, PartitionSpec
    from multiprocess_worker import build_trainer

    from dew._coordination import PeerFailure
    from dew.quantization import QUANTIZATION

    def saved(step: int, held: dict):
        # Saved on the local tier alone, the managers closed after it (`Checkpoints.close`: a closed one
        # reopens on its next use); where `lost`, process 1's copy then gone, as a replaced host's is, so
        # its restore reads process 0's replica (`Checkpoints._recovered`).
        written = dataclasses.replace(placed, step=placed.step + step,
                                      variables={**placed.variables, QUANTIZATION: held})
        with checkpoints:
            checkpoints.save_local(step, written, None)
        if lost:
            if jax.process_index() == 1:
                shutil.rmtree(Path(args.local_dir) / "process1")
            multihost_utils.sync_global_devices(f"local copy {step} lost")
        return written

    trainer = build_trainer(args.name, args.run_dir, args.fsdp_size, local_dir=args.local_dir, local_every=1)
    checkpoints = trainer.checkpoints
    assert checkpoints is not None
    placed, _, _ = trainer.place()
    packed = _packed(NamedSharding(trainer.device_mesh, PartitionSpec()))
    state = saved(1, packed)
    try:
        with checkpoints:
            restored, _ = checkpoints.restore(_template(state), 1)
    except (ValueError, PeerFailure) as error:
        # Its message beside the refusal's type: a failing assertion shows what refused the read.
        return {"read": {**refusal(error), "message": str(error)}, "foreign": None}
    read = _read(packed, restored.variables[QUANTIZATION])
    saved(2, _foreign(packed))
    try:
        with checkpoints:
            checkpoints.restore(_template(state), 2)
    except (ValueError, PeerFailure) as error:
        return {"read": read, "foreign": refusal(error)}
    return {"read": read, "foreign": None}


def local_packed_lost(args) -> dict:
    return local_packed(args, lost=True)


def stage_packed(args) -> dict:
    """A two-stage pipeline's train state whose stages are each one process's devices (as
    stage_save_worker's), holding a frozen quantized collection on the second stage's devices alone, so
    process 0 holds no shard of its matrices nor of their digests: written by the processes holding it
    (`Checkpoints.save` of a pipeline's rows held apart) and restored into its template, each matrix as
    written where it is held; then a step whose window is under another encoder's form, refused on every
    process alike, the one reading the digest by its check and the other by the pool's agreement."""
    import jax
    from jax.sharding import NamedSharding, PartitionSpec
    from multiprocess_worker import pipeline_trainer
    from stage_save_worker import paired_devices

    from dew._coordination import PeerFailure
    from dew.checkpoints import Checkpoints, _Laid
    from dew.quantization import QUANTIZATION
    from dew.training import MeshSpec
    from dew.training.stages import stage_mesh

    paired = paired_devices()
    trainer = pipeline_trainer(2, 2, 2)
    trainer.device_mesh = MeshSpec(fsdp=2, stage=2).build(paired)
    placed, _, _ = trainer.place()
    packed = _packed(NamedSharding(stage_mesh(trainer.device_mesh, 1), PartitionSpec()))
    # The pipeline's own seams (a run's checkpoint envelope's and its resume's): its rows saved held apart
    # where its staging holds them so, and restored onto its stages' devices and restacked there.
    staging = trainer._staging
    laid, template = _Laid(held_apart=staging.apart), _template(
        dataclasses.replace(placed, variables={**placed.variables, QUANTIZATION: packed}))
    reports = {}
    for step, held in ((1, packed), (2, _foreign(packed))):
        # The state a step left, which the step it is saved under names (`_filled`).
        state = dataclasses.replace(placed, step=placed.step + step,
                                    variables={**placed.variables, QUANTIZATION: held})
        directory = Path(args.run_dir) / f"stage-{step}"
        checkpoints = Checkpoints(str(directory))
        checkpoints.save(step, staging.saved(state), _laid=laid)
        checkpoints.wait()
        try:
            restored, _ = staging.restore(Checkpoints(str(directory)), template, step)
        except (ValueError, PeerFailure) as error:
            # Its message beside the refusal's type: a failing assertion shows what refused the read.
            reports[step] = refusal(error) if step == 2 else {**refusal(error), "message": str(error)}
            continue
        reports[step] = _read(packed, restored.variables[QUANTIZATION])
    return {"holds": bool(packed["window"].words.addressable_shards), "read": reports[1],
            "foreign": reports[2]}


MODES = {"local_packed": local_packed, "local_packed_lost": local_packed_lost, "stage_packed": stage_packed}


if __name__ == "__main__":
    parser = worker_parser(MODES, __doc__)
    parser.add_argument("--name", default="packed")
    parser.add_argument("--run-dir", help="the run's persistent checkpoints")
    parser.add_argument("--local-dir", help="every process's local checkpoint directory")
    serve_mode(MODES, parser.parse_args(sys.argv[1:]))
