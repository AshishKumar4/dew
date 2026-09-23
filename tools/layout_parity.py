#!/usr/bin/env python3
"""Every layout of a model against one device: the loss and each gradient leaf.

Each process builds the one-device reference on its first local device, then
each layout's trainer on the global mesh, from one key and one global batch,
and runs the trainer's own compiled step. The optimizer is `stash` before
adam, so the gradient the step handed the optimizer is kept whole and
compared leaf by leaf.

A layout changes the order of the sums over rows, tokens and split widths
and nothing else, so its error is held to the reference's own deviation
under that kind of change: the batch with its rows in PERMUTATIONS orders,
and pooled from 2 and 4 accumulated slices of consecutive rows and of
strided rows, which is the per-device shapes and partial sums a split batch
or a pipeline's microbatches compute. The floor is the largest deviation per
leaf, and a layout `works` when every leaf is within `FLOOR_FACTOR` of it
and the loss within as much of its own.

An objective that draws noise per row (a DiT's diffusion) draws other noise
for permuted or pooled rows, so neither is a reassociation of its step. Its
floor is data parallelism over every device instead, the layout that splits
the batch sum and nothing else; the language models check that layout
against the permutation floor.

The same command runs in one process or under `dew launch`, where the global
batch is placed from every process alike:

    python tools/layout_parity.py --models dense --layouts data4,fsdp4,tensor4
    dew launch --processes-per-host 4 --devices-per-process 1 -- \\
        python tools/layout_parity.py --models dense,dit --out parity.json

Process 0 prints a line per layout and writes the rows; the exit status is
nonzero when a layout mismatches or fails.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated, Any

import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))

FLOOR_FACTOR = 4.0
PERMUTATIONS = 16

LAYOUTS: dict[str, dict[str, int]] = {
    "data4": {},
    "fsdp4": {"fsdp": 4},
    "tensor4": {"tensor": 4},
    "expert4": {"expert": 4},
    "sequence4": {"sequence": 4},
    "stage4": {"stage": 4, "microbatches": 4},
    "replicas4": {"replicas": 4},
    "fsdp2_tensor2": {"fsdp": 2, "tensor": 2},
    "replicas2_fsdp2": {"fsdp": 2, "replicas": 2},
    "data2_expert2": {"expert": 2},
    "fsdp2_sequence2": {"fsdp": 2, "sequence": 2},
    "tensor2_sequence2": {"tensor": 2, "sequence": 2},
    "stage2_fsdp2": {"fsdp": 2, "stage": 2, "microbatches": 4},
}
"""Four-device layouts: every axis alone and the combinations worth running."""


def zoo() -> dict[str, Any]:
    """Small models of every family a layout splits differently: a dense
    decoder, an MoE decoder, a Mamba-2 hybrid and a DiT, each at widths
    every layout above divides."""
    from benchmark_step import Case

    dense = {"vocab_size": 512, "emb_features": 64, "num_layers": 4, "num_heads": 8,
             "num_kv_heads": 4, "head_dim": 8, "mlp_features": 128, "max_seq_len": 33}
    moe = {**dense, "mixture": {"experts": 8, "top_k": 2, "expert_features": 32,
                                "layers": (0, 1, 2, 3)}}
    hybrid = {**dense, "layer_types": ("mamba", "attention") * 2,
              "kinds": {"mamba": {"mixer": {"kind": "mamba2", "num_heads": 4, "head_dim": 16,
                                            "state_size": 8, "n_groups": 1, "chunk_size": 8}},
                        "attention": {}}}
    dit = {"patch_size": 2, "emb_features": 64, "num_layers": 4, "num_heads": 4, "mlp_ratio": 2,
           "output_channels": 4}
    lm = {"batch_size": 8, "seq_len": 32, "fsdp_min_param_size": 256}
    return {
        "dense": Case("causal_transformer", dense, **lm),
        "moe": Case("causal_transformer", moe, **lm),
        "hybrid": Case("causal_transformer", hybrid, **lm),
        "dit": Case("simple_dit", dit, batch_size=8, image_size=8, channels=4,
                    fsdp_min_param_size=256),
    }


def stash():
    """Pass the updates on unchanged and keep them as the optimizer state."""
    import jax
    import optax

    return optax.GradientTransformation(
        lambda params: {"gradient": jax.tree.map(jax.numpy.zeros_like, params)},
        lambda updates, state, params=None: (updates, {"gradient": updates}))


def placed(batch, mesh):
    """The global batch on `mesh`, the same on every process."""
    import jax
    import numpy as np

    from dew.training.distributed import batch_shardings

    def put(leaf, sharding):
        array = np.asarray(leaf)
        return jax.make_array_from_callback(array.shape, sharding, lambda index: array[index])
    return jax.tree.map(put, batch, batch_shardings(mesh, batch))


def reordered(batch, seed: int):
    """The batch with its rows reversed (seed 0) or drawn in another order."""
    import jax
    import numpy as np

    rows = len(jax.tree.leaves(batch)[0])
    order = np.arange(rows)[::-1] if seed == 0 else np.random.default_rng(seed).permutation(rows)
    return jax.tree.map(lambda leaf: np.asarray(leaf)[order], batch)


def _trainer(case, fields: dict[str, int], *, one_device: bool = False, accumulation: int = 1):
    """The trainer of `case` on the layout `fields` names, or on this
    process's first device, stashing each gradient the optimizer is handed."""
    import benchmark_step as bench
    import jax
    import optax

    from dew.training import Layout, MeshSpec, Trainer, build_mesh

    trainer = Trainer(bench.build_objective(case), optax.chain(stash(), optax.adam(1e-3)),
                      key=jax.random.key(0), mesh=bench.mesh_spec(fields),
                      layout=Layout(min_shard=case.fsdp_min_param_size, tolerance=1.0),
                      accumulation=accumulation, checkpoints=None, tracker=None)
    if one_device:
        trainer.device_mesh = build_mesh(MeshSpec(), [jax.local_devices()[0]])
    return trainer


def _gradient(state):
    import jax
    import numpy as np
    from jax.experimental import multihost_utils

    return jax.tree.map(
        lambda leaf: np.asarray(multihost_utils.process_allgather(leaf, tiled=True)),
        state.opt_state[0]["gradient"])


def trained(case, fields: dict[str, int], batch, *, steps: int, one_device: bool = False
            ) -> tuple[list[float], Any, dict[str, Any]]:
    """The losses of `steps` steps on the layout `fields` names, step one's
    gradient gathered whole, and what the compiler says of the step."""
    trainer = _trainer(case, fields, one_device=one_device)
    state, _, _ = trainer.place()
    data = placed(batch, trainer.device_mesh)
    step = trainer.compile(state, data)
    losses, gradient = [], None
    for _ in range(steps):
        state, loss, _, _, _ = step(state, data)
        losses.append(float(loss))
        gradient = _gradient(state) if gradient is None else gradient
    compiled = {"flops_per_device": trainer.flops_per_step,
                "mesh": {axis: int(size) for axis, size in trainer.device_mesh.shape.items()}}
    return losses, gradient, compiled


def pooled(case, batches, pieces: int) -> list[tuple[float, Any]]:
    """Step one's loss and gradient on one device for each batch, each pooled
    from `pieces` consecutive slices of its rows by accumulation, from one
    initial state through one compiled step."""
    import jax
    import numpy as np

    trainer = _trainer(case, {}, one_device=True, accumulation=pieces)
    step, results = None, []
    for batch in batches:
        state, _, _ = trainer.place()
        rows = len(jax.tree.leaves(batch)[0]) // pieces
        slices = [placed(jax.tree.map(lambda leaf: np.asarray(leaf)[i * rows:(i + 1) * rows], batch),
                         trainer.device_mesh) for i in range(pieces)]
        step = trainer.compile(state, slices[0]) if step is None else step
        losses = []
        for data in slices:
            state, loss, _, _, _ = step(state, data)
            losses.append(float(loss))
        results.append((float(np.mean(losses)), _gradient(state)))
    return results


def leaf_errors(reference, other) -> dict[str, float]:
    """Each leaf's L2 distance from the reference over the reference's norm."""
    import jax
    import numpy as np

    errors = {}
    for (path, want), got in zip(jax.tree_util.tree_flatten_with_path(reference)[0],
                                 jax.tree.leaves(other), strict=True):
        want, got = np.asarray(want, np.float64), np.asarray(got, np.float64)
        norm = np.linalg.norm(want)
        errors[jax.tree_util.keystr(path)] = float(
            np.linalg.norm(got - want) / norm if norm else np.linalg.norm(got))
    return errors


def strided(batch, pieces: int):
    """The batch with rows m, m + pieces, m + 2 * pieces, ... gathered into
    its m-th consecutive slice: a pipeline's microbatch m."""
    import jax
    import numpy as np

    rows = len(jax.tree.leaves(batch)[0])
    order = np.concatenate([np.arange(start, rows, pieces) for start in range(pieces)])
    return jax.tree.map(lambda leaf: np.asarray(leaf)[order], batch)


def floor(case, batch, reference, reference_loss: float) -> tuple[dict[str, float], float]:
    """Per leaf, the largest deviation of the reference from itself under a
    reassociation of the batch's sums, and the same of the step-one loss."""
    if not case.is_lm:
        losses, gradient, _ = trained(case, {}, batch, steps=1)
        return leaf_errors(reference, gradient), abs(losses[0] - reference_loss)
    runs = pooled(case, [reordered(batch, seed) for seed in range(PERMUTATIONS)], 1)
    loss = max(abs(moved - reference_loss) for moved, _ in runs)
    for pieces in (2, 4):
        runs += pooled(case, [batch, strided(batch, pieces)], pieces)
    leaves: dict[str, float] = {}
    for _, gradient in runs:
        for leaf, error in leaf_errors(reference, gradient).items():
            leaves[leaf] = max(leaves.get(leaf, 0.0), error)
    return leaves, loss


def judged(errors: dict[str, float], floors: dict[str, float], loss: float,
           loss_floor: float, reference_loss: float) -> dict[str, Any]:
    """Every leaf against FLOOR_FACTOR times its floor, fp32 epsilon at least."""
    import numpy as np

    eps = float(np.finfo(np.float32).eps)
    ratios = {leaf: error / (FLOOR_FACTOR * max(floors[leaf], eps)) for leaf, error in errors.items()}
    worst = max(ratios, key=ratios.__getitem__)
    loss_bound = FLOOR_FACTOR * max(loss_floor, eps * abs(reference_loss))
    return {"worst_leaf": worst, "worst_leaf_error": errors[worst], "worst_leaf_floor": floors[worst],
            "worst_ratio": ratios[worst], "loss_error": loss, "loss_bound": loss_bound,
            "status": "works" if ratios[worst] <= 1.0 and loss <= loss_bound else "MISMATCH"}


def run(models: Sequence[str], layouts: Sequence[str], *, dtype: str, steps: int,
        speak: Callable[[str], None]) -> list[dict[str, Any]]:
    import benchmark_step as bench
    import jax

    jax.config.update("jax_default_matmul_precision", "highest")
    rows = []
    for model in models:
        case = dataclasses.replace(zoo()[model], dtype=dtype)
        batch = bench.global_batch(case)
        ref_losses, ref_gradient, ref_compiled = trained(case, {}, batch, steps=steps,
                                                         one_device=True)
        floors, loss_floor = floor(case, batch, ref_gradient, ref_losses[0])
        speak(f"[{model}] reference losses {ref_losses}, largest floor {max(floors.values()):.2e}")
        for name in layouts:
            row: dict[str, Any] = {"model": model, "layout": name, "processes": jax.process_count(),
                                   "reference_losses": ref_losses}
            started = time.perf_counter()
            try:
                losses, gradient, compiled = trained(case, LAYOUTS[name], batch, steps=steps)
                row.update(compiled, losses=losses, **judged(
                    leaf_errors(ref_gradient, gradient), floors, abs(losses[0] - ref_losses[0]),
                    loss_floor, ref_losses[0]))
                devices = jax.device_count()
                if compiled["flops_per_device"] and ref_compiled["flops_per_device"]:
                    # Above one when devices compute what one device need not.
                    row["flops_ratio"] = (compiled["flops_per_device"] * devices
                                          / ref_compiled["flops_per_device"])
            except Exception as error:  # a failing layout is a row of the matrix
                row.update(status="error", error=f"{type(error).__name__}: {error}"[:2000],
                           traceback=traceback.format_exc()[-4000:])
            row["seconds"] = round(time.perf_counter() - started, 1)
            rows.append(row)
            speak(f"[{model}/{name}] {row['status']} "
                  + (f"leaf {row['worst_ratio']:.2f} of bound at {row['worst_leaf']}, "
                     f"loss {row['loss_error']:.1e} of {row['loss_bound']:.1e}, "
                     f"flops x{row.get('flops_ratio', float('nan')):.2f}"
                     if "worst_ratio" in row else row["error"][:300]))
    return rows


def main(models: Annotated[tuple[str, ...], tyro.conf.arg(help="zoo() names")] = ("dense",),
         layouts: Annotated[tuple[str, ...], tyro.conf.arg(help="LAYOUTS names")] = tuple(LAYOUTS),
         dtype: str = "float32", steps: int = 3, out: Path | None = None) -> None:
    """Run the layouts of each model against one device; see the module docstring."""
    from dew.training.runtime import prepare_process

    prepare_process()
    import jax

    speaker = jax.process_index() == 0
    rows = run(models, layouts, dtype=dtype, steps=steps,
               speak=lambda line: print(line, flush=True) if speaker else None)
    if speaker and out is not None:
        out.write_text(json.dumps(rows, indent=1))
    if any(row["status"] != "works" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    tyro.cli(main)
