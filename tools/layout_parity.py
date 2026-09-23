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

A split sequence, a tensor axis and an expert axis also reassociate sums
inside a row (over positions, heads and widths), which no reordering of rows
samples. With `--anchor` (and JAX_ENABLE_X64=1), each leaf's floor is at
least the reference's own distance from the same step computed in fp64, so
any reassociation of fp32 sums is held to fp32's own rounding of the step,
while a defect lands orders of magnitude past it.

An objective that draws noise per row (a DiT's diffusion, a DiffusionGemma
canvas's masks) draws other noise for permuted or pooled rows, so neither is
a reassociation of its step. Its floor is data parallelism over every device
instead, the layout that splits the batch sum and nothing else; the
next-token models check that layout against the permutation floor.

The same command runs in one process or under `dew launch`, where the global
batch is placed from every process alike:

    python tools/layout_parity.py --models dense --layouts data4,fsdp4,tensor4
    python tools/layout_parity.py --models moe --layouts expert4,data2_expert2,expert2_fsdp2 \\
        --mixture '{"experts": 32, "top_k": 8, "dispatch": "exchange"}' --objective '{"aux_loss_alpha": 0.01}'
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

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
    "expert2_fsdp2": {"expert": 2, "fsdp": 2},
    "sequence2": {"sequence": 2},
    "fsdp2_sequence2": {"fsdp": 2, "sequence": 2},
    "tensor2_sequence2": {"tensor": 2, "sequence": 2},
    "stage2_sequence2": {"stage": 2, "sequence": 2, "microbatches": 4},
    "stage2_fsdp2": {"fsdp": 2, "stage": 2, "microbatches": 4},
}
"""Four-device layouts: every axis alone and the combinations worth running."""


def _fixture(name: str) -> dict[str, Any]:
    """The decoder config Dew builds from a model's Hugging Face config."""
    from dew.interop.hf_decoders import translate_config

    config = json.loads((REPO / "tests/fixtures/hf" / name / "config.json").read_text())
    return dict(translate_config(config.get("text_config", config)))


def zoo() -> dict[str, Any]:
    """Small models of every family a layout splits differently: a dense
    decoder, MoE decoders with 8 and with Qwen3-30B-A3B's 128 experts, a
    Mamba-2 hybrid, a DiT and DiffusionGemma, each at widths every layout
    above divides. The last two MoE models and DiffusionGemma keep their
    released configs' routing and layer kinds. The layers a sequence axis
    splits differently come too: latent attention (MLA), a window (12 rows:
    a sequence split two ways reads it from one neighbour, four ways through
    the exchange), Mamba-2 alone, and packed rows of three documents whose
    boundaries fall inside the shards, in the dense decoder and in Rigel's
    three Mamba-2 layers to one windowed layer."""
    from benchmark_step import Case

    dense = {"vocab_size": 512, "emb_features": 64, "num_layers": 4, "num_heads": 8,
             "num_kv_heads": 4, "head_dim": 8, "mlp_features": 128, "max_seq_len": 33}
    moe = {**dense, "mixture": {"experts": 8, "top_k": 2, "expert_features": 32,
                                "layers": (0, 1, 2, 3)}}
    moe128 = {**dense, "mixture": {**_fixture("qwen3-30b-a3b")["mixture"], "expert_features": 32,
                                   "layers": (0, 1, 2, 3)}}
    # Four periods of five sliding layers and one global one, so each stage
    # of two or four holds whole periods; the trunk is the causal encoder
    # view DiffusionGemma clones its bidirectional decoder from.
    released = _fixture("diffusiongemma-26b")
    dgemma = {**released, "vocab_size": 512, "emb_features": 64, "num_heads": 4,
              "num_kv_heads": 2, "head_dim": 16, "mlp_features": 64, "num_layers": 24,
              "layer_types": tuple(released["layer_types"][:6]) * 4, "max_seq_len": 32,
              "layer_scalar": "frozen", "causal": True,
              "kinds": {"sliding_attention": {"window": 8, "rope_theta": 10000.0},
                        "full_attention": {"head_dim": 32, "num_kv_heads": 1}},
              "mixture": {**released["mixture"], "experts": 8, "top_k": 2, "expert_features": 32}}
    mamba = {"mixer": {"kind": "mamba2", "num_heads": 4, "head_dim": 16, "state_size": 8,
                       "n_groups": 1, "chunk_size": 8}}
    hybrid = {**dense, "layer_types": ("mamba", "attention") * 2,
              "kinds": {"mamba": mamba, "attention": {}}}
    window = {**dense, "layer_types": ("sliding",) * 4, "kinds": {"sliding": {"window": 12}}}
    mla = {**dense, "mixer": {"kind": "mla", "kv_lora_rank": 32, "qk_nope_head_dim": 16,
                              "qk_rope_head_dim": 8, "v_head_dim": 8}}
    mamba2 = {**dense, "layer_types": ("mamba",) * 4, "kinds": {"mamba": mamba}}
    rigel = {**dense, "layer_types": ("mamba",) * 3 + ("sliding",),
             "kinds": {"mamba": mamba, "sliding": {"window": 12}}}
    dit = {"patch_size": 2, "emb_features": 64, "num_layers": 4, "num_heads": 4, "mlp_ratio": 2,
           "output_channels": 4}
    lm = {"batch_size": 8, "seq_len": 32, "fsdp_min_param_size": 256}
    return {
        "dense": Case("causal_transformer", dense, **lm),
        "moe": Case("causal_transformer", moe, **lm),
        "moe128": Case("causal_transformer", moe128, **lm),
        "hybrid": Case("causal_transformer", hybrid, **lm),
        "window": Case("causal_transformer", window, **lm),
        "mla": Case("causal_transformer", mla, **lm),
        "mamba2": Case("causal_transformer", mamba2, **lm),
        "dense_packed": Case("causal_transformer", dense, packed_documents=3, **lm),
        "rigel_packed": Case("causal_transformer", rigel, packed_documents=3, **lm),
        "dit": Case("simple_dit", dit, batch_size=8, image_size=8, channels=4,
                    fsdp_min_param_size=256),
        "dgemma": Case("diffusion_gemma", dgemma, canvas={"prompt_length": 16, "canvas_size": 8},
                       batch_size=8, seq_len=31, fsdp_min_param_size=256),
    }


def reassociates(case) -> bool:
    """Whether reordering or pooling the batch's rows only reassociates the
    step's sums: a next-token loss draws nothing per row, a diffusion
    objective draws its noise by row."""
    return case.is_lm and case.canvas is None


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
    """The gradient the optimizer was handed, whole on every process. A
    one-device reference's leaves are already whole where they are, and
    gathering those would stack every process's copy."""
    import jax
    import numpy as np
    from jax.experimental import multihost_utils

    return jax.tree.map(
        lambda leaf: np.asarray(leaf if leaf.is_fully_addressable
                                else multihost_utils.process_allgather(leaf, tiled=True)),
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


def leaf_errors(reference, other, dtype: str) -> dict[str, float]:
    """Each leaf's L2 distance from the reference over the reference leaf's
    norm, or over the compute dtype's rounding of the whole gradient's norm
    where the leaf is smaller than that.

    A leaf's reassociation error scales with its terms, not with their sum,
    and a leaf whose terms cancel has a gradient below the rounding of the
    step: a scale just ahead of a normalisation that undoes it, as a
    decoder's last layer scalar is ahead of the final RMSNorm. Relative to
    its own norm that is noise over noise (DiffusionGemma's read 2.2), so it
    is measured against what rounding the whole step moves instead; every
    other leaf is measured against itself.
    """
    import jax
    import numpy as np

    pairs = [(jax.tree_util.keystr(path), np.asarray(want, np.float64), np.asarray(got, np.float64))
             for (path, want), got in zip(jax.tree_util.tree_flatten_with_path(reference)[0],
                                          jax.tree.leaves(other), strict=True)]
    noise = rounding_limit(dtype) * float(np.sqrt(sum(np.sum(want ** 2) for _, want, _ in pairs)))
    return {name: float(np.linalg.norm(got - want) / max(float(np.linalg.norm(want)), noise))
            for name, want, got in pairs}


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
    if not reassociates(case):
        losses, gradient, _ = trained(case, {}, batch, steps=1)
        return leaf_errors(reference, gradient, case.dtype), abs(losses[0] - reference_loss)
    runs = pooled(case, [reordered(batch, seed) for seed in range(PERMUTATIONS)], 1)
    loss = max(abs(moved - reference_loss) for moved, _ in runs)
    for pieces in (2, 4):
        runs += pooled(case, [batch, strided(batch, pieces)], pieces)
    leaves: dict[str, float] = {}
    for _, gradient in runs:
        for leaf, error in leaf_errors(reference, gradient, case.dtype).items():
            leaves[leaf] = max(leaves.get(leaf, 0.0), error)
    return leaves, loss


def exact(case, reference_gradient) -> dict[str, float]:
    """Each leaf's distance, relative to the fp64 gradient, of the fp32
    reference's step one from the same step in fp64: the model built with no
    dtype of its own, on the reference's own initial variables widened to
    fp64. The trainer draws those from its key's first split, so they come
    from the trainer, not from the objective's `init` on the key itself."""
    import benchmark_step as bench
    import jax
    import jax.numpy as jnp

    from dew.objectives.base import Step, scalar_loss
    from dew.registry import models

    state = jax.jit(_trainer(case, {}, one_device=True).initial_state)()
    wide = jax.tree.map(lambda leaf: leaf.astype(jnp.float64)
                        if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf, state.params)
    objective = bench.lm_objective(case, models.build(case.architecture, **case.config, dtype=None))
    step = Step(step=state.step, key=state.key, ema=None)

    def loss(params, batch):
        return scalar_loss(objective, {**wide, "params": params}, batch, step)[0]

    gradient = jax.jit(jax.grad(loss))(wide["params"], bench.global_batch(case))
    return leaf_errors(gradient, reference_gradient, case.dtype)


def rounding_limit(dtype: str) -> float:
    """The farthest the reference may sit from the fp64 step, and the widest
    a floor may be, and still be the compute dtype's rounding of the step:
    the square root of its machine epsilon, where a reassociation has cost
    half the significand (3.5e-4 for fp32, 8.8e-2 for bf16). Past it, two
    runs compute different steps, which no floor may absorb."""
    import jax.numpy as jnp

    return float(jnp.finfo(dtype).eps) ** 0.5


def widest_floor(floors: dict[str, float], dtype: str) -> str:
    """The leaf with the widest floor, refused past `rounding_limit`: a floor
    that wide passes any layout at its leaf, and where the floor is data
    parallelism's own deviation, it would pass that layout's own defect."""
    widest = max(floors, key=floors.__getitem__)
    limit = rounding_limit(dtype)
    if floors[widest] > limit:
        raise ValueError(
            f"the floor at {widest} is {floors[widest]:.2e}, past {dtype} rounding "
            f"({limit:.1e}), so no layout of this model can be judged by it")
    return widest


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


def run(models: Sequence[str], layouts: Sequence[str], *, dtype: str, steps: int, anchor: bool,
        mixture: dict[str, Any], objective: dict[str, Any], speak: Callable[[str], None],
        keep: Callable[[list[dict[str, Any]]], None]) -> list[dict[str, Any]]:
    """Every layout of every model, one row each, `keep` handed the rows so
    far after each. A reference and a layout run as agreed phases: a failure
    on one process fails that row on every process, or, where the others
    wait in a collective it left, ends the pool within the failure grace
    (dew.artifacts) with the rows kept so far."""
    import benchmark_step as bench
    import jax

    from dew.artifacts import agreed

    jax.config.update("jax_default_matmul_precision", "highest")
    rows = []
    for model in models:
        case = dataclasses.replace(zoo()[model], dtype=dtype)
        if mixture:
            if "mixture" not in case.config:
                raise ValueError(f"{model} has no mixture for --mixture to change")
            case = dataclasses.replace(case, config={
                **case.config, "mixture": {**case.config["mixture"], **mixture}})
        case = dataclasses.replace(case, objective={**case.objective, **objective})
        batch = bench.global_batch(case)
        # The exchange needs an expert axis; one device computes the same
        # layer through the global dispatch.
        reference = case if case.config.get("mixture", {}).get("dispatch") != "exchange" else (
            dataclasses.replace(case, config={
                **case.config, "mixture": {**case.config["mixture"], "dispatch": "global"}}))
        try:
            ref_losses, ref_gradient, ref_compiled = agreed(
                f"reference of {model}",
                lambda: trained(reference, {}, batch, steps=steps, one_device=True))
            floors, loss_floor = floor(reference, batch, ref_gradient, ref_losses[0])
            if anchor and reassociates(reference):
                rounding = exact(reference, ref_gradient)
                farthest = max(rounding, key=rounding.__getitem__)
                if rounding[farthest] > rounding_limit(dtype):
                    raise ValueError(
                        f"the {dtype} reference is {rounding[farthest]:.2e} from the fp64 step at "
                        f"{farthest}, past {dtype} rounding ({rounding_limit(dtype):.1e}): the two "
                        f"compute different steps, so the anchor cannot bound the layouts")
                floors = {leaf: max(value, rounding[leaf]) for leaf, value in floors.items()}
            widest = widest_floor(floors, dtype)
        except Exception as error:  # no reference judges no layout: the model's one row
            rows.append({"model": model, "layout": "reference", "processes": jax.process_count(),
                         "status": "error", "error": f"{type(error).__name__}: {error}"[:2000],
                         "traceback": traceback.format_exc()[-4000:]})
            speak(f"[{model}] reference error {rows[-1]['error'][:300]}")
            keep(rows)
            continue
        speak(f"[{model}] reference losses {ref_losses}, largest floor {floors[widest]:.2e} at {widest}")
        for name in layouts:
            row: dict[str, Any] = {"model": model, "mixture": case.config.get("mixture"),
                                   "objective": case.objective, "layout": name,
                                   "processes": jax.process_count(),
                                   "reference_losses": ref_losses}
            started = time.perf_counter()
            try:
                losses, gradient, compiled = agreed(
                    f"{model} on {name}", lambda: trained(case, LAYOUTS[name], batch, steps=steps))
                row.update(compiled, losses=losses, **judged(
                    leaf_errors(ref_gradient, gradient, dtype), floors, abs(losses[0] - ref_losses[0]),
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
            keep(rows)
            speak(f"[{model}/{name}] {row['status']} "
                  + (f"leaf {row['worst_ratio']:.2f} of bound at {row['worst_leaf']}, "
                     f"loss {row['loss_error']:.1e} of {row['loss_bound']:.1e}, "
                     f"flops x{row.get('flops_ratio', float('nan')):.2f}"
                     if "worst_ratio" in row else row["error"][:300]))
    return rows


def main(models: Annotated[tuple[str, ...], tyro.conf.arg(help="zoo() names")] = ("dense",),
         layouts: Annotated[tuple[str, ...], tyro.conf.arg(help="LAYOUTS names")] = tuple(LAYOUTS),
         dtype: str = "float32", steps: int = 3, anchor: bool = False,
         out: Path | None = None,
         mixture: Annotated[str, tyro.conf.arg(
             help="JSON merged into each model's mixture, e.g. '{\"dispatch\": \"exchange\"}'")] = "{}",
         objective: Annotated[str, tyro.conf.arg(
             help="JSON of LMObjective keywords, e.g. '{\"aux_loss_alpha\": 0.01}'")] = "{}",
         ) -> None:
    """Run the layouts of each model against one device; see the module docstring."""
    from dew.training.runtime import prepare_process

    prepare_process()
    import jax

    if anchor and not jax.config.jax_enable_x64:
        raise SystemExit("--anchor computes the step in fp64, which needs JAX_ENABLE_X64=1")
    speaker = jax.process_index() == 0

    def keep(rows: list[dict[str, Any]]) -> None:
        if speaker and out is not None:
            out.write_text(json.dumps(rows, indent=1))

    rows = run(models, layouts, dtype=dtype, steps=steps, anchor=anchor, mixture=json.loads(mixture),
               objective=json.loads(objective), speak=lambda line: print(line, flush=True) if speaker else None, keep=keep)
    if any(row["status"] != "works" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    tyro.cli(main)
