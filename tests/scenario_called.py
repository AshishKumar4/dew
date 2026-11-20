"""The scenario runs a fresh process calls (tests/test_scenario_runs.py's `called`,
`scenario_contract.Called`): each trained through Dew's API in this process in a phase
(`scenario_contract.Phase`), returning its persistent-cache counts. A module of their own, importing what
they run and none of the scenarios' own file, which a fresh process imported whole: 22.7 s of each of a
resume claim's four processes, where these take about 6 (row 7006)."""

import dataclasses
import itertools
from collections.abc import Mapping
from pathlib import Path

import jax
import numpy as np
import optax
from recording import RecordingTracker
from scenario_contract import CELL, Phase, assert_steps_finite

from dew.checkpoints import Checkpoints
from dew.data import Memory
from dew.distributed.placement import Layout, MeshSpec
from dew.nn.kernels.generation import run_precision
from dew.nn.precision import numerics_of
from dew.training import Trainer

STEPS = 3
"""A scenario run's steps, every scenario's (tests/test_scenario_runs.py's `settings`)."""


def diffusion_trained(cell: Mapping, root: str, phase: Mapping) -> dict[str, float]:
    """A tiny DiT's diffusion run over 8-pixel images at the cell's settings (`test_quantization`'s smallest
    run), its model's dtype left for the run's precision to derive, trained in this process in `phase`
    (`Phase`): its process's persistent-cache counts."""
    from test_quantization import RES, RUN_BATCH, diffusion_run

    from dew.quantization import Quantization

    phase = Phase(**phase)
    phase.entered()
    trainer = {"precision": numerics_of(cell["precision"]), "log_every": 1,
               "mesh": MeshSpec(fsdp=cell["devices"]), "layout": Layout(min_shard=64),
               **({"quantization": Quantization()} if cell["quantization"] == "qat" else {})}
    config = diffusion_run(Path(root) / "runs", RUN_BATCH, **trainer)
    fields = {name: value for name, value in config.model.fields.items() if name != "dtype"}
    config = dataclasses.replace(
        config, model=dataclasses.replace(config.model, fields={**fields, "remat": cell["memory"]}),
        trainer=dataclasses.replace(config.trainer, steps=STEPS, checkpoint_final=True))
    config = config.assigned([f"{path}={value}" for path, value in phase.settings().items()])
    images = np.random.RandomState(0).randint(0, 256, (4 * RUN_BATCH, RES, RES, 3), np.uint8)
    config.train(config.build(), Memory([{"image": image} for image in images]).batch(RUN_BATCH),
                 name=phase.name)
    from dew.telemetry.instrumentation import compilation_cache_counts

    return compilation_cache_counts()


DIFFUSION_RL = {"flow_grpo": {"clip_range": 0.2, "beta": 0.0}}
"""The flow RL methods an online post-training cell trains a diffusion policy by
(tests/test_scenario_runs.py), each with its objective's settings as tests/test_diffusion_run_sources.py's
runs name them. DiffusionNFT's run records a guidance its objective does not take and refuses to train, so
it joins with row 9668's fix."""


def diffusion_rl_trained(cell: Mapping, root: str, phase: Mapping) -> dict[str, float]:
    """A tiny DiT trained over 8-pixel images by the cell's flow RL method (`DIFFUSION_RL`) at the cell's
    settings: each step a rollout of two of the policy's samples of each image over three time points, scored
    by their PSNR against it (`FlowRolloutConfig`), and the policy's update on them; its model's dtype left
    for the run's precision to derive (`diffusion_trained`'s run), trained in this process in `phase`
    (`Phase`). Its process's persistent-cache counts."""
    from test_quantization import RES, RUN_BATCH, diffusion_run

    from dew.config import ObjectiveConfig
    from dew.config.diffusion import FlowRolloutConfig
    from dew.diffusion.presets import Flow
    from dew.sampling import Euler
    from dew.telemetry.instrumentation import compilation_cache_counts

    phase = Phase(**phase)
    phase.entered()
    trainer = {"precision": numerics_of(cell["precision"]), "log_every": 1,
               "mesh": MeshSpec(fsdp=cell["devices"]), "layout": Layout(min_shard=64)}
    config = diffusion_run(Path(root) / "runs", RUN_BATCH, **trainer)
    fields = {name: value for name, value in config.model.fields.items() if name != "dtype"}
    method = cell["method"]
    config = dataclasses.replace(
        config, model=dataclasses.replace(config.model, fields=fields), preset=Flow(),
        objective=ObjectiveConfig(method, {**DIFFUSION_RL[method], "solver": Euler(), "steps": 2}),
        rollout=FlowRolloutConfig(reward="psnr", groups=2, steps=3, train_steps=2),
        trainer=dataclasses.replace(config.trainer, steps=STEPS, checkpoint_final=True))
    config = config.assigned([f"{path}={value}" for path, value in phase.settings().items()])
    objective = config.build()
    images = np.random.RandomState(0).randint(0, 256, (4 * RUN_BATCH, RES, RES, 3), np.uint8)
    config.train(objective, Memory([{"image": image} for image in images]).batch(RUN_BATCH), name=phase.name,
                 rollout=config.rollout.build(objective))
    return compilation_cache_counts()


def jepa_trained(cell: Mapping, root: str, phase: Mapping) -> dict[str, float]:
    """A JEPA encoder trained over images read as the cell reads, its modules computing in the cell's dtype,
    its mesh its devices' fsdp split, in this process in `phase` (`Phase`): every step's gradient finite;
    and, the cell's own run, its target encoder (the context encoder's average) moved from where it started
    and apart from the live encoder, and the trained encoder's representations of a held batch finite and
    not all one value. Its process's persistent-cache counts."""
    from test_jepa import GRID, RES, image_batches, images, make_encoder, make_predictor

    from dew.batch import part
    from dew.inputs import Field
    from dew.objectives.jepa import JepaObjective, MultiBlockMask
    from dew.telemetry.instrumentation import compilation_cache_counts

    phase = Phase(**phase)
    phase.entered()
    compute = {"dtype": numerics_of(cell["precision"]).dtypes.compute}
    objective = JepaObjective(make_encoder(**compute), make_predictor(**compute),
                              MultiBlockMask.for_grid(GRID, num_targets=4, scale=(0.15, 0.2)),
                              sample=Field("image", (RES, RES, 3)), momentum=(0.5, 0.5), momentum_steps=1)
    rows = Memory([dict(zip(("image", "label"), (image, label), strict=True))
                   for batch in itertools.islice(image_batches(8), 2)
                   for image, label in zip(batch["image"], batch["label"], strict=True)])
    tracker = RecordingTracker()
    trainer = Trainer(objective, optax.adam(1e-3), key=jax.random.key(0), mesh=MeshSpec(fsdp=cell["devices"]),
                      layout=Layout(min_shard=256), tracker=tracker, data_resume=phase.data_resume,
                      checkpoints=Checkpoints(str(Path(root) / "runs" / phase.name)))
    own = phase == CELL
    start = jax.tree.map(np.asarray, trainer.initial_state().ema) if own else None
    with run_precision(numerics_of(cell["precision"])):
        state = trainer.fit(rows.read(cell["read"]).batch(8), steps=STEPS, log_every=1, stop=phase.stop,
                            resume=phase.resume)
    assert_steps_finite(tracker.scalars, "JEPA")
    if not own:
        return compilation_cache_counts()
    target, began = (jax.tree.leaves(tree["params"]["context_encoder"]) for tree in (state.ema, start))
    live = jax.tree.leaves(state.variables["params"]["context_encoder"])
    assert target and any(not np.array_equal(np.asarray(a), b) for a, b in zip(target, began, strict=True))
    assert any(not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(target, live, strict=True))
    represented = np.asarray(objective.encode(part(state.variables, "context_encoder"), images(batch=4)),
                             np.float32)
    assert np.isfinite(represented).all() and np.ptp(represented) > 0
    return compilation_cache_counts()


GGUF = ("Q8_0", "Q4_0")
"""The GGUF encodings a quantized interchange cell writes its llama in (tests/test_scenario_runs.py's
`quantized_written`)."""
CODECS = Path(__file__).resolve().parent / "fixtures" / "codecs"
"""The quantized checkpoints a quantized interchange cell loads as their formats' authors wrote them, by their
directories here (tests/test_codec_checkpoints.py's)."""


def quantized_loaded(form: str, root: Path):
    """Format `form`'s quantized checkpoint loaded (`Pretrained.load`), its weights decoded into float32
    masters: the GGUF file of that encoding written under `root` (`GGUF`), or the codec checkpoint of that
    name (`CODECS`)."""
    from dew.interop import Pretrained

    held = {"dtype": "float32", "attention_impl": "reference"}
    if form in GGUF:
        return Pretrained.load(root, gguf_file="model.gguf", **held)
    return Pretrained.load(CODECS / form, **held)


def quantized_trained(cell: Mapping, root: str, phase: Mapping) -> dict[str, float]:
    """The cell's quantized checkpoint (`quantized_loaded`) trained on from its decoded weights, in this
    process in `phase` (`Phase`): every step's gradient finite. Its process's persistent-cache counts."""
    from dew.objectives.lm import LMObjective
    from dew.telemetry.instrumentation import compilation_cache_counts

    phase = Phase(**phase)
    phase.entered()
    source = quantized_loaded(cell["format"], Path(root))
    tracker = RecordingTracker()
    trainer = Trainer(LMObjective(source.model, 8, variables=jax.device_get(source.variables)),
                      optax.adam(1e-3), key=jax.random.key(0), tracker=tracker, data_resume=phase.data_resume,
                      checkpoints=Checkpoints(str(Path(root) / "runs" / phase.name)))
    rows = np.random.default_rng(1).integers(3, 30, (8, 9)).astype(np.int32)
    trainer.fit(Memory([{"text": row} for row in rows]).batch(8), steps=STEPS, log_every=1, stop=phase.stop,
                resume=phase.resume)
    assert_steps_finite(tracker.scalars, f"a {cell['format']} checkpoint")
    return compilation_cache_counts()
