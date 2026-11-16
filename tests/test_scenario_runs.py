"""Every scenario's covering set over its settings axes (docs/design/testing.md, settings and devices): each
cell one short run, a user's `dew train` or the scenario's own API in a fresh process that sees the cell's
devices, held to the contracts its flow touches (tests/scenario_contract.py) on the run it made. The
contracts a run of their own shows are held once a scenario, on its first cell (`first`): its run resumed,
its sampling rerun, and its first step under other rows."""

import dataclasses
import functools
import itertools
import json
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn
from placement_parity import layouts_agree
from recording import RecordingTracker
from run_kinds import cli_settings, lm, saved_record, scalars
from scenario_contract import (
    CELL,
    Called,
    Phase,
    Recorded,
    assert_resumes_onto_its_uninterrupted_run,
    assert_samples_from_its_cached_programs,
    assert_steps_finite,
    covering,
    first_steps,
)
from selection_margins import MARGIN, NARROW_MARGIN, clearing, top_margin
from test_mixers import KIND_FIELDS

import dew
from dew.checkpoints import Checkpoints
from dew.config import ObjectiveConfig, RunConfig
from dew.config.lm import LMRunConfig
from dew.data import Memory, presets
from dew.distributed.placement import Layout, MeshSpec
from dew.inference import TextGeneration
from dew.nn.kernels.generation import run_precision
from dew.nn.mixer_base import MixerBase
from dew.nn.precision import numerics_of
from dew.quantization import QuantizedDtype
from dew.registry import mixers, to_record
from dew.sampling import Sampling
from dew.training import Trainer

STEPS = 3
MODULE = "test_scenario_runs"
"""This module as a fresh process imports it, to call a run or a sampling made through Dew's API."""


@dataclass(frozen=True)
class Scenario:
    """A scenario's settings axes (the values each takes on the CPU tier) and its cell: one short run at the
    cell's settings under `root`, held to its claims. Where it trains, its training run at a cell
    (`trained`), which the resume claim pauses and resumes; where it samples, this module's function that
    samples a cell (`sampled`), called on the cell and the directory it samples from: the cell's trained run,
    or the cell's root, where it samples what it writes there itself (`sampled_from`). Where its runs read the
    LM data path, the axes over whose every value its first cell holds the rows claims (`laid`, none for its
    first cell alone; `LAID`)."""

    axes: Mapping[str, tuple]
    run: Callable[[Mapping, Path], None]
    trained: Callable[[Mapping, Path], Recorded | Called] | None = None
    sampled: str | None = None
    sampled_from: str = "run"
    laid: tuple[str, ...] | None = None


def settings(cell: Mapping, root: Path, more: Mapping[str, str] | None = None) -> dict[str, str]:
    """A cell's `dew train` settings (`run_kinds.cli_settings`): a short run logging every step, at the cell's
    precision over its devices' fsdp split, small leaves sharded, and `more`."""
    return cli_settings(root, "cell", {"trainer.steps": str(STEPS), "trainer.log_every": "1",
                                       "trainer.precision": cell.get("precision", "32-true"),
                                       "trainer.mesh.fsdp": str(cell.get("devices", 1)),
                                       "trainer.layout.min_shard": "64", **(more or {})})


EVALUATED = {"trainer.eval_every": "1"}
"""A run that holds validation data evaluates it every step: a resumed run then evaluates before the last
step it logs, whose persistent-cache counts show its evaluation's programs too."""


def called(function: str) -> Callable[[Mapping, Path], Called]:
    """A scenario's run made through Dew's API by this module's `function` in a fresh process that sees the
    cell's devices (`Called`)."""
    return lambda cell, root: Called(MODULE, function, cell, root, cell.get("devices", 1))


# --- dense language-model pretraining ----------------------------------------------------------------------

def lm_run(cell: Mapping, root: Path) -> LMRunConfig:
    """The shared LM run over a token store's documents (`run_kinds.lm`), laid as the cell lays its rows
    (padded, one document a row, where it names none) and read as the cell reads (memory where it names
    none), its model's dtype left for the run's precision to derive."""
    return lm(root, drawn=False, rows=cell.get("rows", "padded"), reading=cell.get("read", "memory"),
              dtype=None)


def assert_generates_as_its_checkpoint(run: Path) -> None:
    """Export then reload is identical: the run's task (`dew.pipeline`) generates, greedily, what a
    generation over its model and its final checkpoint's tree generates."""
    task = dew.pipeline(str(run))
    restored = TextGeneration(task.model, Checkpoints(str(run)).variables())
    greedy = Sampling(temperature=0)
    np.testing.assert_array_equal(task("the ", key=1, sampling=greedy).host().tokens,
                                  restored([list(b"the ")], 4, key=1, sampling=greedy).host().tokens)


def generated(cell: Mapping, run: str) -> np.ndarray:
    """The run's task's greedy generation of a prompt (`dew.pipeline`)."""
    return np.asarray(dew.pipeline(run)("the ", key=1, sampling=Sampling(temperature=0)).host().tokens)


def dense_lm_trained(cell: Mapping, root: Path) -> Recorded:
    """A decoder built and trained over a token store's documents packed or padded, read as the cell reads,
    at the cell's memory rung."""
    return Recorded(saved_record(lm_run(cell, root), root, "cell"),
                    settings(cell, root, {**EVALUATED, "max_new_tokens": "4",
                                          "model.remat": str(cell["memory"])}),
                    cell["devices"])


def dense_lm(cell: Mapping, root: Path) -> None:
    """Build a decoder, train it and reload its run: every step's gradient finite (`assert_steps_finite`), and
    the reloaded task generating what its final checkpoint's model generates."""
    run, _ = dense_lm_trained(cell, root).trained()
    assert_steps_finite(scalars(run).items(), "dense LM")
    assert_generates_as_its_checkpoint(run)


# --- a served decoder's continuation, held to its float64 forward -------------------------------------------

def float64_twin(served: TextGeneration, canonical: nn.Module) -> tuple[nn.Module, dict]:
    """The truth of a served task's logits: its canonical float model (`canonical`, the model the task serves
    before any quantization) in float64 on the reference kernel, and the served weights in float64, each
    quantized one its integers times its scales, their exact function (`test_quantization`'s split quantized
    tree)."""
    from dew.quantization import Annotated

    def value(node):
        if isinstance(node, Annotated):
            import qwix

            return np.asarray(qwix.dequantize(node.array.astype(jnp.float64)), np.float64)
        held = np.asarray(node)
        return held.astype(np.float64) if np.issubdtype(held.dtype, np.floating) else held

    with jax.enable_x64(new_val=True):
        weights = jax.tree.map(value, dict(served.variables),
                               is_leaf=lambda node: isinstance(node, Annotated))
    return canonical.clone(dtype=jnp.float64, attention_impl="reference"), weights


def assert_served_as_its_forward(served: TextGeneration, canonical: nn.Module, prompts: int, compute,
                                 steps: int = 4) -> None:
    """A served task's greedy continuation is its model's (testing.md: an argmax held at a stated margin): on
    `prompts` prompts of two tokens drawn so that every step of each one's float64 greedy walk
    (`model_support.greedy_walk` over `float64_twin`) clears the top-2 margin the served `compute` dtype's
    rounding moves a logit by (`selection_margins`: MARGIN, or NARROW_MARGIN in bfloat16), the served tokens
    are the walk's and every generated token's log-probability is finite."""
    from model_support import greedy_walk

    wide, weights = float64_twin(served, canonical)
    least = NARROW_MARGIN if jnp.dtype(compute) == jnp.bfloat16 else MARGIN

    def walked(prompt: np.ndarray) -> tuple[np.ndarray, float]:
        margins = []

        def margined(sequence, logits):
            margins.append(float(top_margin(logits, 1).min()))
            return logits

        with jax.enable_x64(new_val=True):
            sequence = greedy_walk(wide, weights, prompt[None], steps, margined)
        return sequence[0], min(margins)

    def drawn(seed: int) -> np.ndarray:
        return np.random.default_rng([0x5E7, seed]).integers(1, canonical.vocab_size, (prompts, 2)).astype(
            np.int32)

    rows = clearing(drawn, lambda rows: all(walked(row)[1] > least for row in rows))(0)
    result = served([list(row) for row in rows], steps, key=0, sampling=Sampling(temperature=0)).host()
    for index, row in enumerate(rows):
        length = int(result.lengths[index])
        assert length > 0
        np.testing.assert_array_equal(np.asarray(result.tokens)[index, :len(row) + length],
                                      walked(row)[0][:len(row) + length])
        assert np.isfinite(np.asarray(result.raw_log_probs)[index, :length]).all()


# --- pretrained fine-tuning ---------------------------------------------------------------------------------

def pretrained_trained(cell: Mapping, root: Path) -> Recorded:
    """A published decoder (a local HF-layout export; its checkpoint decides its geometry and its remat)
    fine-tuned in its full weights or a LoRA over them, quantization-aware where the cell trains so."""
    from test_lm_recipe import export_tiny_decoder

    from dew.lora import LoRA
    from dew.quantization import Quantization

    shared = lm_run(cell, root)
    config = LMRunConfig(data=shared.data, validation=shared.validation,
                         pretrained=str(export_tiny_decoder(root / "checkpoint")),
                         lora=LoRA(rank=4, modules=("q_proj",)) if cell["adapter"] == "lora" else None)
    if cell["quantization"] == "qat":
        config = dataclasses.replace(config, trainer=dataclasses.replace(config.trainer,
                                                                        quantization=Quantization()))
    return Recorded(saved_record(config, root, "cell"),
                    settings(cell, root, {**EVALUATED, "max_new_tokens": "4"}), cell["devices"])


def pretrained(cell: Mapping, root: Path) -> None:
    """Load a published decoder, fine-tune it and reload the run: every step's gradient finite, and its task
    generating what its checkpoint generates; served quantized after training (PTQ, weight-only) where the
    cell serves so, its continuation its float64 forward's (`assert_served_as_its_forward`)."""
    from dew.quantization import Quantization

    run, _ = pretrained_trained(cell, root).trained()
    assert_steps_finite(scalars(run).items(), "pretrained fine-tuning")
    assert_generates_as_its_checkpoint(run)
    if cell["quantization"] == "ptq":
        task = dew.pipeline(str(run))
        served = task.quantized(Quantization(weight_only=True))
        numerics = numerics_of(cell["precision"])
        with run_precision(numerics):
            assert_served_as_its_forward(served, task.model, 1, numerics.dtypes.compute)


# --- diffusion language models -----------------------------------------------------------------------------

MASK = 255
"""The masked diffusion LM's mask id: the byte vocabulary's last."""
PROMPT = [5, 6, 7]


def masked_trained(cell: Mapping, root: Path) -> Recorded:
    """A masked diffusion LM over the shared LM data path, its documents packed or padded, a bidirectional
    decoder masking at the vocabulary's last id."""
    config = dataclasses.replace(lm_run(cell, root),
                                 objective=ObjectiveConfig("masked_diffusion", {"ema_decay": None}))
    return Recorded(saved_record(config, root, "cell"),
                    settings(cell, root, {**EVALUATED, "max_new_tokens": "0", "model.causal": "False",
                                          "model.mask_token_id": str(MASK)}), cell["devices"])


def unmasked(cell: Mapping, run: str) -> np.ndarray:
    """The run's sampler's continuation of a prompt (`dew.pipeline`), two denoising steps."""
    return np.asarray(dew.pipeline(run)([PROMPT], 4, key=0, steps=2).host().tokens)


def masked_diffusion(cell: Mapping, root: Path) -> None:
    """Train a masked diffusion LM and reload its run to a sampler: every step's gradient finite, and its
    sample keeping its prompt and leaving no position masked."""
    run, _ = masked_trained(cell, root).trained()
    assert_steps_finite(scalars(run).items(), "masked diffusion LM")
    tokens = unmasked(cell, str(run))
    assert list(tokens[0, :len(PROMPT)]) == PROMPT and (tokens[0, len(PROMPT):len(PROMPT) + 4] != MASK).all()


# --- hybrid and state-space models --------------------------------------------------------------------------

LENGTHS = (16, 32)
"""The two lengths a tiny decoder of each kind is built for, whose caches `recurrent` compares."""


@functools.cache
def recurrent() -> tuple[str, ...]:
    """The registry's recurrent mixer kinds (`mixers.paths`): each whose tiny decoder (`test_mixers.tiny`, at
    the kind's fields there, `KIND_FIELDS`) states a cache (`cache_spec`, each holder's own statement of it)
    of the same leaf shapes built for either of `LENGTHS`: a state of its own size, where keys and values
    grow with the sequence."""
    from test_mixers import tiny

    def shapes(kind: str, length: int):
        fields, overrides = KIND_FIELDS.get(kind, ({}, {}))
        model = tiny(mixer=mixers.build(kind, **fields), max_seq_len=length, **overrides)
        variables = nn.unbox(jax.eval_shape(model.init, jax.random.key(0), jnp.zeros((1, 8), jnp.int32)))
        spec = model.apply(variables, 1, method=lambda module, rows: module.cache_spec(rows))
        return jax.tree.map(lambda leaf: tuple(leaf.shape), spec)

    stated = {kind: [shapes(kind, length) for length in LENGTHS] for kind in sorted(mixers.paths)}
    # A kind that states no cache (a token-wise MLP's) holds no state to decode from.
    return tuple(kind for kind, (short, long) in stated.items() if jax.tree.leaves(short) and short == long)


def mixer_settings(kind: str) -> dict[str, str]:
    """The `dew train` settings of a decoder of mixer `kind`: the kind's record (`to_record`) at its tiny
    fields (`test_mixers.KIND_FIELDS`), and the model's fields the kind needs beside it."""
    fields, overrides = KIND_FIELDS.get(kind, ({}, {}))
    return {"model.mixer": json.dumps(to_record(mixers.build(kind, **fields), MixerBase | None)),
            **{f"model.{name}": json.dumps(value) for name, value in overrides.items()}}


def hybrid_trained(cell: Mapping, root: Path) -> Recorded:
    """A decoder of the cell's recurrent mixer trained through the shared components."""
    return Recorded(saved_record(lm_run(cell, root), root, "cell"),
                    settings(cell, root, {**EVALUATED, "max_new_tokens": "4",
                                          **mixer_settings(cell["mixer"])}),
                    cell["devices"])


def hybrid(cell: Mapping, root: Path) -> None:
    """Train a decoder of the cell's recurrent mixer and decode it from its run: every step's gradient finite,
    and its task generating what its checkpoint generates."""
    run, _ = hybrid_trained(cell, root).trained()
    assert_steps_finite(scalars(run).items(), f"a {cell['mixer']} decoder")
    assert_generates_as_its_checkpoint(run)


# --- distributed training -----------------------------------------------------------------------------------

def distributed_trained(cell: Mapping, root: Path) -> Recorded:
    """A dense decoder or a mixture of experts trained over a mesh of the cell's split, on two devices."""
    moe = {"model.mixture": json.dumps({"experts": 4, "top_k": 2})} if cell["model"] == "moe" else {}
    mesh = {"trainer.mesh.fsdp": "2"} if cell["mesh"] == "fsdp" else {"trainer.mesh.fsdp": "1",
                                                                      "trainer.mesh.expert": "2"}
    return Recorded(saved_record(lm_run(cell, root), root, "cell"),
                    settings(cell, root, {**EVALUATED, "max_new_tokens": "4", **moe, **mesh}), 2)


def distributed(cell: Mapping, root: Path) -> None:
    """Train a dense decoder or a mixture of experts over a mesh, its experts sharded where the cell shards
    them: every step's gradient finite and its run reloading."""
    run, _ = distributed_trained(cell, root).trained()
    assert_steps_finite(scalars(run).items(), f"a {cell['model']} decoder over {cell['mesh']}")
    assert_generates_as_its_checkpoint(run)


# --- diffusion training -------------------------------------------------------------------------------------

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


def imaged(cell: Mapping, run: str) -> np.ndarray:
    """The run's pipeline's sample (`dew.pipeline`), two steps at one key."""
    return np.asarray(dew.pipeline(run)("", steps=2, key=0).host().images)


def diffusion(cell: Mapping, root: Path) -> None:
    """Train an image model, checkpointing it, and reload its run: every step's gradient finite, and the
    reloaded pipeline sampling what a pipeline over its final checkpoint's tree samples."""
    run, _ = called("diffusion_trained")(cell, root).trained()
    assert_steps_finite(scalars(run).items(), "diffusion training")
    task = dew.pipeline(str(run))
    restored = task.bind(Checkpoints(str(run)).variables())
    np.testing.assert_array_equal(np.asarray(task("", steps=2, key=0).host().images),
                                  np.asarray(restored("", steps=2, key=0).host().images))


# --- representation learning --------------------------------------------------------------------------------

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


def jepa(cell: Mapping, root: Path) -> None:
    """Representation learning: a JEPA encoder trained and its representations read, in a process of the
    cell's devices (`jepa_trained`)."""
    called("jepa_trained")(cell, root).trained()


# --- online post-training -----------------------------------------------------------------------------------

def online_rl_trained(cell: Mapping, root: Path) -> Recorded:
    """The recorded GRPO run (`rl_support.recorded_grpo_config`), its rollout's method the cell's (GRPO or
    RLOO), collecting rollouts from its policy and scoring them by the reward its import path names, its
    model's dtype left for the run's precision to derive."""
    import rl_support

    config = rl_support.recorded_grpo_config(root, "--rollout.method", cell["method"])
    fields = {name: value for name, value in config.model.fields.items() if name != "dtype"}
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, fields=fields))
    return Recorded(saved_record(config, root, "cell"), settings(cell, root), cell["devices"],
                    trust=(rl_support.__name__,))


def online_rl(cell: Mapping, root: Path) -> None:
    """Language RL: collect rollouts, score them, update the policy, `dew train` as a user runs it: every
    step's gradient finite. The run's scalars show no estimator, so which method it ran is not read here
    (the methods' own tests hold each)."""
    run, _ = online_rl_trained(cell, root).trained()
    assert_steps_finite(scalars(run).items(), f"online {cell['method']}")


# --- generation and serving ---------------------------------------------------------------------------------

PROMPTS = (1, 3)
"""A served batch's prompts: one alone, or three together."""


def serving_task(cell: Mapping) -> tuple[TextGeneration, nn.Module]:
    """The cell's served task and the canonical float model it serves: a decoder computing in the cell's
    dtype, quantized after training (PTQ, weight-only) where the cell serves so."""
    from model_support import serving_task as tiny_task

    from dew.quantization import Quantization

    task = tiny_task(sampling=Sampling(temperature=0))
    canonical = task.model.clone(dtype=numerics_of(cell["precision"]).dtypes.compute)
    task = TextGeneration(canonical, task.variables, sampling=Sampling(temperature=0))
    if cell["quantization"] is not None:
        task = task.quantized(Quantization(dtype=cell["quantization"], weight_only=True))
    return task, canonical


def served(cell: Mapping, root: str) -> np.ndarray:
    """The cell's served greedy continuation of its batch's prompts."""
    task, _ = serving_task(cell)
    prompts = [[3, 4], [5, 6], [8, 9]][:cell["prompts"]]
    with run_precision(numerics_of(cell["precision"])):
        return np.asarray(task(prompts, 4, key=0).host().tokens)


def serving(cell: Mapping, root: Path) -> None:
    """Generation and serving: a decoder computing in the cell's dtype, quantized after training as the cell
    serves, over one prompt or a batch of three: its greedy continuation its float64 forward's
    (`assert_served_as_its_forward`)."""
    task, canonical = serving_task(cell)
    with run_precision(numerics_of(cell["precision"])):
        assert_served_as_its_forward(task, canonical, cell["prompts"], canonical.dtype)


# --- quantized interchange ----------------------------------------------------------------------------------

def gguf_written(cell: Mapping, root: Path) -> Path:
    """A llama written as a GGUF file of the cell's quantization under `root`: its directory."""
    import gguf
    from test_gguf import hf_tensors, write_gguf

    rng = np.random.default_rng(0)
    write_gguf(root / "model.gguf", "llama", getattr(gguf.GGMLQuantizationType, cell["format"]),
               hf_tensors("llama", rng), rng)
    return root


def gguf_loaded(root: Path):
    """The llama of the GGUF file under `root` (`gguf_written`), loaded (`Pretrained.load`)."""
    from dew.interop import Pretrained

    return Pretrained.load(root, dtype="float32", attention_impl="reference", gguf_file="model.gguf")


def gguf_run(cell: Mapping, root: Path) -> Called:
    """A GGUF file of the cell's quantization written under `root`, and its llama's training run there
    (`gguf_trained`), in a fresh process."""
    gguf_written(cell, root)
    return called("gguf_trained")(cell, root)


def gguf_trained(cell: Mapping, root: str, phase: Mapping) -> dict[str, float]:
    """A GGUF file's llama trained on from its dequantized weights, in this process in `phase` (`Phase`):
    every step's gradient finite. Its process's persistent-cache counts."""
    from dew.objectives.lm import LMObjective
    from dew.telemetry.instrumentation import compilation_cache_counts

    phase = Phase(**phase)
    phase.entered()
    source = gguf_loaded(Path(root))
    tracker = RecordingTracker()
    trainer = Trainer(LMObjective(source.model, 8, variables=jax.device_get(source.variables)),
                      optax.adam(1e-3), key=jax.random.key(0), tracker=tracker, data_resume=phase.data_resume,
                      checkpoints=Checkpoints(str(Path(root) / "runs" / phase.name)))
    rows = np.random.default_rng(1).integers(3, 30, (8, 9)).astype(np.int32)
    trainer.fit(Memory([{"text": row} for row in rows]).batch(8), steps=STEPS, log_every=1, stop=phase.stop,
                resume=phase.resume)
    assert_steps_finite(tracker.scalars, f"a {cell['format']} GGUF's llama")
    return compilation_cache_counts()


def gguf_served(cell: Mapping, root: str) -> np.ndarray:
    """A GGUF file of the cell's quantization written under `root`, its llama served: its greedy continuation
    of a prompt."""
    source = gguf_loaded(gguf_written(cell, Path(root)))
    task = TextGeneration(source.model, source.variables, sampling=Sampling(temperature=0))
    return np.asarray(task([[3, 4, 5]], 4, key=0).host().tokens)


def quantized_interchange(cell: Mapping, root: Path) -> None:
    """Quantized interchange: a llama written as a GGUF file of the cell's quantization, loaded, then trained
    on from its dequantized weights (`gguf_trained`) or served, its greedy continuation its float64
    forward's."""
    if cell["use"] == "train":
        gguf_run(cell, root).trained()
        return
    source = gguf_loaded(gguf_written(cell, root))
    task = TextGeneration(source.model, source.variables, sampling=Sampling(temperature=0))
    assert_served_as_its_forward(task, source.model, 1, source.model.dtype)


# --- published diffusion pipelines --------------------------------------------------------------------------

def published_task(cell: Mapping, root: Path):
    """The cell's released pipeline, extracted under `root` and loaded at the cell's dtype."""
    import tarfile

    from test_pipeline_streaming import PIPELINES

    from dew.interop.pretrained import Pretrained

    archive, inside = PIPELINES[cell["family"]]
    with tarfile.open(Path(__file__).resolve().parents[1] / f"tests/fixtures/{archive}.tar.xz") as opened:
        opened.extractall(root / archive, filter="data")
    loaded = Pretrained.load(str(root / archive / inside), dtype=cell["dtype"], attention_impl="xla")
    return loaded.text_to_image()


def pipeline_sampled(cell: Mapping, root: str) -> np.ndarray:
    """The cell's released pipeline's sample of a prompt, two steps at one key."""
    return np.asarray(published_task(cell, Path(root))(["a red bird"], steps=2, key=3).host().images)


def published_pipeline(cell: Mapping, root: Path) -> None:
    """Published diffusion pipelines: a tiny released pipeline loaded and sampled end to end at the cell's
    family, its images finite and the same each call at one key."""
    task = published_task(cell, root)
    first, second = (np.asarray(task(["a red bird"], steps=2, key=3).host().images, np.float32)
                     for _ in range(2))
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def _pipelines() -> tuple[str, ...]:
    from test_pipeline_streaming import PIPELINES

    return tuple(PIPELINES)


PRECISIONS = ("32-true", "bf16-mixed")
"""The precisions a scenario's CPU cells train at (fp8 needs widths a multiple of 128)."""

SCENARIOS: Mapping[str, Scenario] = {
    "Dense language-model pretraining": Scenario(
        {"precision": PRECISIONS, "rows": ("padded", "packed"), "memory": (None, "full"), "devices": (1, 2),
         "read": ("memory", "indexed", "streamed")}, dense_lm, dense_lm_trained, "generated", laid=()),
    "Pretrained fine-tuning": Scenario(
        {"precision": PRECISIONS, "adapter": ("full", "lora"), "quantization": (None, "qat", "ptq"),
         "devices": (1, 2)}, pretrained, pretrained_trained, "generated", laid=()),
    "Diffusion language models": Scenario(
        {"precision": PRECISIONS, "rows": ("padded", "packed"), "devices": (1, 2)}, masked_diffusion,
        masked_trained, "unmasked"),
    "Hybrid and state-space models": Scenario(
        {"mixer": recurrent(), "precision": PRECISIONS, "rows": ("padded", "packed"), "devices": (1, 2)},
        hybrid, hybrid_trained, "generated", laid=("mixer",)),
    "Distributed training": Scenario(
        {"model": ("dense", "moe"), "mesh": ("fsdp", "expert"), "precision": PRECISIONS}, distributed,
        distributed_trained, "generated", laid=()),
    "Diffusion training": Scenario(
        {"precision": PRECISIONS, "quantization": (None, "qat"), "memory": (False, "full"),
         "devices": (1, 2)}, diffusion, called("diffusion_trained"), "imaged"),
    "Representation learning": Scenario(
        {"precision": PRECISIONS, "devices": (1, 2), "read": ("memory", "indexed", "streamed")}, jepa,
        called("jepa_trained")),
    "Online post-training": Scenario(
        {"method": ("grpo", "rloo"), "precision": PRECISIONS, "devices": (1, 2)}, online_rl,
        online_rl_trained),
    "Generation and serving": Scenario(
        {"precision": PRECISIONS, "quantization": (None, *typing.get_args(QuantizedDtype)),
         "prompts": PROMPTS}, serving, sampled="served", sampled_from="root"),
    "Quantized interchange": Scenario(
        {"format": ("Q8_0", "Q4_0"), "use": ("train", "serve")}, quantized_interchange, gguf_run,
        "gguf_served", "root"),
    "Published diffusion pipelines": Scenario(
        {"family": _pipelines(), "dtype": ("float32", "bfloat16")}, published_pipeline,
        sampled="pipeline_sampled", sampled_from="root"),
}
"""Every scenario of docs/design/testing.md's table this file runs, by its name there."""


def _named(cell: Mapping) -> str:
    return ",".join(f"{axis}={value}" for axis, value in cell.items())


CELLS = [pytest.param(name, cell, id=f"{name}-{_named(cell)}")
         for name, scenario in SCENARIOS.items() for cell in covering(scenario.axes)]


@pytest.mark.parametrize(("name", "cell"), CELLS)
def test_a_scenarios_cell_holds_the_contracts_its_flow_touches(name, cell, tmp_path):
    """One cell of a scenario's covering set: its run at the cell's settings, held to its claims."""
    SCENARIOS[name].run(cell, tmp_path)


def first(name: str) -> dict:
    """A scenario's first cell, where it holds the claims a run of their own shows."""
    return covering(SCENARIOS[name].axes)[0]


@pytest.mark.parametrize("name", [name for name, scenario in SCENARIOS.items() if scenario.trained])
def test_a_scenarios_run_resumed_ends_on_its_uninterrupted_run_from_its_compiled_programs(name, tmp_path):
    """Resume equals the uninterrupted run, and a resumed run reuses its compiled programs (testing.md's
    contracts), once a scenario on its first cell (`assert_resumes_onto_its_uninterrupted_run`)."""
    assert_resumes_onto_its_uninterrupted_run(SCENARIOS[name].trained(first(name), tmp_path), tmp_path)


@pytest.mark.parametrize("name", [name for name, scenario in SCENARIOS.items() if scenario.sampled])
def test_a_scenarios_sampling_reruns_from_its_compiled_programs(name, tmp_path):
    """A repeated run reuses its compiled programs, sampling (testing.md's contracts), once a scenario on its
    first cell: its sampling of the cell's trained run, or of what it writes under the cell's root, twice in
    fresh processes over one persistent cache (`assert_samples_from_its_cached_programs`)."""
    scenario, cell = SCENARIOS[name], first(name)
    source = scenario.trained(cell, tmp_path).trained()[0] if scenario.sampled_from == "run" else tmp_path
    assert_samples_from_its_cached_programs(MODULE, scenario.sampled, cell, str(source), root=tmp_path)


# --- a first step however its rows are laid or split ------------------------------------------------------

LENGTH, DOCUMENTS = 11, 6
"""A drawn document's tokens and a draw's documents: three documents fill a row of 32 inputs and their
targets whole, so packed rows hold no padding, and a row of one document's own length none either."""

ROWS = {"packed": ({"pack": "first-fit", "bins": 2}, 3 * LENGTH - 1, DOCUMENTS // 3),
        "separate": ({"pack": "first-fit", "segments": 1}, 3 * LENGTH - 1, DOCUMENTS),
        "unpadded": ({"pack": "first-fit", "segments": 1}, LENGTH - 1, DOCUMENTS)}
"""Each way the LM data path (`presets.lm`) lays a draw's documents: its pack, its rows' inputs and the rows
it makes of them. Packed beside each other; one a row, right-padded; one a row of its own length."""

LAID = {"packed": ("packed rows equal their documents run separately", ("packed", 1, 1),
                   ("separate", 1, 1)),
        "padded": ("right-padded rows equal unpadded ones on the real tokens", ("separate", 1, 1),
                   ("unpadded", 1, 1)),
        "split": ("rows split over two devices equal one device's read in two microbatches", ("packed", 2, 1),
                  ("packed", 1, 2))}
"""Each claim of a scenario's rows (testing.md's contracts): its statement, the arm it holds and its
reference, each the way its rows are laid, the devices they are split over and the microbatches a device
reads them in. The suite's CPU sums a whole batch's gradient reduction in float64 and rounds it once
(`WIDE_CPU_ARITHMETIC`'s widened reductions, which the gate keeps, `arm_arithmetic.summed_as_run`), where any
split of it, over devices or microbatches, rounds each part and their sum: no split holds a leaf whose error
is that one rounding (Mamba's A_log, three roundings to one, row 6459), so the rows over two devices are held
to one device reading them as two microbatches, the same split and the same roundings."""


def laid(rows: str, seed) -> dict:
    """Draw `seed`'s documents (`DOCUMENTS` of `LENGTH` token ids each) as the LM data path lays them `rows`'s
    way, every row in one batch."""
    from test_presets import first_batch

    pack, inputs, count = ROWS[rows]
    rng = np.random.default_rng([0x1A1D, int(seed)])
    documents = Memory([{"text": rng.integers(1, 250, LENGTH).astype(np.int32)} for _ in range(DOCUMENTS)])
    return first_batch(documents.pipe(presets.lm, None, seq_len=inputs, batch=count, **pack))


LAID_CELLS = [pytest.param(name, {**first(name), **dict(zip(scenario.laid, values, strict=True))}, site,
                           id=site)
              for name, scenario in SCENARIOS.items() if scenario.laid is not None
              for values in itertools.product(*(scenario.axes[axis] for axis in scenario.laid))
              for site in ["-".join([name.split()[0].lower(), *map(str, values)])]]
"""The scenarios whose runs read the LM data path, by their first cells (`Scenario.laid`): a hybrid's at
each recurrent mixer, each of which resets its state at a document's start in its own arithmetic, a
distributed run's its dense decoder's (a mixture's routing is a selection, which the rows claims' draws do not
hold clear of its boundaries, `selection_margins`; the parallelism matrix holds its layouts). The masked
diffusion LM draws one noise time a row, every document packed in it sharing it, where each run alone draws
its own, and a mask over a row's every position, a row right-padded drawing other masks for its real tokens
than its own length's: its rows are held over devices alone, by the parallelism matrix's sequence and fsdp
cells (tests/test_parallelism_matrix.py, dense_mdlm)."""

LAID_SEEDS = {"dense": {"packed": 8, "padded": 128, "split": 256},
              "pretrained": {"packed": 128, "padded": 128, "split": 128},
              "distributed": {"packed": 8, "padded": 128, "split": 256},
              "hybrid-gated_delta_net": {"packed": 128, "padded": 128, "split": 128},
              "hybrid-kimi_delta_attention": {"packed": 128, "padded": 128, "split": 256},
              "hybrid-lightning_attention": {"packed": 32, "padded": 128, "split": 256},
              "hybrid-mamba": {"packed": 8, "padded": 64, "split": 256},
              "hybrid-mamba2": {"packed": 16, "padded": 256, "split": 256},
              "hybrid-mamba3": {"packed": 256, "padded": 256, "split": 256},
              "hybrid-memory": {"packed": 256, "padded": 256, "split": 256},
              "hybrid-short_conv": {"packed": 8, "padded": 64, "split": 128}}
"""Each site's declared check-1 seeds, by its cell and claim: `seeds_for` paired, on the CPU lane at
5b91fd892, pretrained's and distributed's at 88edfcc4a (armada, 2026-11-15)."""


CHECK1_SHARDS = {
    "test_a_scenarios_first_step_is_the_same_however_its_rows_are_laid[hybrid-mamba-split]":
        {"seeds": 256, "per_task": 64},
    "test_a_scenarios_first_step_is_the_same_however_its_rows_are_laid[hybrid-kimi_delta_attention-split]":
        {"seeds": 256, "per_task": 64},
    "test_a_scenarios_first_step_is_the_same_however_its_rows_are_laid[hybrid-mamba2-padded]":
        {"seeds": 256, "per_task": 64},
    "test_a_scenarios_first_step_is_the_same_however_its_rows_are_laid[hybrid-mamba2-split]":
        {"seeds": 256, "per_task": 64},
    "test_a_scenarios_first_step_is_the_same_however_its_rows_are_laid[hybrid-memory-split]":
        {"seeds": 256, "per_task": 64},
    "test_a_scenarios_first_step_is_the_same_however_its_rows_are_laid[hybrid-mamba3-split]":
        {"seeds": 256, "per_task": 64},
}
"""The rows sites whose check-1 seeds run as map tasks (`check1_shards`), by node id, a literal the CI plan
reads as it reads a mark's (tools/armada/ci.py): each takes 62 to 118 s whole on the CPU lane (main's CI
durations) and 122 to 144 s under a loaded train's (queue train 922, row 6885), past the group target, its
setup some 15 to 20 s of it, so 64 seeds a task keeps each well inside it."""


def laid_sites(test: str) -> list:
    """Every rows site (`LAID_CELLS`, each claim of `LAID`), as `test`'s parameters: each marked to run its
    seeds as map tasks where `CHECK1_SHARDS` names its node, its declared count `LAID_SEEDS`'."""
    def param(cell, claim: str):
        name, values, site = cell.values
        held = CHECK1_SHARDS.get(f"{test}[{site}-{claim}]")
        assert held is None or held["seeds"] == LAID_SEEDS[site][claim], (site, claim, held)
        return pytest.param(name, values, site, claim, id=f"{site}-{claim}",
                            marks=[pytest.mark.check1_shards(**held)] if held else [])

    return [param(cell, claim) for cell in LAID_CELLS for claim in LAID]


@pytest.mark.parametrize(("name", "cell", "site", "claim"),
                         laid_sites("test_a_scenarios_first_step_is_the_same_however_its_rows_are_laid"))
def test_a_scenarios_first_step_is_the_same_however_its_rows_are_laid(name, cell, site, claim, tmp_path):
    """Packed rows equal their documents run separately, right-padded rows equal unpadded ones on the real
    tokens, and rows split over devices equal one device's read in as many microbatches (testing.md's
    contracts, `LAID`): the scenario's first step, its loss and the gradient it hands the optimizer, as its
    own run builds its objective (`RunConfig.prepare`, at the cell's settings, its model in float64 for the
    truth) over the LM data path's batches of drawn documents (`laid`), on par by check 1
    (`placement_parity.layouts_agree`) over the site's declared seeds (`LAID_SEEDS`)."""
    recorded = SCENARIOS[name].trained(cell, tmp_path)
    run = RunConfig.load(str(recorded.record))
    given = [f"{path}={value}" for path, value in {**recorded.settings, **Phase().settings()}.items()]

    def objective_of(wide: bool):
        return run.assigned([*given, f"model.dtype={'float64' if wide else 'float32'}"]).prepare().objective

    stated, arm, reference = LAID[claim]
    layouts_agree(first_steps(objective_of, laid, 64, reference), lambda seed: (np.int32(seed),),
                  f"{name}: {stated}", arm, reference, seeds=LAID_SEEDS[site][claim])
