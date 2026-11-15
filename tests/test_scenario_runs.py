"""Every scenario's covering set over its settings axes (docs/design/testing.md, settings and devices): each
cell one short run, a user's `dew train` in a fresh process that sees the cell's devices, held to the
contracts its flow touches (tests/scenario_contract.py) on the run it made."""

import dataclasses
import itertools
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
import optax
import pytest
from lane_environment import without_flags
from process_support import fresh
from recording import RecordingTracker
from run_kinds import saved_record, scalars, trained_record
from scenario_contract import assert_steps_finite, covering
from token_support import write_token_corpus

import dew
from dew.checkpoints import Checkpoints
from dew.config import ObjectiveConfig
from dew.config.lm import LMRunConfig
from dew.data import Loading, Memory, presets
from dew.data.sources.tokens import TokenFiles
from dew.distributed.placement import Layout, MeshSpec
from dew.inference import TextGeneration
from dew.nn.kernels.generation import run_precision
from dew.nn.precision import numerics_of
from dew.sampling import Sampling
from dew.training import Trainer

DEVICES = "--xla_force_host_platform_device_count"
STEPS = 3
SEQ = 32


@dataclass(frozen=True)
class Scenario:
    """A scenario's settings axes (the values each takes on the CPU tier) and its cell: one short run at the
    cell's settings under `root`, held to its claims."""

    axes: Mapping[str, tuple]
    run: Callable[[Mapping, Path], None]


def in_fresh_process(devices: int, function: str, *arguments) -> None:
    """This module's `function` called on `arguments` (JSON's values) in a fresh process that sees `devices`
    CPU devices: a run's device count is its process's, fixed as its backend opens."""
    flags = f"{without_flags(os.environ.get('XLA_FLAGS', ''), DEVICES)} {DEVICES}={devices}".strip()
    script = ("import json, sys\nimport test_scenario_runs\n"
              "getattr(test_scenario_runs, sys.argv[1])(*json.loads(sys.argv[2]))\nprint(json.dumps(0))")
    assert fresh(script, function, json.dumps(arguments), flags=flags, platforms="cpu") == 0


def common(cell: Mapping, root: Path) -> dict[str, str]:
    """Every scenario's run settings at the cell's: a short run logging each step, its checkpoint at its last,
    its precision and its devices' fsdp split, small leaves sharded."""
    return {"trainer.name": "cell", "trainer.checkpoint_dir": str(root / "runs"), "trainer.steps": str(STEPS),
            "trainer.log_every": "1", "trainer.eval_every": "None", "trainer.checkpoint_every": "None",
            "trainer.multi_host": "False", "trainer.compilation_cache_dir": "None",
            "trainer.precision": cell["precision"], "trainer.mesh.fsdp": str(cell["devices"]),
            "trainer.layout.min_shard": "64"}


# --- dense language-model pretraining ----------------------------------------------------------------------

def corpus(root: Path):
    """A token store of documents (an end of text every seventh id), a training split and a held one."""
    rng = np.random.RandomState(0)
    train, val = (rng.randint(1, 250, count).astype(np.uint8) for count in (40 * SEQ, 8 * SEQ))
    train[::7] = val[::7] = 0
    return write_token_corpus(root / "tokens", train, val, eos_id=0)


def lm_data(cell: Mapping, root: Path, rows: str = "padded") -> dict:
    """An LM run's data and validation over `corpus`, its documents packed beside each other or one a row
    padded (`presets.lm`'s `segments=1`) as the cell lays its rows (`rows` where it names none), read as the
    cell reads (memory where it names none)."""
    tokens = corpus(root)
    # Packed: documents first-fit beside each other, segment ids apart; padded: one document a row, padded.
    layout = {"pack": "first-fit", "bins": 2} if cell.get("rows", rows) == "packed" else \
        {"pack": "first-fit", "segments": 1}

    def windows(store: str):
        chain = TokenFiles(store).pipe(presets.lm, None, seq_len=SEQ, batch=8, **layout)
        return chain.read(cell.get("read", "memory")).loading(Loading(workers=0))

    return {"data": windows(tokens.train), "validation": windows(tokens.val)}


def assert_generates_as_its_checkpoint(run: Path) -> None:
    """Export then reload is identical: the run's task (`dew.pipeline`) generates, greedily, what a
    generation over its model and its final checkpoint's tree generates."""
    task = dew.pipeline(str(run))
    restored = TextGeneration(task.model, Checkpoints(str(run)).variables())
    greedy = Sampling(temperature=0)
    np.testing.assert_array_equal(task("the ", key=1, sampling=greedy).host().tokens,
                                  restored([list(b"the ")], 4, key=1, sampling=greedy).host().tokens)


TINY = {"model.emb_features": "16", "model.num_layers": "1", "model.num_heads": "2", "max_new_tokens": "4"}


def dense_lm(cell: Mapping, root: Path) -> None:
    """Build a decoder, train it over a token store's documents packed or padded, read as the cell reads,
    and reload its run: every step's gradient finite (`assert_steps_finite`), and the reloaded task
    generating what its final checkpoint's model generates."""
    run = trained_record(saved_record(LMRunConfig(**lm_data(cell, root)), root, "cell"),
                         {**common(cell, root), **TINY, "model.remat": str(cell["memory"])}, cell["devices"])
    assert_steps_finite(scalars(run).items(), "dense LM")
    assert_generates_as_its_checkpoint(run)


def pretrained(cell: Mapping, root: Path) -> None:
    """Load a published decoder (a local HF-layout export; its checkpoint decides its geometry and its
    remat, the planner's rung its memory), fine-tune its full weights or a LoRA over them,
    quantization-aware where the cell trains so, and reload the run: every step's gradient finite, and its
    task generating what its checkpoint generates, served quantized after training (PTQ) where the cell
    serves so."""
    from test_lm_recipe import export_tiny_decoder

    from dew.lora import LoRA
    from dew.quantization import Quantization

    config = LMRunConfig(**lm_data(cell, root), pretrained=str(export_tiny_decoder(root / "checkpoint")),
                         lora=LoRA(rank=4, modules=("q_proj",)) if cell["adapter"] == "lora" else None)
    if cell["quantization"] == "qat":
        config = dataclasses.replace(config, trainer=dataclasses.replace(config.trainer,
                                                                        quantization=Quantization()))
    run = trained_record(saved_record(config, root, "cell"), {**common(cell, root), "max_new_tokens": "4"},
                         cell["devices"])
    assert_steps_finite(scalars(run).items(), "pretrained fine-tuning")
    assert_generates_as_its_checkpoint(run)
    if cell["quantization"] == "ptq":
        served = dew.pipeline(str(run)).quantized(Quantization())
        assert served("the ", key=1, sampling=Sampling(temperature=0)).host().tokens.shape[-1] > len("the ")


MASK = 255
"""The masked diffusion LM's mask id: the byte vocabulary's last."""


def masked_diffusion(cell: Mapping, root: Path) -> None:
    """Train a masked diffusion LM over the shared LM data path, its documents packed or padded, a
    bidirectional decoder masking at the vocabulary's last id: every step's gradient finite and its run
    reloading to a sampler."""
    config = LMRunConfig(**lm_data(cell, root),
                         objective=ObjectiveConfig("masked_diffusion", {"ema_decay": None}))
    run = trained_record(saved_record(config, root, "cell"),
                         {**common(cell, root), **TINY, "max_new_tokens": "0", "model.causal": "False",
                          "model.mask_token_id": str(MASK)}, cell["devices"])
    assert_steps_finite(scalars(run).items(), "masked diffusion LM")
    prompt = [5, 6, 7]
    drawn = dew.pipeline(str(run))([prompt], 4, key=0, steps=2).host()
    response = np.asarray(drawn.tokens)[0, len(prompt):len(prompt) + 4]
    assert list(np.asarray(drawn.tokens)[0, :len(prompt)]) == prompt and (response != MASK).all()


MIXERS = {"gdn": {"class": "gated_delta_net", "fields": {
              "linear_num_key_heads": 2, "linear_num_value_heads": 2, "linear_key_head_dim": 8,
              "linear_value_head_dim": 8}},
          "kda": {"class": "kimi_delta_attention", "fields": {"linear_num_heads": 2, "linear_head_dim": 8}},
          "mamba": {"class": "mamba2", "fields": {"num_heads": 4, "head_dim": 8, "state_size": 8,
                                                  "n_groups": 1, "chunk_size": 8}}}
"""Each recurrent mixer the hybrid scenario trains and decodes, at a tiny width."""


def hybrid(cell: Mapping, root: Path) -> None:
    """Train a decoder of the cell's recurrent mixer through the shared components and decode it from its
    run: every step's gradient finite, and its task generating what its checkpoint generates."""
    run = trained_record(saved_record(LMRunConfig(**lm_data(cell, root)), root, "cell"),
                         {**common(cell, root), **TINY, "model.mixer": json.dumps(MIXERS[cell["mixer"]])},
                         cell["devices"])
    assert_steps_finite(scalars(run).items(), f"a {cell['mixer']} decoder")
    assert_generates_as_its_checkpoint(run)


def distributed(cell: Mapping, root: Path) -> None:
    """Train a dense decoder or a mixture of experts over a mesh of the cell's split, its experts sharded
    where the cell shards them: every step's gradient finite and its run reloading."""
    moe = {"model.mixture": json.dumps({"experts": 4, "top_k": 2})} if cell["model"] == "moe" else {}
    mesh = {"trainer.mesh.fsdp": "2"} if cell["mesh"] == "fsdp" else {"trainer.mesh.expert": "2"}
    run = trained_record(saved_record(LMRunConfig(**lm_data(cell, root)), root, "cell"),
                         {**common({**cell, "devices": 1}, root), **TINY, **moe, **mesh}, 2)
    assert_steps_finite(scalars(run).items(), f"a {cell['model']} decoder over {cell['mesh']}")
    assert_generates_as_its_checkpoint(run)


def diffusion_trained(cell: Mapping, root: str) -> None:
    """A tiny DiT's diffusion run over 8-pixel images at the cell's settings (`test_quantization`'s smallest
    run), trained in this process: its run directory `root/runs/cell`."""
    from test_quantization import RUN_BATCH, diffusion_run, image_batches

    from dew.data.dataset import Loader
    from dew.quantization import Quantization

    trainer = {"precision": numerics_of(cell["precision"]), "log_every": 1,
               "mesh": MeshSpec(fsdp=cell["devices"]), "layout": Layout(min_shard=64),
               **({"quantization": Quantization()} if cell["quantization"] == "qat" else {})}
    config = diffusion_run(Path(root) / "runs", RUN_BATCH, **trainer)
    # The run's numerics reach its model as a run's do (`RunConfig.numerics_fields`): its compute dtype.
    compute = numerics_of(cell["precision"]).dtypes.compute
    model = dataclasses.replace(config.model, fields={**config.model.fields, "remat": cell["memory"],
                                                      "dtype": compute})
    trainer_config = dataclasses.replace(config.trainer, steps=STEPS, checkpoint_final=True)
    config = dataclasses.replace(config, model=model, trainer=trainer_config)
    config.train(config.build(), Loader(lambda partition: image_batches(RUN_BATCH)(), None, None, RUN_BATCH),
                 name="cell")


def diffusion(cell: Mapping, root: Path) -> None:
    """Train an image model, checkpointing it, and reload its run: every step's gradient finite, and the
    reloaded pipeline sampling a finite image."""
    in_fresh_process(cell["devices"], "diffusion_trained", dict(cell), str(root))
    run = root / "runs" / "cell"
    assert_steps_finite(scalars(run).items(), "diffusion training")
    sampled = dew.pipeline(str(run))("", steps=2, key=0).host().images
    assert sampled is not None and np.isfinite(np.asarray(sampled, np.float32)).all()


def jepa_trained(cell: Mapping, root: str) -> None:
    """A JEPA encoder trained over images read as the cell reads, its modules computing in the cell's dtype,
    its mesh its devices' fsdp split, in this process: every step's gradient finite; its target encoder (the
    context encoder's average) moved from where it started and apart from the live encoder; and the
    trained encoder's representations of a held batch finite and not all one value."""
    from test_jepa import GRID, RES, image_batches, images, make_encoder, make_predictor

    from dew.batch import part
    from dew.inputs import Field
    from dew.objectives.jepa import JepaObjective, MultiBlockMask

    compute = {"dtype": numerics_of(cell["precision"]).dtypes.compute}
    objective = JepaObjective(make_encoder(**compute), make_predictor(**compute),
                              MultiBlockMask.for_grid(GRID, num_targets=4, scale=(0.15, 0.2)),
                              sample=Field("image", (RES, RES, 3)), momentum=(0.5, 0.5), momentum_steps=1)
    rows = Memory([dict(zip(("image", "label"), (image, label), strict=True))
                   for batch in itertools.islice(image_batches(8), 2)
                   for image, label in zip(batch["image"], batch["label"], strict=True)])
    tracker = RecordingTracker()
    trainer = Trainer(objective, optax.adam(1e-3), key=jax.random.key(0), mesh=MeshSpec(fsdp=cell["devices"]),
                      layout=Layout(min_shard=256), tracker=tracker)
    start = jax.tree.map(np.asarray, trainer.initial_state().ema)
    with run_precision(numerics_of(cell["precision"])):
        state = trainer.fit(rows.read(cell["read"]).batch(8), steps=STEPS, log_every=1,
                            checkpoint_final=False)
    assert_steps_finite(tracker.scalars, "JEPA")
    target, began = (jax.tree.leaves(tree["params"]["context_encoder"]) for tree in (state.ema, start))
    live = jax.tree.leaves(state.variables["params"]["context_encoder"])
    assert target and any(not np.array_equal(np.asarray(a), b) for a, b in zip(target, began, strict=True))
    assert any(not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(target, live, strict=True))
    represented = np.asarray(objective.encode(part(state.variables, "context_encoder"), images(batch=4)),
                             np.float32)
    assert np.isfinite(represented).all() and np.ptp(represented) > 0


def jepa(cell: Mapping, root: Path) -> None:
    """Representation learning: a JEPA encoder trained and its representations read, in a process of the
    cell's devices (`jepa_trained`)."""
    in_fresh_process(cell["devices"], "jepa_trained", dict(cell), str(root))


def online_rl(cell: Mapping, root: Path) -> None:
    """Language RL: an LM run that names GRPO (or RLOO) collects rollouts from its policy, scores them by the
    reward its import path names and updates the policy, `dew train` as a user runs it: every step's
    gradient finite and its run reloading with its rollout."""
    import rl_support

    from dew.config.lm import PromptRollout
    from dew.registry import ModelConfig

    tokenizer = str(Path(__file__).resolve().parent / "fixtures" / "tokenizers" / "tiny-chat")
    rows = [{"prompt": "hi there " * 8, "data_source": "rule", "ground_truth": "a"}] * 8
    config = LMRunConfig(
        data=Memory(rows).pipe(presets.prompts, tokenizer, max_prompt_len=8, batch=8), tokenizer=tokenizer,
        objective=ObjectiveConfig("grpo"), max_new_tokens=0,
        rollout=PromptRollout(reward=f"{rl_support.__name__}:completion_length", groups=2, max_new_tokens=4,
                              method=cell["method"]),
        model=ModelConfig("causal_transformer", {"emb_features": 16, "num_layers": 1, "num_heads": 2}))
    run = trained_record(saved_record(config, root, "cell"), common(cell, root), cell["devices"],
                         trust=(rl_support.__name__,))
    assert_steps_finite(scalars(run).items(), f"online {cell['method']}")
    assert LMRunConfig.load(str(run)).rollout.method == cell["method"]


def assert_served_as_its_forward(task, prompts: list[list[int]], steps: int = 4) -> None:
    """A served text task's greedy continuation of each prompt is its model's own, a full forward a token
    over the task's weights (`model_support.greedy_walk`), every generated token's log-probability
    finite."""
    from model_support import greedy_walk

    served = task(prompts, steps, key=0).host()
    for row, prompt in enumerate(prompts):
        walked = greedy_walk(task.model, task.variables, np.asarray([prompt], np.int32), steps)[0]
        # A row's tokens hold its prompt and then its response, which `lengths` counts and
        # `raw_log_probs` holds alone.
        length = int(served.lengths[row])
        assert length > 0
        np.testing.assert_array_equal(np.asarray(served.tokens)[row, :len(prompt) + length],
                                      walked[:len(prompt) + length])
        assert np.isfinite(np.asarray(served.raw_log_probs)[row, :length]).all()


def serving(cell: Mapping, root: Path) -> None:
    """Generation and serving: a decoder computing in the cell's dtype, quantized after training as the cell
    serves (PTQ, int8 or fp8), over one prompt or a batch of three: its greedy continuation each prompt's
    full forward's (`assert_served_as_its_forward`)."""
    from model_support import serving_task

    from dew.quantization import Quantization

    task = serving_task(sampling=Sampling(temperature=0))
    task = TextGeneration(task.model.clone(dtype=numerics_of(cell["precision"]).dtypes.compute),
                          task.variables, sampling=Sampling(temperature=0))
    if cell["quantization"] is not None:
        task = task.quantized(Quantization(dtype=cell["quantization"]))
    prompts = [[3, 4]] if cell["prompts"] == 1 else [[3, 4], [5, 6], [8, 9]]
    with run_precision(numerics_of(cell["precision"])):
        assert_served_as_its_forward(task, prompts)


def quantized_interchange(cell: Mapping, root: Path) -> None:
    """Quantized interchange: a llama written as a GGUF file of the cell's quantization, loaded, then
    trained on from its dequantized weights or served: every step's gradient finite where it trains, and
    its greedy continuation the same each call where it serves."""
    import gguf
    from test_gguf import hf_tensors, write_gguf

    from dew.interop import Pretrained
    from dew.objectives.lm import LMObjective

    rng = np.random.default_rng(0)
    write_gguf(root / "model.gguf", "llama", getattr(gguf.GGMLQuantizationType, cell["format"]),
               hf_tensors("llama", rng), rng)
    source = Pretrained.load(root, dtype="float32", attention_impl="reference", gguf_file="model.gguf")
    if cell["use"] == "train":
        tracker = RecordingTracker()
        trainer = Trainer(LMObjective(source.model, 8, variables=jax.device_get(source.variables)),
                          optax.adam(1e-3), key=jax.random.key(0), tracker=tracker)
        rows = np.random.default_rng(1).integers(3, 30, (8, 9)).astype(np.int32)
        trainer.fit(Memory([{"text": row} for row in rows]).batch(8), steps=STEPS, log_every=1,
                    checkpoint_final=False)
        assert_steps_finite(tracker.scalars, f"a {cell['format']} GGUF's llama")
    else:
        task = TextGeneration(source.model, source.variables, sampling=Sampling(temperature=0))
        assert_served_as_its_forward(task, [[3, 4, 5]])


def published_pipeline(cell: Mapping, root: Path) -> None:
    """Published diffusion pipelines: a tiny released pipeline loaded and sampled end to end at the cell's
    family, its images finite and the same each call at one key."""
    import tarfile

    from test_pipeline_streaming import PIPELINES

    from dew.interop.pretrained import Pretrained

    archive, inside = PIPELINES[cell["family"]]
    with tarfile.open(Path(__file__).resolve().parents[1] / f"tests/fixtures/{archive}.tar.xz") as opened:
        opened.extractall(root / archive, filter="data")
    loaded = Pretrained.load(str(root / archive / inside), dtype=cell["dtype"], attention_impl="xla")
    task = loaded.text_to_image()
    first, second = (np.asarray(task(["a red bird"], steps=2, key=3).host().images, np.float32)
                     for _ in range(2))
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


SCENARIOS: Mapping[str, Scenario] = {
    "dense LM": Scenario({"precision": ("32-true", "bf16-mixed"), "rows": ("padded", "packed"),
                          "memory": (None, "full"), "devices": (1, 2),
                          "read": ("memory", "indexed", "streamed")}, dense_lm),
    "pretrained": Scenario({"precision": ("32-true", "bf16-mixed"), "adapter": ("full", "lora"),
                            "quantization": (None, "qat", "ptq"), "devices": (1, 2)}, pretrained),
    "masked diffusion LM": Scenario({"precision": ("32-true", "bf16-mixed"), "rows": ("padded", "packed"),
                                     "devices": (1, 2)}, masked_diffusion),
    "hybrid": Scenario({"mixer": tuple(MIXERS), "precision": ("32-true", "bf16-mixed"),
                        "rows": ("padded", "packed"), "devices": (1, 2)}, hybrid),
    "distributed": Scenario({"model": ("dense", "moe"), "mesh": ("fsdp", "expert"),
                             "precision": ("32-true", "bf16-mixed")}, distributed),
    "diffusion": Scenario({"precision": ("32-true", "bf16-mixed"), "quantization": (None, "qat"),
                           "memory": (False, "full"), "devices": (1, 2)}, diffusion),
    "JEPA": Scenario({"precision": ("32-true", "bf16-mixed"), "devices": (1, 2),
                      "read": ("memory", "indexed", "streamed")}, jepa),
    "online RL": Scenario({"method": ("grpo", "rloo"), "precision": ("32-true", "bf16-mixed"),
                           "devices": (1, 2)}, online_rl),
    "serving": Scenario({"precision": ("32-true", "bf16-mixed"), "quantization": (None, "int8", "fp8"),
                         "prompts": (1, 3)}, serving),
    "quantized interchange": Scenario({"format": ("Q8_0", "Q4_0"), "use": ("train", "serve")},
                                      quantized_interchange),
    "published pipeline": Scenario({"family": ("sd", "flux", "z_image"), "dtype": ("float32", "bfloat16")},
                                   published_pipeline),
}


def _named(cell: Mapping) -> str:
    return ",".join(f"{axis}={value}" for axis, value in cell.items())


CELLS = [pytest.param(name, cell, id=f"{name}-{_named(cell)}")
         for name, scenario in SCENARIOS.items() for cell in covering(scenario.axes)]


@pytest.mark.parametrize(("name", "cell"), CELLS)
def test_a_scenarios_cell_holds_the_contracts_its_flow_touches(name, cell, tmp_path):
    """One cell of a scenario's covering set: its run at the cell's settings, held to its claims."""
    SCENARIOS[name].run(cell, tmp_path)
