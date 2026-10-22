"""LoRA SFT of DiffusionGemma on chat data, with the base weights host-streamed.

The optimizer updates only the adapter, and `Layout(host=("variables",))`
keeps the whole train state on the host between steps, with the scanned stack
fetching one layer at a time. The 26B-A4B base has not been run through it:

    python examples/sft_diffusion_gemma.py \\
        --model google/diffusiongemma-26B-A4B-it \\
        --chat data/tulu-3-sft.parquet --steps 2000

`--chat` is an `hf://` Hub dataset, a `.jsonl` file or a parquet file of
conversations, which the `canvas` preset reads and renders with the
checkpoint's own chat template. Each assistant turn becomes one row: the
conversation before it is the clean prompt, and the turn alone fills the
canvases the denoiser scores. The run writes the PEFT adapter directory
`Adapter.save` produces, which transformers loads, then loads the base again
through `dew.pipeline`, merges the adapter into it and writes the decoded
canvases to `samples.txt`. Image-conditioned training on real Oxford Flowers
is `examples/sft_diffusion_gemma_images.py`; it uses a fresh small model,
not the released 26B model's qualification.

    JAX_PLATFORMS=cpu python examples/sft_diffusion_gemma.py --smoke --out /tmp/dg-smoke
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import tyro

import dew
from dew.config import OptimConfig
from dew.data import Dataset, presets
from dew.data.sources import source as data_source
from dew.interop import PretrainedBlockDecoder
from dew.lora import LoRA
from dew.objectives.diffusion.block import BlockDiffusionObjective
from dew.training import Checkpoints, Layout, MeshSpec, Trainer

FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures"
SMOKE_SOURCE = FIXTURES / "hf/diffusion-gemma-workflow"

CONVERSATIONS = [
    [{"role": "user", "content": "t5 t7"}, {"role": "assistant", "content": "t9 t11"}],
    [{"role": "user", "content": "t6 t8"}, {"role": "assistant", "content": "t10 t12"}],
    [{"role": "user", "content": "t13 t15"}, {"role": "assistant", "content": "t17 t19"},
     {"role": "user", "content": "t21 t23"}, {"role": "assistant", "content": "t25 t27 t29 t31 t33"}],
]

PROMPTS = ["<bos> t5 t7 t9 t11", "<bos> t6 t8 t10 t12"]


@dataclass
class Config:
    model: str = "google/diffusiongemma-26B-A4B-it"
    """Hub repo or local directory the base weights and tokenizer come from."""
    chat: str | None = None
    """Hub dataset id, .jsonl or parquet of conversations; --smoke writes its own."""
    out: Path = Path("runs/diffusion-gemma-lora")
    prompt_tokens: int = 256
    """Clean prompt prefix of a training row; the canvases follow it."""
    canvases: int = 4
    """Canvases per row. The row is prompt_tokens + canvases * canvas_length."""
    batch_size: int = 8
    steps: int = 2000
    learning_rate: float = 1e-4
    rank: int = 16
    alpha: float = 32.0
    modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    """PEFT's target_modules: which projections carry a delta."""
    response_tokens: int = 32
    smoke: bool = False
    """Fine-tune the committed tiny checkpoint on canned turns instead."""


def smoke_config(config: Config) -> Config:
    """The run over the committed tiny checkpoint, which carries the tokenizer
    too, and the canned turns as the JSONL file the preset reads line by line."""
    config.out.mkdir(parents=True, exist_ok=True)
    jsonl = config.out / "chat.jsonl"
    jsonl.write_text("".join(json.dumps({"messages": turns}) + "\n" for turns in CONVERSATIONS))
    return replace(config, model=str(SMOKE_SOURCE), chat=str(jsonl), prompt_tokens=8, canvases=2,
                   batch_size=2, steps=2, rank=2, alpha=4.0, response_tokens=4)


def adapted_objective(config: Config) -> tuple[PretrainedBlockDecoder, BlockDiffusionObjective]:
    """The base with its adapter, and the SFT objective over it."""
    # On the host: the trainer streams the stack from there (Layout(host=...)).
    source = PretrainedBlockDecoder.load(config.model, dtype=jnp.bfloat16, param_dtype=jnp.float32,
                                         on_host=True).adapt(
        LoRA(rank=config.rank, modules=tuple(config.modules), alpha=config.alpha), key=0)
    # A host layout streams the stack one layer per scan iteration, so the
    # decoder runs as a scan; a plain loop's fetches would all be hoisted and
    # the whole base would land on the device this is keeping it off.
    scanned = source.model.clone(text=source.model.text.clone(scan_layers=True))
    return source, BlockDiffusionObjective(
        scanned, prompt_length=config.prompt_tokens,
        num_canvases=config.canvases, variables=source.variables,
        pad_token_id=int(source.config["text_config"]["pad_token_id"]))


def chat_rows(config: Config, objective: BlockDiffusionObjective, conversations: Dataset) -> Dataset:
    """`conversations` as the objective's rows, one per assistant turn, in its geometry, in batches."""
    return conversations.pipe(
        presets.canvas, config.model, prompt_length=objective.prompt_length,
        canvas_size=objective.canvas_size, canvases=objective.num_canvases, batch=config.batch_size,
        pad_id=objective.pad_token_id)


def main(config: Config) -> Path:
    if config.smoke:
        config = smoke_config(config)
    elif config.chat is None:
        raise ValueError("--chat names the conversations to fine-tune on: a hub "
                         "dataset id, a .jsonl file or a parquet file")

    source, objective = adapted_objective(config)
    data = chat_rows(config, objective, data_source(config.chat).shuffle(seed=0).repeat())

    checkpoints = Checkpoints(str(config.out / "checkpoints"), keep=1)
    trainer = Trainer(objective, OptimConfig(learning_rate=config.learning_rate), key=jax.random.key(1),
                      mesh=MeshSpec(fsdp=jax.device_count()),
                      layout=Layout(host=("variables",)), checkpoints=checkpoints)
    state = trainer.fit(data, steps=config.steps, log_every=1,
                        checkpoint_every=config.steps)
    checkpoints.wait()

    adapter_dir = config.out / "adapter"
    assert source.adapter is not None
    source.adapter.save(state.variables, adapter_dir)

    # The other half of the workflow, from the files alone: the base weights
    # back through `dew.pipeline`, the adapter directory read onto them, and
    # the factors folded into the kernels so the task runs the base model.
    base = dew.pipeline(config.model, dtype=jnp.float32)
    trained = LoRA.load(base.model, base.variables, adapter_dir, layouts=source.layouts)
    task = base.bind(trained.merge(trained.variables))
    generated = task(PROMPTS, config.response_tokens, key=3)
    (config.out / "samples.txt").write_text("\n".join(task.decode(generated)) + "\n")
    print(f"adapter {adapter_dir}  samples {config.out / 'samples.txt'}")
    return adapter_dir


if __name__ == "__main__":
    main(tyro.cli(Config))
