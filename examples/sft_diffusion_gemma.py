"""LoRA SFT of DiffusionGemma on chat data, with the base weights host-streamed.

The optimizer updates only the adapter, and `Layout(host=("variables",))`
keeps the whole train state on the host between steps, with the scanned stack
fetching one layer at a time. The 26B-A4B base has not been run through it:

    python examples/sft_diffusion_gemma.py \\
        --model google/diffusiongemma-26B-A4B-it \\
        --chat data/tulu-3-sft.parquet --steps 2000

`--chat` is a Hub dataset id, a `.jsonl` file or a parquet file of
conversations, which is what `dew.data.ChatMessages` reads and renders with
the checkpoint's own chat template. The run writes the PEFT adapter directory
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
from dew.data import ChatMessages, Loading
from dew.interop import PretrainedBlockDecoder
from dew.lora import LoRA
from dew.objectives.diffusion.block import BlockDiffusionObjective
from dew.training import Checkpoints, Layout, MeshSpec, Trainer

FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures"
SMOKE_SOURCE = FIXTURES / "hf/diffusion-gemma-workflow"

CONVERSATIONS = [
    [{"role": "user", "content": "t5 t7"}, {"role": "assistant", "content": "t9 t11"}],
    [{"role": "user", "content": "t6 t8"}, {"role": "assistant", "content": "t10 t12"}],
    [{"role": "user", "content": "t13 t15"}, {"role": "assistant", "content": "t17 t19"}],
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
    """Fine-tune the committed tiny checkpoint on three canned turns instead."""


def smoke_conversations(out: Path) -> Path:
    """The canned turns as the JSONL `ChatMessages` reads line by line."""
    jsonl = out / "chat.jsonl"
    jsonl.write_text("".join(json.dumps({"messages": turns}) + "\n" for turns in CONVERSATIONS))
    return jsonl


def main(config: Config) -> Path:
    if config.smoke:
        config.out.mkdir(parents=True, exist_ok=True)
        tokenizer = str(SMOKE_SOURCE)
        config = replace(config, model=tokenizer, chat=str(smoke_conversations(config.out)),
                         prompt_tokens=8, canvases=2, batch_size=2, steps=2, rank=2,
                         alpha=4.0, response_tokens=4)
    else:
        tokenizer = config.model
        if config.chat is None:
            raise ValueError("--chat names the conversations to fine-tune on: a hub "
                             "dataset id, a .jsonl file or a parquet file")

    source = PretrainedBlockDecoder.load(config.model, dtype=jnp.bfloat16, param_dtype=jnp.float32).adapt(
        LoRA(rank=config.rank, modules=tuple(config.modules), alpha=config.alpha), key=0)
    sequence_length = config.prompt_tokens + config.canvases * source.model.canvas_length
    # A host layout streams the stack one layer per scan iteration, so the
    # decoder runs as a scan; a plain loop's fetches would all be hoisted and
    # the whole base would land on the device this is keeping it off.
    scanned = source.model.clone(text=source.model.text.clone(scan_layers=True))
    objective = BlockDiffusionObjective(
        scanned, prompt_length=config.prompt_tokens,
        num_canvases=config.canvases, variables=source.variables,
        pad_token_id=int(source.config["text_config"]["pad_token_id"]))

    # A packed window is seq_len + 1 ids wide, and the objective reads rows of
    # exactly prompt + canvases: the spec is asked for one less.
    data = ChatMessages(tokenizer=tokenizer, path=config.chat,
                        seq_len=sequence_length - 1, val_batches=None,
                        loading=Loading(workers=0, threads=1, read_buffer=2,
                                        worker_buffer=1)).load(batch=config.batch_size)

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
