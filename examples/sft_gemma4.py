"""Full-weight SFT of a Gemma 4 text decoder on a Hub chat dataset.

    python examples/sft_gemma4.py --model google/gemma-4-E2B \\
        --dataset allenai/tulu-3-sft-mixture --steps 4000 --out runs/gemma4-sft

The run packs whole conversations into windows and counts the loss only on
assistant targets. It shards the weights over the visible devices and
accumulates micro-batches into one update, then exports the trained weights
in the Hugging Face layout. Both transformers and `Pretrained.load` can read
the export, and the harness can score the run directory itself:

    python -m dew.eval --model dew --model_args run=runs/gemma4-sft/checkpoints/gemma4-sft \\
        --tasks hellaswag --limit 64

`dew.data.ChatMessages` reads the Hub dataset named on the command line,
renders every conversation with the checkpoint's own chat template and
packs them into windows.

    JAX_PLATFORMS=cpu python examples/sft_gemma4.py --smoke --out /tmp/gemma4-smoke
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import tyro

from dew.config import ModelConfig, OptimConfig, TrainerConfig
from dew.data import ChatMessages, HFTokenizer, Loading
from dew.data.chat import Role
from dew.inference import RunProcessor
from dew.interop import PretrainedDecoder
from dew.objectives.lm import LMObjective, LMRunConfig, Perplexity, Samples
from dew.training import MeshSpec, TrainState, prepare_process

FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures"
SMOKE_MODEL = FIXTURES / "hf/gemma4-ple"
# The decoder fixtures have weights but no tokenizer. This vocabulary has
# the same 64 IDs, one per `t<n>` word, so canned turns yield distinct targets
# rather than unknowns. It includes the prefix-preserving chat template
# expected from an instruct checkpoint.
SMOKE_VOCAB = FIXTURES / "hf/diffusion-gemma-workflow"
SMOKE_CONVERSATIONS = [
    [{"role": "user", "content": f"t{first} t{first + 2}"},
     {"role": "assistant", "content": f"t{first + 4} t{first + 6}"}]
    for first in range(5, 45, 5)]


@dataclass
class Config:
    model: str = "google/gemma-4-E2B"
    dataset: str = "allenai/tulu-3-sft-mixture"
    """Hub chat dataset ID, a parquet file or a .jsonl file of conversations."""
    split: str = "train"
    column: str = "messages"
    out: Path = Path("runs/gemma4-sft")
    sequence_length: int = 4096
    batch_size: int = 64
    accumulation: int = 4
    """Micro-batches per optimizer update, so a big effective batch fits."""
    steps: int = 4000
    learning_rate: float = 1e-5
    rows: int | None = None
    """Conversations to take from the start of the split; unset uses them all."""
    smoke: bool = False
    """Fine-tune the committed tiny Gemma 4 on canned turns instead."""


def write_conversations(conversations: list, out: Path) -> Path:
    """Write the canned turns as JSONL for `ChatMessages`."""
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "chat.jsonl"
    jsonl.write_text("".join(json.dumps({"messages": turns}) + "\n" for turns in conversations))
    return jsonl


def run_config(config: Config, tokenizer: str, chat: str) -> LMRunConfig:
    """Configure the run apart from its model, which comes from the checkpoint.

    `main` records that loaded model.
    """
    smoke = config.smoke
    # A row budget is the split slice `datasets` already understands.
    split = config.split if config.rows is None else f"{config.split}[:{config.rows}]"
    return LMRunConfig(
        data=ChatMessages(tokenizer=tokenizer, path=chat, val_path=chat, column=config.column,
                          split=split, seq_len=config.sequence_length, val_batches=1,
                          loading=Loading(workers=0, threads=1, read_buffer=2,
                                          worker_buffer=1) if smoke else Loading(workers=4)),
        tokenizer=tokenizer,
        sample_tokens=4 if smoke else 64,
        optim=OptimConfig(learning_rate=config.learning_rate, weight_decay=0.0,
                          clip_grads=1.0),
        # Smoke runs stay out of process pools. A real run joins the pool
        # supplied by its cluster, or runs alone when there is none.
        trainer=TrainerConfig(checkpoint_dir=str(config.out / "checkpoints"),
                              batch_size=config.batch_size, steps=config.steps,
                              accumulation=config.accumulation, log_every=1 if smoke else 20,
                              eval_every=config.steps, checkpoint_every=config.steps,
                              multi_host=False if smoke else None,
                              compilation_cache_dir=None if smoke else
                              TrainerConfig().compilation_cache_dir))


def main(config: Config) -> Path:
    if config.smoke:
        config = replace(config, model=str(SMOKE_MODEL), sequence_length=15, batch_size=2,
                         accumulation=2, steps=2)
        tokenizer = str(SMOKE_VOCAB)
        chat = str(write_conversations(SMOKE_CONVERSATIONS, config.out))
    else:
        tokenizer = config.model
        chat = config.dataset

    run = run_config(config, tokenizer, chat)
    prepare_process(run.trainer.wandb, run.trainer.multi_host, run.trainer.xla_flags,
                    run.trainer.compilation_cache_dir, layout=run.trainer.layout)
    # Counting devices opens the backend, so the pool must form first.
    # In a pool, the count includes devices from every process.
    run = replace(run, trainer=replace(run.trainer, mesh=MeshSpec(fsdp=jax.device_count())))

    source = PretrainedDecoder.load(config.model, dtype=jnp.float32 if config.smoke else jnp.bfloat16,
                                    attention_impl="xla" if config.smoke else "auto",
                                    max_seq_len=config.sequence_length + run.sample_tokens)
    words = HFTokenizer(tokenizer)
    # The run decodes through this tokenizer, so its checkpoints record its name.
    objective = LMObjective(
        source, config.sequence_length,
        loss_role=Role.ASSISTANT,
        processor=RunProcessor(words),
        samples=Samples(words.encode("user : hello "), run.sample_tokens,
                        sampling=run.sampling, decode=words.decode))

    # run.json records the model as loaded, so `dew.pipeline` and the export
    # rebuild the checkpoint's own architecture at the precision it trained in.
    run = replace(run, model=ModelConfig.from_model(source.model))

    data = run.data.load(batch=run.trainer.batch_size)
    name = config.out.name
    state: TrainState = run.train(objective, data, name=name,
                                  metrics=(Perplexity(),),
                                  summary={"model": dict(source.model_config)})

    run_dir = Path(run.trainer.checkpoint_dir) / name
    export = config.out / "export"
    PretrainedDecoder.from_run(str(run_dir)).save(export)
    print(f"trained {int(state.step)} steps; run {run_dir}; exported {export}")
    return export


if __name__ == "__main__":
    main(tyro.cli(Config))
