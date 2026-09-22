"""Full-weight SFT of a Gemma 4 text decoder on a Hub chat dataset.

    python examples/sft_gemma4.py --model google/gemma-4-E2B \\
        --dataset allenai/tulu-3-sft-mixture --steps 4000 --out runs/gemma4-sft

The run packs whole conversations into windows, counts the loss on assistant
targets alone, shards the weights over the visible devices and accumulates
micro-batches into one update. It ends by exporting the trained weights to
the Hugging Face layout, so transformers and `load_pretrained` both read
them, and the run directory itself scores through the harness:

    python -m dew.eval --model dew --model_args run=runs/gemma4-sft/gemma4-sft \\
        --tasks hellaswag --limit 64

`dew.data.ChatMessages` reads the Hub dataset itself, so the id on the
command line is what the run trains on: it renders every conversation with
the checkpoint's own chat template and packs them into windows.

    JAX_PLATFORMS=cpu python examples/sft_gemma4.py --smoke --out /tmp/gemma4-smoke
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import jax
import tyro

from dew.config import ModelConfig, OptimConfig, TrainerConfig
from dew.data import ChatMessages, Loading, tokenizer_for
from dew.data.chat import Role
from dew.interop import export_run, load_pretrained
from dew.objectives.lm import LMObjective, LMRunConfig, Samples
from dew.registry import metrics
from dew.training import MeshSpec, TrainState, prepare_process

FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures"
SMOKE_MODEL = FIXTURES / "hf/gemma4-ple"
# The decoder fixtures ship weights and no tokenizer. This one holds the same
# 64 ids they were written against, one per `t<n>` word, so a canned turn
# tokenizes to distinct targets instead of a row of unknowns, and it carries
# the prefix-preserving chat template an instruct checkpoint always does.
SMOKE_VOCAB = FIXTURES / "hf/diffusion-gemma-workflow"
SMOKE_CONVERSATIONS = [
    [{"role": "user", "content": f"t{first} t{first + 2}"},
     {"role": "assistant", "content": f"t{first + 4} t{first + 6}"}]
    for first in range(5, 45, 5)]


@dataclass
class Config:
    model: str = "google/gemma-4-E2B"
    dataset: str = "allenai/tulu-3-sft-mixture"
    """Hub chat dataset id, a parquet file or a .jsonl file of conversations."""
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
    """Conversations taken from the head of the split; unset takes them all."""
    smoke: bool = False
    """Fine-tune the committed tiny Gemma 4 on canned turns instead."""


def write_conversations(conversations: list, out: Path) -> Path:
    """The canned turns as the JSONL `ChatMessages` reads line by line."""
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "chat.jsonl"
    jsonl.write_text("".join(json.dumps({"messages": turns}) + "\n" for turns in conversations))
    return jsonl


def run_config(config: Config, tokenizer: str, chat: str) -> LMRunConfig:
    """Everything the run is, before the checkpoint decides the architecture."""
    smoke = config.smoke
    # A row budget is the split slice `datasets` already understands.
    split = config.split if config.rows is None else f"{config.split}[:{config.rows}]"
    return LMRunConfig(
        model=ModelConfig("causal_transformer", {}, dtype="float32" if smoke else "bfloat16",
                          attention_impl="xla" if smoke else "auto"),
        data=ChatMessages(tokenizer=tokenizer, path=chat, val_path=chat, column=config.column,
                          split=split, seq_len=config.sequence_length, val_batches=1,
                          loading=Loading(workers=0, threads=1, read_buffer=2,
                                          worker_buffer=1) if smoke else Loading(workers=4)),
        tokenizer=tokenizer,
        sample_tokens=4 if smoke else 64,
        optim=OptimConfig(learning_rate=config.learning_rate, weight_decay=0.0,
                          clip_grads=1.0),
        trainer=TrainerConfig(checkpoint_dir=str(config.out / "checkpoints"),
                              batch_size=config.batch_size, steps=config.steps,
                              accumulation=config.accumulation, log_every=1 if smoke else 20,
                              eval_every=config.steps, checkpoint_every=config.steps,
                              mesh=MeshSpec(fsdp=jax.device_count()),
                              multi_host=not smoke,
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

    source = load_pretrained(config.model, dtype=run.model.dtype,
                             attention_impl=run.model.attention_impl,
                             max_seq_len=config.sequence_length + run.sample_tokens)
    words = tokenizer_for(tokenizer)
    objective = LMObjective(
        source.model, config.sequence_length, pretrained=source.variables,
        loss_role=Role.ASSISTANT,
        samples=Samples(words.encode("user : hello "), run.sample_tokens,
                        sampling=run.sampling, decode=words.decode))

    # run.json records the model as built, so `dew.pipeline` and the export
    # rebuild the checkpoint's own architecture rather than these defaults.
    resolved = {name: value for name, value in source.model_config.items()
                if name not in run.model.precision_settings()}
    run = replace(run, model=replace(run.model, config=resolved))

    data = run.data.load(batch=run.trainer.batch_size)
    name = config.out.name
    state: TrainState = run.train(objective, data, name=name,
                                  metrics=(metrics.perplexity(),),
                                  summary={"model": dict(source.model_config)})

    run_dir = Path(run.trainer.checkpoint_dir) / name
    export = config.out / "export"
    export_run(str(run_dir), export)
    print(f"trained {int(state.step)} steps; run {run_dir}; exported {export}")
    return export


if __name__ == "__main__":
    main(tyro.cli(Config))
