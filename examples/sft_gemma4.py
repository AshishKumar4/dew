"""Full-weight SFT of a Gemma 4 text decoder on a Hub chat dataset.

    python examples/sft_gemma4.py --model google/gemma-4-E2B \\
        --dataset hf://datasets/allenai/tulu-3-sft-mixture --steps 4000 --out runs/gemma4-sft

The run holds the first `--held-out` conversations out for its validation
perplexity, packs the rest into windows, counts the loss on assistant targets
alone, shards the weights over the visible devices and accumulates
micro-batches into one update. It ends by exporting the trained weights to
the Hugging Face layout, so transformers and `Pretrained.load` both read
them, and the run directory itself scores through the harness:

    python -m dew.eval --model dew --model_args run=runs/gemma4-sft/checkpoints/gemma4-sft \\
        --tasks hellaswag --limit 64

The `sft` preset reads the Hub dataset through its `hf://` source, so the URL
on the command line is what the run trains on: it renders every conversation
with the checkpoint's own chat template and packs them into windows.

    JAX_PLATFORMS=cpu python examples/sft_gemma4.py --smoke --out /tmp/gemma4-smoke
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import tyro

from dew.config import ModelConfig, OptimConfig, TrainerConfig
from dew.config.lm import LMRunConfig
from dew.data import HFTokenizer, Loading, presets
from dew.data.chat import Role
from dew.data.sources import Read, source as data_source
from dew.inference import RunProcessor
from dew.interop import PretrainedDecoder
from dew.objectives.lm import LMObjective, Perplexity, Samples
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
    dataset: str = "hf://datasets/allenai/tulu-3-sft-mixture"
    """A Hub source URL, Parquet file or JSONL conversations."""
    data_read: Read = "memory"
    """Materialize rendering and packing so four workers can share even one physical source shard."""
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
    held_out: int = 512
    """The first conversations, which validation scores and training never reads."""
    smoke: bool = False
    """Fine-tune the committed tiny Gemma 4 on canned turns instead."""


def write_conversations(conversations: list, out: Path) -> Path:
    """The canned turns as the JSONL file the `sft` preset reads line by line."""
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "chat.jsonl"
    jsonl.write_text("".join(json.dumps({"messages": turns}) + "\n" for turns in conversations))
    return jsonl


def smoke_config(config: Config) -> tuple[Config, str, str]:
    """The run over the committed tiny Gemma 4: its config, the vocabulary
    its ids were written against, and the canned turns written beside it."""
    small = replace(config, model=str(SMOKE_MODEL), sequence_length=15, batch_size=2, accumulation=2,
                    steps=2, held_out=2)
    return small, str(SMOKE_VOCAB), str(write_conversations(SMOKE_CONVERSATIONS, config.out))


def run_config(config: Config, tokenizer: str, chat: str) -> LMRunConfig:
    """Everything the run is but its model, which the checkpoint decides
    (`main` records the loaded one)."""
    smoke = config.smoke
    # A row budget is the split slice `datasets` already understands.
    split = config.split if config.rows is None else f"{config.split}[:{config.rows}]"
    # A Hub split is downloaded and prepared where the run first reads it; a local file, as the
    # smoke run's, reads as it is.
    train, held = data_source(chat, split=split).holdout(records=config.held_out)
    loading = Loading(workers=0, threads=1, read_buffer=2, worker_buffer=1) if smoke else Loading(workers=4)
    windows = {"seq_len": config.sequence_length, "batch": config.batch_size, "messages": config.column}
    return LMRunConfig(
        data=(train.shuffle(seed=0).repeat().pipe(presets.sft, tokenizer, **windows, long="chunk")
              .read(config.data_read).loading(loading)),
        validation=(held.pipe(presets.sft, tokenizer, **windows, records=config.batch_size)
                    .read(config.data_read).loading(loading)),
        tokenizer=tokenizer,
        max_new_tokens=4 if smoke else 64,
        optim=OptimConfig(learning_rate=config.learning_rate, weight_decay=0.0,
                          clip_grads=1.0),
        # A smoke run stays out of any process pool; a real run joins one
        # when a cluster started it and runs alone otherwise.
        trainer=TrainerConfig(checkpoint_dir=str(config.out / "checkpoints"),
                              steps=config.steps,
                              accumulation=config.accumulation, log_every=1 if smoke else 20,
                              eval_every=config.steps, checkpoint_every=config.steps,
                              multi_host=False if smoke else None,
                              compilation_cache_dir=None if smoke else
                              TrainerConfig().compilation_cache_dir))


def main(config: Config) -> Path:
    if config.smoke:
        config, tokenizer, chat = smoke_config(config)
    else:
        tokenizer, chat = config.model, config.dataset

    run = run_config(config, tokenizer, chat)
    prepare_process(run.trainer.wandb, run.trainer.multi_host, run.trainer.xla_flags,
                    run.trainer.compilation_cache_dir, layout=run.trainer.layout)
    # Only now: counting devices opens the backend, which has to come after
    # the pool forms, and a pool's count is every process's devices.
    run = replace(run, trainer=replace(run.trainer, mesh=MeshSpec(fsdp=jax.device_count())))

    source = PretrainedDecoder.load(config.model, dtype=jnp.float32 if config.smoke else jnp.bfloat16,
                                    attention_impl="xla" if config.smoke else "auto",
                                    max_seq_len=config.sequence_length + run.max_new_tokens,
                                    on_host=True)  # the trainer places it over its fsdp mesh
    words = HFTokenizer(tokenizer)
    # The run decodes through this tokenizer, so its checkpoints record its name.
    objective = LMObjective(
        source, config.sequence_length,
        loss_role=Role.ASSISTANT,
        processor=RunProcessor(words),
        samples=Samples(words.encode("user : hello "), run.max_new_tokens,
                        sampling=run.sampling, decode=words.decode))

    # run.json records the model as loaded, so `dew.pipeline` and the export
    # rebuild the checkpoint's own architecture at the precision it trained in.
    run = replace(run, model=ModelConfig.from_model(source.model))

    name = config.out.name
    state: TrainState = run.train(objective, run.data, validation=run.validation, name=name,
                                  metrics=(Perplexity(),),
                                  summary={"model": dict(source.model_config)})

    run_dir = Path(run.trainer.checkpoint_dir) / name
    export = config.out / "export"
    PretrainedDecoder.from_run(str(run_dir)).save(export)
    print(f"trained {int(state.step)} steps; run {run_dir}; exported {export}")
    return export


if __name__ == "__main__":
    main(tyro.cli(Config))
