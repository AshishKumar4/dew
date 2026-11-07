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

from dew.data import Dataset, HFTokenizer, Loading, presets
from dew.data.chat import Role
from dew.data.sources import Read, source as data_source
from dew.inference import RunProcessor
from dew.interop import PretrainedDecoder
from dew.objectives.lm import LMObjective, Perplexity, Samples
from dew.optim import OptimConfig
from dew.sampling import Sampling
from dew.training import Checkpoints, LocalTracker, MeshSpec, Trainer, prepare_process

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


def windows(config: Config, tokenizer: str, chat: str) -> tuple[Dataset, Dataset]:
    """The run's training windows and its validation's: every conversation but the first `held_out`
    rendered with the checkpoint's chat template and packed, and those first ones alone."""
    # A row budget is the split slice `datasets` already understands.
    split = config.split if config.rows is None else f"{config.split}[:{config.rows}]"
    # A Hub split is downloaded and prepared where the run first reads it; a local file, as the
    # smoke run's, reads as it is.
    train, held = data_source(chat, split=split).holdout(records=config.held_out)
    loading = (Loading(workers=0, threads=1, read_buffer=2, worker_buffer=1) if config.smoke
               else Loading(workers=4))
    shape = {"seq_len": config.sequence_length, "batch": config.batch_size, "messages": config.column}
    return ((train.shuffle(seed=0).repeat().pipe(presets.sft, tokenizer, **shape, long="chunk")
             .read(config.data_read).loading(loading)),
            (held.pipe(presets.sft, tokenizer, **shape, records=config.batch_size)
             .read(config.data_read).loading(loading)))


def main(config: Config) -> Path:
    if config.smoke:
        config, tokenizer, chat = smoke_config(config)
    else:
        tokenizer, chat = config.model, config.dataset
    data, validation = windows(config, tokenizer, chat)
    # A smoke run stays out of any process pool; a real run joins one when a cluster started it and runs
    # alone otherwise.
    prepare_process(multi_host=False if config.smoke else None)

    new_tokens = 4 if config.smoke else 64
    source = PretrainedDecoder.load(config.model, dtype=jnp.float32 if config.smoke else jnp.bfloat16,
                                    attention_impl="xla" if config.smoke else "auto",
                                    max_seq_len=config.sequence_length + new_tokens,
                                    on_host=True)  # the trainer places it over its fsdp mesh
    words = HFTokenizer(tokenizer)
    # The run decodes through this tokenizer, so its checkpoints record its name.
    objective = LMObjective(
        source, config.sequence_length,
        loss_role=Role.ASSISTANT,
        processor=RunProcessor(words),
        samples=Samples(words.encode("user : hello "), new_tokens,
                        sampling=Sampling(temperature=0.8, top_k=40), decode=words.decode))

    run_dir = config.out / "checkpoints" / config.out.name
    # Only now: counting devices opens the backend, which has to come after the pool forms, and a pool's
    # count is every process's devices.
    trainer = Trainer(objective, OptimConfig(learning_rate=config.learning_rate, weight_decay=0.0,
                                             clip_grads=1.0).build(config.steps),
                      key=jax.random.key(0), mesh=MeshSpec(fsdp=jax.device_count()),
                      accumulation=config.accumulation, checkpoints=Checkpoints(str(run_dir)),
                      tracker=LocalTracker(run_dir / "tracking"))
    state = trainer.fit(data, validation=validation, steps=config.steps, log_every=1 if config.smoke else 20,
                        eval_every=config.steps, checkpoint_every=config.steps, metrics=[Perplexity()])

    # The checkpoint records the decoder as it was loaded, so the export rebuilds its own architecture.
    export = config.out / "export"
    PretrainedDecoder.from_run(str(run_dir)).save(export)
    print(f"trained {int(state.step)} steps; run {run_dir}; exported {export}")
    return export


if __name__ == "__main__":
    main(tyro.cli(Config))
