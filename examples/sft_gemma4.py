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

A Hub chat dataset is fetched once and written as the parquet
`dew.data.ChatMessages` renders: that spec reads a local file, and the `hf`
provider hands back raw rows with no chat template behind them.

    JAX_PLATFORMS=cpu python examples/sft_gemma4.py --smoke --out /tmp/gemma4-smoke
"""

import json
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path

import jax
import tyro

from dew.config import ModelConfig, OptimConfig, RunConfig, TrainerConfig
from dew.data import ChatMessages, HFOptions, Loading, tokenizer_for
from dew.data.chat import Role
from dew.interop import export_run, load_pretrained
from dew.objectives.lm import LMObjective, Samples
from dew.registry import metrics
from dew.sampling import Sampling
from dew.training import MeshSpec, TrainState, prepare_process

FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures"
SMOKE_MODEL = FIXTURES / "hf/gemma4-ple"
# The decoder fixtures ship weights and no tokenizer. This one holds the same
# 64 ids they were written against, one per `t<n>` word, so a canned turn
# tokenizes to distinct targets instead of a row of unknowns. The chat
# template is this file's, since no fixture tokenizer carries one and an
# instruct checkpoint always does.
SMOKE_VOCAB = FIXTURES / "hf/diffusion-gemma-workflow"
SMOKE_TEMPLATE = ("{% for message in messages %}{{ message['role'] }} : "
                  "{{ message['content'] }} {% endfor %}"
                  "{% if add_generation_prompt %}assistant : {% endif %}")
SMOKE_CONVERSATIONS = [
    [{"role": "user", "content": f"t{first} t{first + 2}"},
     {"role": "assistant", "content": f"t{first + 4} t{first + 6}"}]
    for first in range(5, 45, 5)]


@dataclass(frozen=True)
class ChatSFTRun(RunConfig):
    """The run record a chat SFT publishes.

    `RunConfig` holds the model, the data, the optimizer and the trainer;
    what a saved decoder needs beside them is the tokenizer its ids came
    from and the preview policy, which is what `dew.pipeline` and
    `dew.interop.export_run` read back out of `run.json`. The LM recipe's
    own config declares the same three and trains on token files, so a chat
    spec needs this one.
    """

    objective: str = "lm"
    model: ModelConfig = field(default_factory=lambda: ModelConfig("causal_transformer"))
    data: ChatMessages = field(default_factory=lambda: ChatMessages(tokenizer="byte"))
    tokenizer: str = "byte"
    sample_tokens: int = 16
    sampling: Sampling = field(default_factory=lambda: Sampling(temperature=0.8, top_k=40))


@dataclass
class Config:
    model: str = "google/gemma-4-E2B"
    dataset: str = "allenai/tulu-3-sft-mixture"
    """Hub chat dataset; its `messages` column holds the conversations."""
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
    """The conversations as JSONL, then as the parquet ChatMessages reads.

    The JSONL is the form a chat corpus arrives in and the form a reader can
    open; the parquet is what `ConversationSource` indexes.
    """
    import pyarrow
    import pyarrow.parquet

    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "chat.jsonl"
    jsonl.write_text("".join(json.dumps({"prompt": turns}) + "\n" for turns in conversations))
    rows = [json.loads(line)["prompt"] for line in jsonl.read_text().splitlines()]
    parquet = out / "chat.parquet"
    pyarrow.parquet.write_table(pyarrow.table({"prompt": rows}), parquet)
    return parquet


def hub_conversations(config: Config, out: Path) -> Path:
    """One Hub split's chat column, through `datasets.load_dataset`.

    `HFOptions` is the same value `datasets["hf"]` forwards, so the download,
    the cache and the revision are the library's own on the library's terms.
    """
    split = HFOptions().load(config.dataset, config.split, streaming=False)
    if config.rows is not None:
        split = split.select(range(config.rows))
    return write_conversations([list(row) for row in split[config.column]], out)


def smoke_tokenizer(out: Path) -> str:
    """The fixture's vocabulary with a prefix-preserving chat template on it."""
    directory = out / "tokenizer"
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy(SMOKE_VOCAB / "tokenizer.json", directory / "tokenizer.json")
    record = json.loads((SMOKE_VOCAB / "tokenizer_config.json").read_text())
    (directory / "tokenizer_config.json").write_text(
        json.dumps({**record, "chat_template": SMOKE_TEMPLATE}, indent=2))
    return str(directory)


def run_config(config: Config, tokenizer: str, parquet: Path) -> ChatSFTRun:
    """Everything the run is, before the checkpoint decides the architecture."""
    smoke = config.smoke
    return ChatSFTRun(
        model=ModelConfig("causal_transformer", {}, dtype="float32" if smoke else "bfloat16",
                          attention_impl="xla" if smoke else "auto"),
        data=ChatMessages(tokenizer=tokenizer, path=str(parquet), val_path=str(parquet),
                          seq_len=config.sequence_length, val_batches=1,
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
        tokenizer = smoke_tokenizer(config.out)
        parquet = write_conversations(SMOKE_CONVERSATIONS, config.out)
    else:
        tokenizer = config.model
        parquet = hub_conversations(config, config.out)

    run = run_config(config, tokenizer, parquet)
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
