"""Train MDLM on TinyStories, score its held-out perplexity bound and unmask stories.

    python examples/train_masked_lm.py --out runs/mdlm-tinystories
    python examples/train_masked_lm.py --dataset Salesforce/wikitext \\
        --config wikitext-103-raw-v1 --prompts "The game" --out runs/mdlm-wikitext
    JAX_PLATFORMS=cpu python examples/train_masked_lm.py --smoke --out /tmp/mdlm-smoke

The first run reads the dataset's train split with `datasets` and tokenizes it
once into Dew's cache, each row a document ended by the
tokenizer's eos id. Later runs read the cached ids offline. The head 1% of
that stream is held out, and training never reads it. A bidirectional
`CausalTransformer` learns MDLM's negative ELBO (`MaskedDiffusionObjective`).
The mask is one id past the tokenizer's vocabulary, so no text can contain it.
Every `--eval-every` steps the trainer scores the held-out windows, and
`val/perplexity` is the exponential of the NELBO per token, the bound MDLM
reports.

At the end the script loads the run back with `dew.pipeline`, which returns a
`MaskedGeneration` task over the saved weights. It scores the whole held-out
split with those weights, unmasks a continuation of each of `--prompts`, and
writes `result.json` and `samples.txt` beside the run. `--smoke` writes a few
lines of text as JSON, byte-tokenizes them and trains a tiny model for four
steps on one CPU device. It needs no network.
"""

import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import jax
import tyro

import dew
from dew.config import ObjectiveConfig, TrainerConfig
from dew.config.lm import LMRunConfig
from dew.data import ByteTokenizer, Files, HFTokenizer, Hub, Loading, presets, text
from dew.data.compile import iterate
from dew.objectives.diffusion.masked import MaskedDiffusionObjective
from dew.objectives.lm import Perplexity
from dew.optim import Cosine, OptimConfig
from dew.registry import ModelConfig
from dew.training import Evaluation, MeshSpec

SMOKE_TEXT = (
    "The cat sat on the warm mat and watched the rain.\n",
    "A small boat drifted past the old stone bridge.\n",
    "She found a red kite caught in the apple tree.\n",
    "The baker gave the children fresh bread after school.\n",
)


@dataclass
class Config:
    dataset: str = "roneneldan/TinyStories"
    """A Hugging Face dataset id whose train split has a `text` column."""
    config: str | None = None
    """The dataset's configuration name, such as wikitext-103-raw-v1."""
    revision: str | None = None
    tokenizer: str = "gpt2"
    """"byte", or a Hugging Face tokenizer name."""
    sequence_length: int = 256
    batch_size: int = 128
    steps: int = 12_000
    learning_rate: float = 6e-4
    warmup_steps: int = 1000
    model: dict = field(default_factory=lambda: {
        "emb_features": 512, "num_layers": 8, "num_heads": 8})
    eval_every: int = 2000
    val_batches: int | None = 32
    """Held-out batches each periodic evaluation scores; the final score reads them all."""
    sample_steps: int = 128
    max_new_tokens: int = 160
    prompts: tuple[str, ...] = ("Once upon a time", "Tom and Lily went to the park.",
                                "The little bird was sad because")
    out: Path = Path("runs/mdlm-tinystories")
    smoke: bool = False
    """Train a tiny model on a few local lines, byte-tokenized, on one CPU device."""


def smoke_config(config: Config) -> tuple[Config, Files]:
    """The same run over a JSON file of text written here, read through the
    `json` builder `datasets` ships, so nothing is downloaded."""
    config.out.mkdir(parents=True, exist_ok=True)
    rows = config.out / "smoke.jsonl"
    rows.write_text("".join(json.dumps({"text": line}) + "\n" for line in SMOKE_TEXT * 160))
    source = Files(str(rows), format="jsonl")
    return replace(config, dataset="json", tokenizer="byte", sequence_length=32, batch_size=8,
                   steps=4, warmup_steps=1, model={"emb_features": 32, "num_layers": 1, "num_heads": 2},
                   eval_every=2, val_batches=1, sample_steps=4, max_new_tokens=8,
                   prompts=("The cat", "A small boat")), source


def main(config: Config) -> Path:
    source = Hub(config.dataset, name=config.config, revision=config.revision)
    if config.smoke:
        config, source = smoke_config(config)
    tokenizer = ByteTokenizer() if config.tokenizer == "byte" else HFTokenizer(config.tokenizer)
    documents = source.map(text.tokenize(config.tokenizer, eos=True))
    train, held = documents.holdout(tokens=0.01, fields=("text",))
    # The window is one id shorter than a row: the lm preset adds the id a
    # next-token target shifts by, and MDLM denoises the whole row. The ids
    # are tokenized already, so the preset takes no tokenizer.
    window = config.sequence_length - 1
    loading = Loading(workers=0, threads=4)
    # The windows are cut first, then shuffled one by one, as the held-out split's are read.
    data = (train.pipe(presets.lm, None, seq_len=window, batch=None, pack=None)
            .shuffle(seed=0).repeat().batch(config.batch_size)
            .read("memory").loading(loading))
    records = None if config.val_batches is None else config.val_batches * config.batch_size
    validation = (held.pipe(presets.lm, None, seq_len=window, batch=config.batch_size, pack=None,
                            records=records)
                  .read("memory").loading(loading))
    # The final score reads every held-out window.
    whole = held.pipe(presets.lm, None, seq_len=window, batch=config.batch_size, pack=None).read("memory")
    name = f"mdlm-{config.dataset.rsplit('/', 1)[-1].lower()}"
    trainer = TrainerConfig(name=name, checkpoint_dir=str(config.out / "checkpoints"), keep=1,
                            steps=config.steps,
                            log_every=1 if config.smoke else 100, eval_every=config.eval_every,
                            checkpoint_every=config.eval_every, mesh=MeshSpec(fsdp=1), multi_host=False)
    # The mask is the id past the tokenizer's, so no text can contain it.
    run = LMRunConfig(
        model=ModelConfig("causal_transformer", {
            **config.model, "causal": False, "mask_token_id": tokenizer.vocab_size,
            "dtype": "float32" if config.smoke else "bfloat16"}),
        data=data, validation=validation, tokenizer=config.tokenizer,
        objective=ObjectiveConfig("masked_diffusion", {
            "steps": config.sample_steps, "ema_decay": 0.9 if config.smoke else 0.999,
            "eos_token_ids": () if tokenizer.eos_id is None else (tokenizer.eos_id,)}),
        optim=OptimConfig(weight_decay=0.03, clip_grads=1.0, learning_rate=Cosine(
            peak_value=config.learning_rate, warmup_steps=config.warmup_steps,
            end_value=config.learning_rate / 10)),
        trainer=replace(trainer, compilation_cache_dir=None) if config.smoke else trainer)
    started = time.perf_counter()
    run.run()
    trained = time.perf_counter()

    # The other half, from the files alone: the run directory loads as the
    # task its objective saved, and both the score and the samples read it.
    run_dir = config.out / "checkpoints" / name
    task = dew.pipeline(str(run_dir))
    scored = Evaluation.run(MaskedDiffusionObjective(task.model, task.process, config.sequence_length),
                            task.variables, whole, key=jax.random.key(0), metrics=[Perplexity()],
                            loss=True)
    generated = task(list(config.prompts), config.max_new_tokens, key=1)
    samples = [prompt + text for prompt, text in zip(config.prompts, task.decode(generated), strict=True)]
    token_rows = iterate(documents, read="memory", loading=loading)
    reader = iter(token_rows)
    try:
        total_tokens = sum(len(row["text"]) for row in reader)
    finally:
        reader.close()
    held_tokens = int(total_tokens * 0.01)
    report = {"dataset": config.dataset, "config": config.config, "tokenizer": config.tokenizer,
              "train_tokens": total_tokens - held_tokens, "held_out_tokens": held_tokens,
              "device": jax.devices()[0].device_kind, "steps": config.steps,
              "batch_size": config.batch_size, "sequence_length": config.sequence_length,
              "train_seconds": round(trained - started, 1), "held_out": dict(scored.scores),
              "samples": samples}
    (config.out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    (config.out / "samples.txt").write_text("\n\n".join(samples) + "\n")
    print(json.dumps(report, indent=2))
    return run_dir


if __name__ == "__main__":
    main(tyro.cli(Config))
