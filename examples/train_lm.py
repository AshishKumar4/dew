"""Train a byte-level language model on a token store, then generate.

The text is split into lines to train on and lines held out, and each is
tokenized into a store of its own by `dew data export`: training reads every
token of its store, and the held-out store is what
examples/evaluate_and_serve.py scores.

    curl -o data/shakespeare.txt --create-dirs \\
        https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    head -n 36000 data/shakespeare.txt > data/shakespeare-train.txt
    tail -n +36001 data/shakespeare.txt > data/shakespeare-val.txt
    dew data export data/shakespeare-train.txt data/shakespeare --tokenizer byte --eos --read memory
    dew data export data/shakespeare-val.txt data/shakespeare-val --tokenizer byte --eos --read memory
    python examples/train_lm.py --tokens data/shakespeare --epochs 4
    python examples/train_lm.py --tokens data/shakespeare --steps 20 --sequence-length 32   # smoke run
"""
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import tyro

from dew.config import OptimConfig
from dew.data import ByteTokenizer, presets
from dew.data.sources import TokenFiles
from dew.inference import RunProcessor
from dew.nn.backbones import CausalTransformer
from dew.objectives.lm import LMObjective, Samples
from dew.sampling import Sampling
from dew.training import Checkpoints, Trainer


@dataclass
class Config:
    tokens: Path = Path("data/shakespeare")
    sequence_length: int = 256
    batch_size: int = 64
    epochs: int = 4
    steps: int | None = None
    """Run length in steps; unset trains for `epochs` passes over the data."""
    learning_rate: float = 1e-3
    model: dict = field(default_factory=lambda: {
        "emb_features": 384, "num_layers": 6, "num_heads": 6})
    prompt: str = "ROMEO:"
    max_new_tokens: int = 300
    out: Path = Path("runs/shakespeare")


def main(config: Config):
    tokenizer = ByteTokenizer()
    # Every token of the store, cut into windows by index and shuffled window by window.
    data = (TokenFiles(str(config.tokens)).shuffle(seed=0).repeat()
            .pipe(presets.token_windows, seq_len=config.sequence_length, batch=config.batch_size)
            .workers(4))

    prompt = tokenizer.encode(config.prompt)
    model = CausalTransformer(**config.model, vocab_size=tokenizer.vocab_size,
                              max_seq_len=max(config.sequence_length, len(prompt) + config.max_new_tokens),
                              dtype=jnp.bfloat16)
    objective = LMObjective(
        model,
        config.sequence_length,
        samples=Samples(
            prompt,
            config.max_new_tokens,
            sampling=Sampling(temperature=0.8, top_k=40),
            decode=tokenizer.decode,
        ),
        # Recorded with the run, so what loads it (examples/evaluate_and_serve.py) reads text.
        processor=RunProcessor(tokenizer),
    )

    trainer = Trainer(objective, OptimConfig(learning_rate=config.learning_rate), key=jax.random.key(0),
                      checkpoints=Checkpoints(str(config.out / "checkpoints")))
    state = trainer.fit(data, steps=config.steps,
                        epochs=None if config.steps else config.epochs, log_every=50)

    # No reload is needed; the weights stay where the trainer placed them, and
    # the tokenizer decodes the rows.
    task = objective.pipeline(state)
    text = config.prompt + task(config.prompt, key=1).text[0]
    config.out.mkdir(parents=True, exist_ok=True)
    (config.out / "sample.txt").write_text(text)
    print(text)
    return state


if __name__ == "__main__":
    main(tyro.cli(Config))
