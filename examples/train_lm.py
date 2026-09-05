"""Train a byte-level language model on a directory of token files, then generate.

    python tools/tokenize_text.py --input data/shakespeare.txt --out data/shakespeare --tokenizer byte
    python examples/train_lm.py --tokens data/shakespeare --epochs 4
    python examples/train_lm.py --tokens data/shakespeare --steps 20 --sequence-length 32   # smoke run
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import tyro

from dew.data import ByteTokenizer, Loading, TokenWindows
from dew.objectives.lm import LMObjective, Samples
import dew.nn.backbones  # registers the models
from dew.registry import models
from dew.sampling import generate
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
    model: dict = field(default_factory=lambda: dict(emb_features=384, num_layers=6, num_heads=6))
    prompt: str = "ROMEO:"
    sample_tokens: int = 300
    out: Path = Path("runs/shakespeare")


def main(config: Config):
    meta = json.loads((config.tokens / "meta.json").read_text())
    tokenizer = ByteTokenizer()
    data = TokenWindows(path=str(config.tokens), seq_len=config.sequence_length,
                        loading=Loading(workers=4)).load(batch=config.batch_size)
    steps = config.steps or data.epoch_steps(config.epochs)

    prompt = tokenizer.encode(config.prompt)
    model = models.build("causal_transformer", **config.model, vocab_size=int(meta["vocab_size"]),
                         max_seq_len=max(config.sequence_length, len(prompt) + config.sample_tokens),
                         dtype="bfloat16")
    objective = LMObjective(model, config.sequence_length,
                            samples=Samples(prompt, config.sample_tokens, temperature=0.8, top_k=40,
                                            decode=tokenizer.decode))

    trainer = Trainer(objective, optax.adamw(config.learning_rate), key=jax.random.key(0),
                      checkpoints=Checkpoints(str(config.out / "checkpoints")))
    state = trainer.fit(data, steps=steps, log_every=50)

    tokens = generate(model, state.averaged, jnp.asarray([prompt], jnp.int32),
                      config.sample_tokens, key=jax.random.key(1), temperature=0.8, top_k=40)
    text = tokenizer.decode(tokens[0])
    config.out.mkdir(parents=True, exist_ok=True)
    (config.out / "sample.txt").write_text(text)
    print(text)
    return state


if __name__ == "__main__":
    main(tyro.cli(Config))
