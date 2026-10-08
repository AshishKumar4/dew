"""Train a byte-level language model on a directory of token files, then generate.

    curl -o data/shakespeare.txt --create-dirs \\
        https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    dew tokenize --input data/shakespeare.txt --out data/shakespeare --tokenizer byte
    python examples/train_lm.py --tokens data/shakespeare --epochs 4
    python examples/train_lm.py --tokens data/shakespeare --steps 20 --sequence-length 32   # smoke run
"""
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import tyro

from dew.config import OptimConfig
from dew.data import ByteTokenizer, Loading, TokenCorpus, TokenWindows
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
    corpus = TokenCorpus.read(config.tokens)
    tokenizer = ByteTokenizer()
    data = TokenWindows(path=str(config.tokens), seq_len=config.sequence_length,
                        loading=Loading(workers=4)).load(batch=config.batch_size)
    steps = config.steps or data.epoch_steps(config.epochs)

    prompt = tokenizer.encode(config.prompt)
    model = CausalTransformer(**config.model, vocab_size=corpus.vocab_size,
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
    )

    trainer = Trainer(objective, OptimConfig(learning_rate=config.learning_rate), key=jax.random.key(0),
                      checkpoints=Checkpoints(str(config.out / "checkpoints")))
    state = trainer.fit(data, steps=steps, log_every=50)

    # No reload is needed; the weights stay where the trainer placed them, and
    # the tokenizer decodes the rows.
    task = objective.pipeline(state, processor=RunProcessor(tokenizer))
    text = config.prompt + task(config.prompt, key=1).text[0]
    config.out.mkdir(parents=True, exist_ok=True)
    (config.out / "sample.txt").write_text(text)
    print(text)
    return state


if __name__ == "__main__":
    main(tyro.cli(Config))
