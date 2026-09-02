"""Train a byte-level language model on a tokenized corpus, then generate from it.

Tokenize first (Tiny Shakespeare takes a second):

    python tools/tokenize_text.py --input shakespeare.txt --out data/shakespeare --tokenizer byte --val-fraction 0.02
    python examples/train_lm.py --tokens data/shakespeare --epochs 4
    python examples/train_lm.py --tokens data/shakespeare --epochs 1 --steps-per-epoch 20 --num-layers 1   # smoke run
"""
import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import tyro

from dew.data.dataloaders import get_token_dataset_grain
from dew.data.text import ByteTokenizer
from dew.objectives.lm import LMObjective
from dew.registry import apply_precision_policy, build_model
from dew.sampling.text import generate
from dew.training import ObjectiveTrainer


@dataclass
class Config:
    tokens: Path = Path("data/shakespeare")
    sequence_length: int = 256
    batch_size: int = 64
    epochs: int = 4
    steps_per_epoch: int = 300
    learning_rate: float = 1e-3
    emb_features: int = 384
    num_layers: int = 6
    num_heads: int = 6
    prompt: str = "ROMEO:"
    sample_tokens: int = 300
    out: Path = Path("runs/shakespeare")


def main(config: Config):
    meta = json.loads((config.tokens / "meta.json").read_text())
    tokenizer = ByteTokenizer()
    data = get_token_dataset_grain(
        config.tokens / "train.bin", config.tokens / "val.bin",
        batch_size=config.batch_size, seq_len=config.sequence_length, worker_count=4,
    )

    # The KV cache is sized when the model is built, so the context covers the longest sample.
    model_config = apply_precision_policy("causal_transformer", dict(
        vocab_size=meta["vocab_size"], emb_features=config.emb_features,
        num_layers=config.num_layers, num_heads=config.num_heads,
        max_seq_len=max(config.sequence_length, len(config.prompt) + config.sample_tokens),
    ), dtype="bfloat16", attention_impl="auto")
    model = build_model("causal_transformer", model_config)
    objective = LMObjective(model, config.sequence_length, vocab_size=meta["vocab_size"])

    trainer = ObjectiveTrainer(
        model, optax.adamw(config.learning_rate), objective=objective, input_config=None,
        rngs=jax.random.PRNGKey(0), name=config.out.name, checkpoint_base_path=str(config.out / "checkpoints"),
    )
    state = trainer.fit(data, training_steps_per_epoch=config.steps_per_epoch, epochs=config.epochs)

    prompt = jnp.asarray([tokenizer.encode(config.prompt)], jnp.int32)
    tokens = generate(model, state.ema_params, prompt, max_new_tokens=config.sample_tokens,
                      rng=jax.random.PRNGKey(0), temperature=0.8, top_k=40)
    text = tokenizer.decode(tokens[0])
    (config.out / "sample.txt").write_text(text)
    print(text)
    return state


if __name__ == "__main__":
    main(tyro.cli(Config))
