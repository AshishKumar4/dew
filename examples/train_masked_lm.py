"""Train MDLM on real byte-tokenized text, then unmask a sample.

Prepare WikiText or TinyStories with the existing tokenizer tool:

    dew tokenize --input data/wikitext.txt --out data/wikitext --tokenizer byte
    python examples/train_masked_lm.py --tokens data/wikitext --steps 2000
    python examples/train_masked_lm.py --tokens data/wikitext --smoke --out runs/mdlm-smoke

The mask is an extra vocabulary entry, not a byte the corpus can contain.
`--smoke` makes the model and run small; it still reads the supplied real
corpus. Its sample demonstrates the workflow, not language quality.
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import tyro

from dew.config import OptimConfig
from dew.data import ByteTokenizer, DataPartition, Loading, TokenCorpus, TokenWindows
from dew.diffusion.discrete import DiscreteProcess, LogLinear
from dew.inference import RunProcessor
from dew.nn.backbones import CausalTransformer
from dew.objectives.base import Step
from dew.objectives.diffusion.masked import MaskedDiffusionObjective
from dew.training import Checkpoints, Trainer


@dataclass
class Config:
    tokens: Path
    sequence_length: int = 256
    batch_size: int = 32
    steps: int = 2000
    learning_rate: float = 1e-3
    features: int = 384
    layers: int = 6
    heads: int = 6
    sample_tokens: int = 128
    sample_steps: int = 64
    prompt: str = "Once upon a time"
    out: Path = Path("runs/masked-lm")
    smoke: bool = False


def main(config: Config):
    if config.smoke:
        config = replace(config, sequence_length=64, batch_size=4, steps=8,
                         features=64, layers=2, heads=4, sample_tokens=32, sample_steps=8)
    corpus = TokenCorpus.read(config.tokens)
    if corpus.tokenizer != "byte" or corpus.vocab_size != 256:
        raise ValueError("this example expects a corpus prepared with --tokenizer byte")
    tokenizer = ByteTokenizer()
    # TokenWindows ordinarily yields S+1 ids for next-token training. MDLM
    # predicts the same S input positions, so request a window one id shorter.
    data = TokenWindows(path=str(config.tokens), seq_len=config.sequence_length - 1,
                        val_batches=1, loading=Loading(workers=0, threads=2)).load(batch=config.batch_size)
    prompt = tokenizer.encode(config.prompt)
    model = CausalTransformer(vocab_size=257, causal=False,
                              emb_features=config.features, num_layers=config.layers, num_heads=config.heads,
                              mlp_features=4 * config.features, dtype=jnp.float32,
                              precision=jax.lax.Precision.HIGHEST, attention_impl="reference",
                              max_seq_len=max(config.sequence_length, len(prompt) + config.sample_tokens))
    objective = MaskedDiffusionObjective(model, DiscreteProcess(LogLinear(), mask_id=256),
                                        config.sequence_length,
                                        ema_decay=None, steps=config.sample_steps, decode=tokenizer.decode)
    config.out.mkdir(parents=True, exist_ok=True)
    checkpoints = Checkpoints(str(config.out / "checkpoints"), keep=1)
    trainer = Trainer(objective, OptimConfig(learning_rate=config.learning_rate).build(config.steps),
                      key=jax.random.key(0),
                      checkpoints=checkpoints)
    stream = data.train(DataPartition())
    try:
        probe = next(stream)
    finally:
        stream.close()
    score = jax.jit(lambda params: objective.scalar_loss(params, probe,
                   Step(step=jnp.asarray(0), key=jax.random.key(7), ema=None))[0])
    initial_loss = float(score(trainer.initial_state().variables))
    state = trainer.fit(data, steps=config.steps, log_every=1, checkpoint_every=config.steps)
    checkpoints.wait()
    final_loss = float(score(state.variables))
    task = objective.pipeline(state, ema=False, processor=RunProcessor(tokenizer))
    generated = task(config.prompt, config.sample_tokens, key=1).text[0]
    (config.out / "sample.txt").write_text(config.prompt + generated + "\n")
    report = {"corpus": str(config.tokens), "train_tokens": corpus.train_tokens,
              "device": jax.devices()[0].device_kind, "steps": int(state.step),
              "updates": int(state.updates), "probe_nelbo_before": initial_loss,
              "probe_nelbo_after": final_loss, "sample": config.prompt + generated}
    (config.out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return state


if __name__ == "__main__":
    main(tyro.cli(Config))
