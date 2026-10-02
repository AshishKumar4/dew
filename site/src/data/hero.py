"""Train a byte decoder on Tiny Shakespeare, evaluate, and generate two samples."""
import argparse
import json
import urllib.request
from pathlib import Path

import jax
import numpy as np
import optax

from dew import Trainer
from dew.nn.backbones import CausalTransformer
from dew.data import ByteTokenizer, Loading, TokenWindows
from dew.objectives.lm import LMObjective, Perplexity
from dew.sampling import Sampling, generate

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=1000)
steps = parser.parse_args().steps

# Prepare a byte corpus once; the final 50,000 tokens are held out.
tokens = Path(__file__).with_name("tokens")
tokenizer = ByteTokenizer()
if not tokens.exists():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = urllib.request.urlopen(url).read().decode()
    ids = np.asarray(tokenizer.encode(text), np.uint16)
    tokens.mkdir()
    ids[:-50_000].tofile(tokens / "train.bin")
    ids[-50_000:].tofile(tokens / "val.bin")
    (tokens / "meta.json").write_text(json.dumps({
        "tokenizer": "byte", "vocab_size": tokenizer.vocab_size, "dtype": "uint16",
        "train_tokens": len(ids) - 50_000, "val_tokens": 50_000, "eos_id": tokenizer.eos_id,
    }))

data = TokenWindows(path=str(tokens), seq_len=256, stride=1, val_batches=None,
                    loading=Loading(workers=0, threads=1, read_buffer=2)).load(batch=64)
model = CausalTransformer(vocab_size=tokenizer.vocab_size,
                     emb_features=384, num_layers=6, num_heads=6, mlp_features=1024, max_seq_len=256,
                     dropout_rate=0.2, embedding_dropout_rate=0.2, attention_dropout_rate=0.2,
                     qk_norm=False, initializer_range=0.02, depth_scaled_init=True)
objective = LMObjective(model, seq_len=256, ema_decay=None)
optimizer = optax.chain(optax.clip_by_global_norm(1), optax.adamw(
    optax.warmup_cosine_decay_schedule(0., 1e-3, min(100, steps - 1), steps, end_value=1e-4),
    b2=0.99, weight_decay=0.1, mask=lambda params: jax.tree.map(lambda p: p.ndim >= 2, params)))
trainer = Trainer(objective, optimizer, key=jax.random.key(0))
state = trainer.fit(data, steps=steps, log_every=100, eval_every=500, metrics=(Perplexity(),))

for index, prompt in enumerate(("ROMEO:", "JULIET:")):
    result = generate(model, state.variables, [tokenizer.encode(prompt)], max_new_tokens=200,
                      key=jax.random.key(index), sampling=Sampling(
                          temperature=0.5, top_k=40, eos_id=(46, 33, 63), pad_id=32))
    print(tokenizer.decode(result.tokens[0]), end="\n\n")
