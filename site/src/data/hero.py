"""Train a small Shakespeare MoE on one GPU, evaluate, and generate two samples."""
import json
import urllib.request
from pathlib import Path

import jax
import numpy as np
import optax

from dew import Trainer, models
from dew.data import ByteTokenizer, Loading, TokenWindows
from dew.objectives.lm import LMObjective, Perplexity
from dew.sampling import Sampling, generate

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

data = TokenWindows(path=str(tokens), seq_len=128, val_batches=4,
                    loading=Loading(workers=0, threads=1, read_buffer=2)).load(batch=16)
model = models.build("causal_transformer", vocab_size=tokenizer.vocab_size,
                     emb_features=64, num_layers=2, num_heads=4, max_seq_len=256,
                     mixture={"experts": 8, "top_k": 2, "dispatch": "global"})
objective = LMObjective(model, seq_len=128, aux_loss_alpha=0.01, ema_decay=None)
trainer = Trainer(objective, optax.adamw(3e-3), key=jax.random.key(0))
state = trainer.fit(data, steps=200, log_every=10, eval_every=50, metrics=(Perplexity(),))

for index, prompt in enumerate(("ROMEO:", "JULIET:")):
    result = generate(model, state.params, [tokenizer.encode(prompt)], max_new_tokens=60,
                      key=jax.random.key(index), sampling=Sampling(temperature=0.8, top_k=40))
    print(tokenizer.decode(result.tokens[0]), end="\n\n")
