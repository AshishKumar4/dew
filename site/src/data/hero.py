import urllib.request

import jax
import numpy as np
import optax

from dew import Dataset, Trainer, models
from dew.data import ByteTokenizer
from dew.objectives.lm import LMObjective
from dew.sampling import Sampling, generate

url = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
       "data/tinyshakespeare/input.txt")
tokenizer = ByteTokenizer()
ids = np.asarray(tokenizer.encode(urllib.request.urlopen(url).read().decode()), np.int32)


def windows(partition):
    """Random 257-byte windows of the text: 256 inputs and their next bytes."""
    rng = np.random.default_rng(partition.index)
    while True:
        starts = rng.integers(0, len(ids) - 257, partition.rows(64))
        yield {"text": np.stack([ids[s:s + 257] for s in starts])}


data = Dataset(train=windows, val=None, records=len(ids) // 257, batch=64)
model = models.build("causal_transformer", vocab_size=tokenizer.vocab_size,
                     emb_features=384, num_layers=6, num_heads=6, dropout_rate=0.2,
                     max_seq_len=512, dtype="bfloat16")
objective = LMObjective(model, seq_len=256, ema_decay=0.999)
schedule = optax.warmup_cosine_decay_schedule(0.0, 1e-3, 150, 1500, 1e-4)
trainer = Trainer(objective, optax.adamw(schedule), key=jax.random.key(0))
state = trainer.fit(data, steps=1500, log_every=150)

prompt = [tokenizer.encode("ROMEO:")]
out = generate(model, state.averaged, prompt, max_new_tokens=300, key=jax.random.key(1),
               sampling=Sampling(temperature=0.8, top_k=40))
print(tokenizer.decode(out.tokens[0]))
