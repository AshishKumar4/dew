import itertools

import jax
import numpy as np
import optax

from dew import Dataset, Trainer, models
from dew.data import ByteTokenizer
from dew.objectives.lm import LMObjective
from dew.sampling import Sampling, generate

tokenizer = ByteTokenizer()
text = tokenizer.encode("dew trains jax models. " * 3)
batch = {"text": np.tile(np.asarray(text[:65], np.int32), (8, 1))}
data = Dataset(train=lambda partition: itertools.repeat(batch),
               val=None, records=8, batch=8)

model = models.build(
    "causal_transformer", vocab_size=tokenizer.vocab_size,
    emb_features=64, num_layers=2, num_heads=4,
    mlp_features=256, max_seq_len=128)
objective = LMObjective(model, seq_len=64)
trainer = Trainer(objective, optax.adamw(3e-3),
                  key=jax.random.key(0))
state = trainer.fit(data, steps=100, log_every=25)

prompt = [tokenizer.encode("dew")]
out = generate(model, state.params, prompt, max_new_tokens=40,
               key=jax.random.key(1),
               sampling=Sampling(temperature=0))
print(tokenizer.decode(out.tokens[0]))
