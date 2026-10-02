import urllib.request

import jax
import numpy as np
import optax

from dew import Dataset, Layout, MeshSpec, Trainer, models
from dew.data import ByteTokenizer
from dew.inference import RunProcessor, TextGeneration
from dew.inference.serving import Server
from dew.objectives.lm import LMObjective, Perplexity
from dew.sampling import Sampling

# Eight CPU devices in one process, standing in for eight accelerators.
jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_num_cpu_devices", 8)

url = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
       "data/tinyshakespeare/input.txt")
tokenizer = ByteTokenizer()
ids = np.asarray(tokenizer.encode(urllib.request.urlopen(url).read().decode()), np.int32)
train, val = ids[:-50_000], ids[-50_000:]


def windows(partition):
    """Random 129-byte windows of the training text: 128 inputs and their next bytes."""
    rng = np.random.default_rng(partition.index)
    while True:
        starts = rng.integers(0, len(train) - 129, partition.rows(32))
        yield {"text": np.stack([train[s:s + 129] for s in starts])}


def held_out(partition):
    """The held-out text in order, 32 windows at a time."""
    rows = np.lib.stride_tricks.sliding_window_view(val, 129)[::129][:256]
    for batch in rows.reshape(-1, 32, 129):
        yield {"text": batch[partition.index::partition.count]}


data = Dataset(train=windows, val=held_out, records=len(train) // 129, batch=32)

# Every layer routes each token to 2 of 8 experts. The mesh splits the batch,
# the experts and the weights two ways each, and the tokens travel to their
# experts' devices in an all-to-all.
model = models.build("causal_transformer", vocab_size=tokenizer.vocab_size,
                     emb_features=128, num_layers=4, num_heads=4, max_seq_len=256,
                     mixture={"experts": 8, "top_k": 2, "dispatch": "exchange"})
objective = LMObjective(model, seq_len=128, aux_loss_alpha=0.01, ema_decay=None)
trainer = Trainer(objective, optax.adamw(3e-3), key=jax.random.key(0),
                  mesh=MeshSpec(expert=2, fsdp=2), layout=Layout(min_shard=1))
state = trainer.fit(data, steps=300, log_every=10, eval_every=100, metrics=[Perplexity()])

# Serve the trained weights where they are, on the same mesh.
task = TextGeneration(model, state.params, RunProcessor(tokenizer),
                      sampling=Sampling(temperature=0.8, top_k=40))
server = Server.from_task(task, slots=8, capacity=256)
prompts = ["ROMEO:", "JULIET:"]
for prompt, generation in zip(prompts, server(prompts, 160, key=0)):
    print(prompt + generation.host().text[0], end="\n\n")
