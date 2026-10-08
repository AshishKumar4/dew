"""Train a byte decoder on Tiny Shakespeare, evaluate, and generate two samples."""
import argparse

from dew import Trainer
from dew.config import OptimConfig
from dew.data import ByteTokenizer, load
from dew.nn.backbones import CausalTransformer
from dew.objectives.lm import LMObjective, Perplexity
from dew.sampling import Sampling, generate
from dew.training.optim import Cosine

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=1000)
steps = parser.parse_args().steps

data = load("hf/winglian/tiny-shakespeare", batch=64, tokenizer="byte", seq_len=256)
model = CausalTransformer(vocab_size=256, emb_features=384, num_layers=6, num_heads=6,
                          mlp_features=1024, max_seq_len=256, dropout_rate=0.2)
optimizer = OptimConfig(schedule=Cosine(peak=1e-3, warmup_steps=10, end=1e-4),
                        b2=0.99, weight_decay=0.1, clip_grads=1.0)
trainer = Trainer(LMObjective(model, seq_len=256), optimizer, key=0)
state = trainer.fit(data, steps=steps, log_every=100, eval_every=500, metrics=(Perplexity(),))

tokenizer = ByteTokenizer()
for key, prompt in enumerate(("ROMEO:", "JULIET:")):
    result = generate(model, state.variables, [tokenizer.encode(prompt)], max_new_tokens=200, key=key,
                      sampling=Sampling(temperature=0.5, top_k=40, eos_id=(46, 33, 63), pad_id=32))
    print(tokenizer.decode(result.tokens[0]), end="\n\n")
