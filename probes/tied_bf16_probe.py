"""Probe: where the chunked head's tied bf16 losses leave the whole logits' (test_bf16_states_from_the_backbone)."""
import sys
sys.path.insert(0, "tests")
import jax, jax.numpy as jnp, numpy as np, optax
import test_chunked_cross_entropy as t
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.protocols import OutputTable
from dew.objectives.lm import chunked
from dew.objectives.lm.chunked import chunked_cross_entropy

dtype = jnp.bfloat16
model = t.small_model(tie_embeddings=True, dtype=dtype)
rng = jax.random.PRNGKey(0)
ids = jax.random.randint(rng, (2, 12), 0, 97)
variables = model.init(rng, ids)
table = chunked.head_table(model, variables)
print("table", table.vocab_major, table.compute, table.precision, table.matrix.dtype, table.matrix.shape)
plain = OutputTable(t.head_matrix(model, variables), vocab_major=False)
arms = {
    "chunked model table": lambda h, y: chunked_cross_entropy(h, table, y, 4)[0],
    "chunked model table tile None": lambda h, y: chunked_cross_entropy(h, table, y, 4, tile=None)[0],
    "chunked plain table": lambda h, y: chunked_cross_entropy(h, plain, y, 4)[0],
    "whole model_logits": lambda h, y: optax.softmax_cross_entropy_with_integer_labels(chunked.model_logits(model, variables, h), y),
    "whole table.logits": lambda h, y: optax.softmax_cross_entropy_with_integer_labels(table.logits(h), y),
}
worst = {name: [] for name in arms}
for seed in range(40):
    keys = jax.random.split(jax.random.fold_in(jax.random.key(1), seed))
    drawn = jax.random.randint(keys[0], (2, 12), 0, 97)
    hidden = model.apply(variables, drawn, method=CausalTransformer.hidden_states)
    targets = jax.random.randint(keys[1], (2, 12), 0, 97)
    with jax.enable_x64(True):
        truth = np.asarray(t.true_losses(table, np.asarray(hidden, np.float64), targets, dtype))
    with jax.default_matmul_precision("default"):
        for name, arm in arms.items():
            error = np.abs(np.asarray(jax.jit(arm)(hidden, targets), np.float64) - truth)
            worst[name].append(error.max())
    if seed in (0, 1):
        with jax.default_matmul_precision("default"):
            a = np.asarray(table.logits(hidden)); b = np.asarray(chunked.model_logits(model, variables, hidden))
            c = np.asarray(plain.logits(hidden))
        print("seed", seed, "table.logits==model_logits", np.array_equal(a, b), "plain==model", np.array_equal(c, b),
              "differing entries", int((a != c).sum()))
for name, values in worst.items():
    values = np.asarray(values)
    print(f"{name:32s} rms-of-worst {np.sqrt(np.mean(values**2)):.3e} max {values.max():.3e} at seed {values.argmax()}")
