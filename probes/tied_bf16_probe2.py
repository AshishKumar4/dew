"""Probe 2: the tied bf16 head's worst token, term by term."""
import sys
sys.path.insert(0, "tests")
import jax, jax.numpy as jnp, numpy as np, optax, ml_dtypes
import test_chunked_cross_entropy as t
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.lm import chunked
from dew.objectives.lm.chunked import chunked_cross_entropy

dtype = jnp.bfloat16
for tied in (True, False):
    model = t.small_model(tie_embeddings=tied, dtype=dtype)
    rng = jax.random.PRNGKey(0)
    ids = jax.random.randint(rng, (2, 12), 0, 97)
    variables = model.init(rng, ids)
    table = chunked.head_table(model, variables)
    seed = 7
    keys = jax.random.split(jax.random.fold_in(jax.random.key(1), seed))
    drawn = jax.random.randint(keys[0], (2, 12), 0, 97)
    hidden = model.apply(variables, drawn, method=CausalTransformer.hidden_states)
    targets = jax.random.randint(keys[1], (2, 12), 0, 97)
    with jax.enable_x64(True):
        truth = np.asarray(t.true_losses(table, np.asarray(hidden, np.float64), targets, dtype))
        true_logits = np.asarray(t.true_logits(table, np.asarray(hidden, np.float64), dtype))
    with jax.default_matmul_precision("default"):
        logits = np.asarray(table.logits(hidden), np.float64)
        whole = np.asarray(optax.softmax_cross_entropy_with_integer_labels(table.logits(hidden), targets), np.float64)
        results = {c: [np.asarray(v, np.float64) for v in jax.jit(lambda h, y, c=c: chunked_cross_entropy(h, table, y, c))(hidden, targets)]
                   for c in (1, 4)}
    print(f"tied={tied} logits==truth logits: {np.array_equal(logits, true_logits)}; logits range [{logits.min():.2f}, {logits.max():.2f}]")
    for c, (losses, predicted, log_z) in results.items():
        err = np.abs(losses - truth)
        k = np.unravel_index(err.argmax(), err.shape)
        true_z = np.log(np.sum(np.exp(true_logits - true_logits.max(-1, keepdims=True)), -1)) + true_logits.max(-1)
        print(f"  chunks={c}: worst loss err {err.max():.3e} at {k}: loss {truth[k]:.6f} chunked {losses[k]:.9f} whole {whole[k]:.9f}"
              f" | log Z err {np.abs(log_z - true_z).max():.3e} | target logit {true_logits[k][targets[k]]:.4f} peak {true_logits[k].max():.4f}")
    print(f"  whole worst loss err {np.abs(whole - truth).max():.3e}")
