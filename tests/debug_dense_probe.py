"""Debug (not for landing): Dew's and the reference's float64 dense-decoder gradients, leaf by leaf."""
import sys
sys.path.insert(0, "tests")
import jax, jax.numpy as jnp, numpy as np, optax, torch
from reference_error import torch_with_gradients, widened, with_gradients
from test_gemma3n import STEP_LENGTH, clipped_gemma3n, reference_decoder
from tools.diffusers_wan_reference import float64

loaded, model, clip = clipped_gemma3n()
sparsity = (0.0,) * len(model.activation_sparsity_pattern)
names = list(loaded.export(loaded.variables))
rows = np.random.default_rng([0xDE, 0]).integers(0, int(model.vocab_size), (4, STEP_LENGTH)).astype(np.int32)
with jax.enable_x64(True):
    dense = model.clone(activation_sparsity_pattern=sparsity, dtype=jnp.float64)
    params = widened(loaded.variables["params"])
    stored = dense.projected(params, updated=False)

    def loss(p):
        logits = dense.apply({"params": p}, jnp.asarray(rows[:, :-1]), train=True)
        return optax.softmax_cross_entropy_with_integer_labels(logits, jnp.asarray(rows[:, 1:])).mean()

    value, grads = jax.value_and_grad(loss)(stored)
    mine = loaded.export({"params": grads})
reference = reference_decoder(loaded.source, clip, torch.float64, sparsity)
tokens = torch.from_numpy(rows).long()
with float64():
    with torch.no_grad():
        reference(input_ids=tokens)
    held = {n: l.detach().numpy() for n, l in reference.named_parameters()}
    out, theirs = torch_with_gradients(lambda w: torch.func.functional_call(reference, w, (), {"input_ids": tokens, "labels": tokens}).loss)(held, np.float64(1))
print("LOSS dew", float(value), "ref", float(out))
for name in names:
    a, b = np.asarray(mine[name], np.float64), np.asarray(theirs[name], np.float64)
    rel = float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))
    if rel > 1e-9:
        extra = ""
        if a.ndim >= 2:
            za, zb = np.all(a == 0, axis=-1), np.all(b == 0, axis=-1)
            extra = f" rows zero dew {int(za.sum())} ref {int(zb.sum())} disagree {int((za != zb).sum())}"
        print("DIFF", name, a.shape, f"rel {rel:.2e}", extra)
