"""Debug (not for landing): per-seed errors of the quantized kv_b_proj reference against its truth."""
import sys
import jax
import jax.numpy as jnp
import numpy as np
from reference_error import widened, with_gradients
from test_sparse_attention import HEADS, NOPE, RANK, ROPE, VALUE, hidden_draw

from dew.nn.kernels.generation import device_generation
from dew.nn.mla import MultiHeadLatentAttention
from dew.quantization import QUANTIZATION, InputQuantization

quantized = sys.argv[1]
layer = MultiHeadLatentAttention(
    emb_features=32, num_heads=HEADS, max_seq_len=64, q_lora_rank=16, kv_lora_rank=RANK,
    qk_nope_head_dim=NOPE, qk_rope_head_dim=ROPE, v_head_dim=VALUE,
    index_topk=5, index_n_heads=2, index_head_dim=16, attention_impl="reference")
variables = layer.init(jax.random.key(3), jnp.zeros((2, 21, 32)))
if quantized == "input":
    record = InputQuantization(sites=(("kv_b_proj", "fp8"),))
    held = {"kv_b_proj": {"fp8_input_scale": jnp.asarray(0.01, jnp.float32)}}
elif quantized == "none":
    record = None
    held = {}
else:
    record = InputQuantization(frozen_fp8=("kv_b_proj",), measured_on=device_generation())
    held = record.packed(variables["params"])
model = record.apply(layer) if record else layer

def run(params, qk):
    def call(params, x):
        tree = {"params": params, QUANTIZATION: held} if held else {"params": params}
        return model.apply(tree, x, mutable=["qk"])[0] if qk else model.apply(tree, x)
    return lambda hidden, cotangent: with_gradients(call)(params, hidden, cotangent)

params = variables["params"]
ref = jax.jit(run(params, True))
def latents(params, x):
    _, state = model.apply({"params": params, QUANTIZATION: held} if held else {"params": params}, x,
                           mutable=["qk", "intermediates"],
                           capture_intermediates=lambda m, name: m.name == "kv_a_layernorm")
    return jax.tree.leaves(state["intermediates"])[0]
for seed in range(int(sys.argv[2])):
    (hidden,) = hidden_draw(seed)
    cot = jax.random.normal(jax.random.fold_in(jax.random.key(4), seed), hidden.shape)
    got = ref(hidden, cot)
    with jax.enable_x64(True):
        truth = run(widened(params), True)(*widened((hidden, cot)))
        lat64 = np.asarray(latents(widened(params), widened(hidden)))
    lat32 = np.asarray(latents(params, hidden))
    out_err = float(np.sqrt(np.mean((np.asarray(got[0], np.float64) - np.asarray(truth[0])) ** 2)))
    codes32 = (lat32 * 100).astype(jnp.float8_e4m3fn).astype(np.float32)
    codes64 = (lat64.astype(np.float32) * 100).astype(jnp.float8_e4m3fn).astype(np.float32)
    print(f"SEED {seed} out_rms {out_err:.3e} latent_rel {np.max(np.abs(lat32 - lat64) / (np.abs(lat64) + 1e-30)):.2e} code_flips {int(np.sum(codes32 != codes64))}", flush=True)

if len(sys.argv) > 3:
    import reference_error as gate
    truth_fn = run(widened(params), True)
    def draw(seed):
        (hidden,) = hidden_draw(seed)
        return hidden, jax.random.normal(jax.random.fold_in(jax.random.key(4), seed), hidden.shape)
    first = gate._drawn(ref, ref, draw, range(8), truth_fn)
    second = gate._drawn(ref, ref, draw, range(8, 16), truth_fn)
    for index, name in enumerate(first.names):
        low, high = gate._interval(first.theirs[:, index], control=second.theirs[:, index])
        print("LEAF", name, "first", np.array2string(first.theirs[:, index], precision=2),
              "second", np.array2string(second.theirs[:, index], precision=2), f"interval {low:.3f} {high:.3f}",
              flush=True)
