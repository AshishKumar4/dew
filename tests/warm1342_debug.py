"""Receipt warm1342.py, with only memory_stats made safe for CPU replay."""

import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

compiles = []
jax.monitoring.register_event_duration_secs_listener(
    lambda event, seconds, *, fun_name=None, **_: compiles.append(round(seconds, 2))
    if event == "/jax/core/compile/backend_compile_duration" and fun_name == "jit(_stepped)" else None)

from dew.inference import TextGeneration  # noqa: E402
from dew.inference.serving import Server  # noqa: E402
from dew.nn.backbones.causal_transformer import CausalTransformer  # noqa: E402
from dew.sampling import Sampling  # noqa: E402

LAYERS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
SCAN = "scan" in sys.argv[2:] or "banked" in sys.argv[2:]
BANKED = "banked" in sys.argv[2:]
SMOKE = "smoke" in sys.argv[2:]
width, experts, expert_features = (64, 8, 32) if SMOKE else (2048, 128, 768)
model = CausalTransformer(
    vocab_size=151936 if not SMOKE else 512, emb_features=width, num_layers=LAYERS, num_heads=4 if SMOKE else 32,
    num_kv_heads=2 if SMOKE else 4, head_dim=16 if SMOKE else 128, mlp_features=4 * width, max_seq_len=1024,
    qk_norm=True, dtype=jnp.bfloat16, scan_layers=SCAN,
    mixture={"experts": experts, "top_k": 2 if SMOKE else 8, "expert_features": expert_features,
             "layers": tuple(range(LAYERS))})
began = time.perf_counter()
shapes = jax.eval_shape(model.init, jax.random.key(0), jnp.ones((1, 2), jnp.int32))
params = jax.jit(lambda key: jax.tree.map(
    lambda leaf, k: (jax.random.normal(k, leaf.shape, jnp.bfloat16) * 0.02).astype(jnp.bfloat16),
    shapes, jax.tree.unflatten(jax.tree.structure(shapes), list(jax.random.split(key, len(jax.tree.leaves(shapes)))))))(
    jax.random.key(0))
jax.block_until_ready(params)
if BANKED:
    from dew.inference.banks import HeldBanks
    stored = params
    params = jax.block_until_ready(HeldBanks(stored).place(model))
    for leaf in jax.tree.leaves(stored):
        leaf.delete()
    del stored
record = {"device": jax.devices()[0].device_kind, "layers": LAYERS, "scan": SCAN, "banked": BANKED,
          "init_s": round(time.perf_counter() - began, 1),
          "peak_gib": round((jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0) / 2**30, 1)}
task = TextGeneration(model, params, None, sampling=Sampling(temperature=0))
began = time.perf_counter()
server = Server.from_task(task, slots=64, capacity=512, admission=8)
record["build_s"], record["build_compiles"] = round(time.perf_counter() - began, 1), list(compiles)
rng = np.random.default_rng(0)
for round_, rows in enumerate((8, 1, 2, 4, 8, 1)):
    before = len(compiles)
    began = time.perf_counter()
    for result in server([rng.integers(100, 400, 256).tolist() for _ in range(rows)], 4, key=0):
        result.host()
    record[f"admit{round_}x{rows}_s"] = round(time.perf_counter() - began, 1)
    record[f"admit{round_}x{rows}_compiles"] = compiles[before:]
record["step_compiles"], record["compile_s"] = len(compiles), round(sum(compiles), 1)
prompts = [rng.integers(100, 400, 256).tolist() for _ in range(64)]
for result in server(prompts[:8], 8, key=0):
    result.host()
began = time.perf_counter()
drawn = sum(int(result.host().lengths[0]) for result in server(prompts, 64, key=0))
seconds = time.perf_counter() - began
record["steady_tokens"], record["steady_tok_s"] = drawn, round(drawn / seconds, 1)
record["steady_compiles"] = len(compiles) - record["step_compiles"]
print("WARM", json.dumps(record), flush=True)
