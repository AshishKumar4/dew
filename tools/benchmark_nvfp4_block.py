#!/usr/bin/env python3
"""One pinned Qwen3-8B NVFP4 decoder block, range-read and measured on the 4080.

The smallest RedHatAI Qwen3 local-NVFP4 model decodes to 15.3 GiB of bf16
weights alone, above the lane's 90%-of-16GB budget. This measures its real
layer 0, not full-model throughput. Training is one 1x2048 forward/backward
with fp32 master weights; decode is batch eight, one token, with a 2048-slot
synthetic KV cache. Each pair alternates order over ten warmed measurements.
Both compute dtypes keep the model's normal Dense precision policy; fp32
asks for highest to match full-fp32 products.

    PYTHONPATH=src python tools/benchmark_nvfp4_block.py fetch SCRATCH_DIR
    PYTHONPATH=src ~/.cache/dew/dew-gpu-run --short \
        /home/mrwhite0racle/Desktop/dew/.venv/bin/python \
        tools/benchmark_nvfp4_block.py run SCRATCH_DIR float32

Repeat run with bfloat16. Fetch uses idle I/O and stores Hub metadata in
the default HF cache. Range-read block weights stay in SCRATCH_DIR, since
putting an incomplete shard in the Hub cache would corrupt that cache.
"""

import json
import struct
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import requests
from huggingface_hub import hf_hub_download, hf_hub_url

from dew.interop import hf_decoders
from dew.interop.codecs import source_quantization
from dew.interop.safetensors_io import _STORED_DTYPES, read_weights, save_hf_layout
from dew.nn.backbones import CausalTransformer
from dew.registry import from_record
from dew.training.quantization import NVFP4Input, checkpoint_input_quantization, nvfp4_input_qdq

REPO = "RedHatAI/Qwen3-8B-NVFP4"
REVISION = "e391349c110709b87bfc2ad2fde3f50dc5839fd8"
PREFIX = "model.layers.0."


def fetch(url: str, start: int, end: int) -> bytes:
    response = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
    response.raise_for_status()
    if response.status_code != 206 or len(response.content) != end - start + 1:
        raise ValueError(f"Hub did not return the requested byte range: {response.status_code}")
    return response.content


def download(directory: Path) -> None:
    config = json.loads(Path(hf_hub_download(REPO, "config.json", revision=REVISION)).read_text())
    index_path = Path(hf_hub_download(REPO, "model.safetensors.index.json", revision=REVISION))
    index = json.loads(index_path.read_text())["weight_map"]
    headers, weights = {}, {}
    for name, shard in index.items():
        if not name.startswith(PREFIX):
            continue
        url = hf_hub_url(REPO, shard, revision=REVISION)
        if shard not in headers:
            size = struct.unpack("<Q", fetch(url, 0, 7))[0]
            headers[shard] = size, json.loads(fetch(url, 8, size + 7))
        size, header = headers[shard]
        meta = header[name]
        start, end = meta["data_offsets"]
        weights[name] = np.frombuffer(fetch(url, 8 + size + start, 8 + size + end - 1),
                                      _STORED_DTYPES[meta["dtype"]]).reshape(meta["shape"])
    save_hf_layout(weights, config, directory)
    print(REPO, REVISION, "layer 0", len(weights), "tensors", sum(value.nbytes for value in weights.values()))


def paired(functions: dict, arguments: dict, iterations: int) -> dict:
    times = {name: [] for name in functions}
    for _ in range(3):
        for name, function in functions.items():
            jax.block_until_ready(function(*arguments[name]))
    names = tuple(functions)
    for turn in range(iterations):
        for name in names if turn % 2 == 0 else names[::-1]:
            start = time.perf_counter()
            jax.block_until_ready(functions[name](*arguments[name]))
            times[name].append(time.perf_counter() - start)
    return {name: {"median_ms": 1000 * float(np.median(values)),
                   "min_ms": 1000 * min(values), "max_ms": 1000 * max(values)}
            for name, values in times.items()}


def benchmark(directory: Path, dtype: str) -> None:
    jax.config.update("jax_default_matmul_precision", "highest")
    config = json.loads((directory / "config.json").read_text())
    stored = read_weights(directory)
    codec = source_quantization(config)
    assert codec is not None and codec.input_scale_dtype == "unrounded"
    decoded = codec.dequantize(stored)
    record = hf_decoders.translate_config(config)
    record["max_seq_len"] = 2048
    decoder = from_record(CausalTransformer, {**record, "dtype": dtype, "attention_impl": "reference"})
    plain = decoder.bind({}).layers[0].clone(parent=None, name=None)
    parameters = hf_decoders.translate_weights(decoded, record, "qwen3")["params"]["layers_0"]
    variables = {"params": jax.tree.map(jnp.asarray, parameters)}
    inputs = {name.removeprefix(PREFIX).removesuffix('.input_global_scale').replace('.', '/'):
              NVFP4Input(float(value.reshape(())), e4m3_scale=False)
              for name, value in stored.items() if name.endswith('.input_global_scale')}
    models = {"weight_only": plain, "qdq": checkpoint_input_quantization(plain, inputs)}
    del decoded, parameters, stored
    random = np.random.default_rng(2048)
    x = jnp.asarray(random.standard_normal((1, 2048, 4096)).astype(np.float32), getattr(jnp, dtype))
    steps, compile_times = {}, {}
    for name, model in models.items():
        def loss(params, model=model):
            output = model.apply({"params": params}, x, train=True)
            return jnp.mean(jnp.square(output.astype(jnp.float32)))

        start = time.perf_counter()
        steps[name] = jax.jit(jax.value_and_grad(loss)).lower(variables["params"]).compile()
        compile_times[name] = time.perf_counter() - start
    train = paired(steps, dict.fromkeys(models, (variables["params"],)), 10)
    token = jnp.asarray(random.standard_normal((8, 1, 4096)).astype(np.float32), getattr(jnp, dtype))
    cache = plain.apply(variables, token, decode=True, mutable=["cache"])[1]["cache"]

    def populated(path, value):
        key = jax.tree_util.keystr(path)
        if key.endswith("['cache_index']"):
            return jnp.full(value.shape, 2047, value.dtype)
        if "cached_key" in key or "cached_value" in key:
            return jnp.asarray(random.standard_normal(value.shape).astype(np.float32), value.dtype)
        return value

    cache = jax.tree_util.tree_map_with_path(populated, cache)
    calls = {}
    for name, model in models.items():
        calls[name] = jax.jit(lambda state, model=model:
                             model.apply(state, token, decode=True, mutable=["cache"]))
    decode = paired(calls, {name: ({**variables, "cache": cache},) for name in models}, 30)
    for value in decode.values():
        value["one_block_tokens_per_s"] = 8000 / value["median_ms"]
    spec = inputs["self_attn/q_proj"]
    quantize = jax.jit(lambda values: nvfp4_input_qdq(values, spec))
    qdq = paired({"training": quantize, "decode": quantize},
                 {"training": (x,), "decode": (token,)}, 30)
    print(json.dumps({"repo": REPO, "revision": REVISION, "block": 0, "dtype": dtype,
                      "param_dtype": "float32", "matmul_precision": "highest",
                      "device": str(jax.devices()[0]), "jax": jax.__version__,
                      "training_shape": [1, 2048, 4096], "decode_shape": [8, 1, 4096],
                      "cache_context": 2048, "train_compile_s": compile_times,
                      "forward_backward": train, "decode": decode,
                      "one_4096_wide_projection_input_qdq": qdq}, indent=2))


if __name__ == "__main__":
    directory = Path(sys.argv[2])
    if sys.argv[1] == "fetch":
        download(directory)
    else:
        benchmark(directory, sys.argv[3])
