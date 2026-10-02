#!/usr/bin/env python3
"""Same-weight disk/host-bank logits and cached greedy decode on a checkpoint.

`depth` selects a prefix of real decoder layers, keeping real embeddings,
norm and head. This bounds the host-resident reference, not just the disk
case, on memory-limited machines. Separate processes can record each mode's
outputs to avoid holding host banks beside streaming staging buffers.
"""

from __future__ import annotations

import dataclasses
import json
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from dew import models
from dew.inference.banks import SafetensorsBanks, host_banked, layer_index, stream_banked
from dew.interop.hf_decoders import translate_config
from dew.objectives.base import merge
from dew.registry import with_precision
from dew.training import Layout
from dew.training.distributed import build_mesh


class PrefixBanks(SafetensorsBanks):
    """The checkpoint's first `depth` layers, without reading discarded weights."""

    def __init__(self, directory, *, depth, read_ahead):
        super().__init__(directory, cache_bytes=0, param_dtype="auto", read_ahead=read_ahead)
        self.depth = depth

    def shapes(self):
        return {collection: {name: leaf for name, leaf in tree.items()
                             if (index := layer_index(name)) is None or index < self.depth}
                for collection, tree in super().shapes().items()}


@dataclass(frozen=True)
class ParityConfig:
    checkpoint: Path
    output: Path
    mode: Literal["host", "disk"] = "disk"
    depth: int = 3
    dtype: str = "bfloat16"
    precision: str = "highest"
    prompts: tuple[tuple[int, ...], ...] = ((1, 2, 3, 4), (13, 17, 23, 29), (101, 3, 47, 5))
    tokens: int = 3
    reference: Path | None = None
    read_ahead: bool = True


def check(config):
    jax.config.update("jax_default_matmul_precision", config.precision)
    started = time.perf_counter()
    with PrefixBanks(config.checkpoint, depth=config.depth, read_ahead=config.read_ahead) as source:
        record = translate_config(source.config)
        if not 0 < config.depth <= record["num_layers"]:
            raise ValueError("depth must select a nonempty prefix of the decoder")
        record["num_layers"] = config.depth
        if record.get("layer_types") is not None:
            record["layer_types"] = record["layer_types"][:config.depth]
        record["max_seq_len"] = max(map(len, config.prompts)) + config.tokens + 1
        built = with_precision("causal_transformer", record, dtype=config.dtype, attention_impl="xla")
        model = models.build("causal_transformer", {**built, "scan_layers": True})
        loader = host_banked if config.mode == "host" else stream_banked
        layout = Layout(min_shard=1, tolerance=1.0,
                        host_parameters=("params/layers_*",) if config.mode == "host" else ())
        variables = loader(model, source, mesh=build_mesh(devices=[jax.devices()[0]]), layout=layout)
        loaded_seconds = time.perf_counter() - started
        print("weights loaded", loaded_seconds, "peak RSS",
              resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, file=sys.stderr, flush=True)
        cache = model.apply(variables, 1, method=model.init_cache, mutable=["cache"])[1]
        prefill = jax.jit(lambda read, held, tokens: model.apply(
            merge(read, held), tokens, decode=True, mutable=["cache"]))
        decode = jax.jit(lambda read, held, token: model.apply(
            merge(read, held), token, decode=True, mutable=["cache"]))
        outputs = {}
        for index, prompt in enumerate(config.prompts):
            logits, held = prefill(variables, cache, jnp.asarray([prompt], jnp.int32))
            outputs[f"prefill_{index}"] = np.asarray(logits)
            emitted, scores = [], []
            for _ in range(config.tokens):
                token = jnp.argmax(logits[:, -1:], axis=-1).astype(jnp.int32)
                emitted.append(np.asarray(token))
                logits, held = decode(variables, held, token)
                scores.append(np.asarray(logits))
            outputs[f"tokens_{index}"] = np.concatenate(emitted, axis=1)
            outputs[f"decode_{index}"] = np.concatenate(scores, axis=1)
        if config.reference:
            with np.load(config.reference) as reference:
                for name, actual in outputs.items():
                    np.testing.assert_array_equal(actual, reference[name], err_msg=name)
        config.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(config.output, **outputs)
        print(json.dumps({"device": jax.devices()[0].device_kind, "config": dataclasses.asdict(config),
                          "load_seconds": loaded_seconds, "elapsed_seconds": time.perf_counter() - started,
                          "bitwise_equal": config.reference is not None,
                          "host_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                          "device_memory": jax.devices()[0].memory_stats(),
                          "reads": dataclasses.asdict(source.stats())}, default=str))


if __name__ == "__main__":
    check(tyro.cli(ParityConfig))
