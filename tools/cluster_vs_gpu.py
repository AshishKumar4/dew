#!/usr/bin/env python3
"""One training run, the same on a CPU cluster and on one GPU, timed to a target loss.

    # an armada gang of N containers, one process each:
    echo '[{"gang": 64}]' | armada map --commit=<sha> --items=- -- \\
        .venv-3.12/bin/python tools/cluster_vs_gpu.py --shape dense --out /tmp/out.json
    # one A100:
    python tools/cluster_vs_gpu.py --shape dense --out a100.json

Every shape is a byte-level decoder (`CausalTransformer`) trained with AdamW on TinyStories'
validation text (pinned by revision), at one global batch and learning rate however many devices
hold it, so its loss curve is the model's own and the platforms differ only in how long each step
takes. A run stops at `--target` (the mean of the last 10 training losses) or after `--steps`;
compilation is reported apart. In a gang (ARMADA_WORLD > 1) every rank joins jax.distributed at
rank0 and the mesh is the shape's layout over every host's device; on one host, the same model
runs on the one device.

The shapes:
  dense   a 14M-parameter decoder, data parallel.
  fsdp    a 1.6B-parameter decoder, its parameters and optimizer state sharded over every device
          (19 GB of fp32 parameters and Adam moments: more than one 12 GiB container holds).
  moe     a decoder of 64 experts, top 2, an expert per device.
  pipe    a 32-layer decoder in stages over the hosts, 16 microbatches (more than one device).

Rank 0 writes one JSON record: the shape, the platform and its devices, the parameter count, the
steps run, each step's loss and seconds, the step and the time the target was reached, and
compilation's seconds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

CORPUS = ("https://huggingface.co/datasets/roneneldan/TinyStories/resolve/"
          "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/TinyStories-valid.txt")
"""TinyStories' validation split, about 19 MB of English, at a pinned revision."""

SEQ_LEN = 256
SHAPES = {
    "dense": {"model": {"emb_features": 384, "num_layers": 6, "num_heads": 6, "mlp_features": 1536},
              "mesh": {}, "batch": 256, "lr": 1e-3, "target": 1.2},
    "fsdp": {"model": {"emb_features": 2048, "num_layers": 24, "num_heads": 16, "mlp_features": 8192},
             "mesh": {"fsdp": -1}, "batch": 64, "lr": 2e-4, "target": 1.2},
    "moe": {"model": {"emb_features": 384, "num_layers": 6, "num_heads": 6, "mlp_features": 1536,
                      "mixture": {"experts": 64, "top_k": 2, "expert_features": 512, "layers": (1, 3, 5)}},
            "mesh": {"expert": -1}, "batch": 256, "lr": 1e-3, "target": 1.2},
    "pipe": {"model": {"emb_features": 512, "num_layers": 32, "num_heads": 8, "mlp_features": 2048},
             "mesh": {"stage": -1}, "microbatches": 16, "batch": 128, "lr": 5e-4, "target": 1.2},
}


def corpus(cache: Path) -> Path:
    """The pinned text as a produced ArrayRecord token store."""
    import numpy as np

    from dew.data import ByteTokenizer
    from dew.data.sinks import export
    from dew.data.storage import EXPORT_MANIFEST

    tokens = cache / "tokens"
    if (tokens / EXPORT_MANIFEST).is_file():
        return tokens
    cache.mkdir(parents=True, exist_ok=True)
    text = cache / "TinyStories-valid.txt"
    if not text.is_file():
        part = text.with_suffix(".part")
        urllib.request.urlretrieve(CORPUS, part)
        part.rename(text)
    print("corpus sha256", hashlib.sha256(text.read_bytes()).hexdigest(), flush=True)
    ids = np.asarray(ByteTokenizer().encode(text.read_text()), np.int32)
    held = int(len(ids) * 0.01)
    metadata = {"token_store": True, "tokenizer": "byte", "vocab_size": 256,
                "dtype": "int32", "eos_id": None}
    export([{"text": ids[held:]}], str(tokens), format="array_record", metadata=metadata, length_field="text")
    export([{"text": ids[:held]}], str(cache / "held_tokens"), format="array_record", metadata=metadata,
           length_field="text")
    return tokens


def windows(tokens: Path, batch: int):
    """The token store's stream cut into windows of SEQ_LEN, in a run's order, in batches of `batch`,
    read into memory."""
    from dew.data import presets
    from dew.data.compile import compile
    from dew.data.sources import TokenFiles

    order = TokenFiles(str(tokens)).shuffle(0).repeat()
    return compile(order.pipe(presets.token_windows, seq_len=SEQ_LEN, batch=batch), read="memory")


def collectives(hlo: str) -> dict:
    """Each kind of collective (`dew.telemetry.trace.COLLECTIVES`) in compiled HLO text: how many ops
    there are, how many buffers they move (a combined collective is one op with a tuple of them),
    their bytes, and the sizes of the five largest ops. An async pair counts once, by its `-done`."""
    from dew.telemetry.instrumentation import hlo_instructions
    from dew.telemetry.trace import collective

    found: dict = {}
    for instruction in hlo_instructions(hlo).values():
        kind = collective(instruction.op)
        if kind is None or instruction.op.endswith("-start"):
            continue
        size = sum(instruction.buffers)
        entry = found.setdefault(kind, {"ops": 0, "buffers": 0, "bytes": 0, "sizes": []})
        entry["ops"] += 1
        entry["buffers"] += len(instruction.buffers)
        entry["bytes"] += size
        entry["sizes"].append(size)
    for entry in found.values():
        entry["sizes"] = sorted(entry["sizes"], reverse=True)[:5]
    return found


def compiled_step(executable) -> dict:
    """A compiled step's collectives (`collectives`) and the memory it holds."""
    memory = executable.memory_analysis()
    return {"collectives": collectives(executable.as_text()), "temp_GB": memory.temp_size_in_bytes / 1e9,
            "arguments_GB": memory.argument_size_in_bytes / 1e9}


def mesh_of(layout: dict, devices: int) -> dict:
    """The shape's layout over `devices`: an axis given as -1 takes every device."""
    return {axis: devices if size == -1 else size for axis, size in layout.items()}


def joined_pool(world: int, rank: int) -> float:
    """The seconds this process took to join a pool of `world`, one process a host at rank0, as
    `dew launch` leaves one; none on one host."""
    began = time.monotonic()
    from dew.clusters import Launched

    if world > 1:
        os.environ.update({**Launched.environ_of("rank0:8476", world, rank),
                           "JAX_CPU_COLLECTIVES_IMPLEMENTATION": "gloo"})
    import jax

    from dew.training.runtime import prepare_process

    # The compile seconds reported apart are a cold compile's, which no cached executable stands in for.
    jax.config.update("jax_enable_compilation_cache", val=False)
    prepare_process(multi_host=world > 1)
    return time.monotonic() - began


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=sorted(SHAPES), required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seconds", type=float, default=float("inf"),
                        help="stop after this many seconds of steps, to time a step the run cannot finish")
    parser.add_argument("--batch", type=int, help="the global batch; the shape's own by default")
    parser.add_argument("--compile-only", action="store_true",
                        help="print the step's collectives and memory, and train nothing")
    parser.add_argument("--target", type=float, help="the loss to stop at; the shape's own by default")
    parser.add_argument("--dtype", default=None, help="compute dtype; float32 on CPU, bfloat16 on a GPU")
    parser.add_argument("--scan-layers", action="store_true",
                        help="the decoder's layers under one scan, so a step gathers a layer's weights at a "
                             "time")
    parser.add_argument("--diloco", type=int, metavar="H",
                        help="DiLoCo (Douillard et al. 2023): each host trains its own copy on its own "
                             "share of every global batch, and every H steps the hosts sync "
                             "(`dew.training.diloco`, one group a host)")
    parser.add_argument("--cache", type=Path,
                        default=Path(os.environ.get("TMPDIR", "/tmp")) / "cluster-bench")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    world = int(os.environ.get("ARMADA_WORLD", "1"))
    rank = int(os.environ.get("ARMADA_RANK", "0"))
    joined = joined_pool(world, rank)
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from jax.experimental import multihost_utils

    from dew.data.dataset import DataPartition
    from dew.distributed.batches import shard_batch
    from dew.nn.backbones import CausalTransformer
    from dew.objectives.lm import LMObjective
    from dew.training import Layout, MeshSpec, Trainer
    from dew.training.ahead import step_batch
    from dew.training.diloco import DiLoCo

    shape = SHAPES[args.shape]
    platform = jax.devices()[0].platform
    dtype = args.dtype or ("bfloat16" if platform == "gpu" else "float32")
    target = shape["target"] if args.target is None else args.target
    tokens = corpus(args.cache)
    batch_size = shape["batch"] if args.batch is None else args.batch
    data = windows(tokens, batch_size)
    model = CausalTransformer(**shape["model"], vocab_size=256, max_seq_len=SEQ_LEN + 1,
                              dtype=getattr(jnp, dtype), attention_impl="auto", scan_layers=args.scan_layers)
    trainer = Trainer(LMObjective(model, SEQ_LEN, ema_decay=None),
                      optax.adamw(shape["lr"], weight_decay=0.1), key=jax.random.key(0),
                      # Under DiLoCo one group a host, the shape's layout over the host's devices.
                      mesh=(MeshSpec(diloco=world, **mesh_of(shape["mesh"], jax.local_device_count()))
                            if args.diloco else MeshSpec(**mesh_of(shape["mesh"], jax.device_count()))),
                      microbatches=shape.get("microbatches"), layout=Layout(), checkpoints=None, tracker=None,
                      diloco=DiLoCo(every=args.diloco) if args.diloco else None)
    # Each process reads its own rows of every global batch, which shard_batch assembles: under DiLoCo a
    # group's share.
    partition = DataPartition.of(trainer.device_mesh)
    state, _, _ = trainer.place()
    parameters = sum(leaf.size for leaf in jax.tree.leaves(state.variables["params"]))
    batches = iter(data.train(partition))
    compiled, losses, seconds, reached, compile_seconds = None, [], [], None, 0.0
    for step in range(1, args.steps + 1):
        # The batch as `fit` hands it to its step (`step_batch`), so the step timed is a run's.
        batch = shard_batch(trainer.device_mesh, step_batch(trainer, next(batches)))
        start = time.monotonic()
        if compiled is None:
            compiled = trainer.compile(state, batch)
            compile_seconds = time.monotonic() - start
            if rank == 0:
                print(json.dumps(compiled_step(trainer.executable)), flush=True)
            if args.compile_only:
                args.out.with_suffix(".hlo.txt").write_text(trainer.executable.as_text())
                return
            start = time.monotonic()
        state, loss, _, _, _, _ = compiled(state, batch)
        loss = float(loss)
        seconds.append(time.monotonic() - start)
        losses.append(loss)
        if rank == 0 and (step % 10 == 0 or step < 5):
            print(f"step {step} loss {loss:.4f} {seconds[-1]:.3f} s", flush=True)
        # The mean of the last 10 losses; under DiLoCo the outer model is what reaches the target, read at
        # each meeting of the groups, every H steps, over the groups' mean losses since the last.
        window = args.diloco or 10
        meets = not args.diloco or step % args.diloco == 0
        if meets and len(losses) >= window and sum(losses[-window:]) / window <= target:
            reached = {"step": step, "seconds": sum(seconds)}
            break
        # Rank 0's clock decides, so every process stops at the same step.
        over = sum(seconds) > args.seconds
        if world > 1:
            over = bool(multihost_utils.broadcast_one_to_all(np.asarray(over)))
        if over:
            break
    if rank == 0:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "shape": args.shape, "platform": platform, "device_kind": jax.devices()[0].device_kind,
            "processes": jax.process_count(), "devices": jax.device_count(), "dtype": dtype,
            "parameters": int(parameters), "batch": batch_size, "seq_len": SEQ_LEN, "target": target,
            "join_seconds": joined, "compile_seconds": compile_seconds, "steps": len(losses),
            "reached": reached, "losses": losses, "step_seconds": seconds, "diloco": args.diloco}))
        print(json.dumps({"shape": args.shape, "devices": jax.device_count(), "reached": reached,
                          "median_step_s": sorted(seconds)[len(seconds) // 2]}), flush=True)


if __name__ == "__main__":
    main()
