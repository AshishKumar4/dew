"""Measure the per-generation kernel choices, one measurement per process.

Each mode prints one JSON line. `peak_bytes_in_use` is a process high-water
mark, so a row is one process.

- `projection`: `dew.nn.moe.expert_projection` forward and forward plus
  backward, at lm-moe's shape (8192 rows, 8 experts, bf16 compute, Dirichlet
  routing), against a float64 oracle of the rounded operands.
- `adam`: one AdamW update over the lm-dense parameter tree, fp32 or bf16
  moment storage (`OptimConfig.state_dtype`).
- `step`: a whole training step of `lm-dense` (359.8M) or `lm-moe` (321.8M,
  8 experts top-2), through `tools/benchmark_step.py`'s trainer, with the
  grouped matmul, the state dtype and the vocabulary head chosen.
- `mxfp4-projection`: one expert projection at gpt-oss-20b's shapes (32
  experts, 2880 to 5760 or 2880 to 2880) over bf16 experts or MXFP4 ones
  (`dew.nn.moe.MXFP4Experts`): the Pallas kernel that decodes MXFP4 tiles
  (`fused`), every expert decoded and then the bf16 grouped matmul
  (`decoded`), or the routed experts gathered, decoded and multiplied
  (`gathered`), with whether the output is the bf16 path's bit for bit.
- `mxfp4-decode`: greedy decoding of a gpt-oss-20b-shaped decoder cut to
  `--layers` layers, random weights drawn as MXFP4 and held as bf16 or as
  MXFP4: milliseconds a token, parameter bytes, the process's peak, and
  the prefill logits and tokens to `--out` for a bitwise comparison.

The docs/performance.md section "Kernel choices per generation" was measured
with these commands, for example:

    PYTHONPATH=src python tools/benchmark_kernels.py projection --implementation pallas
    PYTHONPATH=src python tools/benchmark_kernels.py adam --state-dtype bfloat16
    PYTHONPATH=src python tools/benchmark_kernels.py step --path lm-moe --batch 4 \\
        --implementation auto --state-dtype bfloat16
    PYTHONPATH=src python tools/benchmark_kernels.py mxfp4-projection --path fused --rows 4
    PYTHONPATH=src python tools/benchmark_kernels.py mxfp4-decode --layers 8 --storage mxfp4 --out mxfp4.npz
"""

import argparse
import json
import time
import zlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from benchmark_cases import Case
from benchmark_models import batches, build_trainer

from dew.config import OptimConfig
from dew.interop.codecs import MXFP4_PARTS
from dew.nn.moe import MXFP4Experts, expert_projection
from dew.telemetry.profile import capture_options
from dew.training.distributed import DevicePrefetchIterator

VOCAB = 50304
SEQUENCE = 1024


def timed(function, arguments, repeats: int) -> dict[str, float]:
    """Ratio, minimum and median wall time of `function`, each call synced."""
    for _ in range(5):
        jax.block_until_ready(function(*arguments))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(function(*arguments))
        samples.append((time.perf_counter() - start) * 1e3)
    return {"mean_ms": float(np.mean(samples)), "min_ms": float(np.min(samples)),
            "median_ms": float(np.median(samples))}


def peak_bytes() -> int | None:
    stats = jax.local_devices()[0].memory_stats() or {}
    return stats.get("peak_bytes_in_use")


def projection(args: argparse.Namespace) -> dict[str, object]:
    rows, experts = 8192, 8
    width, features = (768, 2048) if args.shape == "up" else (2048, 768)
    rng = np.random.default_rng(0)
    sizes = np.floor(rng.dirichlet(np.ones(experts)) * rows).astype(np.int32)
    sizes[-1] += rows - sizes.sum()
    x = jnp.asarray(rng.normal(size=(rows, width)), jnp.bfloat16)
    kernel = jnp.asarray(rng.normal(size=(experts, width, features)) / np.sqrt(width), jnp.float32)
    cotangent = jnp.asarray(rng.normal(size=(rows, features)), jnp.bfloat16)
    group_sizes = jnp.asarray(sizes)

    def project(x, kernel):
        return jnp.asarray(expert_projection(x, kernel, group_sizes, jnp.bfloat16,
                                             args.implementation, None))

    def loss(x, kernel):
        return jnp.sum(project(x, kernel).astype(jnp.float32) * cotangent.astype(jnp.float32))

    forward = jax.jit(project)
    both = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))
    output = np.asarray(forward(x, kernel), np.float64)
    _, (dx, dk) = both(x, kernel)
    xr = np.asarray(x, np.float64)
    kr = np.asarray(kernel.astype(jnp.bfloat16), np.float64)
    cr = np.asarray(cotangent, np.float64)
    starts = np.cumsum(sizes) - sizes
    oracle, dx_oracle = np.zeros((rows, features)), np.zeros((rows, width))
    dk_oracle = np.zeros((experts, width, features))
    for expert in range(experts):
        block = slice(starts[expert], starts[expert] + sizes[expert])
        oracle[block] = xr[block] @ kr[expert]
        dx_oracle[block] = cr[block] @ kr[expert].T
        dk_oracle[expert] = xr[block].T @ cr[block]

    def relative(value, reference):
        return float(np.abs(np.asarray(value, np.float64) - reference).max() / np.abs(reference).max())

    return {"implementation": args.implementation, "shape": [rows, width, features, experts],
            "groups": sizes.tolist(), "forward": timed(forward, (x, kernel), args.repeats),
            "forward_backward": timed(both, (x, kernel), args.repeats),
            "forward_error": relative(output, oracle), "input_gradient_error": relative(dx, dx_oracle),
            "kernel_gradient_error": relative(dk, dk_oracle)}


def mxfp4_parts(key: jax.Array, shape: tuple[int, ...]) -> MXFP4Experts:
    """Random MXFP4 matrices `[exp, in, out]`: uniform codes under exponent
    bytes 117 to 121, the scales of weights near 0.02."""
    experts, inputs, outputs = shape
    codes, exponents = jax.random.split(key)
    return MXFP4Experts(
        jax.random.bits(codes, (experts, outputs, inputs // 2), jnp.uint8),
        jax.random.randint(exponents, (experts, outputs, inputs // 32), 117, 122).astype(jnp.uint8))


@jax.jit
def bf16(held: MXFP4Experts) -> jax.Array:
    return held.decoded(jnp.bfloat16)


def mxfp4_projection(args: argparse.Namespace) -> dict[str, object]:
    experts, width = 32, 2880
    features = 2 * width if args.shape == "gate_up" else width
    rng = np.random.default_rng(0)
    if args.rows <= experts:
        # A decode step: each row its own expert, as one token's top-k are.
        sizes = np.zeros(experts, np.int32)
        sizes[rng.choice(experts, args.rows, replace=False)] = 1
    else:
        sizes = np.floor(rng.dirichlet(np.ones(experts)) * args.rows).astype(np.int32)
        sizes[-1] += args.rows - sizes.sum()
    group_sizes = jnp.asarray(sizes)
    held = mxfp4_parts(jax.random.key(0), (experts, width, features))
    x = jnp.asarray(rng.normal(size=(args.rows, width)), jnp.bfloat16)
    routed = min(args.rows, experts)

    def project(x, kernel, sizes=group_sizes, implementation="auto"):
        return jnp.asarray(expert_projection(x, kernel, sizes, jnp.bfloat16, implementation, None))

    def gathered(x, held):
        present = jnp.nonzero(group_sizes, size=routed, fill_value=0)[0]
        sizes = jnp.where(jnp.arange(routed) < jnp.count_nonzero(group_sizes), group_sizes[present], 0)
        return project(x, bf16(MXFP4Experts(held.codes[present], held.exponents[present])), sizes)

    paths = {"bf16": project, "fused": lambda x, held: project(x, held, implementation="pallas"),
             "decoded": lambda x, held: project(x, bf16(held)), "gathered": gathered}
    operand = bf16(held) if args.path == "bf16" else held
    forward = jax.jit(paths[args.path])
    reference = np.asarray(jax.jit(project)(x, bf16(held)), np.float32)
    output = np.asarray(forward(x, operand), np.float32)
    return {"path": args.path, "shape": [args.rows, width, features, experts],
            "routed": int((sizes > 0).sum()), "forward": timed(forward, (x, operand), args.repeats),
            "operand_bytes": sum(leaf.nbytes for leaf in jax.tree.leaves(operand)),
            "bitwise_equal_to_bf16": bool(np.array_equal(output.view(np.uint32), reference.view(np.uint32))),
            "max_abs_difference": float(np.max(np.abs(output - reference)))}


def gpt_oss_decoder(layers: int, storage: str, implementation: str, length: int):
    """gpt-oss-20b's decoder (tests/fixtures/hf/gpt-oss-20b) cut to `layers`,
    holding its experts as `storage`."""
    from dew.interop.hf_decoders import translate_config
    from dew.registry import models

    source = Path(__file__).resolve().parent.parent / "tests/fixtures/hf/gpt-oss-20b/config.json"
    config = json.loads(source.read_text())
    config = {**config, "num_hidden_layers": layers, "layer_types": config["layer_types"][:layers]}
    fields = translate_config(config)
    mixture = fields["mixture"]
    assert isinstance(mixture, dict)
    mixture.update(expert_storage=storage, implementation=implementation)
    return models.build("causal_transformer", {**fields, "max_seq_len": length, "dtype": "bfloat16"})


def drawn(shapes, storage: str):
    """Weights for the MXFP4 decoder's `shapes`, each drawn from a key its path
    names, so both storages hold one model: the experts' parts as
    `mxfp4_parts` draws them, decoded to bf16 on the device one leaf at a
    time under 'float', and every other weight N(0, 0.02) in bf16."""
    def walk(path, node):
        key = jax.random.fold_in(jax.random.key(0), zlib.crc32(jax.tree_util.keystr(path).encode()))
        if isinstance(node, dict) and set(node) == set(MXFP4_PARTS):
            codes = node["codes"].shape
            held = mxfp4_parts(key, (codes[0], 2 * codes[2], codes[1]))
            if storage == "mxfp4":
                return dict(zip(MXFP4_PARTS, held, strict=True))
            return jax.block_until_ready(bf16(held))
        if isinstance(node, dict):
            return {name: walk((*path, jax.tree_util.DictKey(name)), child) for name, child in node.items()}
        if not jnp.issubdtype(node.dtype, jnp.floating):
            return jnp.zeros(node.shape, node.dtype)
        return (jax.random.normal(key, node.shape, jnp.float32) * 0.02).astype(jnp.bfloat16)

    return walk((), shapes)


def mxfp4_decode(args: argparse.Namespace) -> dict[str, object]:
    from dew.sampling import Sampling, generate

    length = args.prompt + args.tokens + 8
    model = gpt_oss_decoder(args.layers, args.storage, args.implementation, length)
    probe = jnp.zeros((1, 8), jnp.int32)
    held = jax.eval_shape(gpt_oss_decoder(args.layers, "mxfp4", args.implementation, length).init,
                          jax.random.key(0), probe)
    variables = drawn(held, args.storage)
    own = jax.eval_shape(model.init, jax.random.key(0), probe)
    assert jax.tree.map(np.shape, variables) == jax.tree.map(np.shape, own), "drawn for another tree"
    ids = jnp.asarray(np.random.default_rng(0).integers(100, 200000, (1, args.prompt)), jnp.int32)
    logits = np.asarray(jax.jit(model.apply)(variables, ids), np.float32)

    def greedy(tokens: int):
        return generate(model, variables, ids, tokens, key=jax.random.key(1),
                        sampling=Sampling(temperature=0))

    # One token is the prefill and its sample; the difference to `--tokens` is the decode steps alone.
    samples: dict[int, list[float]] = {}
    for tokens in (1, args.tokens):
        jax.block_until_ready(greedy(tokens).tokens)
        for _ in range(args.repeats):
            start = time.perf_counter()
            jax.block_until_ready(greedy(tokens).tokens)
            samples.setdefault(tokens, []).append(time.perf_counter() - start)
    generated = greedy(args.tokens)
    if args.out:
        np.savez(args.out, logits=logits, tokens=np.asarray(generated.tokens),
                 log_probs=np.asarray(generated.raw_log_probs, np.float32))
    per_token = (np.median(samples[args.tokens]) - np.median(samples[1])) / (args.tokens - 1)
    experts = [leaf.nbytes for path, leaf in jax.tree_util.tree_leaves_with_path(variables)
               if ".experts." in jax.tree_util.keystr(path, simple=True, separator=".")
               and "bias" not in jax.tree_util.keystr(path)]
    return {"layers": args.layers, "storage": args.storage, "implementation": args.implementation,
            "prompt": args.prompt, "tokens": args.tokens, "ms_per_token": float(per_token * 1e3),
            "tokens_per_second": float(1 / per_token), "seconds": {str(k): v for k, v in samples.items()},
            "parameter_bytes": sum(leaf.nbytes for leaf in jax.tree.leaves(variables)),
            "expert_bytes": sum(experts)}


def lm_dense_tree() -> dict[str, jax.Array]:
    """The lm-dense model's parameters as one flat tree, initialized."""
    case = case_for("lm-dense", 1, None)
    model = build_trainer(case).objective.model
    variables = jax.eval_shape(lambda: model.init(jax.random.key(0),
                                                  jnp.zeros((1, 8), jnp.int32)))
    leaves = jax.tree.leaves(variables["params"])
    keys = jax.random.split(jax.random.key(0), len(leaves))
    return {str(index): jax.random.normal(key, leaf.shape, jnp.float32) * 0.02
            for index, (key, leaf) in enumerate(zip(keys, leaves, strict=True))}


def adam(args: argparse.Namespace) -> dict[str, object]:
    params = lm_dense_tree()
    keys = jax.random.split(jax.random.key(1), len(params))
    grads = {name: jax.random.normal(key, leaf.shape, jnp.float32) * 1e-3
             for key, (name, leaf) in zip(keys, params.items(), strict=True)}
    solver = OptimConfig(optimizer="adamw", learning_rate=1e-4, weight_decay=0.1,
                                         state_dtype=args.state_dtype).build(1000)

    def update(params, state, grads):
        updates, state = solver.update(grads, state, params)
        return optax.apply_updates(params, updates), state

    run = jax.jit(update, donate_argnums=(0, 1))
    state = solver.init(params)
    for _ in range(3):
        params, state = run(params, state, grads)
    samples = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        params, state = run(params, state, grads)
        jax.block_until_ready((params, state))
        samples.append((time.perf_counter() - start) * 1e3)
    return {"state_dtype": args.state_dtype, "params": sum(leaf.size for leaf in params.values()),
            "update_mean_ms": float(np.mean(samples)), "update_min_ms": float(np.min(samples)),
            "state_bytes": sum(leaf.size * leaf.dtype.itemsize for leaf in jax.tree.leaves(state))}


def case_for(path: str, batch: int, args: argparse.Namespace | None) -> Case:
    def decoder(layers: int, width: int, heads: int, mlp: int) -> dict[str, object]:
        return {"vocab_size": VOCAB, "emb_features": width, "num_layers": layers,
                "num_heads": heads, "mlp_features": mlp, "max_seq_len": SEQUENCE}

    remat = None if args is None else args.remat
    if path == "dit":
        # DiT-L/2's width and depth on 64x64 inputs, 1024 tokens a sample.
        config: dict[str, object] = {"patch_size": 2, "emb_features": 1024, "num_layers": 24,
                                     "num_heads": 16, "mlp_ratio": 4, "remat": remat != "none"}
        return Case("simple_dit", config, dtype="bfloat16", batch_size=batch,
                                   image_size=64)
    if path == "lm-dense":
        config = decoder(24, 1024, 16, 2816)
    else:
        # Every second layer routes, Qwen3-MoE's decoder_sparse_step 2.
        mixture: dict[str, object] = {"experts": 8, "top_k": 2, "layers": [1, 3, 5, 7, 9, 11],
                                      "dispatch": "global"}
        if args is not None:
            mixture["implementation"] = args.implementation
        config = {**decoder(12, 768, 12, 2048), "mixture": mixture}
    config["remat"] = None if remat in (None, "none") else remat
    return Case("causal_transformer", config, dtype="bfloat16",
                               batch_size=batch, seq_len=SEQUENCE)


def step(args: argparse.Namespace) -> dict[str, object]:
    if args.path == "dit" and args.remat == "full":
        # A DiT's remat is a flag for the dots policy; full recomputation is
        # that block under no policy.
        import functools

        import dew.nn.backbones.dit as dit
        dit.remat_block = functools.partial(dit.remat_block, policy=None)
    case = case_for(args.path, args.batch, args)
    trainer = build_trainer(case, optimizer=OptimConfig(
        optimizer="adam", learning_rate=1e-4, state_dtype=args.state_dtype).build(1000))
    source = DevicePrefetchIterator(batches(case, trainer.device_mesh), trainer.device_mesh)
    with source:
        abstract = jax.eval_shape(trainer.initial_state)
        state = jax.jit(trainer.initial_state, out_shardings=trainer.shardings(abstract))()
        first = next(source)
        start = time.perf_counter()
        compiled = trainer.compile(state, first)
        compile_seconds = time.perf_counter() - start
        losses = []
        for _ in range(5):
            state, loss, *_ = compiled(state, next(source))
        losses.append(float(loss))
        start = time.perf_counter()
        for _ in range(args.steps):
            state, loss, *_ = compiled(state, next(source))
        loss.block_until_ready()
        step_ms = (time.perf_counter() - start) / args.steps * 1e3
        if args.trace:
            with jax.profiler.trace(args.trace, profiler_options=capture_options()):
                for _ in range(3):
                    state, loss, *_ = compiled(state, next(source))
                loss.block_until_ready()
        losses.append(float(loss))
    return {"path": args.path, "batch": args.batch, "implementation": args.implementation,
            "state_dtype": args.state_dtype, "remat": args.remat, "ms_per_step": step_ms,
            "compile_seconds": compile_seconds, "loss_after_warmup": losses[0],
            "loss_last": losses[-1]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    modes = parser.add_subparsers(dest="mode", required=True)
    project = modes.add_parser("projection")
    project.add_argument("--implementation", default="auto")
    project.add_argument("--shape", choices=("up", "down"), default="up")
    project.add_argument("--repeats", type=int, default=50)
    update = modes.add_parser("adam")
    update.add_argument("--state-dtype", choices=("float32", "bfloat16"), default="float32")
    update.add_argument("--repeats", type=int, default=30)
    run = modes.add_parser("step")
    run.add_argument("--path", choices=("lm-dense", "lm-moe", "dit"), required=True)
    run.add_argument("--remat", default="none",
                     help="none, full, a REMAT_POLICIES name (decoders) or dots (DiT)")
    run.add_argument("--batch", type=int, required=True)
    run.add_argument("--implementation", default="auto")
    run.add_argument("--state-dtype", choices=("float32", "bfloat16"), default="float32")
    run.add_argument("--steps", type=int, default=30)
    run.add_argument("--trace", default=None,
                     help="a directory for a jax.profiler trace of three steady-state steps")
    mxfp4 = modes.add_parser("mxfp4-projection")
    mxfp4.add_argument("--path", choices=("bf16", "fused", "decoded", "gathered"), required=True)
    mxfp4.add_argument("--shape", choices=("gate_up", "down"), default="gate_up")
    mxfp4.add_argument("--rows", type=int, default=4)
    mxfp4.add_argument("--repeats", type=int, default=100)
    decoding = modes.add_parser("mxfp4-decode")
    decoding.add_argument("--layers", type=int, default=8)
    decoding.add_argument("--storage", choices=("float", "mxfp4"), required=True)
    decoding.add_argument("--implementation", default="auto")
    decoding.add_argument("--prompt", type=int, default=16)
    decoding.add_argument("--tokens", type=int, default=65)
    decoding.add_argument("--repeats", type=int, default=3)
    decoding.add_argument("--out", default=None, help="an .npz for the prefill logits and greedy tokens")
    args = parser.parse_args(argv)
    result = {"projection": projection, "adam": adam, "step": step, "mxfp4-projection": mxfp4_projection,
              "mxfp4-decode": mxfp4_decode}[args.mode](args)
    result.update(device_kind=jax.devices()[0].device_kind, jax=jax.__version__,
                  peak_bytes=peak_bytes())
    print(json.dumps(result))


if __name__ == "__main__":
    main()
