"""Qualify a dense or MoE training run, including recovery after SIGKILL.

Uses a local text corpus and the committed tokenizer. Reference models have
random weights, so this establishes workflow correctness, not model quality.
Run under a process-tree memory limit with JAX_PLATFORMS=cpu or cuda.
CUDA bitwise recovery uses --attention-impl xla and
XLA_FLAGS=--xla_gpu_deterministic_ops=true; this does not qualify cuDNN
determinism or physical multi-device execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "tests/fixtures/tokenizers/tiny-tools"
STEPS = 24
CHECKPOINT_STEP = 5
INTERRUPT_STEP = 7
SEQUENCE = 32
BATCH = 8


def worker(directory: Path, mode: str, dtype: str) -> None:
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax

    from dew.checkpoints import Checkpoints
    from dew.data import Loading, TokenWindows
    from dew.data.dataset import GlobalStream
    from dew.interop import load_pretrained
    from dew.objectives.base import Step
    from dew.objectives.lm import LMObjective
    from dew.training import Trainer
    from dew.training.tracker import LocalTracker

    run = directory / ("baseline" if mode == "baseline" else "restarted")
    checkpoints = Checkpoints(str(run / "checkpoints"), keep=2)

    class Recorder(LocalTracker):
        def log(self, scalars, step):
            super().log(scalars, step)
            if mode == "interrupted" and step == INTERRUPT_STEP:
                checkpoints.wait()
                if checkpoints.latest != CHECKPOINT_STEP:
                    raise RuntimeError(
                        "Interruption did not follow the intended checkpoint"
                    )
                (directory / "ready-to-kill").write_text(str(os.getpid()))
                threading.Event().wait()

    configuration = json.loads((directory / "configuration.json").read_text())
    source = load_pretrained(
        directory / "source",
        dtype=dtype,
        attention_impl=configuration["attention_impl"],
    )
    objective = LMObjective(
        source.model.clone(dropout_rate=0.1),
        SEQUENCE,
        pretrained=source.variables,
        ema_decay=0.9,
    )
    data = TokenWindows(
        path=str(directory / "tokens"),
        seq_len=SEQUENCE,
        seed=17,
        val_batches=1,
        loading=Loading(workers=0, threads=1),
    ).load(batch=BATCH)
    trainer = Trainer(
        objective,
        optax.chain(
            optax.clip_by_global_norm(0.5), optax.adamw(0.003, weight_decay=0.01)
        ),
        key=jax.random.key(11),
        accumulation=2,
        dynamic_scale=True,
        checkpoints=checkpoints,
        tracker=Recorder(run / "metrics"),
    )
    before, _, before_position = trainer.place()
    if mode == "resumed" and (
        int(before.microstep) % 2 != 1
        or before.accumulation is None
        or before.accumulation.mass is None
        or float(before.accumulation.mass) <= 0
    ):
        raise RuntimeError("Resume lost the partially accumulated gradient")
    stream = data.train()
    if not isinstance(stream, GlobalStream):
        raise TypeError(
            "Qualification requires the token loader's global-position stream"
        )
    try:
        probe = next(stream)
    finally:
        stream.close()
    step = Step(step=jnp.int32(0), key=jax.random.key(19), ema=None)
    stats, _, _ = objective.predict(before.params, probe, step, train=False)
    initial_loss = float(objective.reduce_loss(stats)[0])
    if mode == "baseline":
        parity = LMObjective(
            source.model.clone(dtype=jnp.float32, attention_impl=None),
            SEQUENCE,
            ema_decay=None,
        )

        def reference_loss(params):
            stats, _ = parity.loss({"params": params}, probe, step)
            return parity.reduce_loss(stats)[0]

        loss, gradients = jax.value_and_grad(reference_loss)(before.params["params"])
        reference_arrays = {"ids": np.asarray(probe["text"]), "loss": np.asarray(loss)}
        for layout in source.weight_layouts:
            reference_arrays["gradient/" + layout.name] = layout.export(
                {"params": gradients}
            )
        np.savez(run / "reference.npz", allow_pickle=False, **reference_arrays)
    started = time.perf_counter()
    state = trainer.fit(
        data, steps=STEPS, log_every=1, checkpoint_every=CHECKPOINT_STEP
    )
    elapsed = time.perf_counter() - started
    stats, _, _ = objective.predict(state.params, probe, step, train=False)
    final_loss = float(objective.reduce_loss(stats)[0])
    restored, _, position = trainer.place()
    if not position:
        raise RuntimeError("Final checkpoint has no data position")

    arrays = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]:
        value = (
            jax.random.key_data(leaf)
            if jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key)
            else leaf
        )
        arrays[jax.tree_util.keystr(path)] = np.asarray(value)
    for left, right in zip(
        jax.tree.leaves(state), jax.tree.leaves(restored), strict=True
    ):
        if jnp.issubdtype(left.dtype, jax.dtypes.prng_key):
            left, right = jax.random.key_data(left), jax.random.key_data(right)
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    np.savez(run / "state.npz", **arrays)
    source.save(run / "export", variables=state.params)
    reloaded = load_pretrained(
        run / "export", dtype="float32", attention_impl="reference"
    )
    ids = jnp.asarray(probe["text"][:, :-1])
    expected = source.model.clone(dtype=jnp.float32, attention_impl=None).apply(
        state.params, ids
    )
    actual = reloaded.model.apply(reloaded.variables, ids)
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), atol=1e-4, rtol=0
    )
    np.savez(run / "export_logits.npz", ids=np.asarray(ids), logits=np.asarray(actual))
    report = {
        "mode": mode,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "dtype": dtype,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "elapsed_seconds": elapsed,
        "restored_step": int(before.step),
        "restored_microstep": int(before.microstep),
        "restored_updates": int(before.updates),
        "restored_position": None
        if before_position is None
        else before_position.decode(),
        "step": int(state.step),
        "microstep": int(state.microstep),
        "updates": int(state.updates),
        "position": position.decode(),
        "state_structure": str(jax.tree.structure(state)),
        "state_leaves": len(arrays),
        "jax": jax.__version__,
    }
    (run / "result.json").write_text(json.dumps(report, indent=2) + "\n")


def prepare(
    directory: Path, corpus: Path, family: str, dtype: str, attention_impl: str
) -> None:
    import torch
    from transformers import (
        AutoTokenizer,
        LlamaConfig,
        LlamaForCausalLM,
        MixtralConfig,
        MixtralForCausalLM,
    )

    from tokenize_text import TokenizeArgs, main as tokenize

    directory.mkdir(parents=True, exist_ok=False)
    text = corpus.read_bytes()
    (directory / "corpus.txt").write_bytes(text)
    tokenize(
        TokenizeArgs(
            input=str(directory / "corpus.txt"),
            out=str(directory / "tokens"),
            tokenizer=str(TOKENIZER),
            val_fraction=0.1,
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    fields = dict(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=SEQUENCE,
        tie_word_embeddings=False,
    )
    torch.manual_seed(13)
    model = (
        LlamaForCausalLM(LlamaConfig.from_dict(fields))
        if family == "dense"
        else MixtralForCausalLM(
            MixtralConfig.from_dict(
                dict(
                    fields,
                    num_local_experts=4,
                    num_experts_per_tok=2,
                    sliding_window=None,
                )
            )
        )
    )
    model.save_pretrained(directory / "source", safe_serialization=True)
    tokenizer.save_pretrained(directory / "source")
    record = dict(
        family=family,
        dtype=dtype,
        attention_impl=attention_impl,
        model=model.config.to_dict(),
        steps=STEPS,
        sequence=SEQUENCE,
        global_batch=BATCH,
        dropout=0.1,
        accumulation=2,
        dynamic_scale=True,
        ema_decay=0.9,
        seed=11,
        data_seed=17,
        optimizer={
            "name": "adamw",
            "learning_rate": 0.003,
            "weight_decay": 0.01,
            "clip_global_norm": 0.5,
        },
        corpus_sha256=hashlib.sha256(text).hexdigest(),
        corpus_path=str(corpus.resolve()),
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        xla_flags=os.environ.get("XLA_FLAGS", ""),
    )
    (directory / "configuration.json").write_text(json.dumps(record, indent=2) + "\n")


def run_child(directory: Path, mode: str, dtype: str) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--out",
        str(directory),
        "--worker",
        mode,
        "--dtype",
        dtype,
    ]
    with (directory / f"{mode}.log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            if mode == "interrupted":
                deadline = time.monotonic() + 600
                marker = directory / "ready-to-kill"
                while not marker.exists():
                    if process.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError(
                            f"Worker did not reach interruption; see {mode}.log"
                        )
                    time.sleep(0.05)
                if int(marker.read_text()) != process.pid:
                    raise RuntimeError("Interruption marker belongs to another process")
                os.killpg(process.pid, signal.SIGKILL)
                if process.wait(timeout=30) != -signal.SIGKILL:
                    raise RuntimeError("Worker was not killed by SIGKILL")
            elif process.wait(timeout=600) != 0:
                raise RuntimeError(f"Worker failed; see {mode}.log")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)


def compare(directory: Path) -> None:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    runs = [directory / name for name in ("baseline", "restarted")]
    baseline, resumed = [json.loads((run / "result.json").read_text()) for run in runs]
    for field in ("position", "state_structure", "step", "microstep", "updates"):
        if baseline[field] != resumed[field]:
            raise AssertionError(f"Recovery changed {field}")
    if (
        resumed["restored_step"] != CHECKPOINT_STEP
        or resumed["restored_updates"] != CHECKPOINT_STEP // 2
    ):
        raise AssertionError(
            "Resume did not restore the partial accumulation checkpoint"
        )
    if baseline["final_loss"] >= baseline["initial_loss"]:
        raise AssertionError("Training did not improve the fixed-batch loss")
    with (
        np.load(runs[0] / "state.npz") as left,
        np.load(runs[1] / "state.npz") as right,
    ):
        if set(left.files) != set(right.files):
            raise AssertionError("Recovery changed the state leaves")
        for name in left.files:
            np.testing.assert_array_equal(left[name], right[name], err_msg=name)
    reference = AutoModelForCausalLM.from_pretrained(
        directory / "source",
        dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
    ).eval()
    gradient_errors = {}
    with np.load(runs[0] / "reference.npz") as data:
        ids = torch.from_numpy(data["ids"]).long()
        logits = reference(ids).logits[:, :-1]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)
        )
        np.testing.assert_allclose(
            loss.detach().numpy(), data["loss"], atol=1e-4, rtol=0
        )
        loss.backward()
        gradients = {}
        for name, parameter in reference.named_parameters():
            if parameter.grad is None:
                raise AssertionError(f"Reference parameter has no gradient: {name}")
            gradient = parameter.grad.numpy()
            if reference.config.model_type == "mixtral":
                name = name.replace(".mlp.", ".block_sparse_moe.")
                if name.endswith(".experts.gate_up_proj"):
                    prefix = name.removesuffix("gate_up_proj")
                    for index, expert in enumerate(gradient):
                        gate, up = np.split(expert, 2, axis=0)
                        gradients[f"{prefix}{index}.w1.weight"] = gate
                        gradients[f"{prefix}{index}.w3.weight"] = up
                    continue
                if name.endswith(".experts.down_proj"):
                    prefix = name.removesuffix("down_proj")
                    for index, expert in enumerate(gradient):
                        gradients[f"{prefix}{index}.w2.weight"] = expert
                    continue
            gradients[name] = gradient
        if {"gradient/" + name for name in gradients} != set(data.files) - {
            "ids",
            "loss",
        }:
            raise AssertionError(
                "Reference gradient coverage differs from the source tensor layout"
            )
        for name, wanted in gradients.items():
            delta = float(np.max(np.abs(data["gradient/" + name] - wanted)))
            scaled = delta / max(1.0, float(np.max(np.abs(wanted))))
            if scaled > 1e-4:
                raise AssertionError(f"Gradient mismatch for {name}: {scaled}")
            gradient_errors[name] = scaled
    errors = []
    for run in runs:
        reference = AutoModelForCausalLM.from_pretrained(
            run / "export",
            dtype=torch.float32,
            attn_implementation="eager",
            local_files_only=True,
        ).eval()
        with np.load(run / "export_logits.npz") as data, torch.no_grad():
            actual = reference(torch.from_numpy(data["ids"]).long()).logits.numpy()
            np.testing.assert_allclose(actual, data["logits"], atol=1e-4, rtol=0)
            errors.append(float(np.max(np.abs(actual - data["logits"]))))
    result = {
        "baseline": baseline,
        "resumed": resumed,
        "all_state_leaves_equal": True,
        "fp32_reference_gradient_scaled_errors": gradient_errors,
        "reference_export_max_errors": errors,
        "interruption": "SIGKILL after step 7, restore step 5",
    }
    (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=ROOT / "CONTRIBUTING.md")
    parser.add_argument("--family", choices=("dense", "moe"), default="dense")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument(
        "--attention-impl",
        choices=("auto", "xla", "cudnn", "reference", "tpu"),
        default="auto",
    )
    parser.add_argument("--worker", choices=("baseline", "interrupted", "resumed"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        compare(args.out)
        return
    if args.worker:
        worker(args.out, args.worker, args.dtype)
        return
    prepare(args.out, args.corpus, args.family, args.dtype, args.attention_impl)
    for mode in ("baseline", "interrupted", "resumed"):
        run_child(args.out, mode, args.dtype)
    compare(args.out)


if __name__ == "__main__":
    main()
