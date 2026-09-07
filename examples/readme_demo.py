"""Train a toy language model, resume it, learn preferences, and generate images.

Run from an installed Dew checkout, without downloads:
    JAX_PLATFORMS=cpu python examples/readme_demo.py --out runs/readme-demo

Use a new output directory for each invocation. The generated data demonstrates
mechanics, not language ability, preference quality, or useful image generation.
"""
from dataclasses import dataclass
from pathlib import Path
import itertools
import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro

from dew import Checkpoints, Field, InputSpec, Trainer, metrics, models
from dew.data import Dataset, Loading, PreferencePairs, TokenWindows
from dew.diffusion.presets import Flow
from dew.objectives.base import Step
from dew.objectives.diffusion import DiffusionObjective
from dew.objectives.lm import LMObjective
from dew.objectives.rl import DPOObjective
from dew.sampling import Euler, generate


@dataclass
class Config:
    out: Path = Path("runs/readme-demo")
    """A new directory for generated inputs, checkpoints, and results."""


def language_model(out: Path):
    """Train and resume a decoder on a repeating eight-symbol vocabulary."""
    tokens = out / "tokens"
    tokens.mkdir()
    for split, repeats in (("train", 256), ("val", 32)):
        np.tile(np.array([1, 2, 3, 4], dtype=np.uint8), repeats).tofile(
            tokens / f"{split}.bin")
    (tokens / "meta.json").write_text(json.dumps({
        "tokenizer": "demo-symbols", "vocab_size": 8, "dtype": "uint8",
        "train_tokens": 1024, "val_tokens": 128, "eos_id": None,
    }))
    data = TokenWindows(
        path=str(tokens), seq_len=8, val_batches=1,
        loading=Loading(workers=0, threads=1, read_buffer=2),
    ).load(batch=8)
    model = models.build(
        "causal_transformer", vocab_size=8, emb_features=16, num_layers=1,
        num_heads=2, mlp_features=32, max_seq_len=16,
        dtype=jnp.float32, attention_impl="xla",
    )
    objective = LMObjective(model, seq_len=8, ema_decay=0.9)
    optimizer = optax.adam(0.01)
    checkpoints = Checkpoints(str(out / "lm-checkpoints"))
    trainer = Trainer(objective, optimizer, key=jax.random.key(0),
                      checkpoints=checkpoints)
    first = trainer.fit(data, steps=20, log_every=10,
                        eval_every=20, checkpoint_every=20,
                        metrics=(metrics.perplexity(),))
    first_step = int(first.step)
    # A new Trainer restores model, optimizer, random key, and read position.
    resumed = Trainer(objective, optimizer, key=jax.random.key(0),
                      checkpoints=checkpoints)
    state = resumed.fit(data, steps=24, log_every=4,
                        eval_every=4, checkpoint_every=4,
                        metrics=(metrics.perplexity(),))
    prompt = jnp.array([[1, 2]], dtype=jnp.int32)
    continuation = np.asarray(generate(
        model, state.averaged, prompt, max_new_tokens=8,
        key=jax.random.key(1), temperature=0.0,
    ))[0].tolist()
    print(f"LM resumed: {first_step} -> {int(state.step)}; tokens: {continuation}")
    return model, state, {"first_step": first_step, "resumed_step": int(state.step),
                          "generated_ids": continuation}


def preferences(model, pretrained):
    """Start DPO from the trained policy; freeze that policy as the reference."""
    row = {"chosen": [1, 2, 3, 4], "rejected": [1, 2, 6, 4],
           "chosen_mask": [0, 0, 1, 1], "rejected_mask": [0, 0, 1, 1]}
    data = PreferencePairs(
        records=(json.dumps(row),) * 8, seq_len=4,
        loading=Loading(workers=0, threads=1, read_buffer=2),
    ).load(batch=8)
    # The objective predicts three next tokens from each four-ID sequence.
    objective = DPOObjective(model, seq_len=3, beta=0.1, pretrained=pretrained)
    reference = jax.tree.map(lambda x: np.array(x, copy=True), pretrained)
    trainer = Trainer(objective, optax.adam(0.001), key=jax.random.key(2))
    state = trainer.fit(data, steps=4, log_every=1)
    delta = max(float(np.max(np.abs(np.asarray(after) - before)))
                for before, after in zip(jax.tree.leaves(reference),
                                         jax.tree.leaves(state.ema), strict=True))
    print(f"DPO: {int(state.step)} updates; reference max change: {delta:.1f}")
    return {"updates": int(state.step), "reference_max_change": delta}


def flow_images(out: Path):
    """Train a tiny image flow model, then write a separate generated preview."""
    images = np.zeros((8, 8, 8, 3), dtype=np.uint8)
    images[:, :, ::2, :] = 255
    batch = {"image": images}
    data = Dataset(train=lambda: itertools.repeat(batch), val=None,
                   records=8, batch=8)
    model = models.build(
        "simple_dit", patch_size=4, emb_features=16, num_layers=1,
        num_heads=2, mlp_ratio=2, dtype=jnp.float32, attention_impl="xla",
    )
    objective = DiffusionObjective(
        model, Flow()(), InputSpec(Field("image", (8, 8, 3))),
        sampler=Euler(), guidance=None, steps=4,
    )
    trainer = Trainer(objective, optax.adam(0.001), key=jax.random.key(3))
    state = trainer.fit(data, steps=3, log_every=1)
    preview = objective.preview(
        state.params, batch,
        Step(step=state.step, key=jax.random.key(4), ema=state.averaged),
    )
    generated = np.asarray(preview.images)
    np.save(out / "flow-preview.npy", generated)
    pixels = np.round((generated + 1) * 127.5).clip(0, 255).astype(np.uint8)
    grid = np.concatenate(list(pixels), axis=1)
    # PPM is a standard RGB image format; no image library is needed to write it.
    header = f"P6\n{grid.shape[1]} {grid.shape[0]}\n255\n".encode("ascii")
    (out / "flow-preview.ppm").write_bytes(header + grid.tobytes())
    finite = bool(np.isfinite(generated).all())
    print(f"Flow: {int(state.step)} updates; preview {generated.shape}; finite={finite}")
    return {"updates": int(state.step), "preview_shape": list(generated.shape),
            "finite": finite, "range": [float(generated.min()), float(generated.max())]}


def main(config: Config):
    out = config.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    print("Devices:", jax.devices())
    print("Output:", out)
    model, state, lm = language_model(out)
    dpo = preferences(model, state.params)
    flow = flow_images(out)
    summary = {"language_model": lm, "dpo": dpo, "flow": flow}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Saved summary.json, LM checkpoints, and flow-preview.npy/.ppm.")


if __name__ == "__main__":
    main(tyro.cli(Config))
