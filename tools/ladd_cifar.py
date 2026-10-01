"""LADD on CIFAR-10 (32 px, pixel space), at the paper's renoising and others.

    PYTHONPATH=src python tools/ladd_cifar.py TEACHER_STEPS LADD_STEPS '<list of overrides as JSON>'

Run once with LADD_STEPS 0 to train and save the teacher, then with
TEACHER_STEPS 0 to distill it. Outputs go to $LADD_OUTPUT (the teacher's
weights and the synthetic set are cached there). An override set holds the
objective's fields and the run's: lr, head_lr, batch, accumulation, seed,
synthetic (train on the teacher's own samples, as LADD does) and
eval_every (FID-5k at 1 and 4 steps every that many optimizer steps).

A flow teacher (simple_dit, patch 2, width 256, 6 blocks, class-name
prompts) trains
for TEACHER_STEPS at batch 128 and is scored at 1, 4 and 25 Euler steps
(CFG 1.5). Each override set then distills it with LADD for LADD_STEPS and is
scored at 1 and 4 Consistency steps. FID-5k against 5,000 training images;
every number is one seed.
"""

import json
import os
import sys
import time

import datasets
import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew.diffusion import presets
from dew.eval.fid import fid
from dew.inputs import CharTable, Condition, Field, InputSpec
from dew.objectives.diffusion import DiffusionObjective
from dew.objectives.diffusion.adversarial import AdversarialDistillationObjective
from dew.registry import models
from dew.sampling import CFG, Consistency, Euler, sample
from dew.training import Trainer

teacher_steps, ladd_steps, variants = int(sys.argv[1]), int(sys.argv[2]), json.loads(sys.argv[3])
data = datasets.load_dataset("uoft-cs/cifar10", split="train")
images = np.stack([np.asarray(image) for image in data["img"]])
names = data.features["label"].names
labels = np.asarray(data["label"])
reference = images[np.random.default_rng(1).choice(len(images), 5000, replace=False)]


def spec():
    return InputSpec(Field("image", (32, 32, 3)), {"textcontext": Condition(CharTable.from_pretrained("char_table"))})


def network():
    return models.SimpleDiT(patch_size=2, emb_features=256, num_layers=6, num_heads=4, mlp_ratio=4)


def train(task, steps, lr, log_every=0, head_lr=None, seed=0, rows_per_step=128, accumulation=1, data=None,
          evaluate=None, eval_every=0):
    pixels, classes = (images, labels) if data is None else data
    optimizer = optax.adamw(lr)
    if head_lr is not None:
        from dew.objectives.diffusion.objective import DISCRIMINATOR
        optimizer = optax.multi_transform(
            {"heads": optax.adamw(head_lr), "student": optax.adamw(lr)},
            lambda params: {name: "heads" if name == DISCRIMINATOR else "student" for name in params})
    trainer = Trainer(task, optimizer, key=jax.random.PRNGKey(seed), accumulation=accumulation)
    state = trainer.initial_state()
    rng = np.random.default_rng(seed)
    text = CharTable.from_pretrained("char_table")
    step = None
    for index in range(steps * accumulation):
        rows = rng.integers(0, len(pixels), rows_per_step // accumulation)
        batch = {"image": pixels[rows], "text": text.tokenize([names[c] for c in classes[rows]])}
        step = step or trainer.compile(state, batch)
        state, _, metrics, *_ = step(state, batch)
        if log_every and index % (log_every * accumulation) == 0:
            print(index // accumulation, {k: round(float(v), 3) for k, v in jax.device_get(metrics).items()},
                  flush=True)
        done = (index + 1) // accumulation
        if evaluate is not None and eval_every and (index + 1) % accumulation == 0 and done % eval_every == 0:
            print(json.dumps({"step": done, **evaluate(state.params)}), flush=True)
    return state


def score(task, params, steps, solver, guidance=None):
    text = CharTable.from_pretrained("char_table")
    generated = []
    for start in range(0, 5000, 250):
        rows = np.arange(start, start + 250) % len(images)
        prompts = text.tokenize([names[c] for c in labels[rows]])
        given, unconditional = task._conditions(params, {"text": prompts, "image": images[rows]},
                                                jax.random.PRNGKey(0), dropout=False)
        denoise = task.process.denoiser(task.model, task.trainable(params), given,
                                        None if guidance is None else unconditional)
        x = sample(denoise, task.process.noise(jax.random.PRNGKey(start), (250, 32, 32, 3)), steps + 1,
                   solver=solver, guidance=guidance, key=jax.random.PRNGKey(1))
        generated.append(np.asarray(np.clip((np.asarray(x) + 1) * 127.5, 0, 255), np.uint8))
    return round(fid(np.concatenate(generated), reference), 2)


flow = presets.Flow()()
OUTPUT = os.environ.get("LADD_OUTPUT", "/mnt/scratch/dew/runs/diffusion-gaps")
saved = f"{OUTPUT}/ladd_cifar_teacher.npz"
if teacher_steps:
    started = time.time()
    teacher = DiffusionObjective(network(), flow, spec(), guidance=None, sampler=Euler(), steps=2, ema_decay=None)
    state = train(teacher, teacher_steps, 2e-4)
    teacher_model = teacher.trainable(state.params)
    leaves, _ = jax.tree.flatten(teacher_model)
    np.savez(saved, *[np.asarray(leaf) for leaf in leaves])
    print(json.dumps({"teacher": {f"euler_{n}_cfg1.5": score(teacher, state.params, n, Euler(), CFG(1.5))
                                  for n in (1, 4, 25)}, "seconds": round(time.time() - started)}), flush=True)
    sys.exit()
template = DiffusionObjective(network(), flow, spec(), guidance=None, sampler=Euler(), steps=2, ema_decay=None)
structure = jax.tree.structure(template.trainable(template.init(jax.random.PRNGKey(0))))
stored = np.load(saved)
teacher_model = jax.tree.unflatten(structure, [jnp.asarray(stored[f"arr_{i}"]) for i in range(len(stored.files))])
SYNTHETIC = f"{OUTPUT}/ladd_cifar_synthetic.npz"


def synthetic_data(count=20000):
    """The teacher's own samples, as LADD trains on: 25 Euler steps at a
    constant CFG of 1.5, one per class name in turn; cached."""
    import os
    if os.path.exists(SYNTHETIC):
        stored = np.load(SYNTHETIC)
        return stored["images"], stored["labels"]
    teacher = DiffusionObjective(network(), flow, spec(), guidance=None, sampler=Euler(), steps=2, ema_decay=None)
    params = {**teacher.init(jax.random.PRNGKey(0)), **teacher_model}
    text = CharTable.from_pretrained("char_table")
    classes = np.arange(count) % len(names)
    generated = []
    for start in range(0, count, 250):
        chunk = classes[start:start + 250]
        given, unconditional = teacher._conditions(
            params, {"text": text.tokenize([names[c] for c in chunk]), "image": images[:len(chunk)]},
            jax.random.PRNGKey(0), dropout=False)
        denoise = teacher.process.denoiser(teacher.model, teacher.trainable(params), given, unconditional)
        x = sample(denoise, teacher.process.noise(jax.random.PRNGKey(10_000 + start), (len(chunk), 32, 32, 3)), 26,
                   solver=Euler(), guidance=CFG(1.5), key=jax.random.PRNGKey(2))
        generated.append(np.asarray(np.clip((np.asarray(x) + 1) * 127.5, 0, 255), np.uint8))
    pixels = np.concatenate(generated)
    np.savez(SYNTHETIC, images=pixels, labels=classes)
    return pixels, classes


for overrides in variants:
    started = time.time()
    fields = {"feature_layers": ("dit_block_1", "dit_block_3", "dit_block_5"), **overrides}
    seed = fields.pop("seed", 0)
    rows_per_step = fields.pop("batch", 128)
    accumulation = fields.pop("accumulation", 1)
    data = synthetic_data() if fields.pop("synthetic", False) else None
    lr = fields.pop("lr", 1e-5)
    head_lr = fields.pop("head_lr", None)
    task = AdversarialDistillationObjective(network(), flow, spec(), teacher=jax.tree.map(jnp.copy, teacher_model),
                                            ema_decay=None, **fields)
    eval_every = fields.pop("eval_every", 0)

    def evaluate(params, task=task):
        return {f"consistency_{n}": score(task, params, n, Consistency()) for n in (1, 4)}

    distilled = train(task, ladd_steps, lr, log_every=500, head_lr=head_lr, seed=seed, rows_per_step=rows_per_step,
                      accumulation=accumulation, data=data, evaluate=evaluate, eval_every=eval_every)
    print(json.dumps({"ladd": overrides, **{f"consistency_{n}": score(task, distilled.params, n, Consistency())
                                            for n in (1, 4)}, "seconds": round(time.time() - started)}), flush=True)
