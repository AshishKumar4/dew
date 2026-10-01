"""Small runnable examples behind the landing page, using the public API.

python site/snippets/framework.py --section lm --out /tmp/dew-landing
Every displayed block is between its show/end markers; the surrounding code
prepares the small synthetic fixtures. --smoke uses an offline tiny checkpoint
in place of the Hub model and shortens training for CI, not for the recording.
"""

# Command-line examples print results for capture_snippets.py.
# ruff: noqa: T201

import argparse
import itertools
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew import Checkpoints, Dataset, Field, InputSpec, MeshSpec, Trainer, models
from dew.data import ByteTokenizer, Loading, Prompts, TokenWindows
from dew.diffusion.presets import EDM, Flow
from dew.inference import RunProcessor, TextGeneration
from dew.inference.serving import Server
from dew.interop import load_pretrained
from dew.objectives.diffusion import DiffusionObjective
from dew.objectives.jepa import JepaObjective, multi_block_mask
from dew.objectives.lm import LMObjective
from dew.objectives.rl import GRPOObjective, SampledRollout
from dew.sampling import Euler, Heun, Sampling
from dew.training.distributed import build_mesh

ROOT = Path(__file__).resolve().parents[2]
PROFILE = True


def text_fixture(out):
    tokenizer = ByteTokenizer()
    tokens = np.asarray(tokenizer.encode("dew trains jax models. " * 100), np.uint8)
    corpus = out / "tokens"
    corpus.mkdir(exist_ok=True)
    tokens.tofile(corpus / "train.bin")
    tokens[:520].tofile(corpus / "val.bin")
    (corpus / "meta.json").write_text(json.dumps({"tokenizer": "byte", "vocab_size": 256,
                                               "dtype": "uint8", "train_tokens": len(tokens),
                                               "val_tokens": 520, "eos_id": 255}))
    data = TokenWindows(path=str(corpus), seq_len=64, loading=Loading(workers=0)).load(batch=8)
    return tokenizer, data


def image_fixture(size):
    images = np.zeros((8, size, size, 3), np.uint8)
    images[:, :, ::2] = 255
    return Dataset(train=lambda partition: itertools.repeat({"image": images}), val=None, records=8, batch=8)


def decoder():
    return models.build("causal_transformer", vocab_size=256, emb_features=32,
                        num_layers=1, num_heads=2, mlp_features=64, max_seq_len=128)


def lm(out, smoke):
    _, data = text_fixture(out)
    # Begin snippet: lm
    model = models.build("causal_transformer", vocab_size=256,
                         emb_features=32, num_layers=1, num_heads=2,
                         mlp_features=64, max_seq_len=128)
    objective = LMObjective(model, seq_len=64, ema_decay=None)
    trainer = Trainer(objective, optax.adamw(1e-3), key=jax.random.key(0))
    state = trainer.fit(data, steps=3)
    # End snippet: lm
    assert int(state.step) == 3 and int(state.updates) == 3
    return {"steps": int(state.step), "parameters": sum(x.size for x in jax.tree.leaves(state.params))}


def diffusion(out, smoke):
    data = image_fixture(8)
    # Begin snippet: diffusion
    model = models.build("simple_dit", patch_size=4, emb_features=16,
                         num_layers=1, num_heads=2, mlp_ratio=2)
    objective = DiffusionObjective(
        model, Flow(), InputSpec(Field("image", (8, 8, 3))),
        sampler=Euler(), steps=4)
    trainer = Trainer(objective, optax.adamw(1e-3), key=jax.random.key(0))
    state = trainer.fit(data, steps=3)
    # End snippet: diffusion
    assert int(state.step) == 3 and int(state.updates) == 3
    # Begin snippet: processes
    flow, flow_solver = Flow(), Euler()
    edm, edm_solver = EDM(regime="pixel"), Heun()
    # End snippet: processes
    assert flow is not edm and type(flow_solver) is not type(edm_solver)
    return {"steps": int(state.step), "processes": [type(flow).__name__, type(edm).__name__],
            "solvers": [type(flow_solver).__name__, type(edm_solver).__name__]}


def jepa(out, smoke):
    data = image_fixture(32)
    # Begin snippet: jepa
    encoder = models.build("jepa_encoder", patch_size=4, emb_features=32,
                           num_layers=1, num_heads=2)
    predictor = models.build("jepa_predictor", grid=(8, 8), emb_features=32,
                             predictor_features=16, num_layers=1, num_heads=2)
    objective = JepaObjective(
        encoder, predictor, mask=multi_block_mask((8, 8)),
        sample=Field("image", (32, 32, 3)), momentum_steps=3)
    trainer = Trainer(objective, optax.adamw(1e-3), key=jax.random.key(0))
    state = trainer.fit(data, steps=3)
    # End snippet: jepa
    assert int(state.step) == 3 and int(state.updates) == 3
    return {"steps": int(state.step), "averaged": sorted(state.averaged)}


def grpo(out, smoke):
    model = decoder()
    tokenizer = ByteTokenizer()
    initial = model.init(jax.random.key(0), jnp.zeros((1, 8), jnp.int32))
    from dew import LocalTracker
    tracker = LocalTracker(out / "tracking")
    steps = 2 if smoke else 16
    # Begin snippet: grpo
    def reward(data_source, completion, ground_truth, extra_info):
        return sum(character.isalpha() for character in completion) / 8

    data = Prompts(tokenizer="byte", records=(json.dumps({"prompt": "dew"}),) * 8,
                   max_prompt_len=8, loading=Loading(workers=0)).load(batch=8)
    objective = GRPOObjective(model, seq_len=15, beta=0.01, pretrained=initial)
    rollout = SampledRollout(objective, reward=reward, groups=4,
                             max_new_tokens=8, decode=tokenizer.decode)
    trainer = Trainer(objective, optax.adamw(1e-4), key=jax.random.key(0),
                      rollout=rollout, tracker=tracker)
    state = trainer.fit(data, steps=steps, log_every=1)
    # End snippet: grpo
    tracker.close()
    assert int(state.updates) == steps
    curve = [{"step": row["step"], "reward": row["scalars"]["reward/mean"]}
             for line in (out / "tracking/scalars.jsonl").read_text().splitlines()
             if "reward/mean" in (row := json.loads(line))["scalars"]]
    return {"steps": steps, "reward": "Alphabetic characters / 8 response bytes", "curve": curve}


def pretrained(out, smoke):
    # Begin snippet: pretrained-source
    source = "Qwen/Qwen3-0.6B"
    # End snippet: pretrained-source
    if smoke:
        source = str(ROOT / "tests/fixtures/hf/qwen3-tiny")
    # Begin snippet: pretrained
    bundle = load_pretrained(source, dtype="bfloat16", max_seq_len=128)
    task = bundle.text_generation(sampling=Sampling(temperature=0))
    # End snippet: pretrained
    if smoke:
        task = replace(task, processor=RunProcessor(ByteTokenizer()))
        prompt = np.load(ROOT / "tests/fixtures/hf/qwen3-tiny/input_ids.npy")[:1, :8]
        training_tokens = np.pad(prompt, ((0, 0), (0, 9 - prompt.shape[1])))
    else:
        prompt = "The capital of France is"
        training_tokens = np.asarray(bundle.processor("The capital of France is Paris.").tokens[:, :9], np.int32)
    data = Dataset(train=lambda partition: itertools.repeat({"text": training_tokens}), val=None, records=1, batch=1)
    # Begin snippet: finetune
    text = task(prompt, 12, seed=0).text
    objective = bundle.lm_objective(seq_len=training_tokens.shape[1] - 1, ema_decay=None)
    trainer = Trainer(objective, optax.sgd(1e-5), key=jax.random.key(0))
    state = trainer.fit(data, steps=1)
    bundle.save(out / "export", variables=state.params)
    # End snippet: finetune
    assert int(state.updates) == 1
    assert any(not np.array_equal(np.asarray(before), np.asarray(after))
               for before, after in zip(jax.tree.leaves(bundle.variables), jax.tree.leaves(state.params), strict=True))
    assert (out / "export/config.json").is_file()
    return {"source": source, "text": list(text), "steps": int(state.step), "export": "export"}


def serving(out, smoke):
    tokenizer, data = text_fixture(out)
    model = decoder()
    state = Trainer(LMObjective(model, 64, ema_decay=None), optax.adamw(3e-3),
                    key=jax.random.key(0)).fit(data, steps=3 if smoke else 150)
    # Begin snippet: serving
    task = TextGeneration(model, state.params, RunProcessor(tokenizer),
                          sampling=Sampling(temperature=0))
    server = Server.from_task(task, slots=4, capacity=128)
    results = server(["dew", "jax"], 24, seed=0)
    print([result.text[0] for result in results])
    # End snippet: serving
    # Begin snippet: int8
    from dew.training.quantization import Quantization
    int8 = task.quantized(Quantization(dtype="int8", weight_only=True))
    # End snippet: int8
    # Begin snippet: fp8
    fp8 = task.quantized(Quantization(dtype="fp8", weight_only=True))
    # End snippet: fp8
    for variant in (int8, fp8):
        quantized = Server.from_task(variant, slots=4, capacity=128)
        assert len(quantized(["dew"], 2, seed=0)) == 1
    return {"text": [result.text[0] for result in results], "weight_formats": ["int8", "fp8"]}


def mesh(out, smoke):
    _, data = text_fixture(out)
    model = decoder()
    objective = LMObjective(model, 64, ema_decay=None)
    # Begin snippet: mesh
    trainer = Trainer(objective, optax.adamw(1e-3), key=jax.random.key(0),
                      mesh=MeshSpec(fsdp=2, tensor=2))
    state = trainer.fit(data, steps=3)
    # End snippet: mesh
    assert int(state.step) == 3 and int(state.updates) == 3
    return {"devices": jax.device_count(), "axes": dict(build_mesh(trainer.mesh).shape), "steps": int(state.step)}


def reliability(out, smoke):
    _, data = text_fixture(out)
    objective = LMObjective(decoder(), 64, ema_decay=None)
    # Begin snippet: reliability
    checkpoints = Checkpoints(str(out / "checkpoints"))
    trainer = Trainer(objective, optax.adamw(1e-3), key=jax.random.key(0),
                      checkpoints=checkpoints)
    state = trainer.fit(data, steps=2, checkpoint_every=1)
    resumed = trainer.fit(data, steps=3, checkpoint_every=1)
    # End snippet: reliability
    baseline = Trainer(objective, optax.adamw(1e-3), key=jax.random.key(0)).fit(data, steps=3)
    differences = []
    left, _ = jax.tree_util.tree_flatten_with_path(resumed.params)
    right = jax.tree.leaves(baseline.params)
    for (path, actual), expected in zip(left, right, strict=True):
        actual, expected = np.asarray(actual), np.asarray(expected)
        if not np.array_equal(actual, expected):
            differences.append({"path": jax.tree_util.keystr(path), "shape": list(actual.shape),
                                "count": int(np.count_nonzero(actual != expected)),
                                "max_abs": float(np.max(np.abs(actual - expected)))})
    (out / "resume-comparison.json").write_text(json.dumps(differences, indent=2) + "\n")
    exact = not differences
    assert exact, "resuming the checkpoint changed the parameters"
    if PROFILE:
        # Begin snippet: profile
        import dew
        with dew.profile(out / "profile"):
            logits = objective.model.apply(resumed.params, jnp.zeros((1, 8), jnp.int32))
            logits.block_until_ready()
        # End snippet: profile
    return {"saved_step": int(state.step), "resumed_step": int(resumed.step), "bit_exact": exact}


SECTIONS = {function.__name__: function for function in (lm, diffusion, jepa, grpo, pretrained, serving, mesh, reliability)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--section", choices=SECTIONS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--topology-only", action="store_true")
    parser.add_argument("--no-profile", action="store_true")
    options = parser.parse_args()
    global PROFILE
    PROFILE = not options.no_profile
    options.out.mkdir(parents=True, exist_ok=True)
    if options.section == "mesh":
        jax.config.update("jax_num_cpu_devices", 4)
    if options.topology_only:
        mesh = build_mesh(MeshSpec(fsdp=2, tensor=2))
        result = {"axes": dict(mesh.shape), "devices": [device.id for device in mesh.devices.flat],
                  "backend": jax.default_backend(), "training": False}
    else:
        result = SECTIONS[options.section](options.out, options.smoke)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(result) if isinstance(x, np.ndarray))
    (options.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
