"""Small runnable examples behind the landing page, using the public API.

python site/snippets/framework.py --section lm --out /tmp/dew-landing
The page shows the code between each "Begin snippet" and "End snippet" marker.
The code around the markers builds small synthetic fixtures. For CI, --smoke
swaps the Hub model for an offline tiny checkpoint and trains for fewer steps;
the recordings on the page do not use it.
"""

# Each section prints its result for capture_snippets.py.

import argparse
import itertools
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew import Checkpoints, Dataset, Field, InputSpec, MeshSpec, Trainer
from dew.data import ByteTokenizer, Loading, Prompts, TokenWindows
from dew.diffusion.presets import EDM, Flow
from dew.inference import RunProcessor
from dew.inference.serving import Server
from dew.interop import PretrainedDecoder
from dew.nn.backbones import CausalTransformer, SimpleDiT
from dew.objectives.diffusion import DiffusionObjective
from dew.objectives.jepa import JepaEncoder, JepaObjective, JepaPredictor, MultiBlockMask
from dew.objectives.lm import LMObjective
from dew.objectives.rl import GRPOObjective, SampledRollout
from dew.sampling import Euler, Heun, Sampling

ROOT = Path(__file__).resolve().parents[2]
PROFILE = True


# Begin snippet: text-fixture
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
# End snippet: text-fixture


# Begin snippet: image-fixture
def image_fixture(size):
    images = np.zeros((8, size, size, 3), np.uint8)
    images[:, :, ::2] = 255
    return Dataset(train=lambda partition: itertools.repeat({"image": images}), val=None, records=8, batch=8)
# End snippet: image-fixture


# Begin snippet: decoder
def decoder():
    return CausalTransformer(vocab_size=256, emb_features=32,
                             num_layers=1, num_heads=2, mlp_features=64, max_seq_len=128)
# End snippet: decoder


def lm(out, smoke):
    _, data = text_fixture(out)
    # Begin snippet: lm
    model = CausalTransformer(vocab_size=256,
                              emb_features=32, num_layers=1, num_heads=2,
                              mlp_features=64, max_seq_len=128)
    objective = LMObjective(model, seq_len=64, ema_decay=None)
    trainer = Trainer(objective, optax.adamw(1e-3), key=jax.random.key(0))
    state = trainer.fit(data, steps=3)
    # End snippet: lm
    assert int(state.step) == 3 and int(state.updates) == 3
    return {"steps": int(state.step), "parameters": sum(x.size for x in jax.tree.leaves(state.variables))}


def diffusion(out, smoke):
    data = image_fixture(8)
    # Begin snippet: diffusion
    model = SimpleDiT(patch_size=4, emb_features=16,
                      num_layers=1, num_heads=2, mlp_ratio=2)
    objective = DiffusionObjective(
        model, Flow(), InputSpec(Field("image", (8, 8, 3))),
        solver=Euler(), steps=4)
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


def sample_public(out, smoke):
    if smoke:
        import sys
        sys.path.insert(0, str(ROOT / "tests"))
        from test_inference import make_run

        import dew.interop.hub as hub
        snapshot = out / "run"
        make_run(snapshot, preset=Flow())
        hub.pull_from_hub = lambda repo_id, revision=None: snapshot
    # Begin snippet: sample-public
    from dew.sampling import CFG, DPMSolverMultistep, TextToImage
    pipe = TextToImage.from_pretrained("dewml/hybrid-dit-176m",
                                       revision="7187bb75a425dfb9fa0055951b8f0f7520185b87")
    result = pipe(["green and purple northern lights over a frozen lake"],
                  key=5, steps=20, solver=DPMSolverMultistep(), guidance=CFG(5))
    result.pil()[0].save(out / "sample.png")
    # End snippet: sample-public
    assert (out / "sample.png").is_file()
    return {"shape": list(result.host().images.shape), "image": "sample.png"}


def jepa(out, smoke):
    data = image_fixture(32)
    # Begin snippet: jepa
    encoder = JepaEncoder(patch_size=4, emb_features=32,
                          num_layers=1, num_heads=2)
    predictor = JepaPredictor(grid=(8, 8), emb_features=32,
                              predictor_features=16, num_layers=1, num_heads=2)
    objective = JepaObjective(
        encoder, predictor, mask=MultiBlockMask.for_grid((8, 8)),
        sample=Field("image", (32, 32, 3)), momentum_steps=3)
    trainer = Trainer(objective, optax.adamw(1e-3), key=jax.random.key(0))
    state = trainer.fit(data, steps=3)
    # End snippet: jepa
    assert int(state.step) == 3 and int(state.updates) == 3
    return {"steps": int(state.step), "averaged": sorted(state.averaged)}


def grpo(out, smoke):
    from dew import LocalTracker
    tracker = LocalTracker(out / "tracking")
    steps = 2 if smoke else 150
    rng = np.random.default_rng(0)
    records = tuple(json.dumps({"prompt": rng.integers(0, 13, 4).tolist()}) for _ in range(512))
    # Begin snippet: grpo
    model = CausalTransformer(vocab_size=13, emb_features=64,
                              num_layers=2, num_heads=4, head_dim=16,
                              mlp_features=128, max_seq_len=16)

    def reward(data_source, completion, ground_truth, extra_info):
        tokens = [int(token) for token in completion.split()]
        pairs = list(itertools.pairwise(tokens))
        return sum(b == (a + 1) % 13 for a, b in pairs) / max(len(pairs), 1)

    data = Prompts(tokenizer="byte", records=records, max_prompt_len=4,
                   loading=Loading(workers=0)).load(batch=8)
    objective = GRPOObjective(model, seq_len=11, beta=0.02)
    rollout = SampledRollout(objective, reward=reward, groups=4, max_new_tokens=8,
                             sampling=Sampling(temperature=1.0))
    trainer = Trainer(objective, optax.adamw(1e-3), key=jax.random.key(0),
                      rollout=rollout, tracker=tracker)
    state = trainer.fit(data, steps=steps, log_every=1)
    # End snippet: grpo
    tracker.close()
    assert int(state.updates) == steps
    curve = [{"step": row["step"], "reward": row["scalars"]["rollout/reward/mean"]}
             for line in (out / "tracking/scalars.jsonl").read_text().splitlines()
             if "rollout/reward/mean" in (row := json.loads(line))["scalars"]]
    assert len(curve) == steps and all(0 <= row["reward"] <= 1 for row in curve)
    return {"steps": steps, "reward": "Adjacent response tokens counting up modulo 13",
            "reports": "tracking/scalars.jsonl"}


def pretrained(out, smoke):
    # Begin snippet: pretrained-source
    source = "Qwen/Qwen3-0.6B"
    # End snippet: pretrained-source
    if smoke:
        source = str(ROOT / "tests/fixtures/hf/qwen3-tiny")
    # Begin snippet: finetune-load
    bundle = PretrainedDecoder.load(source, dtype=jnp.bfloat16, max_seq_len=128)
    # End snippet: finetune-load
    task = bundle.text_generation(sampling=Sampling(temperature=0))
    if smoke:
        task = replace(task, processor=RunProcessor(ByteTokenizer()))
        prompt = np.load(ROOT / "tests/fixtures/hf/qwen3-tiny/input_ids.npy")[:1, :8]
        training_tokens = np.pad(prompt, ((0, 0), (0, 9 - prompt.shape[1])))
    else:
        # Begin snippet: finetune-data
        prompt = "The capital of France is"
        processed = bundle.processor("The capital of France is Paris.")
        training_tokens = np.asarray(processed.tokens[:, :9], np.int32)
        # End snippet: finetune-data
    # Begin snippet: finetune-dataset
    data = Dataset(train=lambda partition: itertools.repeat({"text": training_tokens}),
                   val=None, records=1, batch=1)
    # End snippet: finetune-dataset
    text = task(prompt, 12, key=0).text
    # Begin snippet: finetune
    objective = LMObjective(bundle, seq_len=training_tokens.shape[1] - 1, ema_decay=None)
    trainer = Trainer(objective, optax.sgd(1e-5), key=jax.random.key(0))
    state = trainer.fit(data, steps=1)
    bundle.save(out / "export", variables=state.variables, max_shard_size="128MB")
    # End snippet: finetune
    assert int(state.updates) == 1
    pairs = zip(jax.tree.leaves(bundle.variables), jax.tree.leaves(state.variables), strict=True)
    assert any(not np.array_equal(np.asarray(before), np.asarray(after)) for before, after in pairs)
    assert (out / "export/config.json").is_file()
    return {"source": source, "text": list(text), "steps": int(state.step), "export": "export"}


def serving(out, smoke):
    source = str(ROOT / "tests/fixtures/hf/qwen3-tiny") if smoke else "Qwen/Qwen3-0.6B"
    bundle = PretrainedDecoder.load(source, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16,
                                    max_seq_len=128, mesh=MeshSpec())
    if smoke:
        bundle = replace(bundle, processor=RunProcessor(ByteTokenizer()))
        prompts = ["dew", "jax"]
    else:
        prompts = ["The capital of France is", "The capital of Japan is"]
    # Begin snippet: serving
    task = bundle.text_generation(sampling=Sampling(temperature=0))
    server = Server.from_task(task, slots=4, capacity=128)
    results = server(prompts, 24, key=0)
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
        assert len(quantized(["dew"], 2, key=0)) == 1
    return {"source": source, "prompts": prompts, "text": [result.text[0] for result in results],
            "weight_formats": ["int8", "fp8"]}


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
    return {"devices": jax.device_count(), "axes": dict(trainer.mesh.build().shape), "steps": int(state.step)}


def reliability(out, smoke):
    _, data = text_fixture(out)
    objective = LMObjective(decoder(), 64, ema_decay=None)
    # Begin snippet: reliability
    checkpoints = Checkpoints(str(out / "checkpoints"))
    trainer = Trainer(objective, optax.adamw(1e-3), key=jax.random.key(0),
                      checkpoints=checkpoints)
    state = trainer.fit(data, steps=2, checkpoint_every=1)
    resumed = Trainer(objective, optax.adamw(1e-3), key=jax.random.key(0),
                      checkpoints=Checkpoints(str(out / "checkpoints"))).fit(
        data, steps=3, checkpoint_every=1)
    # End snippet: reliability
    baseline = Trainer(objective, optax.adamw(1e-3), key=jax.random.key(0)).fit(data, steps=3)
    differences = []
    assert jax.tree.structure(resumed) == jax.tree.structure(baseline)
    left, _ = jax.tree_util.tree_flatten_with_path(resumed)
    right = jax.tree.leaves(baseline)
    for (path, actual), expected in zip(left, right, strict=True):
        if jax.dtypes.issubdtype(actual.dtype, jax.dtypes.prng_key):
            actual, expected = jax.random.key_data(actual), jax.random.key_data(expected)
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
        with dew.Profiler(out / "profile"):
            logits = objective.model.apply(resumed.variables, jnp.zeros((1, 8), jnp.int32))
            logits.block_until_ready()
        # End snippet: profile
    return {"saved_step": int(state.step), "resumed_step": int(resumed.step), "bit_exact": exact}


SECTIONS = {function.__name__: function for function in
            (lm, diffusion, sample_public, jepa, grpo, pretrained, serving, mesh, reliability)}


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
    if options.section == "reliability":
        # Begin snippet: determinism
        from dew.training import prepare_process
        prepare_process(multi_host=False, xla_flags="--xla_gpu_deterministic_ops=true")
        # End snippet: determinism
    options.out.mkdir(parents=True, exist_ok=True)
    if options.section == "mesh":
        jax.config.update("jax_num_cpu_devices", 4)
    if options.topology_only:
        mesh = MeshSpec(fsdp=2, tensor=2).build()
        result = {"axes": dict(mesh.shape), "devices": [device.id for device in mesh.devices.flat],
                  "backend": jax.default_backend(), "training": False}
    else:
        result = SECTIONS[options.section](options.out, options.smoke)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(result) if isinstance(x, np.ndarray))
    (options.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
