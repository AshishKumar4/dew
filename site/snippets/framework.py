"""Small runnable examples behind the landing page, using the public API.

python site/snippets/framework.py --section lm --out /tmp/dew-landing
Each cell on the page is the code between one "Begin snippet" and "End snippet"
marker, a whole program that runs in the --out directory. For CI, --smoke
serves the Hub datasets and models the cells name from small offline fixtures;
the recordings on the page do not use it.
"""

# Each section prints its result for capture_snippets.py.

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROFILE = True


def lm():
    # Begin snippet: lm
    from dew import Trainer
    from dew.config import OptimConfig
    from dew.data import load
    from dew.nn.backbones import CausalTransformer
    from dew.objectives.lm import LMObjective

    data = load("hf/winglian/tiny-shakespeare", batch=8, tokenizer="byte", seq_len=64)
    model = CausalTransformer(vocab_size=256, emb_features=32, num_layers=1, num_heads=2,
                              mlp_features=64, max_seq_len=128)
    trainer = Trainer(LMObjective(model, seq_len=64), OptimConfig(learning_rate=1e-3), key=0)
    state = trainer.fit(data, steps=3)
    # End snippet: lm
    assert int(state.step) == 3 and int(state.updates) == 3
    return {"steps": int(state.step), "parameters": sum(x.size for x in jax.tree.leaves(state.variables))}


def diffusion():
    # Begin snippet: diffusion
    from dew import Field, InputSpec, Trainer
    from dew.config import OptimConfig
    from dew.data import HFImages
    from dew.diffusion.presets import Flow
    from dew.nn.backbones import SimpleDiT
    from dew.objectives.diffusion import DiffusionObjective
    from dew.sampling import Euler

    data = HFImages(name="uoft-cs/cifar10", image_column="img", image_size=32).load(batch=64)
    model = SimpleDiT(patch_size=4, emb_features=32, num_layers=1, num_heads=2, mlp_ratio=2)
    objective = DiffusionObjective(model, Flow(), InputSpec(Field("image", (32, 32, 3))),
                                   solver=Euler(), steps=4)
    state = Trainer(objective, OptimConfig(learning_rate=1e-3), key=0).fit(data, steps=3)
    # End snippet: diffusion
    assert int(state.step) == 3 and int(state.updates) == 3
    return {"steps": int(state.step)}


def sample_public():
    # Begin snippet: sample-public
    from dew.sampling import CFG, DPMSolverMultistep, TextToImage

    pipe = TextToImage.from_pretrained("dewml/hybrid-dit-176m",
                                       revision="73b7e9ed118eaa46fb7f199e688334bb37216cdd")
    result = pipe(["green and purple northern lights over a frozen lake"],
                  key=5, steps=20, solver=DPMSolverMultistep(), guidance=CFG(5))
    result.pil()[0].save("sample.png")
    # End snippet: sample-public
    assert Path("sample.png").is_file()
    return {"shape": list(result.host().images.shape), "image": "sample.png"}


def jepa():
    # Begin snippet: jepa
    from dew import Field, Trainer
    from dew.config import OptimConfig
    from dew.data import HFImages
    from dew.objectives.jepa import JepaEncoder, JepaObjective, JepaPredictor, MultiBlockMask

    data = HFImages(name="uoft-cs/cifar10", image_column="img", image_size=32).load(batch=64)
    encoder = JepaEncoder(patch_size=4, emb_features=32, num_layers=1, num_heads=2)
    predictor = JepaPredictor(grid=(8, 8), emb_features=32, predictor_features=16,
                              num_layers=1, num_heads=2)
    objective = JepaObjective(encoder, predictor, mask=MultiBlockMask.for_grid((8, 8)),
                              sample=Field("image", (32, 32, 3)), momentum_steps=3)
    state = Trainer(objective, OptimConfig(learning_rate=1e-3), key=0).fit(data, steps=3)
    # End snippet: jepa
    assert int(state.step) == 3 and int(state.updates) == 3
    return {"steps": int(state.step), "averaged": sorted(state.averaged)}


def grpo():
    # Begin snippet: grpo
    import itertools
    import json

    import numpy as np

    from dew import LocalTracker, Trainer
    from dew.config import OptimConfig
    from dew.data import Prompts
    from dew.nn.backbones import CausalTransformer
    from dew.objectives.rl import GRPOObjective, SampledRollout
    from dew.sampling import Sampling

    def reward(data_source, completion, ground_truth, extra_info):
        tokens = [int(token) for token in completion.split()]
        pairs = list(itertools.pairwise(tokens))
        return sum(b == (a + 1) % 13 for a, b in pairs) / max(len(pairs), 1)

    rng = np.random.default_rng(0)
    records = tuple(json.dumps({"prompt": rng.integers(0, 13, 4).tolist()}) for _ in range(512))
    data = Prompts(tokenizer="byte", records=records, max_prompt_len=4).load(batch=8)
    model = CausalTransformer(vocab_size=13, emb_features=64, num_layers=2, num_heads=4,
                              head_dim=16, mlp_features=128, max_seq_len=16)
    objective = GRPOObjective(model, seq_len=11, beta=0.02)
    rollout = SampledRollout(objective, reward=reward, groups=4, max_new_tokens=8,
                             sampling=Sampling(temperature=1.0))
    tracker = LocalTracker("tracking")
    trainer = Trainer(objective, OptimConfig(learning_rate=1e-3), key=0, rollout=rollout, tracker=tracker)
    state = trainer.fit(data, steps=150, log_every=1)
    tracker.close()
    # End snippet: grpo
    assert int(state.updates) == 150
    lines = Path("tracking/scalars.jsonl").read_text().splitlines()
    reports = [json.loads(line)["scalars"] for line in lines]
    curve = [scalars["rollout/reward/mean"] for scalars in reports if "rollout/reward/mean" in scalars]
    assert len(curve) == 150 and all(0 <= value <= 1 for value in curve)
    return {"steps": 150, "reward": "Adjacent response tokens counting up modulo 13",
            "reports": "tracking/scalars.jsonl"}


def pretrained():
    # Begin snippet: pretrained
    import jax.numpy as jnp

    from dew import Trainer
    from dew.config import OptimConfig
    from dew.data import load
    from dew.interop import PretrainedDecoder
    from dew.objectives.lm import LMObjective
    from dew.sampling import Sampling

    bundle = PretrainedDecoder.load("Qwen/Qwen3-0.6B", dtype=jnp.bfloat16, param_dtype=jnp.bfloat16,
                                    max_seq_len=128)
    task = bundle.text_generation(sampling=Sampling(temperature=0))
    print(task("The capital of France is", 12, key=0).text[0])

    data = load("hf/winglian/tiny-shakespeare", batch=1, tokenizer="Qwen/Qwen3-0.6B", seq_len=64)
    trainer = Trainer(LMObjective(bundle, seq_len=64), OptimConfig(learning_rate=1e-5), key=0)
    state = trainer.fit(data, steps=1)
    bundle.save("export", variables=state.variables)
    # End snippet: pretrained
    assert int(state.updates) == 1
    pairs = zip(jax.tree.leaves(bundle.variables), jax.tree.leaves(state.variables), strict=True)
    assert any(not np.array_equal(np.asarray(before), np.asarray(after)) for before, after in pairs)
    return {"steps": int(state.step), "export": sorted(path.name for path in Path("export").iterdir())}


def decide():
    # Begin snippet: decide
    import json

    from dew.decision import Decide

    decide = Decide.from_pretrained("convaiinnovations/laya")
    response = decide.systemone({
        "state": "Hi, we were billed twice for March and I want it reversed today.",
        "questions": {
            "department": {"type": "choice", "instructions": "Which department should handle this?",
                           "criteria": {"billing": "invoices, refunds", "technical": "bugs, outages"}},
            "urgency": {"type": "score", "instructions": "How urgent is this?",
                        "criteria": ["not urgent", "soon", "blocking"]},
            "churn_risk": {"type": "noul", "instructions": "Does the user threaten to leave?"},
        },
    })
    print(json.dumps(response["answers"], indent=2))
    # End snippet: decide
    assert set(response["answers"]) == {"department", "urgency", "churn_risk"}
    return response


def serving():
    # Begin snippet: serving
    import jax.numpy as jnp

    from dew.inference.serving import Server
    from dew.interop import PretrainedDecoder
    from dew.sampling import Sampling

    bundle = PretrainedDecoder.load("Qwen/Qwen3-0.6B", dtype=jnp.bfloat16, param_dtype=jnp.bfloat16,
                                    max_seq_len=128)
    task = bundle.text_generation(sampling=Sampling(temperature=0))
    server = Server.from_task(task, slots=4, capacity=128)
    results = server(["The capital of France is", "The capital of Japan is"], 24, key=0)
    print([result.text[0] for result in results])
    # End snippet: serving
    # Begin snippet: formats
    import jax.numpy as jnp

    from dew.interop import PretrainedDecoder
    from dew.sampling import Sampling
    from dew.training.quantization import Quantization

    bundle = PretrainedDecoder.load("Qwen/Qwen3-0.6B", dtype=jnp.bfloat16, param_dtype=jnp.bfloat16,
                                    max_seq_len=128)
    task = bundle.text_generation(sampling=Sampling(temperature=0))
    int8 = task.quantized(Quantization(dtype="int8", weight_only=True))
    fp8 = task.quantized(Quantization(dtype="fp8", weight_only=True))
    # End snippet: formats
    for variant in (int8, fp8):
        quantized = Server.from_task(variant, slots=4, capacity=128)
        assert len(quantized(["dew"], 2, key=0)) == 1
    return {"text": [result.text[0] for result in results], "weight_formats": ["int8", "fp8"]}


def mesh():
    # Begin snippet: mesh
    import jax

    jax.config.update("jax_num_cpu_devices", 4)

    from dew import MeshSpec, Trainer
    from dew.config import OptimConfig
    from dew.data import load
    from dew.nn.backbones import CausalTransformer
    from dew.objectives.lm import LMObjective

    data = load("hf/winglian/tiny-shakespeare", batch=8, tokenizer="byte", seq_len=64)
    model = CausalTransformer(vocab_size=256, emb_features=32, num_layers=1, num_heads=2,
                              mlp_features=64, max_seq_len=128)
    trainer = Trainer(LMObjective(model, seq_len=64), OptimConfig(learning_rate=1e-3), key=0,
                      mesh=MeshSpec(fsdp=2, tensor=2))
    state = trainer.fit(data, steps=3)
    # End snippet: mesh
    assert int(state.step) == 3 and int(state.updates) == 3
    return {"devices": [device.id for device in trainer.mesh.build().devices.flat],
            "axes": dict(trainer.mesh.build().shape), "steps": int(state.step)}


def reliability():
    # Begin snippet: reliability
    from dew import Checkpoints, Trainer
    from dew.config import OptimConfig
    from dew.data import load
    from dew.nn.backbones import CausalTransformer
    from dew.objectives.lm import LMObjective
    from dew.training import prepare_process

    prepare_process(multi_host=False, xla_flags="--xla_gpu_deterministic_ops=true")
    data = load("hf/winglian/tiny-shakespeare", batch=8, tokenizer="byte", seq_len=64)
    model = CausalTransformer(vocab_size=256, emb_features=32, num_layers=1, num_heads=2,
                              mlp_features=64, max_seq_len=128)
    objective = LMObjective(model, seq_len=64)
    state = Trainer(objective, OptimConfig(learning_rate=1e-3), key=0,
                    checkpoints=Checkpoints("checkpoints")).fit(data, steps=2, checkpoint_every=1)
    # A new trainer on the same directory resumes from step 2.
    resumed = Trainer(objective, OptimConfig(learning_rate=1e-3), key=0,
                      checkpoints=Checkpoints("checkpoints")).fit(data, steps=3, checkpoint_every=1)
    # End snippet: reliability
    baseline = Trainer(objective, OptimConfig(learning_rate=1e-3), key=0).fit(data, steps=3)
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
    Path("resume-comparison.json").write_text(json.dumps(differences, indent=2) + "\n")
    exact = not differences
    assert exact, "resuming the checkpoint changed the parameters"
    if PROFILE:
        # Begin snippet: profile
        from dew import ProfileWindow, Trainer
        from dew.config import OptimConfig
        from dew.data import load
        from dew.nn.backbones import CausalTransformer
        from dew.objectives.lm import LMObjective

        data = load("hf/winglian/tiny-shakespeare", batch=8, tokenizer="byte", seq_len=64)
        model = CausalTransformer(vocab_size=256, emb_features=32, num_layers=1, num_heads=2,
                                  mlp_features=64, max_seq_len=128)
        trainer = Trainer(LMObjective(model, seq_len=64), OptimConfig(learning_rate=1e-3), key=0,
                          profile=ProfileWindow("profile", steps=2))
        trainer.fit(data, steps=5)
        # End snippet: profile
    return {"saved_step": int(state.step), "resumed_step": int(resumed.step), "bit_exact": exact}


SECTIONS = {function.__name__: function for function in
            (lm, diffusion, sample_public, jepa, grpo, pretrained, decide, serving, mesh, reliability)}


def offline(out):
    """Serve the Hub names the cells read from small fixtures, without the network."""
    import datasets
    from PIL import Image

    import dew.data.text
    import dew.interop.hub as hub
    from dew.data import ByteTokenizer, HFOptions
    from dew.decision import Decide
    from dew.diffusion.presets import Flow
    from dew.inference import RunProcessor
    from dew.interop import PretrainedDecoder

    # The cells tokenize into dew's cache, which must not keep these fixtures.
    os.environ["XDG_CACHE_HOME"] = str(out / "cache")
    # More images than the four validation batches HFImages holds out.
    images = [Image.new("RGB", (32, 32), (255, 0, 255))] * 320
    tables = {"winglian/tiny-shakespeare": {"text": ["dew trains jax models. " * 100] * 4},
              "uoft-cs/cifar10": {"img": images, "label": [0] * len(images)}}
    HFOptions.load = lambda self, path, split, *, streaming: datasets.Dataset.from_dict(tables[path])
    tokenizer_for = dew.data.text.tokenizer_for
    dew.data.text.tokenizer_for = lambda name: tokenizer_for("byte" if name == "Qwen/Qwen3-0.6B" else name)
    load, generation = PretrainedDecoder.load.__func__, PretrainedDecoder.text_generation

    def tiny(cls, name, **options):
        fixture = ROOT / "tests/fixtures/hf/qwen3-tiny"
        return load(cls, fixture if name == "Qwen/Qwen3-0.6B" else name, **options)

    # The fixture has no tokenizer of its own.
    def byte_text(self, **options):
        return replace(generation(self, **options), processor=RunProcessor(ByteTokenizer()))

    def run(repo_id, revision=None):
        sys.path.insert(0, str(ROOT / "tests"))
        from test_inference import make_run
        if not (out / "run").exists():
            make_run(out / "run", preset=Flow())
        return out / "run"

    PretrainedDecoder.load, PretrainedDecoder.text_generation = classmethod(tiny), byte_text
    laya, tiny_laya = Decide.from_pretrained.__func__, ROOT / "tests/fixtures/laya/tiny"
    Decide.from_pretrained = classmethod(lambda cls, name, **options: laya(cls, tiny_laya, **options))
    hub.pull_from_hub = run


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
    out = options.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if options.smoke:
        offline(out)
    os.chdir(out)
    if options.topology_only:
        from dew import MeshSpec
        jax.config.update("jax_num_cpu_devices", 4)
        mesh = MeshSpec(fsdp=2, tensor=2).build()
        result = {"axes": dict(mesh.shape), "devices": [device.id for device in mesh.devices.flat],
                  "backend": jax.default_backend(), "training": False}
    else:
        result = SECTIONS[options.section]()
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(result) if isinstance(x, np.ndarray))
    Path("result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
