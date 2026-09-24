"""Every example runs end to end on stub data: train, evaluate, export.

The older three are called in process, with a stub dataset built here. The
end-to-end four ship their own `--smoke` mode over the repo's fixtures, and
run the way a reader runs them: their own process, their own command line,
one artifact each to show they got to the end.
"""

import importlib.util
import itertools
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
import pytest
from test_diffusion_objective import RES, TOKENS, StubText

from dew.data import Dataset
from dew.inputs import Condition, Field, InputSpec
from dew.interop import load_params

REPO_ROOT = Path(__file__).resolve().parents[1]


def smoke(name, out, *arguments, offline=True):
    """One example's `--smoke` run, in its own process, on one CPU device.

    The environment is the one the docstrings tell a reader to use, minus
    the suite's eight simulated devices: a smoke run is a single-device run,
    and `HF_HUB_OFFLINE` keeps a fixture path from becoming a download. A
    harness suite reads its documents from the Hub, so that one run asks for
    the network and carries the marker. No smoke reaches a paid endpoint, so
    none is handed the caller's OpenAI key.
    """
    environment = {**{key: value for key, value in os.environ.items() if key != "OPENAI_API_KEY"},
                   "PYTHONPATH": str(REPO_ROOT / "src"),
                   "JAX_PLATFORMS": "cpu",
                   "XLA_FLAGS": "--xla_force_host_platform_device_count=1",
                   "HF_HUB_OFFLINE": "1" if offline else "0",
                   "TOKENIZERS_PARALLELISM": "false"}
    finished = subprocess.run(
        [sys.executable, str(REPO_ROOT / "examples" / f"{name}.py"), "--smoke",
         "--out", str(out), *arguments],
        cwd=REPO_ROOT, env=environment, capture_output=True, text=True, timeout=900)
    assert finished.returncode == 0, (
        f"{name} --smoke exited {finished.returncode}\n"
        f"--- stdout ---\n{finished.stdout}\n--- stderr ---\n{finished.stderr}")
    return finished


def load_example(name):
    path = REPO_ROOT / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"example_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _batches(batch, classes=None, size=RES):
    """Endless captioned batches of noise images; labels when `classes` is set."""
    def stream():
        rng = np.random.RandomState(0)
        while True:
            record = {"image": rng.randint(0, 256, (batch, size, size, 3), np.uint8),
                      "text": {"input_ids": np.ones((batch, TOKENS), np.int32),
                               "attention_mask": np.ones((batch, TOKENS), np.int32)}}
            if classes is not None:
                record["label"] = rng.randint(0, classes, (batch,), np.int32)
            yield record
    return stream


def fake_dataset(batch, classes=None, size=RES):
    return Dataset(train=lambda partition: _batches(batch, classes, size)(),
                   val=lambda partition: itertools.islice(_batches(batch, classes, size)(), 1),
                   records=4 * batch, batch=batch)


@pytest.mark.mesh
def test_train_diffusion_example_trains_samples_and_exports(tmp_path):
    example = load_example("train_diffusion")
    config = example.Config(image_size=RES, batch_size=8, steps=3, prompts=("a", "b"),
                            model=dict(patch_size=4, emb_features=16, num_layers=1, num_heads=2),
                            out=tmp_path)
    inputs = InputSpec(Field("image", (RES, RES, 3)),
                       {"textcontext": Condition(StubText.from_pretrained("stub"))})

    state = example.main(config, data=fake_dataset(8), inputs=inputs)

    assert int(state.step) == 3
    grid = np.asarray(__import__("PIL.Image").Image.open(tmp_path / "samples.png"))
    assert grid.shape == (RES, 2 * RES, 3) and grid.dtype == np.uint8
    assert (tmp_path / "export" / "model.safetensors").exists()
    assert (tmp_path / "export" / "config.json").exists()
    assert (tmp_path / "checkpoints" / "3").is_dir(), "the last step was not checkpointed"


@pytest.mark.mesh
def test_train_jepa_example_trains_probes_and_saves_the_encoder(tmp_path):
    example = load_example("train_jepa")
    # An 8x8 patch grid, the smallest the default mask geometry fits on.
    config = example.Config(classes=5, image_size=32, patch_size=4, batch_size=8, steps=3,
                            model=dict(emb_features=32, num_layers=2, num_heads=2), out=tmp_path)

    state = example.main(config, data=fake_dataset(8, classes=5, size=32))

    assert int(state.step) == 3
    saved = load_params(tmp_path / "encoder.safetensors")
    averaged = state.averaged["params"]["context_encoder"]
    assert jax.tree.structure(saved) == jax.tree.structure(averaged), "not the encoder's tree"
    assert all(np.array_equal(np.asarray(a), b) for a, b in zip(
        jax.tree.leaves(averaged), jax.tree.leaves(saved), strict=True))


@pytest.mark.mesh
def test_train_lm_example_trains_and_generates(tmp_path):
    tokens = tmp_path / "tokens"
    tokens.mkdir()
    text = ("ab" * 2000).encode()
    (tokens / "train.bin").write_bytes(text[:3600])
    (tokens / "val.bin").write_bytes(text[3600:])
    (tokens / "meta.json").write_text(json.dumps(
        {"tokenizer": "byte", "vocab_size": 256, "dtype": "uint8"}))
    example = load_example("train_lm")
    config = example.Config(tokens=tokens, sequence_length=32, batch_size=8, steps=3,
                            model=dict(emb_features=16, num_layers=1, num_heads=2),
                            prompt="ab", sample_tokens=8, out=tmp_path / "run")

    state = example.main(config)

    assert int(state.step) == 3
    sample = (tmp_path / "run" / "sample.txt").read_text()
    assert sample.startswith("ab") and len(sample) > 2


# ---------------------------------------------------------------------------------
# The end-to-end scripts, run the way their docstrings say to run them
# ---------------------------------------------------------------------------------

def test_train_flowers_tpu_smoke_samples_a_grid_and_scores_it(tmp_path):
    """The diffusion run's whole arc: synthetic ArrayRecords in, a run
    directory with its record and checkpoint, a samples grid out of
    `dew.pipeline`, and the CLIPScore and FID of that grid, both scored
    against a committed fixture rather than a download."""
    smoke("train_flowers_tpu", tmp_path)

    grid = np.asarray(__import__("PIL.Image").Image.open(tmp_path / "samples.png"))
    assert grid.shape == (16, 4 * 16, 3) and grid.dtype == np.uint8
    scored = json.loads((tmp_path / "eval.json").read_text())
    assert "clip_score" in scored and scored["fid"] >= 0
    assert (tmp_path / "checkpoints" / "smoke" / "run.json").is_file()


def test_sft_diffusion_gemma_smoke_writes_an_adapter_and_generates_from_it(tmp_path):
    """LoRA over a host-streamed base: the PEFT directory the run publishes,
    and the canvas `dew.pipeline` decodes once that directory is read back
    onto the base weights."""
    smoke("sft_diffusion_gemma", tmp_path)

    adapter = tmp_path / "adapter"
    assert (adapter / "adapter_config.json").is_file()
    assert (adapter / "adapter_model.safetensors").is_file()
    config = json.loads((adapter / "adapter_config.json").read_text())
    assert config["peft_type"] == "LORA" and config["target_modules"]
    assert len((tmp_path / "samples.txt").read_text().splitlines()) == 2


def test_sft_gemma4_smoke_trains_on_chat_rows_and_exports_the_decoder(tmp_path):
    """Full-weight SFT over packed conversations: the run record `dew.pipeline`
    reads, and the Hugging Face directory `export_run` writes beside it."""
    smoke("sft_gemma4", tmp_path)

    export = tmp_path / "export"
    assert {entry.name for entry in export.iterdir()} >= {
        "config.json", "generation_config.json", "model.safetensors"}
    run = tmp_path / "checkpoints" / tmp_path.name
    assert json.loads((run / "run.json").read_text())["objective"] == "lm"
    assert json.loads((export / "generation_config.json").read_text())["tokenizer_name"]


@pytest.mark.parametrize("new_tokens", [1, 128])
def test_train_rlvr_starts_sglang_with_room_for_a_prompt_at_the_window_and_its_full_budget(tmp_path, new_tokens):
    """SGLang 0.5.20 refuses an input of `context - 6` ids or more
    (`max_req_input_len`) and caps a budget at `context - 2 - input`
    (`max_req_len - input - 1`). A prompt truncated to `--prompt-tokens`
    must pass the first and keep `--new-tokens` under the second, or its
    draw ends short with a "length" the rollout refuses."""
    example = load_example("train_rlvr")
    config = example.Config(backend="sglang", prompt_tokens=128, new_tokens=new_tokens, out=tmp_path)
    command, _ = example.engine_command(config, tmp_path / "served")
    context = int(command[command.index("--context-length") + 1])
    assert config.prompt_tokens < context - 6
    assert min(new_tokens, context - 2 - config.prompt_tokens) == new_tokens


def test_train_rlvr_smoke_commits_every_update_one_behind(tmp_path):
    """The smoke's completions all run out of their eight tokens, so a run
    that masked truncations would commit nothing and never push. The
    example scores them: both updates commit, and the second trains on
    draws submitted one update earlier."""
    smoke("train_rlvr", tmp_path)
    summary = json.loads((tmp_path / "rewards.json").read_text())
    assert summary["updates"] == 2 and summary["max_lag"] == 1


def test_train_rlvr_turns_smoke_trains_through_environment_source(tmp_path):
    """`--turns 2` runs its sessions through EnvironmentSource and commits both updates, one behind.
    The tiny model's attempts end on the token budget, so the feedback path
    is the next test's."""
    smoke("train_rlvr", tmp_path, "--turns", "2")
    summary = json.loads((tmp_path / "rewards.json").read_text())
    assert summary["updates"] == 2 and summary["max_lag"] == 1


def test_train_rlvr_turns_feed_a_failed_attempt_back_and_run_each_program_once():
    """The example's own Attempts and verifier through EnvironmentSource, on a
    server that draws a failing program and then a passing one: the second
    call's prompt extends the first call's ids with the test report, both pack
    into one chain, and each program runs once."""
    from concurrent.futures import Future

    from dew.inference import Draw
    from dew.objectives.rl import Status, Task, pack
    from dew.sampling import Sampling

    example = load_example("train_rlvr")
    eos, failing, passing = 9, (5, 5), (6, 6)
    runs = []

    class Server:
        sampling = Sampling(temperature=1.0, eos_id=eos)
        version = 0

        def submit(self, prompt, max_new_tokens, *, seed):
            program = failing if not runs else passing
            future = Future()
            future.set_result(Draw(tuple(prompt), (*program, eos), (-.5,) * 3, None, True, 0))
            return future

    def reward(source, completion, cases, info):
        runs.append(completion)
        return 1.0 if completion == "6 6" else 1 / 3

    config = example.Config(prompt_tokens=8, new_tokens=4, turns=2, prompts=1, groups=1)
    source = example.attempts_source(Server(), reward, lambda ids: " ".join(map(str, ids)),
                                     lambda text: [7, 7], config)
    session = source.submit(Task("t", {"prompt": (1, 2), "truth": "cases"}), 1, version=0)[0].result(timeout=10)
    source.close()
    assert session.status == Status.COMPLETED and session.reward == 1.0
    first, second = session.calls
    assert second.prompt_ids == (*first.prompt_ids, *first.sampled_ids, 7, 7)
    assert pack([session], 32)["text_segment_ids"].max() == 1
    assert runs == ["5 5", "6 6"]


def test_train_harbor_waits_for_the_gateway_it_is_told_to_launch(tmp_path):
    """The real path writes the served policy and then waits for the gateway the user starts; with none
    up it gives up with the readiness error instead of failing on the first gateway call."""
    example = load_example("train_harbor")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    config = example.Config(model=str(REPO_ROOT / "tests/fixtures/hf/qwen2-tiny"), tasks=(tmp_path,), width=64,
                            gateway=f"http://127.0.0.1:{port}", served=tmp_path / "served", out=tmp_path / "out",
                            ready_timeout=1.0)
    with pytest.raises(RuntimeError, match="no healthy worker"):
        example.main(config)
    assert (tmp_path / "served" / "config.json").is_file()


def test_train_harbor_smoke_trains_on_gateway_recorded_harness_calls(tmp_path):
    """Two updates of HarborSource trials whose model calls the stand-in gateway records from the policy's
    own draws; the second trains on sessions submitted one update earlier."""
    smoke("train_harbor", tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["updates"] == 2 and summary["max_lag"] == 1


def test_evaluate_and_serve_smoke_reports_perplexity_and_a_greedy_continuation(tmp_path):
    """The evaluation report of a run the script trains first: the perplexity
    `evaluate` scores over the held-out split, a greedy continuation, and a
    served comparison that neither SDK can reach. An installed SDK reports
    the endpoint unreachable, an absent one is skipped, and neither needs an
    OpenAI key, which the smoke's environment does not carry."""
    smoke("evaluate_and_serve", tmp_path,
          "--openai-base-url", "http://127.0.0.1:1/v1", "--ollama-host", "http://127.0.0.1:1")

    report = json.loads((tmp_path / "report.json").read_text())
    assert report["perplexity"]["val/perplexity"] > 0
    assert len(report["greedy"]) == 8
    for sdk, answer in report["served"].items():
        installed = importlib.util.find_spec(sdk) is not None
        assert answer.startswith("unreachable: http://127.0.0.1:1" if installed
                                 else f"skipped: pip install {sdk}"), (sdk, answer)
    assert set(report["served"]) == {"openai", "ollama"}


def test_evaluate_and_serve_smoke_scores_a_diffusion_run_it_is_pointed_at(tmp_path):
    """The image half, over the run the diffusion example trains. Both
    metrics read a committed fixture instead of the checkpoint each default
    names, so the pair scores with no download: the tiny CLIP, and the FID
    extractor the smoke passes as `--inception-weights`."""
    images = tmp_path / "images"
    smoke("train_flowers_tpu", images)

    smoke("evaluate_and_serve", tmp_path / "report",
          "--image-run", str(images / "checkpoints" / "smoke"),
          "--clip-model", str(REPO_ROOT / "tests/fixtures/clip/tiny"))

    report = json.loads((tmp_path / "report" / "report.json").read_text())
    assert "clip_score" in report["images"]
    assert report["images"]["fid"] >= 0, "the offline FID did not score"


@pytest.mark.network
def test_evaluate_and_serve_smoke_runs_an_lm_eval_harness_task(tmp_path):
    """`DewLM` behind the harness's own `simple_evaluate`, on two documents
    of a real suite; the task's data comes from the Hub."""
    smoke("evaluate_and_serve", tmp_path, "--tasks", "hellaswag",
          "--harness-limit", "2", offline=False)

    harness = json.loads((tmp_path / "report.json").read_text())["harness"]
    assert {name.split("/")[0] for name in harness} == {"hellaswag"}
    assert all(0.0 <= value <= 1.0 for name, value in harness.items() if name.endswith("acc,none"))


def test_train_rlvr_native_holds_one_copy_of_the_served_weights_after_pushes(tmp_path, monkeypatch):
    """The native backend pushes the policy into Dew's own server every
    update. After a push the device holds the served weights once: the
    trainer's copy is float32 and the server's bfloat16, so every live
    bfloat16 array of a served leaf's shape is the server's. A second copy is
    a model's worth of memory the run keeps for good (1.1 GiB on Qwen3-0.6B,
    the headroom its single-turn run ran out of at update 19)."""
    import collections

    import jax

    from dew.objectives.rl import RolloutScheduler

    example = load_example("train_rlvr")
    counts = []
    schedule = RolloutScheduler.__call__

    def counted(self, state, batch, key):
        packed = schedule(self, state, batch, key)
        served = collections.Counter((array.shape, array.dtype) for array in jax.live_arrays()
                                     if array.dtype == jax.numpy.bfloat16)
        counts.append(served)
        return packed

    monkeypatch.setattr(RolloutScheduler, "__call__", counted)
    example.main(example.Config(smoke=True, out=tmp_path))
    source = example.load_pretrained(str(example.SMOKE_MODEL), dtype="float32")
    tree = collections.Counter((leaf.shape, jax.numpy.dtype(jax.numpy.bfloat16))
                               for leaf in jax.tree.leaves(source.variables))
    # The second call pushed update 1's weights before it returned.
    assert len(counts) == 2
    assert all(counts[-1][key] == number for key, number in tree.items()), \
        {str(key): (counts[-1][key], number) for key, number in tree.items() if counts[-1][key] != number}


def test_no_constant_answer_passes_a_quarter_of_a_train_rlvr_task():
    """A policy that prints one number whatever the input earns that number's
    share of a task's cases. With a and b uniform in 1..99, print(1) passed
    61% of the gcd cases and print(0) 49% of the multiples ones, a reward for
    guessing that the policy learns. Over 16,000 records every task's most
    common answer covers under a quarter of its cases."""
    import collections
    import json

    example = load_example("train_rlvr")
    answers = collections.defaultdict(collections.Counter)
    for row in map(json.loads, example.records(16_000, seed=0)):
        for case in json.loads(row["ground_truth"]):
            answers[row["prompt"][0]["content"].split(" and prints ", 1)[1][:60]][case["stdout"]] += 1
    shares = {task: counts.most_common(1)[0][1] / sum(counts.values()) for task, counts in answers.items()}
    assert len(shares) == len(example.TASKS)
    assert max(shares.values()) < 0.25, {task: round(share, 3) for task, share in shares.items()}
