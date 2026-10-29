"""Every example runs end to end on stub data: train, evaluate, export.

The older three are called in process, with a stub dataset built here. The
end-to-end four ship their own `--smoke` mode over the repo's fixtures, and
run the way a reader runs them: their own process, their own command line,
one artifact each to show they got to the end.
"""

import itertools
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
import pytest
from diffusion_stubs import RES, TOKENS, StubText
from PIL import Image
from process_support import dew_train, logged_losses
from test_inference import make_run
from tool_support import load

from dew import Checkpoints
from dew.artifacts import uint8_pixels
from dew.data.dataset import DataPartition, Loader
from dew.diffusion.presets import Flow
from dew.inference import TextToImage
from dew.inputs import Condition, Field, InputSpec
from dew.interop import load_params

REPO_ROOT = Path(__file__).resolve().parents[1]


def single_device(offline=True, devices=1) -> dict[str, str]:
    """One CPU device by default, or an isolated mesh for the MoE example.
    `HF_HUB_OFFLINE` keeps a fixture path from becoming a download.
    A harness suite reads its documents from the Hub, so that one
    run asks for the network and carries the marker. No smoke reaches a paid
    endpoint, so none is handed the caller's OpenAI key.
    """
    return {**{key: value for key, value in os.environ.items() if key != "OPENAI_API_KEY"},
            "PYTHONPATH": str(REPO_ROOT / "src"),
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}",
            "HF_HUB_OFFLINE": "1" if offline else "0",
            "TOKENIZERS_PARALLELISM": "false"}


def smoke(name, out, *arguments, offline=True, script=None, smoke_args=True, devices=1):
    """Run an example's smoke mode or existing small controls in an isolated CPU process."""
    finished = subprocess.run(
        [sys.executable, str(script or REPO_ROOT / "examples" / f"{name}.py"),
         *(["--smoke", "--out", str(out)] if smoke_args else []), *arguments],
        cwd=REPO_ROOT, env=single_device(offline, devices), capture_output=True, text=True, timeout=900)
    assert finished.returncode == 0, (
        f"{name} --smoke exited {finished.returncode}\n"
        f"--- stdout ---\n{finished.stdout}\n--- stderr ---\n{finished.stderr}")
    return finished


def test_readme_demo_resumes_updates_its_policy_freezes_the_reference_and_draws_flow(tmp_path):
    """The public chain restores a real checkpoint and keeps DPO's reference unchanged."""
    out = tmp_path / "demo"
    smoke("readme_demo", out, "--out", str(out), smoke_args=False)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["language_model"]["first_step"] == 20
    assert summary["language_model"]["resumed_step"] == summary["language_model"]["updates"] == 24
    assert summary["dpo"] == {"updates": 4, "reference_max_change": 0.0}
    checkpoints = Checkpoints(str(out / "lm-checkpoints"))
    first, _ = checkpoints.restore(step=20)
    resumed, _ = checkpoints.restore(step=24)
    assert int(first["updates"]) == 20 and int(resumed["updates"]) == 24
    assert any(not np.array_equal(a, b) for a, b in zip(
        jax.tree.leaves(first["variables"]["params"]),
        jax.tree.leaves(resumed["variables"]["params"]), strict=True))
    preview = np.load(out / "flow-preview.npy")
    assert summary["flow"]["updates"] == 3 and preview.shape == (4, 8, 8, 3) and np.isfinite(preview).all()
    np.testing.assert_array_equal(np.asarray(Image.open(out / "flow-preview.ppm")),
                                  np.concatenate(list(uint8_pixels(preview)), axis=1))


def test_text_to_image_example_records_requested_seeds_and_their_real_pixels(tmp_path, monkeypatch):
    """A trained local DiT and real tiny CLIP render the manifest's prompts and keys."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    run, out = tmp_path / "run", tmp_path / "samples"
    make_run(run, Flow(), encoder="clip_text", checkpoint=str(REPO_ROOT / "tests/fixtures/clip/tiny"),
             steps=1)
    example = load("sample_text_to_image", "examples")
    config = example.Config(model=str(run), out=out, prompts=("a bird", "a flower"),
                            seeds=(3, 17), samplers=("heun40",), negative="a cat")
    example.main(config)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["model"] == str(run) and manifest["negative"] == "a cat"
    assert [(batch["seed"], batch["images"]) for batch in manifest["batches"]] == [(3, 2), (17, 2)]
    pipe = TextToImage.from_run(str(run))
    solver, steps, guidance = example.SAMPLERS["heun40"]
    for index, seed in enumerate(config.seeds):
        prepared = pipe.prepare(config.prompts, key=seed, steps=steps, unconditional=config.negative)
        images = pipe(prepared, key=seed, steps=steps, solver=solver, guidance=guidance).host().images
        pixels = uint8_pixels(images)
        for row, prompt in enumerate(config.prompts):
            record = manifest["images"][2 * index + row]
            assert record == {"file": f"heun40/p{row}_s{seed}.png", "prompt": prompt, "seed": seed,
                              "sampler": "Heun", "steps": 40, "guidance": 5.0,
                              "guidance_interval": [0.0, 1.0]}
            np.testing.assert_array_equal(np.asarray(Image.open(out / record["file"])), pixels[row])
    assert (out / "heun40/p0_s3.png").read_bytes() != (out / "heun40/p0_s17.png").read_bytes()


def test_moe_mesh_example_routes_and_serves_on_eight_cpu_devices(tmp_path):
    """Existing tiny controls train exchange dispatch and record actual routed generations."""
    out = tmp_path / "mesh"
    smoke("moe_mesh", out, "--out", str(out), "--steps", "2", "--snapshot-every", "1",
          "--batch-size", "8", "--sequence-length", "12", "--experts", "2", "--top-k", "1",
          smoke_args=False, devices=8)
    record = json.loads((out / "moe-mesh.json").read_text())
    assert record["mesh"] == {"data": 2, "expert": 2, "fsdp": 2, "tensor": 1, "sequence": 1, "stage": 1}
    assert len(record["devices"]) == 8 and record["dispatch"] == "exchange" and record["all_to_all_ops"] > 0
    assert [step["step"] for step in record["steps"]] == [1, 2]
    assert all(np.isfinite(step["loss"]) for step in record["steps"])
    assert len(record["generated"]) == 8 and all(record["generated"])
    assert record["frames"][-1]["step"] is None and len(record["frames"]) == 3
    assert record["frames"][-1]["rows"] == [[row, row + 1] for row in range(8)]
    for frame in record["frames"]:
        for layer in frame["layers"]:
            sent = np.asarray(layer["sent"])
            assert sent.shape == (8, 8) and sent.sum() == sum(layer["per_expert"]) == 96
            assert (sent.sum() - np.trace(sent)) > 0 and layer["rounds"] >= 1
            assert np.asarray(layer["by_class"]).sum() == 96
    assert "<svg" in (out / "moe-mesh.svg").read_text()


@pytest.mark.parametrize("section", ["reliability"])
def test_landing_snippet_runs(section, tmp_path):
    """Each landing cell runs, offline from small fixtures, and does what it shows."""
    smoke("landing", tmp_path, "--section", section,
          script=REPO_ROOT / "site/snippets/framework.py")
    result = json.loads((tmp_path / "result.json").read_text())
    assert result
    assert result["bit_exact"] and result["resumed_step"] == 3


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
    return Loader(train=lambda partition: _batches(batch, classes, size)(),
                   val=lambda partition: itertools.islice(_batches(batch, classes, size)(), 1),
                   records=4 * batch, batch=batch)


@pytest.mark.mesh
def test_train_diffusion_example_trains_samples_and_exports(tmp_path):
    example = load("train_diffusion", "examples")
    config = example.Config(image_size=RES, batch_size=8, steps=3, prompts=("a", "b"),
                            model={"patch_size": 4, "emb_features": 16, "num_layers": 1, "num_heads": 2},
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


def test_train_flowers_trains_on_every_record_and_its_profiling_run_draws_nothing(tmp_path):
    example = load("train_flowers", "examples")
    config = example.Config(data=tmp_path, output=tmp_path / "run", steps=4, batch=8, image_size=RES,
                            profile=2)
    state = example.main(config, data=fake_dataset(8))

    assert int(state.step) == 4
    assert any((tmp_path / "run" / "profile").rglob("*.xplane.pb")), "no trace was written"
    assert not (tmp_path / "run" / "samples.png").exists(), "a profiling run drew samples"


@pytest.mark.mesh
def test_train_jepa_example_trains_probes_and_saves_the_encoder(tmp_path):
    example = load("train_jepa", "examples")
    # An 8x8 patch grid, the smallest the default mask geometry fits on.
    config = example.Config(classes=5, image_size=32, patch_size=4, batch_size=8, steps=3,
                            model={"emb_features": 32, "num_layers": 2, "num_heads": 2}, out=tmp_path)

    state = example.main(config, data=fake_dataset(8, classes=5, size=32))

    assert int(state.step) == 3
    saved = load_params(tmp_path / "encoder.safetensors")
    averaged = state.averaged["params"]["context_encoder"]
    assert jax.tree.structure(saved) == jax.tree.structure(averaged), "not the encoder's tree"
    assert all(np.array_equal(np.asarray(a), b) for a, b in zip(
        jax.tree.leaves(averaged), jax.tree.leaves(saved), strict=True))


# ---------------------------------------------------------------------------------
# The end-to-end scripts, run the way their docstrings say to run them
# ---------------------------------------------------------------------------------

def test_a_python_experiment_trains_and_its_record_trains_the_same_losses_in_a_new_process(tmp_path):
    """`dew train examples/train_supervised.py --set ...` trains the run the
    file builds, each `--set` read as its field's type reads it. A new
    process given only the run's `run.json`, trusting the file's module,
    rebuilds the same model, loss and data and trains the same loss, step
    for step: the record carries the run exactly."""
    environment = {**single_device(), "XDG_CACHE_HOME": str(tmp_path / "cache"),
                   "PYTHONPATH": os.pathsep.join([str(REPO_ROOT / "src"), str(REPO_ROOT / "examples")])}

    def train(*arguments):
        dew_train(tmp_path, environment, *arguments)

    def losses(directory):
        return logged_losses(directory / "train_supervised")

    train(str(REPO_ROOT / "examples" / "train_supervised.py"),
          "--set", f"trainer.checkpoint_dir={tmp_path / 'first'}", "--set", "trainer.steps=6",
          "--set", "trainer.log_every=1", "--set", "model.hidden=8")
    record = tmp_path / "first" / "train_supervised" / "run.json"
    written = json.loads(record.read_text())["fields"]
    assert written["trainer"]["steps"] == 6 and written["model"]["fields"]["hidden"] == 8
    assert written["objective"]["fields"]["loss"] == {"function": "train_supervised:cross_entropy"}

    train(str(record), "--trust", "train_supervised", "--set", f"trainer.checkpoint_dir={tmp_path / 'again'}")
    assert len(losses(tmp_path / "first")) == 6
    assert losses(tmp_path / "again") == losses(tmp_path / "first")


def test_train_flowers_tpu_smoke_samples_a_grid_and_scores_it(tmp_path):
    """The diffusion run's whole arc: synthetic ArrayRecords in, a run
    directory with its record and checkpoint, a samples grid out of
    `dew.pipeline`, and the held-out split's CLIPScore and FID, both scored
    against a committed fixture rather than a download."""
    smoke("train_flowers_tpu", tmp_path)

    grid = np.asarray(__import__("PIL.Image").Image.open(tmp_path / "samples.png"))
    assert grid.shape == (16, 4 * 16, 3) and grid.dtype == np.uint8
    scored = json.loads((tmp_path / "eval.json").read_text())
    assert "val/clip_score" in scored and scored["val/fid"] >= 0
    assert (tmp_path / "checkpoints" / "smoke" / "run.json").is_file()


def test_sft_diffusion_gemma_images_smoke_trains_on_two_splits_and_captions_the_third(tmp_path):
    """The image-reading run's arc: the train and validation splits train it,
    with random crops and flips, every step's loss finite and its weights moved
    from one checkpoint to the next, and the run read back with `dew.pipeline`
    captions every image of the test split it never trained on, each caption
    the one the trained model in the run's own process writes for that image."""
    import pyarrow.parquet as parquet

    smoke("sft_diffusion_gemma_images", tmp_path)

    report = json.loads((tmp_path / "result.json").read_text())
    assert (report["train_images"], report["test_images"], report["classes"]) == (16, 8, 2)
    # The test file shares no image with the two that train, and the run's own batches, read as it
    # reads them, train on ids of those two alone and caption each test id once.
    example = load("sft_diffusion_gemma_images", "examples")
    files = {split: tmp_path / "data" / f"{split}.parquet" for split in example.SPLITS}
    images = {split: {row["bytes"] for row in parquet.read_table(path).column("image").to_pylist()}
              for split, path in files.items()}
    assert len(images["test"]) == 8 and not images["test"] & (images["train"] | images["validation"])
    ids = {split: set(parquet.read_table(path).column("image_id").to_pylist())
           for split, path in files.items()}
    config = example.sized(example.Config(smoke=True, out=tmp_path))
    parts = {split: example.Files(str(path), format="parquet") for split, path in files.items()}
    data = example.caption_data(config, parts, example.class_names(parts))
    trained = [int(image) for batch in itertools.islice(data.train(DataPartition()), config.steps)
               for image in np.asarray(batch["image_id"])]
    assert set(trained) <= ids["train"] | ids["validation"] and set(trained) & ids["validation"]
    from dew.batch import VALID_ROWS

    # A batch marks its repeats where it holds any (`valid_rows`), as the example reads it.
    captioned = [int(image) for batch in data.val(DataPartition())
                 for image, real in zip(np.asarray(batch["image_id"]),
                                        batch.get(VALID_ROWS, np.ones(len(batch["image_id"]), bool)),
                                        strict=True)
                 if real]
    assert sorted(captioned) == sorted(ids["test"])
    run = tmp_path / "checkpoints" / "flowers-caption"
    journal = [json.loads(row) for row in (run / "tracking" / "scalars.jsonl").read_text().splitlines()]
    losses = [row["scalars"]["train/loss"] for row in journal if "train/loss" in row["scalars"]]
    assert len(losses) == 4 and all(math.isfinite(loss) for loss in losses)
    checkpoints = Checkpoints(str(run))
    early, _ = checkpoints.restore(step=2)
    late, _ = checkpoints.restore(step=4)
    assert (int(early["updates"]), int(late["updates"])) == (2, 4)
    assert any(not np.array_equal(a, b) for a, b in zip(
        jax.tree.leaves(early["variables"]["params"]), jax.tree.leaves(late["variables"]["params"]),
        strict=True))
    captions = [json.loads(row) for row in (tmp_path / "captions.jsonl").read_text().splitlines()]
    assert len(captions) == 8 and {row["label"] for row in captions} == {"pink rose", "yellow tulip"}
    assert report["restored_matches_live"] is True


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


def test_sft_diffusion_gemma_puts_each_assistant_turn_alone_on_the_canvases(tmp_path):
    """The example's rows, as its objective reads them: one per assistant
    turn, the template's render of the conversation before the turn as the
    clean prompt (its tail, past `prompt_tokens`), and the turn's completion
    then EOS as the only canvas targets. Packed windows split at a fixed
    prompt length put user and template ids on the canvases and cut turns
    apart."""
    from transformers import AutoTokenizer

    example = load("sft_diffusion_gemma", "examples")
    config = example.smoke_config(example.Config(smoke=True, out=tmp_path))
    _, objective = example.adapted_objective(config)
    from dew.data.sources import source

    # One pass over the conversations in their order, batch by batch.
    batches = list(example.chat_rows(config, objective, source(config.chat)))
    template = AutoTokenizer.from_pretrained(config.model)
    expected = []
    for turns in example.CONVERSATIONS:
        for index in range(1, len(turns), 2):
            prompt = template.apply_chat_template(turns[:index], add_generation_prompt=True,
                                                  tokenize=True, return_dict=False)
            whole = template.apply_chat_template(turns[:index + 1], tokenize=True, return_dict=False)
            expected.append((prompt[:objective.prompt_length], whole[len(prompt):]))

    batch = {name: np.concatenate([np.asarray(part[name]) for part in batches])
             for name in ("text", "canvas_mask")}
    _, tokens, response, _, valid, targets, _ = objective._row(batch)
    assert len(batch["text"]) == len(expected) == 4
    for row, (prompt, completion) in enumerate(expected):
        head = np.asarray(tokens[row, :objective.prompt_length])
        assert head[np.asarray(valid[row, :objective.prompt_length])].tolist() == prompt
        scored = np.asarray(response[row])[np.asarray(targets[row])].tolist()
        closing = -len(completion) % objective.canvas_size
        assert scored == completion + [template.eos_token_id] * closing


def test_sft_gemma4_smoke_trains_on_chat_rows_and_exports_the_decoder(tmp_path):
    """Full-weight SFT over packed conversations: the run record `dew.pipeline`
    reads, and the Hugging Face directory `Pretrained.from_run(...).save(...)` writes beside it."""
    smoke("sft_gemma4", tmp_path)

    export = tmp_path / "export"
    assert {entry.name for entry in export.iterdir()} >= {
        "config.json", "generation_config.json", "model.safetensors"}
    run = tmp_path / "checkpoints" / tmp_path.name
    objective = json.loads((run / "run.json").read_text())["fields"]["objective"]
    assert objective["name"] == "dew.objectives.lm.objective:LMObjective"
    assert (export / "tokenizer_config.json").is_file()
    assert "tokenizer_name" not in json.loads((export / "generation_config.json").read_text())


def test_sft_gemma4_validates_on_conversations_it_never_trains_on(tmp_path):
    """The run's validation windows hold the first `held_out` conversations,
    and no training window holds an id of theirs: the canned conversations
    share no word, so a shared id would be a shared conversation. Validation
    used to read the training split again and score what the run trained on."""
    from transformers import AutoTokenizer

    from dew.data.compile import compile

    example = load("sft_gemma4", "examples")
    config, tokenizer, chat = example.smoke_config(example.Config(smoke=True, out=tmp_path))
    run = example.run_config(config, tokenizer, chat)
    data = compile(run.data, validation=run.validation, provision=True)
    vocabulary = AutoTokenizer.from_pretrained(tokenizer)
    words = [{token for turn in turns
              for token in vocabulary.encode(turn["content"], add_special_tokens=False)}
             for turns in example.SMOKE_CONVERSATIONS]
    held, kept = set().union(*words[:config.held_out]), set().union(*words[config.held_out:])

    def read(batches) -> set[int]:
        return {int(token) for batch in batches for token in np.asarray(batch["text"]).ravel()}

    assert data.val is not None
    assert read(data.val(DataPartition())) & (held | kept) == held
    trained = read(itertools.islice(data.train(DataPartition()), 4 * (data.steps_per_epoch or 1)))
    assert trained & (held | kept) and trained & held == set()


def test_sft_gemma4_runs_its_documented_configuration_on_one_machine(tmp_path):
    """Outside --smoke the run is the documented one: bf16, four loader
    workers, the persistent cache, and a process pool when a cluster started
    the run. On one machine it runs alone. The committed tiny Gemma 4 and the
    vocabulary its ids were written against stand in for the Hub checkpoint,
    one directory as a published checkpoint is."""
    fixtures = REPO_ROOT / "tests/fixtures/hf"
    model = tmp_path / "model"
    shutil.copytree(fixtures / "gemma4-ple", model)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(fixtures / "diffusion-gemma-workflow" / name, model / name)
    example = load("sft_gemma4", "examples")
    chat = example.write_conversations(example.SMOKE_CONVERSATIONS, tmp_path)

    finished = subprocess.run(
        [sys.executable, str(REPO_ROOT / "examples/sft_gemma4.py"), "--model", str(model),
         "--dataset", str(chat), "--sequence-length", "15", "--batch-size", "2",
         "--accumulation", "2", "--steps", "2", "--held-out", "2", "--out", str(tmp_path / "run")],
        cwd=REPO_ROOT, env=single_device(), capture_output=True, text=True, timeout=900)

    assert finished.returncode == 0, finished.stderr[-3000:]
    assert (tmp_path / "run" / "export" / "model.safetensors").is_file()


def commits_both_updates_one_behind(example, tmp_path, summary, *arguments):
    """`example`'s smoke, then its `summary` file's account: both updates committed, the second on draws
    submitted one update earlier."""
    smoke(example, tmp_path, *arguments)
    written = json.loads((tmp_path / summary).read_text())
    assert written["updates"] == 2 and written["max_lag"] == 1


def test_train_rlvr_smoke_commits_every_update_one_behind(tmp_path):
    """The smoke's completions all run out of their eight tokens, so a run
    that masked truncations would commit nothing and never push. The
    example scores them: both updates commit, and the second trains on
    draws submitted one update earlier."""
    commits_both_updates_one_behind("train_rlvr", tmp_path, "rewards.json")


def test_train_rlvr_turns_smoke_trains_through_environment_source(tmp_path):
    """`--turns 2` runs its sessions through EnvironmentSource and commits both updates, one behind.
    The tiny model's attempts end on the token budget, so the feedback path
    is the next test's."""
    commits_both_updates_one_behind("train_rlvr", tmp_path, "rewards.json", "--turns", "2")


def test_train_harbor_smoke_trains_on_gateway_recorded_harness_calls(tmp_path):
    """Two updates of Harbor trials whose model calls the stand-in gateway records from the policy's
    own draws; the second trains on sessions submitted one update earlier."""
    commits_both_updates_one_behind("train_harbor", tmp_path, "summary.json")


def test_evaluate_and_serve_smoke_reports_perplexity_and_a_greedy_continuation(tmp_path):
    """The evaluation report of a run the script trains first: the perplexity
    `Evaluation.run` scores over the held-out split, a greedy continuation, and a
    served comparison that neither SDK can reach. An installed SDK reports
    the endpoint unreachable, an absent one is skipped, and neither needs an
    OpenAI key, which the smoke's environment does not carry."""
    smoke("evaluate_and_serve", tmp_path,
          "--openai-base-url", "http://127.0.0.1:1/v1", "--ollama-host", "http://127.0.0.1:1")

    report = json.loads((tmp_path / "report.json").read_text())
    assert report["perplexity"]["val/perplexity"] > 0
    assert len(report["greedy"]) == 8
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


@pytest.mark.parametrize("arguments", [(), ("--damping", "1e-8")], ids=["adam", "gauss-newton"])
def test_pinn_burgers_smoke_trains_the_profile_in_float64_and_reports_its_residuals(tmp_path, arguments):
    """The self-similar Burgers profile through a four-level Fourier feature pyramid, by Adam and by
    Gauss-Newton, in float64: the run writes the paper's two figures on its diagnostic grid, finite."""
    out = tmp_path / "burgers"
    smoke("pinn_burgers", out, *arguments)
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["steps"] == 20 and np.isfinite(metrics["pde_mse"])
    assert np.isfinite(metrics["log10_max_residual"])


BURGERS_PIECES = """
import importlib.util, json, sys
import jax, jax.numpy as jnp, numpy as np

spec = importlib.util.spec_from_file_location("pinn_burgers_under_test", sys.argv[1])
burgers = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = burgers
spec.loader.exec_module(burgers)
model = burgers.Profile(pyramid=0, width=5, depth=3)
profile = burgers.BurgersProfile(model)
variables = profile.fresh_variables(jax.random.key(0), None)
value, slope = jax.value_and_grad(lambda at: profile.equation(variables, at))(jnp.float64(0.0))
drawn = [model.init(jax.random.key(seed), jnp.zeros((1,), jnp.float64))["params"] for seed in range(200)]
largest = {layer: float(max(np.abs(np.asarray(params[layer]["bias"])).max() for params in drawn))
           for layer in ("Dense_0", "Dense_2")}
print(json.dumps({"zero": [float(value), float(slope)], "largest": largest}))
"""
"""The Burgers example's residual and its slope at eta = 0, and its layers' largest bias of 200 draws."""


def test_the_burgers_residual_is_finite_where_its_points_start_and_its_biases_are_bounded_by_their_inputs(
        tmp_path):
    """The profile equation and its derivative are finite at eta = 0, inside the points' [0, c), where
    the tail's factored form gave a negative power of zero times zero (row 1993). Each layer's bias is
    drawn as PyTorch's Linear draws it, within 1 / sqrt of its inputs: the first layer's one input lets
    a bias past 1 / sqrt(5), and the last layer's five inputs keep one within it, where the bias was
    bounded by its output width (row 1994)."""
    script = tmp_path / "burgers_pieces.py"
    script.write_text(BURGERS_PIECES)
    finished = smoke("pinn_burgers", tmp_path, str(REPO_ROOT / "examples" / "pinn_burgers.py"), script=script,
                     smoke_args=False)
    pieces = json.loads(finished.stdout.strip().splitlines()[-1])
    assert np.isfinite(pieces["zero"]).all()
    bound = 1 / math.sqrt(5)
    assert pieces["largest"]["Dense_0"] > bound >= pieces["largest"]["Dense_2"]


@pytest.mark.parametrize("method", ["perturbation", "backprop"])
def test_forward_only_lm_smoke_trains_a_decoder_with_and_without_backpropagation(tmp_path, method):
    """A two-layer decoder of Dust's layout on random rows, by node perturbation and by backpropagation:
    each run writes its training and validation losses, finite, for every step and every evaluation."""
    out = tmp_path / method
    smoke("forward_only_lm", out, "--method", method)
    curves = json.loads((out / "curves.json").read_text())
    assert curves["method"] == method and curves["steps"] == 4
    assert [step for step, _ in curves["train"]] == [1, 2, 3, 4] and curves["val"]
    assert np.isfinite([loss for _, loss in curves["train"] + curves["val"]]).all()
