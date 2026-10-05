"""A loaded source trained with every training feature at once, killed mid-accumulation and resumed.

The other recovery tests each hold one feature: CUDA bitwise resume
(test_trainer), partial accumulation (test_training_transactions), a real
SIGKILL (test_multiprocess), an export transformers reads
(test_decoder_export). This one runs them together, as a fine-tune does. A
tiny transformers Llama, loaded through `Pretrained.load`, trains with
dropout, EMA, two-micro-step accumulation, dynamic loss scaling and a clipped
AdamW (tests/qualification_worker.py). One run goes uninterrupted. A second
is SIGKILLed after step 7, holding the step-5 checkpoint, which sits between
the two micro-steps of an update. A third resumes from it. The resumed run
must end on every state leaf of the uninterrupted one, bit for bit. The
baseline's fp32 loss and source-tensor gradients must match transformers', and
each run's export must give its trained logits when transformers reads it.
The runs use XLA attention, the path the README's CUDA bitwise claim names.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import qualification_worker as worker
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM

from dew.data import TokenCorpus

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "tests" / "fixtures" / "tokenizers" / "tiny-tools"


def spawn(directory: Path, mode: str) -> subprocess.Popen:
    """One run of the worker in its own session, so a kill takes its children
    too. It inherits the session's platform and the lane's flags
    (tests/lane_environment.py), which on CUDA make the reductions repeatable."""
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    return subprocess.Popen([sys.executable, str(Path(worker.__file__)), str(directory), mode],
                            cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True)


def finish(process: subprocess.Popen, mode: str) -> None:
    try:
        log = process.communicate(timeout=900)[0]
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        pytest.fail(f"the {mode} run did not finish within 900s\n{process.communicate()[0]}")
    assert process.returncode == 0, f"the {mode} run exited {process.returncode}\n{log}"


def kill_when_ready(process: subprocess.Popen, marker: Path) -> None:
    """SIGKILL the interrupted run once it is blocked past its checkpoint."""
    deadline = time.monotonic() + 900
    while not marker.exists():
        if process.poll() is not None or time.monotonic() >= deadline:
            os.killpg(process.pid, signal.SIGKILL)
            pytest.fail(f"the interrupted run never blocked\n{process.communicate()[0]}")
        time.sleep(0.05)
    assert int(marker.read_text()) == process.pid
    os.killpg(process.pid, signal.SIGKILL)
    assert process.wait(timeout=60) == -signal.SIGKILL, "the run was not killed"


def source_gradients(directory: Path) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
    """transformers' fp32 loss and per-tensor gradient on the baseline's probe batch."""
    reference = AutoModelForCausalLM.from_pretrained(directory / "source", dtype=torch.float32,
                                                     attn_implementation="eager",
                                                     local_files_only=True).eval()
    with np.load(directory / "baseline" / "reference.npz") as recorded:
        ids = torch.from_numpy(recorded["ids"]).long()
    logits = reference(ids).logits[:, :-1]
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
    loss.backward()
    return loss.detach(), {name: parameter.grad.numpy() for name, parameter in reference.named_parameters()}


@pytest.fixture(scope="module")
def qualified(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("qualification")
    (directory / "corpus.txt").write_bytes((ROOT / "CONTRIBUTING.md").read_bytes())
    TokenCorpus.write(directory / "corpus.txt", directory / "tokens", tokenizer=str(TOKENIZER),
                      val_fraction=0.1)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    torch.manual_seed(13)
    LlamaForCausalLM(LlamaConfig(
        vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=worker.SEQUENCE,
        tie_word_embeddings=False)).save_pretrained(directory / "source")
    tokenizer.save_pretrained(directory / "source")
    finish(spawn(directory, "baseline"), "baseline")
    kill_when_ready(spawn(directory, "interrupted"), directory / "ready-to-kill")
    finish(spawn(directory, "resumed"), "resumed")
    return directory


@pytest.mark.slow
def test_a_run_killed_mid_accumulation_resumes_onto_every_leaf_of_the_uninterrupted_run(qualified):
    baseline, resumed = (json.loads((qualified / run / "result.json").read_text())
                         for run in ("baseline", "restarted"))
    assert (resumed["restored_step"], resumed["restored_updates"]) == (worker.CHECKPOINT_STEP,
                                                                       worker.CHECKPOINT_STEP // 2)
    for field in ("position", "state_structure", "step", "microstep", "updates"):
        assert resumed[field] == baseline[field], field
    assert baseline["final_loss"] < baseline["initial_loss"], "training did not lower the probe's loss"
    with (np.load(qualified / "baseline" / "state.npz") as left,
          np.load(qualified / "restarted" / "state.npz") as right):
        assert set(left.files) == set(right.files)
        for name in left.files:
            assert left[name].dtype == right[name].dtype, name
            np.testing.assert_array_equal(left[name], right[name], err_msg=name)


@pytest.mark.slow
def test_the_loaded_sources_loss_and_gradients_are_transformers(qualified):
    loss, gradients = source_gradients(qualified)
    with np.load(qualified / "baseline" / "reference.npz") as recorded:
        np.testing.assert_allclose(loss.numpy(), recorded["loss"], atol=1e-4, rtol=0)
        assert {"gradient/" + name for name in gradients} == set(recorded.files) - {"ids", "loss"}
        for name, wanted in gradients.items():
            gap = np.abs(recorded["gradient/" + name] - wanted).max() / max(1.0, np.abs(wanted).max())
            assert gap <= 1e-4, f"{name}: {gap}"


@pytest.mark.slow
@pytest.mark.parametrize("run", ["baseline", "restarted"])
def test_each_runs_export_reads_in_transformers_as_its_trained_logits(qualified, run):
    exported = AutoModelForCausalLM.from_pretrained(qualified / run / "export", dtype=torch.float32,
                                                    attn_implementation="eager",
                                                    local_files_only=True).eval()
    with np.load(qualified / run / "export_logits.npz") as recorded, torch.no_grad():
        actual = exported(torch.from_numpy(recorded["ids"]).long()).logits.numpy()
        np.testing.assert_allclose(actual, recorded["logits"], atol=1e-4, rtol=0)
