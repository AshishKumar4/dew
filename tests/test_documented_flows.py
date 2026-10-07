"""The documented flows, each step in its own clean process.

docs/recipes.md's language-model flow (`dew tokenize`, the LM recipe, `dew
export`, then loading the export and generating) and the diffusion recipe's
(train, `TextToImage.from_run`, sample) run here one command at a time, each a
new interpreter with an environment built from nothing: no import, cache or
variable survives from one step to the next, so a step that only works after
another one imported something fails. Every step runs offline against local
fixtures, under a hard limit on open files lower than a workstation's.
"""

import os
import resource
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "tests" / "fixtures" / "tfds" / "dew_images" / "1.0.0"
FILES = (1024, 4096)
"""The soft and hard RLIMIT_NOFILE every step starts under."""


def step(directory: Path, *command: str, timeout: int = 900) -> str:
    """Run `command` with this interpreter in `directory`, in a fresh
    environment and under the lowered descriptor limit; its output on success."""
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(directory),
           "PYTHONPATH": str(ROOT / "src"), "JAX_PLATFORMS": "cpu",
           "HF_HOME": str(directory / "hf-home"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "HF_DATASETS_OFFLINE": "1", "WANDB_MODE": "disabled"}

    def lowered():
        resource.setrlimit(resource.RLIMIT_NOFILE, FILES)

    done = subprocess.run([sys.executable, *command], cwd=directory, env=env, preexec_fn=lowered,
                          capture_output=True, text=True, timeout=timeout)
    assert done.returncode == 0, (f"{' '.join(command[:3])} failed:\n"
                                  f"{done.stdout[-3000:]}\n{done.stderr[-3000:]}")
    return done.stdout


@pytest.mark.mesh(devices=0)
def test_the_language_model_flow_trains_exports_loads_and_generates(tmp_path):
    (tmp_path / "corpus.txt").write_text("the quick brown fox jumps over the lazy dog. " * 200)
    step(tmp_path, "-m", "dew.cli.main", "tokenize", "--input", "corpus.txt", "--out", "tokens",
         "--tokenizer", "byte", "--val-fraction", "0.1")
    step(tmp_path, str(ROOT / "recipes" / "lm" / "train.py"), "data:token-windows",
         "--data.path", "tokens", "--data.seq-len", "16", "--data.loading.workers", "0",
         "--data.loading.threads", "1", "--data.loading.read-buffer", "2", "--data.val-batches", "2",
         "--model.dtype", "float32", "--model.attention-impl", "reference",
         "--model.emb_features", "16", "--model.num_layers", "1",
         "--model.num_heads", "2", "--model.mlp_features", "32",
         "--trainer.batch-size", "8", "--trainer.steps", "2", "--trainer.log-every", "1",
         "--trainer.eval-every", "2", "--trainer.checkpoint-every", "2", "--trainer.checkpoint-dir", "runs",
         "--trainer.name", "byte-demo", "--trainer.multi-host", "False", "--sample-tokens", "0",
         "--sampling.temperature", "0.5", "--sampling.top-k", "7")
    step(tmp_path, "-m", "dew.cli.main", "export", "runs/byte-demo", "export")
    generated = step(tmp_path, "-c", (
        "from dew.interop import PretrainedDecoder\n"
        "source = PretrainedDecoder.load('export', dtype='float32')\n"
        "task = source.text_generation()\n"
        "print('policy', task.sampling.temperature, task.sampling.top_k)\n"
        "print('tokens', task('the ', 4, key=0).host().tokens.shape)\n"))
    # The run's own policy reached the export and was read back from it.
    assert "policy 0.5 7" in generated and "tokens (1, 8)" in generated, generated


@pytest.mark.mesh(devices=0)
def test_the_diffusion_flow_trains_restores_and_samples(tmp_path):
    step(tmp_path, str(ROOT / "recipes" / "diffusion" / "train.py"), "data:tfds-images",
         "--data.path", str(IMAGES), "--data.image-size", "8", "--data.augmentation", "none",
         "--data.val-batches", "0", "--data.loading.workers", "0", "--data.loading.threads", "1",
         "--data.loading.read-buffer", "2", "preset:flow", "--objective.solver", '{"class": "euler"}',
         "--objective.guidance", "None",
         "--text.encoder", "char_table", "--text.checkpoint", "char_table",
         "--model", "simple_dit", "--model.patch_size", "2", "--model.emb_features", "16",
         "--model.num_layers", "1", "--model.num_heads", "2",
         "--model.dtype", "float32", "--model.attention-impl", "xla", "--objective.steps", "2",
         "--objective.ema-decay", "None", "--val-metrics", "--trainer.batch-size", "4",
         "--trainer.steps", "2",
         "--trainer.log-every", "1", "--trainer.eval-every", "None", "--trainer.checkpoint-every", "2",
         "--trainer.checkpoint-dir", "runs", "--trainer.name", "demo", "--trainer.multi-host", "False")
    sampled = step(tmp_path, "-c", (
        "import numpy as np\n"
        "from dew.inference import TextToImage\n"
        "pipe = TextToImage.from_run('runs/demo')\n"
        "images = pipe(['a red flower'], key=0).host().images\n"
        "print('images', type(pipe.model).__name__, images.shape, bool(np.isfinite(images).all()))\n"))
    assert "images SimpleDiT (1, 8, 8, 3) True" in sampled, sampled
