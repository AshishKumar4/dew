"""Score a finished run four ways, then compare it against a served model.

Perplexity over held-out tokens, an lm-evaluation-harness suite, greedy
continuations, and — for a diffusion run — FID and CLIPScore of a sampled
grid against a reference set. Everything is read through `dew.pipeline`, so
a run directory, a published checkpoint and a Hub repository all work.

    python examples/evaluate_and_serve.py --run runs/shakespeare/lm-shakespeare \\
        --tokens data/shakespeare --tasks hellaswag arc_easy --harness-limit 200
    python examples/evaluate_and_serve.py --run runs/shakespeare/lm-shakespeare \\
        --image-run runs/flowers-tpu/checkpoints/flowers-256 \\
        --reference-images data/flowers-heldout

The same suites run from the command line, which is the harness's own entry
point with Dew's model registered:

    python -m dew.eval --model dew --model_args run=runs/shakespeare/lm-shakespeare \\
        --tasks hellaswag --limit 200

`--openai-base-url` adds a served comparison: the same prompts through the
OpenAI SDK (a vLLM endpoint speaks it too), or `--ollama-host` through
ollama's. Both are optional extras; without them the report says so and the
rest of the run is unaffected.

    JAX_PLATFORMS=cpu python examples/evaluate_and_serve.py --smoke --out /tmp/eval-smoke
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import jax
import numpy as np
import tyro

import dew
from dew.config import ModelConfig, OptimConfig, TrainerConfig
from dew.data import Loading, TokenWindows
from dew.eval import clip_score, fid
from dew.inference import TextGeneration, TextToImage
from dew.objectives.lm import LMObjective, LMRunConfig
from dew.registry import metrics
from dew.sampling import Sampling
from dew.training import evaluate

GREEDY = Sampling(temperature=0.0)
FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures"
# An InceptionV3 at a sixteenth of every channel width with drawn
# parameters: the FID path runs offline on it, and the value is its own.
SMOKE_INCEPTION = FIXTURES / "inception/tiny/inception_v3_fid.safetensors"


@dataclass
class Config:
    run: Path | None = None
    """The run directory, published checkpoint or Hub repo to score."""
    tokens: Path | None = None
    """Token directory from tools/tokenize_text.py; its val split is the perplexity set."""
    out: Path = Path("reports/evaluation")
    sequence_length: int = 256
    batch_size: int = 8
    prompt: str = "ROMEO:"
    max_new_tokens: int = 64
    tasks: tuple[str, ...] = ()
    """lm-eval-harness task names; empty runs no suite."""
    harness_limit: int = 64
    """Documents per task, the harness's own --limit."""
    image_run: Path | None = None
    """A diffusion run to sample and score; unset skips the image metrics."""
    reference_images: Path | None = None
    """Directory of PNGs FID is measured against; --smoke draws its own."""
    inception_weights: Path | None = None
    """The FID extractor's parameters as a file; unset downloads the
    published checkpoint. --smoke reads the committed tiny one."""
    image_prompts: tuple[str, ...] = ("a water lily", "a sunflower", "a red rose", "a purple orchid")
    image_steps: int = 40
    clip_model: str = "openai/clip-vit-large-patch14"
    openai_base_url: str | None = None
    """An OpenAI-compatible endpoint (vLLM included) to compare against."""
    openai_model: str = "gpt-4o-mini"
    openai_provider: Literal["openai", "vllm"] = "openai"
    """vLLM takes top-k, min-p and stop ids that the OpenAI API has no field for."""
    ollama_host: str | None = None
    ollama_model: str = "llama3.2"
    smoke: bool = False
    """Train a tiny byte-level run here first and score that instead."""


def smoke_run(out: Path) -> tuple[Path, Path]:
    """A two-step byte-level LM run and the token files it read.

    The same shape tests/test_inference.py's `make_lm_run` builds: a tiny
    causal decoder, its checkpoint and the `run.json` that names the model,
    the tokenizer and the preview budget.
    """
    tokens = out / "tokens"
    tokens.mkdir(parents=True, exist_ok=True)
    text = ("to be or not to be, that is the question. " * 200).encode()
    (tokens / "train.bin").write_bytes(text[:6000])
    (tokens / "val.bin").write_bytes(text[6000:])
    (tokens / "meta.json").write_text(json.dumps(
        {"tokenizer": "byte", "vocab_size": 256, "dtype": "uint8"}))

    fields = {"emb_features": 16, "num_layers": 1, "num_heads": 2, "head_dim": 8,
              "mlp_features": 32, "vocab_size": 256, "max_seq_len": 48}
    run = LMRunConfig(
        model=ModelConfig("causal_transformer", fields, dtype="float32",
                          attention_impl="reference"),
        data=TokenWindows(path=str(tokens), seq_len=32,
                          loading=Loading(workers=0, threads=1, read_buffer=2,
                                          worker_buffer=1)),
        sample_tokens=8,
        optim=OptimConfig(learning_rate=1e-3),
        trainer=TrainerConfig(checkpoint_dir=str(out / "checkpoints"), batch_size=4, steps=2,
                              log_every=1, eval_every=None, checkpoint_every=2,
                              multi_host=False, compilation_cache_dir=None))
    objective = LMObjective(run.model.build(), run.data.seq_len, ema_decay=0.9)
    run.train(objective, run.data.load(batch=run.trainer.batch_size), name="smoke")
    return Path(run.trainer.checkpoint_dir) / "smoke", tokens


def perplexity(task: TextGeneration, config: Config) -> dict[str, float]:
    """The run's loss over a held-out split, through the evaluation contract.

    `evaluate` is what the trainer calls at a validation step, minus the
    optimizer and the tracker: the same objective, the same metric, one
    finite pass, and scalars every rank agrees on.
    """
    if config.tokens is None:
        return {}
    data = TokenWindows(path=str(config.tokens), seq_len=config.sequence_length,
                        loading=Loading(workers=0)).load(batch=config.batch_size)
    if data.val is None:
        raise ValueError(f"{config.tokens} holds no val split to score")
    scored = evaluate(LMObjective(task.model, config.sequence_length, ema_decay=None),
                      task.variables, data.val, key=jax.random.key(0),
                      metrics=[metrics.perplexity()])
    return dict(scored.scores)


def harness(task: TextGeneration, config: Config) -> dict[str, float]:
    """One lm-evaluation-harness suite over the same weights.

    `DewLM` is the adapter the `dew` model name resolves to, so this is what
    `python -m dew.eval --model dew` runs, with the suite chosen in Python.
    """
    if not config.tasks:
        return {}
    import lm_eval

    from dew.eval.harness import DewLM

    results = lm_eval.simple_evaluate(model=DewLM(task, batch_size=config.batch_size),
                                      tasks=list(config.tasks), limit=config.harness_limit)
    return {f"{name}/{metric}": value
            for name, scores in results["results"].items()
            for metric, value in scores.items() if isinstance(value, float)}


def draw(pipe: TextToImage, config: Config, *, seed: int) -> np.ndarray:
    """The prompts sampled once, as the uint8 images both metrics read."""
    drawn = pipe(list(config.image_prompts), steps=config.image_steps, seed=seed).host().images
    return np.clip(np.rint((drawn + 1.0) * 127.5), 0, 255).astype(np.uint8)


def image_metrics(config: Config) -> dict[str, float]:
    """FID and CLIPScore of a diffusion run's samples against a reference set."""
    if config.image_run is None:
        return {}
    pipe = dew.pipeline(str(config.image_run))
    if not isinstance(pipe, TextToImage):
        raise TypeError(f"--image-run holds a {type(pipe).__name__}, not a diffusion run")
    generated = draw(pipe, config, seed=0)
    scores = {"clip_score": clip_score(generated, list(config.image_prompts),
                                       modelname=config.clip_model)}
    if config.reference_images is not None:
        from PIL import Image

        reference = np.stack([np.asarray(Image.open(path).convert("RGB"))
                              for path in sorted(config.reference_images.glob("*.png"))])
    elif config.smoke:
        # A held-out set is what FID is measured against, and a smoke has
        # none: a second draw of the same run is a population to measure, so
        # the metric runs end to end on a number that says nothing.
        reference = draw(pipe, config, seed=1)
    else:
        return scores
    scores["fid"] = fid(generated, reference, weights=config.inception_weights)
    return scores


def served(config: Config) -> dict[str, str]:
    """The same prompt through a served model, when one is configured.

    Both adapters bind an SDK client the caller owns, and both SDKs are
    optional extras; an absent one is reported rather than raised, because
    the local numbers above do not depend on it.
    """
    answers: dict[str, str] = {}
    if config.openai_base_url is not None:
        try:
            from openai import OpenAI
        except ImportError:
            answers["openai"] = "skipped: pip install openai"
        else:
            from dew.inference import OpenAICompletion

            client = OpenAICompletion(config.openai_model, OpenAI(base_url=config.openai_base_url),
                                      provider=config.openai_provider)
            answers["openai"] = client(config.prompt, config.max_new_tokens,
                                       sampling=GREEDY).texts[0]
    if config.ollama_host is not None:
        try:
            from ollama import Client
        except ImportError:
            answers["ollama"] = "skipped: pip install ollama"
        else:
            from dew.inference import OllamaCompletion

            client = OllamaCompletion(config.ollama_model, Client(host=config.ollama_host))
            answers["ollama"] = client(config.prompt, config.max_new_tokens,
                                       sampling=GREEDY).texts[0]
    return answers


def main(config: Config) -> Path:
    if config.smoke:
        config.out.mkdir(parents=True, exist_ok=True)
        run, tokens = smoke_run(config.out)
        config = replace(config, run=run, tokens=tokens, sequence_length=32, batch_size=2,
                         prompt="to be", max_new_tokens=8, image_steps=2,
                         inception_weights=SMOKE_INCEPTION)
    if config.run is None:
        raise ValueError("--run names the run directory, checkpoint or Hub repo to score")

    task = dew.pipeline(str(config.run))
    if not isinstance(task, TextGeneration):
        raise TypeError(f"--run holds a {type(task).__name__}; --image-run takes a diffusion run")

    report = {"run": str(config.run),
              "perplexity": perplexity(task, config),
              "harness": harness(task, config),
              "images": image_metrics(config),
              "served": served(config),
              "greedy": task(config.prompt, config.max_new_tokens, sampling=GREEDY, seed=0).text[0]}
    config.out.mkdir(parents=True, exist_ok=True)
    path = config.out / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return path


if __name__ == "__main__":
    main(tyro.cli(Config))
