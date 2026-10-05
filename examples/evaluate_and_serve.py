"""Score a finished run four ways, then compare it against a served model.

The script measures held-out perplexity, runs an lm-evaluation-harness suite
and generates greedy continuations. For diffusion, it computes FID and
CLIPScore on a sampled grid, using a reference set for FID. `dew.pipeline`
loads a run directory, published checkpoint or Hub repository.

    python examples/evaluate_and_serve.py --run runs/shakespeare/lm-shakespeare \\
        --tokens data/shakespeare --tasks hellaswag arc_easy --harness-limit 200
    python examples/evaluate_and_serve.py --run runs/shakespeare/lm-shakespeare \\
        --image-run runs/flowers-tpu/checkpoints/flowers-256 \\
        --reference-images data/flowers-heldout

To run these suites from the harness's command-line entry point, use Dew's
registered model:

    python -m dew.eval --model dew --model_args run=runs/shakespeare/lm-shakespeare \\
        --tasks hellaswag --limit 200

To compare server output for the same prompts, set `--openai-base-url` for
the OpenAI SDK (including vLLM endpoints), or `--ollama-host` for ollama's SDK.
Both SDKs are optional extras. Missing SDKs or unreachable endpoints are
reported without interrupting local evaluation. The OpenAI client uses
`OPENAI_API_KEY` when set. Otherwise it sends vLLM's placeholder key, which
a vLLM server accepts when started without `--api-key`.

    JAX_PLATFORMS=cpu python examples/evaluate_and_serve.py --smoke --out /tmp/eval-smoke
"""

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import tyro

import dew
from dew.artifacts import uint8_pixels
from dew.config import ModelConfig, OptimConfig, TrainerConfig
from dew.data import ByteTokenizer, Loading, TokenWindows
from dew.eval import FID, CLIPScore
from dew.inference import RunProcessor, TextGeneration, TextToImage
from dew.nn.backbones import CausalTransformer
from dew.objectives.lm import LMObjective, LMRunConfig, Perplexity
from dew.sampling import Sampling
from dew.training import Evaluation

GREEDY = Sampling(temperature=0.0)
FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures"
# This InceptionV3 has one-sixteenth of each channel width and untrained
# parameters. It exercises FID offline; its scores are specific to this fixture.
SMOKE_INCEPTION = FIXTURES / "inception/tiny/inception_v3_fid.safetensors"


@dataclass
class Config:
    run: Path | None = None
    """The run directory, published checkpoint or Hub repo to score."""
    tokens: Path | None = None
    """Token directory from `dew tokenize`; perplexity uses its val split."""
    out: Path = Path("reports/evaluation")
    sequence_length: int = 256
    batch_size: int = 8
    prompt: str = "ROMEO:"
    max_new_tokens: int = 64
    tasks: tuple[str, ...] = ()
    """lm-eval-harness task names. An empty tuple skips the suite."""
    harness_limit: int = 64
    """Documents per task, passed as the harness's --limit."""
    image_run: Path | None = None
    """A diffusion run to sample and score; unset skips the image metrics."""
    reference_images: Path | None = None
    """Reference PNG directory for FID. --smoke generates its own reference."""
    inception_weights: str | None = None
    """FID extractor weights file. If unset, download the published checkpoint.
    --smoke uses the committed tiny checkpoint."""
    image_prompts: tuple[str, ...] = ("a water lily", "a sunflower", "a red rose", "a purple orchid")
    image_steps: int = 40
    clip_model: str = "openai/clip-vit-large-patch14"
    openai_base_url: str | None = None
    """An OpenAI-compatible endpoint (vLLM included) to compare against."""
    openai_model: str = "gpt-4o-mini"
    openai_provider: Literal["openai", "vllm"] = "openai"
    """vLLM accepts top-k, min-p and stop IDs absent from the OpenAI API."""
    ollama_host: str | None = None
    ollama_model: str = "llama3.2"
    smoke: bool = False
    """Train and score a tiny byte-level run locally."""


def smoke_run(out: Path) -> tuple[Path, Path]:
    """Train a two-step byte-level LM and return its run and token directories.

    This uses the same shape as tests/test_inference.py's `make_lm_run`.
    The tiny causal decoder's checkpoint record includes the model, the
    objective's byte tokenizer and the preview budget.
    """
    tokens = out / "tokens"
    tokens.mkdir(parents=True, exist_ok=True)
    text = ("to be or not to be, that is the question. " * 200).encode()
    (tokens / "train.bin").write_bytes(text[:6000])
    (tokens / "val.bin").write_bytes(text[6000:])
    (tokens / "meta.json").write_text(json.dumps(
        {"tokenizer": "byte", "vocab_size": 256, "dtype": "uint8"}))

    model = CausalTransformer(emb_features=16, num_layers=1, num_heads=2, head_dim=8, mlp_features=32,
                              vocab_size=256, max_seq_len=48, dtype=jnp.float32,
                              attention_impl="reference")
    run = LMRunConfig(
        model=ModelConfig.from_model(model),
        data=TokenWindows(path=str(tokens), seq_len=32,
                          loading=Loading(workers=0, threads=1, read_buffer=2,
                                          worker_buffer=1)),
        sample_tokens=8,
        optim=OptimConfig(learning_rate=1e-3),
        trainer=TrainerConfig(checkpoint_dir=str(out / "checkpoints"), batch_size=4, steps=2,
                              log_every=1, eval_every=None, checkpoint_every=2,
                              multi_host=False, compilation_cache_dir=None))
    objective = LMObjective(model, run.data.seq_len, ema_decay=0.9,
                            processor=RunProcessor(ByteTokenizer()))
    run.train(objective, run.data.load(batch=run.trainer.batch_size), name="smoke")
    return Path(run.trainer.checkpoint_dir) / "smoke", tokens


def perplexity(task: TextGeneration, config: Config) -> dict[str, float]:
    """Score the run's held-out split with the trainer's evaluation API.

    `Evaluation.run` uses the same objective and metric as trainer validation.
    It makes one finite pass and returns scalars agreed by every rank,
    without an optimizer or tracker.
    """
    if config.tokens is None:
        return {}
    data = TokenWindows(path=str(config.tokens), seq_len=config.sequence_length,
                        loading=Loading(workers=0)).load(batch=config.batch_size)
    if data.val is None:
        raise ValueError(f"{config.tokens} holds no val split to score")
    scored = Evaluation.run(LMObjective(task.model, config.sequence_length),
                      task.variables, data.val, key=jax.random.key(0),
                      metrics=[Perplexity()])
    return dict(scored.scores)


def harness(task: TextGeneration, config: Config) -> dict[str, float]:
    """One lm-evaluation-harness suite over the same weights.

    The `dew` model name resolves to the `DewLM` adapter. This runs the same
    evaluation as `python -m dew.eval --model dew`, with tasks chosen in Python.
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


def draw(pipe: TextToImage, config: Config, *, key: int) -> np.ndarray:
    """Sample the prompts once and return uint8 images for both metrics."""
    drawn = pipe(list(config.image_prompts), steps=config.image_steps, key=key).host().images
    return uint8_pixels(drawn)


def image_metrics(config: Config) -> dict[str, float]:
    """FID and CLIPScore of a diffusion run's samples against a reference set."""
    if config.image_run is None:
        return {}
    pipe = dew.pipeline(str(config.image_run))
    if not isinstance(pipe, TextToImage):
        raise TypeError(f"--image-run holds a {type(pipe).__name__}, not a diffusion run")
    generated = draw(pipe, config, key=0)
    scores = {"clip_score": CLIPScore(config.clip_model).score(generated, list(config.image_prompts))}
    if config.reference_images is not None:
        from PIL import Image

        reference = np.stack([np.asarray(Image.open(path).convert("RGB"))
                              for path in sorted(config.reference_images.glob("*.png"))])
    elif config.smoke:
        # With no held-out set, smoke mode uses a second draw as the FID reference.
        # This exercises the metric end to end but cannot measure model quality.
        reference = draw(pipe, config, key=1)
    else:
        return scores
    scores["fid"] = FID(weights=config.inception_weights).score(generated, reference)
    return scores


def served(config: Config) -> dict[str, str]:
    """Generate from a configured server using the local evaluation's prompt.

    Both adapters use caller-owned SDK clients from optional extras.
    Missing SDKs and unreachable endpoints are reported without raising,
    since local evaluation does not depend on the server.
    """
    answers: dict[str, str] = {}
    if config.openai_base_url is not None:
        try:
            from openai import APIConnectionError, OpenAI
        except ImportError:
            answers["openai"] = "skipped: pip install openai"
        else:
            from dew.inference import OpenAICompletion

            # The SDK requires a key even when a local vLLM endpoint does not.
            # "EMPTY" is the placeholder used in vLLM's OpenAI-client examples.
            sdk = OpenAI(base_url=config.openai_base_url,
                         api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
            client = OpenAICompletion(config.openai_model, sdk, provider=config.openai_provider)
            try:
                answers["openai"] = client(config.prompt, config.max_new_tokens,
                                           sampling=GREEDY).texts[0]
            except APIConnectionError as error:
                answers["openai"] = f"unreachable: {config.openai_base_url}: {error}"
    if config.ollama_host is not None:
        try:
            from ollama import Client
        except ImportError:
            answers["ollama"] = "skipped: pip install ollama"
        else:
            from dew.inference import OllamaCompletion

            client = OllamaCompletion(config.ollama_model, Client(host=config.ollama_host))
            try:
                answers["ollama"] = client(config.prompt, config.max_new_tokens,
                                           sampling=GREEDY).texts[0]
            except ConnectionError as error:
                answers["ollama"] = f"unreachable: {config.ollama_host}: {error}"
    return answers


def main(config: Config) -> Path:
    if config.smoke:
        config.out.mkdir(parents=True, exist_ok=True)
        run, tokens = smoke_run(config.out)
        config = replace(config, run=run, tokens=tokens, sequence_length=32, batch_size=2,
                         prompt="to be", max_new_tokens=8, image_steps=2,
                         inception_weights=str(SMOKE_INCEPTION))
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
              "greedy": task(config.prompt, config.max_new_tokens, sampling=GREEDY, key=0).text[0]}
    config.out.mkdir(parents=True, exist_ok=True)
    path = config.out / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return path


if __name__ == "__main__":
    main(tyro.cli(Config))
