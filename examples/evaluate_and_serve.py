"""Score a finished run four ways, then compare it against a served model.

Perplexity over held-out tokens, an lm-evaluation-harness suite, greedy
continuations, and, for a diffusion run, CLIPScore of a sampled grid against
its prompts and FID against a reference set. Everything is read through `dew.pipeline`, so
a run directory, a published checkpoint and a Hub repository all work.

    python examples/evaluate_and_serve.py --run runs/shakespeare/checkpoints \\
        --tokens data/shakespeare-val --tasks hellaswag arc_easy --harness-limit 200
    python examples/evaluate_and_serve.py --run runs/shakespeare/checkpoints \\
        --image-run runs/flowers-tpu/checkpoints/flowers-256 \\
        --reference-images data/flowers-heldout

The same suites run from the command line, which is the harness's own entry
point with Dew's model registered:

    python -m dew.eval --model dew --model_args run=runs/shakespeare/checkpoints \\
        --tasks hellaswag --limit 200

`--openai-base-url` adds a served comparison: the same prompts through the
OpenAI SDK (vLLM serves the same API), or `--ollama-host` through
ollama's. Each SDK is an optional extra, needed only when its flag is given;
with the endpoint unreachable, the report says so and the rest of the run is
unaffected. The
OpenAI key is `OPENAI_API_KEY` when set; a vLLM server started without
`--api-key` takes any key, so an unset one sends vLLM's placeholder.

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
from dew.config import TrainerConfig
from dew.config.lm import LMRunConfig
from dew.data import ByteTokenizer, Loading, Memory, presets, text
from dew.data.sources import TokenFiles
from dew.eval import FID, CLIPScore
from dew.inference import RunProcessor, TextGeneration, TextToImage
from dew.nn.backbones import CausalTransformer
from dew.objectives.lm import LMObjective, Perplexity
from dew.optim import OptimConfig
from dew.registry import ModelConfig
from dew.sampling import Sampling
from dew.training import Evaluation

GREEDY = Sampling(temperature=0.0)
FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures"
# An InceptionV3 with every channel width cut to a sixteenth and randomly
# drawn parameters. FID runs offline on it; its values mean nothing elsewhere.
SMOKE_INCEPTION = FIXTURES / "inception/tiny/inception_v3_fid.safetensors"


@dataclass
class Config:
    run: Path | None = None
    """The run directory, published checkpoint or Hub repo to score."""
    tokens: Path | None = None
    """The held-out token store whose tokens are the perplexity set, as
    `dew data export --tokenizer` writes one (examples/train_lm.py)."""
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
    inception_weights: str | None = None
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
    """A two-step byte-level LM run and the held-out tokens it validated on.

    The same shape tests/test_inference.py's `make_lm_run` builds: a tiny
    causal decoder and its checkpoint, whose record names the model, the
    byte tokenizer the objective decodes through and the preview budget.
    """
    play = "to be or not to be, that is the question. " * 200
    # The held-out tokens as the store `--tokens` names, as `dew data export --tokenizer byte` writes one.
    held = out / "held-out"
    text.export_tokens(Memory([{"text": play[6000:]}]), str(held), "byte", eos=False)
    loading = Loading(workers=0, threads=1, read_buffer=2, worker_buffer=1)

    model = CausalTransformer(emb_features=16, num_layers=1, num_heads=2, head_dim=8, mlp_features=32,
                              vocab_size=256, max_seq_len=48, dtype=jnp.float32,
                              attention_impl="reference")
    run = LMRunConfig(
        model=ModelConfig.from_model(model),
        data=Memory([{"text": play[:6000]}]).repeat().pipe(
            presets.lm, "byte", seq_len=32, batch=4, pack=None, eos=False).loading(loading),
        validation=TokenFiles(str(held)).pipe(presets.token_windows, seq_len=32, batch=4).loading(loading),
        max_new_tokens=8,
        optim=OptimConfig(learning_rate=1e-3),
        trainer=TrainerConfig(checkpoint_dir=str(out / "checkpoints"), steps=2,
                              log_every=1, eval_every=None, checkpoint_every=2,
                              multi_host=False, compilation_cache_dir=None))
    objective = LMObjective(model, 32, ema_decay=0.9, processor=RunProcessor(ByteTokenizer()))
    run.train(objective, run.data, validation=run.validation, name="smoke")
    return Path(run.trainer.checkpoint_dir) / "smoke", held


def perplexity(task: TextGeneration, config: Config) -> dict[str, float]:
    """The run's loss over the held-out tokens, through the evaluation contract.

    `Evaluation.run` is what the trainer calls at a validation step, minus the
    optimizer and the tracker: the same objective, the same metric, one
    finite pass, and scalars every rank agrees on.
    """
    if config.tokens is None:
        return {}
    # Every window of the held-out store is scored.
    held = TokenFiles(str(config.tokens)).pipe(presets.token_windows, seq_len=config.sequence_length,
                                               batch=config.batch_size)
    scored = Evaluation.run(LMObjective(task.model, config.sequence_length),
                            task.variables, held, key=jax.random.key(0), metrics=[Perplexity()])
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


def draw(pipe: TextToImage, config: Config, *, key: int) -> np.ndarray:
    """The prompts sampled once, as the uint8 images both metrics read."""
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
        # FID needs a held-out reference set and a smoke run has none, so a
        # second draw of the same run stands in. The metric runs end to end,
        # but the number says nothing about the model.
        reference = draw(pipe, config, key=1)
    else:
        return scores
    scores["fid"] = FID(weights=config.inception_weights).score(generated, reference)
    return scores


def served(config: Config) -> dict[str, str]:
    """The same prompt through a served model, when one is configured.

    Both adapters bind an SDK client the caller owns. An endpoint that does
    not answer is reported rather than raised, because the local numbers
    above do not depend on it.
    """
    answers: dict[str, str] = {}
    if config.openai_base_url is not None:
        from openai import APIConnectionError, OpenAI

        from dew.inference import OpenAICompletion

        # The SDK refuses to build a client with no key, and a local
        # vLLM endpoint needs none: "EMPTY" is the placeholder vLLM's
        # own OpenAI-client examples send.
        sdk = OpenAI(base_url=config.openai_base_url,
                     api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
        client = OpenAICompletion(config.openai_model, sdk, provider=config.openai_provider)
        try:
            answers["openai"] = client(config.prompt, config.max_new_tokens,
                                       sampling=GREEDY).texts[0]
        except APIConnectionError as error:
            answers["openai"] = f"unreachable: {config.openai_base_url}: {error}"
    if config.ollama_host is not None:
        from ollama import Client

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
