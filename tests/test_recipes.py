"""The shipped recipes' documented commands (their docstrings' `python recipes/...` lines, `RUN`), run
as written at a tiny size on CPU: their flags as documented, with only the model's sizes, the run's
length, the data and the pretrained source swapped for small local ones (`TINY`). A recipe whose
defaults refuse its own command is red: the decision recipe's `--trainer.steps 2000` was refused at its
default cadence ("epoch"), which no test of the recipes ran."""

import ast
import shlex
from collections.abc import Callable
from pathlib import Path

import pytest
from objective_data import IMAGES
from run_kinds import QWEN_TINY, TINY_LAYA, pictures
from tool_support import REPO_ROOT, load

from dew.data import TFDS, Memory
from dew.data.text import export_tokens


def documented(recipe: str) -> list[list[str]]:
    """The commands `recipes/<recipe>.py`'s docstring documents for it, each its arguments as a shell reads
    them."""
    path = REPO_ROOT / "recipes" / f"{recipe}.py"
    written = ast.get_docstring(ast.parse(path.read_text())) or ""
    script = f"python recipes/{recipe}.py"
    return [shlex.split(line.strip())[2:] for line in written.replace("\\\n", " ").splitlines()
            if line.strip().startswith(script)]


def scaled(args: list[str], sizes: dict[str, str]) -> list[str]:
    """`args` with each flag `sizes` names (`--model.num-layers`, or a `--set` path such as
    `data.images.size`) given its value there, every other argument as written."""
    out, at = [], 0
    while at < len(args):
        flag = args[at]
        if flag == "--set" and args[at + 1].split("=", 1)[0] in sizes:
            path = args[at + 1].split("=", 1)[0]
            out += ["--set", f"{path}={sizes[path]}"]
            at += 2
        elif flag in sizes:
            out += [flag, sizes[flag]]
            at += 2
        else:
            out.append(flag)
            at += 1
    return out


def tickets():
    """A table of support tickets as BANKING77's rows are, each a `text` and its `label_text`, a tenth
    held out."""
    asked = [("charged twice for March", "billing"), ("the site is down", "technical"),
             ("how much is the pro plan", "sales"), ("refund my last invoice", "billing")]
    return Memory([{"text": f"{text}, ticket {index}", "label_text": label}
                   for index in range(40) for text, label in asked]).holdout(0.1)


def byte_store(root: Path, name: str) -> str:
    """A small store of byte tokens at `root/name`, as `dew data export` writes one."""
    documents = [{"text": f"document {index}: " + "to be or not to be " * (1 + index % 3)}
                 for index in range(64)]
    export_tokens(Memory(documents), str(root / name), "byte")
    return str(root / name)


TRAINED = ["--trainer.multi-host", "False", "--trainer.compilation-cache-dir", "None"]
"""What every run here adds: one process, no shared compilation cache."""

def decision(args: list[str], root: Path) -> Callable[[], object]:
    from dew.decision.config import DecisionRunConfig

    recipe = load("train", "recipes/decision")
    pretrained = str(QWEN_TINY if "--lora.rank" in args else TINY_LAYA)
    sized = scaled(args, {"--pretrained": pretrained, "--trainer.epochs": "1", "--trainer.steps": "2",
                          "--lora.rank": "2"})
    sized = sized if "--pretrained" in sized else ["--pretrained", pretrained, *sized]
    config = DecisionRunConfig.cli([*sized, *TRAINED, "--trainer.checkpoint-dir", str(root / "runs")],
                                   default=recipe.default(*tickets()))
    return config.run


IMAGE_MODEL = {"--model.patch-size": "2", "--model.emb-features": "16", "--model.num-layers": "1",
               "--model.num-heads": "2"}


def diffusion(args: list[str], root: Path) -> Callable[[], object]:
    from dew.config.diffusion import DiffusionRunConfig

    recipe = load("train", "recipes/diffusion")
    sized = scaled(args, {"data.images.size": "8", "data.images.batch": "8", "--trainer.epochs": "1",
                          **IMAGE_MODEL})
    config = DiffusionRunConfig.cli([*sized, *TRAINED, "--trainer.checkpoint-dir", str(root / "runs"),
                                     "--set", "validation.images.size=8",
                                     "--set", "validation.images.batch=8",
                                     # Its default scores read a CLIP model from the Hub.
                                     "--val-metrics", "[]",
                                     "--text.encoder", "char_table", "--text.checkpoint", "char_table"],
                                    default=recipe.default(*pictures()))
    return config.run


def jepa(args: list[str], root: Path) -> Callable[[], object]:
    from dew.config.jepa import JepaRunConfig

    recipe = load("train", "recipes/jepa")
    sized = scaled(args, {"data.images.size": "16", "data.images.batch": "8", "--trainer.epochs": "1",
                          "--probe-classes": "2", **IMAGE_MODEL})
    images = TFDS("dew_images", path=str(IMAGES), split="train")
    config = JepaRunConfig.cli([*sized, *TRAINED, "--trainer.checkpoint-dir", str(root / "runs"),
                                "--set", "validation.images.size=16", "--set", "validation.images.batch=8"],
                               default=recipe.default(*images.holdout(split="test")))
    return config.run


def lm(args: list[str], root: Path) -> Callable[[], object]:
    from dew.config.lm import LMRunConfig

    recipe = load("train", "recipes/lm")
    sized = scaled(args, {"data.source.url": byte_store(root, "shakespeare-byte"),
                          "data.token_windows.seq_len": "16", "data.token_windows.batch": "8",
                          "--trainer.epochs": "1",
                          "--model.emb-features": "16", "--model.num-layers": "1", "--model.num-heads": "2"})
    config = LMRunConfig.cli([*sized, *TRAINED, "--trainer.checkpoint-dir", str(root / "runs"),
                              "--max-new-tokens", "0"], default=recipe.default())
    return config.run


def rigel(args: list[str], root: Path) -> Callable[[], object]:
    recipe = load("rigel", "recipes/lm")
    at = args.index("--corpora") + 1
    named = args[at:at + next((index for index, word in enumerate(args[at:]) if word.startswith("--")),
                              len(args) - at)]
    stores = {directory: byte_store(root, name)
              for name, directory in zip(named[::2], named[1::2], strict=True)}
    # Eight steps, the fewest whose rescaled phases each end past the one before.
    sizes = {"--width": "128", "--steps": "8", "--batch-size": "8", "--seq-len": "16"}
    sized = [stores.get(word, word) for word in scaled(args, sizes)]
    rigel_flags = [*sized, *(["--width", "128"] if "--width" not in sized else []),
                   *(["--steps", "8"] if "--steps" not in sized else []), "--tokenizer", "byte"]
    sized_run = ["--model.num-layers", "4", "--model.layer-types", "mamba", "mamba", "mamba", "attention",
                 "--max-new-tokens", "0", *TRAINED, "--trainer.checkpoint-dir", str(root / "runs")]
    return lambda: recipe.main([*scaled(rigel_flags, {"--trainer.checkpoint-dir": str(root / "runs")}),
                                *sized_run])


def mixture(args: list[str], root: Path) -> Callable[[], object]:
    import tyro

    recipe = load("mixture", "recipes/lm")
    return lambda: recipe.main(tyro.cli(recipe.Args, args=scaled(args, {"--show": "8"})))


TINY = {"decision/train": decision, "diffusion/train": diffusion, "jepa/train": jepa, "lm/train": lm,
        "lm/rigel": rigel, "lm/mixture": mixture}
"""How each recipe's documented commands run here: their documented flags at a tiny size, over local data."""

RUN = (("decision/train", 0), ("decision/train", 1), ("diffusion/train", 0), ("jepa/train", 0),
       ("lm/train", 0), ("lm/rigel", 1), ("lm/mixture", 0))
"""The documented commands this file runs, each by its recipe and its place among the recipe's commands: those
that run as documented; a fix that makes another run adds it."""

NETWORK = {"lm/mixture"}
"""The recipes whose commands read the Hub, as their documented runs do: the mixture's sources and
tokenizer."""

COMMANDS = [pytest.param(recipe, documented(recipe)[index], id=f"{recipe}-{index}",
                         marks=[pytest.mark.network] if recipe in NETWORK else [])
            for recipe, index in RUN]


@pytest.mark.mesh
@pytest.mark.parametrize(("recipe", "args"), COMMANDS)
def test_a_recipes_documented_command_runs_at_a_tiny_size(recipe, args, tmp_path):
    """The command runs to its end at a tiny size: its run trains, or the mixture prints its counts."""
    TINY[recipe](args, tmp_path)()
