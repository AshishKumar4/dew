"""The runs of each kind the option tests, the saved-run corpus and the GPU cells build: a language
model's, a decision's, a diffusion model's and a JEPA's, at their small sizes, each beside the run it
is set from, and the `dew train` step that trains one (`trained_by_cli`)."""

import csv
import dataclasses
import json
import typing
from pathlib import Path

import numpy as np
from objective_data import IMAGES
from test_lm_recipe import SEQ, write_token_files
from token_support import write_token_corpus

from dew import registry
from dew.cli.main import main
from dew.config import ObjectiveConfig, RunConfig, SourceOptions, TrainerConfig
from dew.config.diffusion import DiffusionRunConfig
from dew.config.jepa import JepaRunConfig
from dew.config.lm import LMRunConfig
from dew.data import TFDS, Files, Loading, images as image_steps, presets
from dew.data.pipeline import Rejected
from dew.data.sources.tokens import TokenFiles
from dew.decision import decisions
from dew.decision.config import DecisionRunConfig, Letters
from dew.decision.layout import Labels, QuestionSegment, built_in
from dew.diffusion.presets import Flow
from dew.inputs import TextCondition
from dew.registry import ModelConfig
from dew.sampling import Euler
from dew.training import Layout, MeshSpec

TINY_LAYA = Path(__file__).parent / "fixtures" / "laya" / "tiny"
QWEN_TINY = Path(__file__).parent / "fixtures" / "hf" / "qwen38-dense-tiny"
DIFFUSION_GEMMA = Path(__file__).parent / "fixtures" / "hf" / "diffusion-gemma-sft"
RATE = 0.0123
"""A learning rate no kind takes by default, which the run's optimizer logs as it steps."""
@dataclasses.dataclass(frozen=True)
class MostlyKept:
    """A record kept by its draw three times in four (`Pipeline.random_filter_map`): which records a
    batch holds is then the run's data seed's."""

    identity = "mostly-kept@1"

    def __call__(self, row, rng: np.random.Generator):
        return row if rng.random() < 0.75 else Rejected("drawn out")


@dataclasses.dataclass(frozen=True)
class Dimmed:
    """An image dimmed by a factor its record draws (`presets.images`' `augment`): what a batch's
    pixels hold is then the run's data seed's."""

    identity = "dimmed@1"

    def __call__(self, row, rng: np.random.Generator):
        image = np.asarray(row["image"])
        return {**row, "image": (image * rng.uniform(0.5, 1.0)).astype(image.dtype)}

    def declared(self, facts):
        # The image keeps its shape and dtype.
        return facts


def read(chain):
    return chain.read("memory").loading(Loading(workers=0))


def lm(root: Path, drawn: bool) -> RunConfig:
    corpus = write_token_files(root / "tokens", 40 * SEQ, 8 * SEQ)

    def windows(path, *, drawn: bool):
        chain = TokenFiles(path).pipe(presets.lm, None, seq_len=SEQ, batch=None, pack=None)
        return read((chain.random_filter_map(MostlyKept()) if drawn else chain).batch(8))
    return LMRunConfig(model=ModelConfig("causal_transformer", {"emb_features": 16, "num_layers": 1,
                                                                "num_heads": 2, "dtype": "float32"}),
                       data=windows(corpus.train, drawn=drawn), validation=windows(corpus.val, drawn=False))


def decision(root: Path, drawn: bool) -> RunConfig:
    table = root / "tickets.csv"
    asked = [("charged twice for March", "billing"), ("the site is down", "technical"),
             ("how much is the pro plan", "sales"), ("refund my last invoice", "billing")]
    # Each ticket its own text, so the held-out split, cut by a row's content, holds some of each.
    tickets = [(f"{text}, ticket {index}", label) for index in range(8) for text, label in asked]
    with table.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["text", "label"])
        writer.writerows(tickets)
    train, held = Files(str(table), format="csv").holdout(0.25)

    def asked(rows, *, drawn: bool):
        chain = rows.pipe(decisions, question="team").read("memory")
        kept = chain.random_filter_map(MostlyKept()) if drawn else chain
        return kept.batch(8).loading(Loading(workers=0))
    return DecisionRunConfig(pretrained=str(TINY_LAYA), model=ModelConfig("causal_transformer",
                                                                          {"dtype": "float32"}),
                             data=asked(train, drawn=drawn), validation=asked(held, drawn=False))


def pictures():
    return TFDS("dew_images", path=str(IMAGES), split="all").holdout(records=4)


def diffusion(root: Path, drawn: bool) -> RunConfig:
    del root
    train, held = pictures()

    def captioned(rows, *, drawn: bool):
        return read(rows.pipe(presets.images, 4, batch=8, caption=image_steps.caption(),
                              augment=Dimmed() if drawn else False))
    return DiffusionRunConfig(
        model=ModelConfig("simple_dit", {"patch_size": 2, "emb_features": 16, "num_layers": 1,
                                         "num_heads": 2}),
        data=captioned(train, drawn=drawn), validation=captioned(held, drawn=False), preset=Flow(),
        val_metrics=(),
        text=TextCondition(encoder="char_table", checkpoint="char_table"),
        objective=ObjectiveConfig("diffusion", {"solver": Euler(), "steps": 2}))


def jepa(root: Path, drawn: bool) -> RunConfig:
    del root
    train, held = pictures()

    def patched(rows, *, drawn: bool):
        return read(rows.pipe(presets.images, 16, batch=8, augment=Dimmed() if drawn else False))
    return JepaRunConfig(
        model=ModelConfig("jepa_encoder", {"patch_size": 2, "emb_features": 16, "num_layers": 1,
                                           "num_heads": 2, "mlp_ratio": 2, "dtype": "float32",
                                           "attention_impl": "xla"}),
        predictor={"num_layers": 1, "predictor_features": 8}, num_target_blocks=2,
        data=patched(train, drawn=drawn), validation=patched(held, drawn=False))


KINDS = {"lm": lm, "decision": decision, "diffusion": diffusion, "jepa": jepa}
ADAPTED = {"lm": ("q_proj",), "decision": ("q_proj",), "diffusion": ("ada_proj",), "jepa": ("up_proj",)}
"""A module of each kind's model a `lora` adapts."""
DEFAULT_METRICS = {"lm": "val/perplexity", "decision": "val/accuracy"}
"""A metric each kind scores where `val_metrics` names none, for the kinds that score one."""
def scalars(run: Path) -> dict[int, dict[str, float]]:
    """Each step's scalars the run logged, joined."""
    logged = {}
    for line in (run / "tracking" / "scalars.jsonl").read_text().splitlines():
        row = json.loads(line)
        logged.setdefault(row["step"], {}).update(row["scalars"])
    return logged


def trained(run: RunConfig, root: Path, name: str, **fields):
    """`run` trained one step from Python (`prepare().train`) over a mesh of four data shards by two
    fsdp shards, its trainer the fields given, over its defaults here: its state, and its step's
    scalars."""
    trainer = TrainerConfig(**{"checkpoint_dir": str(root / "runs"), "steps": 1, "log_every": 1,
                               "eval_every": None, "checkpoint_every": None, "checkpoint_final": False,
                               "multi_host": False, "mesh": MeshSpec(fsdp=2), "layout": Layout(min_shard=64),
                               **fields.pop("trainer", {})})
    state = dataclasses.replace(run, trainer=trainer, **fields).prepare().train(name)
    return state, scalars(root / "runs" / name)[1]


def masked(root: Path, drawn: bool) -> RunConfig:
    """`lm`'s run as a masked-diffusion one, its model bidirectional with a mask id, which previews
    nothing."""
    run = lm(root, drawn)
    model = ModelConfig("causal_transformer", {**run.model.fields, "causal": False, "mask_token_id": 0})
    return dataclasses.replace(run, objective=ObjectiveConfig("masked_diffusion"), model=model,
                               max_new_tokens=0)


def dreamed(root: Path, drawn: bool, release: str = "dream") -> RunConfig:
    """A masked-diffusion run over a Dream `release` (dream-tiny as it ships, `test_dream_walk.snapshot`,
    under `root / release`), its walk the release's own, its grid's endpoint among it (ledger 2563), over
    token files in the release's tokenizer."""
    from test_dream_walk import snapshot

    source = snapshot(release, root)
    corpus = write_token_corpus(root / "dream-tokens", np.random.RandomState(0).randint(4, 100, 400),
                                tokenizer=str(source), vocab_size=128, eos_id=1)
    data = TokenFiles(corpus.train).pipe(presets.lm, None, seq_len=11, batch=8, pack=None).loading(
        Loading(workers=0))
    model = ModelConfig("causal_transformer", {"dtype": "float32", "attention_impl": "reference"})
    return dataclasses.replace(lm(root, drawn), data=data, validation=None, pretrained=str(source),
                               tokenizer=str(source), model=model, max_new_tokens=0,
                               objective=ObjectiveConfig("masked_diffusion"))


def prefixed(root: Path, drawn: bool) -> RunConfig:
    """`lm`'s run over a decoder whose attention reads a prefix's blocks, which a prefix run needs, in two
    layers: one alone answers a generation from the prompt's last place, which a prefix's block shows
    nothing new, where a second reads what the block's places saw."""
    from dew.nn.backbones.causal_transformer import CausalTransformer, LayerKind
    from dew.nn.mixers.attention import AttentionMixer

    run = lm(root, drawn)
    kinds = {"blocks": LayerKind(mixer=AttentionMixer(bidirectional_blocks=True))}
    record = registry.to_record(kinds, typing.get_type_hints(CausalTransformer)["kinds"])
    return dataclasses.replace(run, model=ModelConfig("causal_transformer", {
        **run.model.fields, "num_layers": 2, "layer_types": ["blocks", "blocks"], "kinds": record}))


def state_then_question() -> tuple[QuestionSegment, ...]:
    """The built-in question row's segments (`built_in`) with the state first and the question's header
    straight after it, no separator between, so a state's last word and the header's first abut: a row
    whose unit (`QuestionLayout.unit`) its tokens show."""
    begin, header, separator, lines, _, state, end = built_in(None, "question_first")
    return begin, state, header, separator, lines, end


BESIDE = {"layout.framing": {"segments": built_in(None, "question_first"), "max_len": 256},
          "layout.unit": {"segments": state_then_question(), "max_len": 256},
          "context_encoder": {"context_layers": (0,), "context_length": 64}}
def lettered(root: Path, drawn: bool) -> RunConfig:
    """A decision run of a fresh letter readout (`--head letters`) over a tiny causal source, its rows
    lettered option lines read at an answer cue, the layout's labels the head's."""
    # The tiny tokenizer spells each letter as one token bare ("{}"), not after a space.
    return dataclasses.replace(decision(root, drawn), pretrained=str(QWEN_TINY),
                               head=Letters(labels=Labels(variants=("{}",)), count=4),
                               source=SourceOptions(text_only=True))


def merged(root: Path, drawn: bool) -> RunConfig:
    """A decision run of a fresh head over the tiny causal source with a tokenizer that merges "n" and "c"
    (`merging_source`), where a row encoded whole and one encoded piece by piece are other tokens, which
    its byte-level tokenizer alone never makes them."""
    return dataclasses.replace(decision(root, drawn), pretrained=str(merging_source(root)),
                               source=SourceOptions(text_only=True))


def merging_source(root: Path) -> Path:
    """The tiny Qwen source under `root`, its byte-level tokenizer given one merge, "n" and "c", in place of
    the last byte's token (U+0143, byte 0xff), which no text here holds: a vocabulary of the model's size
    in which a state's last word ("down") and a header's first ("choice") read whole as other tokens."""
    import shutil

    source = Path(shutil.copytree(QWEN_TINY, root / "merging-qwen", ignore=shutil.ignore_patterns(
        "*.npy", "*.npz", "video*", "images*", "preprocessor_config.json")))
    tokenizer = json.loads((source / "tokenizer.json").read_text())
    vocabulary = tokenizer["model"]["vocab"]
    vocabulary["nc"] = vocabulary.pop("Ń")
    tokenizer["model"]["merges"] = [["n", "c"]]
    (source / "tokenizer.json").write_text(json.dumps(tokenizer))
    return source


def blockwise(root: Path, drawn: bool, release: str = str(DIFFUSION_GEMMA)) -> RunConfig:
    """A block-diffusion run over a release whose text decoder its model nests (`DiffusionGemma.text`), its
    windows token files in the release's own tokenizer, both named as `release` names them."""
    corpus = write_token_corpus(root / "gemma-tokens", np.random.RandomState(160).randint(4, 32, 160),
                                tokenizer=release, vocab_size=32, eos_id=1)
    spans = TokenFiles(corpus.train).pipe(presets.lm, None, seq_len=11, batch=8, pack=None).loading(
        Loading(workers=0))
    model = ModelConfig("causal_transformer", {"dtype": "float32", "attention_impl": "xla"})
    return dataclasses.replace(lm(root, drawn), pretrained=release, tokenizer=release, model=model,
                               data=spans, validation=None,
                               objective=ObjectiveConfig("block_diffusion", {"prompt_length": 4}),
                               max_new_tokens=0)


def blocked(root: Path, drawn: bool) -> RunConfig:
    """`decision`'s run as a fresh head over a tiny causal backbone whose full attention reads blocks
    (`--source.attention`), whose rows a split takes, which a release's rows never are."""
    return dataclasses.replace(decision(root, drawn), pretrained=str(QWEN_TINY), source=SourceOptions(
        text_only=True, attention={"full_attention": "block_bidirectional"}))


def trained_by_cli(run: RunConfig, root: Path, name: str, assigned: dict[str, str]) -> Path:
    """The run directory one step of `dew train` writes for `run` with each `path=value` of `assigned`,
    its programs compiled into the suite's cache, which a run naming its own would leave this process
    holding."""
    record, runs = root / f"{name}-record", root / "runs"
    run.save(str(record))
    given = {"trainer.name": name, "trainer.checkpoint_dir": str(runs), "trainer.steps": "1",
             "trainer.eval_every": "None", "trainer.checkpoint_every": "None",
             "trainer.multi_host": "False", "trainer.compilation_cache_dir": "None", **assigned}
    assert main(["train", str(record / "run.json"),
                 *(part for path, value in given.items() for part in ("--set", f"{path}={value}"))]) == 0
    return runs / name


def requested(task) -> object:
    """One request of a loaded task, as its kind's users make it, at the task's own defaults but a
    generation's budget of two tokens, its answers: a decision's with a context beside its state where its
    backbone reads one."""
    from dew.decision import Choice
    from dew.decision.task import Decide
    from dew.inference.images import TextToImage
    from dew.inference.tasks import BlockGeneration, MaskedGeneration, TextGeneration

    if isinstance(task, TextGeneration):
        return task("ab", max_new_tokens=2, key=0)
    if isinstance(task, MaskedGeneration | BlockGeneration):
        # Token ids, which a source with no tokenizer of its own (dream-tiny) reads too, within every tiny
        # source's vocabulary (diffusion-gemma-sft's 32).
        return task([[4, 5, 6, 7]], max_new_tokens=2, key=0)
    if isinstance(task, Decide):
        question = {"team": Choice("which team", ["billing", "technical", "sales"])}
        # A context beside the state where the backbone's cross-attention reads one (`context_length`),
        # unless its layout reads the question there (`spans_in_context`), which reads no other.
        reads = task.context_length is not None and not task.layout.spans_in_context
        context = {"plan": "enterprise"} if reads else None
        # A state of plain text, which a row lays as written, its last word "down".
        return task([("the site is down", question)], contexts=[context])
    if isinstance(task, TextToImage):
        return task(["a cat"], key=0)
    raise AssertionError(f"no request is written for a loaded {type(task).__name__}")
