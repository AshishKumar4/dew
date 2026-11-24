"""The runs of each kind the option tests, the saved-run corpus and the GPU cells build: a language
model's, a decision's, a diffusion model's and a JEPA's, at their small sizes, each beside the run it
is set from, and the `dew train` step that trains one (`trained_by_cli`)."""

import csv
import dataclasses
import hashlib
import json
import os
import typing
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np
from objective_data import IMAGES
from process_support import REPO_ROOT, dew_train, worker_env
from test_lm_recipe import SEQ, write_token_files
from token_support import write_token_corpus

from dew import registry
from dew._cli.main import main
from dew.config import LowerTarget, ObjectiveConfig, RunConfig, SourceOptions, TrainerConfig
from dew.config.diffusion import DEFAULT_MODEL_CONFIG, DiffusionRunConfig
from dew.config.jepa import JepaRunConfig
from dew.config.lm import LMRunConfig
from dew.data import TFDS, Files, Loading, images as image_steps, presets
from dew.data.pipeline import Rejected
from dew.data.sources.tokens import TokenFiles
from dew.decision import decisions
from dew.decision.config import DecisionRunConfig, Letters
from dew.decision.layout import Labels, QuestionSegment, built_in, per_question
from dew.diffusion.presets import Flow
from dew.inputs import TextCondition
from dew.registry import ModelConfig
from dew.sampling import Euler
from dew.training import Layout, LocalTracker, MeshSpec
from dew.training.budget import Horizon

TINY_LAYA = Path(__file__).parent / "fixtures" / "laya" / "tiny"
QWEN_TINY = Path(__file__).parent / "fixtures" / "hf" / "qwen38-dense-tiny"
GEMMA_TINY = Path(__file__).parent / "fixtures" / "hf" / "gemma3-native-tiny"
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


def read(chain, reading: str = "memory"):
    """`chain` read as `reading` names (memory where it names none), in this process."""
    return chain.read(reading).loading(Loading(workers=0))


ROWS = {"windows": {"pack": None}, "packed": {"pack": "first-fit", "bins": 2},
        "padded": {"pack": "first-fit", "segments": 1}}
"""How `lm`'s rows lay its tokens: fixed windows over the store, documents first-fit beside each other
(their segment ids apart), or one document a row, padded (`presets.lm`'s `segments=1`)."""


def lm(root: Path, drawn: bool, *, rows: str = "windows", reading: str = "memory",
       dtype: str | None = "float32") -> RunConfig:
    """A tiny decoder's language-model run over a token store, its rows laid as `rows` names (`ROWS`, its
    store of documents, an end of text every seventh token, where they are packed or padded), read as
    `reading` names, its training rows passed through a random filter where `drawn`; its model's `dtype`
    stated, or none (None), so the run's precision derives it (`RunConfig.numerics_fields`)."""
    corpus = write_token_files(root / "tokens", 40 * SEQ, 8 * SEQ, eos_id=None if rows == "windows" else 0)

    def windows(path, *, drawn: bool):
        chain = TokenFiles(path).pipe(presets.lm, None, seq_len=SEQ, batch=None, **ROWS[rows])
        chain = chain.random_filter_map(MostlyKept()) if drawn else chain
        return read(chain.batch(8), reading)
    fields = {"emb_features": 16, "num_layers": 1, "num_heads": 2,
              **({} if dtype is None else {"dtype": dtype})}
    return LMRunConfig(model=ModelConfig("causal_transformer", fields),
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


def captioned(rows, size: int, *, drawn: bool):
    """`rows`' pictures at `size` square in batches of 8, each with its caption, dimmed where `drawn`."""
    return read(rows.pipe(presets.images, size, batch=8, caption=image_steps.caption(),
                          augment=Dimmed() if drawn else False))


def diffusion(root: Path, drawn: bool) -> RunConfig:
    del root
    train, held = pictures()
    return DiffusionRunConfig(
        model=ModelConfig("simple_dit", {"patch_size": 2, "emb_features": 16, "num_layers": 1,
                                         "num_heads": 2}),
        data=captioned(train, 4, drawn=drawn), validation=captioned(held, 4, drawn=False), preset=Flow(),
        val_metrics=(),
        text=TextCondition(encoder="char_table", checkpoint="char_table"),
        objective=ObjectiveConfig("diffusion", {"solver": Euler(), "steps": 2}))


def published(root: Path, drawn: bool) -> RunConfig:
    """A diffusion run over the tiny SD pipeline (tiny_diffusers.tar.xz's `sd`, unpacked under `root`) naming
    no preset, so it trains the pipeline's own process over its grid (`ImageSettings.grid_process`)."""
    from interop_support import tiny_pipeline

    pipeline = root / "tiny_diffusers" / "sd"
    if not pipeline.exists():
        tiny_pipeline(root / "tiny_diffusers", "sd")
    train, held = pictures()
    return DiffusionRunConfig(
        pretrained=str(pipeline), preset=None,
        model=ModelConfig("unet", {**DEFAULT_MODEL_CONFIG, "dtype": "float32"}),
        data=captioned(train, 32, drawn=drawn), validation=captioned(held, 32, drawn=False), val_metrics=(),
        objective=ObjectiveConfig("diffusion", {"steps": 2}))


def jepa(root: Path, drawn: bool) -> RunConfig:
    del root
    train, held = pictures()

    def patched(rows, *, drawn: bool):
        return read(rows.pipe(presets.images, 16, batch=8, augment=Dimmed() if drawn else False))
    return JepaRunConfig(
        model=ModelConfig("jepa_encoder", {"patch_size": 2, "emb_features": 16, "num_layers": 1,
                                           "num_heads": 2, "mlp_ratio": 2, "dtype": "float32"}),
        predictor={"num_layers": 1, "predictor_features": 8}, num_target_blocks=2,
        data=patched(train, drawn=drawn), validation=patched(held, drawn=False))


KINDS = {"lm": lm, "decision": decision, "diffusion": diffusion, "jepa": jepa}
ADAPTED = {"lm": ("q_proj",), "decision": ("q_proj",), "diffusion": ("ada_proj",), "jepa": ("up_proj",)}
"""A module of each kind's model a `lora` adapts."""
DEFAULT_METRICS = {"lm": "val/perplexity", "decision": "val/accuracy"}
"""A metric each kind scores where `val_metrics` names none, for the kinds that score one."""
def scalars(run: Path) -> dict[int, dict[str, float]]:
    """Each step's scalars the run logged (`LocalTracker.scalars`), joined."""
    logged = {}
    for step, values in LocalTracker(run / "tracking").scalars():
        logged.setdefault(step, {}).update(values)
    return logged


def trained(run: RunConfig, root: Path, name: str, lowering: LowerTarget | None = None, **fields):
    """`run` trained one step from Python (`prepare().train`) over a mesh of four data shards by two
    fsdp shards, its trainer the fields given, over its defaults here: its state, and its step's
    scalars; lowered for `lowering` in place of training where given (`Prepared.lower`), which ends by
    raising."""
    trainer = TrainerConfig(**{"checkpoint_dir": str(root / "runs"), "horizon": Horizon(steps=1),
                               "log_every": 1, "eval_every": None, "checkpoint_every": None,
                               "checkpoint_final": False, "multi_host": False, "mesh": MeshSpec(fsdp=2),
                               "layout": Layout(min_shard=64), **fields.pop("trainer", {})})
    prepared = dataclasses.replace(run, trainer=trainer, **fields).prepare()
    if lowering is not None:
        prepared.lower(name, lowering)
    state = prepared.train(name)
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
    model = ModelConfig("causal_transformer", {"dtype": "float32"})
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
    whose unit (`SegmentLayout.unit`) its tokens show."""
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


def shared(root: Path, drawn: bool) -> RunConfig:
    """A decision run of a fresh letter readout over the tiny Gemma's text decoder, whose every layer reads a
    row's key runs, its rows asking a request's questions over one copy of its state (`shared_state`), its
    labels words of that tokenizer's vocabulary (`token4`...), which spells "A" to "D" as one unknown
    token."""
    run = lettered(root, drawn)
    assert isinstance(run.head, Letters)
    labels = dataclasses.replace(run.head.labels, names=("token4", "token5", "token6", "token7"))
    return dataclasses.replace(run, pretrained=str(GEMMA_TINY), shared_state=3,
                               head=dataclasses.replace(run.head, labels=labels))


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


def blockwise(root: Path, drawn: bool, release: str = str(DIFFUSION_GEMMA), tokens: int = 160) -> RunConfig:
    """A block-diffusion run over a release whose text decoder its model nests (`DiffusionGemma.text`), its
    windows token files of `tokens` drawn ids in the release's own tokenizer, both named as `release` names
    them."""
    corpus = write_token_corpus(root / "gemma-tokens", np.random.RandomState(160).randint(4, 32, tokens),
                                tokenizer=release, vocab_size=32, eos_id=1)
    spans = TokenFiles(corpus.train).pipe(presets.lm, None, seq_len=11, batch=8, pack=None).loading(
        Loading(workers=0))
    model = ModelConfig("causal_transformer", {"dtype": "float32"})
    return dataclasses.replace(lm(root, drawn), pretrained=release, tokenizer=release, model=model,
                               data=spans, validation=None,
                               objective=ObjectiveConfig("block_diffusion", {"prompt_length": 4}),
                               max_new_tokens=0)


def blocked(root: Path, drawn: bool) -> RunConfig:
    """`decision`'s run as a fresh head over a tiny causal backbone whose full attention reads blocks
    (`--source.attention`), whose rows a split takes, which a release's rows never are."""
    return dataclasses.replace(decision(root, drawn), pretrained=str(QWEN_TINY), source=SourceOptions(
        text_only=True, attention={"full_attention": "block_bidirectional"}))


def saved_record(run: RunConfig, root: Path, name: str) -> Path:
    """`run` saved as the run record `dew train` replays, under `root`."""
    record = root / f"{name}-record"
    run.save(str(record))
    return record


def cli_settings(root: Path, name: str, assigned: Mapping[str, object] = MappingProxyType({})
                 ) -> dict[str, object]:
    """A run's settings, by dotted path (`RunConfig.override`), for a run named `name` under `root`: one
    step, no evaluation or checkpoint between steps, one host, its programs compiled into the suite's cache,
    which a run naming its own would leave the training process holding; each of `assigned` over them."""
    return {"trainer.name": name, "trainer.checkpoint_dir": str(root / "runs"), "trainer.horizon.steps": 1,
            "trainer.eval_every": None, "trainer.checkpoint_every": None, "trainer.multi_host": False,
            "trainer.compilation_cache_dir": None, **assigned}


def stated[R: RunConfig](run: R, *, model: str | None = None,
                         model_fields: Mapping[str, object] = MappingProxyType({}),
                         objective: str | None = None,
                         objective_fields: Mapping[str, object] = MappingProxyType({})) -> R:
    """`run` with its model of the class `model` names (its own where None) and `model_fields` over its
    fields, and its objective alike, each field as its record holds it: what a run file states of what the
    run trains."""
    held = run.model if model is None else ModelConfig(model)
    goal = run.objective if objective is None else ObjectiveConfig(objective)
    assert goal is not None or not objective_fields, "a run of no objective takes none of its fields"
    return dataclasses.replace(
        run, model=dataclasses.replace(held, fields={**held.fields, **model_fields}),
        objective=None if goal is None else dataclasses.replace(goal,
                                                                fields={**goal.fields, **objective_fields}))


def settled(settings: Mapping[str, object]) -> dict[str, object]:
    """`settings` by dotted path as `RunConfig.override` sets them: a value given as text read as its JSON
    where it is JSON ("4", "null"; Python's None, True and False as theirs), any other as it is."""
    def value(given: object) -> object:
        if not isinstance(given, str):
            return given
        text = {"None": "null", "True": "true", "False": "false"}.get(given, given)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return given

    return {path: value(given) for path, given in settings.items()}


def set_record(record: Path, settings: Mapping[str, object], trust: tuple[str, ...] = ()) -> Path:
    """The run saved at `record` with each of `settings` set at its dotted path (`settled`), saved beside it
    as the run record a command reads: a run is set in its record or its Python file, which `dew train`
    trains as it is."""
    given = settled(settings)
    run = RunConfig.load(str(record), trust=trust).override(given)
    # Its own directory for its own settings: runs set from one record side by side write each its own.
    digest = hashlib.sha256(json.dumps(given, sort_keys=True, default=str).encode()).hexdigest()[:12]
    written = record.parent / f"{record.name}-set-{digest}"
    run.save(str(written))
    return written


def trained_record(record: Path, settings: Mapping[str, object], devices: int | None = None,
                   trust: tuple[str, ...] = ()) -> Path:
    """The run directory `dew train` writes for the run saved at `record` with `settings` set in its record
    (`set_record`) and `trust` trusted, as a user's command trains it: in this process where `devices` is
    None, else in a fresh one that sees `devices` CPU devices (`process_support.worker_env`), a run's device
    count being its process's, fixed as its backend opens. It writes where its trainer names
    (`TrainerConfig.directory`), its record's name and directory under those `settings` sets: a run resumed
    from its record its own."""
    written = set_record(record, settings, trust)
    trainer = RunConfig.load(str(written), trust=trust).trainer
    directory = trainer.directory(str(trainer.name))
    arguments = [str(written / "run.json"), *(part for package in trust for part in ("--trust", package))]
    if devices is None:
        assert main(["train", *arguments]) == 0
    else:
        dew_train(REPO_ROOT, {**worker_env(devices), "PYTHONPATH": os.pathsep.join(
            (str(REPO_ROOT / "src"), str(REPO_ROOT / "tests")))}, *arguments)
    return Path(directory)


def trained_by_cli(run: RunConfig, root: Path, name: str, assigned: dict[str, str]) -> Path:
    """The run directory one step of `dew train` writes for `run` with each of `assigned` set at its path
    (`saved_record`, `cli_settings`, `trained_record`), in this process."""
    return trained_record(saved_record(run, root, name), cli_settings(root, name, assigned))


def requested(task) -> object:
    """One request of a loaded task, as its kind's users make it, at the task's own defaults but a
    generation's budget of two tokens, its answers: a decision's with a context beside its state where its
    backbone reads one."""
    from dew.decision import Choice
    from dew.decision.task import Decide
    from dew.inference.images import TextToImage
    from dew.inference.tasks import BlockGeneration, MaskedGeneration, TextGeneration

    if isinstance(task, TextGeneration) and task.processor is None:
        # A run of token ids alone (preference pairs) recorded no tokenizer: its task continues ids.
        return task([[1, 2, 3]], max_new_tokens=2, key=0)
    if isinstance(task, TextGeneration):
        return task("ab", max_new_tokens=2, key=0)
    if isinstance(task, MaskedGeneration | BlockGeneration):
        # Token ids, which a source with no tokenizer of its own (dream-tiny) reads too, within every tiny
        # source's vocabulary (diffusion-gemma-sft's 32).
        return task([[4, 5, 6, 7]], max_new_tokens=2, key=0)
    if isinstance(task, Decide):
        # Options described, as a user's are, so a row's writing of them (`SegmentLayout.option_text`) shows.
        question = {"team": Choice("which team", {"billing": "invoices and charges",
                                                  "technical": "the site and the app",
                                                  "sales": "plans and prices"})}
        # A context beside the state where the backbone's cross-attention reads one (`context_length`),
        # unless its layout reads the question there (`spans_in_context`), which reads no other.
        reads = task.context_length is not None and not task.layout.spans_in_context
        context = {"plan": "enterprise"} if reads else None
        if per_question(task.layout) and task.layout.shares:
            # Two questions, which a row over one state asks together, each reading the state once.
            question = {**question,
                        "urgent": Choice("how urgent", {"now": "blocks work", "later": "can wait"})}
        # A state of plain text, which a row lays as written, its last word "down".
        return task([("the site is down", question)], contexts=[context])
    if isinstance(task, TextToImage):
        return task(["a cat"], key=0)
    raise AssertionError(f"no request is written for a loaded {type(task).__name__}")
