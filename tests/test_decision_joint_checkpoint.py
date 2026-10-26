"""A joint-head release's layout and head against the release's own code, under the float64 rule.

The release and its commit are tests/fixtures/joint_checkpoint/tiny/source.json's,
and tools/joint_checkpoint_reference.py wrote the fixtures from its code."""

import base64
import io
import json
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
import optax
import pytest
from decision_support import decision_rows
from interop_support import allocated_bytes, written_tensors
from PIL import Image
from reference_error import assert_as_exact_as_the_reference, float64_forward, ieee_fixture
from safetensors.numpy import load_file

from dew.checkpoints import Checkpoints
from dew.data.text import HFTokenizer
from dew.decision import (
    Choice,
    ChoiceAnswer,
    Decide,
    DecisionInputs,
    DecisionObjective,
    Encoded,
    Example,
    JointLayout,
    Noul,
    Question,
    Schema,
    Specials,
    TopProbability,
)
from dew.decision.calibration import softmax
from dew.decision.images import Images
from dew.decision.layout import JOINT_SEGMENTS
from dew.interop import Pretrained
from dew.nn.kernels.generation import measured_kernel
from dew.training.trainer import Trainer

FIXTURES = Path(__file__).parent / "fixtures"
TINY = FIXTURES / "joint_checkpoint" / "tiny"
BACKBONE = FIXTURES / "hf" / "qwen38-dense-tiny"
LAYA_TINY = FIXTURES / "laya" / "tiny"
CASES = json.loads((FIXTURES / "laya" / "cases.json").read_text())


def test_a_shapes_only_readout_load_avoids_eager_native_buffers(monkeypatch):
    """The standalone joint readout needs no backbone: its shapes-only load retains every
    parameter's geometry while allocating fewer native-layout buffers than its eager load."""
    expected = Pretrained.load(TINY, dtype="float32", attention_impl="xla", on_host=True)
    _, eager_bytes = allocated_bytes(monkeypatch, lambda: Pretrained.load(
        TINY, dtype="float32", attention_impl="xla", on_host=True))
    loaded, deferred_bytes = allocated_bytes(monkeypatch, lambda: Pretrained.load(
        TINY, dtype="float32", attention_impl="xla", weights=False))
    assert eager_bytes > 0 and deferred_bytes < eager_bytes
    assert jax.tree.structure(loaded.variables) == jax.tree.structure(expected.variables)
    for actual, wanted in zip(jax.tree.leaves(loaded.variables), jax.tree.leaves(expected.variables),
                              strict=True):
        assert isinstance(actual, jax.ShapeDtypeStruct)
        assert actual.shape == wanted.shape and actual.dtype == wanted.dtype


@pytest.fixture(scope="module")
def tokenizer() -> HFTokenizer:
    return HFTokenizer(str(BACKBONE), local_files_only=True)


def test_a_transformers_tokenizer_gives_the_special_tokens_its_dew_wrapper_gives(tokenizer):
    """Specials.of reads the special tokens off a loaded transformers tokenizer as it does off the
    HFTokenizer over the same files, with no wrapper to build first; a record naming it reads its
    argument against a type its module holds at runtime."""
    from transformers import AutoTokenizer

    from dew.data.text import Tokenizer
    from dew.decision.layout import TokenizerSpecials
    from dew.registry import parameter_type

    assert Specials.of(AutoTokenizer.from_pretrained(str(BACKBONE), local_files_only=True)) == Specials.of(
        tokenizer)
    assert parameter_type(Specials.of, "tokenizer") == Tokenizer | TokenizerSpecials


def test_any_tokenizer_holding_the_special_tokens_gives_them(tokenizer):
    """A tokenizer of no transformers class that holds the ids and token text
    `TokenizerSpecials` declares gives the same Specials as the transformers
    tokenizer it copies; one that holds none, a ByteTokenizer, is refused."""
    from types import SimpleNamespace

    from dew.data.text import ByteTokenizer

    hf = tokenizer.tokenizer
    names = ("cls_token_id", "sep_token_id", "eos_token_id", "bos_token_id", "mask_token_id", "pad_token_id")
    own = SimpleNamespace(**{name: getattr(hf, name) for name in names},
                          convert_ids_to_tokens=hf.convert_ids_to_tokens)
    assert Specials.of(own) == Specials.of(tokenizer)
    with pytest.raises(ValueError):
        Specials.of(ByteTokenizer())


@pytest.mark.parametrize("zero", ["[PAD]", "[MASK]"])
def test_a_special_token_at_id_zero_is_the_tokenizers_own(zero):
    """A vocabulary may put its padding or mask token at id 0, as BERT's puts
    [PAD]; Specials.of keeps that id rather than taking 0 for a missing one
    and laying out the end token in its place."""
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    words = [zero, *(word for word in ("[PAD]", "[MASK]", "[SEP]", "[UNK]") if word != zero)]
    vocabulary = {word: index for index, word in enumerate(words)}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", mask_token="[MASK]",
                                        sep_token="[SEP]", unk_token="[UNK]")
    specials = Specials.of(tokenizer)
    assert (specials.pad, specials.marker) == (tokenizer.pad_token_id, tokenizer.mask_token_id)
    assert 0 in (specials.pad, specials.marker)


def laid_out(tokenizer: HFTokenizer) -> dict[str, tuple[dict[str, Question], Encoded]]:
    """Each case's questions and its one joint row."""
    rows = {}
    for case, request in CASES.items():
        questions = {name: Question.from_wire(wire) for name, wire in request["questions"].items()}
        [row] = JointLayout().rows(tokenizer, Specials.of(tokenizer), request["state"], questions)
        rows[case] = questions, row
    return rows


def test_the_joint_layout_lays_out_what_the_release_does(tokenizer):
    """Token for token and span for span: the release's prompt, a choice's
    options by key, a noul's true first with the release's descriptions,
    structured values as compact sorted JSON, and every option's span."""
    expected = json.loads((TINY / "layouts.json").read_text())
    for case, (questions, row) in laid_out(tokenizer).items():
        assert list(row.tokens) == expected[case]["ids"], case
        for laid, reference in zip(row.questions, expected[case]["questions"], strict=True):
            assert laid.name == reference["id"]
            assert list(laid.span) == reference["span"], (case, laid.name)
            assert [list(span) for span in laid.options] == reference["options"], (case, laid.name)
            assert [questions[laid.name].options[slot] for slot in laid.order] == reference["option_ids"]


def test_a_joint_schema_read_in_context_is_the_schema_the_row_holds(tokenizer):
    """A joint layout that reads its schema as the row's context (`Schema(stream="context")`) lays the
    release's schema out token for token, as the row that holds it does: the context is the schema's
    tokens in that row, each span the same tokens' place in the context, and the row the rest, its
    images and its state as that row holds them."""
    segments = tuple(Schema(stream="context") if isinstance(segment, Schema) else segment
                     for segment in JOINT_SEGMENTS)
    for case, (questions, held) in laid_out(tokenizer).items():
        request = CASES[case]
        [read] = JointLayout(segments=segments).rows(tokenizer, Specials.of(tokenizer), request["state"],
                                                    questions)
        assert read.spans_in_context and not held.spans_in_context and read.context is not None
        offset = held.questions[0].span[0] - read.questions[0].span[0]
        assert held.tokens[offset:offset + len(read.context)] == read.context, case
        assert held.tokens[:offset] + held.tokens[offset + len(read.context):] == read.tokens, case
        for inline, contextual in zip(held.questions, read.questions, strict=True):
            assert (inline.name, inline.order) == (contextual.name, contextual.order)
            assert inline.span == tuple(at + offset for at in contextual.span), (case, inline.name)
            assert inline.options == tuple(tuple(at + offset for at in option)
                                           for option in contextual.options), (case, inline.name)
        assert (held.state_tokens, held.state_kept, held.media) == (read.state_tokens, read.state_kept,
                                                                     read.media)


def head_logits(tokenizer: HFTokenizer) -> tuple[np.ndarray, ...]:
    """Dew's head, the release's fp32 head, its float64 head, and the terms
    Dew's float64 head sums for each logit (`float64_forward`), all on the
    same fp32 backbone states, over every question of every case."""
    loaded = Pretrained.load(TINY, dtype="float32", attention_impl="xla")
    table = load_file(BACKBONE / "model.safetensors")["lm_head.weight"]
    found, reference, truth, terms = [], [], [], []
    with np.load(TINY / "states.npz") as states, np.load(TINY / "logits.npz") as logits:
        for case, (_, row) in laid_out(tokenizer).items():
            inputs = DecisionInputs.collate([row], 0)
            given = (loaded.variables, states[case][None], inputs, table)
            scores = loaded.model.apply(*given)
            _, summed = float64_forward(loaded.model, *given)
            for slot, laid in enumerate(row.questions):
                key = f"{case}/{laid.name}"
                found.append(np.asarray(scores)[0, slot, :len(laid.options)])
                terms.append(np.asarray(summed)[0, slot, :len(laid.options)])
                reference.append(logits[key])
                truth.append(logits[f"{key}/head64"])
    return tuple(map(np.concatenate, (found, reference, truth, terms)))


def test_the_stored_joint_head_matches_its_release_under_the_float64_rule(tokenizer):
    """Every question of every case, read off the same fp32 backbone states,
    held to twice the release's own head's fp32 distance from its float64
    run (measured: 1.42 times)."""
    found, reference, truth, terms = head_logits(tokenizer)
    assert_as_exact_as_the_reference(found, reference, truth, "stored joint head logits", terms=terms)


def test_a_padded_batch_scores_every_row_as_it_scores_alone(tokenizer):
    """All five cases in one batch, padded to the longest row and the most
    questions and options: padding reaches none of the head's pooling or
    attention, so each question scores as it does alone."""
    loaded = Pretrained.load(TINY, dtype="float32", attention_impl="xla")
    table = load_file(BACKBONE / "model.safetensors")["lm_head.weight"]
    rows = laid_out(tokenizer)
    inputs = DecisionInputs.collate([row for _, row in rows.values()], 0)
    with np.load(TINY / "states.npz") as states:
        length = inputs.tokens.shape[1]
        padded = np.stack([np.pad(states[case], ((0, length - len(states[case])), (0, 0))) for case in rows])
        batched = np.asarray(loaded.model.apply(loaded.variables, padded, inputs, table))
        for index, (case, (_, row)) in enumerate(rows.items()):
            alone = np.asarray(loaded.model.apply(loaded.variables, states[case][None],
                                                 DecisionInputs.collate([row], 0), table))
            for slot, laid in enumerate(row.questions):
                width = len(laid.options)
                np.testing.assert_allclose(batched[index, slot, :width], alone[0, slot, :width], atol=2e-5)

    def total(rows_of_table):
        scores = loaded.model.apply(loaded.variables, padded, inputs, rows_of_table)
        return jax.numpy.sum(jax.numpy.where(jax.numpy.isfinite(scores), scores, 0.0))

    gradient = np.asarray(jax.grad(total)(jax.numpy.asarray(table)))
    assert np.all(np.isfinite(gradient)) and np.any(gradient)


def joint_logits(decide) -> np.ndarray:
    """Dew's logits of every question the cases lay out, in the fixture's order."""
    found = []
    for request in CASES.values():
        questions = {name: Question.from_wire(wire) for name, wire in request["questions"].items()}
        [row] = decide.layout.rows(decide.tokenizer, decide.specials, request["state"], questions)
        scores = np.asarray(decide.model.logits(decide.variables, DecisionInputs.collate([row], 0)))
        found += [scores[0, slot, :len(laid.options)] for slot, laid in enumerate(row.questions)]
    return np.concatenate(found)


def test_a_joint_release_answers_as_its_own_model(tmp_path):
    """`Decide.from_pretrained` reads a joint-head release, the Qwen 3.5
    backbone with the head beside it. Dew's backbone and head together are
    held to twice the release's own fp32 model's distance from its float64
    run, and `systemone`, with the release's confidence (the top
    probability), gives each option the probability the release's fp32
    logits give it, within the fourth decimal both round to.

    Where the device's target row names the delta rule's port, Dew's Gated
    DeltaNet runs fla's algorithm, one plain fp32 decay prefix per chunk, not
    the release's torch math: there the reference is the same release with
    HF's Qwen 3.5 on fla's own kernels (logits_fla.npz,
    tools/joint_checkpoint_reference.py --fla)."""
    for source in [*BACKBONE.iterdir(), TINY / "joint_head_config.json", TINY / "joint_head.safetensors"]:
        (tmp_path / source.name).symlink_to(source.resolve())
    decide = replace(Decide.from_pretrained(tmp_path, attention_impl="xla"), confidence=TopProbability())
    layouts = json.loads((TINY / "layouts.json").read_text())
    keys, reference, truth = [], [], []
    with np.load(TINY / "logits.npz") as logits:
        for case, request in CASES.items():
            questions = {name: Question.from_wire(wire) for name, wire in request["questions"].items()}
            [row] = decide.layout.rows(decide.tokenizer, decide.specials, request["state"], questions)
            answers = decide.systemone(request)["answers"]
            for laid, kept in zip(row.questions, layouts[case]["questions"], strict=True):
                key = f"{case}/{laid.name}"
                keys.append(key)
                reference.append(logits[key])
                truth.append(logits[f"{key}/f64"])
                probabilities = softmax(logits[key].astype(np.float64), 1.0)
                expected = dict(zip(kept["option_ids"], probabilities, strict=True))
                answer = answers[laid.name]
                given = ({"true": answer["noul"]} if answer["type"] == "noul" else answer["probabilities"])
                for option, value in given.items():
                    assert abs(value - expected[option]) <= 1e-4, (case, laid.name, option)
                if answer["type"] != "noul":
                    assert answer["confidence"] == max(given.values())
    if measured_kernel("gated_delta_rule", "xla") == "fla":
        on_fla = ieee_fixture(TINY / "logits_fla.npz")
        reference = [on_fla[key] for key in keys]
    assert_as_exact_as_the_reference(joint_logits(decide), np.concatenate(reference), np.concatenate(truth),
                                     "the joint head over Dew's Qwen 3.5")


def test_a_joint_release_saves_as_the_release_it_was_read_from(tmp_path):
    """A loaded joint-head release writes its own format back: the head's sizes, and every head
    tensor under the release's name, shape and dtype with its stored values, beside the backbone.
    The written release loads to the same task, every variable and answer the read one's."""
    release = tmp_path / "release"
    release.mkdir()
    for source in [*BACKBONE.iterdir(), TINY / "joint_head_config.json", TINY / "joint_head.safetensors"]:
        (release / source.name).symlink_to(source.resolve())
    bundle = Pretrained.load(release, dtype="float32", attention_impl="xla")
    bundle.save(tmp_path / "saved")
    assert json.loads((tmp_path / "saved" / "joint_head_config.json").read_text()) == json.loads(
        (TINY / "joint_head_config.json").read_text())
    written, stored = (load_file(path / "joint_head.safetensors") for path in (tmp_path / "saved", TINY))
    assert written.keys() == stored.keys()
    for name, tensor in stored.items():
        assert written[name].dtype == tensor.dtype, name
        np.testing.assert_array_equal(written[name], tensor, err_msg=name, strict=True)
    restored = Pretrained.load(tmp_path / "saved", dtype="float32", attention_impl="xla")
    for actual, expected in zip(jax.tree.leaves((restored.variables, restored.head_variables)),
                                jax.tree.leaves((bundle.variables, bundle.head_variables)), strict=True):
        np.testing.assert_array_equal(actual, expected)
    request = CASES["quickstart"]
    assert Decide.of(restored, name="release").systemone(request) == Decide.of(
        bundle, name="release").systemone(request)


def released(kind: str, directory: Path) -> Path:
    """A release of each decision format laid out at `directory`: the joint head beside its Qwen 3.5
    backbone, a slot-and-pointer head (`readout_head.pt`) beside the same backbone, or Laya's tiny
    checkpoint."""
    if kind == "laya":
        return LAYA_TINY
    from test_slot_sources import slot_release

    if kind == "slot":
        source, _ = slot_release(directory, "head")
        return source
    directory.mkdir()
    for source in [*BACKBONE.iterdir(), TINY / "joint_head_config.json", TINY / "joint_head.safetensors"]:
        (directory / source.name).symlink_to(source.resolve())
    return directory


@pytest.mark.parametrize("kind", ["joint", "slot", "laya"])
def test_a_decision_release_exports_every_tensor_it_saves(tmp_path, kind):
    """A release's `export` is every tensor its `save` writes, the readout's beside the backbone's, by
    its name in the release's files, so a weight push (`NCCLPush`) sends the whole task, where an
    inherited export once gave the backbone's tensors alone; `backbone` exports the backbone alone,
    the saved backbone's tensors."""
    bundle = Pretrained.load(released(kind, tmp_path / "release"), dtype="float32", attention_impl="xla")
    exported = bundle.export()
    bundle.save(tmp_path / "saved")
    written = written_tensors(tmp_path / "saved")
    assert exported.keys() == written.keys()
    for name, tensor in written.items():
        np.testing.assert_array_equal(np.asarray(exported[name]), tensor, err_msg=name, strict=True)
    if kind != "laya":
        backbone = bundle.backbone()
        assert not hasattr(backbone, "head_variables")
        backbone.save(tmp_path / "backbone")
        alone = written_tensors(tmp_path / "backbone")
        assert alone.keys() < exported.keys()
        for name, tensor in backbone.export().items():
            np.testing.assert_array_equal(alone[name], np.asarray(tensor), err_msg=name, strict=True)


def test_a_joint_release_fine_tunes_averages_and_reloads_from_its_runs(tmp_path):
    """A joint-head release is a whole task to fine-tune: its Qwen 3.5 backbone, joint head and layout
    train together on rows of several questions, and the saved run loads back as a task with identical
    probabilities. Two such runs average into one task, its temperatures refit on held-out answers
    (`Decide.averaged`), which saves to a directory of its own and reloads through `from_run` and
    `dew.pipeline` answering as it did (1505)."""
    import dataclasses

    from dew.decision import Calibration
    from dew.inference.loading import pipeline

    release = tmp_path / "release"
    release.mkdir()
    for source in [*BACKBONE.iterdir(), TINY / "joint_head_config.json", TINY / "joint_head.safetensors"]:
        (release / source.name).symlink_to(source.resolve())
    team = Choice("Which team?", {"billing": "payments", "technical": "outages", "sales": "pricing"})
    urgent = Noul("Is it urgent?")
    answered = (("charged twice", "billing", "true"), ("site is down", "technical", "true"),
                ("how much is pro", "sales", "false"), ("refund please", "billing", "false"))
    examples = [Example(state, {"team": team, "urgent": urgent}, {"team": label, "urgent": hurry})
                for state, label, hurry in answered * 2]
    # Training pads every row to the layout's max_len; the release serves up to 16384.
    objective = DecisionObjective(Decide.from_pretrained(release, attention_impl="xla"),
                                  layout=JointLayout(max_len=1024))
    data = decision_rows(examples, 8, examples, objective=objective.data())
    start = objective.init(jax.random.key(0))
    runs, states = [], []
    for index, steps in enumerate((2, 1)):
        run = tmp_path / f"run{index}"
        checkpoints = Checkpoints(str(run), keep=1)
        states.append(Trainer(objective, optax.adam(1e-3), key=index, checkpoints=checkpoints).fit(
            data, steps=steps, log_every=100, checkpoint_every=steps))
        checkpoints.wait()
        runs.append(str(run))
    before = jax.tree.leaves(start["params"]["head"])
    after = jax.tree.leaves(states[0].variables["params"]["head"])
    assert any(not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(before, after, strict=True))

    live, reloaded = objective.pipeline(states[0]), Decide.from_run(runs[0])
    assert live.processor is not None  # the release's, which reads images
    asked = {"team": team, "urgent": urgent}
    first, second = live("charged twice!", asked), reloaded("charged twice!", asked)
    for answer, again in zip(first.values(), second.values(), strict=True):
        np.testing.assert_array_equal(answer.probabilities, again.probabilities)

    task = Decide.averaged(runs, calibrate_on=examples * 4)
    assert task.calibration != Calibration()
    averaged = str(tmp_path / "averaged")
    task.save(averaged)
    for loaded in (Decide.from_run(averaged), pipeline(averaged)):
        assert isinstance(loaded, Decide)
        assert loaded.layout == task.layout and loaded.weights == task.weights
        assert loaded.calibration == task.calibration
        for state in ("charged twice!", "is the site down?"):
            found, expected = loaded(state, asked), task(state, asked)
            for name in expected:
                np.testing.assert_equal(dataclasses.asdict(found[name]), dataclasses.asdict(expected[name]))


def test_a_joint_release_reads_images_as_its_own_model_does(tmp_path):
    """A request's images, laid out before the state by the backbone's own
    processor as the release's `systemone` lays them out: the same token ids,
    and logits held to twice the release's fp32 model's distance from its
    float64 run,
    the vision tower included. `systemone` reads them as data URLs."""
    for source in [*BACKBONE.iterdir(), TINY / "joint_head_config.json", TINY / "joint_head.safetensors"]:
        (tmp_path / source.name).symlink_to(source.resolve())
    decide = Decide.from_pretrained(tmp_path, attention_impl="xla")
    pictures = [Image.fromarray(array) for array in np.load(BACKBONE / "images.npy")]
    request = CASES["quickstart"]
    questions = {name: Question.from_wire(wire) for name, wire in request["questions"].items()}
    assert decide.processor is not None
    media = Images.of(decide.processor, pictures, decide.processor.image_text())
    [row] = decide.layout.rows(decide.tokenizer, decide.specials, request["state"], questions,
                               media=media.tokens)
    assert list(row.tokens) == json.loads((TINY / "images.json").read_text())["ids"]
    scores = np.asarray(decide.model.logits(decide.variables, DecisionInputs.collate([row], 0),
                                            media=media.inputs(decide.processor, row)))
    with np.load(TINY / "images.npz") as logits:
        names = [laid.name for laid in row.questions]
        found = [scores[0, slot, :len(laid.options)] for slot, laid in enumerate(row.questions)]
        reference = [logits[name] for name in names]
        truth = [logits[f"{name}/f64"] for name in names]
        assert_as_exact_as_the_reference(np.concatenate(found), np.concatenate(reference),
                                         np.concatenate(truth), "joint head over Dew's Qwen 3.5, with images")

    def url(picture: Image.Image) -> str:
        encoded = io.BytesIO()
        picture.save(encoded, format="PNG")
        return "data:image/png;base64," + base64.b64encode(encoded.getvalue()).decode()

    served = decide.systemone({**request, "images": [url(picture) for picture in pictures]})
    assert served == decide.systemone({**request, "images": pictures})
    assert served["usage"]["input_tokens"] == len(row.tokens)
    seen = decide(request["state"], questions, images=pictures)["department"]
    assert isinstance(seen, ChoiceAnswer)
    shown = {option: round(float(p), 4) for option, p in zip(seen.options, seen.probabilities, strict=True)}
    assert served["answers"]["department"]["probabilities"] == shown
    assert decide.systemone(request)["answers"] != served["answers"]
