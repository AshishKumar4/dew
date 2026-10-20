"""Each preset prepares its kind of record for a model, and a run trains and validates on it."""

import dataclasses
import json
from pathlib import Path

import jax
import numpy as np
import optax
import pytest
from recording import RecordingTracker

from dew import Field, InputSpec, Trainer
from dew.data import Loading, Memory, images, mix, presets
from dew.data.chat import Role
from dew.data.compile import compile
from dew.data.dataset import DataPartition
from dew.data.steps.generic import Fields
from dew.data.steps.text import ValidTargets
from dew.diffusion.presets import Flow
from dew.nn.backbones import CausalTransformer, SimpleDiT
from dew.objectives.diffusion import DiffusionObjective
from dew.objectives.lm import LMObjective, Perplexity
from dew.sampling import Euler

CHAT = Path(__file__).resolve().parent / "fixtures" / "tokenizers" / "tiny-chat"


def trained(objective, train, held, **scoring) -> RecordingTracker:
    """Three steps of `train`, validating on `held` after each, as a run logs them."""
    tracker = RecordingTracker()
    Trainer(objective, optax.adam(1e-3), key=0, tracker=tracker).fit(
        train, validation=held, steps=3, log_every=1, eval_every=1, **scoring)
    return tracker


def finite(tracker: RecordingTracker, name: str) -> list[float]:
    values = [scalars[name] for _, scalars in tracker.scalars if name in scalars]
    assert values and all(np.isfinite(values)), (name, tracker.scalars)
    return values


def first_batch(dataset):
    stream = compile(dataset, loading=Loading(workers=0, threads=1, read_buffer=2)).train(DataPartition())
    try:
        return next(stream)
    finally:
        stream.close()


def test_images_train_a_denoiser_with_device_augmentation_and_validate_unaugmented():
    """Pictures of several sizes, one encoded as PNG bytes, become 16-pixel squares: augmented on
    the accelerator for training, as they are for validation, and a diffusion run learns on both.
    Each keeps its catalog URL, which stays on the host."""
    import io

    from PIL import Image

    rng = np.random.default_rng(0)
    pictures = [rng.integers(0, 256, (20 + index % 3 * 7, 24, 3), np.uint8) for index in range(64)]
    encoded = io.BytesIO()
    Image.fromarray(pictures[0]).save(encoded, format="PNG")
    rows = [{"image": encoded.getvalue(), "label": 0, "url": "https://example/0"},
            *({"image": picture, "label": 1, "url": f"https://example/{index + 1}"}
              for index, picture in enumerate(pictures))]
    train, held = Memory(rows).holdout(records=8)
    objective = DiffusionObjective(SimpleDiT(patch_size=4, emb_features=16, num_layers=1, num_heads=2,
                                             mlp_ratio=1),
                                   Flow(), InputSpec(Field("image", (16, 16, 3))), solver=Euler(), steps=2)
    batch = first_batch(train.shuffle(0).repeat().pipe(presets.images, 16, batch=8, augment=True))
    assert batch["image"].shape == (8, 16, 16, 3) and batch["image"].dtype == np.uint8
    assert batch["label"].shape == (8,) and batch["url"].shape == (8,)
    logged = trained(objective, train.shuffle(0).repeat().pipe(presets.images, 16, batch=8, augment=True),
                     held.pipe(presets.images, 16, batch=8), preview=True)
    finite(logged, "train/loss")
    assert logged.artifacts, "the validation batches were sampled from"


def test_images_apply_a_user_random_step_on_the_host_right_after_resize():
    pixels = np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)
    source = Memory([{"image": pixels, "label": index} for index in range(8)])

    def mirror(record, rng):
        assert record["image"].shape == (16, 16, 3)
        assert isinstance(record["image"], np.ndarray)
        return {**record, "image": record["image"][:, ::-1], "draw": rng.integers(2**16)}

    baseline = first_batch(source.pipe(presets.images, 16, batch=8))
    augmented = first_batch(source.pipe(presets.images, 16, batch=8, augment=mirror))
    equivalent = first_batch(source.filter_map(images.DecodeImage("image")).map(images.Resize(16))
                             .random_map(mirror).batch(8))
    np.testing.assert_array_equal(augmented["image"], baseline["image"][:, :, ::-1])
    np.testing.assert_array_equal(augmented["label"], baseline["label"])
    for name in augmented:
        np.testing.assert_array_equal(augmented[name], equivalent[name])


def test_lm_packs_documents_into_rows_that_keep_every_target():
    """Byte documents longer and shorter than a row pack into 33-token rows, each document its own
    segment, and a decoder trains and validates on them."""
    documents = [{"text": "dew trains jax models " * (1 + index % 4)} for index in range(64)]
    train, held = Memory(documents).holdout(records=8)
    batch = next(train.pipe(presets.lm, "byte", seq_len=32, batch=8).iterate())
    laid = batch["text_coordinates"]
    assert batch["text"].shape == laid.segments.shape == laid.positions.shape == (8, 33)
    assert laid.segments.max() > 1
    model = CausalTransformer(vocab_size=256, emb_features=32, num_layers=1, num_heads=2, mlp_features=64,
                              max_seq_len=64)
    logged = trained(LMObjective(model, seq_len=32), train.shuffle(0).repeat().pipe(
        presets.lm, "byte", seq_len=32, batch=8), held.pipe(presets.lm, "byte", seq_len=32, batch=8),
        metrics=(Perplexity(),))
    finite(logged, "train/loss")
    finite(logged, "val/perplexity")


def test_sft_renders_conversations_and_trains_on_the_assistant_turns():
    """Conversations through the tokenizer's chat template pack with each token's speaker beside it,
    and an assistant-only loss trains and validates on them."""
    conversations = [{"messages": [{"role": "user", "content": f"what is {index} plus one"},
                                   {"role": "assistant", "content": f"{index + 1}"}]} for index in range(32)]
    train, held = Memory(conversations).holdout(records=8)
    batch = next(train.pipe(presets.sft, str(CHAT), seq_len=48, batch=8).iterate())
    assert batch["text"].shape == batch["text_roles"].shape == (8, 49)
    assert (batch["text_roles"] == int(Role.ASSISTANT)).any()
    vocab = json.loads((CHAT / "tokenizer.json").read_text())["model"]["vocab"]
    model = CausalTransformer(vocab_size=len(vocab) + 8, emb_features=32, num_layers=1, num_heads=2,
                              mlp_features=64, max_seq_len=64)
    objective = LMObjective(model, seq_len=48, loss_role=Role.ASSISTANT)
    logged = trained(objective, train.shuffle(0).repeat().pipe(presets.sft, str(CHAT), seq_len=48, batch=8),
                     held.pipe(presets.sft, str(CHAT), seq_len=48, batch=8), metrics=(Perplexity(),))
    finite(logged, "train/loss")
    finite(logged, "val/perplexity")


def test_sft_keeps_every_turn_of_a_conversation_longer_than_its_rows():
    """A conversation longer than a row is cut into pieces that keep every turn (P37), not cut short:
    the pieces' ids, the boundary each shares with the next once, reassemble to the whole render."""
    from dew.data.chat import template

    turns = [{"role": role, "content": f"turn {index} of a long talk"}
             for index in range(12) for role in ("user", "assistant")]
    whole = next(Memory([{"messages": turns}]).map(template(str(CHAT))).iterate())["text"]
    assert len(whole) > 49
    pieces = list(Memory([{"messages": turns}]).pipe(presets.sft, str(CHAT), seq_len=48, batch=1).iterate())
    kept = [row[row_segments > 0] for batch in pieces
            for row, row_segments in zip(batch["text"], batch["text_coordinates"].segments, strict=True)]
    rebuilt = np.concatenate([kept[0], *(piece[1:] for piece in kept[1:])])
    np.testing.assert_array_equal(rebuilt, whole)


def sft_tokens(marker, targets, *, empty_rows=0):
    """Equal-length tokenized conversations with `targets` assistant tokens per accepted row."""
    roles = np.full(9, int(Role.USER), np.int8)
    if targets:
        roles[-targets:] = int(Role.ASSISTANT)
    rows = [{"text": np.full(9, marker, np.int32),
             "text_roles": np.full(9, int(Role.USER), np.int8) if index < empty_rows else roles}
            for index in range(64)]
    return Memory(rows).named(f"sft-{marker}").pack(9, fields=("text", "text_roles")).shuffle(marker).repeat()


@pytest.mark.parametrize("weights", [(0.5, 0.5), (0.25, 0.75)])
@pytest.mark.parametrize("empty_rows", [0, 32])
@pytest.mark.parametrize("projected", [False, True])
def test_sft_token_mixture_weighs_final_assistant_targets_and_resumes(weights, empty_rows, projected):
    """One versus seven assistant targets per packed row get the requested scored-token shares."""
    model = CausalTransformer(vocab_size=16, emb_features=8, num_layers=1, num_heads=2,
                              mlp_features=16, max_seq_len=8)
    policy = LMObjective(model, seq_len=8, loss_role=Role.ASSISTANT).data()
    field = "tokens" if projected else "text"
    if projected:
        names = ("text", "text_roles", "text_segment_ids")
        projection = Fields(names, rename=tuple((name, name.replace("text", field)) for name in names))
        policy = dataclasses.replace(policy, train=(*policy.train, projection))
    train = mix([sft_tokens(1, 1, empty_rows=empty_rows), sft_tokens(2, 7)], weights=weights, by="tokens")
    data = compile(train.batch(8), objective=policy, read="memory", loading=Loading(workers=0))
    reader = data.train(DataPartition())
    targets = ValidTargets(field=field, role=int(Role.ASSISTANT), measured=False)
    delivered = {1: 0, 2: 0}
    try:
        for _ in range(64):
            batch = next(reader)
            for index, marker in enumerate(batch[field][:, 0]):
                row = jax.tree.map(lambda values, index=index: values[index], batch)
                delivered[int(marker)] += targets(row)
        saved = reader.get_state()
        remaining = [next(reader)[field].tolist() for _ in range(3)]
    finally:
        reader.close()
    assert [delivered[marker] / sum(delivered.values()) for marker in (1, 2)] == pytest.approx(weights,
                                                                                          abs=0.01)
    assert data.counts["sft-1"].accepted == 64 - empty_rows
    assert data.counts["sft-1"].rejected["zero_targets"] == empty_rows
    resumed = compile(train.batch(8), objective=policy, read="memory",
                      loading=Loading(workers=0)).train(DataPartition())
    try:
        resumed.set_state(saved)
        assert [next(resumed)[field].tolist() for _ in range(3)] == remaining
    finally:
        resumed.close()


def test_sft_token_mixture_refuses_a_member_with_no_assistant_targets():
    """A corpus with no objective targets cannot silently disappear from a token-weighted mix."""
    model = CausalTransformer(vocab_size=16, emb_features=8, num_layers=1, num_heads=2,
                              mlp_features=16, max_seq_len=8)
    policy = LMObjective(model, seq_len=8, loss_role=Role.ASSISTANT).data()
    train = mix([sft_tokens(1, 0), sft_tokens(2, 7)], by="tokens").batch(8)
    with pytest.raises(ValueError):
        compile(train, objective=policy, read="memory", loading=Loading(workers=0))


def test_lm_over_a_mixture_of_raw_stores_tokenizes_and_packs_after_mixing():
    """Two raw text stores mixed, then the lm preset: the documents are mixed first, then tokenized
    and packed together, and the run's own token steps read the packed rows."""
    stories = Memory([{"text": "once upon a time " * (1 + index % 3),
                       "origin": "stories"} for index in range(32)])
    code = Memory([{"text": "def f(x): return x " * (1 + index % 2),
                    "origin": "code"} for index in range(32)])
    chain = mix([stories.shuffle(0).repeat(), code.shuffle(1).repeat()], weights=[0.5, 0.5])
    model = CausalTransformer(vocab_size=256, emb_features=32, num_layers=1, num_heads=2, mlp_features=64,
                              max_seq_len=64)
    held = mix([stories, code]).pipe(presets.lm, "byte", seq_len=32, batch=8)
    logged = trained(LMObjective(model, seq_len=32), chain.pipe(presets.lm, "byte", seq_len=32, batch=8),
                     held,
                     metrics=(Perplexity(),))
    finite(logged, "train/loss")
    finite(logged, "val/perplexity")
