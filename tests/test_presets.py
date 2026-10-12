"""Each preset prepares its kind of record for a model, and a run trains and validates on it."""

import json
from pathlib import Path

import numpy as np
import optax
from recording import RecordingTracker

from dew import Field, InputSpec, Trainer
from dew.data import Memory, presets
from dew.data.chat import Role
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


def test_images_train_a_denoiser_with_device_augmentation_and_validate_unaugmented():
    """Pictures of several sizes, one encoded as PNG bytes, become 16-pixel squares: augmented on
    the accelerator for training, as they are for validation, and a diffusion run learns on both."""
    import io

    from PIL import Image

    rng = np.random.default_rng(0)
    pictures = [rng.integers(0, 256, (20 + index % 3 * 7, 24, 3), np.uint8) for index in range(64)]
    encoded = io.BytesIO()
    Image.fromarray(pictures[0]).save(encoded, format="PNG")
    rows = [{"image": encoded.getvalue(), "label": 0}, *({"image": picture,
                                                          "label": 1} for picture in pictures)]
    train, held = Memory(rows).holdout(records=8)
    objective = DiffusionObjective(SimpleDiT(patch_size=4, emb_features=16, num_layers=1, num_heads=2,
                                             mlp_ratio=1),
                                   Flow(), InputSpec(Field("image", (16, 16, 3))), solver=Euler(), steps=2)
    batch = next(train.shuffle(0).repeat().pipe(presets.images(16, batch=8, augment=True)).iterate())
    assert batch["image"].shape == (8, 16, 16, 3) and batch["image"].dtype == np.uint8
    assert batch["label"].shape == (8,)
    logged = trained(objective, train.shuffle(0).repeat().pipe(presets.images(16, batch=8, augment=True)),
                     held.pipe(presets.images(16, batch=8)), preview=True)
    finite(logged, "train/loss")
    assert logged.artifacts, "the validation batches were sampled from"


def test_lm_packs_documents_into_rows_that_keep_every_target():
    """Byte documents longer and shorter than a row pack into 33-token rows, each document its own
    segment, and a decoder trains and validates on them."""
    documents = [{"text": "dew trains jax models " * (1 + index % 4)} for index in range(64)]
    train, held = Memory(documents).holdout(records=8)
    batch = next(train.pipe(presets.lm("byte", seq_len=32, batch=8)).iterate())
    assert {name: batch[name].shape for name in ("text", "text_segment_ids",
                                                 "text_positions")} == dict.fromkeys(
        ("text", "text_segment_ids", "text_positions"), (8, 33))
    assert batch["text_segment_ids"].max() > 1
    model = CausalTransformer(vocab_size=256, emb_features=32, num_layers=1, num_heads=2, mlp_features=64,
                              max_seq_len=64)
    logged = trained(LMObjective(model, seq_len=32), train.shuffle(0).repeat().pipe(
        presets.lm("byte", seq_len=32, batch=8)), held.pipe(presets.lm("byte", seq_len=32, batch=8)),
        metrics=(Perplexity(),))
    finite(logged, "train/loss")
    finite(logged, "val/perplexity")


def test_sft_renders_conversations_and_trains_on_the_assistant_turns():
    """Conversations through the tokenizer's chat template pack with each token's speaker beside it,
    and an assistant-only loss trains and validates on them."""
    conversations = [{"messages": [{"role": "user", "content": f"what is {index} plus one"},
                                   {"role": "assistant", "content": f"{index + 1}"}]} for index in range(32)]
    train, held = Memory(conversations).holdout(records=8)
    batch = next(train.pipe(presets.sft(str(CHAT), seq_len=48, batch=8)).iterate())
    assert batch["text"].shape == batch["text_roles"].shape == (8, 49)
    assert (batch["text_roles"] == int(Role.ASSISTANT)).any()
    vocab = json.loads((CHAT / "tokenizer.json").read_text())["model"]["vocab"]
    model = CausalTransformer(vocab_size=len(vocab) + 8, emb_features=32, num_layers=1, num_heads=2,
                              mlp_features=64, max_seq_len=64)
    objective = LMObjective(model, seq_len=48, loss_role=Role.ASSISTANT)
    logged = trained(objective, train.shuffle(0).repeat().pipe(presets.sft(str(CHAT), seq_len=48, batch=8)),
                     held.pipe(presets.sft(str(CHAT), seq_len=48, batch=8)), metrics=(Perplexity(),))
    finite(logged, "train/loss")
    finite(logged, "val/perplexity")
