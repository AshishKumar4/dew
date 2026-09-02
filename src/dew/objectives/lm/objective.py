"""The autoregressive language modelling objective.

Next-token prediction over packed token ids. A batch row holds `seq_len + 1`
ids, the model sees all but the last, and the targets are the same row shifted
by one; the causal mask lives in the backbone, so nothing here has to know how
the model keeps the future out of a prediction.

Cross entropy is computed in float32 even when the model runs in bfloat16: a
bf16 logsumexp over a large vocabulary loses enough precision to move the loss
and, through it, the gradient. Padding is excluded only when the run names the
pad id, because packed token files have no padding and masking out a real id
would quietly drop those tokens from the average.

Validation reports the teacher-forced cross entropy, which the perplexity
metric scores, and the text the model writes from a fixed prompt, which is the
only part of a training curve a human can read.
"""

from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew.objectives.base import EMASpec, Objective, shape_and_dtype

TEXT_KEY = "text"
"""Batch key the token pipeline packs `[B, seq_len + 1]` int32 ids under."""


def prompt_batch(prompt) -> jax.Array:
    """`[B, P]` int32 ids from one prompt, or several of the same length."""
    try:
        ids = np.asarray(prompt)
    except ValueError as ragged:
        raise ValueError("sample prompts have to be of equal length") from ragged
    if ids.ndim == 1:
        ids = ids[None]
    if ids.ndim != 2 or ids.shape[1] == 0:
        raise ValueError(f"a sample prompt is non-empty [P] or [B, P] ids, got {ids.shape}")
    return jnp.asarray(ids, jnp.int32)


def validation_params(val_state):
    """The EMA copy when the trainer keeps one, the live params otherwise."""
    ema_params = getattr(val_state, "ema_params", None)
    return val_state.params if ema_params is None else ema_params


class LMObjective(Objective):
    """Shifted cross entropy, with generated text as its validation artifact."""

    tag = "lm"

    def __init__(
        self,
        model,
        seq_len: int,
        *,
        vocab_size: int,
        ema_decay: float = 0.999,
        pad_id: Optional[int] = None,
        samples: Optional[Dict[str, Any]] = None,
    ):
        """`samples` configures the text logged at validation: a `prompt` of
        int32 ids (one, or several of equal length), `max_new_tokens`, a
        `temperature` (0 is greedy), an optional `top_k`, and the `decode`
        that turns ids back into a string. Unset logs no text."""
        self.model = model
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.samples = samples
        self.ema = EMASpec(decay=lambda step: ema_decay)

    @property
    def input_shapes(self) -> Dict[str, Any]:
        """One int32 token sequence, which is all the model is fed."""
        return {"tokens": ((self.seq_len,), jnp.int32)}

    def init_params(self, rng):
        shape, dtype = shape_and_dtype(self.input_shapes["tokens"])
        return self.model.init(rng, jnp.zeros((1, *shape), dtype))

    def shifted_cross_entropy(self, params, tokens, train: bool = False, rngs=None,
                              segment_ids=None, positions=None):
        """Next-token cross entropy over a `[B, seq_len + 1]` batch, and its telemetry.

        A packed batch carries `segment_ids` for the same rows: the last token
        of a document does not predict the first of the next one, so that
        transition is dropped from the loss and the accuracy, and the model
        reads the per-document `positions` for its rotary angles.
        """
        if tokens.shape[-1] != self.seq_len + 1:
            raise ValueError(
                f"a {self.seq_len}-token context needs {self.seq_len + 1} ids per row "
                f"so the targets can be the shifted input, got {tokens.shape[-1]}")
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        logits = self.model.apply(params, inputs, train=train, rngs=rngs,
                                  positions=None if positions is None else positions[:, :-1],
                                  segment_ids=None if segment_ids is None else segment_ids[:, :-1])
        logits = logits.astype(jnp.float32)
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        correct = (jnp.argmax(logits, axis=-1) == targets).astype(losses.dtype)

        weights = (jnp.ones_like(losses) if self.pad_id is None
                   else (targets != self.pad_id).astype(losses.dtype))
        if segment_ids is not None:
            # A target counts only inside a document: the first token of the
            # next packed document, the padding after the last one (segment 0,
            # which the seg==seg comparison alone would keep), and every
            # cross-boundary transition drop out of loss and accuracy alike.
            weights = weights * (
                (segment_ids[:, 1:] == segment_ids[:, :-1])
                & (segment_ids[:, 1:] != 0)).astype(losses.dtype)
        # A batch that is entirely padding would divide by zero and take the
        # whole run down with a nan.
        counted = jnp.maximum(jnp.sum(weights), 1.0)
        ce = jnp.sum(losses * weights) / counted
        return ce, {
            "ce": ce,
            "perplexity": jnp.exp(ce),
            "token_accuracy": jnp.sum(correct * weights) / counted,
        }

    def loss(self, params, ema_params, batch, rng, step):
        tokens = jnp.asarray(batch[TEXT_KEY], jnp.int32)
        segment_ids = batch.get("text_segment_ids")
        segment_ids = None if segment_ids is None else jnp.asarray(segment_ids, jnp.int32)
        positions = batch.get("text_positions")
        positions = None if positions is None else jnp.asarray(positions, jnp.int32)
        return self.shifted_cross_entropy(params, tokens, train=True, rngs={"dropout": rng},
                                          segment_ids=segment_ids, positions=positions)

    def make_validation_step(self, **kwargs) -> Callable[[Any, Any], Dict[str, Any]]:
        """Teacher-forced cross entropy, plus the sampled text when asked for.

        Both read the EMA copy, so the perplexity and the samples describe the
        same weights.
        """
        samples = self.samples
        if samples is not None:
            prompt = prompt_batch(samples["prompt"])
            max_new_tokens = int(samples["max_new_tokens"])
            temperature = float(samples.get("temperature", 1.0))
            top_k = samples.get("top_k")
            # Deferred: a run that logs no text pulls in no sampler.
            from dew.sampling.text import generate

        teacher_forced_ce = jax.jit(
            lambda params, tokens, segment_ids=None, positions=None:
                self.shifted_cross_entropy(params, tokens,
                                           segment_ids=segment_ids, positions=positions)[0])

        def validate(val_state, batch):
            params = validation_params(val_state)
            tokens = jnp.asarray(batch[TEXT_KEY], jnp.int32)
            segment_ids = batch.get("text_segment_ids")
            segment_ids = None if segment_ids is None else jnp.asarray(segment_ids, jnp.int32)
            positions = batch.get("text_positions")
            positions = None if positions is None else jnp.asarray(positions, jnp.int32)
            artifacts = {"ce": teacher_forced_ce(params, tokens, segment_ids, positions)}
            if samples is not None:
                artifacts["tokens"] = generate(
                    self.model, params, prompt, max_new_tokens,
                    # Folded so successive validations write different text
                    # from the same prompt.
                    rng=jax.random.fold_in(val_state.rngs, val_state.step),
                    temperature=temperature, top_k=top_k,
                )
            return artifacts

        return validate

    def log_validation_artifacts(self, wandb, artifacts, step: int):
        tokens = artifacts.get("tokens")
        if tokens is None:
            return

        from wandb import Table
        decode = self.samples.get("decode", lambda ids: str(ids))
        rows = [[index, decode(row.tolist())]
                for index, row in enumerate(np.asarray(tokens))]
        wandb.log({"val/samples": Table(columns=["sample", "text"], data=rows)}, step=step)
