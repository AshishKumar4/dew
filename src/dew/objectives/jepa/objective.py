"""The I-JEPA / V-JEPA objective.

Predict the representation of masked target blocks from the representation of
the visible context, in latent space. Three moving parts:

  - the context encoder sees only the context tokens and is trained;
  - the target encoder sees the whole image and is the EMA of the context
    encoder. It has no parameter subtree of its own and lives in the
    trainer's EMA copy, and its branch is stop_gradient'd;
  - the predictor maps context embeddings plus target positions to the target
    representations.

Targets are layer-normalized (no learned affine) before the L2 loss, which
keeps the scale of the prediction problem fixed as the encoder drifts. The
loss is the paper's L2. The reference implementation uses smooth-L1. The LN
already bounds the target scale, so L2 has no outliers to absorb and stays
directly comparable to the paper.

The characteristic failure is silent. Both encoders can agree on a constant
and the loss goes to zero. representation_health is reported on every step so
that collapse shows in the training curves, before a probe run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from dew.artifacts import Representations
from dew.inputs import Field, InputSpec, unit_range
from dew.nn.sharding import explicit_spec, whole_spec
from dew.objectives.base import (
    Aux,
    EMASpec,
    Objective,
    ProgramModule,
    Ratio,
    Shown,
    Step,
    Variables,
    joined,
    model_rngs,
    part,
    thaw,
    under,
)

from .masking import MultiBlockMask

CONTEXT_ENCODER = "context_encoder"
PREDICTOR = "predictor"
LABEL_KEY = "label"


def representation_health(z) -> dict[str, jax.Array]:
    """Return two collapse measures for pooled embeddings `[B, D]`, as metrics.

    `repr_std` is the per-dimension standard deviation across the batch. It
    goes to zero exactly when the encoder stops distinguishing inputs.
    `repr_cov_offdiag` is the RMS magnitude of the off-diagonal covariance,
    which rises when the dimensions become redundant (dimensional collapse)
    while `repr_std` holds.

    Both are computed in fp32, so a run's compute dtype does not set the
    noise floor of the drift, and bf16 and fp32 runs read off the same curves.
    """
    batch_size, dim = z.shape
    z = z.astype(jnp.float32)
    centered = z - jnp.mean(z, axis=0, keepdims=True)
    cov = jnp.matmul(centered.T, centered, out_sharding=whole_spec(centered, 2)) / max(batch_size - 1, 1)
    off_diagonal = cov * (1.0 - jnp.eye(dim, dtype=cov.dtype))
    return {
        "repr_std": jnp.mean(jnp.std(z, axis=0)),
        "repr_cov_offdiag": jnp.sqrt(jnp.sum(off_diagonal ** 2) / max(dim * (dim - 1), 1)),
    }


def normalize_targets(x, epsilon: float = 1e-6):
    """Layer-normalize `x` over its feature axis, with no learned affine.

    The objective applies it to the target encoder's output, so the
    prediction problem keeps a fixed scale as the encoder drifts, and
    shrinking the representation does not lower the loss.
    """
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(variance + epsilon)


@runtime_checkable
class _Scanned(Protocol):
    """A module that declares the order it sequences an image's or a clip's
    tokens in, as Dew's JEPA encoder and predictor do."""

    @property
    def scan_order(self) -> str: ...


class JepaObjective(Objective[Ratio]):
    """Trains a JEPA encoder and predictor over images (B,H,W,C) or video (B,T,H,W,C).

    Evaluation returns the pooled target-encoder embeddings of a batch with
    its labels, which the probe metrics score. `momentum` is the target
    encoder's EMA decay, which rises linearly from its first value to its
    second over `momentum_steps` updates.

    `encoder_variables` and `predictor_variables` are the trees each module
    starts from, as `model.init`, `Pretrained.load` or `LoRA.apply` return
    them; a module whose tree is None is initialized from the key. Every
    collection of each tree is kept under the module's name. For an adapted
    encoder, the factors train under `params` and the base stays under
    `frozen`, and the target encoder is the trainer's average of the trained
    leaves over that same frozen base.
    """
    artifact = Representations
    # A collapsing encoder's spread falls to zero; a redundant one's
    # off-diagonal covariance rises.
    shown: Mapping[str, Shown] = {
        "repr_std": Shown(better="higher"),
        "repr_cov_offdiag": Shown(better="lower"),
    }

    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        mask: MultiBlockMask,
        sample: Field,
        momentum: tuple[float, float] = (0.996, 1.0),
        momentum_steps: int = 100_000,
        label_key: str = LABEL_KEY,
        encoder_variables: Variables | None = None,
        predictor_variables: Variables | None = None,
    ):
        # A module that declares the order it sequences tokens in reads the
        # mask's indices in that sequence.
        for role, module in (("encoder", encoder), ("predictor", predictor)):
            if isinstance(module, _Scanned) and module.scan_order != mask.scan_order:
                raise ValueError(f"the {role} scans in {module.scan_order!r} order and the mask in "
                                 f"{mask.scan_order!r}; they must share one scan order")
        self.encoder = encoder
        self.predictor = predictor
        self.mask = mask
        self.sample = sample
        self.label_key = label_key
        self.encoder_variables = encoder_variables
        self.predictor_variables = predictor_variables
        self.is_video = len(sample.shape) == 4
        self.inputs = InputSpec(sample=sample)
        self.ema = EMASpec(
            decay=optax.linear_schedule(momentum[0], momentum[1], momentum_steps),
            select=under("params", CONTEXT_ENCODER),
        )

    def program_key(self) -> tuple[ProgramModule, ...]:
        """The encoder, which also runs as the target, then the predictor."""
        return (ProgramModule(self.encoder, None, trained=True),
                ProgramModule(self.predictor, None, trained=True))

    def substitute(self, modules: Sequence[nn.Module]) -> None:
        self.encoder, self.predictor = modules

    def held_variables(self) -> Variables | None:
        """The modules' given starting trees, by module, or None when both are drawn."""
        given = {name: tree for name, tree in ((CONTEXT_ENCODER, self.encoder_variables),
                                               (PREDICTOR, self.predictor_variables)) if tree is not None}
        return given or None

    def init(self, key, variables: Variables | None = None):
        given = dict(self.held_variables() or {}) if variables is None else dict(variables)
        encoder_key, predictor_key = jax.random.split(key)
        sample = jnp.ones((1, *self.sample.shape))
        context_idx = jnp.arange(self.mask.num_context, dtype=jnp.int32)[None]
        target_idx = jnp.arange(self.mask.block_area, dtype=jnp.int32)[None]

        encoder = given.get(CONTEXT_ENCODER)
        if encoder is None:
            encoder = self.encoder.init(encoder_key, sample, context_idx)
        predictor = given.get(PREDICTOR)
        if predictor is None:
            context = self.encoder.apply(thaw(encoder), sample, context_idx)
            predictor = self.predictor.init(predictor_key, context, context_idx, target_idx)
        return joined({CONTEXT_ENCODER: encoder, PREDICTOR: predictor})

    def encode(self, encoder_variables, samples, token_idx=None, train=False, rngs=None) -> jax.Array:
        """The encoder over `encoder_variables`, its own tree in every collection,
        a frozen split merged back."""
        features = self.encoder.apply(thaw(encoder_variables), samples, token_idx, train=train, rngs=rngs)
        # `mutable` is unset, so apply returns the output alone, not a pair.
        assert not isinstance(features, tuple)
        return features

    def _target_variables(self, step: Step):
        """The target encoder's variables: the EMA copy of the context
        encoder's trained leaves over whatever the encoder keeps frozen.

        The objective declares an EMASpec, so the trainer always hands it an
        EMA tree. Without one there is no target branch to run.
        """
        if step.ema is None:
            raise ValueError("the JEPA target branch needs the trainer's EMA variables")
        return part(step.ema, CONTEXT_ENCODER)

    def loss(self, variables, batch, step: Step):
        samples = unit_range(batch[self.sample.key])
        batch_size = samples.shape[0]
        mask_key, dropout_key = jax.random.split(step.key)
        context_idx, target_idx = self.mask.sample(mask_key, batch_size)
        num_targets = self.mask.num_targets

        # The target branch reads the whole view through the EMA encoder, without gradients.
        full = normalize_targets(self.encode(self._target_variables(step), samples))
        # [B, (T,) S, F] -> [B, M, (T,) n_tgt, F]
        frame_axis = (1,) if self.is_video else ()
        gather_idx = target_idx.reshape(batch_size, num_targets, *frame_axis, -1, 1)
        targets = jax.lax.stop_gradient(
            jnp.take_along_axis(full[:, None], gather_idx, axis=-2))

        context = self.encode(
            part(variables, CONTEXT_ENCODER), samples, context_idx,
            train=step.training, rngs=model_rngs(dropout_key, training=step.training))

        # Each target block is predicted from the same context. Fold the block
        # axis into the batch so one predictor call covers all M of them
        repeated = jnp.repeat(context, num_targets, axis=0, out_sharding=explicit_spec(context))
        predictions = self.predictor.apply(
            thaw(part(variables, PREDICTOR)),
            repeated,
            jnp.repeat(context_idx, num_targets, axis=0),
            target_idx.reshape(batch_size * num_targets, -1),
            train=step.training, rngs=model_rngs(dropout_key, training=step.training),
        )
        # `mutable` is unset, so apply returns the output alone, not a pair.
        assert not isinstance(predictions, tuple)
        predictions = predictions.reshape(targets.shape)

        squared = (predictions.astype(jnp.float32) - targets.astype(jnp.float32)) ** 2
        loss = self.row_mean(squared, batch)
        pooled = jnp.mean(full, axis=tuple(range(1, full.ndim - 1)))
        return loss, Aux(representation_health(pooled))

    def evaluate(self, params, batch, step: Step):
        """Return the pooled embeddings of the target encoder, the EMA copy, with the batch labels."""
        features = self._embed(self._target_variables(step), batch[self.sample.key])
        return Representations(features=features, labels=jnp.asarray(batch[self.label_key]))

    @property
    def _embed(self):
        def embed(encoder_variables, pixels):
            features = self.encode(encoder_variables, unit_range(pixels))
            return jnp.mean(features, axis=tuple(range(1, features.ndim - 1)))

        return self._compiled_program('embed', embed)
