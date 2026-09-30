"""The denoising diffusion objective.

Sample a noise level, corrupt, predict, weight. The convention (schedule,
parameterization, weighting) is the `Process`; the sample field and the
conditions are the `InputSpec`; every draw comes from the step's key. The
frozen encoders' weights live in the tree's `encoders` collection, so they
reach the compiled step as arguments and the optimizer never sees them. The
unconditional branch is a pure function of those frozen weights and a fixed
prompt, so the objective encodes it once, when it is built, and the step
reads that: the tower runs over the batch and nothing else, once a step.

Evaluation samples a few images from the validation batch's conditions with
the averaged weights, through the same `sample` inference uses.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from dew.artifacts import ImageGrid, VideoGrid, agreed, collective_host
from dew.diffusion.process import Process, aligned_conditions
from dew.diffusion.schedules import expand
from dew.diffusion.transforms import broadcast_rates
from dew.inputs import InputSpec, unit_range
from dew.nn.autoencoders import AutoEncoder
from dew.nn.autoencoders.api import ModuleAutoEncoder
from dew.nn.autoencoders.kl import AutoencoderKL, posterior_latent
from dew.nn.mp import Uncertainty
from dew.objectives.base import Aux, EMASpec, Mean, Objective, Step, Variables, under
from dew.objectives.diffusion.alignment import ALIGNMENT, REPRESENTATION, Alignment
from dew.objectives.diffusion.end_to_end import AUTOENCODER, LATENT_STATS, EndToEnd
from dew.registry import objectives
from dew.sampling.guidance import CFG, Guidance
from dew.sampling.sample import sample
from dew.sampling.solvers import DDIM, Solver

if TYPE_CHECKING:
    from dew.sampling.pipelines import TextToImage
    from dew.training.state import TrainState

# Samples a validation batch draws, conditioned or not.
VALIDATION_SAMPLES = 4

UNCERTAINTY = "loss_uncertainty"
"""Where the learned loss weighting's head lives in the `params` and
`constants` collections, beside the model's own modules."""


LOSS_HEADS = (UNCERTAINTY, ALIGNMENT, AUTOENCODER)
"""What trains beside the model under `params` and the model never reads."""


def _without_loss_heads(variables: Variables) -> Variables:
    """`variables` without what the model never reads: the uncertainty head,
    the alignment projector, the frozen representation encoder, and an
    autoencoder trained end to end with its latent statistics."""
    return {name: ({key: value for key, value in tree.items() if key not in LOSS_HEADS}
                   if name in ("params", "constants") else tree)
            for name, tree in variables.items() if name not in (REPRESENTATION, LATENT_STATS)}


def check_solver(process, sampler, steps: int) -> None:
    """Trace a solver step on the objective's actual sampling grid."""
    x = jnp.zeros((1, 1), jnp.float32)
    with jax.ensure_compile_time_eval():
        times = process.times(steps)
    state = sampler.init(x, times, process, key=jax.random.PRNGKey(0))
    if times.shape[0] < 2:
        return
    t, t_next = times[:1], times[1:2]
    key = jax.ShapeDtypeStruct((2,), jnp.uint32)
    jax.eval_shape(
        lambda x, key: sampler.step(x, t, t_next, x, x, state, key, process,
                                    lambda x_t, t_: (x_t, x_t)),
        x, key)


class TunedLatents(NamedTuple):
    """An end-to-end step's latents: the autoencoder's raw draw, the batch
    normalized samples the model trains on and the running statistics
    after them, and the autoencoder's regularizer with its terms."""

    raw: jax.Array
    samples: jax.Array
    statistics: Variables
    regularizer: jax.Array
    terms: dict[str, jax.Array]


@objectives("diffusion")
class DiffusionObjective(Objective[Mean]):
    """Denoising diffusion: sample a noise level, corrupt, predict, weight."""

    def __init__(
        self,
        model: nn.Module,
        process: Process,
        inputs: InputSpec,
        *,
        autoencoder: AutoEncoder | None = None,
        unconditional_prob: float = 0.12,
        ema_decay: float | None = 0.999,
        sampler: Solver = DDIM(),
        guidance: Guidance | None = CFG(3.0),
        steps: int = 200,
        pretrained: Variables | None = None,
        uncertainty: int | None = None,
        alignment: Alignment | None = None,
        end_to_end: EndToEnd | None = None,
    ):
        """Build a denoising objective over `model` for the `inputs` field.

        `sampler`, `guidance` and `steps` are how evaluation samples;
        `guidance` None is the plain conditional prediction.

        `uncertainty` learns EDM2's loss weighting (Karras et al. 2024,
        Eq. 21): a head u of the model time, `dew.nn.mp.Uncertainty` with
        this many Fourier channels, and the loss w / e^u ||D - y||^2 + u in
        place of w ||D - y||^2, halved as Dew's L2 loss is. Its minimum over
        u is at the log of the weighted error each noise level leaves, so
        every level contributes about equally. The head trains beside the
        model under `params`, as `UNCERTAINTY`, and a published task drops it.
        None keeps the preset's fixed weighting.

        `alignment` adds REPA's or iREPA's representation alignment
        (`Alignment`) of the model's hidden tokens at one layer with a
        frozen encoder's features of the clean sample: its projector trains
        under `params` as `ALIGNMENT`, the encoder's weights ride frozen
        under `REPRESENTATION`, and a published task drops both.

        `end_to_end` trains the autoencoder with the model, REPA-E's tuning
        (`EndToEnd`): it needs `alignment` and a KL autoencoder, whose
        weights then train under `params` as `AUTOENCODER` and whose
        latents a batch norm normalizes, its running statistics held in
        the `LATENT_STATS` collection. A published task carries the tuned
        autoencoder with those statistics as its latent normalization.
        """
        self.model = model
        self.process = process
        self.inputs = inputs
        self.autoencoder = autoencoder
        self.pretrained = pretrained
        self.uncertainty = None if uncertainty is None else Uncertainty(uncertainty)
        self.alignment = alignment
        self.end_to_end = end_to_end
        if end_to_end is not None and (
                alignment is None or not isinstance(autoencoder, ModuleAutoEncoder)
                or not isinstance(autoencoder.model, AutoencoderKL) or inputs.mask is not None):
            raise ValueError("end-to-end tuning trains a KL autoencoder through REPA's loss; it "
                             "needs `alignment`, a KL autoencoder and no masked-image input")
        if inputs.mask is not None and autoencoder is None:
            raise ValueError("Masked-image conditioning requires an autoencoder")
        # The unconditional branch is a pure function of the frozen towers
        # and each condition's fixed datum, so it is encoded here, once, and
        # not inside every step and every sample. A few hundred kilobytes of
        # host arrays, which a compiled step takes as a constant; the towers
        # themselves stay in the state, for the reason `held_variables` gives.
        self.unconditional_conditions = jax.tree.map(
            np.asarray, self.encode(self.encoder_params()))
        self.unconditional_prob = unconditional_prob
        self.sampler = sampler
        self.guidance = guidance
        self.steps = steps
        self.ema = (None if ema_decay is None else
                    EMASpec(decay=optax.constant_schedule(ema_decay), select=under("params")))
        self.artifact = VideoGrid if len(inputs.sample.shape) == 4 else ImageGrid
        check_solver(process, sampler, steps)
        self._sample = jax.jit(self._sample_impl, static_argnames=("count",))

    def pipeline(self, state: TrainState, *, ema: bool = True) -> TextToImage:
        """The model over the state's published weights as a `TextToImage`
        task, sampling the way this objective's evaluation does."""
        from dew.sampling.pipelines import TextToImage

        return TextToImage.from_objective(self, self._pipeline_weights(state, ema))

    @property
    def latent_shape(self) -> tuple[int, ...]:
        """The per-example shape the model denoises: the sample field's, or
        its latent when an autoencoder sits in front of the model."""
        shape = self.inputs.sample.shape
        return shape if self.autoencoder is None else self.autoencoder.latent_shape(shape)

    def encoder_params(self) -> dict:
        if self.pretrained is not None:
            return dict(self.pretrained["encoders"])
        return {keyword: condition.encoder.params
                for keyword, condition in self.inputs.conditions.items()}

    def encode(self, encoders, tokens: dict | None = None) -> dict:
        """Encode conditions under the supplied parameters; omitted tokens
        select each condition's configured unconditional datum."""
        if tokens is None:
            tokens = {keyword: condition.encoder.tokenize([condition.unconditional])
                      for keyword, condition in self.inputs.conditions.items()}
        return {keyword: condition.encoder.encode(encoders[keyword], tokens[keyword])
                for keyword, condition in self.inputs.conditions.items()}

    def blank_conditions(self, like: dict) -> dict:
        """Cast the stored unconditional conditions to the conditional branch's dtypes.

        `like` is the conditional branch. The values themselves were
        encoded once at construction, so this only changes dtype.
        """
        return jax.tree.map(lambda blank, value: jnp.asarray(blank, value.dtype),
                            self.unconditional_conditions, like)

    def held_variables(self) -> Variables:
        """Every array `init` starts from rather than draws: a whole pretrained
        tree, or the frozen towers.

        A text tower and a VAE are released weights: hundreds of megabytes
        that a nullary trace would compile into the state executable as
        constants. One mapping, so an objective that starts from more than
        the towers extends this and `init` together.
        """
        if self.pretrained is not None:
            return self.pretrained
        held: dict[str, Any] = {"encoders": self.encoder_params()}
        if self.autoencoder is not None:
            held["autoencoder"] = self.autoencoder.params
        if self.alignment is not None:
            held[REPRESENTATION] = self.alignment.variables
        return held

    def init(self, key, variables: Variables | None = None) -> Variables:
        held = self.held_variables() if variables is None else variables
        head_key = jax.random.fold_in(key, 1)
        if "params" in held:
            state: dict[str, Any] = dict(held)
        else:
            conditions = self.encode(held["encoders"])
            if self.inputs.mask is not None:
                conditions = {**conditions, "mask": jnp.zeros((1, *self.latent_shape[:-1], 1)),
                              "masked_image": jnp.zeros((1, *self.latent_shape))}
            drawn = self.model.init(key, jnp.ones((1, *self.latent_shape)), jnp.ones((1,)),
                                    **conditions)
            state = {**drawn, "encoders": held["encoders"]}
            for frozen in ("autoencoder", REPRESENTATION):
                # The frozen weights are state, like the encoders'. They ride in
                # as an argument to the compiled step for the layout to place.
                if frozen in held:
                    state[frozen] = held[frozen]
        if self.uncertainty is not None and UNCERTAINTY not in state["params"]:
            head = self.uncertainty.init(head_key, jnp.ones((1,)))
            for collection, value in head.items():
                state[collection] = {**state.get(collection, {}), UNCERTAINTY: value}
        if self.end_to_end is not None and AUTOENCODER not in state["params"]:
            assert self.autoencoder is not None
            state["params"] = {**state["params"], AUTOENCODER: state.pop("autoencoder")}
            state[LATENT_STATS] = self.end_to_end.initial_statistics(
                self.autoencoder.latent_shift, self.autoencoder.latent_scale, self.autoencoder.latent_channels)
        if self.alignment is not None and ALIGNMENT not in state["params"]:
            state["params"] = {**state["params"], ALIGNMENT: self._projector_init(
                jax.random.fold_in(key, 2), state)}
        return state

    def _projector_init(self, key, state) -> Variables:
        """The projector's parameters, shaped by the model's hidden tokens at
        the aligned layer and the encoder's feature width."""
        alignment = self.alignment
        assert alignment is not None
        conditions = jax.tree.map(lambda value: value[:1], self.unconditional_conditions)

        def hidden(variables):
            _, captured = self.model.apply(
                variables, jnp.ones((1, *self.latent_shape)), jnp.ones((1,)), **conditions,
                capture_intermediates=alignment.captures, mutable=["intermediates"])
            return self._captured(captured)

        tokens = jax.eval_shape(hidden, self.trainable(state))
        features = jax.eval_shape(alignment.targets, state[REPRESENTATION],
                                  jnp.zeros((1, *self.inputs.sample.shape)))
        return alignment.init(key, tokens.shape[-1], tokens.shape[1], features.shape[-1])["params"]

    def _captured(self, captured) -> jax.Array:
        assert self.alignment is not None
        kept = captured.get("intermediates", {}).get(self.alignment.layer)
        if kept is None:
            raise ValueError(f"the model has no submodule {self.alignment.layer!r} to align")
        return kept["__call__"][0]

    def trainable(self, params) -> Variables:
        """Return the model's own collections, without the frozen towers or
        the loss's own heads."""
        return _without_loss_heads({name: value for name, value in params.items()
                                    if name not in ("encoders", "autoencoder")})

    def encoded_conditions(self, params, batch) -> dict:
        """Encode each condition's own batch field under the tree's frozen towers."""
        tokens = {keyword: batch[condition.field]
                  for keyword, condition in self.inputs.conditions.items()}
        return self.encode(params["encoders"], tokens)

    def denoiser(self, params, given, unconditional):
        """Build the process's denoiser over the model's own collections.

        The unconditional branch is passed only when this objective is
        guided; without guidance the sampler never evaluates it.
        """
        return self.process.denoiser(self.model, self.trainable(params), given,
                                     None if self.guidance is None else unconditional)

    def _conditions(self, params, batch, key, *, dropout):
        given = self.encoded_conditions(params, batch)
        # The unconditional prompt is fixed, so its encoding is a constant
        # and the text tower runs over the batch alone, once per step.
        unconditional = self.blank_conditions(given)
        if dropout:
            count = batch[self.inputs.sample.key].shape[0]
            dropped = jax.random.bernoulli(key, self.unconditional_prob, (count,))
            given = jax.tree.map(
                lambda value, blank: jnp.where(
                    expand(dropped, value), jnp.broadcast_to(blank, value.shape), value),
                given, aligned_conditions(given, unconditional))
        if self.inputs.mask is not None:
            from dew.inputs.diffusion import latent_image_conditions
            spatial = latent_image_conditions(self.autoencoder, params["autoencoder"],
                unit_range(batch[self.inputs.sample.key]), batch[self.inputs.mask.key], jax.random.fold_in(key, 1))
            return {**given, **spatial}, {**unconditional, **spatial}
        return given, unconditional

    def _sampling_batch(self, batch):
        fields = [condition.field for condition in self.inputs.conditions.values()]
        if self.inputs.mask is not None:
            fields.extend((self.inputs.sample.key, self.inputs.mask.key))
        return {name: batch[name] for name in fields}

    def loss(self, params, batch, step: Step):
        images = unit_range(batch[self.inputs.sample.key])
        encode_key, drop_key, time_key, noise_key, dropout_key = jax.random.split(step.key, 5)
        conditions, _ = self._conditions(params, batch, drop_key, dropout=True)
        schedule = self.process.schedule
        count = images.shape[0]
        t = schedule.sample_t(time_key, count)
        end_to_end = None
        if self.end_to_end is not None:
            end_to_end = self._end_to_end_latents(params, images, encode_key)
            samples = end_to_end.samples
        elif self.autoencoder is not None:
            samples = self.autoencoder.encode(params["autoencoder"], images, encode_key)
        else:
            samples = images
        noise = jax.random.normal(noise_key, samples.shape, dtype=jnp.float32)

        call = {**conditions, "train": True, "rngs": {"dropout": dropout_key}}
        losses, aligned = self._denoised(params, self.trainable(params), samples, t, noise, call, images)
        weighted = losses * expand(self.process.weight(t), losses)
        if self.uncertainty is not None:
            head = {collection: params[collection][UNCERTAINTY] for collection in ("params", "constants")}
            logvar = expand(self.uncertainty.apply(head, schedule.model_time(t)), losses)
            weighted = weighted * jnp.exp(-logvar) + logvar / 2
        mass = jnp.asarray(losses.size, jnp.promote_types(losses.dtype, jnp.float32))
        total = jnp.sum(weighted)
        metrics: dict[str, jax.Array] = {}
        if self.alignment is not None and aligned is not None:
            # REPA adds proj_coeff times its mean to the denoising mean; the
            # denoising term here is halved, as Dew's L2 is, and so is this.
            metrics["alignment"] = aligned
            total = total + self.alignment.weight / 2 * aligned * mass
        if self.end_to_end is None or end_to_end is None:
            return Mean(total, mass), Aux(metrics=metrics)
        # The autoencoder's update: its regularizer and the alignment of its
        # latent, read through the frozen model and projector in evaluation
        # mode (the batch norm on its running statistics, no condition
        # dropped), as REPA-E's `align_only` pass reads them, on the same
        # times and noise.
        latents = self.end_to_end.normalized(end_to_end.raw, params[LATENT_STATS])
        frozen = jax.lax.stop_gradient(params)
        given, _ = self._conditions(params, batch, drop_key, dropout=False)
        _, through = self._denoised(frozen, self.trainable(frozen), latents, t, noise,
                                    {**given, "train": False}, images)
        assert through is not None
        metrics.update(end_to_end.terms, autoencoder_alignment=through)
        total = total + (end_to_end.regularizer + self.end_to_end.align_weight * through) * mass
        return Mean(total, mass), Aux(metrics=metrics, variables={LATENT_STATS: end_to_end.statistics})

    def _denoised(self, params, variables, samples, t, noise, call, images):
        """The per-element denoising loss at `(t, noise)` and, under
        `alignment`, the REPA loss of the model's hidden tokens."""
        schedule = self.process.schedule
        rates = broadcast_rates(schedule, t, samples)
        noisy, c_in, target = self.process.prediction.forward_diffusion(samples, noise, rates)
        inputs = (variables, noisy * c_in, schedule.model_time(t))
        aligned = None
        if self.alignment is None:
            preds = self.model.apply(*inputs, **call)
        else:
            preds, captured = self.model.apply(*inputs, **call, mutable=["intermediates"],
                                               capture_intermediates=self.alignment.captures)
            aligned = self.alignment.loss({"params": params["params"][ALIGNMENT]}, self._captured(captured),
                                          self.alignment.targets(params[REPRESENTATION], images))
        preds = self.process.prediction.pred_transform(noisy, preds, rates, t)
        return optax.l2_loss(preds, target), aligned

    def _end_to_end_latents(self, params, images, key) -> TunedLatents:
        """The trained autoencoder's posterior draw of `images` and what the
        step reads of it."""
        assert self.end_to_end is not None and isinstance(self.autoencoder, ModuleAutoEncoder)
        module, weights = self.autoencoder.model, {"params": params["params"][AUTOENCODER]}
        moments = module.apply(weights, images, method=module.moments)
        raw = posterior_latent(moments, key)
        reconstruction = module.apply(weights, raw, method=module.decode)
        regularizer, terms = self.end_to_end.regularizer(images, reconstruction, moments)
        samples, statistics = self.end_to_end.batch_normalized(jax.lax.stop_gradient(raw),
                                                               params[LATENT_STATS])
        return TunedLatents(raw, samples, statistics, regularizer, terms)

    def published_autoencoder(self, variables: Variables) -> tuple[AutoEncoder | None, Variables]:
        """The autoencoder a task over `variables` decodes with, and its
        weights under `autoencoder`: the frozen one, or under `end_to_end`
        the tuned one, its latents normalized by the running statistics."""
        if self.end_to_end is None or self.autoencoder is None:
            return self.autoencoder, variables
        tuned = copy.copy(self.autoencoder)
        statistics = variables[LATENT_STATS]
        tuned.latent_shift = statistics["mean"]
        tuned.latent_scale = 1.0 / jnp.sqrt(statistics["var"] + self.end_to_end.epsilon)
        tuned.params = variables["params"][AUTOENCODER]
        return tuned, {**variables, "autoencoder": tuned.params}

    def _sample_impl(self, params, batch, key, *, count: int):
        given, unconditional = self._conditions(params, batch, key, dropout=False)
        denoise = self.denoiser(params, given, unconditional)
        noise_key, sample_key = jax.random.split(key)
        x_T = self.process.noise(noise_key, (count, *self.latent_shape))
        samples = sample(denoise, x_T, self.steps, solver=self.sampler,
                         guidance=self.guidance, key=sample_key)
        autoencoder, params = self.published_autoencoder(params)
        if autoencoder is not None:
            samples = autoencoder.decode(params["autoencoder"], samples)
        return jnp.clip(samples, -1.0, 1.0)

    def evaluate(self, params, batch, step: Step):
        """One generated sample for every real row, without display decoding."""
        params = params if step.ema is None else step.ema
        count = batch[self.inputs.sample.key].shape[0]
        samples = self._sample(params, self._sampling_batch(batch), step.key, count=count)
        assert self.artifact is not None
        return self.artifact(samples)

    def preview(self, params, batch, step: Step, *, scored=None):
        """A separate small draw for display, with root-only caption decoding."""
        def setup():
            weights = params if step.ema is None else step.ema
            count = min(VALIDATION_SAMPLES, batch[self.inputs.sample.key].shape[0])
            return weights, count, self._sampling_batch(batch)

        def generate():
            selected = jax.tree.map(lambda value: value[:count], raw_batch)
            return (self._sample(weights, selected, step.key, count=count),
                    {keyword: selected[condition.field]
                     for keyword, condition in self.inputs.conditions.items()})

        weights, count, raw_batch = agreed("diffusion preview setup", setup)
        samples, tokens = agreed("diffusion preview generation", generate)
        samples, tokens = collective_host((samples, tokens), phase="diffusion preview")
        if jax.process_index() != 0:
            return None
        captions = ()
        for keyword, condition in self.inputs.conditions.items():
            captions = condition.encoder.captions(tokens[keyword])
            if captions:
                break
        assert self.artifact is not None
        return self.artifact(samples, captions)
