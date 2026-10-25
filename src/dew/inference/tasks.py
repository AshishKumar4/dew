"""Tasks that hold a model, its weights and its host processing for repeated generation.

A task holds what a generation needs besides the request itself: the native
model, a frozen variables mapping and, when the source ships one, the host
processor that turns text and media into `ModelInputs`. A call takes the same
typed controls training uses (`Sampling`, `BlockProcess`) and returns the
typed records the sampling kernels produce. `bind` returns the same task over
other weights, so a training loop can draw from a policy snapshot without
holding the trainer's mutable mapping. The task uses the array buffers of the
variables it is given without copying them, so do not mutate, donate or
delete those buffers while the task is using them.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, Protocol, Self

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.core import freeze
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.typing import ArrayLike, DTypeLike

from dew._coordination import agreed
from dew._records import integer, record as named_fields, strings, text as named
from dew.batch import PARAMETER_ROOTS, Variables, thaw
from dew.cache import persist_compilations
from dew.diffusion.block import BlockProcess, CanvasGeneration, refuse_non_denoiser
from dew.diffusion.discrete import MDLM_STEPS, DiscreteProcess, Unmask, refuse_causal
from dew.inference.projections import packed_owned
from dew.interop.sources import SourceReference
from dew.interop.weights import decoding_storage
from dew.nn.inputs import Media, ModelInputs, default_placement, mesh_of, request_key
from dew.nn.protocols import Bounded, CacheCapacity, TokenModel
from dew.objectives.base import UNREAD, Objective
from dew.sampling import decoding, vocabulary
from dew.sampling.decoding import LogitsTransform, Stopping, id_row
from dew.sampling.solvers import Solver
from dew.sampling.strategies import Strategy
from dew.sampling.text import Criteria, Generation, Sampling, Transforms, generate, with_ids_of
from dew.telemetry.profile import region

if TYPE_CHECKING:
    from jax.sharding import Mesh

    from dew.distributed.placement import Layout, MeshSpec
    from dew.quantization import Quantization
    from dew.registry import ModelConfig

Rows = ModelInputs | ArrayLike | Sequence[Sequence[int]]
Request = str | Sequence[str] | Rows

SHAPE_BUCKETS = tuple(1 << exponent for exponent in range(21))
"""The shapes a text request is rounded up to: powers of two.

Every distinct prompt width, budget and continuation count costs its own XLA
compile of several seconds. To share compiles, a call pads the prompt on the
left, behind the attention mask, to the smallest bucket that holds it and is
at least 64. It rounds the budget up to the next power of two and trims the
extra decode steps off the result. The call's cache is the padded prompt plus
the rounded budget, rounded up again, and the call uses it in place of the
model's whole `max_seq_len`. The number of rows is not bucketed. A request
whose buckets would exceed `max_seq_len` keeps its own shapes, so the model
refuses exactly the requests it would refuse without buckets. A request with
media or multi-axis positions also keeps its own shapes.
"""


class Processor(Protocol):
    """Interface for a loaded source's host processor, which prepares requests and decodes tokens.

    Calling it turns text, with optional images, into `ModelInputs`.
    `decode` turns token rows into one string per row, `encode` one text into its ids, with the
    special tokens a sequence carries or without, `spelling` a row of ids into the text the
    tokenizer's own decode gives it, its special tokens spelled out, and `bos_id` is the id a
    sequence starts with, or None.
    """

    def __call__(self, text: str | Sequence[str], *, images: Media | None = None) -> ModelInputs: ...

    def decode(self, tokens: ArrayLike) -> list[str]: ...

    def encode(self, text: str, *, specials: bool) -> list[int]: ...

    def spelling(self, ids: Sequence[int]) -> str: ...

    @property
    def bos_id(self) -> int | None: ...


def prepared_inputs(processor: Processor | None, request: Request, *, images: Media | None) -> ModelInputs:
    """Turn a request into `ModelInputs`, tokenizing text through `processor`.

    Token rows pass through with their ids and row order unchanged; only text
    reaches the processor.
    """
    if isinstance(request, str):
        text = [request]
    elif isinstance(request, Sequence) and request and all(isinstance(entry, str) for entry in request):
        text = [entry for entry in request if isinstance(entry, str)]
    else:
        text = None
    if text is not None:
        if processor is None:
            raise ValueError("text requests need a processor: pass processor= where the task is built, "
                             "as objective.pipeline(state, processor=source.processor) does, or request "
                             "ModelInputs or token rows")
        return processor(text, images=images)
    if images is not None:
        raise ValueError(
            "images travel with text through the processor; prepared rows carry them in ModelInputs"
        )
    if isinstance(request, ModelInputs):
        return ModelInputs.from_value(request)
    if isinstance(request, Sequence):
        rows = [row for row in request if not isinstance(row, str)]
        if len(rows) != len(request):
            raise ValueError("a request cannot mix text prompts and numeric token rows")
        # ModelInputs reads the rows by `dew.batch.token_id`'s law.
        return ModelInputs.from_value(rows)
    # A resident array stays where it is; a global one cannot be fetched.
    return ModelInputs.from_value(request)


def _task_inputs(processor: Processor | None, request: Request, *, images: Media | None,
                 collective: bool, max_new_tokens: int | None, default_tokens: int | None,
                 max_length: int | None, key: int | jax.Array | None) -> tuple[ModelInputs, int, jax.Array]:
    def prepared() -> tuple[ModelInputs, int, jax.Array]:
        """Tokenize the request, size its budget and draw its key."""
        random_key = request_key(key)
        inputs = prepared_inputs(processor, request, images=images)
        return inputs, generation_budget(max_new_tokens, default_tokens, max_length,
                                         inputs.tokens.shape[1]), random_key

    return agreed("inference task input preparation", prepared) if collective else prepared()


def decoded_rows(
    processor: Processor | None, tokens: ArrayLike, lengths: ArrayLike, width: int
) -> tuple[str, ...]:
    """Each row's `lengths` tokens past `width`, as text: what a task's `decode` and a generation's
    `text` both return. Without a processor there is no text to return, and it raises."""
    if processor is None:
        raise ValueError("this generation carries no processor to decode with")
    rows, counts = np.asarray(tokens), np.asarray(lengths)
    return tuple(processor.decode(rows[row:row + 1, width:width + int(counts[row])])[0]
                 for row in range(rows.shape[0]))


def generation_budget(requested: int | None, default: int | None, max_length: int | None,
                      prompt_width: int) -> int:
    """The tokens a request generates: the ones it asks for, else its task's
    default `max_new_tokens`, else what its task's `max_length` leaves past
    the request's `prompt_width` tokens (the prompt rows' width, padding
    included). A task call and a server size a request alike through it."""
    if requested is not None:
        return requested
    if default is not None:
        return default
    if max_length is not None:
        if type(max_length) is not int or max_length < prompt_width:
            raise ValueError("source max_length must be an integer at least as large as the prompt width")
        return max_length - prompt_width
    raise ValueError("max_new_tokens is required; the source declares no default budget")


def shape_bucket(value: int, smallest: int) -> int:
    """Return the smallest shape bucket that holds `value`, never below `smallest`."""
    for bucket in SHAPE_BUCKETS:
        if bucket >= value and bucket >= smallest:
            return bucket
    return value


def cache_ceiling(model: nn.Module) -> int | None:
    """Return the largest cache the model admits, or None where it declares none.

    A `Bounded` decoder declares `max_seq_len`, and a wrapper answers for
    the decoder it holds; `dew.sampling.text._validated` reads the same
    field to refuse a request too large for it.
    """
    declared = model.max_seq_len if isinstance(model, Bounded) else None
    return declared if type(declared) is int else None


def _bucketed(inputs: ModelInputs, budget: int, ceiling: int | None
              ) -> tuple[ModelInputs, int, int | None]:
    """Return the request at bucket shapes: padded inputs, scan trips and cache capacity.

    A capacity of None leaves the request its own shapes and the model its
    own cache. That is what a request too large for the ceiling gets, so it
    is refused where it is refused today, and what one carrying media or
    logical positions gets: a filler slot holds no media feature and no
    multi-axis coordinate. One-axis positions, which every source
    processor's text rows carry, pad like validity: the filler is invalid,
    so its position is never read, and the real tokens keep theirs.
    """
    if ceiling is None or budget < 1:
        return inputs, budget, None
    positions = inputs.token_fields.get("positions")
    if (set(inputs.token_fields) - {"attention_mask", "positions"} or inputs.conditioning
            or (positions is not None and positions.shape != inputs.tokens.shape)):
        return inputs, budget, None
    width = shape_bucket(inputs.tokens.shape[1], 64)
    trips = shape_bucket(budget, 1)
    capacity = shape_bucket(width + trips, 64)
    if capacity > ceiling:
        return inputs, budget, None
    return _padded(inputs, width), trips, capacity


def _padded(inputs: ModelInputs, width: int) -> ModelInputs:
    """Return `inputs` left-padded to `width` slots, with the filler marked invalid."""
    extra = width - inputs.tokens.shape[1]
    if extra < 1:
        return inputs
    valid = inputs.token_fields.get("attention_mask")
    if valid is None:
        valid = jnp.ones(inputs.tokens.shape, bool)
    left = ((0, 0), (extra, 0))
    fields = {name: jnp.pad(value, left) for name, value in inputs.token_fields.items()}
    return replace(inputs, tokens=jnp.pad(inputs.tokens, left),
                   token_fields={**fields, "attention_mask": jnp.pad(valid, left)})


@functools.cache
def cache_sized(model: nn.Module, capacity: int | None) -> nn.Module:
    """Return `model` with a decode cache of `capacity` slots, one model per capacity.

    A `CacheCapacity` model gives the same model at another cache size, a
    wrapper sizing the decoder it holds; any other model keeps its own cache.
    The models are kept because the model is a static argument of the
    compiled generation: one object per capacity is one compile per
    capacity rather than one per call.
    """
    if capacity is None or not isinstance(model, CacheCapacity):
        return model
    return model.with_cache_capacity(capacity)


def _requested(generated: Generation, budget: int, padding: int) -> Generation:
    """Return `generated` cut back to the shapes the caller asked for.

    A bucket pads the prompt on the left and scans past the budget, so the
    filler comes off the front of the rows and the extra trips off the back.
    A row the scan stopped in one of those trips did not stop inside the
    budget: it reaches the caller at the budget's length, unterminated,
    which is what the unbucketed scan reports for it.
    """
    trips = generated.behavior_log_probs.shape[1]
    if not padding and trips == budget:
        return generated
    lengths = generated.lengths
    return replace(generated,
                   tokens=generated.tokens[:, padding:generated.tokens.shape[1] - trips + budget],
                   lengths=jnp.minimum(lengths, budget),
                   terminated=generated.terminated & (lengths <= budget),
                   behavior_log_probs=generated.behavior_log_probs[:, :budget],
                   raw_log_probs=generated.raw_log_probs[:, :budget])


@dataclass(frozen=True)
class SavedRun:
    """One checkpoint of a run, pinned, as every loader of a run reads it (a task's `from_run`,
    `Pretrained.from_run`, `Adapter.from_run`, `dew export`): its inference declaration, not the
    training configuration, the objective that wrote it, and the exact step it is, which the
    weights are read at too (`variables`), so a step saved meanwhile cannot pair one step's
    declaration with another's weights. A loader reconstructs its own components from the record
    (`saved_settings`, `saved_model`)."""

    directory: str
    record: Mapping[str, object]
    step: int

    @classmethod
    def pinned(cls, directory: str, step: int | str | None = None, trust: Sequence[str] = ()) -> SavedRun:
        """The selected checkpoint of the run in `directory`, its latest or `step`. A task loaded
        from it compiles into the persistent cache (`persist_compilations`). `trust` names the
        packages outside Dew whose modules the record may import (`dew.registry.imported`)."""
        from dew.checkpoints import Checkpoints
        from dew.registry import import_trusted, objectives
        persist_compilations()
        with Checkpoints(directory) as checkpoints:
            step = checkpoints.pinned(step)
            record = checkpoints.artifact(step)
        import_trusted(record, trust)
        if record is None:
            raise ValueError("this checkpoint's objective declares no inference record; declare "
                             "Objective.inference_record, or call objective.pipeline(state)")
        record = earlier_shapes(named_fields(record, 'checkpoint artifact'))
        if 'unrecorded' in record:
            raise ValueError(f"this run's checkpoints describe no model to load: {record['unrecorded']}")
        objective = objectives[named(record['objective'], 'objective')]
        declared = frozenset() if objective.settings is None else settings_fields(objective.settings)
        stray = set(record) - {'objective', 'model', UNREAD, *declared}
        if stray:
            raise ValueError(f"this {objective.__name__} run records fields its objective does not write: "
                             f"{sorted(stray)}")
        return cls(str(directory), record, step)

    @property
    def objective(self) -> type[Objective]:
        """The objective class the run trained, which its record names."""
        from dew.registry import objectives

        return objectives[named(self.record['objective'], 'objective')]

    def variables(self, *, ema: bool | None, mesh: MeshSpec | Mesh | None = None,
                  layout: Layout | None = None, param_dtype: DTypeLike | None = None,
                  parameter_roots: tuple[tuple[str, ...], ...] = PARAMETER_ROOTS) -> Variables:
        """The model's variables at this step, as every run loader reads them: an objective whose
        average is its frozen reference gives the trained weights whatever `ema` asks
        (`Objective.ema_is_reference`), and one that nests the model beside other networks gives the
        model's own part (`Objective.model_tree`), trained and frozen collections together; the parts
        the record names unread (`UNREAD`) are neither placed nor read. The other arguments are
        `Checkpoints.variables`'."""
        from dew.checkpoints import Checkpoints

        unread = self.record.get(UNREAD, [])
        if not isinstance(unread, list):
            raise ValueError(f"{UNREAD} must be a list of paths")
        objective = self.objective
        with Checkpoints(self.directory) as checkpoints:
            variables = checkpoints.variables(
                step=self.step, ema=False if objective.ema_is_reference else ema, mesh=mesh, layout=layout,
                param_dtype=param_dtype, parameter_roots=parameter_roots,
                unread=tuple(strings(path, UNREAD) for path in unread))
        return objective.model_tree(variables)


_EARLIER_AUTOENCODERS = {
    "sd_vae": ("dew.nn.autoencoders.sd_vae:StableDiffusionVAE", "dew.nn.autoencoders.kl:AutoencoderKL"),
    "dc_ae": ("dew.nn.autoencoders.dc_ae:DCAutoencoder", "dew.nn.autoencoders.dc_ae:DCAE"),
    "wan": ("dew.nn.autoencoders.wan:WanAutoencoder", "dew.nn.autoencoders.wan:WanVAE"),
    "rae": ("dew.nn.autoencoders.rae:RAEAutoencoder", "dew.nn.autoencoders.rae:RAE"),
    "flux2": ("dew.nn.autoencoders.flux2:Flux2Autoencoder", "dew.nn.autoencoders.kl:AutoencoderKL"),
    "qwen_image": ("dew.nn.autoencoders.qwen_image:QwenImageAutoencoder",
                   "dew.nn.autoencoders.qwen_image:QwenImageVAE"),
}
"""The built-in autoencoders an earlier run recorded by name, each the wrapper's and its module's
import paths the record names now (`earlier_shapes`)."""

def earlier_shapes(record: Mapping[str, object]) -> Mapping[str, object]:
    """A run's record with its components in the shapes Dew writes now, every earlier shape read here
    and nowhere else (`SavedRun.pinned`), so no component keeps two readers: a decision head's
    `{"name"}` record, its class by import path or by class name alone in `dew.decision.head`, a
    built-in autoencoder's by its name with its module's bare fields (`_EARLIER_AUTOENCODERS`), a
    decision layout's by class name in `dew.decision.layout`, each read as the registry's class record,
    a processor's source as its encoded string (ledger 1641), and a Qwen or MiMo projector's earlier
    fields (`_earlier_projectors`, ledger 1622). A record in today's shapes passes as it is. Carried to
    1.0, when it is deleted (ledger 1552)."""
    current = dict(record)
    head = record.get("head")
    if isinstance(head, Mapping) and "name" in head and "class" not in head:
        name = named(head["name"], "head.name")
        current["head"] = {"class": name if ":" in name else f"dew.decision.head:{name}",
                           "fields": named_fields(head.get("fields", {}), "head.fields")}
    autoencoder = record.get("autoencoder")
    if isinstance(autoencoder, Mapping) and "name" in autoencoder and "class" not in autoencoder:
        name = named(autoencoder["name"], "autoencoder.name")
        if name not in _EARLIER_AUTOENCODERS:
            raise ValueError(f"{name!r} names no built-in autoencoder an earlier run recorded")
        wrapper, module = _EARLIER_AUTOENCODERS[name]
        fields = dict(named_fields(autoencoder.get("fields", {}), "autoencoder.fields"))
        current["autoencoder"] = {"class": wrapper,
                                  "fields": {**fields, "model": {"class": module, "fields": fields["model"]}}}
    layout = record.get("layout")
    if isinstance(layout, Mapping) and "name" in layout and "class" not in layout:
        current["layout"] = {"class": f"dew.decision.layout:{named(layout['name'], 'layout.name')}",
                             "fields": named_fields(layout.get("fields", {}), "layout.fields")}
    processor = record.get("processor")
    if isinstance(processor, str):
        # A source's identity was an encoded string before typed component references (ledger 1641).
        from dew.interop.sources import SourceReference
        from dew.registry import to_record

        current["processor"] = to_record(SourceReference.parse(processor), SourceReference)
    if "model" in current:
        current = _earlier_projectors(current)
    return current


def _earlier_projectors(record: Mapping[str, object]) -> Mapping[str, object]:
    """A record's earlier Qwen or MiMo projector fields read as today's (`earlier_shapes`), wherever the
    model nests the projector through another recorded wrapper. Carried until 1.0 (ledger 1622);
    current constructors and source codecs accept only their current schema."""
    def current(value):
        if isinstance(value, Mapping):
            component = {key: current(field) for key, field in value.items()}
            name, fields = component.get("class"), component.get("fields")
            if name in ("dew.nn.vision.qwen35:Qwen35Projector", "dew.nn.vision.mimo:MiMoProjector") \
                    and isinstance(fields, dict) and "vision_width" in fields:
                width, merge = fields.pop("vision_width"), fields.pop("merge_size")
                component["fields"] = {**fields, "hidden_size": width, "spatial_merge_size": merge,
                                     "width": width * merge ** 2,
                                     "bias": name == "dew.nn.vision.qwen35:Qwen35Projector"}
            return component
        if isinstance(value, list | tuple):
            return [current(field) for field in value]
        return value

    return {**record, "model": current(record["model"])}


def settings_fields(settings: type) -> frozenset[str]:
    """The fields a saved run records beside its objective and model: those of its task's `settings`."""
    return frozenset(field.name for field in dataclasses.fields(settings))


def saved_settings[SettingsT](record: Mapping[str, object], settings: type[SettingsT]) -> SettingsT:
    """The task `settings` a saved run's record holds, rebuilt through the record machinery
    (`dew.registry.from_record`), each one it leaves out at its default."""
    from dew.registry import from_record

    return from_record(settings, {name: record[name] for name in settings_fields(settings) if name in record},
                       dtypes=False)


@dataclass(frozen=True, kw_only=True)
class TaskSettings:
    """What every generation run records beside its model: `seq_len`, its training row length,
    `max_new_tokens`, its task's response budget (0 for none), and its host processor's source:
    `tokenizer` names the run tokenizer (`recorded_tokenizer`) and `processor` a checkpoint's own
    processor source (`recorded_processor`); either may be None. Each task's settings add what its
    generation reads."""

    seq_len: int
    max_new_tokens: int
    tokenizer: str | None
    processor: SourceReference | None

    def __post_init__(self) -> None:
        if type(self.max_new_tokens) is not int or self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")


@dataclass(frozen=True, kw_only=True)
class TextSettings(TaskSettings):
    """A causal decoder's run's settings (`TextGeneration.from_settings`): its previews' sampling
    policy beside the common ones."""

    sampling: Sampling


@dataclass(frozen=True, kw_only=True)
class MaskedSettings(TaskSettings):
    """A masked-diffusion run's settings (`MaskedGeneration.from_settings`): the masking process,
    the unmasking solver and its step count, and the end-of-sequence ids a response stops at,
    beside the common ones."""

    process: DiscreteProcess
    solver: Solver
    sampling_steps: int
    eos_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        integer(self.sampling_steps, "sampling_steps")
        if len(self.eos_token_ids):
            object.__setattr__(self, "eos_token_ids", id_row("eos_token_ids", self.eos_token_ids))
        if not isinstance(self.solver, Unmask):
            raise ValueError("a masked run's task unmasks with an Unmask solver")


@dataclass(frozen=True, kw_only=True)
class BlockSettings(TaskSettings):
    """A block-diffusion run's settings (`BlockGeneration.from_settings`): the block process, the
    pad id and the end-of-sequence ids a response stops at beside the common ones."""

    process: BlockProcess
    pad_token_id: int
    eos_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        integer(self.pad_token_id, "pad_token_id")
        if len(self.eos_token_ids):
            object.__setattr__(self, "eos_token_ids", id_row("eos_token_ids", self.eos_token_ids))


def saved_model(record: Mapping[str, object], dtype: DTypeLike | None) -> ModelConfig:
    """Read the run's model record, with `dtype` overriding the computation it saved."""
    from dew.registry import ModelConfig, dtype_name, resolve_dtype

    config = ModelConfig.from_dict(named_fields(record["model"], "model"))
    return config.with_dtype(dtype_name(resolve_dtype(dtype)))


def recorded_tokenizer(processor: Processor | None) -> str | None:
    """The tokenizer name a run's record keeps for `processor`, which
    `_saved_processor` rebuilds through `tokenizer_for`: a run tokenizer's
    own name, byte or Hugging Face. Any other processor records none, and the
    run loads as weights that take ids."""
    from dew.data.text import ByteTokenizer, HFTokenizer
    from dew.inference.pipeline import RunProcessor

    if isinstance(processor, RunProcessor) and isinstance(processor.tokenizer, ByteTokenizer | HFTokenizer):
        return processor.tokenizer.name
    return None


def recorded_processor(processor: Processor | None) -> SourceReference | None:
    """The source a run's record keeps for a checkpoint's own processor
    (`dew.interop.processors.Processor.source`), which `_saved_processor`
    reloads from its files; any other processor records none here."""
    from dew.interop.processors import Processor as SourceProcessor

    return processor.source if isinstance(processor, SourceProcessor) else None


def _saved_processor(settings: TaskSettings, config: ModelConfig) -> Processor | None:
    """Build the run's host processor: the checkpoint's own, reloaded from the
    source it recorded beside `config`'s model, or its run tokenizer."""
    from dew.data.text import tokenizer_for
    from dew.inference.pipeline import RunProcessor
    from dew.interop.pretrained import load_processor

    if settings.processor is not None:
        return load_processor(settings.processor, config.build())
    return None if settings.tokenizer is None else RunProcessor(tokenizer_for(settings.tokenizer))


def resident(model: nn.Module, variables: Variables) -> Variables:
    """`variables` as decoding reads them (`decoding_storage`), each leaf held in host memory, as
    `Pretrained.load(on_host=True)` returns them, cast on the host and then placed on the device
    once: replicated on the mesh the tree's arrays sit on, or committed to this process's default
    device as it is now (`jax.default_device`), so a later call made elsewhere runs where the
    weights are rather than moving them. A jax.Array stays where it is, a pinned-host bank's among
    them. A task's generations and a server's steps read their weights so, where a host leaf
    would be copied to the device again at each call."""
    return held(decoding_storage(model, variables))


def held(variables: Variables) -> Variables:
    """`variables` placed on the device once (`resident`'s placement, its storage as given): each host
    leaf replicated on the mesh the tree's arrays sit on, or committed to this process's default
    device, one leaf at a time (`placed_leaves`), into a tree of its own; a jax.Array stays where it
    is."""
    from dew.distributed.host import placed_leaves

    mesh = mesh_of(variables)
    target = default_placement() if mesh is None else NamedSharding(mesh, P())
    return placed_leaves(variables, lambda _, leaf: jax.device_put(leaf, target)
                         if isinstance(leaf, (np.ndarray, np.generic)) else leaf, in_place=False)


class _NativeTextTask:
    """The lifecycle Dew's three text tasks share, over their own `model`,
    `variables` and `processor`: frozen weights, rebinding, loading a run or
    its Hub copy, and the row decoder their generations carry. Each task
    keeps its own controls, loading checks and device call."""

    model: nn.Module
    variables: Variables
    processor: Processor | None

    def __post_init__(self) -> None:
        # A tree split for training, an adapter's or `dew.batch.freeze`'s,
        # is read whole; a frozen dataclass's field is written through `object.__setattr__`.
        object.__setattr__(self, "variables", freeze(dict(thaw(self.variables))))

    @functools.cached_property
    def placed(self) -> Variables:
        """The weights this task's generations read, placed on the device at the first of them
        (`resident`): `variables` stay as given, so host weights quantize on the host
        (`quantized`)."""
        return resident(self.model, self.variables)

    def bind(self, variables: Variables) -> Self:
        """Return the same task over other weights, such as a policy snapshot."""
        return replace(self, variables=variables)

    if TYPE_CHECKING:
        # Each task is a dataclass, which `bind` replaces a field of.
        __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[object]]]

        # Each task loads its own kind of run; `from_pretrained` reads the Hub copy through it.
        @classmethod
        def from_run(cls, directory: str, *, ema: bool | None = None, step: int | str | None = None,
                     mesh: MeshSpec | Mesh | None = None, layout: Layout | None = None,
                     dtype: DTypeLike | None = None, param_dtype: DTypeLike | None = None,
                     trust: Sequence[str] = ()) -> Self: ...

    @classmethod
    def _loaded[SettingsT: (TextSettings, MaskedSettings, BlockSettings)](
            cls, directory: str, settings: type[SettingsT], checked: Callable[[nn.Module, SettingsT], None],
            *, ema: bool | None, step: int | str | None, mesh: MeshSpec | Mesh | None,
            layout: Layout | None, dtype: DTypeLike | None, param_dtype: DTypeLike | None,
            trust: Sequence[str]) -> tuple[nn.Module, Variables, SettingsT, Processor | None]:
        """A run's model at `dtype`, as `checked` accepts it before any weight loads, its weights
        as its objective restores them, its task settings, and its host processor. The task owns
        the weights it restored: their projections pack here, once, as a source's do at its load
        (`dew.pipeline`), so every route reads the same packed weights."""
        run = SavedRun.pinned(directory, step, trust)
        config = saved_model(run.record, dtype)
        held = saved_settings(run.record, settings)
        model = config.build()
        checked(model, held)
        variables = run.variables(ema=ema, mesh=mesh, layout=layout, param_dtype=param_dtype)
        return model, packed_owned(model, variables), held, _saved_processor(held, config)

    @classmethod
    def from_pretrained(cls, repo_id: str, *, revision: str | None = None,
                        ema: bool | None = None, step: int | str | None = None,
                        mesh: MeshSpec | Mesh | None = None, layout: Layout | None = None,
                        dtype: DTypeLike | None = None,
                        param_dtype: DTypeLike | None = None, trust: Sequence[str] = ()) -> Self:
        """Load a run directory published to the Hugging Face Hub.

        You publish one by uploading the run directory itself with
        `HfApi().upload_folder`. `revision` pins a Hub revision, and the
        other arguments work as in `from_run`.
        """
        from huggingface_hub import snapshot_download

        return cls.from_run(snapshot_download(repo_id=repo_id, revision=revision),
                            ema=ema, step=step, mesh=mesh, layout=layout,
                            dtype=dtype, param_dtype=param_dtype, trust=trust)

    def _decoded[GenerationT: (Generation, CanvasGeneration)](self, generated: GenerationT) -> GenerationT:
        """`generated` carrying the processor's row decoder, or none without a processor."""
        return replace(generated, decoder=None if self.processor is None
                       else functools.partial(decoded_rows, self.processor))


def _canvas_text(processor: Processor | None, generation: CanvasGeneration,
                 trace: str) -> tuple[str, ...]:
    """Decode a canvas generation's rows past their prompt, traced under `trace`.

    A generation with no prompt width names no boundary to decode from, so it
    is refused rather than decoded from the start of the canvas.
    """
    with region(trace):
        if generation.prompt_width is None:
            raise ValueError("this generation has no prompt width")
        rows = generation.host()
        return decoded_rows(processor, rows.tokens, rows.lengths, generation.prompt_width)


@dataclass(frozen=True)
class TextGeneration(_NativeTextTask):
    """Generates next tokens from a decoder, its weights and its processor.

    A call runs the shared cached prefill and decode, and returns the
    `Generation` a training rollout consumes, with the log-probability of
    every drawn action under the policy that drew it and under the raw model.
    When a call passes none, it uses the task's `sampling` policy, its
    `max_new_tokens` budget and its `n` continuations per prompt; a loaded
    source sets all three from its generation config. The `n` continuations
    of a prompt come back as `n` consecutive rows, in prompt order. The
    weights stay where they were placed. On a mesh, the rows are split over
    its batch axes and the results keep that sharding.

    `logits` is the whole logits transform chain, and None means the chain
    `sampling` compiles to. `stopping` holds the criteria that run beside
    the policy's EOS and stop-string criteria, and `strategy` is the device
    loop. A call that passes any of the three replaces the task's value
    whole, so to add a transform you pass
    `logits=task.sampling.transforms() + (mine,)`. A call that passes
    `sampling=` also drops the task's own `logits` chain, because that chain
    was built from the policy the call replaces. Where a call's policy
    leaves the EOS and pad ids None, it takes the task's. So to change one
    control, pass `sampling=replace(task.sampling, repetition_penalty=1.1)`
    or a fresh `Sampling(...)`, and either one still stops where the task
    does. `sampling.stop` strings compile against `processor` once for the
    task's policy, and again on each call whose policy has other stop strings.

    A call pads the request to `SHAPE_BUCKETS` shapes, with a cache sized to
    the buckets, and returns results at the shapes the request asked for. So
    two requests of nearby lengths share one compiled executable, and neither
    pays for the model's whole context.
    """

    model: nn.Module
    variables: Variables
    """The weights generations read. A task loaded from a source (`dew.pipeline`) or a saved run
    (`from_run`) holds them in the serving layout, its projections packed
    (`dew.inference.projections`); one over a live objective's arrays holds those as they are. An
    objective built over them, and an export of them, read the model's own tree
    (`projections.unpacked`)."""
    processor: Processor | None = None
    sampling: Sampling = dataclasses.field(default_factory=Sampling)
    max_new_tokens: int | None = None
    max_length: int | None = None
    n: int = 1
    logits: tuple[LogitsTransform, ...] | None = None
    stopping: tuple[Stopping, ...] = ()
    strategy: Strategy | None = None
    _stops: tuple[Stopping, ...] = dataclasses.field(default=(), init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "_stops", self._stop_criteria(self.sampling.stop))

    def _stop_criteria(self, strings: tuple[str, ...]) -> tuple[Stopping, ...]:
        """Compile stop strings against the processor's vocabulary, sized for the model's head."""
        if not strings:
            return ()
        if not isinstance(
                self.processor, (vocabulary.Referencing, vocabulary.Tokenizing, vocabulary.Vocabulary)):
            raise ValueError("stop strings compile against the task's processor, and this task has none "
                             "that lists its vocabulary; pass processor=")
        return (decoding.stop_strings(self.processor, strings,
                                      self.model.vocab_size if isinstance(self.model, Bounded) else None),)

    def _controls(self, sampling: Sampling | None, logits: Transforms | None, stopping: Criteria | None
                  ) -> tuple[Sampling, Transforms | None, tuple[Stopping, ...]]:
        """The policy, chain and criteria a call runs, the stop strings compiled
        into criteria and the EOS and pad ids filled from the task's policy."""
        policy = self.sampling if sampling is None else with_ids_of(sampling, self.sampling)
        stops = self._stops if policy.stop == self.sampling.stop else self._stop_criteria(policy.stop)
        chain = (self.logits if sampling is None else None) if logits is None else logits
        given = self.stopping if stopping is None else stopping
        # As given: `resolve` tells the criteria that truncate from those that end a row by their kind.
        criteria = (tuple(given) if isinstance(given, Sequence) else (given,)) + stops
        return replace(policy, stop=(), pad_token_id=policy.pad), chain, criteria

    def quantized(self, spec: Quantization, example: Rows = ((0,),)) -> TextGeneration:
        """Return the task with the weights matched by `spec` stored as int8 or fp8 through Qwix.

        Requires `dewml[quantization]`. `example` is one prepared model
        input, token rows or `ModelInputs`, for Qwix's abstract trace; for a
        multimodal model it must include the media fields too. The processor
        and decoding controls stay unchanged. Weights held as host NumPy
        arrays are quantized one kernel at a time on the default device and
        then return to host storage. If one kernel will not fit on that
        device, load the weights onto the task's mesh before quantizing them.
        """
        from dew.quantization import quantize_for_serving

        inputs = prepared_inputs(None, example, images=None)
        model, variables = quantize_for_serving(self.model, self.variables, spec,
                                                inputs.tokens, **inputs.kwargs())
        return replace(self, model=model, variables=variables)

    @classmethod
    def from_run(cls, directory: str, *, ema: bool | None = None, step: int | str | None = None,
                 mesh: MeshSpec | Mesh | None = None, layout: Layout | None = None,
                 dtype: DTypeLike | None = None, param_dtype: DTypeLike | None = None,
                 trust: Sequence[str] = ()) -> TextGeneration:
        """Load the causal language-model run in `directory` as a task.

        The task rebuilds the model that the checkpoint's inference record
        (`Checkpoints.artifact`) declares, the way the recipe built it, over
        the weights of the latest checkpoint
        (or of `step`), and decodes text through the run's own tokenizer.
        The run's preview budget and sampling policy become the task's
        defaults, and a PPO run loads only its policy weights.

        `ema` selects the averaged weights; None reads them when the run
        kept them. When the run's objective keeps its average as a reference
        policy, the task always loads the trained weights. With `mesh`, the
        weights are restored directly onto that mesh under `layout`, the way
        the trainer places them. `dtype` overrides the computation dtype and
        `param_dtype` the parameter storage dtype; None keeps what the
        checkpoint stored. `trust` names the packages outside Dew the run's
        record may import, as `trust_remote_code` does in transformers.
        """
        model, variables, settings, processor = cls._loaded(
            directory, TextSettings, lambda model, settings: None, ema=ema, step=step, mesh=mesh,
            layout=layout, dtype=dtype, param_dtype=param_dtype, trust=trust)
        return cls.from_settings(model, variables, settings, processor)

    @classmethod
    def from_settings(cls, model: nn.Module, variables: Variables, settings: TextSettings,
                      processor: Processor | None) -> TextGeneration:
        """The task `settings` describe over `model` and `variables`, reading text through
        `processor`: the previews' policy and budget become its defaults. A live objective
        (`LMObjective.build_task`) and a saved run (`from_run`) build it alike."""
        return cls(model, variables, processor, sampling=settings.sampling,
                   max_new_tokens=settings.max_new_tokens or None)



    def __call__(self, request: Request, max_new_tokens: int | None = None, *,
                 key: int | jax.Array | None = None, n: int | None = None,
                 sampling: Sampling | None = None, images: Media | None = None,
                 logits: Transforms | None = None, stopping: Criteria | None = None,
                 strategy: Strategy | None = None) -> Generation:
        with region("inference.text"):
            inputs, budget, random_key = _task_inputs(self.processor, request, images=images,
                                          collective=mesh_of(self.variables) is not None,
                                          max_new_tokens=max_new_tokens, default_tokens=self.max_new_tokens,
                                          max_length=self.max_length, key=key)
            shaped, trips, capacity = _bucketed(inputs, budget, cache_ceiling(self.model))
            policy, chain, criteria = self._controls(sampling, logits, stopping)
            generated = generate(cache_sized(self.model, capacity), self.placed, shaped, budget,
                                 key=random_key, sampling=policy, n=self.n if n is None else n, logits=chain,
                                 stopping=criteria, strategy=self.strategy if strategy is None else strategy,
                                 trips=trips)
            padding = shaped.tokens.shape[1] - inputs.tokens.shape[1]
            return self._decoded(_requested(generated, budget, padding))

    def decode(self, generation: Generation) -> tuple[str, ...]:
        """Return each row's valid continuation as text; raises ValueError without a processor."""
        with region("inference.text.decode"):
            rows = generation.host()
            return decoded_rows(self.processor, rows.tokens, rows.lengths, generation.prompt_width)


@dataclass(frozen=True)
class BlockGeneration(_NativeTextTask):
    """Generates block-diffusion canvases from a block denoiser and its weights.

    A call runs prefill, refinement and the commits of clean tokens as one
    device computation. It returns a `CanvasGeneration`, which has no
    autoregressive likelihoods. When a call passes none, it uses the task's
    `process` (the published sampler configuration), its `max_new_tokens`
    budget and its `n` continuations per prompt. The `n` continuations of a
    prompt come back as `n` consecutive rows, in prompt order. The model
    is a `dew.nn.protocols.BlockDenoiser`, as DiffusionGemma is.
    """

    model: nn.Module
    variables: Variables
    process: BlockProcess
    processor: Processor | None = None
    eos_token_ids: tuple[int, ...] = ()
    pad_token_id: int = 0
    max_new_tokens: int | None = None
    max_length: int | None = None
    n: int = 1

    @classmethod
    def from_run(cls, directory: str, *, ema: bool | None = None, step: int | str | None = None,
                 mesh: MeshSpec | Mesh | None = None, layout: Layout | None = None,
                 dtype: DTypeLike | None = None, param_dtype: DTypeLike | None = None,
                 trust: Sequence[str] = ()) -> BlockGeneration:
        """Load the block-diffusion run in `directory` as a task.

        The task rebuilds the block denoiser that the checkpoint's inference
        record declares, over the weights of the latest checkpoint (or of `step`),
        and samples with the run's saved `BlockProcess` over the canvas the
        model declares. The run's preview budget becomes the task's
        `max_new_tokens`, and its pad id the task's `pad_token_id`. A run
        whose model is not a `BlockDenoiser` raises `TypeError` naming the
        operations it lacks.

        The arguments work as in `TextGeneration.from_run`: `ema` selects
        the averaged weights, `mesh` and `layout` place them, and the two
        dtypes override computation and storage.
        """
        model, variables, settings, processor = cls._loaded(
            directory, BlockSettings, lambda model, settings: refuse_non_denoiser(model), ema=ema,
            step=step, mesh=mesh, layout=layout, dtype=dtype, param_dtype=param_dtype, trust=trust)
        return cls.from_settings(model, variables, settings, processor)

    @classmethod
    def from_settings(cls, model: nn.Module, variables: Variables, settings: BlockSettings,
                      processor: Processor | None) -> BlockGeneration:
        """The task `settings` describe over `model` and `variables`, reading text through
        `processor`: their process, pad id, end-of-sequence ids and canvas budget. A live objective
        (`BlockDiffusionObjective.build_task`) and a saved run (`from_run`) build it alike."""
        return cls(model, variables, settings.process, processor, eos_token_ids=settings.eos_token_ids,
                   pad_token_id=settings.pad_token_id, max_new_tokens=settings.max_new_tokens or None)



    def __call__(self, request: Request, max_new_tokens: int | None = None, *,
                 key: int | jax.Array | None = None, n: int | None = None,
                 process: BlockProcess | None = None, images: Media | None = None) -> CanvasGeneration:
        with region("inference.block"):
            inputs, budget, random_key = _task_inputs(self.processor, request, images=images, collective=True,
                                          max_new_tokens=max_new_tokens, default_tokens=self.max_new_tokens,
                                          max_length=self.max_length, key=key)
            generated = (self.process if process is None else process).generate(
                self.model, self.placed, inputs, budget, key=random_key,
                n=self.n if n is None else n,
                eos_token_ids=self.eos_token_ids, pad_token_id=self.pad_token_id)
            return self._decoded(generated)

    def decode(self, generation: CanvasGeneration) -> tuple[str, ...]:
        """Return each row's valid continuation as text; raises ValueError without a processor."""
        return _canvas_text(self.processor, generation, "inference.block.decode")


@dataclass(frozen=True)
class MaskedGeneration(_NativeTextTask):
    """Samples a whole response with native MDLM, holding the prompt fixed.

    It does not reproduce the source-specific remasking recipes of LLaDA or
    Dream. EOS trims the finished response after the bidirectional
    denoising, and does not end that denoising early. Results hold
    refinement counts and no autoregressive action likelihoods. When a call
    passes none, it uses the task's `steps`, its `max_new_tokens` response
    length and its `n` continuations per prompt.
    """

    model: nn.Module
    variables: Variables
    process: DiscreteProcess
    processor: Processor | None = None
    solver: Unmask = dataclasses.field(default_factory=Unmask)
    steps: int = MDLM_STEPS
    eos_token_ids: tuple[int, ...] = ()
    pad_token_id: int = 0
    max_new_tokens: int | None = None
    max_length: int | None = None
    n: int = 1

    @classmethod
    def from_run(cls, directory: str, *, ema: bool | None = None, step: int | str | None = None,
                 mesh: MeshSpec | Mesh | None = None, layout: Layout | None = None,
                 dtype: DTypeLike | None = None, param_dtype: DTypeLike | None = None,
                 trust: Sequence[str] = ()) -> MaskedGeneration:
        """Load the masked-diffusion run in `directory` as a task.

        The task rebuilds the bidirectional model that the checkpoint's inference
        record declares, over the weights of the latest checkpoint (or of `step`),
        and refines responses with MDLM over the run's own mask token, using
        the solver and step count the run saved. The model must score the
        whole row with every position reading the others (`Logits`, and
        `TokenModel` with `causal` False) and name a mask token (`TokenModel.mask_token_id`)
        that matches the saved process, or loading raises `ValueError`.

        The arguments work as in `TextGeneration.from_run`, and the run's
        preview budget becomes the response length a call omits.
        """
        def checked(model: nn.Module, settings: MaskedSettings) -> None:
            refuse_causal(model)
            mask_id = model.mask_token_id if isinstance(model, TokenModel) else None
            if type(mask_id) is not int:
                raise ValueError(f"a saved masked run requires a model that names its mask token "
                                 f"(TokenModel.mask_token_id), and this {type(model).__name__} names none")
            if settings.process.mask_id != mask_id:
                raise ValueError("model and process mask token disagree")

        model, variables, settings, processor = cls._loaded(
            directory, MaskedSettings, checked, ema=ema, step=step, mesh=mesh, layout=layout, dtype=dtype,
            param_dtype=param_dtype, trust=trust)
        return cls.from_settings(model, variables, settings, processor)

    @classmethod
    def from_settings(cls, model: nn.Module, variables: Variables, settings: MaskedSettings,
                      processor: Processor | None) -> MaskedGeneration:
        """The task `settings` describe over `model` and `variables`, reading text through
        `processor`: their process, solver, step count, end-of-sequence ids and response budget. A
        live objective (`MaskedDiffusionObjective.build_task`) and a saved run (`from_run`) build it
        alike."""
        assert isinstance(settings.solver, Unmask)
        return cls(model, variables, settings.process, processor, solver=settings.solver,
                   steps=settings.sampling_steps, eos_token_ids=settings.eos_token_ids,
                   max_new_tokens=settings.max_new_tokens or None)



    def __call__(self, request: Request, max_new_tokens: int | None = None, *,
                 key: int | jax.Array | None = None, n: int | None = None,
                 steps: int | None = None, images: Media | None = None) -> CanvasGeneration:
        with region("inference.masked"):
            inputs, budget, random_key = _task_inputs(self.processor, request, images=images, collective=True,
                max_new_tokens=max_new_tokens, default_tokens=self.max_new_tokens,
                max_length=self.max_length, key=key)
            generated = self.process.generate(self.model, self.placed, inputs, budget, key=random_key,
                solver=self.solver, steps=self.steps if steps is None else steps,
                n=self.n if n is None else n,
                eos_token_ids=self.eos_token_ids, pad_token_id=self.pad_token_id)
            return self._decoded(generated)

    def decode(self, generation: CanvasGeneration) -> tuple[str, ...]:
        """Return each row's valid response as text; raises ValueError without a processor."""
        return _canvas_text(self.processor, generation, "inference.masked.decode")


__all__ = ["SHAPE_BUCKETS", "BlockGeneration", "MaskedGeneration", "Processor", "TextGeneration"]
