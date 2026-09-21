"""One trained source-format export per routed decoder family, measured.

The reproduction command behind the numbers in tests/test_decoder_export.py.
Each case loads a committed tiny checkpoint through `load_pretrained`, runs
one real `Trainer` step of plain SGD under `LMObjective`, writes the trained
weights back into the source's own tensor names with `Pretrained.save`, and
reads the export back twice: with `load_pretrained` for the parameter tree
and the logits, and with transformers 5.16.1 for the reference logits on the
same ids. It prints what moved, what the export holds and where the two
implementations disagree.

    JAX_PLATFORMS=cpu PYTHONPATH=src python tools/decoder_export_reference.py

Nothing is downloaded and nothing is written outside the temporary
directory it works in. The tests import `round_trip` from here, so the
numbers below and the assertions come from one pipeline.
"""

from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from torch.overrides import TorchFunctionMode

from dew.interop import load_pretrained
from dew.interop.pretrained import Pretrained
from dew.objectives.base import Variables
from dew.objectives.lm import LMObjective

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "hf"
RATE = 5e-2
SEED = 3


@dataclass(frozen=True)
class Case:
    """A family's fixture, the loss terms its checkpoint supports and the
    class the reference reads it with.

    `balance_rate` needs a mixture that keeps a routing bias, so only the
    sigmoid-scored families carry one; `mtp_weight` needs a configured
    prediction depth, which GLM's checkpoint is the one to ship.

    `reference_class` names the transformers class for a released
    model_type `AutoModelForCausalLM` reads no mapping for. Kimi K2 and
    Kimi K2.5 are the families that ship that way. K2's `auto_map` points
    at its own copy of DeepSeek V3's modeling code, and transformers says
    the same thing where it does read the name:
    `Kimi_K25Config.__post_init__` turns a text config of model_type
    `kimi_k2` into a `deepseek_v3` one
    (models/kimi_k25/configuration_kimi_k25.py:80-92), so
    `DeepseekV3ForCausalLM` is upstream's implementation of these weights.
    K2.5's class exists but is registered for image-text-to-text, not
    causal LM, so the case names `Kimi_K25ForConditionalGeneration` and the
    export runs on input_ids alone, which is its text half.
    `AutoModelForCausalLM` reads every other family here off its model_type.

    `rate` is the step this case trains with, `RATE` unless it says
    otherwise. Torch and JAX can choose different valid indexer sets at
    equal scores. The V4 MTP-enabled step preserves the observed tied-score
    failure at 5e-3; a smaller step is not evidence resolving that failure.
    """

    name: str
    fixture: str
    balance_rate: float | None = None
    mtp_weight: float | None = None
    reference_class: str | None = None
    reference_module: str = 'transformers'
    conversion_type: str | None = None
    rate: float = RATE


CASES = (
    Case("mixtral", "mixtral-tiny"),
    Case("qwen3_moe", "qwen3-moe-tiny"),
    Case("glm4_moe", "glm4-moe-tiny", balance_rate=1e-2, mtp_weight=0.3),
    Case("glm_moe_dsa", "glm-moe-dsa-tiny", balance_rate=1e-2, mtp_weight=0.3),
    Case("deepseek_v2", "deepseek-v2-tiny"),
    Case("deepseek_v3", "deepseek-v3-tiny", balance_rate=1e-2),
    Case("deepseek_v32", "deepseek-v32-tiny", balance_rate=1e-2),
    Case("deepseek_v4", "deepseek-v4-tiny", balance_rate=1e-2, mtp_weight=0.3, rate=5e-3),
    Case("kimi_k2", "kimi-k2-tiny", balance_rate=1e-2,
         reference_class="DeepseekV3ForCausalLM"),
    Case("kimi_k25", "kimi-k25-tiny", balance_rate=1e-2,
         reference_class="Kimi_K25ForConditionalGeneration"),
    Case("llama4_text", "llama4-tiny"),
    Case("olmo3", "olmo3-yarn-tiny"),
    Case("qwen3_next", "qwen3-next-tiny", mtp_weight=0.3),
    Case("glm5_next", "glm5-next-tiny", balance_rate=1e-2, mtp_weight=0.3,
         reference_class="Glm5NextTextForCausalLM", reference_module="tools.hf_reference_b",
         conversion_type="glm5_next"),
)


@dataclass(frozen=True)
class RoundTrip:
    """One family trained, exported and read back by both implementations."""

    case: Case
    source: Pretrained
    trained: Variables
    export: Path
    ids: np.ndarray
    ours: np.ndarray
    reloaded: Pretrained
    theirs: np.ndarray
    source_tensors: dict[str, np.ndarray] = field(repr=False)
    exported_tensors: dict[str, np.ndarray] = field(repr=False)


def source_tensors(directory: Path) -> dict[str, np.ndarray]:
    """The checkpoint's own tensor table, as the loader reads it."""
    from dew.interop.hf_decoders import _load_shards

    return _load_shards(directory)


def logits(source: Pretrained, variables: Variables, ids: np.ndarray) -> np.ndarray:
    return np.asarray(source.model.apply(variables, jnp.asarray(ids, jnp.int32)), np.float32)


def train(case: Case, source: Pretrained, ids: np.ndarray):
    """One real `Trainer` step of plain SGD over the loaded checkpoint.

    The rows are the fixture's own ids repeated to fill one batch per
    device, so the step runs on whatever devices the process has.
    """
    import grain.python as pygrain

    from dew.data.dataset import Dataset
    from dew.training import Trainer

    objective = LMObjective(source.model, seq_len=int(ids.shape[1]) - 1, ema_decay=None,
                            pretrained=source.variables, balance_rate=case.balance_rate,
                            mtp_weight=case.mtp_weight)
    count = math.lcm(int(ids.shape[0]), jax.device_count())
    rows = np.concatenate([ids] * (count // int(ids.shape[0])), axis=0)
    entries = [{"text": rows[row]} for row in range(count)]
    stream = (pygrain.MapDataset.source(entries).repeat().to_iter_dataset()
              .batch(count, drop_remainder=True))
    data = Dataset(train=lambda: iter(stream), val=None, records=count, batch=count)
    state = Trainer(objective, optax.sgd(case.rate), key=jax.random.key(SEED)).fit(
        data, steps=1, log_every=1)
    return state


def _text_half_prefixes(model) -> tuple[str, ...]:
    """Where a wrapper reference keeps the decoder a text-only claim covers.

    Kimi K2.5's reference is `Kimi_K25ForConditionalGeneration`, run on
    input_ids alone: its vision tower and projector see no input and hold
    no gradient. That is not missing participation, it is the half this
    qualifies nothing about, and the source retains those tensors byte for
    byte instead of binding them. A plain causal LM has no such half and
    names no prefix, so every parameter of it stays in scope.
    """
    inner = getattr(getattr(model, 'model', None), 'language_model', None)
    return () if inner is None else ('model.language_model.', 'lm_head.')


def _prediction_prefixes(config) -> tuple[str, ...]:
    """The declared prediction tensors omitted by the upstream trunk class."""
    if config.model_type in ('glm4_moe', 'glm_moe_dsa', 'glm5_next_text'):
        return tuple(f'model.layers.{config.num_hidden_layers + depth}.'
                     for depth in range(getattr(config, 'num_nextn_predict_layers', 0)))
    return ('mtp.',) if config.model_type == 'qwen3_next' else ()


class StableSelectorTopK(TorchFunctionMode):
    def __torch_function__(self, func: Callable[..., object], types: Sequence[type],
                           args: tuple[object, ...] = (), kwargs: dict[str, object] | None = None) -> object:
        keywords = {} if kwargs is None else kwargs
        if func not in (torch.topk, torch.Tensor.topk):
            return func(*args, **keywords)
        operand = args[0]
        count = args[1] if len(args) > 1 else keywords['k']
        axis = args[2] if len(args) > 2 else keywords.get('dim', -1)
        largest = args[3] if len(args) > 3 else keywords.get('largest', True)
        if not isinstance(operand, torch.Tensor) or type(count) is not int or type(axis) is not int:
            raise TypeError('Selector top-k requires a tensor and integer count/axis')
        if largest is not True or keywords.get('out') is not None:
            raise ValueError('Selector qualification requires descending top-k without out buffers')
        indices = torch.argsort(operand, dim=axis, descending=True, stable=True).narrow(axis, 0, count)
        values = torch.gather(operand, axis, indices)
        return torch.return_types.topk((values, indices))


def _stable_forward(original: Callable[..., object]) -> Callable[..., object]:
    def forward(*args: object, **kwargs: object) -> object:
        with StableSelectorTopK():
            return original(*args, **kwargs)
    return forward


@contextmanager
def stable_selector_ties(model: torch.nn.Module, selector_type: type[torch.nn.Module]) -> Iterator[int]:
    """Temporarily scope stable top-k to one concrete reference indexer class."""
    held = [(module, module.forward) for module in model.modules() if isinstance(module, selector_type)]
    if not held:
        raise ValueError('No matching indexers; refusing to label an unmodified run as qualified')
    try:
        for module, original in held:
            module.forward = _stable_forward(original)
        yield len(held)
    finally:
        for module, original in held:
            module.forward = original


@contextmanager
def reference_tie_contract(model) -> Iterator[None]:
    """Dew's selector tie rule over a reference that specifies none.

    `jax.lax.top_k` is stable, so equal selector scores go to the lower
    token index; torch's `topk` promises no tie order, and a sparse
    indexer is the only place that ordering reaches the logits. Scope the
    stable ordering to that one class per family that has one, and leave
    every other family's reference run untouched.
    """
    selectors = {'glm5_next_text': ('transformers.models.glm5_next.modeling_glm5_next',
                                    'Glm5NextTextIndexer'),
                 'deepseek_v4': ('transformers.models.deepseek_v4.modeling_deepseek_v4',
                                 'DeepseekV4Indexer')}
    named = selectors.get(model.config.model_type)
    if named is None:
        yield
        return
    from importlib import import_module

    module, class_name = named
    with stable_selector_ties(model, getattr(import_module(module), class_name)):
        yield


def reference_model(case: Case, directory: Path):
    """The reference implementation over the exported directory, with its
    loading report.

    A family whose released model_type transformers registers a config for
    loads through `AutoModelForCausalLM`. Kimi K2's does not exist upstream,
    so the case names the class its release's `auto_map` points at and this
    registers that class's own tensor conversion under the checkpoint's
    model_type. transformers keys the per-expert conversion by model_type,
    so without it a Kimi checkpoint's one tensor per expert would reach no
    converter and arrive unread while the fused parameters stayed random.
    The arithmetic is transformers' own either way.
    """
    import torch
    import transformers

    factory = transformers.AutoModelForCausalLM
    if case.reference_class is not None:
        from transformers.conversion_mapping import (
            get_checkpoint_conversion_mapping, register_checkpoint_conversion_mapping,
        )

        from importlib import import_module
        factory = getattr(import_module(case.reference_module), case.reference_class)
        model_type = json.loads((directory / "config.json").read_text())["model_type"]
        conversion = get_checkpoint_conversion_mapping(case.conversion_type or factory.config_class.model_type)
        if conversion is None:
            raise ValueError(f"no reference conversion for {model_type}")
        register_checkpoint_conversion_mapping(
            model_type, conversion, overwrite=True)
    loaded = factory.from_pretrained(
        str(directory), dtype=torch.float32, local_files_only=True, output_loading_info=True,
        experts_implementation="eager")
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise TypeError("output_loading_info must return a model and its loading report")
    model, report = loaded
    for category in ("missing_keys", "mismatched_keys", "error_msgs"):
        if report.get(category):
            raise ValueError(f"reference load {category}: {report[category]}")
    unexpected = report.get("unexpected_keys", [])
    # Transformers has no GLM or Qwen prediction module: Glm4MoePreTrainedModel
    # reads the depths past num_hidden_layers as unexpected and
    # Qwen3NextPreTrainedModel ignores `^mtp.*` on load
    # (modeling_qwen3_next.py:877). Only those may remain unconsumed; an
    # unrelated tensor is an export bug. DeepSeek V4 ships a depth too and
    # reports nothing for it, so it declares no prefix here: its own class
    # filters every `mtp.` key out of the report
    # (`_keys_to_ignore_on_load_unexpected`, modeling_deepseek_v4.py:1212).
    prefixes = _prediction_prefixes(model.config)
    if any(not name.startswith(prefixes) for name in unexpected):
        raise ValueError(f"reference load unexpected tensors: {unexpected}")
    return model, report


def reference_logits(case: Case, directory: Path, ids: np.ndarray) -> np.ndarray:
    """transformers 5.16.1 over the exported directory, fp32 on the eager path."""
    import torch

    model, _ = reference_model(case, directory)
    model.eval()
    model.set_attn_implementation("eager")
    with torch.no_grad(), reference_tie_contract(model):
        out = model(input_ids=torch.from_numpy(np.asarray(ids, np.int64)), use_cache=False)
    return out.logits.to(torch.float32).numpy()


def glm5_prediction_logits(case: Case, directory: Path, ids: np.ndarray) -> np.ndarray:
    """Read the trained GLM depth, using the unchanged reference composition.

    Native NextN at SGLang 97c6978 passes final-normalized target states
    (glm5_next.py:1068-1075, eagle_worker_v2.py:1229-1235) into the plain
    NoPE block (deepseek_nextn.py:177-187,247-298). This reads new weights;
    it neither redraws a tensor nor changes the existing fixture oracle.
    """
    from concurrent.futures import Future
    from copy import deepcopy

    import torch
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping
    from transformers.core_model_loading import WeightConverter, dot_natural_key
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextIndexer
    from tools.hf_reference_b import glm5_next_mtp

    model, _ = reference_model(case, directory)
    if model.config.model_type != 'glm5_next_text':
        raise ValueError('GLM prediction composition requires glm5_next_text')
    model.eval()
    model.set_attn_implementation('eager')
    depth = glm5_next_mtp(model.config).float().eval()
    registered = get_checkpoint_conversion_mapping('glm5_next')
    if registered is None:
        raise ValueError('glm5_next has no reference checkpoint conversion')
    rules = deepcopy(registered)
    prefix = f'model.layers.{model.config.num_hidden_layers}.'
    tensors = source_tensors(directory)
    prepared: dict[str, torch.Tensor] = {}
    pending: dict[str, WeightConverter] = {}
    for source_name in sorted((name for name in tensors if name.startswith(prefix)),
                              key=dot_natural_key):
        name = source_name.removeprefix(prefix).replace('shared_head.norm.', 'shared_head_norm.')
        value = torch.from_numpy(tensors[source_name])
        for rule in rules:
            target, matched = rule.rename_source_key(name)
            if matched is None:
                continue
            if isinstance(rule, WeightConverter):
                converter = pending.get(target)
                if converter is None:
                    converter = deepcopy(rule)
                    pending[target] = converter
                future: Future[torch.Tensor] = Future()
                future.set_result(value)
                converter.add_tensor(target, name, matched, future)
                break
            name = target
        else:
            prepared[name] = value
    for target, converter in pending.items():
        prepared.update(converter.convert(target, config=model.config))
    depth.load_state_dict(prepared, strict=True)
    tokens = torch.from_numpy(np.asarray(ids, np.int64))
    # The depth is its own sparse block with its own indexer, so it needs the
    # tie contract the trunk runs under, not just the trunk's.
    with torch.no_grad(), reference_tie_contract(model), \
            stable_selector_ties(depth, Glm5NextTextIndexer):
        hidden = model.model(input_ids=tokens, use_cache=False).last_hidden_state
        logits = depth(model, hidden[:, :-1], tokens[:, 1:])
    return logits.float().numpy()


def round_trip(case: Case, workspace: Path) -> RoundTrip:
    """`case` loaded, trained for one step, exported and read back."""
    directory = FIXTURES / case.fixture
    ids = np.load(directory / "input_ids.npy")
    source = load_pretrained(str(directory), dtype="float32", attention_impl="reference")
    state = train(case, source, ids)
    export = workspace / case.name
    source.save(export, variables=state.params)
    reloaded = load_pretrained(str(export), dtype="float32", attention_impl="reference")
    return RoundTrip(case, source, state.params, export, ids,
                     logits(source, state.params, ids), reloaded,
                     reference_logits(case, export, ids),
                     source_tensors(directory), source_tensors(export))


def moved(trip: RoundTrip) -> dict[str, float]:
    """How far the trained export moved from the checkpoint, by tensor kind.

    Only the kinds this checkpoint names appear: a dense family holds no
    expert and no router, Mixtral routes every layer and holds no dense
    feed-forward, and DeepSeek V4's only unrouted feed-forward is its
    shared expert. Llama 4 spells its feed-forward `feed_forward` where
    most say `mlp`, V4 spells the block's halves `attn` and `ffn` and its
    embedding `embed`, and a wrapper repo nests the decoder's own names
    under its language model.

    Only bound tensors are measured. A retained one carries its source
    bytes out by construction, which
    `test_every_source_tensor_is_bound_or_retained_and_written_back` holds
    to account, and a training step cannot move it.
    """
    bound = {layout.name for layout in trip.source.weight_layouts}
    kinds = {"embedding": lambda name: name.endswith(("model.embed_tokens.weight",
                                                      "embed.weight")),
             "attention": lambda name: ".self_attn." in name or ".attn." in name,
             "feedforward": lambda name: (
                 (".mlp." in name or ".feed_forward." in name or ".ffn." in name)
                 and ".experts." not in name
                 and not name.endswith(("mlp.gate.weight", "router.weight",
                                        "ffn.gate.weight", "ffn.gate.bias", "ffn.gate.tid2eid"))),
             "expert": lambda name: ".experts." in name,
             "router": lambda name: name.endswith(("mlp.gate.weight", "router.weight",
                                                   "ffn.gate.weight",
                                                   "block_sparse_moe.gate.weight")),
             "balancing bias": lambda name: name.endswith(("e_score_correction_bias",
                                                           "ffn.gate.bias"))}
    distances = {}
    for kind, belongs in kinds.items():
        moves = [float(np.max(np.abs(trip.exported_tensors[name].astype(np.float32)
                                     - tensor.astype(np.float32))))
                 for name, tensor in trip.source_tensors.items()
                 if name in bound and belongs(name)]
        if moves:
            distances[kind] = max(moves)
    return distances


def gradient_parity(trip: RoundTrip) -> dict[str, float]:
    """The two implementations' gradients of the mean next-token cross
    entropy over the source checkpoint, per source tensor, as
    `max |ours - theirs| / max(1, |ours|, |theirs|)`.

    Each bound layout writes dew's gradient tree back into the source's
    own tensor layout the way the export writes the weights, so the same
    slices, transposes and concatenations that carry a trained weight out
    carry its gradient to the reference's `.grad`. A per-expert tensor
    reads its slice of the reference's fused `experts.gate_up_proj` and
    `experts.down_proj`, the parameters transformers converts them into.
    Tensors the reference holds no gradient for (an MTP depth, the
    balancing bias buffer, the indexer it runs under no_grad) are not
    compared.

    Every gradient is keyed the way a checkpoint is written back, by the
    reversal transformers saves through
    (`core_model_loading.revert_weight_conversion`), so a release whose
    names are not its module names at all — DeepSeek V4 spells the block's
    halves `attn` and `ffn` and its experts `w1`/`w2`/`w3` — is named the
    way the source and every binding name it, per-expert tensors included.
    What the reversal keeps is the module tree's own nesting prefix, which
    a release need not carry: V4 holds its stack at `layers.N.*` and Kimi
    K2.5's `model.language_model.X` is the source's `language_model.model.X`.
    Only those prefixes move, so both sides are indexed with them stripped,
    a collision there is raised rather than guessed at, and the two indexes
    must be a bijection.
    """
    if trip.case.name == 'deepseek_v4' and trip.case.mtp_weight is not None:
        return v4_training_gradient_parity(trip)
    import torch

    source = trip.source
    ids = jnp.asarray(trip.ids, jnp.int32)

    def loss(params):
        logits = jnp.asarray(source.model.apply({**source.variables, "params": params}, ids),
                             jnp.float32)
        return -jnp.mean(jnp.take_along_axis(
            jax.nn.log_softmax(logits[:, :-1], axis=-1), ids[:, 1:, None], axis=-1))

    grads = {"params": jax.grad(loss)(source.variables["params"])}
    model, _ = reference_model(trip.case, FIXTURES / trip.case.fixture)
    model.train(False)
    model.set_attn_implementation("eager")
    labels = torch.from_numpy(np.asarray(trip.ids, np.int64))
    model(input_ids=labels, labels=labels, use_cache=False).loss.backward()
    from transformers.core_model_loading import revert_weight_conversion

    parameters = dict(model.named_parameters())
    prediction_prefixes = _prediction_prefixes(model.config)
    text_half = _text_half_prefixes(model)
    excluded = {name for name in parameters
                if '.indexer.' in name or name.startswith(prediction_prefixes)
                or (text_half and not name.startswith(text_half))}
    unexpected_missing = {name for name, parameter in parameters.items()
                          if parameter.requires_grad and parameter.grad is None and name not in excluded}
    unexpected_present = {name for name in excluded if parameters[name].grad is not None}
    if unexpected_missing or unexpected_present:
        raise ValueError(f'reference gradient participation differs: missing {sorted(unexpected_missing)}, '
                         f'excluded but present {sorted(unexpected_present)}')
    upstream = {name: parameter.grad for name, parameter in parameters.items()
                if parameter.grad is not None}
    # The inverse converter processes every supplied upstream name, including
    # unmatched pass-throughs (core_model_loading.py:1833-1860). Check its
    # scalar count as well, so a rename collision cannot silently drop data.
    converted = revert_weight_conversion(model, upstream)
    if sum(value.numel() for value in upstream.values()) != sum(value.numel() for value in converted.values()):
        raise ValueError('reference gradient conversion lost or duplicated scalar entries')

    def bare(name: str) -> str:
        """`name` without the nesting prefixes the two sides disagree on.

        The reversal keys a gradient the way the module tree is named, which
        is not always how the release spells the same tensor: DeepSeek V4
        holds its stack at `layers.N.*` with no `model.`, and Kimi K2.5's
        `model.language_model.X` is the source's `language_model.model.X`.
        Only those prefixes move, so both sides are indexed with them
        stripped and a collision is raised rather than guessed at.
        """
        parts = name.split('.')
        while parts[:1] in (['model'], ['language_model']):
            parts.pop(0)
        return '.'.join(parts)

    def indexed(named):
        held: dict[str, tuple[str, object]] = {}
        for name, value in named:
            key = bare(name)
            if key in held:
                raise ValueError(f'{name!r} and {held[key][0]!r} share the bare name {key!r}')
            held[key] = (name, value)
        return held

    # The reversal writes the checkpoint spelling of the model_type it was
    # handed, which is not always the release's own: DeepSeek V4 names the
    # block's halves `attn`/`ffn`, and Kimi K2.5's repo holds its text
    # stack at `layers.N` where the reversal writes `blocks.N`.
    if model.config.model_type == 'deepseek_v4':
        from tools.deepseek_v4_reference import deepseek_v4_source_name
        converted = {deepseek_v4_source_name(name): value for name, value in converted.items()}
    elif model.config.model_type == 'kimi_k25':
        converted = {name.replace('.blocks.', '.layers.'): value
                     for name, value in converted.items()}
    # Prediction-owned serialized embedding/head copies alias trunk leaves;
    # exclude their declared source prefix as well as independent MTP paths.
    layouts = indexed(
        (layout.name, layout) for layout in source.weight_layouts
        if layout.paths[0][0] == 'params' and 'indexer' not in layout.paths[0]
        and not layout.paths[0][1].startswith('mtp_')
        and not layout.name.startswith(prediction_prefixes))
    # The reversal keys a gradient by the checkpoint spelling of the
    # model_type it was handed, which is not always the release's: Kimi
    # K2.5's text stack is `layers.N` in its repo and `blocks.N` through
    # the reversal. A tensor the reversal only renamed is the parameter
    # itself, so where the module tree already names it the way a binding
    # does, that name stands; what the reversal built rather than passed
    # through, a fused expert kernel split apart, keeps the name it was
    # built under.
    named = {id(grad): name for name, grad in upstream.items() if bare(name) in layouts}
    reference = indexed((named.get(id(value), name), value)
                        for name, value in converted.items())
    if set(reference) != set(layouts):
        raise ValueError(f'gradient source/layout bijection differs: '
                         f'upstream only {sorted(reference[key][0] for key in set(reference) - set(layouts))}, '
                         f'Dew only {sorted(layouts[key][0] for key in set(layouts) - set(reference))}')
    for layout in source.weight_layouts:
        path = layout.paths[0]
        if path[0] == 'params' and ('indexer' in path or path[1].startswith('mtp_')):
            if np.any(layout.export(grads) != 0):
                raise ValueError(f'{layout.name} has an unexpected trunk-loss gradient')
    errors = {}
    for key, (_, upstream_grad) in reference.items():
        name, layout = layouts[key]
        ours = layout.export(grads)
        theirs = upstream_grad.to(torch.float32).numpy()
        if ours.shape != theirs.shape:
            raise ValueError(f'{name} gradient shapes differ: {ours.shape} versus {theirs.shape}')
        errors[name] = float(np.max(
            np.abs(ours - theirs) / np.maximum(1.0, np.maximum(np.abs(ours), np.abs(theirs)))))
    return errors


def v4_training_gradient_parity(trip: RoundTrip) -> dict[str, float]:
    """Actual LMObjective gradients, including V4's raw-stream prediction loss.

    The CPU reference composes unchanged Transformers modules according to
    the released inference/model.py MTPBlock. Name coverage is the full
    loaded parameter tree, except selector parameters proven to have no
    gradient through top-k; missing reference gradients are errors.
    """
    import torch
    from transformers.core_model_loading import revert_weight_conversion
    from dew.objectives.base import Step, scalar_loss
    from tools.deepseek_v4_reference import deepseek_v4_source_name, load_mtp_reference

    source, case = trip.source, trip.case
    weight = case.mtp_weight
    if weight is None:
        raise ValueError('V4 training gradient comparison requires mtp_weight')
    objective = LMObjective(source.model, seq_len=trip.ids.shape[1] - 1, ema_decay=None,
                            pretrained=source.variables, mtp_weight=case.mtp_weight)
    step = Step(step=jnp.asarray(0), key=jax.random.key(SEED), ema=None)
    batch = {'text': jnp.asarray(trip.ids, jnp.int32)}

    def loss(params):
        return scalar_loss(objective, {**source.variables, 'params': params}, batch, step)[0]

    our_loss, gradients = jax.value_and_grad(loss)(source.variables['params'])
    model, _ = reference_model(case, FIXTURES / case.fixture)
    model.eval()
    model.set_attn_implementation('eager')
    depth = load_mtp_reference(FIXTURES / case.fixture, model.config)
    tokens = torch.from_numpy(trip.ids.astype(np.int64))
    streams: list[torch.Tensor] = []

    def capture(module, args, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError('V4 decoder layer must return raw residual streams')
        streams.append(output)

    handle = model.model.layers[-1].register_forward_hook(capture)
    try:
        logits = model(input_ids=tokens[:, :-1], use_cache=False).logits
    finally:
        handle.remove()
    predicted = depth(model, streams[0][:, :-1], tokens[:, 1:-1])
    denominator = tokens[:, 1:].numel()
    main_loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1), reduction='sum')
    prediction_loss = torch.nn.functional.cross_entropy(
        predicted.reshape(-1, predicted.shape[-1]), tokens[:, 2:].reshape(-1), reduction='sum')
    reference_loss = (main_loss + weight * prediction_loss) / denominator
    reference_loss.backward()
    if abs(float(our_loss) - float(reference_loss.detach())) > 1e-4:
        raise ValueError('V4 LMObjective and reference prediction losses differ')

    def converted(reference):
        grads = {name: parameter.grad for name, parameter in reference.named_parameters()
                 if parameter.grad is not None}
        return {deepseek_v4_source_name(name): value
                for name, value in revert_weight_conversion(reference, grads).items()}

    expected = converted(model)
    for name, value in converted(depth.reference_model).items():
        if name.startswith('layers.0.'):
            expected['mtp.0.' + name.removeprefix('layers.0.')] = value
        elif name.startswith('hc_head_') or name == 'norm.weight':
            expected['mtp.0.' + name] = value
    for name in ('e_proj', 'h_proj', 'enorm', 'hnorm'):
        for leaf, parameter in getattr(depth, name).named_parameters():
            if parameter.grad is not None:
                expected[f'mtp.0.{name}.{leaf}'] = parameter.grad
    layouts = {layout.name: layout for layout in source.weight_layouts
               if layout.paths[0][0] == 'params' and 'indexer' not in layout.paths[0]}
    if set(expected) != set(layouts):
        raise ValueError(f"V4 gradient coverage mismatch: missing {sorted(set(layouts) - set(expected))}, "
                         f"unexpected {sorted(set(expected) - set(layouts))}")
    errors = {}
    for name, layout in layouts.items():
        ours = layout.export({'params': gradients})
        theirs = expected[name].detach().float().numpy()
        errors[name] = float(np.max(np.abs(ours - theirs) /
                                   np.maximum(1.0, np.maximum(np.abs(ours), np.abs(theirs)))))
    for layout in source.weight_layouts:
        if layout.paths[0][0] == 'params' and 'indexer' in layout.paths[0]:
            if np.any(layout.export({'params': gradients}) != 0):
                raise ValueError(f'{layout.name} differentiates through discrete selection')
    return errors


def measure(trip: RoundTrip) -> dict[str, object]:
    """The row this tool prints for one family."""
    ours, theirs = trip.ours, trip.theirs
    indexed = [layout for layout in trip.source.weight_layouts if layout.expert_index is not None]
    return {"source tensors": len(trip.source_tensors),
            "bound": len(trip.source.weight_layouts),
            "indexed experts": len(indexed),
            "retained": len(trip.source.retained_tensors),
            "exported": len(trip.exported_tensors),
            "argmax equal": bool(np.array_equal(np.argmax(ours, -1), np.argmax(theirs, -1))),
            "max |logit difference|": float(np.max(np.abs(ours - theirs)))}


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="dew-decoder-export-") as workspace:
        for case in CASES:
            trip = round_trip(case, Path(workspace))
            row = " ".join(f"{key} {value}" for key, value in measure(trip).items())
            moves = ", ".join(f"{kind} {value:.3e}" for kind, value in moved(trip).items())
            print(f"{case.name:13s} {row} | moved: {moves}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
