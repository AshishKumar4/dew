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
    otherwise. DeepSeek V4's lightning indexer keeps two of the compressed
    entries a query may attend, and its fixture's top-k margin is 0.022, so
    a step of `RATE` puts five query rows on exactly tied zero scores.
    Torch and JAX choose different sets on three of those rows even when
    handed identical scores. This fixture steps at 5e-3, retaining a strict
    top-k boundary; no parity claim covers tied indexer selections.
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
    Case("deepseek_v4", "deepseek-v4-tiny", balance_rate=1e-2, rate=5e-3),
    Case("kimi_k2", "kimi-k2-tiny", balance_rate=1e-2,
         reference_class="DeepseekV3ForCausalLM"),
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
    excluded = {name for name in parameters
                if '.indexer.' in name or name.startswith(prediction_prefixes)}
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

    reference = indexed(converted.items())
    # Prediction-owned serialized embedding/head copies alias trunk leaves;
    # exclude their declared source prefix as well as independent MTP paths.
    layouts = indexed(
        (layout.name, layout) for layout in source.weight_layouts
        if layout.paths[0][0] == 'params' and 'indexer' not in layout.paths[0]
        and not layout.paths[0][1].startswith('mtp_')
        and not layout.name.startswith(prediction_prefixes))
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
