"""One trained source-format export per masked-diffusion decoder family, measured.

The reproduction command behind the numbers in
tests/test_masked_diffusion_export.py. Each case loads a committed tiny
checkpoint through `load_pretrained`, runs one real `Trainer` step of plain
SGD under `MaskedDiffusionObjective` started from those weights, writes the
trained weights back into the source's own tensor names with
`Pretrained.save`, and reads the export back twice: with `load_pretrained`
for the parameter tree and the logits, and with transformers 5.16.1 for the
reference logits on the same ids.

    JAX_PLATFORMS=cpu PYTHONPATH=src python tools/masked_diffusion_export_reference.py

transformers 5.16.1 carries no LLaDA or Dream class; both releases ship
remote code. What it does carry is the block each of them is: LLaDA is the
Llama block with no causal mask anywhere (modeling_llada.py, LLaDAModel's
bidirectional bias and LLaDALlamaBlock is_causal=False) and Dream is the
Qwen2 block with the same mask dropped (modeling_dream.py, DreamAttention
hard-codes is_causal=False over biased q/k/v and a bias-free o_proj). So the
reference here is stock `LlamaForCausalLM` and `Qwen2ForCausalLM` over the
export's own tensors, loaded by `from_pretrained` and run with an all-visible
4-D attention mask, which `masking_utils._preprocess_mask_arguments` returns
untouched instead of building a causal one. LLaDA's tensors are renamed onto
the llama layout for that load, the correspondence the release's own names
describe; Dream's are already the qwen2 layout and are copied as they are.
The loading report has to name no missing, unexpected or mismatched tensor,
so the export is a complete checkpoint for the reference, not a subset it
happened to accept.

Nothing is downloaded and nothing is written outside the temporary directory
it works in. The tests import `round_trip` from here, so the numbers in them
and the assertions come from one pipeline.
"""

from __future__ import annotations

import json
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew.diffusion.discrete import MDLM
from dew.interop import load_pretrained
from dew.interop.pretrained import Pretrained
from dew.objectives.base import Variables

if TYPE_CHECKING:
    from transformers import PretrainedConfig

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "hf"
RATE = 5e-2
SEED = 3

_LLADA_TRUNK = {
    "model.transformer.wte.weight": "model.embed_tokens.weight",
    "model.transformer.ln_f.weight": "model.norm.weight",
    "model.transformer.ff_out.weight": "lm_head.weight",
}
_LLADA_BLOCK = {
    "attn_norm": "input_layernorm", "attn_out": "self_attn.o_proj",
    "ff_norm": "post_attention_layernorm", "ff_proj": "mlp.gate_proj",
    "up_proj": "mlp.up_proj", "ff_out": "mlp.down_proj",
    "q_proj": "self_attn.q_proj", "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
}


def _whole(config: Mapping[str, object], name: str) -> int:
    """One width of the exported config, refused if it is not a count."""
    value = config[name]
    if type(value) is not int:
        raise ValueError(f"{name} is a count, got {value!r}")
    return value


def _real(config: Mapping[str, object], name: str) -> float:
    value = config[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} is numeric, got {value!r}")
    return float(value)


def _llada_reference(config: Mapping[str, object], tensors: Mapping[str, np.ndarray]
                     ) -> tuple[PretrainedConfig, dict[str, np.ndarray]]:
    """LLaDA's export as the llama checkpoint transformers reads.

    The release stores the OLMo-style names its own modeling file declares;
    the block underneath is Llama's, so each name is respelled and the
    tensors are handed over untouched.
    """
    from transformers import LlamaConfig

    def rename(name: str) -> str:
        renamed = _LLADA_TRUNK.get(name)
        if renamed is not None:
            return renamed
        parts = name.split(".")
        if (len(parts) != 6 or parts[:3] != ["model", "transformer", "blocks"]
                or not parts[3].isdigit() or parts[5] != "weight"
                or parts[4] not in _LLADA_BLOCK):
            raise ValueError(f"{name!r} is not a LLaDA tensor name")
        return f"model.layers.{parts[3]}.{_LLADA_BLOCK[parts[4]]}.weight"

    reference = LlamaConfig(
        hidden_size=_whole(config, "d_model"), num_hidden_layers=_whole(config, "n_layers"),
        num_attention_heads=_whole(config, "n_heads"),
        num_key_value_heads=_whole(config, "n_kv_heads"),
        intermediate_size=_whole(config, "mlp_hidden_size"),
        vocab_size=_whole(config, "vocab_size"),
        max_position_embeddings=_whole(config, "max_sequence_length"),
        rms_norm_eps=_real(config, "rms_norm_eps"),
        rope_parameters={"rope_type": "default", "rope_theta": _real(config, "rope_theta")},
        tie_word_embeddings=bool(config["weight_tying"]), hidden_act="silu",
        attention_bias=False, mlp_bias=False, dtype="float32")
    return reference, {rename(name): tensor for name, tensor in tensors.items()}


def _dream_reference(config: Mapping[str, object], tensors: Mapping[str, np.ndarray]
                     ) -> tuple[PretrainedConfig, dict[str, np.ndarray]]:
    """Dream's export as the qwen2 checkpoint transformers reads.

    Dream ships the qwen2 tensor layout, so the table is what it was; only
    the config is respelled into the class that computes the same block.
    """
    from transformers import Qwen2Config

    reference = Qwen2Config(
        hidden_size=_whole(config, "hidden_size"),
        num_hidden_layers=_whole(config, "num_hidden_layers"),
        num_attention_heads=_whole(config, "num_attention_heads"),
        num_key_value_heads=_whole(config, "num_key_value_heads"),
        intermediate_size=_whole(config, "intermediate_size"),
        vocab_size=_whole(config, "vocab_size"),
        max_position_embeddings=_whole(config, "max_position_embeddings"),
        rms_norm_eps=_real(config, "rms_norm_eps"),
        rope_parameters={"rope_type": "default", "rope_theta": _real(config, "rope_theta")},
        tie_word_embeddings=bool(config["tie_word_embeddings"]),
        hidden_act=str(config["hidden_act"]), use_sliding_window=False, dtype="float32")
    return reference, dict(tensors)


@dataclass(frozen=True)
class Case:
    """A masked-diffusion family's fixture and how its export is read back.

    `moves` names one source tensor per kind of weight a step has to move,
    so a step that only touched the embedding would not pass for a step that
    trained the stack.
    """

    name: str
    fixture: str
    reference: Callable[[Mapping[str, object], Mapping[str, np.ndarray]],
                        tuple["PretrainedConfig", dict[str, np.ndarray]]]
    moves: Mapping[str, str]


CASES = (
    Case("llada", "llada-tiny", _llada_reference, {
        "embedding": "model.transformer.wte.weight",
        "attention": "model.transformer.blocks.0.q_proj.weight",
        "mlp": "model.transformer.blocks.1.ff_proj.weight",
        "norm": "model.transformer.ln_f.weight",
        "head": "model.transformer.ff_out.weight"}),
    Case("dream", "dream-tiny", _dream_reference, {
        "embedding": "model.embed_tokens.weight",
        "attention": "model.layers.0.self_attn.q_proj.weight",
        "attention_bias": "model.layers.0.self_attn.q_proj.bias",
        "mlp": "model.layers.1.mlp.gate_proj.weight",
        "norm": "model.norm.weight",
        "head": "lm_head.weight"}),
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
    report: Mapping[str, list]
    source_tensors: dict[str, np.ndarray] = field(repr=False)
    exported_tensors: dict[str, np.ndarray] = field(repr=False)


def source_tensors(directory: Path) -> dict[str, np.ndarray]:
    """The checkpoint's own tensor table, as the loader reads it."""
    from dew.interop.hf_decoders import _load_shards

    return _load_shards(directory)


def logits(source: Pretrained, variables: Variables, ids: np.ndarray) -> np.ndarray:
    return np.asarray(source.model.apply(variables, jnp.asarray(ids, jnp.int32)), np.float32)


def train(source: Pretrained, ids: np.ndarray):
    """One real `Trainer` step of plain SGD from the loaded checkpoint.

    The objective holds the loaded tree, so the trainer takes it through
    `held_variables` as an argument of its state JIT rather than compiling
    it in. The rows are the fixture's own ids repeated to fill one batch per
    device, so the step runs on whatever devices the process has.
    """
    import grain.python as pygrain

    from dew.data.dataset import Dataset
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.diffusion.masked import MaskedDiffusionObjective
    from dew.training import Trainer

    model = source.model
    if not isinstance(model, CausalTransformer):
        raise TypeError("a masked diffusion source loads as a CausalTransformer")
    mask_id = model.mask_token_id
    if mask_id is None:
        raise ValueError("a masked diffusion checkpoint reserves its mask id")
    objective = MaskedDiffusionObjective(
        model, MDLM(mask_id=int(mask_id))(), seq_len=int(ids.shape[1]),
        ema_decay=None, pretrained=source.variables)
    count = math.lcm(int(ids.shape[0]), jax.device_count())
    rows = np.concatenate([ids] * (count // int(ids.shape[0])), axis=0)
    entries = [{"text": rows[row]} for row in range(count)]
    stream = (pygrain.MapDataset.source(entries).repeat().to_iter_dataset()
              .batch(count, drop_remainder=True))
    data = Dataset(train=lambda: iter(stream), val=None, records=count, batch=count)
    return Trainer(objective, optax.sgd(RATE), key=jax.random.key(SEED)).fit(
        data, steps=1, log_every=1)


def reference_logits(case: Case, export: Path, ids: np.ndarray,
                     workspace: Path) -> tuple[np.ndarray, dict[str, list]]:
    """transformers 5.16.1 over the export, fp32 on the eager path.

    The all-visible 4-D mask is what makes the causal reference compute the
    bidirectional forward both releases hard-code; a 2-D mask or none would
    have it build the causal one instead.
    """
    import torch
    from safetensors.numpy import save_file
    from transformers import AutoModelForCausalLM

    config, tensors = case.reference(json.loads((export / "config.json").read_text()),
                                     source_tensors(export))
    view = workspace / f"{case.name}-reference"
    view.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(str(view))
    save_file(tensors, str(view / "model.safetensors"))
    loaded = AutoModelForCausalLM.from_pretrained(
        str(view), dtype=torch.float32, local_files_only=True, output_loading_info=True)
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise TypeError("output_loading_info must return a model and its loading report")
    model, report = loaded
    named = {category: list(report.get(category, []))
             for category in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")}
    model.eval()
    model.set_attn_implementation("eager")
    rows, width = int(ids.shape[0]), int(ids.shape[1])
    visible = torch.zeros((rows, 1, width, width), dtype=torch.float32)
    with torch.no_grad():
        out = model(input_ids=torch.from_numpy(np.asarray(ids, np.int64)),
                    attention_mask=visible, use_cache=False)
    return out.logits.to(torch.float32).numpy(), named


def round_trip(case: Case, workspace: Path) -> RoundTrip:
    """`case` loaded, trained for one step, exported and read back."""
    directory = FIXTURES / case.fixture
    ids = np.load(directory / "input_ids.npy")
    source = load_pretrained(str(directory), dtype="float32", attention_impl="reference")
    state = train(source, ids)
    export = workspace / case.name
    source.save(export, variables=state.params)
    reloaded = load_pretrained(str(export), dtype="float32", attention_impl="reference")
    theirs, report = reference_logits(case, export, ids, workspace)
    return RoundTrip(case, source, state.params, export, ids,
                     logits(source, state.params, ids), reloaded, theirs, report,
                     source_tensors(directory), source_tensors(export))


def moved(trip: RoundTrip) -> dict[str, float]:
    """How far the trained export moved from the checkpoint, by tensor kind."""
    return {kind: float(np.max(np.abs(trip.exported_tensors[name].astype(np.float32)
                                      - trip.source_tensors[name].astype(np.float32))))
            for kind, name in trip.case.moves.items()}


def measure(trip: RoundTrip) -> dict[str, object]:
    """The row this tool prints for one family."""
    return {"source tensors": len(trip.source_tensors),
            "argmax equal": bool(np.array_equal(np.argmax(trip.ours, -1),
                                                np.argmax(trip.theirs, -1))),
            "max |logit difference|": float(np.max(np.abs(trip.ours - trip.theirs))),
            "logit magnitude": float(np.max(np.abs(trip.theirs)))}


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="dew-masked-export-") as workspace:
        for case in CASES:
            trip = round_trip(case, Path(workspace))
            row = " | ".join(f"{name}: {value}" for name, value in measure(trip).items())
            moves = " ".join(f"{kind} {distance:.3e}" for kind, distance in moved(trip).items())
            print(f"{case.name:6s} {row} | moved: {moves}")


if __name__ == "__main__":
    main()
