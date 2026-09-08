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

import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew.interop import load_pretrained
from dew.interop.pretrained import Pretrained
from dew.objectives.base import Variables
from dew.objectives.lm import LMObjective

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "hf"
RATE = 5e-2
SEED = 3


@dataclass(frozen=True)
class Case:
    """A family's fixture and the loss terms its checkpoint supports.

    `balance_rate` needs a mixture that keeps a routing bias, so only the
    sigmoid-scored families carry one; `mtp_weight` needs a configured
    prediction depth, which GLM's checkpoint is the one to ship.
    """

    name: str
    fixture: str
    balance_rate: float | None = None
    mtp_weight: float | None = None


CASES = (
    Case("mixtral", "mixtral-tiny"),
    Case("qwen3_moe", "qwen3-moe-tiny"),
    Case("glm4_moe", "glm4-moe-tiny", balance_rate=1e-2, mtp_weight=0.3),
    Case("deepseek_v2", "deepseek-v2-tiny"),
    Case("deepseek_v3", "deepseek-v3-tiny", balance_rate=1e-2),
    Case("deepseek_v32", "deepseek-v32-tiny", balance_rate=1e-2),
    Case("llama4_text", "llama4-tiny"),
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
    state = Trainer(objective, optax.sgd(RATE), key=jax.random.key(SEED)).fit(
        data, steps=1, log_every=1)
    return state


def reference_logits(directory: Path, ids: np.ndarray) -> np.ndarray:
    """transformers 5.16.1 over the exported directory, fp32 on the eager path."""
    import torch
    from transformers import AutoModelForCausalLM

    loaded = AutoModelForCausalLM.from_pretrained(
        str(directory), dtype=torch.float32, local_files_only=True, output_loading_info=True)
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise TypeError("output_loading_info must return a model and its loading report")
    model, report = loaded
    for category in ("missing_keys", "mismatched_keys", "error_msgs"):
        if report.get(category):
            raise ValueError(f"reference load {category}: {report[category]}")
    unexpected = report.get("unexpected_keys", [])
    # Transformers has no GLM prediction module. Only its declared MTP
    # depths may remain unconsumed; an unrelated tensor is an export bug.
    config = model.config
    prefixes = (tuple(f"model.layers.{config.num_hidden_layers + depth}."
                      for depth in range(config.num_nextn_predict_layers))
                if config.model_type == "glm4_moe" else ())
    if any(not name.startswith(prefixes) for name in unexpected):
        raise ValueError(f"reference load unexpected tensors: {unexpected}")
    model.eval()
    model.set_attn_implementation("eager")
    with torch.no_grad():
        out = model(input_ids=torch.from_numpy(np.asarray(ids, np.int64)), use_cache=False)
    return out.logits.to(torch.float32).numpy()


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
                     reference_logits(export, ids),
                     source_tensors(directory), source_tensors(export))


def moved(trip: RoundTrip) -> dict[str, float]:
    """How far the trained export moved from the checkpoint, by tensor kind."""
    kinds = {"embedding": lambda name: name == "model.embed_tokens.weight",
             "expert": lambda name: ".experts." in name,
             "router": lambda name: name.endswith(("mlp.gate.weight", "router.weight",
                                                   "block_sparse_moe.gate.weight")),
             "balancing bias": lambda name: name.endswith("e_score_correction_bias")}
    distances = {}
    for kind, belongs in kinds.items():
        moves = [float(np.max(np.abs(trip.exported_tensors[name].astype(np.float32)
                                     - tensor.astype(np.float32))))
                 for name, tensor in trip.source_tensors.items() if belongs(name)]
        if moves:
            distances[kind] = max(moves)
    return distances


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
    main()
