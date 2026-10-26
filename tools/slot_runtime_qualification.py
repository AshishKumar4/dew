"""Qualify released slot engines against the authors' independently streamed float64 runtime.

The native and author phases run in separate processes. LayerBanks owns weight reads and layouts;
the author's Transformers model owns all forward arithmetic and Accelerate owns its offload hooks.
"""

import argparse
import contextlib
import functools
import hashlib
import importlib
import importlib.metadata
import json
import resource
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TorchBankWeights(Mapping):
    """A Transformers state dict backed by one native layer bank, never a full resident checkpoint.

    Source names pass through Transformers' published checkpoint conversion mapping. The inverse
    native weight layouts recover the source tensors; non-decoder entries are read one tensor at a time.
    Each tensor, once recovered, is kept on the host at the narrowest float dtype that holds it exactly
    (the release's stored one: bf16 for a bf16 release), so every later forward reads no bank and
    widens it in NumPy: the release is read and laid out once per run, not once per forward. The
    narrow copy is storage outside Torch, so the float64 oracle's audit sees float64 tensors alone.
    """

    def __init__(self, source, bundle, model):
        import jax
        import numpy as np
        from flax.traverse_util import flatten_dict
        from transformers.conversion_mapping import get_checkpoint_conversion_mapping

        from dew.inference.banks import at_layer, one_layer

        self.source = source
        self.layouts = {}
        self.owner = {}
        self.current = None
        self.held = {}
        self.peak_bank_bytes = 0
        self.reads = 0
        self.kept = {}
        self.seconds = {"read": 0.0, "widen": 0.0}
        """Where a request's weights cost: the layer banks read from disk, and each tensor's export,
        widening to float64 and copy into Torch."""
        self.cpu = jax.local_devices(backend="cpu")[0]
        expected = model.state_dict()
        owners = {}
        shapes = source.shapes()
        for site in bundle.model.bank_sites:
            for first, count in site.view.groups:
                for index in range(first, first + count):
                    layer = at_layer(one_layer(shapes, index, namespace=site.namespace), index,
                                     namespace=site.namespace)
                    owners.update(dict.fromkeys(flatten_dict(layer), (site.namespace, index)))
        conversions = get_checkpoint_conversion_mapping(model.config.model_type)
        for layout in bundle.weight_layouts:
            name = layout.name
            for conversion in conversions:
                renamed, matched = conversion.rename_source_key(layout.name)
                if matched is not None:
                    name = renamed
                    break
            if name not in expected:
                continue
            if name in self.layouts or tuple(expected[name].shape) != layout.shape:
                raise ValueError(f"the source layout does not uniquely supply {name}'s shape")
            held = {owners.get(path) for path in layout.paths}
            if len(held) != 1:
                raise ValueError(f"{name} spans more than one layer bank")
            self.layouts[name] = layout
            self.owner[name] = held.pop()
        if missing := set(expected) - self.layouts.keys():
            raise ValueError(f"the bank source does not supply the author's parameters {sorted(missing)}")
        self.whole_bytes = sum(int(np.prod(leaf.shape)) * np.dtype(leaf.dtype).itemsize
                               for leaf in jax.tree.leaves(shapes))

    def __iter__(self):
        return iter(self.layouts)

    def __len__(self):
        return len(self.layouts)

    def clear(self):
        self.held = {}
        self.current = None

    def __getitem__(self, name):
        import numpy as np
        import torch

        if name not in self.kept:
            self.kept[name] = _narrowest(self._recovered(name))
        began = time.perf_counter()
        kept = self.kept[name]
        tensor = torch.from_numpy(kept.astype(np.float64) if np.issubdtype(kept.dtype, np.floating)
                                  or kept.dtype.name == "bfloat16" else kept)
        self.seconds["widen"] += time.perf_counter() - began
        return tensor

    def _recovered(self, name):
        """`name`'s source tensor in float64, from its layer's bank, read where no tensor of that layer
        was recovered before."""
        import jax
        import numpy as np
        from flax.traverse_util import unflatten_dict

        from dew.inference.banks import at_layer

        layout, owner = self.layouts[name], self.owner[name]
        began = time.perf_counter()
        if owner is None:
            self.clear()
            placement = unflatten_dict({path: jax.sharding.SingleDeviceSharding(self.cpu)
                                        for path in layout.paths})
            variables = self.source.entry(placement)
        else:
            if owner != self.current:
                self.clear()
                namespace, index = owner
                self.held = at_layer(self.source.read(index, namespace=namespace), index, namespace=namespace)
                self.current = owner
                self.reads += 1
                self.peak_bank_bytes = max(self.peak_bank_bytes,
                                          sum(leaf.nbytes for leaf in jax.tree.leaves(self.held)))
            variables = self.held
        # Layout conversion stays on the host; only Accelerate transfers the returned tensor to CUDA.
        with jax.default_device(self.cpu), jax.enable_x64(new_val=True):
            array = layout.export(variables)
        if jax.numpy.issubdtype(array.dtype, jax.numpy.floating):
            array = np.asarray(array, np.float64)
        self.seconds["read"] += time.perf_counter() - began
        return array


def _narrowest(array):
    """`array` at the narrowest float dtype that holds every value exactly (bf16, float32, else
    float64), or as it is where it is no float array."""
    import ml_dtypes
    import numpy as np

    if not np.issubdtype(array.dtype, np.floating):
        return np.ascontiguousarray(array)
    for narrow in (ml_dtypes.bfloat16, np.float32):
        held = np.ascontiguousarray(array.astype(narrow))
        if np.array_equal(held.astype(np.float64), array, equal_nan=True):
            return held
    return np.ascontiguousarray(array, np.float64)


def offloaded_model(directory, source, bundle, *, device):
    """The author's causal model with ordinary Accelerate offload, fed by the bank mapping."""
    import psutil
    import torch
    from accelerate import cpu_offload, init_empty_weights
    from block_override_reference import oracle_precision
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(directory, local_files_only=True).get_text_config()
    with oracle_precision(), init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.float64,
                                                attn_implementation="eager").eval()
    narrow = [name for name, value in (*model.named_parameters(), *model.named_buffers())
              if value.is_floating_point() and value.dtype != torch.float64]
    if narrow:
        raise RuntimeError(f"the offload skeleton must be float64 before installing weight hooks: {narrow}")
    stored = sum(path.stat().st_size for path in Path(directory).glob("*.safetensors"))
    if stored > (free := psutil.virtual_memory().available):
        raise RuntimeError(f"the author's weights are kept on the host as the release stores them, "
                           f"{stored} bytes, and this host has {free} free")
    weights = TorchBankWeights(source, bundle, model)
    classes = sorted({type(layer).__name__ for layer in model.base_model.layers})
    model = cpu_offload(model, execution_device=device, state_dict=weights,
                        offload_buffers=True, preload_module_classes=classes)
    return model, weights


OPTIONS = (2, 3, 4, 8)
"""A request's questions' option counts in turn, so `questions` a multiple of four keeps every draw's
leaf one length."""


def request(seed, *, questions, short_words, long_words):
    """A fixed, seeded member of the qualification population: one state, of varied length, and
    `questions` questions over it, score and choice, of `OPTIONS` options in turn. One request is one
    draw: the author's runtime forks its questions from one pass over the state, so a draw's rows cost
    about one request's state, and its errors average over every question's options."""
    import numpy as np

    rng = np.random.default_rng([0x5107, seed])
    words = ("new", "late", "paid", "ready", "broken", "clear", "urgent", "quiet")
    text = " ".join(rng.choice(words, long_words if seed % 2 else short_words).tolist())
    state = {"event": text, "origin": "recorded request"} if seed % 3 == 0 else text
    asked = {}
    for index in range(questions):
        criteria = [" ".join(rng.choice(words, 2).tolist()) for _ in range(OPTIONS[index % len(OPTIONS)])]
        question = {"type": "score", "instructions": "Rate this state.", "criteria": criteria}
        if (seed + index) % 3:
            question = {"type": "choice", "instructions": "Which option best describes the state?",
                        "criteria": {f"option{option}": value for option, value in enumerate(criteria)}}
        asked[f"case_{seed}_{index}"] = question
    return {"state": state, "questions": asked}


def make_pool(args):
    from reference_error import REDRAW, SIZING_FROM

    if args.count < 8 or args.short_words < 1 or args.long_words < args.short_words:
        raise ValueError("the pool needs at least eight verdict draws and positive ordered state lengths")
    if args.questions < 1 or args.questions % len(OPTIONS):
        raise ValueError(f"a request asks a positive multiple of {len(OPTIONS)} questions")
    verdict = list(range(args.count * (1 + REDRAW)))
    pilot = list(range(SIZING_FROM, SIZING_FROM + 2 * args.count))
    pool = {"count": args.count, "redraw": REDRAW, "sizing_from": SIZING_FROM,
            "questions": args.questions, "short_words": args.short_words, "long_words": args.long_words,
            "verdict": verdict, "pilot": pilot,
            "requests": {str(seed): request(seed, questions=args.questions, short_words=args.short_words,
                                            long_words=args.long_words)
                         for seed in (*verdict, *pilot)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(pool, ensure_ascii=False, indent=2) + "\n")


class ObservedScoring:
    """Observe the actual serving task's raw row logits without changing admission or execution."""

    def __init__(self, task):
        self.task = task
        self.values = {}
        self.batch_rows = []

    def admit(self, body):
        from dew.inference.scoring import Scored

        admitted = self.task.admit(body)
        if len(admitted.rows) != len(body["questions"]):
            raise ValueError("the qualification pool has one question per row")
        asked = tuple((name, len(question["criteria"])) for name, question in body["questions"].items())
        return Scored(admitted.rows, (asked, admitted.payload))

    def fits(self, rows):
        return self.task.fits(rows)

    def forward(self, variables, rows):
        self.batch_rows.append(len(rows))
        return self.task.forward(variables, rows)

    def answer(self, payload, outputs):
        import numpy as np

        asked, admitted = payload
        counts = dict(asked)
        for row, output in zip(admitted.rows, outputs, strict=True):
            [laid] = row.questions
            self.values[laid.name] = np.asarray(output)[0, :counts[laid.name]].copy()
        return self.task.answer(admitted, outputs)


def native(args, pool):
    import jax
    import jax.numpy as jnp

    from dew.batch import part
    from dew.decision import Decide
    from dew.decision.task import Budget
    from dew.inference import ScoringServer
    from dew.interop import Pretrained
    from dew.registry import dtype_name

    bundle = Pretrained.load(args.source, dtype=args.dtype, param_dtype="auto", attention_impl="xla")
    task = replace(Decide.of(bundle), budget=Budget(tokens=4096, rows=4))
    scoring, variables = task.scoring()
    observed = ObservedScoring(scoring)
    server = ScoringServer.from_task(observed, variables)
    # The pilots and the verdict's draws in one run, so the paired sizing and the verdict read one commit.
    seeds = (*pool["pilot"], *pool["verdict"])
    futures = [server.submit(pool["requests"][str(seed)]) for seed in seeds]
    while server.busy:
        server.step()
    for future in futures:
        future.result()
    if max(observed.batch_rows, default=0) < 2:
        raise RuntimeError("the measurement did not execute a batched engine pass")
    values = {name: observed.values[name] for seed in seeds
              for name in pool["requests"][str(seed)]["questions"]}
    leaves = jax.tree.leaves(variables)
    devices = {device for value in leaves for device in value.devices()}
    hidden = jax.eval_shape(lambda: task.model.backbone.apply(
        part(variables, "backbone"), jnp.zeros((1, 1), jnp.int32), method="hidden_states"))
    return values, {"batch_rows": observed.batch_rows,
                    "compute_dtype": dtype_name(hidden.dtype),
                    "parameter_dtypes": sorted({str(value.dtype) for value in leaves}),
                    "storage": "auto",
                    "platforms": sorted({device.platform for device in devices}),
                    "devices": sorted(str(device) for device in devices),
                    "device_memory": jax.devices()[0].memory_stats()}


def _floating(value):
    import torch

    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _floating(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _floating(item)


@contextlib.contextmanager
def float64_operations():
    """Audit actual Torch operation outputs, not just the model's final dtype."""
    import torch
    from torch.overrides import TorchFunctionMode

    counts = {"floating_operations": 0, "cuda_floating_operations": 0, "audit_seconds": 0.0}

    class Checked(TorchFunctionMode):
        def __torch_function__(self, function, types, args=(), kwargs=None):
            result = function(*args, **(kwargs or {}))
            began = time.perf_counter()
            values = tuple(_floating(result))
            if any(value.dtype != torch.float64 for value in values):
                raise RuntimeError(f"the float64 oracle narrowed in {function}")
            counts["floating_operations"] += bool(values)
            counts["cuda_floating_operations"] += any(value.device.type == "cuda" for value in values)
            counts["audit_seconds"] += time.perf_counter() - began
            return result

    with Checked():
        yield counts


@contextlib.contextmanager
def layer_seconds(model, device):
    """Each decoder layer kind's seconds, its weights' offload loading included, and the model's
    forward count: the cost of one request's passes, by where it goes."""
    import torch

    seconds, forwards, began = {}, [0], {}

    def kind(layer):
        return getattr(layer, "layer_type", type(layer).__name__)

    def synchronized():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    def start(layer, _):
        began[layer] = synchronized()

    def stop(layer, _, __):
        seconds[kind(layer)] = seconds.get(kind(layer), 0.0) + synchronized() - began.pop(layer)

    def counted(*_):
        forwards[0] += 1

    handles = [model.register_forward_pre_hook(counted)]
    for layer in model.base_model.layers:
        handles += [layer.register_forward_pre_hook(start), layer.register_forward_hook(stop)]
    try:
        yield seconds, forwards
    finally:
        for handle in handles:
            handle.remove()


def _spent(seconds, forwards, weights, counts):
    """The running totals a request's cost is read off, as their difference across it."""
    return {"layers": dict(seconds), "forwards": forwards[0], "bank_seconds": dict(weights.seconds),
            "bank_reads": weights.reads, "audit_seconds": counts["audit_seconds"],
            "floating_operations": counts["floating_operations"],
            "cuda_floating_operations": counts["cuda_floating_operations"]}


def _across(before, after):
    """`after`'s running totals less `before`'s, key by key, nested ones too."""
    if isinstance(after, dict):
        return {key: _across(before.get(key, 0), value) for key, value in after.items()}
    return after - before


def author(args, pool, *, truth, commit):
    from unittest.mock import patch

    import numpy as np
    import torch
    import transformers
    from slot_readout_reference import oracle_precision, runtime

    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    measured = None
    with contextlib.ExitStack() as stack:
        if truth:
            import jax

            from dew.inference.banks import SafetensorsBanks
            from dew.interop import Pretrained

            stack.enter_context(jax.enable_x64(new_val=True))
            bundle = Pretrained.load(args.source, weights=False, dtype="float64", param_dtype="float64",
                                      attention_impl="reference")
            if bundle.head.tied_labels() is None or jax.tree.leaves(bundle.head_variables):
                raise ValueError("this released-model oracle requires a parameter-free slot_lm readout")
            banks = stack.enter_context(SafetensorsBanks(args.source, param_dtype="float64",
                                                          cache_bytes=0, read_ahead=False))
            model, weights = offloaded_model(args.source, banks, bundle, device=torch.device(args.device))
            if type(model) is not transformers.Qwen3_5ForCausalLM:
                raise ValueError("the qualified releases use the author's Qwen3.5 causal decoder")
            stack.enter_context(patch.object(transformers.Qwen3_5ForCausalLM, "from_pretrained",
                                              return_value=model))
            stack.enter_context(oracle_precision())
            counts = stack.enter_context(float64_operations())
            seconds, forwards = stack.enter_context(layer_seconds(model, torch.device(args.device)))
            measured = functools.partial(_spent, seconds, forwards, weights, counts)
        _, schema, _ = stack.enter_context(runtime())
        engine_type = importlib.import_module("slot_reference.engine").DecisionEngine
        engine = engine_type(str(args.source), device=args.device,
                             dtype="float64" if truth else args.dtype,
                             head_dtype="float64" if truth else "float32", fork_mode="batched")
        if truth and engine.model is not model:
            raise RuntimeError("the author runtime did not retain the bank-fed model")
        if args.partial is not None:
            args.partial.mkdir(parents=True, exist_ok=True)
        pool_hash = hashlib.sha256(args.pool.read_bytes()).hexdigest()
        values, requests = {}, []
        for index, seed in enumerate((*pool["pilot"], *pool["verdict"])):
            body = pool["requests"][str(seed)]
            record = None if args.partial is None else args.partial / f"{seed}.npz"
            if record is not None and record.exists():
                # A request an earlier run of this phase recorded, over this pool, is read back.
                spent = json.loads(record.with_suffix(".json").read_text())
                if spent["pool_sha256"] != pool_hash:
                    raise ValueError(f"{record} was recorded over another request pool")
                with np.load(record) as held:
                    values.update({name: held[name] for name in held.files})
                requests.append(spent)
                continue
            began = time.perf_counter()
            before = {} if measured is None else measured()
            _, details = engine.decide(schema.SystemOneRequest.model_validate(body), debug=True)
            asked = {}
            for name in body["questions"]:
                logits = details["diagnostics"][name]["slot_logits"]
                if len(logits) != 1:
                    raise ValueError("the qualification pool has one uncalibrated branch per question")
                asked[name] = np.asarray(logits[0], np.float64 if truth else np.float32)
            values.update(asked)
            spent = {"seed": seed, "index": index, "seconds": time.perf_counter() - began,
                     **({} if measured is None else _across(before, measured())),
                     "pool_sha256": pool_hash, "commit": commit}
            if torch.device(args.device).type == "cuda":
                spent["peak_device_bytes"] = torch.cuda.max_memory_allocated(args.device)
            requests.append(spent)
            print(json.dumps(spent), flush=True)
            if record is not None:
                partial = record.with_name(f"{seed}.partial.npz")
                np.savez(partial, **asked)
                record.with_suffix(".json").write_text(json.dumps(spent) + "\n")
                partial.replace(record)
        metadata = {"device": str(engine.device),
                    "compute_dtype": str(engine.torch_dtype).removeprefix("torch."),
                    "parameter_dtypes": sorted({str(value.dtype).removeprefix("torch.")
                                                for value in engine.model.parameters()}),
                    "storage": "float64" if truth else args.dtype,
                    "precision": "ieee"}
        if truth:
            # Every request's own operations, across the runs that recorded them.
            counts = {key: sum(spent[key] for spent in requests)
                      for key in ("floating_operations", "cuda_floating_operations")}
            if not counts["floating_operations"]:
                raise RuntimeError("the float64 oracle executed no observed floating operations")
            metadata.update(counts)
            metadata.update(bank_reads=weights.reads, peak_bank_bytes=weights.peak_bank_bytes,
                            whole_checkpoint_bytes=weights.whole_bytes, bank_stats=asdict(banks.stats()))
        if torch.device(args.device).type == "cuda":
            metadata["peak_device_bytes"] = max(
                [torch.cuda.max_memory_allocated(args.device),
                 *(spent["peak_device_bytes"] for spent in requests if "peak_device_bytes" in spent)])
        metadata["requests"] = requests
        return values, metadata


def verdict(args, pool):
    """Check 1 of native against the author's runtime, each question's option log-probabilities from the
    float64 truth's. The count is sized paired, native's errors against the author's on the pilots
    the records hold (`recorded_pilots`), and the verdict judges its own draws at it, never a pilot.
    Pairing holds because native and the author read the same released weights at their stored
    precision and the same recorded inputs, so they share their upstream rounding."""
    import numpy as np
    from reference_error import Inconclusive, assert_as_exact, leaf_summary, recorded_pilots, seeds_for

    phases = ("native", "author", "truth")
    metadata = {phase: json.loads((args.records / f"{phase}.json").read_text()) for phase in phases}
    pool_hash = hashlib.sha256(args.pool.read_bytes()).hexdigest()
    source = source_identity(args.source)
    for phase, record in metadata.items():
        if record["phase"] != phase or record["pool_sha256"] != pool_hash:
            raise ValueError(f"{phase} was not recorded for this phase and request pool")
        if record["compute_dtype"] != ("float64" if phase == "truth" else args.dtype):
            raise ValueError(f"{phase} was recorded at a different compute dtype")
        expected_storage = "auto" if phase == "native" else "float64" if phase == "truth" else args.dtype
        if record["storage"] != expected_storage or not record["parameter_dtypes"]:
            raise ValueError(f"{phase} was recorded at a different parameter storage policy")
        if record["source"] != source:
            raise ValueError(f"{phase} was recorded from a different source manifest")
    if not metadata["native"].get("commit"):
        raise ValueError("native was recorded without its commit")
    device = args.device.partition(":")[0]
    if metadata["native"]["platforms"] != ["gpu" if device == "cuda" else device]:
        raise ValueError("the native computation ran on a different device platform")
    for phase in ("author", "truth"):
        record = metadata[phase]
        if record["device"].partition(":")[0] != device or record["precision"] != "ieee":
            raise ValueError(f"{phase} was recorded on a different device or precision policy")
    truth = metadata["truth"]
    if truth["parameter_dtypes"] != ["float64"] or truth["floating_operations"] < 1:
        raise ValueError("the recorded truth did not execute the float64 model")
    if device == "cuda" and truth["cuda_floating_operations"] < 1:
        raise ValueError("the recorded truth executed no floating CUDA operations")
    loaded = {phase: np.load(args.records / f"{phase}.npz") for phase in phases}
    # Every phase holds the pilots and the verdict's draws, native's in one record of one commit.
    asked = {name for seed in (*pool["pilot"], *pool["verdict"])
             for name in pool["requests"][str(seed)]["questions"]}
    for phase, values in loaded.items():
        if set(values.files) != asked:
            raise ValueError(f"{phase} does not hold this pool's complete disjoint draw set")
        if phase == "truth" and any(value.dtype != np.float64 for value in values.values()):
            raise ValueError("the recorded truth logits were narrowed after execution")

    def log_probabilities(phase, index):
        """A draw's leaf: each question's log-softmax over its own options, in the phase's precision,
        its questions in turn. The claim is the options' probabilities, so their shared shift is out."""
        def normalized(logits):
            top = logits.max()
            return logits - (top + np.log(np.sum(np.exp(logits - top))))

        names = pool["requests"][str(int(index))]["questions"]
        return {"log_probabilities": np.concatenate([normalized(loaded[phase][name]) for name in names])}

    def served(index):
        return log_probabilities("native", index)

    def original(index):
        return log_probabilities("author", index)

    def exact(index):
        return log_probabilities("truth", index)

    def recorded(index):
        return {"log_probabilities": leaf_summary(original(index)["log_probabilities"],
                                                  exact(index)["log_probabilities"])}

    def draw(seed):
        return (np.asarray(seed, np.int64),)

    original.recorded = recorded.recorded = exact.recorded = True
    count = seeds_for(recorded_pilots(original, len(pool["pilot"])), draw, "released slot runtime",
                      dew_fn=served, truth_fn=exact)
    if count > pool["count"]:
        raise Inconclusive(f"the reference requires {count} draws; the fixed set has only {pool['count']}")
    assert_as_exact(served, recorded, draw, "released slot runtime", seeds=count, truth_fn=exact)
    args.output.write_text(json.dumps({"seeds": count, "verdict": "green", "source": source,
                                      "pool_sha256": pool_hash, "phases": metadata}, indent=2) + "\n")


def source_identity(directory):
    """The requested local source, including weights, readout, processor and configuration bytes."""
    from dew.interop.sources import METADATA_PATTERNS

    root = directory.resolve()
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and (path.suffix in (".safetensors", ".pt")
                               or any(path.match(pattern) for pattern in METADATA_PATTERNS)):
            with path.open("rb") as stream:
                files[path.relative_to(root).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"files": files}


def main():
    import numpy as np

    sys.path.insert(0, str(ROOT / "tests"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("pool", "native", "author", "truth", "verdict"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--pool", type=Path)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--partial", type=Path,
                        help="a directory each request's record is written to as it finishes, and an "
                             "earlier run's are read back from, so a stopped phase resumes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--questions", type=int, default=8)
    parser.add_argument("--short-words", type=int, default=8)
    parser.add_argument("--long-words", type=int, default=256)
    args = parser.parse_args()
    if args.phase == "pool":
        make_pool(args)
        return
    if args.pool is None or args.source is None:
        parser.error("a measurement needs its pool and local source")
    pool = json.loads(args.pool.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.phase == "verdict":
        if args.records is None:
            parser.error("a verdict needs its recorded phases")
        verdict(args, pool)
        return
    from reference_source import git_head

    commit = git_head(ROOT)
    values, metadata = (native(args, pool) if args.phase == "native"
                        else author(args, pool, truth=args.phase == "truth", commit=commit))
    metadata.update(phase=args.phase,
                    pool_sha256=hashlib.sha256(args.pool.read_bytes()).hexdigest(), commit=commit,
                    source=source_identity(args.source),
                    peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    packages={name: importlib.metadata.version(name)
                              for name in ("jax", "torch", "transformers", "accelerate", "numpy")})
    np.savez(args.output, **values)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
