"""Qualify released slot engines against the authors' independently streamed float64 runtime.

The native and author phases run in separate processes. LayerBanks owns weight reads and layouts;
the author's Transformers model owns all forward arithmetic and Accelerate owns its offload hooks.
One qualification is one checkout: the pool and every phase record name its commit, and the verdict
refuses records of different commits. On one GPU, from a checkout:

    PYTHONPATH=src:tests:tools python tools/slot_runtime_qualification.py pool --output run/pool.json
    PYTHONPATH=src:tests:tools python tools/slot_runtime_qualification.py native --source RELEASE \
        --pool run/pool.json --output run/native.npz
    AUTHOR=$(PYTHONPATH=src:tests:tools python tools/slot_runtime_qualification.py environment)
    PYTHONPATH=src:tests:tools JAX_PLATFORMS=cpu $AUTHOR tools/slot_runtime_qualification.py author \
        --source RELEASE --pool run/pool.json --partial run/partial --output run/author.npz
    PYTHONPATH=src:tests:tools JAX_PLATFORMS=cpu $AUTHOR tools/slot_runtime_qualification.py truth \
        --source RELEASE --pool run/pool.json --partial run/partial --output run/truth.npz
    PYTHONPATH=src:tests:tools python tools/slot_runtime_qualification.py verdict --source RELEASE \
        --pool run/pool.json --records run --output run/verdict.json

RELEASE is the release's local directory: the authors' runtime runs causal for a Hub id.
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
AUTHOR_OVERRIDES = ROOT / "locks/reference/torch28-cu128-tf519-v1.override.txt"
"""The author and truth phases' environment over the checkout's CPU lock: the reference transformers and
the CUDA 12.8 torch (`author_python`)."""
AUTHOR_HEAD_DTYPE = "float32"
"""The dtype the author phase builds the authors' engine to hold its readout in, beside a narrow backbone."""
JAX_PLATFORMS = {"gpu": "cuda"}
"""JAX's name for a CUDA device's platform, against torch's: the records name every device as torch does."""


class TorchBankWeights(Mapping):
    """A Transformers state dict read from the native layer banks one layer at a time, each recovered
    tensor then kept on the host at its stored precision, so the whole checkpoint is resident there
    after the first forward (`offloaded_model` refuses one larger than the host's free memory).

    Source names pass through Transformers' published checkpoint conversion mapping. The inverse
    native weight layouts recover the source tensors; non-decoder entries are read one tensor at a time.
    Each tensor, once recovered, is kept on the host at the narrowest float dtype that holds it exactly
    (the release's stored one: bf16 for a bf16 release), so every later forward reads no bank and
    widens it in NumPy, and the release is read and laid out once a run. The
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
        for site in bundle.model.bank_sites(shapes):
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
    pool = {"commit": recorded_commit(), "count": args.count,
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

    def compile(self, variables):
        return self.task.compile(variables)

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

    # Each leaf at the dtype the release stores it in, as the verdict reads the release's headers.
    bundle = Pretrained.load(args.source, dtype=args.dtype, param_dtype="auto", attention_impl="xla")
    decide = Decide.of(bundle)
    # Four rows a pass at the layout's shortest pass length (`Decide.pass_shapes`, its window's halvings),
    # so the server's passes batch: a slot release's 32768-token window pads every pass to 4096 or more.
    shortest = min(shape.length for shape in decide.pass_shapes)
    task = replace(decide, budget=Budget(tokens=4 * shortest, rows=4))
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
                    "parameter_dtypes": sorted({str(value.dtype) for value in leaves
                                                if jnp.issubdtype(value.dtype, jnp.floating)}),
                    "device": ",".join(sorted({JAX_PLATFORMS.get(device.platform, device.platform)
                                               for device in devices})),
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
    """Audit the dtype of every floating output each Torch operation returns."""
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


PROVENANCE = ("phase", "commit", "pool_sha256", "source_sha256", "compute_dtype", "device")
"""The run a record was made in, which admits it: a phase's record, and each request record within it."""

IEEE = {"float32_matmul": "highest", "cuda_tf32": False, "cudnn_tf32": False}
"""The runtime settings under which Torch computes float32 products at IEEE float32 (`precision`)."""


def admit(record, run, what):
    """Refuse `record` unless it was made in `run`: the same phase, commit, pool and source digests,
    compute dtype and device type (`PROVENANCE`). A field the record lacks differs."""
    differing = [field for field in PROVENANCE if record.get(field) != run[field]]
    if differing:
        raise ValueError(f"{what} was recorded in another run: its {differing} differ from this run's")


class Partial:
    """A phase's request records, each written as its request finishes under `directory/<phase>`, so a
    stopped phase resumes: a record is read back only where it was made in this run (`admit`)."""

    def __init__(self, directory, run):
        self.run = run
        self.directory = directory / run["phase"]
        self.directory.mkdir(parents=True, exist_ok=True)

    def read(self, seed):
        """Seed's recorded values and cost, or None where no earlier run recorded it."""
        import numpy as np

        record = self.directory / f"{seed}.npz"
        if not record.exists():
            return None
        spent = json.loads(record.with_suffix(".json").read_text())
        admit(spent, self.run, str(record))
        with np.load(record) as held:
            return {name: held[name] for name in held.files}, spent

    def write(self, seed, values, spent):
        import numpy as np

        record = self.directory / f"{seed}.npz"
        partial = record.with_name(f"{seed}.partial.npz")
        np.savez(partial, **values)
        record.with_suffix(".json").write_text(json.dumps(spent) + "\n")
        partial.replace(record)


def precision():
    """What Torch computes float32 products at, read from the runtime (`IEEE`)."""
    import torch

    return {"float32_matmul": torch.get_float32_matmul_precision(),
            "cuda_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32}


def author(args, pool, run):
    """The authors' engine over the pool's pilots and verdict draws, at the narrow precision, or as the
    float64 truth with its weights streamed from the native banks; `run` is the phase's provenance."""
    from unittest.mock import patch

    import numpy as np
    import torch
    import transformers
    from block_override_reference import oracle_precision, release_package

    truth = run["phase"] == "truth"
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
        _, schema, _ = stack.enter_context(release_package())
        engine_type = importlib.import_module("slot_reference.engine").DecisionEngine
        engine = engine_type(str(args.source), device=args.device,
                             dtype="float64" if truth else args.dtype,
                             head_dtype="float64" if truth else AUTHOR_HEAD_DTYPE, fork_mode="batched")
        if truth and engine.model is not model:
            raise RuntimeError("the author runtime did not retain the bank-fed model")
        run = {**run, "compute_dtype": str(engine.torch_dtype).removeprefix("torch."),
               "device": torch.device(engine.device).type}
        partial = None if args.partial is None else Partial(args.partial, run)
        values, requests = {}, []
        for index, seed in enumerate((*pool["pilot"], *pool["verdict"])):
            body = pool["requests"][str(seed)]
            held = None if partial is None else partial.read(seed)
            if held is not None:
                values.update(held[0])
                requests.append(held[1])
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
                     **({} if measured is None else _across(before, measured())), **run}
            if run["device"] == "cuda":
                spent["peak_device_bytes"] = torch.cuda.max_memory_allocated(args.device)
            requests.append(spent)
            print(json.dumps(spent), flush=True)
            if partial is not None:
                partial.write(seed, asked, spent)
        metadata = {"compute_dtype": run["compute_dtype"], "device": run["device"],
                    "parameter_dtypes": sorted({str(value.dtype).removeprefix("torch.")
                                                for value in engine.model.parameters()}),
                    "precision": precision()}
        if truth:
            # Every request's own operations, across the runs that recorded them.
            counts = {key: sum(spent[key] for spent in requests)
                      for key in ("floating_operations", "cuda_floating_operations")}
            if not counts["floating_operations"]:
                raise RuntimeError("the float64 oracle executed no observed floating operations")
            metadata.update(counts)
            metadata.update(bank_reads=weights.reads, peak_bank_bytes=weights.peak_bank_bytes,
                            whole_checkpoint_bytes=weights.whole_bytes, bank_stats=asdict(banks.stats()))
        if run["device"] == "cuda":
            metadata["peak_device_bytes"] = max(
                [torch.cuda.max_memory_allocated(args.device),
                 *(spent["peak_device_bytes"] for spent in requests if "peak_device_bytes" in spent)])
        metadata["requests"] = requests
        return values, metadata


def verdict(args, pool):
    """Check 1 of native against the author's runtime, each question's option log-probabilities from the
    float64 truth's.

    The records are admitted first, each phase's and each of its requests' (`admit`), against the run
    this verdict expects: this pool and source, the pool's commit, the declared compute dtype and device.
    Each arm's parameters are held at what its storage policy gives the release's stored dtypes: native
    keeps each stored dtype, the author holds its backbone at the compute dtype and its readout at
    AUTHOR_HEAD_DTYPE, and the truth holds all of them in float64; the author and truth computed under
    IEEE float32 products.

    The count is sized paired, native's errors against the author's on the pilots the records hold
    (`recorded_pilots`), and the verdict judges its own draws at that count, never a pilot. The arms read
    the same released backbone weights at their stored precision and the same recorded inputs; their
    readouts differ, native's at the compute dtype and the author's at AUTHOR_HEAD_DTYPE. A count the
    shared part does not set leaves the verdict's interval unresolved, which is Inconclusive."""
    import numpy as np
    import torch
    from reference_error import (
        Inconclusive,
        as_given,
        assert_as_exact,
        leaf_summary,
        recorded_pilots,
        seeds_for,
    )
    from reference_source import dew_revision

    if not pool.get("commit"):
        raise ValueError("the pool names no commit, so its phases cannot be held to one")
    source = source_identity(args.source)
    expected = {"commit": pool["commit"], "pool_sha256": file_sha256(args.pool),
                "source_sha256": manifest_sha256(source), "device": torch.device(args.device).type}
    metadata, seeds = {}, [*pool["pilot"], *pool["verdict"]]
    for phase in ("native", "author", "truth"):
        record = json.loads((args.records / f"{phase}.json").read_text())
        run = {**expected, "phase": phase, "compute_dtype": "float64" if phase == "truth" else args.dtype}
        admit(record, run, phase)
        metadata[phase] = record
        if phase == "native":
            continue
        # The author's and truth's runs record each request, a resumed one among them, in the pool's order.
        if "requests" not in record or [spent.get("seed") for spent in record["requests"]] != seeds:
            raise ValueError(f"{phase} does not hold one request record for each of this pool's pilots and "
                             f"verdict draws, in order")
        for spent in record["requests"]:
            admit(spent, run, f"{phase}'s request {spent['seed']}")
    stored = stored_dtypes(args.source)
    for phase, held in (("native", stored), ("author", sorted({args.dtype, AUTHOR_HEAD_DTYPE})),
                        ("truth", ["float64"])):
        if metadata[phase]["parameter_dtypes"] != held:
            raise ValueError(f"{phase} held its parameters at {metadata[phase]['parameter_dtypes']}; its "
                             f"storage policy over the release's {stored} holds them at {held}")
    for phase in ("author", "truth"):
        if metadata[phase]["precision"] != IEEE:
            raise ValueError(f"{phase} computed float32 products at {metadata[phase]['precision']}, "
                             f"not {IEEE}")
    truth = metadata["truth"]
    if truth["floating_operations"] < 1:
        raise ValueError("the recorded truth executed no floating operations")
    if expected["device"] == "cuda" and truth["cuda_floating_operations"] < 1:
        raise ValueError("the recorded truth executed no floating CUDA operations")
    loaded = {phase: np.load(args.records / f"{phase}.npz") for phase in metadata}
    # Every phase holds the pilots and the verdict's draws, native's in one record of one commit.
    asked = {name for seed in (*pool["pilot"], *pool["verdict"])
             for name in pool["requests"][str(seed)]["questions"]}
    for phase, values in loaded.items():
        if set(values.files) != asked:
            raise ValueError(f"{phase} does not hold this pool's complete disjoint draw set")
        if phase == "truth" and any(value.dtype != np.float64 for value in values.values()):
            raise ValueError("the recorded truth logits were narrowed after execution")

    def log_probabilities(phase, index):
        """A draw's leaf: each question's log-softmax over its own options, its questions in turn. The
        claim is the options' probabilities, so their shared shift is out: each phase's recorded logits,
        in the precision that phase computed them, are normalized in float64 (SciPy's `log_softmax`),
        where a shift as large as a float32 logit's adds no rounding of its own."""
        from scipy.special import log_softmax

        names = pool["requests"][str(int(index))]["questions"]
        return {"log_probabilities": np.concatenate([
            log_softmax(np.asarray(loaded[phase][name], np.float64)) for name in names])}

    original = functools.partial(log_probabilities, "author")
    exact = functools.partial(log_probabilities, "truth")

    def recorded(index):
        return {"log_probabilities": leaf_summary(original(index)["log_probabilities"],
                                                  exact(index)["log_probabilities"])}

    def draw(seed):
        return (np.asarray(seed, np.int64),)

    original.recorded = recorded.recorded = exact.recorded = True
    # Every phase is read from the recordings and normalized on the host, as given.
    native = as_given(functools.partial(log_probabilities, "native"))
    count = seeds_for(recorded_pilots(original, draw, len(pool["pilot"])), draw, "released slot runtime",
                      dew_fn=native, truth_fn=exact)
    if count > pool["count"]:
        raise Inconclusive(f"the reference requires {count} draws; the fixed set has only {pool['count']}")
    assert_as_exact(native, recorded, draw, "released slot runtime", seeds=count, truth_fn=exact)
    args.output.write_text(json.dumps({"commit": dew_revision(), "seeds": count, "verdict": "green",
                                      "source": source, "pool_sha256": expected["pool_sha256"],
                                      "phases": metadata}, indent=2) + "\n")


def source_identity(directory):
    """The source's weight files (safetensors and torch) and every metadata file the loader reads
    (`METADATA_PATTERNS`), each by its path in the source and its sha256."""
    from dew.interop.sources import METADATA_PATTERNS

    root = directory.resolve()
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and (path.suffix in (".safetensors", ".pt")
                               or any(path.match(pattern) for pattern in METADATA_PATTERNS)):
            files[path.relative_to(root).as_posix()] = file_sha256(path)
    return {"files": files}


def manifest_sha256(manifest):
    """One digest of a source's manifest (`source_identity`), as each record names its source."""
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stored_dtypes(directory):
    """The floating dtypes the source's safetensors store, read from their headers."""
    import jax.numpy as jnp

    from dew.interop.safetensors_io import read_weights

    return sorted({str(tensor.dtype) for tensor in read_weights(directory).values()
                   if jnp.issubdtype(tensor.dtype, jnp.floating)})


def recorded_commit():
    """The commit the pool and every phase record name, the checkout's. A checkout that changes a tracked
    file or adds one to Dew's source (`dew._checkout.clean`) is refused, as its records would name code
    they did not run."""
    from reference_source import dew_revision

    from dew import _checkout

    if not _checkout.clean():
        raise SystemExit("the checkout has uncommitted changes, so its records cannot name the code they "
                         "ran; record from a clean checkout")
    return dew_revision()


def recorded_run(args):
    """The provenance a phase records, read before it runs and again after: its phase, the checkout's
    commit (`recorded_commit`), and the pool's and source's digests."""
    return {"phase": args.phase, "commit": recorded_commit(), "pool_sha256": file_sha256(args.pool),
            "source_sha256": manifest_sha256(source_identity(args.source))}


def author_python():
    """The author and truth phases' Python: the checkout's CPU lock under `AUTHOR_OVERRIDES`, built once
    by tools/environments.py and reused while its installation is unchanged."""
    from environments import reference

    return reference("slot-author-cu128-tf519", [
        "-r", str(ROOT / "locks/cpu.txt"), "--override", str(AUTHOR_OVERRIDES),
        "--extra-index-url", "https://download.pytorch.org/whl/cu128",
        "--index-strategy", "unsafe-best-match"])


def main():
    import numpy as np

    sys.path.insert(0, str(ROOT / "tests"))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("pool", "native", "environment", "author", "truth", "verdict"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--pool", type=Path)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--partial", type=Path,
                        help="a directory each request's record is written to as it finishes, by phase, and "
                             "an earlier run's are read back from, so a stopped phase resumes")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--questions", type=int, default=8)
    parser.add_argument("--short-words", type=int, default=8)
    parser.add_argument("--long-words", type=int, default=256)
    args = parser.parse_args()
    if args.phase == "environment":
        print(author_python())
        return
    if args.output is None:
        parser.error(f"the {args.phase} phase writes its --output")
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
    run = recorded_run(args)
    values, metadata = native(args, pool) if args.phase == "native" else author(args, pool, run)
    if recorded_run(args) != run:
        raise RuntimeError("the pool or the source changed while the phase ran, so its record names neither")
    metadata.update(run, peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    packages={name: importlib.metadata.version(name)
                              for name in ("jax", "torch", "transformers", "accelerate", "numpy")})
    np.savez(args.output, **values)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
