#!/usr/bin/env python3
"""Read the real gold-v3 token-budget path on armada and report observed shapes."""

from __future__ import annotations

import argparse
import collections
import contextlib
import cProfile
import hashlib
import io
import itertools
import json
import logging
import pstats
import time
from pathlib import Path

import numpy as np

from dew.data.batching import TOKEN_COUNTS
from dew.data.dataset import DataPartition, Loading
from dew.data.recipes import load
from dew.data.text import HFTokenizer
from dew.decision.data import Decisions
from dew.decision.layout import JointLayout, Specials
from dew.decision.objective import DecisionObjective
from dew.nn.backbones.causal_transformer import CausalTransformer


class Requests(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.methods: collections.Counter[str] = collections.Counter()

    def emit(self, entry: logging.LogRecord) -> None:
        arguments = entry.args
        if isinstance(arguments, tuple) and len(arguments) > 3:
            method = arguments[3]
            if isinstance(method, str) and method in ("GET", "HEAD", "PUT", "POST", "DELETE"):
                self.methods[method] += 1


@contextlib.contextmanager
def count_requests():
    logger = logging.getLogger("urllib3.connectionpool")
    level, handlers, propagate = logger.level, logger.handlers, logger.propagate
    counter = Requests()
    logger.setLevel(logging.DEBUG)
    logger.handlers, logger.propagate = [counter], False
    try:
        yield counter
    finally:
        logger.setLevel(level)
        logger.handlers, logger.propagate = handlers, propagate


def main() -> None:
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="r2://dew-data/decision/gold-v3")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--micro-tokens", type=int, default=24576)
    parser.add_argument("--batch-rows", type=int)
    parser.add_argument("--profile-open", type=Path)
    parser.add_argument("--descriptor")
    args = parser.parse_args()
    pinned = snapshot_download(
        "Qwen/Qwen3.5-2B",
        revision="15852e8c16360a2fea060d615a32b45270f8a8fc",
        allow_patterns=["*.json", "*.jinja", "*.model", "*.txt"],
    )
    tokenizer = HFTokenizer(pinned, local_files_only=True)
    objective = DecisionObjective(
        CausalTransformer(
            vocab_size=tokenizer.vocab_size, emb_features=8, num_layers=1, num_heads=1, mlp_features=16
        ),
        tokenizer=tokenizer,
        specials=Specials.of(tokenizer),
        layout=JointLayout(max_len=3072, max_state_tokens=1024),
        shuffle_options=True,
        bucket=256 if args.batch_rows is not None else None,
    )
    started = time.monotonic()
    profiler = cProfile.Profile()
    with count_requests() as requests, profiler:
        dataset = load(
            Decisions(args.root, descriptor=args.descriptor),
            batch=args.batch_rows,
            tokens=args.micro_tokens if args.batch_rows is None else None,
            read="indexed",
            objective=objective.data(),
            seed=0,
            loading=Loading(workers=0, threads=4, read_buffer=16),
        )
    if args.profile_open is not None:
        profiler.dump_stats(args.profile_open)
        report = io.StringIO()
        pstats.Stats(profiler, stream=report).sort_stats("cumulative").print_stats(20)
        args.profile_open.with_suffix(".txt").write_text(report.getvalue())
    opened = time.monotonic()
    reader = dataset.train(DataPartition())
    shapes = collections.Counter()
    real_tokens = rows = 0
    first_seconds = None
    try:
        for batch in itertools.islice(reader, args.batches):
            shape = tuple(batch["tokens"].shape)
            shapes[str(shape)] += 1
            if args.batch_rows is None and shape[0] * shape[1] != args.micro_tokens:
                raise ValueError(f"token budget {args.micro_tokens} emitted {shape}")
            counts = np.sum(batch["valid"], axis=1)
            if args.batch_rows is None:
                np.testing.assert_array_equal(batch[TOKEN_COUNTS], counts)
            real_tokens += int(np.sum(counts))
            rows += shape[0]
            if first_seconds is None:
                first_seconds = time.monotonic() - started
        position = reader.get_state()
        expected = next(reader)
    finally:
        reader.close()
    resumed = dataset.train(DataPartition())
    resumed.set_state(position)
    try:
        after = next(resumed)
        for field in expected:
            np.testing.assert_array_equal(after[field], expected[field], err_msg=field)
    finally:
        resumed.close()
    if not shapes or (args.batch_rows is None and len(shapes) > 6):
        raise ValueError(f"the budget path emitted {len(shapes)} shapes, expected at most six")
    result = {
        "root": args.root,
        "micro_tokens": args.micro_tokens,
        "batch_rows": args.batch_rows,
        "rows": rows,
        "batches": sum(shapes.values()),
        "real_tokens": real_tokens,
        "observed_shapes": dict(shapes),
        "open_seconds": opened - started,
        "open_http_requests": sum(requests.methods.values()),
        "open_http_methods": dict(requests.methods),
        "open_head_requests": requests.methods["HEAD"],
        "descriptor": args.descriptor,
        "first_batch_seconds": first_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "resume_exact": True,
        "position_sha256": hashlib.sha256(position).hexdigest(),
        "plan": None if dataset.plan is None else dataset.plan.record(),
    }
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
