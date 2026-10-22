"""1064 measurement: Dew's first-fit planner against Grain's FirstFitPackIterDataset driven over
length-only records (broadcast zero int8 arrays), its placement read off segment ids; timings and
plan identity printed as JSON."""
import json
import time

import grain.experimental as gx
import grain.python as grain
import numpy as np

from dew.data.tokens import first_fit


def grain_plan(sizes: np.ndarray, window: int, bins: int, segments: int | None):
    """Grain's own first-fit over records of `sizes` tokens, each holding its index in a meta field,
    read back as Dew's (order, starts): the records each window holds in the order Grain placed them."""
    zero = np.zeros((), np.int8)
    records = [{"text": np.broadcast_to(zero, (int(size),)), "index": np.asarray([index], np.int64)}
               for index, size in enumerate(sizes.tolist())]
    packed = gx.FirstFitPackIterDataset(grain.MapDataset.source(records).to_iter_dataset(),
                                        length_struct={"text": window, "index": window}, num_packing_bins=bins,
                                        shuffle_bins=False, max_sequences_per_bin=segments,
                                        meta_features=("index",))
    order, starts = [], [0]
    for row in packed:
        ids = np.asarray(row["text_segment_ids"])
        held = int(ids.max(initial=0))
        if held == 0:
            continue
        order.extend(np.asarray(row["index"])[: held].tolist())
        starts.append(len(order))
    return np.asarray(order, np.int64), np.asarray(starts, np.int64)


def test_probe():
    out = {}
    rng = np.random.default_rng(0)
    for count in (10_000, 100_000, 1_000_000):
        # fineweb-edu-like document lengths: lognormal around 600 tokens, cut to the window.
        sizes = np.minimum(np.maximum(1, rng.lognormal(6.4, 1.0, count).astype(np.int64)), 2048)
        started = time.perf_counter()
        ours = first_fit(sizes, 2048, 32, None)
        dew = time.perf_counter() - started
        started = time.perf_counter()
        theirs = grain_plan(sizes, 2048, 32, None)
        native = time.perf_counter() - started
        out[count] = {"dew_seconds": dew, "grain_seconds": native,
                      "same": bool(np.array_equal(ours[0], theirs[0]) and np.array_equal(ours[1], theirs[1]))}
        print("PROBE1064 " + json.dumps({count: out[count]}), flush=True)
    print("PROBE " + json.dumps(out))
