"""Scratch: profile a thousand-source mixture's compile and steps (not for landing)."""

import cProfile
import io
import itertools
import pstats

from test_mixture_scale import BATCH, READ, sources

from dew.data.compile import compile
from dew.data.dataset import DataPartition
from dew.data.pipeline import mix


def report(profile, label):
    out = io.StringIO()
    pstats.Stats(profile, stream=out).sort_stats("tottime").print_stats(18)
    lines = [line[:150] for line in out.getvalue().splitlines() if "/" in line or "ncalls" in line]
    print(f"===== {label}\n" + "\n".join(lines))


def test_profile():
    profile = cProfile.Profile()
    profile.enable()
    data = compile(mix(sources(1000)).batch(BATCH), read="memory", loading=READ)
    profile.disable()
    report(profile, "compile")
    assert False, "compile"
    stream = data.train(DataPartition())
    list(itertools.islice(stream, 8))
    profile = cProfile.Profile()
    profile.enable()
    list(itertools.islice(stream, 64))
    profile.disable()
    stream.close()
    report(profile, "steps")
