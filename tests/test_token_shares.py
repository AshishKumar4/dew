"""A row's segments share its room (`dew.data.tokens.share_tokens`, row 1616): a segment kept whole takes
all it holds, the others what is left, each up to its cap, max-min within one token, as an exhaustive
search over every allocation finds; a decision row's copies of its state are its first user."""

import itertools

import numpy as np
import pytest

from dew.data.tokens import share_tokens


def leximin(wants: list[int], room: int) -> set[tuple[int, ...]]:
    """Every allocation of at most `room` among segments that want `wants`, each at most its want, that
    gives out all it can and whose shares sorted ascending are the greatest, found by trying them
    all: the max-min fair ones."""
    total = min(sum(wants), max(room, 0))
    full = [shares for shares in itertools.product(*(range(want + 1) for want in wants))
            if sum(shares) == total]
    best = max(sorted(shares) for shares in full)
    return {shares for shares in full if sorted(shares) == best}


def test_segments_share_the_room_each_within_its_cap_and_a_whole_one_takes_all():
    """Segments that fit keep all they want, their lengths up to their caps; where they do not, the
    share is max-min within one token in stable ascending order of want ([10, 10] into 7 keeps [3, 4]),
    never below zero; a whole segment keeps all it holds, past the room and past its cap too, and the
    others share what it leaves."""
    assert share_tokens([10, 10], 100, caps=[None, 4]) == [10, 4]
    assert share_tokens([10, 10], 7) == [3, 4]
    assert share_tokens([10, 10, 10], 11, caps=[2, None, None]) == [2, 4, 5]
    assert share_tokens([100, 10, 10], 4) == [2, 1, 1]
    assert share_tokens([10, 10], -3) == [0, 0]
    assert share_tokens([6, 10, 10], 10, caps=[2, None, None], whole=[True, False, False]) == [6, 2, 2]
    assert share_tokens([12, 10], 10, whole=[True, False]) == [12, 0]
    with pytest.raises(ValueError):
        share_tokens([1, 2], 3, caps=[None])


def test_every_small_allocation_is_max_min_fair():
    """Over drawn small rows (up to four segments of up to four tokens, caps, whole segments and rooms
    below zero to past the total), each cut segment keeps at most its want, and their shares are one of
    the max-min fair allocations an exhaustive search finds."""
    rng = np.random.default_rng(0x5A)
    for _ in range(2000):
        count = int(rng.integers(1, 5))
        lengths = [int(length) for length in rng.integers(0, 5, count)]
        caps = [None if rng.random() < 0.5 else int(rng.integers(0, 5)) for _ in range(count)]
        whole = [bool(rng.random() < 0.25) for _ in range(count)]
        room = int(rng.integers(-2, 18))
        shares = share_tokens(lengths, room, caps=caps, whole=whole)
        cut = [at for at in range(count) if not whole[at]]
        wants = [lengths[at] if caps[at] is None else min(caps[at], lengths[at]) for at in cut]
        assert all(shares[at] == lengths[at] for at in range(count) if whole[at])
        left = room - sum(lengths[at] for at in range(count) if whole[at])
        assert tuple(shares[at] for at in cut) in leximin(wants, left), (lengths, caps, whole, room)
