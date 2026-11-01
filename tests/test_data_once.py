"""dew.data.once: a value an instance computes once, under a lock of its own (row 3216)."""

import threading

from dew.data.once import locked_cached_property


def test_threads_that_first_read_a_value_at_once_read_the_one_its_instance_computed():
    """Of the threads that first read an instance's locked cached property at once, one computes it and
    every one reads that value; another instance computes its own meanwhile, waiting on none of it; a value
    taken from its instance is computed afresh at its next read."""
    started, release = threading.Event(), threading.Event()
    computed = []

    class Held:
        def __init__(self, slow: bool):
            self.slow = slow

        @locked_cached_property
        def value(self) -> list[bool]:
            computed.append(self.slow)
            if self.slow:
                started.set()
                release.wait(60)
            return [self.slow]

    slow, quick = Held(slow=True), Held(slow=False)
    read = []
    threads = [threading.Thread(target=lambda: read.append(slow.value)) for _ in range(6)]
    for thread in threads:
        thread.start()
    started.wait(60)
    try:
        assert quick.value == [False]
    finally:
        release.set()
        for thread in threads:
            thread.join(60)
    assert len(read) == 6 and all(value is read[0] for value in read)
    assert computed == [True, False]
    assert Held.value.taken(slow) is read[0]
    assert slow.value is not read[0] and computed == [True, False, True]
