"""tools/armada/r2.py, the R2 Worker's one client, against the stand-in Worker (conftest's cache_worker): an
object round-trips, a miss is None, a listing pages to its end, and a key or a body the Worker's contract
refuses never reaches it."""

import pytest
from remote_cache import r2


def test_an_object_round_trips_a_miss_is_none_and_a_listing_pages_to_its_end(cache_worker):
    url, _, _ = cache_worker
    store = r2.Store(url, "token", "dew-test", timeout=10)
    for name in ("a", "b", "c"):
        store.put(f"records/{name}", name.encode())
    assert store.get("records/b") == b"b" and store.get("records/d") is None
    assert list(store.keys("records/")) == ["a", "b", "c"]
    with pytest.raises(r2.Refused):
        r2.Store(url, "another", "dew-test", timeout=10).get("records/a")


def test_what_the_contract_refuses_never_reaches_the_worker(cache_worker):
    url, stored, seen = cache_worker
    store = r2.Store(url, "token", "dew-test", timeout=10)
    for key in ("records/../a", "records/a b", "x" * (r2.CONTRACT["max_key_length"] + 1)):
        with pytest.raises(ValueError):
            store.put(key, b"value")
    with pytest.raises(ValueError):
        store.put("records/large", bytes(r2.MAX_BYTES + 1))
    assert seen == [] and stored == {}
