"""URL catalogs and volatile fetches, executed by the ordinary Grain worker path."""

from __future__ import annotations

import hashlib
import http.client
import logging
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fetch:
    """Fetch the bytes at a record's url into `output`, in one attempt: a url
    that fails now is dropped as transient, for a later epoch or a cache
    refill to fetch again, so a reader waits at most `timeout` on it. Only
    permanent failures (404, 410, a malformed url, an oversized object) may
    become cache tombstones."""

    field: str = "url"
    output: str = "bytes"
    timeout: float = 10.0
    max_bytes: int = 64 * 2**20
    volatile: ClassVar[bool] = True
    rejects: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if self.timeout <= 0 or self.max_bytes < 1:
            raise ValueError("Fetch needs a positive timeout and max_bytes")

    def __call__(self, record: Mapping[str, object]):
        from dew.data.pipeline import Rejected

        if self.field not in record:
            raise ValueError(f"Fetch reads its url from {self.field!r}; the record has {sorted(record)}")
        url = record[self.field]
        if not isinstance(url, str):
            return Rejected("invalid_url")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "dew-data/1"})
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = response.read(self.max_bytes + 1)
        except urllib.error.HTTPError as error:
            if error.code in (404, 410):
                return Rejected(f"http_{error.code}")
            _log.debug("fetch %s returned HTTP %d", url, error.code)
            return Rejected("transient", permanent=False)
        except ValueError:
            return Rejected("invalid_url")
        except (OSError, http.client.HTTPException) as error:
            _log.debug("fetch %s failed: %s", url, error)
            return Rejected("transient", permanent=False)
        if len(payload) > self.max_bytes:
            return Rejected("too_large")
        return {**record, self.output: payload, "content_sha256": hashlib.sha256(payload).hexdigest()}
