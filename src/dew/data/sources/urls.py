"""URL catalogs and volatile fetches, executed by the ordinary Grain worker path."""

from __future__ import annotations

import hashlib
import http.client
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from dew.data.dataset import Batch

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fetch:
    """Fetch bytes into a record. Only permanent failures may become cache tombstones."""

    field: str = "url"
    output: str = "bytes"
    timeout: float = 30.0
    retries: int = 2
    max_bytes: int = 64 * 2**20
    volatile: ClassVar[bool] = True
    rejects: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if self.timeout <= 0 or self.retries < 0 or self.max_bytes < 1:
            raise ValueError("Fetch needs positive timeout and max_bytes, and nonnegative retries")

    def __call__(self, record: Mapping[str, object]):
        from dew.data.pipeline import Rejected

        url = record.get(self.field)
        if not isinstance(url, str):
            raise ValueError(f"Fetch needs a string field {self.field!r}")
        for attempt in range(self.retries + 1):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "dew-data/1"})
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = response.read(self.max_bytes + 1)
                if len(payload) > self.max_bytes:
                    return Rejected("too_large")
                outcome: Batch = {
                    **record,
                    self.output: payload,
                    "content_sha256": hashlib.sha256(payload).hexdigest(),
                }
                return outcome
            except urllib.error.HTTPError as error:
                if error.code in (404, 410):
                    return Rejected(f"http_{error.code}")
                _log.debug("fetch %s attempt %d returned HTTP %d", url, attempt + 1, error.code)
            except ValueError:
                return Rejected("invalid_url")
            except (OSError, http.client.HTTPException) as error:
                _log.debug("fetch %s attempt %d failed: %s", url, attempt + 1, error)
            if attempt < self.retries:
                time.sleep(min(0.1 * 2**attempt, 10))
        return Rejected("transient", permanent=False)
