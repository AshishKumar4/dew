"""URL catalogs and volatile fetches, executed by the ordinary Grain worker path."""

from __future__ import annotations

import hashlib
import http.client
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from dew.data.dataset import Batch


@dataclass(frozen=True)
class Fetch:
    """Fetch bytes into a record. Only permanent failures may become cache tombstones."""

    field: str = "url"
    output: str = "bytes"
    timeout: float = 30.0
    retries: int = 2
    max_bytes: int = 64 * 2**20
    volatile: ClassVar[bool] = True

    def __call__(self, record: Mapping[str, object]):
        from dew.data.pipeline import Rejected

        url = record.get(self.field)
        if not isinstance(url, str):
            raise ValueError(f"Fetch needs a string field {self.field!r}")
        for attempt in range(self.retries + 1):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "dew-data/1"})
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = response.read(self.max_bytes + 1)
                if len(data) > self.max_bytes:
                    return Rejected("too_large")
                result: Batch = {**record, self.output: data, "content_sha256": hashlib.sha256(data).hexdigest()}
                return result
            except urllib.error.HTTPError as error:
                if error.code in (404, 410):
                    return Rejected(f"http_{error.code}")
                if error.code < 500 and error.code not in (408, 429):
                    return Rejected(f"http_{error.code}")
            except ValueError:
                return Rejected("invalid_url")
            except (OSError, http.client.HTTPException):
                pass
            if attempt < self.retries:
                time.sleep(0.1 * 2**attempt)
        return Rejected("transient", permanent=False)
