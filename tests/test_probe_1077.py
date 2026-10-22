"""Probe for 1077: the export manifest and success marker bytes, the writer id normalised, and a provisioned
ArrayRecord directory's manifest, printed as JSON for an old/new comparison."""
import hashlib
import json
import re

import numpy as np
from array_record.python.array_record_module import ArrayRecordWriter

from dew.data.sinks import export
from dew.data.sources import Files, Memory


def test_probe(tmp_path):
    rows = [{"id": np.int64(index), "text": np.arange(index % 5, dtype=np.int32), "_dew_length": index % 5 + 1}
            for index in range(23)]
    out = {}
    for format in ("parquet", "array_record"):
        directory = tmp_path / format
        returned = export(Memory(rows), str(directory), format=format, rows_per_shard=5, metadata={"k": 1})
        raw = (directory / "_dew_manifest.json").read_bytes()
        writer = re.search(rb"shards/([0-9a-f]+)/", raw).group(1)
        out[format] = {
            "manifest": raw.replace(writer, b"W").decode(),
            "success_names_manifest": (directory / "_SUCCESS").read_bytes()
            == hashlib.sha256(raw).hexdigest().encode(),
            "returned": json.dumps(returned, sort_keys=True).replace(writer.decode(), "W"),
        }
    provisioned = tmp_path / "provisioned"
    provisioned.mkdir()
    for number in range(2):
        writer = ArrayRecordWriter(str(provisioned / f"part-{number}.array_record"), "group_size:1")
        for index in range(3 + number):
            writer.write(f"row-{number}-{index}".encode())
        writer.close()
    files = Files(str(provisioned), format="array_record")
    files.provision()
    raw = (provisioned / "_dew_manifest.json").read_bytes()
    out["provisioned"] = {"manifest": json.loads(raw), "success": (provisioned / "_SUCCESS").exists(),
                          "rows": len(files.records("indexed"))}
    print("PROBE " + json.dumps(out, sort_keys=True))
