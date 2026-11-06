import json, sys
sys.path.insert(0, "tests")
from uuid import uuid4
from test_parquet_files import written
from dew.data.cache import Cache
from dew.data.compile import publish
from dew.data.generation import Generation
from dew.data.pipeline import Pipeline
from dew.data.sources import Files
url = written(f"probe-{uuid4().hex}")
ms = {}
for v in (9, 10, 10):
    g = publish(Pipeline(Files(url, format="parquet")), Cache(f"memory://probe-{uuid4().hex}"), read="streamed", plan_version=v)
    ms.setdefault(v, []).append(Generation(g.url).manifest)
print(sorted(ms[10][0]))
a, b = ms[10]
print("differ between two v10:", [k for k in a if a[k] != b.get(k)])
print("differ v9 vs v10:", [k for k in a if a[k] != ms[9][0].get(k)])
