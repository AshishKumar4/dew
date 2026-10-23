"""Debug (not for landing): a pytest plugin that runs check 2 (`certified`) on each check-1 site's Dew arm
at its first draw and prints what it finds, before the site's own check 1."""
import json
import os
import sys

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item):
    import op_certificates
    import reference_error

    original = reference_error.assert_as_exact

    def censused(dew_fn, reference_fn, draw, label, *args, **kwargs):
        try:
            _, findings = op_certificates.certified(dew_fn, *draw(0))
        except Exception as error:  # noqa: BLE001 - a census records what it cannot run
            findings = [f"raised {type(error).__name__}: {str(error)[:200]}"]
        print("CENSUS " + json.dumps({"site": os.environ.get("PYTEST_CURRENT_TEST", "").split(" ")[0],
                                      "label": label, "findings": findings[:40]}), flush=True)
        return original(dew_fn, reference_fn, draw, label, *args, **kwargs)

    for module in list(sys.modules.values()):
        if getattr(module, "assert_as_exact", None) is original:
            module.assert_as_exact = censused
