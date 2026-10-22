"""SLOP014: a layer's call reads what its block hands it, and a mixer's `reads` is what it reads."""

import textwrap

import pytest
from tool_support import load


@pytest.fixture(scope="module")
def lint():
    return load("lint_slop")


INPUTS = 'MIXING = ("pairwise_mask", "position_bias", "admitted")\n'
CASES = {
    "a call argument taken and deleted": ("""
        class Layer:
            def __call__(self, x, decode=False, kv_store=None):
                del kv_store
                return x if decode else -x
        """, True),
    "every call argument read": ("""
        class Layer:
            def __call__(self, x, decode=False, kv_store=None):
                return x if decode or kv_store else -x
        """, False),
    "a mask read without being declared": ("""
        class Mixer:
            reads = frozenset()

            def __call__(self, x, call):
                return x if call.metadata.pairwise_mask is None else -x
        """, True),
    "a mask declared and never read": ("""
        class Mixer:
            reads = frozenset({"pairwise_mask"})

            def __call__(self, x, call):
                return x if call.decode else -x
        """, True),
    "a mask declared and read in a helper of the module": ("""
        def masked(metadata):
            return metadata.pairwise_mask

        class Mixer:
            reads = frozenset({"pairwise_mask"})

            def __call__(self, x, call):
                return x if masked(call.metadata) is None else -x
        """, False),
    "a field no call sets": ("""
        class Mixer:
            reads = frozenset({"segments"})

            def __call__(self, x, call):
                return x if call.decode else -x
        """, True),
    "reads spelled some other way": ("""
        class Mixer:
            reads = MASKS

            def __call__(self, x, call):
                return x if call.decode else -x
        """, True),
    "a tokenwise mixer that ignores its call": ("""
        class Mixer:
            reads = TOKENWISE

            def __call__(self, x, call):
                return 2 * x
        """, False),
    "a tokenwise mixer that reads a mask": ("""
        class Mixer:
            reads = TOKENWISE

            def __call__(self, x, call):
                return x if call.metadata.admitted is None else -x
        """, True),
}


@pytest.mark.parametrize("case", list(CASES))
def test_a_call_that_drops_its_arguments_or_misdeclares_its_reads_is_reported(lint, case, tmp_path):
    body, reported = CASES[case]
    (tmp_path / "src" / "dew" / "nn").mkdir(parents=True)
    (tmp_path / "src" / "dew" / "nn" / "inputs.py").write_text(INPUTS)
    (tmp_path / "src" / "dew" / "nn" / "layer.py").write_text(textwrap.dedent(body))
    [module] = [module for module in lint.collect(["src/dew/nn"], tmp_path, "dew")
                if module.path.stem == "layer"]
    assert any(True for _ in lint.calls(module)) is reported


@pytest.mark.parametrize("body, reported", [
    ("assert step._cache_size() == 1\n", True),
    ("with Compiles('step') as compiled:\n    step()\nassert compiled.count == 1\n", False),
], ids=["private cache size", "public compile count"])
def test_a_read_of_jaxs_private_cache_size_is_reported(lint, body, reported, tmp_path):
    """SLOP021: a jitted function's private `_cache_size` is read nowhere; its compiles are counted
    through the public compile event (`dew.telemetry.instrumentation.Compiles`)."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_counted.py").write_text(body)
    [module] = lint.collect(["tests"], tmp_path, "dew")
    assert any(finding.code == "SLOP021" for finding in lint.check(module)) is reported


PULLED_AS_IT_WAS = """
import jax
from reference_error import assert_as_exact


def pulled(fn):
    def run(operands):
        outputs, pull = jax.vjp(fn, operands)
        keys = jax.random.split(jax.random.fold_in(jax.random.key(0), 1 << 20), len(outputs))
        (grads,) = pull(tuple(jax.random.normal(key, out.shape, out.dtype)
                              for key, out in zip(keys, outputs, strict=True)))
        return grads
    return run


def test_gradients(dew, reference, draw):
    assert_as_exact(pulled(dew), pulled(reference), draw, "gradients", seeds=8, truth_fn=pulled(reference))
"""
"""The recurrence stability suite's gradient site as it was: each arm drew its own cotangent at its own
output's dtype, so the float64 truth pulled back other values than the arms and every ratio sat at one
for nothing."""


@pytest.mark.parametrize("files, reported", [
    ({"test_site.py": PULLED_AS_IT_WAS}, True),
    ({"test_site.py": "import jax\nfrom reference_error import assert_as_exact\n\n\n"
                      "def test_g(f, draw):\n"
                      "    assert_as_exact(jax.jit(jax.grad(f)), f, draw, 'g', seeds=8)\n"}, True),
    ({"test_site.py": "from jax import value_and_grad\nfrom reference_error import assert_as_exact\n\n\n"
                      "def loss(f):\n    return value_and_grad(f)\n\n\n"
                      "def test_g(f, draw):\n    assert_as_exact(loss(f), f, draw, 'g', seeds=8)\n"}, True),
    ({"test_site.py": "import jax\nfrom checks import judged\n\n\n"
                      "def test_g(f, draw):\n    judged(jax.grad(f), draw)\n",
      "checks.py": "from reference_error import assert_as_exact\n\n\n"
                   "def judged(fn, draw):\n    assert_as_exact(fn, fn, draw, 'g', seeds=8)\n"}, True),
    ({"test_site.py": "from support import pulled\nfrom reference_error import assert_as_exact\n\n\n"
                      "def test_g(f, draw):\n    assert_as_exact(pulled(f), f, draw, 'g', seeds=8)\n",
      "support.py": "import torch\n\n\ndef pulled(f):\n"
                    "    return lambda x: torch.autograd.grad(f(x).sum(), x)\n"}, True),
    ({"test_site.py": "from reference_error import assert_as_exact, with_gradients\n\n\n"
                      "def test_g(f, draw):\n"
                      "    pulled = with_gradients(f)\n"
                      "    assert_as_exact(pulled, pulled, draw, 'g', seeds=8)\n"}, False),
    ({"test_site.py": "import jax\nfrom reference_error import assert_as_exact\n\n\n"
                      "def test_g(f, draw):\n    grads = jax.grad(f)(1.0)\n"
                      "    assert_as_exact(f, f, draw, 'g', seeds=8)\n"}, False),
], ids=["the recurrence site as it was", "an arm's own grad", "a helper's imported grad",
        "an assertion that passes its arm on", "a support module's torch autograd", "the gate's pull",
        "a gradient beside the check"])
def test_a_check_1_arm_that_pulls_back_its_own_cotangent_is_reported(lint, files, reported, tmp_path):
    """SLOP022: an arm of a check 1 takes its gradients from the gate (`with_gradients`), whose
    cotangent is the draw's; one that reaches a reverse-mode transform itself, through its own file's
    functions, an assertion that passes it on, or a suite module's, is reported."""
    (tmp_path / "tests").mkdir()
    for name, body in files.items():
        (tmp_path / "tests" / name).write_text(body)
    [module] = [module for module in lint.collect(["tests"], tmp_path, "dew")
                if module.path.stem == "test_site"]
    assert any(finding.code == "SLOP022" for finding in lint.check(module)) is reported
