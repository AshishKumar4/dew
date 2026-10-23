"""tools/fingerprints.py, the behavioral duplicate signal: a pure numeric helper is drawn at every dtype
the signal draws, and two helpers computing alike are grouped while one that calls the other is not."""

from tool_support import load

fingerprints = load("fingerprints")


def drawn(tmp_path, monkeypatch, module, source):
    """The candidates of `source`, written as `module` (a name of the test's own, which no other test has
    imported), each fingerprinted on the signal's draws."""
    (tmp_path / f"{module}.py").write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    found = fingerprints.candidates(tmp_path)
    draws = list(fingerprints._draws())
    for candidate in found:
        fingerprints.fingerprint(candidate, draws)
    return {candidate.name: candidate for candidate in found}


def test_a_bfloat16_helper_keeps_its_bfloat16_draws_and_derivatives(tmp_path, monkeypatch):
    found = drawn(tmp_path, monkeypatch, "bent_helpers", "import jax.numpy as jnp\n\n\ndef bent(x):\n"
                                                         "    return x * 2 + jnp.sin(x)\n")
    bf16 = [entry for key, entry in found["bent"].draws.items() if key[1] == "bfloat16"]
    assert len(bf16) == len(fingerprints.SHAPES)
    assert all(derivative is not None for _, derivative, _ in bf16)


def test_helpers_computing_alike_group_and_a_delegating_pair_is_left_out(tmp_path, monkeypatch):
    """Two forms of softplus match, forward and derivative; the wrapper matches both, its pair with the
    function it calls left out as delegation and its pair with the other form kept; tanh matches none."""
    found = drawn(tmp_path, monkeypatch, "softplus_helpers", """import jax.numpy as jnp


def softplus(x):
    return jnp.log1p(jnp.exp(x))


def softplus_again(x):
    return jnp.log(1 + jnp.exp(x))


def wrapped(x):
    return softplus(jnp.asarray(x))


def other(x):
    return jnp.tanh(x)
""")
    groups, delegating = fingerprints.groups(list(found.values()))
    assert [labels for labels, _ in groups] == [
        ["softplus_helpers:softplus", "softplus_helpers:softplus_again", "softplus_helpers:wrapped"]]
    assert delegating == 1
    assert all("forward and derivative" in pair for pair in groups[0][1])
