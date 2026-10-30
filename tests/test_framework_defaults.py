"""Where Dew departs from JAX's, Optax's or a library's default, the user's own setting holds and the
departure has an off switch (row 2506): a run's decay spares biases and vectors unless told not to, a
group following its config; a user's persistent-cache thresholds and tokenizer parallelism stay theirs."""

import dataclasses

import jax
import jax.numpy as jnp

from dew.optim import OptimConfig, ParamGroup

PARAMS = {"dense": {"kernel": jnp.ones((3, 2)), "bias": jnp.ones((2,))}}


def decayed(config: OptimConfig) -> dict[str, bool]:
    """Whether the first update over a zero gradient moves each leaf: only a decay does."""
    solver = config.build(4)
    updates, _ = solver.update(jax.tree.map(jnp.zeros_like, PARAMS), solver.init(PARAMS), PARAMS)
    return {name: bool(jnp.any(leaf != 0)) for name, leaf in updates["dense"].items()}


def test_a_decay_spares_biases_unless_told_not_to_and_a_group_decays_what_it_claims():
    """The ungrouped decay spares a bias, as nanoGPT's, Megatron Core's and timm's split does
    (`OptimConfig.spared`), and decays it where the config says not to spare, as Optax does. A param
    group decays every parameter it claims, as a torch parameter group does, unless it spares; a run's
    default fresh groups decay as its ungrouped optimizer would (`OptimConfig.fresh_groups`)."""
    config = OptimConfig(optimizer="adamw", learning_rate=1e-2, weight_decay=0.1)
    assert decayed(config) == {"kernel": True, "bias": False}
    assert decayed(dataclasses.replace(config, spared=False)) == {"kernel": True, "bias": True}
    claimed = dataclasses.replace(config, param_groups=(ParamGroup("all", ("*",)),))
    assert decayed(claimed) == {"kernel": True, "bias": True}
    sparing = dataclasses.replace(config, param_groups=(ParamGroup("all", ("*",), spared=True),))
    assert decayed(sparing) == {"kernel": True, "bias": False}
    for switch in (True, False):
        assert all(group.spared is switch
                   for group in dataclasses.replace(config, spared=switch).fresh_groups())


def test_a_users_persistent_cache_thresholds_stay_theirs(tmp_path, monkeypatch):
    """A cache Dew enables caches every program, unless the user named JAX's thresholds
    (JAX_PERSISTENT_CACHE_MIN_*), which it leaves as they are."""
    from dew.cache import enable_compilation_cache

    sizes = jax.config.jax_persistent_cache_min_entry_size_bytes
    seconds = jax.config.jax_persistent_cache_min_compile_time_secs
    try:
        monkeypatch.setenv("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "4096")
        monkeypatch.setenv("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "2.5")
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 4096)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 2.5)
        enable_compilation_cache(str(tmp_path / "named"))
        assert (jax.config.jax_persistent_cache_min_entry_size_bytes,
                jax.config.jax_persistent_cache_min_compile_time_secs) == (4096, 2.5)
        monkeypatch.delenv("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES")
        monkeypatch.delenv("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS")
        enable_compilation_cache(str(tmp_path / "unnamed"))
        assert (jax.config.jax_persistent_cache_min_entry_size_bytes,
                jax.config.jax_persistent_cache_min_compile_time_secs) == (-1, 0.0)
    finally:
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", sizes)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", seconds)


def test_a_users_tokenizer_parallelism_stays_theirs(monkeypatch):
    """A process Dew prepares turns HF tokenizers' thread pool off, as grain's forked workers would
    deadlock it, unless the user set TOKENIZERS_PARALLELISM, which stays theirs."""
    import os

    from dew.training.runtime import prepare_process

    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")
    prepare_process(multi_host=False)
    assert os.environ["TOKENIZERS_PARALLELISM"] == "true"
    monkeypatch.delenv("TOKENIZERS_PARALLELISM")
    prepare_process(multi_host=False)
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
