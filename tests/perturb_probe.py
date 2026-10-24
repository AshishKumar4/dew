"""Scratch plugin (`-p tests.perturb_probe`): at the heavy-tail draws of a check-1 site, its float64
truth and both arms at the draw's parameters moved by relative noise of each scale, printed as each
leaf's change from the unmoved draw over the leaf's size (PERTURB lines); the run then stops."""
import numpy as np

SEEDS = (182, 90, 3)
SCALES = (0.0, 1e-12, 1e-9, 1e-7, 1e-6, 1e-5)
WATCHED = ("['text']['layers_0']['self_attn']['q_proj']['kernel']", "['text']['layers_0']['self_attn']['q_norm']['scale']",
           "['text']['layers_1']['mlp']['gate_proj']['kernel']")


def pytest_configure(config):
    import jax
    import reference_error

    def probe(dew_fn, reference_fn, draw, label, *, seeds=None, truth_fn=None, **_):
        for seed in SEEDS:
            (values,) = draw(seed)
            rng = np.random.default_rng(seed)
            noise = jax.tree.map(lambda leaf: rng.standard_normal(np.shape(leaf)), values["params"])
            base = None
            for scale in SCALES:
                moved = {**values, "params": jax.tree.map(
                    lambda leaf, step: np.asarray(np.asarray(leaf, np.float64) * (1 + scale * step), np.float64),
                    values["params"], noise)}
                with jax.enable_x64(new_val=True):
                    truth = (truth_fn or reference_fn)(*reference_error.widened((moved,)))
                narrow = jax.tree.map(lambda leaf: np.asarray(leaf, np.float32), moved)
                arms = {"truth": truth, "dew": dew_fn(narrow), "ref": reference_fn(narrow)}
                flat = {arm: {jax.tree_util.keystr(path): np.asarray(leaf, np.float64)
                              for path, leaf in jax.tree_util.tree_leaves_with_path(tree)}
                        for arm, tree in arms.items()}
                if base is None:
                    base = flat
                for name in WATCHED:
                    if name not in flat["truth"]:
                        continue
                    size = np.sqrt(np.mean(np.square(base["truth"][name])))
                    moved_by = np.sqrt(np.mean(np.square(flat["truth"][name] - base["truth"][name]))) / size
                    errors = {arm: np.sqrt(np.mean(np.square(flat[arm][name] - flat["truth"][name]))) / size
                              for arm in ("dew", "ref")}
                    print(f"PERTURB seed {seed} scale {scale:.0e} {name}: truth moved {moved_by:.3e}; "
                          f"dew error {errors['dew']:.3e}; ref error {errors['ref']:.3e}", flush=True)
        raise SystemExit("perturb probe done")

    reference_error.assert_as_exact = probe
